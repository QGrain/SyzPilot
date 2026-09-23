
import argparse
from datetime import datetime
from glob import glob
import json
import logging
import os
from pathlib import Path
import pickle
from time import time

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs as DDPK
from accelerate.utils import ProjectConfiguration, broadcast
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    GPT2Tokenizer,
    get_cosine_schedule_with_warmup,
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
)

from config import ModelConfig
from logger import TrainLogger


# Set up logger with timestamps
logger = TrainLogger(log_name="SyzEncoder-pretrain")


class SyzProgramDataset(Dataset):
    def __init__(self, prog_dataset_dir, tokenizer, max_len, accelerator=None, split='train', small_test=None):
        if prog_dataset_dir[-1] == '/':
            prog_dataset_dir = prog_dataset_dir[:-1]
        prog_dataset_name = os.path.basename(prog_dataset_dir)
        if prog_dataset_name == 'programs':
            prog_dataset_name = f'{os.path.basename(os.path.dirname(prog_dataset_dir))}_{prog_dataset_name}'

        # Cache the full dataset, not the split data
        cache_fpath = os.path.join(ModelConfig.data_dir, f'{prog_dataset_name}_full_cache.pkl')
        self.split = split

        # First load or create the full dataset
        full_data = self.load_cached_data(cache_fpath)
        if not full_data:
            full_data = []
            try:
                # Use pathlib for efficient file listing with large directories
                prog_path = Path(prog_dataset_dir)
                files = [str(f) for f in prog_path.iterdir() if f.is_file()]
            except Exception as e:
                logger.error(f"[pid={os.getpid()}] Failed to list files in {prog_dataset_dir}: {e}")
                files = []

            # Check if this is the main process (defaults to True if no accelerator)
            is_main_process = accelerator.is_main_process if accelerator is not None else True

            if is_main_process:
                logger.info(f'[pid={os.getpid()}] {cache_fpath} not found, loading programs from {prog_dataset_dir}')
                with tqdm(total=len(files), desc="Loading programs", ncols=100) as pbar:
                    for file in files:
                        try:
                            with open(file, 'r', encoding='utf-8') as f:
                                program = f.read().strip()
                                full_data.append(program)
                        except Exception as e:
                            logger.warning(f"Error reading {file}: {e}")
                        pbar.update(1)
                self.cache_data(cache_fpath, full_data)
                logger.info(f'[pid={os.getpid()}] Cached {len(full_data)} programs to {cache_fpath}')
            else:
                # Non-main process: read data but don't cache
                for file in files:
                    try:
                        with open(file, 'r', encoding='utf-8') as f:
                            program = f.read().strip()
                            full_data.append(program)
                    except Exception as e:
                        continue

        # Wait for all processes to finish data loading (if accelerator is used)
        if accelerator is not None:
            accelerator.wait_for_everyone()

        if accelerator and accelerator.is_main_process:
            logger.info(f"[pid={os.getpid()}] All processes completed data loading, full_data size: {len(full_data)}")

        # Small-scale test
        if small_test is not None and isinstance(small_test, float):
            full_data = full_data[:int(len(full_data) * small_test)]
            if accelerator and accelerator.is_main_process:
                logger.info(f"[pid={os.getpid()}] Using small test dataset: {len(full_data)} samples")

        # Data split: use a fixed seed to ensure consistent splits across all processes
        import random
        seed = 100085
        indices = list(range(len(full_data)))
        random.seed(seed)
        random.shuffle(indices)

        # Split dataset: 95% train, 5% val
        total_samples = len(indices)
        train_end = int(0.9 * total_samples)  # DEBUG set to 0.05

        if split == 'train':
            selected_indices = indices[:train_end]
        elif split == 'val':
            selected_indices = indices[train_end:]  # DEBUG set to indices[train_end:train_end*2]
        else:
            raise ValueError(f"Invalid split: {split}. Only 'train' and 'val' are supported.")

        # Retrieve data by selected indices
        self.data = [full_data[i] for i in selected_indices]

        logger.info(f'[pid={os.getpid()}] [Done] Loaded {len(self.data)} {split} programs (total: {len(full_data)})')
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def load_cached_data(self, cache_fpath):
        if os.path.exists(cache_fpath):
            try:
                with open(cache_fpath, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"[pid={os.getpid()}] Failed to load cache {cache_fpath}: {e}")
                return None
        return None

    def cache_data(self, cache_fpath, data):
        if len(data) == 0:
            logger.warning(f"[pid={os.getpid()}] No data to cache: {cache_fpath}")
            return
        try:
            os.makedirs(os.path.dirname(cache_fpath), exist_ok=True)
            with open(cache_fpath, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"[pid={os.getpid()}] Failed to cache data: {e}")


def create_collate_fn_for_mlm(tokenizer, max_length: int = 1024):
    def collate_fn(programs):
        # Filter out empty programs
        programs = [p for p in programs if p.strip()]
        if not programs:
            programs = ["# empty program"]

        tokenized = tokenizer(
            programs,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True  # MLM pretraining requires [CLS] and other special tokens
        )
        return tokenized["input_ids"], tokenized["attention_mask"]
    return collate_fn


def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    """
    Prepare inputs for MLM: randomly mask some tokens
    """
    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling."
        )

    labels = inputs.clone()
    # Create probability matrix to decide which tokens to mask
    probability_matrix = torch.full(labels.shape, mlm_probability, device=inputs.device)

    # Get mask for special tokens
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool, device=inputs.device)

    # Special tokens are excluded from masking
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    # Padding tokens are excluded from masking
    if tokenizer.pad_token_id is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)

    # Randomly select tokens to mask
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Only compute loss for masked tokens

    # MLM strategy: 80% [MASK], 10% random token, 10% keep original
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=inputs.device)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time replace with a random token
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=inputs.device)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long, device=inputs.device)
    inputs[indices_random] = random_words[indices_random]

    # Remaining 10% keep original (already in inputs)

    return inputs, labels


class PretrainTrainer:
    def __init__(self, config):
        self.config = config
        self.start_time = time()
        config["session_name"] = f"""{config["session_name"]}-{datetime.now().strftime("%Y%m%d-%H%M%S")}"""
        config["log_dir"] = Path(config["log_dir"])
        config["log_dir"].mkdir(parents=True, exist_ok=True)

        self.device = torch.device(config["device"])

        # Initialize accelerator
        kwargs = DDPK(find_unused_parameters=False)
        if config["accelerator"]:
            if config["disable_wandb"]:
                self.accelerator = Accelerator(
                    log_with="tensorboard",
                    gradient_accumulation_steps=config["gradient_acc_step"],
                    kwargs_handlers=[kwargs],
                    project_config=ProjectConfiguration(
                        project_dir=config["log_dir"],
                    ),
                )
                self.accelerator.init_trackers(project_name=config["session_name"])
            else:
                self.accelerator = Accelerator(
                    gradient_accumulation_steps=config["gradient_acc_step"],
                    log_with="wandb",
                    kwargs_handlers=[kwargs],
                )
                self.accelerator.init_trackers(
                    project_name=config["proj_name"],
                    config=config,
                    init_kwargs={
                        "wandb": {
                            "name": config["run_name"],
                            "mode": "online",
                        }
                    },
                )
        else:
            self.accelerator = None

        self.is_main_process = self.accelerator.is_main_process if self.accelerator is not None else True

        # Set up tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])

        # Properly add special tokens
        # special_tokens_dict = {}
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.mask_token is None:
            self.tokenizer.add_special_tokens({"mask_token": "<mask>"})

        # if special_tokens_dict:
        #     num_added_tokens = self.tokenizer.add_special_tokens(special_tokens_dict)
            # logger.info(f"Added {num_added_tokens} special tokens")

        if self.is_main_process:
            logger.info(f"Tokenizer vocab size: {self.tokenizer.vocab_size}")
            logger.info(f"Pad token: {self.tokenizer.pad_token} (id: {self.tokenizer.pad_token_id})")
            logger.info(f"Mask token: {self.tokenizer.mask_token} (id: {self.tokenizer.mask_token_id})")
            for _k in self.config:
                logger.info(f"""config["{_k}"]: {self.config[_k]}""")

        # Create datasets
        train_dataset = SyzProgramDataset(
            config["prog_dataset"],
            self.tokenizer,
            max_len=1024,
            accelerator=self.accelerator,
            split='train',
            small_test=config.get("small_test", None)
        )
        val_dataset = SyzProgramDataset(
            config["prog_dataset"],
            self.tokenizer,
            max_len=1024,
            accelerator=self.accelerator,
            split='val',
            small_test=config.get("small_test", None)
        )

        # Wait for all processes to finish data loading
        if self.accelerator:
            self.accelerator.wait_for_everyone()

        # Create data loaders
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            collate_fn=create_collate_fn_for_mlm(self.tokenizer, 1024),
            pin_memory=True
        )

        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=create_collate_fn_for_mlm(self.tokenizer, 1024),
            pin_memory=True
        )

        # Load model
        self.model = AutoModelForMaskedLM.from_pretrained(config["base_model"])
        if not self.accelerator:
            self.model = self.model.to(self.device)
        if self.is_main_process:
            logger.info(f"Loading MLM model from {config['base_model']}")
            logger.info(f"Model vocab size: {self.model.config.vocab_size}")
            logger.info(f"Tokenizer vocab size: {self.tokenizer.vocab_size}")

        # Check if embedding resize is needed
        if self.model.config.vocab_size < self.tokenizer.vocab_size:
            logger.warning(f"Resizing model embeddings from {self.model.config.vocab_size} to {self.tokenizer.vocab_size}")
            self.model.resize_token_embeddings(self.tokenizer.vocab_size)

        # Optimizer: no weight decay for bias and LayerNorm parameters
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": config["weight_decay"],
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=config["lr"],
            betas=(0.9, 0.98)
        )

        # Prepare with accelerator
        if self.accelerator:
            self.model, self.optimizer, self.train_dataloader, self.val_dataloader = self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader, self.val_dataloader,
            )
        self.val_dataiter = iter(self.val_dataloader)

        # Learning rate scheduler
        total_steps = len(self.train_dataloader) * config["epochs"]
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * 0.05),
            num_training_steps=total_steps
        )
        self.scheduler = self.accelerator.prepare(self.scheduler)

        # Training parameters
        self.total_steps = total_steps
        self.train_dataloader_len = len(self.train_dataloader)
        # self.save_interval = int(self.train_dataloader_len * config["save_interval"])

        # Saving configuration
        self.need_save = config.get("output_dir", None) is not None

        if self.is_main_process:
            logger.info('Starting SyzEncoder MLM pretraining...')
            logger.info(f'Model parameters: {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M')
            logger.info(f"Total steps: {total_steps}")
            logger.info(f"Learning rate: {config['lr']}")
            logger.info(f"Batch size: {config['batch_size']}")
            logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    @torch.no_grad()
    def evaluate_model(self):
        """Evaluate model performance"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        # Reset iterator before each evaluation to ensure fixed and reproducible eval samples
        self.val_dataiter = iter(self.val_dataloader)

        prog_bar = tqdm(range(self.config["test_steps"]), desc="Evaluating", disable=not self.is_main_process, ncols=100)
        while True:
            try:
                input_ids, attention_mask = next(self.val_dataiter)
            except StopIteration:
                self.val_dataiter = iter(self.val_dataloader)
                continue

            masked_input_ids, labels = mask_tokens(input_ids, self.tokenizer, self.config.get("mlm_probability", 0.15))
            outputs = self.model(
                input_ids=masked_input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1

            if num_batches >= self.config["test_steps"]:
                prog_bar.close()
                break
            prog_bar.update(1)

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        avg_loss = self.accelerator.gather(torch.tensor(avg_loss, device=self.accelerator.device)).mean()
        return avg_loss.item()

    def _cleanup_old_checkpoints(self, save_checkpoint_dir, keep=5):
        """Keep only the most recent `keep` training state checkpoints, delete older ones"""
        ckpt_files = sorted(
            save_checkpoint_dir.glob("ckpt-step-*-training_state.pth"),
            key=lambda p: int(p.stem.split("-")[2]),
        )
        for old_ckpt in ckpt_files[:-keep]:
            old_ckpt.unlink()
            logger.info(f"🗑️  Removed old checkpoint: {old_ckpt}")

    def save_checkpoint(self, step, val_loss=None):
        """Save checkpoint"""
        if not self.is_main_process or not self.need_save:
            return

        epoch = step // self.train_dataloader_len
        save_checkpoint_dir = Path(f"{self.config['output_dir']}/checkpoints/")
        save_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save complete model state
        if self.accelerator:
            model_state_dict = self.accelerator.unwrap_model(self.model).state_dict()
        else:
            model_state_dict = self.model.cpu().state_dict()
        torch.save({
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': epoch,
            'step': step,
            'loss': val_loss,
            'config': self.config
        }, save_checkpoint_dir / f"ckpt-step-{step}-training_state.pth")

        # Save encoder model in HuggingFace format
        final_model_dir = save_checkpoint_dir / f"syzencoder-{step}"
        final_model_dir.mkdir(parents=True, exist_ok=True)
        if self.accelerator:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
        else:
            unwrapped_model = self.model.cpu()
        if hasattr(unwrapped_model, 'bert'):
            unwrapped_model.bert.save_pretrained(final_model_dir)
        else:
            unwrapped_model.save_pretrained(final_model_dir)
        self.tokenizer.save_pretrained(final_model_dir)
        logger.info(f"💾 Encoder checkpoint saved to {final_model_dir}")

        # Auto-cleanup old checkpoints, keep only the most recent 5
        self._cleanup_old_checkpoints(save_checkpoint_dir, keep=5)

    def save_best_model(self, val_loss):
        """Save best model"""
        if not self.is_main_process or not self.need_save:
            return

        best_model_dir = f"{self.config['output_dir']}/best_model"
        os.makedirs(best_model_dir, exist_ok=True)
        if self.accelerator:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
        else:
            unwrapped_model = self.model.cpu()

        # Save pure encoder (bert part only), excluding MLM head
        if hasattr(unwrapped_model, 'bert'):
            unwrapped_model.bert.save_pretrained(best_model_dir)
        else:
            unwrapped_model.save_pretrained(best_model_dir)

        self.tokenizer.save_pretrained(best_model_dir)
        logger.info(f"🎉 New best encoder saved! Val loss improved: {val_loss:.6f}")

    def save_final_model(self):
        """Save final model"""
        if not self.is_main_process or not self.need_save:
            return

        final_save_path = f"{self.config['output_dir']}/final_syzencoder"
        os.makedirs(final_save_path, exist_ok=True)

        if self.accelerator:
            self.accelerator.wait_for_everyone()
            unwrapped_model = self.accelerator.unwrap_model(self.model)
        else:
            unwrapped_model = self.model.cpu()

        # Save pure encoder model and tokenizer (excluding MLM head)
        if hasattr(unwrapped_model, 'bert'):
            unwrapped_model.bert.save_pretrained(final_save_path, from_pt=True)
            logger.info(f"Saved pure encoder (bert) to {final_save_path}")
        else:
            unwrapped_model.save_pretrained(final_save_path, from_pt=True)
            logger.info(f"Saved complete model to {final_save_path}")

        self.tokenizer.save_pretrained(final_save_path)

        # Save full configuration
        with open(os.path.join(final_save_path, 'training_config.json'), 'w') as f:
            json.dump(self.config, f, indent=2, default=str)

        logger.info(f"Final SyzEncoder saved to {final_save_path}")

    def run(self):
        """Run training"""
        step_no = 0
        best_val_loss = float('inf')
        start_time = time()

        val_loss = self.evaluate_model()
        logger.info(f"Step {step_no} completed: Val Loss = {val_loss:.6f}, Best Val Loss = {best_val_loss:.6f}")
        self.accelerator.wait_for_everyone()

        # Create progress bar
        prog_bar = tqdm(
            total=self.total_steps,
            disable=not self.is_main_process,
            ncols=100,
        )

        # Start training loop
        train_dataiter = iter(self.train_dataloader)
        while step_no < self.total_steps:
            if step_no >= self.total_steps:
                break

            try:
                input_ids, attention_mask = next(train_dataiter)
            except StopIteration:
                train_dataiter = iter(self.train_dataloader)
                continue

            epoch = step_no // self.train_dataloader_len + 1
            step_no += 1
            self.model.train()

            # Training step (with gradient accumulation)
            masked_input_ids, labels = mask_tokens(input_ids, self.tokenizer, self.config.get("mlm_probability", 0.15))

            if self.accelerator:
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(
                        input_ids=masked_input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.get("max_grad_norm", 1.0))
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
            else:
                masked_input_ids, labels, attention_mask = map(
                    lambda x: x.to(self.device),
                    [masked_input_ids, labels, attention_mask],
                )
                outputs = self.model(
                    input_ids=masked_input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get("max_grad_norm", 1.0))
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            # Update progress bar
            prog_bar.set_description(f"Epoch {epoch}, Step {step_no}: loss={loss.item():.3f}, lr={self.scheduler.get_last_lr()[0]:.2e}")
            prog_bar.update(1)

            # Log metrics
            if step_no % 100 == 0:
                self.accelerator.log({
                    "train/loss": loss.item(),
                    "train/learning_rate": self.scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                }, step=step_no)
            if self.is_main_process:
                logger.debug(
                    f"Epoch {epoch:2}, Step {step_no:4}: " \
                    f"train_loss={loss.item():.3f}, " \
                    f"lr={self.scheduler.get_last_lr()[0]:.2e}"
                )

            # Periodic checkpoint saving and validation
            if step_no % self.config["save_interval"] == 0:
                val_loss = self.evaluate_model()
                if self.is_main_process:
                    logger.info(f"Step {step_no} completed: Val Loss = {val_loss:.6f}, Best Val Loss = {best_val_loss:.6f}")

                    self.save_checkpoint(step_no, val_loss)
                    self.accelerator.log({
                        "epoch/val_loss": val_loss,
                        "epoch/best_val_loss": best_val_loss,
                        "epoch/epoch": epoch,
                        "epoch/time_minutes": (time() - start_time) // 60,
                    }, step=step_no)
                    logger.debug(
                        f"Epoch {epoch}: " \
                        f"val_loss={val_loss:.3f}, " \
                        f"best_val_loss={best_val_loss:.3f}"
                    )

                    # Save best model (unconditionally save current best)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.save_best_model(val_loss)

            self.accelerator.wait_for_everyone()

        prog_bar.close()

        # Save final model
        self.save_final_model()
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()


def main(configs):
    trainer = PretrainTrainer(configs)
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continue pretrain StarEncoder on syz-programs")

    # Training parameters
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate (smaller for continue pretraining)')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm for clipping')
    parser.add_argument('--mlm_probability', type=float, default=0.15, help='MLM masking probability')
    parser.add_argument('--save_interval', type=int, default=6000, help='interval in steps to save the model')
    parser.add_argument('--test_steps', type=int, default=200, help='interval in steps to save the model')  # multi-GPU eval with 2x batch size requires fewer steps
    parser.add_argument('--gradient_acc_step', type=int, default=2, help='gradient accumulation steps')

    parser.add_argument('--accelerator', type=bool, default=True, help='Enable accelerator')
    parser.add_argument('--device', type=str, default='cuda:1')

    # Path parameters
    parser.add_argument('--prog_dataset', type=str, default='/artifact/datasets/prog_dataset_300w')
    parser.add_argument('--base_model', type=str, default='/opt/syzpilot/models/starencoder')
    parser.add_argument('--tokenizer', type=str, default='/opt/syzpilot/models/customized_tokenizer_300w')
    parser.add_argument('--output_dir', type=str, default='/opt/syzpilot/models/syzencoder_300w')
    parser.add_argument('--log_dir', type=str, default='/opt/syzpilot/logs')

    # Logging parameters
    parser.add_argument('--disable_wandb', type=bool, default=True, help='Disable wandb logging')
    parser.add_argument('--proj_name', type=str, default='syzencoder_300w_pretrain', help='Wandb project name')
    parser.add_argument('--session_name', type=str, default='default_session', help='Wandb run name')

    # Test parameters
    parser.add_argument('--small_test', type=float, default=None, help='Use small dataset for testing (e.g., 0.01)')

    args = parser.parse_args()
    main(vars(args))
