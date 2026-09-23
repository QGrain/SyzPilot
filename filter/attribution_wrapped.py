#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer
from model import TraceClassifierForAttribution
from dataset import ProgramDataset
from utils import create_collate_fn, mean_pooling
from config import get_model_max_length
from torch.utils.data import DataLoader, Subset
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
torch.cuda.empty_cache()

# ============================== Data Preprocessing Functions ==============================
def ids2token(input_ids, tokenizer):
    """Convert input_ids to a readable Token sequence"""
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    s = ''.join(tokens)
    s = s.replace('Ġ', ' ').replace('Ċ', '\n').replace('<|endoftext|>', '')
    print(s)

def replace_tokens(tokens):
    """Clean special characters from Tokens"""
    for i in range(len(tokens)):
        tokens[i] = tokens[i].replace('Ġ', ' ').replace('Ċ', '\n').replace('<|endoftext|>', '')

# ============================== Attribution Computation Functions ==============================
def split_invocations(tokens, token_attributions):
    """Split Token attribution data by function call"""
    assert len(tokens) == len(token_attributions)
    invocations_tokens, invocations_attrs = [], []
    one_invocation_tokens, one_invocation_attrs = [], []

    for i, tk in enumerate(tokens):
        one_invocation_tokens.append(tk)
        one_invocation_attrs.append(token_attributions[i])
        if '\n' in tk:
            invocations_tokens.append(one_invocation_tokens)
            invocations_attrs.append(one_invocation_attrs)
            one_invocation_tokens, one_invocation_attrs = [], []

    if one_invocation_tokens:
        invocations_tokens.append(one_invocation_tokens)
        invocations_attrs.append(one_invocation_attrs)

    return invocations_tokens, invocations_attrs

def extract_syscall_from_str(invocation_str):
    """Extract syscall name from function call string"""
    syscall_str = invocation_str.split('(')[0].strip().split(' ')[-1]
    return syscall_str.strip()

def get_syscall_attr(tokenizer, invocations_tokens, invocations_attrs):
    """Compute attribution value for each syscall"""
    assert len(invocations_tokens) == len(invocations_attrs)
    syscall_attr = {}

    for i in range(len(invocations_tokens)):
        invocation_str = ''.join(invocations_tokens[i])
        if '(' not in invocation_str:
            continue
        syscall_str = extract_syscall_from_str(invocation_str)
        if not syscall_str:
            continue
        syscall_attr.setdefault(syscall_str, 0)

        syscall_tokens = tokenizer.tokenize(syscall_str)
        syscall_tokens_num = len(syscall_tokens)

        for j in range(len(invocations_tokens[i]) - syscall_tokens_num + 1):
            if all(syscall_tokens[k] == invocations_tokens[i][j+k].strip() for k in range(syscall_tokens_num)):
                syscall_attr[syscall_str] += sum(invocations_attrs[i][j:j+syscall_tokens_num])
                break

    return syscall_attr

def merge_syscall_attr(total_syscall_attr, new_syscall_attr):
    """Merge syscall attribution values across multiple samples"""
    for syscall_str, attr in new_syscall_attr.items():
        total_syscall_attr[syscall_str] = total_syscall_attr.get(syscall_str, 0) + attr

# ============================== Sorting and Analysis Functions ==============================
def rank_token_attrs(tokens, token_attributions):
    """Sort Token attribution values"""
    token_attrs = {}

    for i in range(len(tokens)):
        token_attrs[tokens[i]] = token_attrs.get(tokens[i], 0) + token_attributions[i]

    sorted_token_attrs = sorted(token_attrs.items(), key=lambda x: x[1], reverse=True)

    for token, attr in sorted_token_attrs:
        print(f'{token}: {attr}')

    return sorted_token_attrs

# ============================== Data Loading and Model Initialization ==============================
def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Run model attribution analysis")
    parser.add_argument("--prog_path", type=str, required=True, help="Path to program dataset")
    parser.add_argument("--label_path", type=str, required=True, help="Path to label dataset")
    parser.add_argument("--base_model", type=str, required=True, help="Path to base model")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer")
    parser.add_argument("--state_path", type=str, required=True, help="Path to model state file")
    parser.add_argument("--output", type=str, default="attribution_results.html", help="Output HTML file")
    parser.add_argument("--analysis_num", type=int, default=50, help="Number of samples to analyze")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use for computation")
    return parser.parse_args()
    '''Example command:
    python attribution_wrapped.py \
    --prog_path "./data/programs_g5_new_10w.pkl" \
    --label_path "./data/labels_g5_new_10w.pkl" \
    --base_model "/opt/syzpilot/models/starencoder" \
    --tokenizer "/opt/syzpilot/models/custom_tokenizer_224w" \
    --state_path "./checkpoints/g5_new_1/latest_state.pt" \
    --output "attribution_results.html" \
    --analysis_num 50
    '''

def load_data(prog_path, label_path, tokenizer_path, base_model_path):
    """Load dataset and create DataLoader"""
    dataset = ProgramDataset(prog_path, label_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    collate_fn = create_collate_fn(tokenizer, get_model_max_length(base_model_path))
    data_loader = DataLoader(Subset(dataset, list(range(1000))), batch_size=1, shuffle=False, collate_fn=collate_fn)
    return dataset, tokenizer, collate_fn, data_loader

def initialize_model(base_model_path, state_path, label_dim, device=None):
    """Initialize model and load weights"""
    model = TraceClassifierForAttribution(base_model_path, label_dim)

    # Auto-detect CUDA availability
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Explicitly specify map_location to prevent errors when GPU is unavailable
    map_location = torch.device(device)
    model.load_state_dict(torch.load(state_path, map_location=map_location))

    model.to(map_location)
    model.eval()
    return model

# ============================== Main Logic ==============================
def main():
    args = parse_arguments()
    dataset, tokenizer, collate_fn, data_loader = load_data(args.prog_path, args.label_path, args.tokenizer, args.base_model)
    model = initialize_model(args.base_model, args.state_path, dataset.label_dim, f"cuda:{args.gpu_id}")
    lig = LayerIntegratedGradients(model, model.base_model.embeddings)
    # pad_token_id = tokenizer.pad_token_id

    vis_data_records = []
    total_syscall_attr = {}
    cnt = 0
    for batch in data_loader:
        input_ids, attention_mask, labels = batch
        # Only proceed for samples whose true label matches the target
        if torch.argmax(labels, dim=1).tolist() != [dataset.label_dim - 1]:
            continue
        device = torch.device(f"cuda:{args.gpu_id}")
        input_ids, attention_mask = input_ids.long().to(device), attention_mask.long().to(device)

        # Compute logits
        logits = model(input_ids, attention_mask)
        pred_labels = torch.argmax(logits, dim=1)
        # Only proceed with model interpretation for samples predicted as target
        if pred_labels.tolist() != [dataset.label_dim - 1]:
            continue
        cnt += 1
        print(f'===== for the {cnt} target program =====')
        print("Input IDs:", input_ids)
        print("Attention Mask:", attention_mask)
        print("True Labels (One-hot):", labels)
        print("Predicted Labels:", pred_labels)

        # Create baseline using pad token
        # baseline_input_ids = pad_token_id * torch.ones_like(input_ids)

        # Compute token-level attributions
        attributions_lig, delta = lig.attribute(
            inputs=input_ids,
            baselines=torch.zeros_like(input_ids),
            additional_forward_args=(attention_mask,),
            target=pred_labels,
            n_steps=50,
            return_convergence_delta=True
        )

        # Get per-token attributions (sum over embedding dimension to get each token's contribution)
        token_attributions = attributions_lig.sum(dim=-1).squeeze(0).tolist()

        # Convert input_ids back to original token sequence
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
        replace_tokens(tokens)

        # Now token_attributions length matches tokens, safe to pass to split_invocations()
        invocation_tokens, invocation_attrs = split_invocations(tokens, token_attributions)
        syscall_attr = get_syscall_attr(tokenizer, invocation_tokens, invocation_attrs)
        merge_syscall_attr(total_syscall_attr, syscall_attr)

        # Store visualization data
        vis_data = viz.VisualizationDataRecord(
            word_attributions=np.array(token_attributions),
            pred_prob=torch.max(F.softmax(logits, dim=1)).item(),
            pred_class=torch.argmax(logits, dim=1).item(),
            true_class=torch.argmax(labels, dim=1).item(),
            attr_class=torch.argmax(logits, dim=1).item(),
            attr_score=np.sum(token_attributions),
            raw_input_ids=tokens,
            convergence_score=delta
        )
        vis_data_records.append(vis_data)

    # Visualize all token attribution results
    html_output = viz.visualize_text(vis_data_records)
    html_str = ''.join(html_output.data)  # Get HTML string

    # Save HTML file
    save_fn = args.output
    with open(save_fn, 'w') as f:
        f.write(html_str)

    print(f"Visualization saved to {save_fn}")

    # Print syscall attribution ranking
    sorted_syscall_attr = sorted(total_syscall_attr.items(), key=lambda x: x[1], reverse=True)
    print("========== Syscall Attribution Rank ==========")
    for one in sorted_syscall_attr[:5]:
        print(one)


if __name__ == "__main__":
    main()
