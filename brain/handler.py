import os
import logging
from abc import ABC
import torch
import torch.nn.functional as F
from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import map_class_to_label
from transformers import AutoTokenizer


# TODO
# 1. remove all the debug print in production scenario
# 2. optimize the tokenizer loading and log file setting.
# 3. fix the warning of nvfuser.

########## CONFIG ##########
TOKEN = ''
config = {
    "tokenizer": os.getenv("TOKENIZER_PATH", "/artifact/assets/models/SyzTokenizer_224w/"),
    "max_length": 1024  # MUST SAME AS THE TRAINING
}
debug_log_file = os.getenv("SERVE_LOG_PATH", "./torchserve_debug.log") # log path
default_cuda = 'cuda:0'
############################


# Set up logging
logger = logging.getLogger(__name__)

# Create a file handler for inference debug logs
file_handler = logging.FileHandler(debug_log_file)
file_handler.setLevel(logging.DEBUG)

# Create a formatter and set it to the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(file_handler)

torch.jit.fuser('off')
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_set_nvfuser_enabled(False)

class ReachablityClassifier(BaseHandler, ABC):

    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"], token=TOKEN)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.force_device = torch.device(default_cuda)

    def initialize(self, context):
        super().initialize(context)
        self.initialized = False
        self.manifest = context.manifest
        properties = context.system_properties
        # test_get_devide = "cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu"
        # logger.debug('test get device: %s', test_get_devide)
        # print('test get device: %s'%test_get_devide)
        if not self.device:
            self.device = self.force_device
        self.model.to(self.device)
        self.initialized = True
        # logger.info('initialize done')

    def preprocess(self, data):
        logger.debug(
            "Preprocessing inference batch: size=%d, device=%s",
            len(data),
            self.device,
        )
        program_batch = []
        for req in data:
            payload = req.get("data")
            if payload is None:
                payload = req.get("body")
            if isinstance(payload, memoryview):
                payload = payload.tobytes()
            # Decode payload if not a str but bytes or bytearray
            if isinstance(payload, (bytes, bytearray)):
                payload = payload.decode("utf-8")
            else:
                raise ValueError(f"Fail to transform payload to str. Unsupported payload type: {type(payload)}")
            program_batch.append(payload)

        tokenized_inputs = self.tokenizer(program_batch, return_tensors="pt", padding="max_length", truncation=True, max_length=config["max_length"])
        # logger.info('preprocess done')
        return tokenized_inputs["input_ids"], tokenized_inputs["attention_mask"]

    def inference(self, inputs, *args, **kwargs):
        with torch.no_grad():
            results = self.model(inputs[0].to(self.device), inputs[1].to(self.device))
        logger.debug("Inference completed: output_shape=%s", tuple(results.shape))
        # Notice: here the results consist of logits and label
        return results

    def postprocess(self, outputs):
        # outputs is logits tensor of shape (batch_size, num_classes)
        results = F.softmax(outputs, dim=-1)
        results = results.tolist()
        logger.debug("Postprocessing completed: batch_size=%d", len(results))
        return map_class_to_label(results, self.mapping)
