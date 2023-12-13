import torch
import warnings
from loguru import logger
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from config import *
import os

class Chatbot:
    def __init__(self, model_id):
        self.model = None
        self.tokenizer = None
        self.device = "cuda"
        self.initialize_chatbot(model_id)

    def initialize_chatbot(self, model_id):
        # Load the language model
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            cache_dir=MISTRAL_MODEL_PATH,
            use_auth_token=os.getenv("auth_token"),
        )

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, padding_side="left", use_auth_token=os.getenv("auth_token"), cache_dir=MISTRAL_TOKENIZER_PATH,

        )
