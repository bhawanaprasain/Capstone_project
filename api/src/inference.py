import torch
import warnings
from loguru import logger
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from src.config import *
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

    def get_response(self, query, message_history, distance, context):
        try:
            if distance and context and distance < 1.5:
                query = f"""
                        As an AI language model use the following context for answering FAQ from user.
                        Give a short answer. Do not add up anything on your own and do not try to give generic answer by using your knowledge.
                        Context: {context} Question: {query}
                        """
                logger.info(query)
                message_history.append({"role": "user", "content": query})
                encodeds = self.tokenizer.apply_chat_template(
                    message_history, return_tensors="pt"
                )
                model_inputs = encodeds.to(self.device)
                generated_ids = self.model.generate(
                    model_inputs, max_new_tokens=300, do_sample=True
                )
                decoded = self.tokenizer.batch_decode(generated_ids)
                return decoded[0]
            else:
                prompt = "As a healthcare AI assistant, prioritize empathetic responses to user questions. Do not provide personal information. When handling complaints or user queries with insufficient information, ask them relevant follow-up questions to understand their problem. Maintain user engagement by consistently asking meaningful follow-up questions when necessary. Now respond to the query from user. Query: "
                initial_followup_prompt = "Ask a necessary follow-up question in case of complains and unclear explanation about the condition, to keep user engaged. If query from user has enough information, give necessary suggestion for the user's query."
                if len(message_history) == 0:
                    message_history.append({"role": "user", "content": prompt +  initial_followup_prompt + query})
                elif len(message_history) < 4:
                    message_history.append({"role": "user", "content":  query})
                else:
                    message_history.append({"role": "user", "content":   query})
                encodeds = self.tokenizer.apply_chat_template(
                    message_history, return_tensors="pt"
                )
                model_inputs = encodeds.to(self.device)
                generated_ids = self.model.generate(
                    model_inputs, max_new_tokens=150, do_sample=True
                )
                decoded = self.tokenizer.batch_decode(generated_ids)
                return decoded[0]

        except Exception as e:
            logger.error(e)




