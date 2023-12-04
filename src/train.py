import torch
from src.config import *
from datasets import load_dataset
from peft import (LoraConfig, get_peft_model, prepare_model_for_kbit_training,)
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling)



class LoRAModelTrainer:
    """
    LoRAModelTrainer class for training LoRA (Low-Rank Adaptation) models on conversation datasets.

    Args:
        model_id (str): Identifier for the pretrained model to be fine-tuned.
        output_dir (str): Directory to save the trained model and related files.
        train_set_path (str): Path to the training dataset in JSON format.
        eval_set_path (str): Path to the evaluation dataset in JSON format.

    Attributes:
        model_id (str): Identifier for the pretrained model.
        output_dir (str): Directory to save the trained model and related files.
        train_set_path (str): Path to the training dataset in JSON format.
        eval_set_path (str): Path to the evaluation dataset in JSON format.
        model (AutoModelForCausalLM): Fine-tuned model for causal language modeling.
        tokenizer (AutoTokenizer): Tokenizer associated with the fine-tuned model.
        train_data (Dataset): Shuffled and tokenized training dataset.
        eval_data (Dataset): Shuffled and tokenized evaluation dataset.

    Methods:
        load_model(): Loads the pretrained model for fine-tuning and applies quantization.
        generate_prompt(data_point): Generates a conversation prompt from a given data point.
        load_datasets(): Loads and tokenizes training and evaluation datasets.
        train_model(): Trains the LoRA model using the specified training arguments.
        save_model(): Saves the trained model to the specified output directory.
    """

    def __init__(self, model_id, output_dir, train_set_path, eval_set_path):
        self.model_id = model_id
        self.output_dir = output_dir
        self.train_set_path = train_set_path
        self.eval_set_path = eval_set_path

    def load_model(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = prepare_model_for_kbit_training(self.model)
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
        )
        self.model = get_peft_model(self.model, config)

    def generate_prompt(self, data_point):
        return data_point["conversation"]

    def load_datasets(self):
        train_data = load_dataset("json", data_files=self.train_set_path)
        eval_data = load_dataset("json", data_files=self.eval_set_path)

        self.train_data = train_data.shuffle().map(
            lambda data_point: self.tokenizer(
                self.generate_prompt(data_point),
                truncation=True,
                max_length=960,
                padding="max_length",
            )
        )

        self.eval_data = eval_data.shuffle().map(
            lambda data_point: self.tokenizer(
                self.generate_prompt(data_point),
                truncation=True,
                max_length=960,
                padding="max_length",
            )
        )

    def train_model(self):
        trainer = Trainer(
            model=self.model,
            train_dataset=self.train_data["train"],
            eval_dataset=self.eval_data["train"],
            args=TrainingArguments(
                per_device_train_batch_size=12,
                gradient_accumulation_steps=12 // 4,
                warmup_steps=100,
                num_train_epochs=10,
                learning_rate=2e-5,
                fp16=True,
                logging_steps=10,
                save_steps=200,
                output_dir=self.output_dir,
                save_total_limit=2,
            ),
            data_collator=DataCollatorForLanguageModeling(
                self.tokenizer, mlm=False
            ),
        )
        self.model.config.use_cache = False
        trainer.train(resume_from_checkpoint=False)
        torch.cuda.empty_cache()

    def save_model(self):
        self.model.save_pretrained(self.output_dir)

# Train model:

loratrainer = LoRAModelTrainer(model_id, output_dir, train_set_path, eval_set_path)
loratrainer.load_model()
loratrainer.load_datasets()
loratrainer.train_model()
loratrainer.save_model()