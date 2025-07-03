import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import json
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class SimpleDataCollator:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]

        max_len = max(len(ids) for ids in input_ids)

        padded_input_ids = []
        padded_labels = []
        attention_mask = []

        for ids, lbls in zip(input_ids, labels):
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [self.tokenizer.pad_token_id] * pad_len)
            padded_labels.append(lbls + [-100] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids),
            "labels": torch.tensor(padded_labels),
            "attention_mask": torch.tensor(attention_mask)
        }

CONFIG = {
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "data_file": "data/medical_compliance_deduplicated.csv",
    "output_dir": "qwen_simple_output",
    "max_length": 2048,
    "batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "num_epochs": 1,
    "warmup_steps": 25,
    "save_steps": 50,
    "eval_steps": 50,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "logging_steps": 50,
    "train_split": 0.95
}

def load_data():
    print("Loading data...")
    df = pd.read_csv(CONFIG["data_file"])

    if 'source_file' in df.columns:
        df = df.drop('source_file', axis=1)

    df = df.dropna(subset=['prompt', 'response'])
    print(f"Loaded {len(df)} rows")
    return df

def format_text(row):
    prompt = str(row['prompt'])
    response = str(row['response'])

    if 'source_input' in row and pd.notna(row['source_input']):
        context = str(row['source_input']).strip()
        if context:
            return f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer: {response}"

    return f"Question: {prompt}\n\nAnswer: {response}"

def setup_model_and_tokenizer():
    print("Loading model and tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float32
    )

    lora_config = LoraConfig(
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=CONFIG["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.train()

    print(f"Trainable parameters: {model.num_parameters(only_trainable=True)}")
    return model, tokenizer

def tokenize_data(examples, tokenizer):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=CONFIG["max_length"],
        return_tensors=None
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def main():
    df = load_data()
    df['text'] = df.apply(format_text, axis=1)
    dataset = Dataset.from_pandas(df[['text']])

    model, tokenizer = setup_model_and_tokenizer()

    tokenized_dataset = dataset.map(
        lambda x: tokenize_data(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )

    train_size = int(CONFIG["train_split"] * len(tokenized_dataset))
    train_dataset = tokenized_dataset.select(range(train_size))
    eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))

    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=CONFIG["num_epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        warmup_steps=CONFIG["warmup_steps"],
        logging_steps=CONFIG["logging_steps"],
        save_steps=CONFIG["save_steps"],
        eval_steps=CONFIG["eval_steps"],
        eval_strategy="steps",
        save_strategy="steps",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0
    )

    data_collator = SimpleDataCollator(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    with open(f"{CONFIG['output_dir']}/info.json", "w") as f:
        json.dump(CONFIG, f, indent=2)

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(CONFIG["output_dir"])

    print("Done!")

if __name__ == "__main__":
    main()