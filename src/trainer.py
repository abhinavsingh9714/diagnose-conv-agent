from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
import torch

def load_data(path="data/processed/output_dataset.jsonl"):
    dataset = load_dataset("json", data_files=path)
    return dataset["train"]

def format_prompt(example):
    instruction = example["instruction"]
    input_text = example["input"]
    output = example["output"]
    return f"""### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"""

def tokenize(example, tokenizer):
    prompt = format_prompt(example)
    return tokenizer(prompt, truncation=True, padding="max_length", max_length=512)

def train():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Force use of CPU
    device = torch.device("cpu")
    torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    model = model.to(device)

    # Apply LoRA
    peft_config = LoraConfig(
        r=64,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, peft_config)

    dataset = load_data()
    tokenized = dataset.map(lambda x: tokenize(x, tokenizer), remove_columns=dataset.column_names)
    tokenized.set_format(type="torch")

    training_args = TrainingArguments(
        output_dir="./models/tinyllama_reverse",
        per_device_train_batch_size=2,  # Reduce batch size for CPU
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,
        learning_rate=2e-5,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False  # Explicitly disable for CPU
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        train_dataset=tokenized,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    trainer.model.save_pretrained("./models/tinyllama_lora_adapter")

if __name__ == "__main__":
    train()