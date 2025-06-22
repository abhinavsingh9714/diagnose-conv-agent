from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, EarlyStoppingCallback
from peft import get_peft_model, LoraConfig, TaskType
import torch

def load_data(path="data/processed/output_dataset.jsonl"):
    dataset = load_dataset("json", data_files=path)["train"]
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    return dataset["train"], dataset["test"]

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

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    model = model.to(device)

    peft_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, peft_config)

    train_dataset, eval_dataset = load_data()
    train_dataset = train_dataset.map(lambda x: tokenize(x, tokenizer), remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(lambda x: tokenize(x, tokenizer), remove_columns=eval_dataset.column_names)
    train_dataset.set_format(type="torch")
    eval_dataset.set_format(type="torch")

    training_args = TrainingArguments(
        output_dir="./models/tinyllama_reverse",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=10,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=False,
        learning_rate=2e-5,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    trainer.model.save_pretrained("./models/tinyllama_lora_adapter")

if __name__ == "__main__":
    train()