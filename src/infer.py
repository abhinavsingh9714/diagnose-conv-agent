from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

from copy import deepcopy

def load_model(base_model_name, lora_path=None):
    device = torch.device("cpu")
    dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model once
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype)
    base_model = base_model.to(device)
    base_model.eval()

    # Clone before LoRA
    if lora_path:
        lora_model = deepcopy(base_model)
        lora_model = PeftModel.from_pretrained(lora_model, lora_path)
        lora_model = lora_model.merge_and_unload()
        lora_model.eval()
        return tokenizer, lora_model, base_model

    return tokenizer, None, base_model


def clean_output(raw_response):
    lines = raw_response.strip().split("\n")
    full_lines = []
    for line in lines:
        if len(line.split()) > 3 and line.strip()[-1] in ".?!":
            full_lines.append(line)
    return "\n".join(full_lines)

def infer(symptom_input, tokenizer, model, base_model):
    prompt = f"""### Instruction:
        Ask relevant follow-up questions based on the patient's symptom description.

        ### Input:
        {symptom_input}

        ### Response:
        """

    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cpu")

    with torch.no_grad():
        # Finetuned model (LoRA)
        lora_output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

        # Base model
        base_output = base_model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    lora_response = tokenizer.decode(lora_output[0], skip_special_tokens=True)
    base_response = tokenizer.decode(base_output[0], skip_special_tokens=True)

    # Strip to just the model's response portion
    lora_response = lora_response.split("### Response:")[-1].strip().split("###")[0].strip()
    base_response = base_response.split("### Response:")[-1].strip().split("###")[0].strip()
    return clean_output(lora_response), base_response

if __name__ == "__main__":
    BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    LORA_PATH = "./models/tinyllama_lora_adapter"

    tokenizer, lora_model, base_model = load_model(BASE_MODEL, LORA_PATH)
    print(">>> Loaded LoRA model:", lora_model.__class__)
    print(">>> Loaded Base model:", base_model.__class__)

    symptom_input = input("Patient: ")
    lora_resp, base_resp = infer(symptom_input, tokenizer, lora_model, base_model)
    print(f"\nLoRA-tuned Assistant:\n{lora_resp}\n")
    print(f"\nBase TinyLlama:\n{base_resp}\n")