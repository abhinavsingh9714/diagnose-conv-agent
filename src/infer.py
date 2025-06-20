from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

def load_model(base_model_name, lora_path):
    device = torch.device("cpu")
    dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype)
    base_model = base_model.to(device)

    model = PeftModel.from_pretrained(base_model, lora_path)
    # model = model.merge_and_unload()
    model.eval()

    return tokenizer, model

def infer(symptom_input, tokenizer, model):
    prompt = f"""### Instruction:
Ask relevant follow-up questions based on the patient's symptom description.

### Input:
{symptom_input}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cpu")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # response = response.split("### Response:")[-1].strip().split("###")[0].strip()

    return response

if __name__ == "__main__":
    BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    LORA_PATH = "./models/tinyllama_lora_adapter"  # ✅ Correct path

    tokenizer, model = load_model(BASE_MODEL, LORA_PATH)
    print(">>> Loaded LoRA model:", model.__class__)

    symptom_input = "I have a headache and feel dizzy."
    response = infer(symptom_input, tokenizer, model)
    print(f"\n🤖 Assistant: {response}")
