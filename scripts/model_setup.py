import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from config import MODEL_NAME, LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES, MAX_SEQ_LENGTH

def setup():
    print("[model_setup] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"  Loading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    torch.set_float32_matmul_precision("high")

    return model, tokenizer

if __name__ == "__main__":
    setup()
