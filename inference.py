import torch
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model_path = "/content/drive/MyDrive/finetuning_results_NEFTune2"  # Update as needed

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # Change to torch.float32 if using CPU
    device_map={"": 0}          # Use device_map="auto" if unsure
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ------------------- GENERATION FUNCTION ------------------- #
def generate_response(prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        do_sample=False,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.0
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
