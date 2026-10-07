

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch

model_path="ibm-granite/granite-3.3-2b-instruct"
if torch.backends.mps.is_available():
    device = torch.device("mps")
    torch_dtype = torch.float16
elif torch.cuda.is_available():
    device = torch.device("cuda")
    torch_dtype = torch.bfloat16
else:
    device = torch.device("cpu")
    torch_dtype = torch.float32

print(f"Using device: {device}")
model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
    )
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(
        model_path
)

conv = [{"role": "user", "content":"a=3, b=5, result=a+b, what is result??"}]

input_ids = tokenizer.apply_chat_template(conv, return_tensors="pt", thinking=False, return_dict=True, add_generation_prompt=True).to(device)

set_seed(42)
output = model.generate(
    **input_ids,
    max_new_tokens=8192,
)

prediction = tokenizer.decode(output[0, input_ids["input_ids"].shape[1]:], skip_special_tokens=True)
print(prediction)