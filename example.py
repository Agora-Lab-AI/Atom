import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("kye/Atom-Z-Tiny-7B")

model = AutoModelForCausalLM.from_pretrained(
  "kye/Atom-Z-Tiny-7B", 
  trust_remote_code=True, 
).to(device)

task = """


[INST] <<SYS>>
<function>Available functions:
<function>{
    "name": "generate_password",
    "description": "Generate a random password with specified criteria",
    "parameters": {
        "type": "object",
        "properties": {
            "length": {
                "type": "integer",
                "description": "The length of the password"
            },
            "include_numbers": {
                "type": "boolean",
                "description": "Include numbers in the password"
            },
            "include_special_characters": {
                "type": "boolean",
                "description": "Include special characters in the password"
            }
        },
        "required": [
            "length"
        ]
    }
}
<</SYS>>

I need a new password. Can you generate one for me? [/INST]


"""

input_ids = tokenizer.encode(task, return_tensors="pt")
output = model.generate(input_ids.to(device), max_length=128, temperature=0.7).cpu()
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_text)
