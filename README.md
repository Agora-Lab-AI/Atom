[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Atom
a suite of finetuned LLMs for atomically precise function calling 🧪

✅ Massive function calling dataset of over 20M samples.

✅ First Model: Atom-Z-Tiny - Zephr trained on 100k samples

✅ Vision function calling coming soon


## Install

You can install the package using pip

```bash
pip install atom-torch
```

## Usage
```python
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


```

### Training

- To train the models, run `accelerate config` and enable DeepSpeed acceleration. `deepspeed/zero3.json` was the configuration file used for training.0
- Then run accelerate launch `accelerate launch finetune.py`


We're finetuning this [model](https://huggingface.co/NousResearch/Yarn-Llama-2-13b-64k) on this [dataset](https://huggingface.co/datasets/kye/all-lucidrain-code-python-tokenized-65536-1)

`sh./run.sh`

- [Dataset](kye/all-lucidrain-code-python-tokenized-65536-1)

### Citation

```bibtex
@misc{peng2023yarn,
    title={YaRN: Efficient Context Window Extension of Large Language Models}, 
    author={Bowen Peng and Jeffrey Quesnelle and Honglu Fan and Enrico Shippole},
    year={2023},
    eprint={2309.00071},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
