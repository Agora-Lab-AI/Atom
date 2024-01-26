from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig


model = AutoModelForCausalLM.from_pretrained(
    "NousResearch/Yarn-Llama-2-13b-64k", load_in_4bit=True
)

dataset = load_dataset("kye/all-lucidrain-code-python-tokenized-65536-1")


peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    model=model,
    trainer_dataset=dataset,
    max_seq_length=16384,
    peft_config=peft_config,
)

trainer.train()
