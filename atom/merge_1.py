from datasets import load_dataset
import jsonlines

# Load the dataset
# dataset = load_dataset("glaiveai/glaive-function-calling-v2")
dataset = load_dataset("rizerphe/glaive-function-calling-v2-llama")

# Select the 'train' split
dataset = dataset['train']

def concatenate_fields(example):
    chat = example['text'] if example['text'] is not None else ""
    return chat.strip()

# Apply the function to the dataset
dataset = dataset.map(concatenate_fields)

# Write the dataset to a JSONL file
with jsonlines.open('glaive_function_calling_v2.jsonl', mode='w') as writer:
    for example in dataset:
        writer.write(example)