from datasets import load_dataset
import jsonlines

# Load the dataset
dataset = load_dataset("glaiveai/glaive-function-calling-v2")

# Function to concatenate specific fields into a single 'text' field
def concatenate_fields(example):
    system = example['system'] if example['system'] is not None else ""
    stringlengths = example['stringlengths'] if example['stringlengths'] is not None else ""
    text = system + " " + stringlengths
    return {'text': text.strip()}

# Apply the function to the dataset
dataset = dataset.map(concatenate_fields)

# Write the dataset to a JSONL file
with jsonlines.open('glaive_function_calling_v2.jsonl', mode='w') as writer:
    for example in dataset:
        writer.write(example)