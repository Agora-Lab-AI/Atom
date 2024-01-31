from datasets import load_dataset
import jsonlines

# List of dataset names
datasets = [
    "kye/all-torvalds-c-code-1",
    "kye/all-huggingface-python-code-2",
    "kye/all-kye-python-code-2",
    "kye/all-allenai-python",
    "kye/all-huggingface-python-code",
    "kye/all-HazyResearch-python-code",
    "kye/all-meta-research-python-code",
    "kye/all-nvidia-python-code",
    "kye/all-deepmind-code-python",
    "kye/all-lucidrain-python-3",
    "kye/all-pytorch-code",
    "kye/all-openai-github-code",
    "kye/all-microsoft-python-code",
    "kye/all-google-ai-python-code",
    "kye/all-meta-research-code",
    "kye/all-Nikita-python-code",
    "kye/all-edwardzhang-python-code",
    "kye/all-kye-code",
    "kye/pytorch-repo-code",
    "kye/lucidrains-code-2",
    "kye/all-lucidrains-code",
    "kye/all-conceptofmind-code",
]


# Function to concatenate specific fields into a single 'text' field
def concatenate_fields(example):
    python_code = example["python_code"] if example["python_code"] is not None else ""
    repo_name = example["repo_name"] if example["repo_name"] is not None else ""
    file_path = example["file_path"] if example["file_path"] is not None else ""
    text = python_code + " " + repo_name + " " + file_path
    return {"text": text.strip()}


# Load and merge datasets
merged_data = []
for dataset_name in datasets:
    try:
        dataset = load_dataset(dataset_name)
        dataset = dataset.map(concatenate_fields)
        merged_data.extend(dataset)
    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {str(e)}")

# Write merged data to JSONL file
with jsonlines.open("merged_data.jsonl", mode="w") as writer:
    for example in merged_data:
        writer.write(example)
