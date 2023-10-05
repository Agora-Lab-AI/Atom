import argparse
import logging
import multiprocessing
from itertools import chain

from datasets import load_dataset
from transformers import AutoTokenizer


class LLamaTokenizer:
    def __init__(self):
        self.tokenizer= AutoTokenizer.from_pretrained(
            "conceptofmind/Yarn-Llama-2-13b-64k",
            eos_token="<eos>",
            pad_token="<pad>",
            extra_ids=0,
            model_max_length=8192
        )

    def tokenize_texts(self, texts):
        return self.tokenizer(
            texts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True
        ).input_ids
    
    def decode(self, texts):
        return self.tokenizer.decode(texts)
    
    def __len__(self):
        num_tokens = len(self.tokenizer)
        return num_tokens



class CFG:
    SEED: int = 42
    SEQ_LEN: int = 8192
    NUM_CPU: int = multiprocessing.cpu_count()
    HF_ACCOUNT_REPO: str = "kye/all-lucidrain-code-python-tokenized-65536"
    TOKENIZER: str = "conceptofmind/Yarn-Llama-2-13b-64k"
    DATASET_NAME: str = "kye/all-lucidrain-python-3"

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(CFG.TOKENIZER)
    train_dataset = load_dataset(CFG.DATASET_NAME, split="train")

    def tokenize_function(example):
        return tokenizer([t + tokenizer.eos_token for t in example["python_code"]])
    
    tokenized_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=CFG.NUM_CPU,
        remove_columns=["repo_name", "python_code", "file_path"],
    )

    block_size = CFG.SEQ_LEN

    # Main data processing function that will concatenate all texts from our dataset 
    # and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it 
        # instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    train_tokenized_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=CFG.NUM_CPU,
    )

    train_tokenized_dataset.push_to_hub(CFG.HF_ACCOUNT_REPO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process and push dataset to Hugging Face Hub")
    parser.add_argument("--seed", type=int, default=CFG.SEED, help="Random seed")
    parser.add_argument("--seq_len", type=int, default=CFG.SEQ_LEN, help="Sequence length for processing")
    parser.add_argument("--hf_account", type=str, default=CFG.HF_ACCOUNT_REPO, help="Hugging Face account name and repo")
    parser.add_argument("--tokenizer", type=str, default=CFG.TOKENIZER, help="Tokenizer model to use")
    parser.add_argument("--dataset_name", type=str, default=CFG.DATASET_NAME, help="Name of the dataset to process")
    args = parser.parse_args()
    main(args)