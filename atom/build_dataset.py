import argparse
import multiprocessing
from itertools import chain
from transformers import AutoTokenizer
from datasets import load_dataset
import logging

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
        return self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True).input_ids
    
    def decode(self, texts):
        return self.tokenizer.decode(texts)
    
    def __len__(self):
        num_tokens = len(self.tokenizer)
        return num_tokens


# class BuildDataset:
#     def __init__(
#             self, 
#             seed=42, 
#             seq_len=65536, 
#             hf_account="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", 
#             dataset_name="kye/all-lucidrain-python-3",
#             dataset_names=["kye/all-lucidrain-python-3"],
#         ):
#         self.SEED = seed
#         self.SEQ_LEN = seq_len
#         self.NUM_CPU = multiprocessing.cpu_count()
#         self.HF_ACCOUNT_REPO = hf_account
#         self.DATASET_NAME = dataset_name
#         self.DATASET_NAMES = dataset_names
#         self.tokenizer = LLamaTokenizer.tokenize_texts

#     def tokenize_function(self, example):
#         return self.tokenizer([t + self.tokenizer.eos_token for t in example["text"]])

#     def group_texts(self, examples):
#         concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
#         total_length = len(concatenated_examples[list(examples.keys())[0]])
#         if total_length >= self.SEQ_LEN:
#             total_length = (total_length // self.SEQ_LEN) * self.SEQ_LEN
#         result = {
#             k: [t[i : i + self.SEQ_LEN] for i in range(0, total_length, self.SEQ_LEN)]
#             for k, t in concatenated_examples.items()
#         }
#         return result

#     def build(self):
#         for dataset_name in self.DATASET_NAMES:
#             try:
#                 logging.info(f"Processing dataset: {dataset_name}")
#                 train_dataset = load_dataset(dataset_name, split="train", streaming=True)
#                 tokenized_dataset = train_dataset.map(
#                     self.tokenize_function,
#                     batched=True,
#                     # num_proc=self.NUM_CPU,
#                     remove_columns=["text"],
#                 )
#                 train_tokenized_dataset = tokenized_dataset.map(
#                     self.group_texts,
#                     batched=True,
#                     # num_proc=self.NUM_CPU,
#                 )
#                 train_tokenized_dataset.save_to_disk(f"{dataset_name}-tokenized")
#                 train_tokenized_dataset.push_to_hub(self.HF_ACCOUNT_REPO)
#                 logging.info(f"Finished processing dataset: {dataset_name}")
#             except Exception as e:
#                 logging.error(f"Error processing dataset {dataset_name}: {e}")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Process and push dataset to Hugging Face Hub")
#     parser.add_argument("--seed", type=int, default=42, help="Random seed")
#     parser.add_argument("--seq_len", type=int, default=65536, help="Sequence length for processing")
#     parser.add_argument("--hf_account", type=str, default="YOUR HUGGINGFACE API KEY", help="Hugging Face account name and repo")
#     parser.add_argument("--dataset_name", type=str, default="kye/all-lucidrain-python-3", help="Name of the dataset to process")
#     parser.add_argument("--dataset_names", type=str, nargs='+', default=["kye/all-lucidrain-python-3"], help="Names of the datasets to process")
#     args = parser.parse_args()
#     dataset_builder = BuildDataset(seed=args.seed, seq_len=args.seq_len, hf_account=args.hf_account, dataset_names=args.dataset_names)
#     dataset_builder.build()

class CFG:
    SEED: int = 42
    SEQ_LEN: int = 8192
    NUM_CPU: int = multiprocessing.cpu_count()
    HF_ACCOUNT_REPO: str = "kye/all-lucidrain-code-python-tokenized-8192"
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
        remove_columns=["python_code"],
    )

    block_size = CFG.SEQ_LEN

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
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