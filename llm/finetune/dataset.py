import os
import json
import sys
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None
    print("Warning: 'datasets' library not found. Run 'pip install datasets' to download Alpaca.")


class Qwen3FineTuningDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_seq_len=512):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.ignore_index = -100 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        user_text = item["prompt"]
        assistant_text = item["response"]

        prompt_text = self.tokenizer._wrap_chat(user_text)
        prompt_ids = self.tokenizer.encode(prompt_text, chat_wrapped=False)
        response_ids = self.tokenizer.encode(assistant_text, chat_wrapped=False) + [self.tokenizer.eos_token_id]

        input_ids = prompt_ids + response_ids
        labels = [self.ignore_index] * len(prompt_ids) + response_ids

        input_ids = input_ids[:self.max_seq_len]
        labels = labels[:self.max_seq_len]

        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            labels = labels + [self.ignore_index] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


def download_and_format_alpaca(output_path="data/alpaca_training_data.json"):
    if load_dataset is None:
        raise ImportError("Cannot download data without the 'datasets' library.")
        
    print("Downloading the cleaned Alpaca dataset from Hugging Face...")
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")

    formatted_data = []
    
    print("Formatting data for Qwen3FineTuningDataset...")
    for example in dataset:
        if example["input"]:
            prompt = f"{example['instruction']}\n\nContext:\n{example['input']}"
        else:
            prompt = example["instruction"]

        formatted_data.append({
            "prompt": prompt,
            "response": example["output"]
        })

    print(f"Saving {len(formatted_data)} examples to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, indent=4)

    print("Data preparation complete!\n")


if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from tokenizer.qween_3_tokenizer import Qwen3Tokenizer
    
    os.makedirs("data", exist_ok=True)
    save_path = os.path.join("data", "alpaca_training_data.json")
    
    if not os.path.exists(save_path):
        download_and_format_alpaca(output_path=save_path)
    else:
        print(f"Found existing training data at: {save_path}\n")
        
    with open(save_path, "r", encoding="utf-8") as f:
        training_data = json.load(f)
        
    tokenizer_path = "tokenizer.json"
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Please provide '{tokenizer_path}' in the root folder.")
    tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)
    
    print("Initializing PyTorch Dataset...")
    dataset = Qwen3FineTuningDataset(training_data, tokenizer, max_seq_len=256)
    print(f"Dataset successfully created with {len(dataset)} examples.\n")
    
    sample = dataset[0]
    print("--- Verification: Sample 0 Tensors ---")
    print(f"Original Prompt Preview: {training_data[0]['prompt'][:50]}...")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Labels shape:    {sample['labels'].shape}")
    
    print("\nLabels Array (Notice the -100s masking the prompt, followed by the answer IDs):")
    print(sample["labels"].tolist())