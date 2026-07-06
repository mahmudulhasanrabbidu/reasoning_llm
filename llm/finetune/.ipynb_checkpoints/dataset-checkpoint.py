import os
import json
import sys
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Safe import for Hugging Face datasets
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None
    print("Warning: 'datasets' library not found. Run 'pip install datasets' to download Alpaca.")


class Qwen3FineTuningDataset(Dataset):
    """
    A PyTorch Dataset designed for Supervised Fine-Tuning (SFT) of the Qwen3 model.
    It automatically handles ChatML formatting, tokenization, padding, and loss masking.
    """
    def __init__(self, data_list, tokenizer, max_seq_len=512):
        """
        Args:
            data_list (list of dicts): E.g., [{"prompt": "...", "response": "..."}]
            tokenizer (Qwen3Tokenizer): Your custom tokenizer instance.
            max_seq_len (int): The maximum number of tokens allowed per sequence.
        """
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        # -100 is the default ignore_index in PyTorch's CrossEntropyLoss
        self.ignore_index = -100 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        user_text = item["prompt"]
        assistant_text = item["response"]

        # 1. Format the prompt using the tokenizer's built-in wrapper
        # This adds <|im_start|>user\n ... <|im_end|>\n<|im_start|>assistant\n
        prompt_text = self.tokenizer._wrap_chat(user_text)
        prompt_ids = self.tokenizer.encode(prompt_text, chat_wrapped=False)
        
        # 2. Tokenize the target response and append the EOS token
        # The EOS token is critical; it teaches the model when to STOP generating.
        response_ids = self.tokenizer.encode(assistant_text, chat_wrapped=False) + [self.tokenizer.eos_token_id]

        # 3. Combine them into the final sequence
        input_ids = prompt_ids + response_ids
        
        # 4. Create the labels array
        # We mask out the prompt tokens with -100 so the model doesn't calculate loss on them.
        labels = [self.ignore_index] * len(prompt_ids) + response_ids

        # 5. Truncate if the combined sequence is too long
        input_ids = input_ids[:self.max_seq_len]
        labels = labels[:self.max_seq_len]

        # 6. Pad if the sequence is too short
        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            labels = labels + [self.ignore_index] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


def download_and_format_alpaca(output_path="data/alpaca_training_data.json"):
    """Downloads the Alpaca dataset and formats it for our PyTorch Dataset."""
    if load_dataset is None:
        raise ImportError("Cannot download data without the 'datasets' library.")
        
    print("Downloading the cleaned Alpaca dataset from Hugging Face...")
    # 'yahma/alpaca-cleaned' is a highly regarded version of the original Stanford dataset
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")

    formatted_data = []
    
    print("Formatting data for Qwen3FineTuningDataset...")
    for example in dataset:
        # Some Alpaca examples have an 'input' context, others just have an 'instruction'
        if example["input"]:
            prompt = f"{example['instruction']}\n\nContext:\n{example['input']}"
        else:
            prompt = example["instruction"]

        formatted_data.append({
            "prompt": prompt,
            "response": example["output"]
        })

    # Save to a local JSON file
    print(f"Saving {len(formatted_data)} examples to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, indent=4)

    print("Data preparation complete!\n")


# -------------------------------------------------------------------------
# Execution Block for Data Preparation and Testing
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Adjust paths assuming this is run from the project root or inside llm/
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from tokenizer.qween3_tokenizer import Qwen3Tokenizer
    
    # 1. Setup paths
    os.makedirs("data", exist_ok=True)
    save_path = os.path.join("data", "alpaca_training_data.json")
    
    # 2. Download and format data if it doesn't exist yet
    if not os.path.exists(save_path):
        download_and_format_alpaca(output_path=save_path)
    else:
        print(f"Found existing training data at: {save_path}\n")
        
    # 3. Load the prepared data
    with open(save_path, "r", encoding="utf-8") as f:
        training_data = json.load(f)
        
    # 4. Initialize Tokenizer
    tokenizer_path = "tokenizer.json"
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Please provide '{tokenizer_path}' in the root folder.")
    tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)
    
    # 5. Initialize the PyTorch Dataset
    print("Initializing PyTorch Dataset...")
    dataset = Qwen3FineTuningDataset(training_data, tokenizer, max_seq_len=256)
    print(f"Dataset successfully created with {len(dataset)} examples.\n")
    
    # 6. Verify the tensors of the first item
    sample = dataset[0]
    print("--- Verification: Sample 0 Tensors ---")
    print(f"Original Prompt Preview: {training_data[0]['prompt'][:50]}...")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Labels shape:    {sample['labels'].shape}")
    
    # 7. Print out the labels to visually verify the -100 masking worked
    print("\nLabels Array (Notice the -100s masking the prompt, followed by the answer IDs):")
    print(sample["labels"].tolist())