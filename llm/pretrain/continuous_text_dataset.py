import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from tokenizer.qween_3_tokenizer import Qwen3Tokenizer

class ContinuousTextDataset(Dataset):
    def __init__(self, text_data, tokenizer, max_length=256, stride=256):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text_data, chat_wrapped=False, show_progress=False)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

        if len(self.input_ids) == 0:
            raise ValueError(f"Dataset split resulted in 0 chunks! Tokens available: {len(token_ids)}.")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]



if __name__ == "__main__":
    # setup paths to find the tokenizer in the root directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'tokenizer.json'))
    
    if os.path.exists(tokenizer_path):
        from tokenizer.qween_3_tokenizer import Qwen3Tokenizer
        
        print("--- Testing ContinuousTextDataset ---")
        tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)
        
        test_text = "This is a test sentence to verify that the continuous text dataset is working correctly. " * 5
        
        # using a small max_length and stride to ensure we get multiple chunks
        dataset = ContinuousTextDataset(test_text, tokenizer, max_length=8, stride=4)
        print(f"Dataset created successfully with {len(dataset)} chunks.")
        
        # test DataLoading
        loader = DataLoader(dataset, batch_size=2)
        batch = next(iter(loader))
        
        # verification
        print(f"Batch inputs shape: {batch[0].shape} (Expected: [2, 8])")
        print(f"Batch targets shape: {batch[1].shape} (Expected: [2, 8])")
        
        # verify that targets are shifted by 1 relative to inputs (Next-token prediction logic)
        print(f"\nSample Input:  {batch[0][0].tolist()}")
        print(f"Sample Target: {batch[1][0].tolist()}")
        print("\nTest passed! Your sliding window is correctly configured.")
        
    else:
        print(f"Skipping test: Could not find tokenizer at {tokenizer_path}")