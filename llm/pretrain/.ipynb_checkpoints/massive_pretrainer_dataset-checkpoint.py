import os
import sys
import math
import torch
from torch.utils.data import IterableDataset, DataLoader

# adjust paths to look two folders up
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from tokenizer.qween_3_tokenizer import Qwen3Tokenizer

# safe import for Hugging Face datasets
try:
    from datasets import load_dataset
except ImportError:
    raise ImportError("Please run 'pip install datasets' to stream Wikipedia.")

class StreamingPretrainDataset(IterableDataset):
    def __init__(self, dataset_path, dataset_name, tokenizer, max_length=1024, split="train"):
        super().__init__()
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split

    def __iter__(self):
        """
        pulls articles one-by-one, tokenizes them, and packs them into a buffer. once the buffer is full, it yields a batch.
        """
        # downloads data chunk-by-chunk
        streamed_dataset = load_dataset(self.dataset_path, self.dataset_name, split=self.split, streaming=True)
        # check if PyTorch is using multiple workers
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            # if multiple workers exist, split the stream so they don't read the same articles
            # Hugging Face provides a built-in sharding method for streaming datasets!
            streamed_dataset = streamed_dataset.shard(num_shards=worker_info.num_workers, index=worker_info.id)
        
        token_buffer = []
        
        for article in streamed_dataset:
            # extract the text
            text = article["text"]
            
            # tokenize the text (chat_wrapped=False because this is raw knowledge, not chat)
            token_ids = self.tokenizer.encode(text, chat_wrapped=False, show_progress=False)
            
            # append an EOS token so the model learns where documents end
            token_ids.append(self.tokenizer.eos_token_id)
            
            # add the tokens to our running buffer
            token_buffer.extend(token_ids)
            
            # pack and Yield: While we have enough tokens to create a full sequence
            # we need max_length + 1 tokens to create inputs and shifted targets
            while len(token_buffer) >= self.max_length + 1:
                # extract the exact length needed
                chunk = token_buffer[:self.max_length + 1]
                
                # remove those tokens from the buffer
                token_buffer = token_buffer[self.max_length:]
                
                # create inputs and shifted targets for Next-Token Prediction
                inputs = chunk[:-1]
                targets = chunk[1:]
                
                yield torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


# testing the Stream
if __name__ == "__main__":
    print("Initializing Wikipedia Streaming Pipeline...")
    
    # setup Tokenizer
    tokenizer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'tokenizer.json'))
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Please provide '{tokenizer_path}' in the root folder.")
    
    tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)
    
    # initialize Streaming Dataset
    # we use "wikipedia" and "20220301.en" for the standard English Wikipedia corpus
    dataset = StreamingPretrainDataset(dataset_path="wikipedia", dataset_name="20220301.en", tokenizer=tokenizer, max_length=256)
    
    # wrap in a standard PyTorch DataLoader
    # iterableDatasets do not support 'shuffle=True' in the DataLoader natively
    dataloader = DataLoader(dataset, batch_size=2)
    
    # test the Stream
    print("\nStarting the stream... (Notice how memory usage stays tiny!)")
    
    for batch_idx, (input_batch, target_batch) in enumerate(dataloader):
        print(f"\n--- Batch {batch_idx + 1} ---")
        print(f"Input Shape:  {input_batch.shape}")
        print(f"Target Shape: {target_batch.shape}")
        
        # decode the first sequence in the batch to verify it works
        sample_text = tokenizer.decode(input_batch[0].tolist())
        print(f"\nDecoded Text Preview:\n{sample_text[:150]}...\n")
        
        # stop after 3 batches so we don't accidentally read the whole internet
        if batch_idx == 2:
            print("Stream test successful! Halting download.")
            break