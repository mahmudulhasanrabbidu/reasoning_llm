import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm.pretrained_weight_loader import PretrainedQweenModel
from llm.base_model import KVCache
from tokenizer.qween_3_tokenizer import Qwen3Tokenizer



class Qwen3InferenceEngine:
    def __init__(self, model, tokenizer, device=None):
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.model.eval()

    def _sample_next_token(self, logits, temperature=1.0, top_p=0.9, generated_ids=None, repetition_penalty=1.2):
        # logits: (V,)

        # Repetition Penalty
        if generated_ids and repetition_penalty > 1.0:
            for idx in set(generated_ids):
                if logits[idx] < 0:
                    logits[idx] *= repetition_penalty
                else:
                    logits[idx] /= repetition_penalty

        # Early Exit for Greedy
        if temperature == 0.0:
            return torch.argmax(logits).item() # scalar (int)

        # Temperature Scaling
        logits = logits / temperature # Shape: (V,)

        # Top-P Filtering
        sorted_logits, sorted_logits_ids = torch.sort(logits, descending=True) # (V,), (V,)
        sorted_probs = torch.softmax(sorted_logits, dim=-1) # (V,)
        cum_sorted_probs = torch.cumsum(sorted_probs, dim=-1) # (V,)

        mask = cum_sorted_probs > top_p # (V,) (Boolean)
        mask[1:] = mask[:-1].clone() # (V,)
        mask[0] = False # (V,)                                          

        ids_to_remove = sorted_logits_ids[mask] # (num_items_dropped,)
        logits[ids_to_remove] = float('-inf') # (V,)

        # Final Sampling
        probs = torch.softmax(logits, dim=-1) # (V,)
        next_idx = torch.multinomial(probs, num_samples=1).item() # scalar(int)

        return next_idx

    def generate_stream(self, prompt, max_seq_len=200, temperature=1.0, top_p=0.9, repetition_penalty=1.2):
        input_ids = self.tokenizer.encode(prompt, chat_wrapped=False) # [seq_len]
        input_tensors = torch.tensor([input_ids], dtype=torch.long, device=self.device) # (1, seq_len)

        generated_ids = input_ids.copy() # [seq_len]

        cache = KVCache(self.model.config.num_trf_blocks)
        self.model.reset_kv_cache()

        with torch.no_grad():
            for _ in range(max_seq_len):
                logits = self.model(input_tensors, cache=cache) # (1, seq_len, v)
                next_token_logits = logits[0, -1, :] # (v,)

                # Fixed hardcoded variables
                next_idx = self._sample_next_token(logits=next_token_logits, temperature=temperature, top_p=top_p, generated_ids=generated_ids, 
                                                   repetition_penalty=repetition_penalty) # scalar(int)

                if next_idx in (self.tokenizer.eos_token_id, self.tokenizer.pad_token_id):
                    break

                next_token = self.tokenizer.decode([next_idx])
                yield next_token

                generated_ids.append(next_idx) # [seq_len + 1]
                input_tensors = torch.tensor([[next_idx]], dtype=torch.long, device=self.device) # (1, 1)

    def stream_to_console(self, prompt, max_new_tokens=200, temperature=0.7, top_p=0.9, repetition_penalty=1.2):
        """
        UI Wrapper: Consumes the stream_generate yield and prints directly to the terminal.
        """
        print(f"\n--- Streaming Output (Temp={temperature}, Top-p={top_p}) ---\n{prompt}", end="", flush=True)
        
        # Consume the generator loop with explicit kwargs
        for token_text in self.generate_stream(prompt=prompt, max_seq_len=max_new_tokens, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty):
            print(token_text, end="", flush=True)
            
        print("\n\n--- Generation Complete ---")
            
        

if __name__ == "__main__":
    tokenizer_path = "tokenizer.json"
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Please provide '{tokenizer_path}' inside the root folder.")
        
    print("Loading Tokenizer...")
    tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)
    
    print("Loading Model and Weights...")
    model, _ = PretrainedQweenModel.from_pretrained(config_path="llm/config.json", weight_path="llm/model.safetensors")
    
    engine = Qwen3InferenceEngine(model=model, tokenizer=tokenizer)
    print(f"Engine ready on device: {engine.device}")
    
    # Testing the wrapper
    test_prompt = "Deep learning is a subset of machine learning that"
    
    engine.stream_to_console(test_prompt, max_new_tokens=40, temperature=0.7, top_p=0.9)