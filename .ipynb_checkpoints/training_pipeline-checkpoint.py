from qwen3 import KVCache
import warnings
import torch
import math
from dataclasses import dataclass
from typing import Optional



@dataclass
class Config:
    temperature: float = 0.5
    top_p: float = 0.8
    max_token: int = 1024
    eos_token: Optional[int] = None


class GptTrainingPipeLine:
    def __init__(self, config):
        self.config = config

    def scale_logits_by_temperature(self, logits):
        # logits: (B, vocab_size)
        tmp = self.config.temperature
        if tmp <= 0:
            raise ValueError("Temperature must be positive")
        return logits / tmp # (B, vocab_size)

    def top_p_filter(self, probas):
        # probas: (B, vocab_size)
        sorted_probas, sorted_ids = torch.sort(probas, dim=1, descending=True) # (B, vocab_size)
        
        cum_prob = torch.cumsum(sorted_probas, dim=1) # (B, vocab_size)
        prefix = cum_prob - sorted_probas # (B, vocab_size)
        musk = prefix < self.config.top_p # (B, vocab_size)
        
        musk[:, 0] = True
        
        keep_sorted = torch.where(musk, sorted_probas, torch.zeros_like(probas)) # (B, vocab_size)
        filtered_prob = torch.zeros_like(probas).scatter(dim=1, index=sorted_ids, src=keep_sorted) # (B, vocab_size)

        su_m = torch.sum(filtered_prob, dim=1, keepdim=True).clamp_min(1e-12) # (B, 1)

        return filtered_prob / su_m # (B, vocab_size)

    @torch.inference_mode()
    def generate_text_top_p_stream_cache(self, model, token_ids):
        # token_ids: (B, seq_len)
        model.eval()
        
        cache = KVCache(n_layers=model.cfg["n_layers"])
        model.reset_kv_cache()

        output = model(token_ids, cache=cache)[:, -1, :] # (B, seq_len, vocab_size) --> (B, vocab_size)
        
        for _ in range(self.config.max_token):
            if self.config.temperature is None or self.config.temperature == 0.0:
                next_tokens = torch.argmax(output, dim=-1, keepdim=True) # (B, 1)
            else:
                logits = self.scale_logits_by_temperature(output) # (B, vocab_size)
                probs = torch.softmax(logits, dim=-1) # (B, vocab_size)
                top_prob = self.top_p_filter(probs) # (B, vocab_size)
                next_tokens = torch.multinomial(top_prob, num_samples=1) # (B, 1)

            if self.config.eos_token is not None and torch.all(next_tokens == self.config.eos_token):
                break

            yield next_tokens # (B, 1)
            
            output = model(next_tokens, cache=cache)[:, -1, :] # (B, 1, vocab_size) --> (B, vocab_size)

    @torch.inference_mode()
    def generate_text_stream_concat_flex(self, model, tokenizer, prompt, device, max_new_tokens, verbose=False, generate_func=None, **generate_kwargs):
        if generate_func is None:
            generate_func = self.generate_text_top_p_stream_cache

        input_ids = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0) # (1, seq_len)

        generated_ids = []
        for token in generate_func(model=model, token_ids=input_ids, max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id, **generate_kwargs): # token: (1, 1)
            next_token_id = token.squeeze(0) # (1,)
            generated_ids.append(next_token_id.item())

            if verbose:
                print(tokenizer.decode(next_token_id.tolist()), end="", flush=True)
        return tokenizer.decode(generated_ids)

    def supervisor_synthesis_loop(self, supervisor_model, worker_models, tokenizer, prompt, device, score_fn=None):
        """
        Args:
            supervisor_model (object): A highly capable model responsible for editing and final synthesis.
            worker_models (list): A list of initialized models (e.g., small or diverse LLMs) that will generate the initial attempts.
        """
        
        # Workers generate their answers
        worker_answers = []
        for i, w_model in enumerate(worker_models):
            ans = self.generate_text_stream_concat_flex(model=w_model, tokenizer=tokenizer, prompt=prompt, device=device, max_new_tokens=1024)
            worker_answers.append(ans)

        # Evaluate and format the worker answers
        compiled_worker_text = ""
        for idx, ans in enumerate(worker_answers):
            score = score_fn(ans, prompt) if score_fn else "N/A"
            compiled_worker_text += f"--- Worker {idx+1} (Score: {score}) ---\n{ans}\n\n"

        # The Supervisor Prompt
        supervisor_prompt = f"""
        You are an expert Supervisor AI. Your task is to provide the perfect final answer to the user's prompt.
        
        USER PROMPT: {prompt}
        
        Below are several attempts by junior worker models to answer this prompt, along with their automated scores. 
        Review their logic, identify any mathematical or logical errors, and combine their best insights into a single, flawless, final response.
        
        WORKER ATTEMPTS:
        {compiled_worker_text}
        
        SUPERVISOR FINAL ANSWER:
        """

        # Supervisor generates the final synthesis
        final_synthesis = self.generate_text_stream_concat_flex(model=supervisor_model, tokenizer=tokenizer, prompt=supervisor_prompt, device=device, max_new_tokens=2048)

        return final_synthesis