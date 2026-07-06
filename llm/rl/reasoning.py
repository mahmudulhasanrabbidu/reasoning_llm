import math
import torch
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

class ConversationalReasoner:
    def __init__(self, inference_engine):
        self.engine = inference_engine

    def _generate_silent(self, prompt, max_new_tokens, temperature, top_p):
        full_text = ""
        for token_text in self.engine.stream_generate(prompt, max_new_tokens, temperature, top_p):
            full_text += token_text
        return full_text

    def _build_critique_prompt(self, user_input, draft):
        return (
            "You are an expert editor and helpful AI assistant. Review the draft response below.\n"
            "Check for clarity, tone, helpfulness, and factual accuracy. "
            "If it is already excellent, say so briefly. Otherwise, provide a concise plan to improve it.\n\n"
            f"User Request:\n{user_input}\n\n"
            f"Draft Response:\n{draft}\n\n"
            "Critique (under 100 words):"
        )

    def _build_refine_prompt(self, user_input, draft, critique):
        return (
            "Revise the draft response based on the critique provided. "
            "Ensure the final response is polite, natural, and directly addresses the user's core request.\n\n"
            f"User Request:\n{user_input}\n\n"
            f"Draft Response:\n{draft}\n\n"
            f"Critique:\n{critique}\n\n"
            "Final Revised Response:"
        )

    def refine(self, user_input, iterations=1, max_response_tokens=512, max_critique_tokens=150, verbose=False, temperature=0.7, top_p=0.9):
        steps = []
        
        if verbose: 
            print("Generating initial draft...")
        current_draft = self._generate_silent(user_input, max_response_tokens, temperature, top_p)

        for it in range(iterations):
            draft_before = current_draft

            critique_prompt = self._build_critique_prompt(user_input, draft_before)
            if verbose: 
                print(f"Critiquing draft {it+1}/{iterations}...")
            critique_text = self._generate_silent(critique_prompt, max_critique_tokens, temperature, top_p)

            refine_prompt = self._build_refine_prompt(user_input, draft_before, critique_text)
            if verbose: 
                print(f"Writing final revision {it+1}/{iterations}...")
            revised_text = self._generate_silent(refine_prompt, max_response_tokens, temperature, top_p)

            step = {
                "iteration": it + 1,
                "draft": draft_before,
                "critique": critique_text,
                "revised": revised_text,
            }
            steps.append(step)

            if verbose:
                print(
                    f"\n[Editing Step {it+1}/{iterations}]"
                    f"\nOriginal Draft: {draft_before.strip()}"
                    f"\nCritique: {critique_text.strip()}"
                    f"\nFinal Revision: {revised_text.strip()}"
                    f"\n{'=' * 40}"
                )

            current_draft = revised_text

        return {
            "final_response": current_draft,
            "steps": steps,
        }


class ReasoningMetrics:
    def __init__(self, inference_engine):
        self.engine = inference_engine
        self.device = self.engine.device
        self.model = self.engine.model
        self.tokenizer = self.engine.tokenizer

    @torch.inference_mode()
    def calc_next_token_probas(self, prompt):
        token_ids = torch.tensor(self.tokenizer.encode(prompt), device=self.device)
        logits = self.model(token_ids.unsqueeze(0)).squeeze(0)
        all_probas = torch.softmax(logits, dim=-1)

        t_idx = torch.arange(0, token_ids.shape[0] - 1, device=self.device)
        next_ids = token_ids[1:]
        next_token_probas = all_probas[t_idx, next_ids]

        print("Next-token probabilities:", [p.item() for p in next_token_probas])
        print("Joint probability:", torch.prod(next_token_probas))

    @torch.inference_mode()
    def calc_next_token_logprobas(self, prompt, show=True):
        token_ids = torch.tensor(self.tokenizer.encode(prompt), device=self.device)
        logits = self.model(token_ids.unsqueeze(0)).squeeze(0)
        
        all_logprobas = torch.log_softmax(logits, dim=-1)
        t_idx = torch.arange(0, token_ids.shape[0] - 1, device=self.device)
        next_ids = token_ids[1:]
        next_token_logprobas = all_logprobas[t_idx, next_ids]

        sum_next_token_logprobas = torch.sum(next_token_logprobas)

        if show:
            print("Next-token log-probabilities:", next_token_logprobas)
            print("Joint log-probability:", sum_next_token_logprobas)
        else:
            return next_token_logprobas, sum_next_token_logprobas

    @torch.inference_mode()
    def avg_logprob_answer(self, prompt, answer):
        prompt_ids = self.tokenizer.encode(prompt)
        answer_ids = self.tokenizer.encode(answer)
        full_ids = torch.tensor(prompt_ids + answer_ids, device=self.device)

        logits = self.model(full_ids.unsqueeze(0)).squeeze(0)
        logprobs = torch.log_softmax(logits, dim=-1)

        start = len(prompt_ids) - 1
        end = full_ids.shape[0] - 1

        t_idx = torch.arange(start, end, device=self.device)
        next_tokens = full_ids[start + 1 : end + 1]
        next_token_logps = logprobs[t_idx, next_tokens]

        return torch.mean(next_token_logps).item()


if __name__ == "__main__":
    import os
    from llm.inference import Qwen3InferenceEngine
    from tokenizer.qween_3_tokenizer import Qwen3Tokenizer
    from llm.pretrained_weight_loader import PretrainedQweenModel
    
    tokenizer_path = "tokenizer.json"
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Please provide '{tokenizer_path}' inside the root folder.")

    print("Loading Core Architecture...")
    tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)
    model, _ = PretrainedQweenModel.from_pretrained("config.json", "model.safetensors")
    base_engine = Qwen3InferenceEngine(model=model, tokenizer=tokenizer)
    
    agent = ConversationalReasoner(base_engine)
    metrics = ReasoningMetrics(base_engine)
    
    user_question = "Explain quantum computing in one short paragraph for a 10-year-old."
    
    print("\n--- Starting Agentic Refinement Loop ---")
    results = agent.refine(
        user_input=user_question,
        iterations=1,
        max_response_tokens=150,
        max_critique_tokens=100,
        verbose=True
    )
    
    print("\nFINAL DELIVERABLE:")
    print(results["final_response"])