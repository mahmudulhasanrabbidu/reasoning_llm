from .base import Tokenizer, get_stat, merge

# only merge tokens
# no special tokens and regex pattern
class BasicTokenizer(Tokenizer):
    def __init__(self):
        # Initialize base Tokenizer fields (merge_tkns, vocab, patterns, etc.)
        super().__init__()

    def train(self, text:str, vocab_size:int, verbose=False):
        # BPE requires at least 256 raw byte tokens
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        # Using "replace" ensures invalid bytes won't crash training
        text_bytes = text.encode(encoding="utf-8", errors="replace")
        ids = list(text_bytes)

        merges = {} # store learned merges: (p0, p1) → new_id
        for i in range(num_merges):
            stats = get_stat(ids)
            # stats: {(a,b): count, ...}
            if not stats:
                if verbose: print("No more pairs left to merge. Stopping early.")
                break
            # Select the most frequent pair
            pair = max(stats, key = stats.get)
            if stats[pair] < 2:
                if verbose: print(f"This pair only occurred once. Stopping early.")
                break
            
            idx = 256 + i
            merges[pair] = idx
            ids = merge(ids, pair, idx)

            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} had {stats[pair]} occurrences")

        self.merge_tkns = merges # Store learned merges
        self.vocab = self._build_vocab() # Rebuild vocab using the learned merges and the parent class logic

    def decode(self, ids:list) -> str:
        # ids = [97, 300, 98] --> b"".join([b"a", b"ab", b"b"]) --> b"aabb" --> aabb
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        return text_bytes.decode(encoding="utf-8", errors="replace")


    def encode(self, text:str) -> list:
        """
        Converts a string of text into a list of integer token IDs 
        by iteratively applying the learned BPE merges.
        """
        text_bytes = text.encode(encoding="utf-8", errors="replace")
        ids = list(text_bytes)

        while len(ids) > 2:
            stats = get_stat(ids)
            # select the one learned earliest from the merge tokens in training
            pair = min(stats, key=lambda p: self.merge_tkns.get(p, float("inf")))
            if pair not in self.merge_tkns:
                break
            idx = self.merge_tkns[pair]
            ids = merge(ids, pair, idx)

        return ids