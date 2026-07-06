import regex as re
from .base import Tokenizer, get_stat, merge


# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""



class RegexTokenizer(Tokenizer):
    def __init__(self, pattern:str = None):
        super().__init__()
        self.patterns = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.patterns)
        self.special_tkns = {}
        self.inverse_special_tkns = {}

    def train(self, text:str, vocab_size:int, verbose:bool = None):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        # text: "Hello World"
        text_chunk = re.findall(self.compiled_pattern, text)
        # text_chunk: ["Hello", " ", "World"]
        ids = [list(ch.encode(encoding="utf-8", errors="replace")) for ch in text_chunk]
        # "Hello" -> [72, 101, 108, 108, 111], " "-> [32], "World" -> [87, 111, 114, 108, 100]
        # ids = [[72, 101, 108, 108, 111], [32], [87, 111, 114, 108, 100]]
        merges = {} # store learned merges: (p0, p1) → new_id
        for i in range(num_merges):
            stats = {}
            # stats: {(a,b): count, ...}
            for chunk_ids in ids:
                get_stat(chunk_ids, stats)
            if not stats:
                if verbose: print("No more pairs left to merge. Stopping early.")
                    break
                
            # Select the most frequent pair
            pair = max(stats, key=stats.get) # (int, int)
            if stats[pair] < 2:
                if verbose: print(f"Best pair {pair} only occurred once. Stopping early.")
                break

            idx = 256 + i
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            merges[pair] = idx

            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} had {stats[pair]} occurrences")

        self.merge_tkns = merges # Store learned merges
        self.vocab = self._build_vocab() # Rebuild vocab using the learned merges and the parent class logic
            

    def register_special_tokens(self, special_tkns:dict):
        # special_tkns: {"<|endoftext|>": 100257}
        self.special_tkns = special_tkns
        self.inverse_special_tkns = {v: k for k, v in self.special_tkns.items()}


    def decode(self, ids:list):
        part_bytes = []
        # store the byte sequences corresponding to each token ID.
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            else:
                raise ValueError(f"Invalid Token Id: {idx}")
        # Concatenate all byte sequences into one continuous bytes object.
        text_bytes = b"".join(part_bytes)
        # Convert the full byte sequence back into a UTF-8 string.
        text = text_bytes.decode(encoding="utf-8")
        return text


    def encode_ordinary(self, text):
        # Split text into chunks according to the compiled regex pattern.
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        # list of token ids for the whole text (built by flattening all chunk encodings).
        for chunk in text_chunks:
            # Convert the chunk string into raw UTF-8 bytes
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = list(chunk_bytes)

            # Continue merging while at least one pair exists
            while len(chunk_ids) > 2:
                # Compute all bigram frequencies inside this chunk (int, int)->int
                stats = get_stat(chunk_ids)
                # Select the pair with the *lowest* merge index
                pair = min(stats, key=lambda p: self.merge_tkns.get(p, float("inf")))
                if pair not in self.merge_tkns:
                    break

                idx = self.merge_tkns[pair]
                # Apply the merge and update the chunk’s id sequence
                chunk_ids = merge(chunk_ids, pair, idx)
            # Flattening instead of nesting
            ids.extend(chunk_ids)
        # Return the fully encoded list of token IDs (flatten)
        return ids




    def encode(self, text, allow_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allow_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """

        special = None
        # allow all special tokens
        if allow_special == "all":
            special = self.special_tkns
        # do not treat anything as special(use bytes label encoding for normal + special)
        elif allow_special == "none":
            special = {}
        # raise error if any special token appears in the text
        elif  allow_special == "none_raise":
            special = {}
            for token in self.special_tkns:
                if token in text:
                    raise ValueError(f"Special token '{token}' found in text but allow_special='none_raise'")
        # Only keep those special tokens allowed by user
        elif isinstance(allow_special, set):
            special = {k: v for k, v in self.special_tkns.items() if k in allow_special}
        # Invalid usage
        else:
            raise ValueError(f"allow_special= {allow_special} is not understood")

        # # If no special tokens are active, fall back to ordinary encoding
        if not special:
            return self.encode_ordinary(text)

        # # Construct a regex that matches any allowed special token
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        # Split text into normal chunks and special-token chunks
        special_text_chunks = re.split(special_pattern, text)

        ids = []
        for chunk in special_text_chunks:
            # If chunk is exactly a special token → map directly to its ID
            if chunk in special:
                ids.append(special[chunk])
            else:
                # Otherwise encode the chunk using the ordinary byte-level encode
                if chunk: # skip empty chunk
                    ids.extend(self.encode_ordinary(chunk))

        return ids


'''

| allowed_special   | special dictionary used    | split chunks                  | output IDs                    |
| ----------------- | -------------------------- | ----------------------------- | ----------------------------- |
| "all"             | {"<BOS>":100, "<EOS>":101} | ["A","<BOS>","B","<EOS>","C"] | [65,100,66,101,67]            |
| "none"            | {}                         | no split                      | raw bytes of full text        |
| "none_raise"      | {} + assertion             | raises AssertionError         | error                         |
| {"<BOS>"}         | {"<BOS>":100}              | ["A","<BOS>","B<EOS>C"]       | [65,100,66,60,69,79,83,62,67] |

'''
