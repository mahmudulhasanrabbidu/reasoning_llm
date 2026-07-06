import unicodedata
import json

def get_stat(ids:list, count:dict=None) -> dict:
    '''
    ids = [10, 20, 20, 10]
    count = {
            (10, 20): 1,
            (20, 20): 1,
            (20, 10): 1
            }
    '''
    count = {} if count is None else count
    for pair in zip(ids, ids[1:]):
        count[pair] = count.get(pair, 0) + 1
    return count

    

def merge(ids:list, pair:tuple, idx:int) -> list:
    '''
    ids  = [10, 20, 30, 20, 30]
    pair = (20, 30)
    idx  = 999
    Output: [10, 999, 999]
    '''
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2 # skip both elements of the pair
        else:
            newids.append(ids[i])
            i += 1 # move to next element
    return newids

# helpfull to visualize vocab
def replace_control_character(s:str) -> str:
    '''
     Replace all Unicode control characters in a string with visible \\uXXXX escapes.
     Non-control characters are kept unchanged.
    '''
    chrs = []
    for c in s:
        if unicodedata.category(c)[0] != "C":
            # category starts with "C" for control chars (Cc, Cf, Cs, Co, Cn)
            chrs.append(c)
        else:
            chrs.append(f"\\u{ord(c):04x}")

    return "".join(chrs)

# helpfull to visualize vocab
def render_token(t: bytes) -> str:
    """
    Convert a byte token into a safe, printable string.
    - First decode bytes → UTF-8 string (invalid bytes become replacement chars)
    - Replace control characters (like \n, \t, etc.) with visible \uXXXX form so the output never breaks the terminal.
    """
    s = t.decode(encoding="utf-8", errors="replace")
    return replace_control_character(s)


class Tokenizer:
    def __init__(self):
        self.merge_tkns = {} # (int, int) -> int
        self.special_tkns = {} # str -> int
        self.patterns = "" # regex used in pre-tokenization
        self.vocab = self._build_vocab()

    def train(self, text, vocab_size, verbose):
        raise NotImplementedError

    def encode(self, text):
        raise NotImplementedError

    def decode(self, tokens):
        raise NotImplementedError

        

    def _build_vocab(self) -> dict:
        """
        Rebuild vocabulary from:
            - base 256 byte tokens
            - learned merge tokens
            - special tokens
        """
        
        vocab = {idx: bytes([idx]) for idx in range(256)}
        
        for pair, idx in sorted(self.merge_tkns.items(), key=lambda x: x[1]):
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        for tkn, idx in self.special_tkns.items():
            # special not in vocab so convert it into bytes by .encode()
            # Special tokens are strings (e.g. "<|endoftext|>"), 
            # so we encode them to bytes to match the rest of the vocab.
            vocab[idx] = tkn.encode("utf-8")

        return vocab



    def save(self, file_prefix):
        model_file = file_prefix + ".model"
        with open(model_file, "w", encoding="utf-8") as f:
            # Write model metadata (needed to reload the tokenizer)
            # write the version, pattern, merge_tokens, special_tokens
            f.write(f"minbpe_v1\n")
            # json.dumps: to safely escape newlines/quotes in the pattern
            f.write(f"{json.dumps(self.patterns)}\n")
            f.write(f"{len(self.special_tkns)}\n")
            for tkn, idx in self.special_tkns.items():
                # IDs must never change → store token + ID
                f.write(f"{idx} {tkn}\n")
    
            for pair, idx in sorted(self.merge_tkns.items(), key=lambda x: x[1]):
                # IDs are determined by merge order → only store pair
                # the ID is just 256 + line_number
                f.write(f"{pair[0]} {pair[1]}\n")
    
    
    
        # Write human-readable vocabulary file (.vocab) ---
        # This file is not used for loading; it's only for debugging/inspection.
        # Shows how tokens map to bytes or merged symbols.
        vocab_file = file_prefix + ".vocab"
        inverted_merge = {idx: pair for pair, idx in self.merge_tkns.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, bts in self.vocab.items():
                s = render_token(bts)
                # for normal token write ([a] 97) and for marged token ([t][h] -> [th] 256)
                if idx in inverted_merge:
                    idx0, idx1 = inverted_merge[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")



    def load(self, model_file):
        assert model_file.endswith(".model")
        merge_tkns = {} # (int, int) → new token ID
        special_tkns = {} # token → ID
        merge_idx = 256 # merged tokens always start after raw bytes 0–255
        with open(model_file, 'r', encoding="utf-8") as f:
            version = f.readline().strip()
            assert version == "minbpe_v1"
            patterns = f.readline().strip()
            num_special_tkns = int(f.readline().strip())
            for _ in range(num_special_tkns):
                # If your special token has a space (e.g., "<|end of text|>"), split() will break it into 3+ parts and crash your loader
                line = f.readline().strip()
                tkn, idx = line.rsplit(' ', 1)
                special_tkns[tkn] = int(idx)

            # the rest of the file contain only merge tokens (each line "p0 p1")
            for line in f:
                p0, p1 = map(int, line.strip().split())
                merge_tkns[(p0, p1)] = merge_idx
                merge_idx += 1

        # Write loaded data into the tokenizer instance
        self.patterns = patterns
        self.merge_tkns = merge_tkns
        self.special_tkns = special_tkns
        # Rebuild vocab from merges + special tokens
        self.vocab = self._build_vocab()