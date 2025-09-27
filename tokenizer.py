import regex as re
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
from assignment1_basics.cs336_basics.pretokenization_example import find_chunk_boundaries
def train_bpe(input_path:str, vocab_size:int, special_tokens:list[str] = []) -> tuple[list[int,bytes], list[tuple[bytes,bytes]]]:
    vocab = {}
    merges = []
    token_dict = {}
    if vocab_size >= 256:
        #初始化vocab
        for i in range(256 + len(special_tokens)):
            if i < 256:
                vocab[i] = i.to_bytes()
            else:
                vocab[i] = special_tokens[i - 256].encode('utf-8')
        #pre-tokenize
        with open(input_path, "rb") as f:
            num_processes = 4
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
            f.seek(0)
            for i in range(num_processes):
                chunk = f.read(boundaries[i+1] - boundaries[i])
                sentences = chunk.split(b"<|endoftext|>")
                for sentence in sentences:
                    if sentence != b'':
                        tokens = re.finditer(PAT,sentence.decode())
                        for token in tokens:#token.group(0) -> str encode-> bytes
                            token_bytes = tuple(bytes([byte]) for byte in token.group(0).encode('utf-8'))
                            token_dict[token_bytes] = token_dict.get(token_bytes,0) + 1

        #merge
        times = vocab_size - 256 - len(special_tokens)
        for i in range(times):
            #init
            max = 0
            max_tuple = tuple()
            pair_counts = {}
            for token in token_dict: #count
                for j in range(len(token) - 1):
                    pair_counts[(token[j],token[j+1])] = pair_counts.get((token[j],token[j+1]),0) + token_dict[token]
                    if max < pair_counts[(token[j],token[j+1])]:
                        max = pair_counts[(token[j],token[j+1])]
                        max_tuple = (token[j],token[j+1])
                    elif max == pair_counts[(token[j],token[j+1])]:
                        bytes_current = b''.join((token[j],token[j+1]))
                        bytes_max = b''.join(max_tuple)
                        if bytes_max < bytes_current:
                            max_tuple = (token[j],token[j+1])

            tokens_toberemoved = []
            tokens_tobeadded = {}
            for token in token_dict: #renew
                token_merged = list()
                value = token_dict[token]
                sign = 0
                need_renew = 0
                for j in range(len(token)):
                    if sign == 0:
                        if token[j] == max_tuple[0]:
                            if j+1 <= len(token) - 1 and token[j+1] == max_tuple[1]:
                                sign = 1
                                need_renew = 1
                            else:
                                token_merged.append(token[j])
                        else:
                            token_merged.append(token[j])
                    else:
                        token_merged.append(b''.join(max_tuple))
                        sign = 0
                        continue
                if need_renew == 1:
                    tokens_toberemoved.append(token)
                    if token_merged[0] != b''.join(token):
                        tokens_tobeadded[tuple(token_merged)] = value
            
            for token in tokens_toberemoved:
                token_dict.pop(token)
            for token in tokens_tobeadded:
                token_dict[token] = tokens_tobeadded[token]

            merges.append(max_tuple)
            vocab[i + 256 + len(special_tokens)] = b''.join(max_tuple)
            #widely wi id de el ly -> for  {[w,i],[i,d],[d,e],[e,l],[l,y]}  when merge happens,e.g. id -> wi/de   
            # now {[w,i,d],[i,d,e],[e,l],[l,y]} affect nearby only 
            # do not happen across pre-tokens ->  check with every merge? pop()
    return vocab, merges

vocab, merges = train_bpe('data\\TinyStoriesV2-GPT4-valid.txt',2000)
