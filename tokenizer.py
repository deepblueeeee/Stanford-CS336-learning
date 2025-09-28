import regex as re
import heapq
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
from assignment1_basics.cs336_basics.pretokenization_example import find_chunk_boundaries

class MaxStr:
    def __init__(self, s):
        self.s = s

    def __lt__(self, other):
        return self.s > other.s
    
    def __eq__(self, other):
        self.s == other.s

    def __repr__(self):
        return f"MaxStr({self.s!r})"

def train_bpe(input_path:str, vocab_size:int, special_tokens:list[str] = []) -> tuple[list[int,bytes], list[tuple[bytes,bytes]]]:
    vocab = dict()
    merges = list()
    token_counts = dict()
    token_split = dict()
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
                            token_bytes = token.group(0).encode('utf-8')
                            token_counts[token_bytes] = token_counts.get(token_bytes,0) + 1
        byte_list = list(bytes([i]) for i in range(256))
        for token in token_counts:
            token_split[token] = list(byte_list[i] for i in token)

        pair_counts = dict()
        bp_location = dict()
        for token in token_split: #construct pair-counts
            for index in range(len(token_split[token]) - 1):
                pair_counts[(token_split[token][index],token_split[token][index+1])] = pair_counts.get((token_split[token][index],token_split[token][index+1]),0) + token_counts[token]
                bp_location[(token_split[token][index],token_split[token][index+1])] = bp_location.get((token_split[token][index],token_split[token][index+1]),list())+[token]

        max_heap = [(-value, MaxStr(key)) for key,value in pair_counts.items()]#max_heap based on pair_counts
        heapq.heapify(max_heap)
        times = vocab_size - 256 - len(special_tokens)
        for i in range(times):
            while True:
                count, pair = heapq.heappop(max_heap)
                count = -count
                pair = pair.s
                if count == pair_counts.get(pair, 0):
                    break
            pair_merged = b''.join(pair)
            merges.append(pair)
            vocab[i + 256 + len(special_tokens)] = b''.join(pair)
            pairs_toBeAdded = set()
            #renew: 1、pair  2、counts
            for token in bp_location[pair]:
                j = 0   #统计新增pair减少pair
                while j < len(token_split[token]) - 1:
                    if token_split[token][j] == pair[0] and token_split[token][j + 1] == pair[1]:
                        token_split[token].pop(j)
                        token_split[token].pop(j)
                        token_split[token].insert(j,pair_merged)
                        #renew pair_counts new = sum(affected_token); old = old - sum(affected_token)
                        if j - 1 >= 0:
                            pair_front = (token_split[token][j-1],pair_merged)
                            pair_counts[pair_front] = pair_counts.get(pair_front,0) + token_counts[token]
                            bp_location[pair_front] = bp_location.get(pair_front,list())+[token]
                            pairs_toBeAdded.add(pair_front)
                            pair_affected = (token_split[token][j-1],pair[0])
                            pair_counts[pair_affected] -= token_counts[token]
                            pairs_toBeAdded.add(pair_affected)
                        if j + 1 <= len(token_split[token]) - 1:
                            pair_back = (pair_merged,token_split[token][j+1])
                            pair_counts[pair_back] = pair_counts.get(pair_back,0) + token_counts[token]
                            bp_location[pair_back] = bp_location.get(pair_back,list())+[token]
                            pairs_toBeAdded.add(pair_back)
                            pair_affected = (pair[1],token_split[token][j+1])
                            pair_counts[pair_affected] -= token_counts[token]
                            pairs_toBeAdded.add(pair_affected)
                    j += 1
            #更新heap
            for pair_add in pairs_toBeAdded:
                if pair_counts[pair_add] != 0:
                    heapq.heappush(max_heap,(-pair_counts[pair_add],MaxStr(pair_add)))
                else:
                    pair_counts.pop(pair_add)
                    bp_location.pop(pair_add)
            
            pair_counts.pop(pair)
            bp_location.pop(pair)
            #dict(bp:dict(token:index))
            #wit
            #widely wi id de el ly -> for  {[w,i],[i,d],[d,e],[e,l],[l,y]}  when merge happens,e.g. id -> wi/de   
            # now {[w,i,d],[i,d,e],[e,l],[l,y]} affect nearby only 
            # do not happen across pre-tokens ->  check with every merge? pop()
    return vocab, merges

vocab, merges = train_bpe('data\\TinyStoriesV2-GPT4-valid.txt',5000)
vocab
