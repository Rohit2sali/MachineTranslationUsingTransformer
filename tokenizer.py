import torch
from collections import OrderedDict
import torch.nn as nn
import numpy as np

class Tokenization(nn.Module):
    def __init__(self, vocab, max_seq_len): # only string or list of strings is acceptable
        super(Tokenization, self).__init__()
        self.max_seq_len = max_seq_len
        self.vocab = vocab
        self.inv_vocab = {id : token for token, id in self.vocab.items()}
        
    def process_sentence(self, input_text, generate):
        token = input_text.lower().split()
        if(len(token) > self.max_seq_len -2):
            token = token[:self.max_seq_len-2] 
        if generate:
            token = token
        else:
            token = ["<sos>"] + token + ["<eos>"]
        token_ids = [self.vocab.get(token, self.vocab["<unk>"]) for token in token]
        token_ids = token_ids + [self.vocab["<pad>"]] * (self.max_seq_len - len(token_ids))
        return token_ids

    def tokenize(self, input_text, generate=False):
        tokens = []
        if (type(input_text) == str or type(input_text) == np.str_):
            tokens.append(self.process_sentence(input_text, generate))
        else:
            for sentence in input_text:
                tokens.append(self.process_sentence(sentence, generate))
        tokens = torch.tensor(tokens)
        return tokens
    
    def decode(self, token_ids, mode):
        tokens = token_ids.tolist()
        text = []
        if((type(tokens) == list) and (type(tokens[0]) != list)):
            text.append([self.inv_vocab.get(id) for id in tokens])
        else:
            for batch in tokens:
                token = [self.inv_vocab.get(id, self.vocab["<unk>"]) for id in batch]
                text.append(token) 
        if mode == "generate":
            text = text[0]
            for i in text:
                if i == "<sos>" or i == "<eos>" or i == "<unk>":
                    text.remove(i)
            text = " ".join(text)
        return text
