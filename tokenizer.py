import torch
from collections import OrderedDict
import torch.nn as nn

class Tokenization(nn.Module):
    def __init__(self, input): # only string or list of strings is acceptable
        super(Tokenization, self).__init__()
        if not isinstance(input, (str, list)):
            raise ValueError("Input must be string or list of string")
        if isinstance(input, list) and not all(isinstance(s, str) for s in input):
            raise ValueError("if input is a list all elemensts in it must be a string")
        all_words = []
        if(type(input) == str):
            all_words = input.lower().split()
        else:
            for sentence in input:
                all_words += sentence.lower().split()
        unique_tokens = list(OrderedDict.fromkeys(all_words))
        unique_tokens.insert(0, "<sos>")
        unique_tokens.extend(["<eos>", "<unk>", "<pad>"])
        self.vocab = {token : id for id, token in enumerate(unique_tokens)}
        self.inv_vocab = {id : token for token, id in self.vocab.items()}

    def get_vocab_len(self):
        return len(self.vocab)
            
    def process_sentence(self, input_text, generate):
        token = input_text.lower().split()
        if(len(token) > self.max_seq_len -2):
            token = token[:self.max_seq_len-2]
        if generate:
            token = token
        else:
            token = ["<sos>"] + token + ["<eos>"]
        token_ids = [self.vocab.get(token, self.vocab["<unk>"]) for token in token]
        token_ids += [self.vocab["<pad>"]] * (self.max_seq_len - len(token_ids))
        return token_ids

    def tokenize(self, input_text, max_seq_len, generate):
        self.max_seq_len = max_seq_len
        tokens = []
        if (type(input_text) == str):
            tokens.append(self.process_sentence(input_text, generate))
        else:
            for sentence in input_text:
                tokens.append(self.process_sentence(sentence, generate))
        tokens = torch.tensor(tokens)
        return tokens, self.vocab
    
    def decode(self, token_ids):
        tokens = token_ids.tolist()
        # print("tokens", tokens)
        if((type(tokens) == list) and (type(tokens[0]) != list)):
            text = [self.inv_vocab.get(id) for id in tokens]
            return text
        else:
            text = []
            for batch in tokens:
                token = [self.inv_vocab.get(id) for id in batch]
                text.append(token)
        return text
        