
import torch
import torch.nn as nn
import numpy as np
from transformers import MarianTokenizer

tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
tokenizer.add_special_tokens({'bos_token': '<sos>'})

class Tokenization(nn.Module):
    def __init__(self, max_seq_len):
        super(Tokenization, self).__init__()
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        
    def get_vocab_len(self):
        return len(self.tokenizer)

    def get_tokenizer(self):
        return self.tokenizer
    
    def process_sentence(self, input_sentences, layer):
        token_ids = self.tokenizer.encode(input_sentences.lower(), return_tensors=None)
        if layer == "decoder":
            if(len(token_ids) > self.max_seq_len - 1):
                token_ids = token_ids[:self.max_seq_len - 1]
            token_ids = [self.tokenizer.bos_token_id] + token_ids 
        else:
            if(len(token_ids) > self.max_seq_len):
                token_ids = token_ids[:self.max_seq_len]
            token_ids = token_ids
        
        token_ids += [self.tokenizer.pad_token_id] * (self.max_seq_len - len(token_ids)) 
        return token_ids
    
    def tokenize(self, input_text, layer):
        tokens = []
        if (type(input_text) == str or type(input_text) == np.str_):
            tokens.append(self.process_sentence(input_text.lower(), layer))
        else:
            for sentence in input_text:
                tokens.append(self.process_sentence(sentence.lower(), layer))
        tokens = torch.tensor(tokens)
        return tokens
    
    def decode(self, token_ids):
        output_tokens = token_ids.tolist()
        tokens = [token for token in output_tokens if token != self.tokenizer.pad_token_id]
        text = []
        if((type(tokens) == list) and (type(tokens[0]) != list)):
            text.append(self.tokenizer.decode(tokens, skip_special_tokens=True))
        else:
            for batch in tokens:
                token = [self.tokenizer.decode(batch, skip_special_tokens=True)]
                text.append(token) 
        return text
