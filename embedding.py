import torch
import torch.nn as nn
import math
class word_embedding(nn.Module):
    def __init__(self, vocab_len, max_seq_len : int, d_model): 
        super(word_embedding, self).__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.embedding = torch.nn.Embedding(vocab_len, d_model)
        positional_embedding = torch.zeros(self.max_seq_len, self.d_model)
        for pos in range(self.max_seq_len):
            for i in range(0 , self.d_model, 2): 
                positional_embedding[pos][i] = math.sin(pos / (10000 ** ((2* i) / self.d_model)))
                positional_embedding[pos][i+1] = math.cos(pos / (10000 ** ((2*i+1) / self.d_model)))
        self.positional_embedding = positional_embedding.reshape(1, self.max_seq_len, self.d_model)

    def get_embedding(self, tokens):
        tokens = tokens.long()
        word_embedding = self.embedding(tokens)
        positional_embedding = self.positional_embedding.to(word_embedding.device)
        word_embedding = word_embedding + positional_embedding
        return word_embedding
