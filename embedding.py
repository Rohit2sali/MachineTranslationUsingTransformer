import torch
import torch.nn as nn

class word_embedding(nn.Module):
    def __init__(self, voocab_len, max_seq_len : int, d_model):
        super(word_embedding, self).__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.embedding = torch.nn.Embedding(voocab_len, d_model)

    def get_embedding(self, tokens):
        tokens = tokens.long()
        word_embedding = self.embedding(tokens)
        positional_embedding = torch.zeros(self.max_seq_len, self.d_model)
        for pos in range(self.max_seq_len):
            for i in range(self.d_model):
                angle = torch.tensor(pos / (10000 ** ((2* (i // 2)) / self.d_model)))
                if i % 2 == 0:
                    positional_embedding[pos][i] = torch.sin(angle)
                else:
                    positional_embedding[pos][i] = torch.cos(angle)
        positional_embedding = positional_embedding.reshape(1, self.max_seq_len, self.d_model)
        word_embedding += positional_embedding
        # shape of word embedding will be (batch_size, max_seq_len, d_model)
        return word_embedding

