from Attention import MultiHeadAttention
from LayerNorm import LayerNorm
import torch.nn as nn

class Encoder(nn.Module): 
    def __init__(self, d_model, n_heads, fnn_hidden_dim, eps): # input must be str or list of str
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads 
        self.attention = MultiHeadAttention(d_model)

        self.fnn = nn.Sequential(
            nn.Linear(d_model, fnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fnn_hidden_dim, d_model)
        )

        self.layernorm1 = LayerNorm(d_model, eps)
        self.layernorm2 = LayerNorm(d_model, eps)

    def forward(self, embedding_output, padding_mask):
        attention_output = self.attention.forward(embedding_output, embedding_output, embedding_output, padding_mask, lookahead_mask=None, n_heads=self.n_heads)
        input = embedding_output + attention_output 

        layernorm_output = self.layernorm1.forward(input)
        input = self.fnn(layernorm_output) 

        input = input + layernorm_output
        input = self.layernorm2.forward(input)
        return input
