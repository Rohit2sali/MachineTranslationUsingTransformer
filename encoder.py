import torch
import torch.nn as nn
from Attention import MultiHeadAttention
class Encoder(nn.Module): 
    def __init__(self, d_model, n_heads, fnn_hidden_dim): # input must be str or list of str
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

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)

    def forward(self, embedding_output, padding_mask):
        x = self.layernorm1(embedding_output)
        attention_output = self.attention(x, x, x, padding_mask, lookahead_mask=None, n_heads=self.n_heads)
        attention_output = embedding_output + attention_output
        
        layernorm_output = self.layernorm2(attention_output)
        fnn_output = self.fnn(layernorm_output) 

        output = fnn_output + attention_output
        return output
