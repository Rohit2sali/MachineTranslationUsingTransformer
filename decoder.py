from Attention import MultiHeadAttention
from LayerNorm import LayerNorm
import torch.nn as nn

class Decoder(nn.Module): 
    def __init__(self, d_model, n_heads, fnn_hidden_dim, eps):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.maskedattention = MultiHeadAttention(d_model)
        self.cross_attention = MultiHeadAttention(d_model)

        self.fnn = nn.Sequential(
            nn.Linear(d_model, fnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fnn_hidden_dim, d_model)
        )

        self.layernorm1 = LayerNorm(d_model, eps)
        self.layernorm2 = LayerNorm(d_model, eps)
        self.layernorm3 = LayerNorm(d_model, eps)
        
    def forward(self, embedding_output, query, decoder_padding_mask, encoder_padding_mask, lookahead_mask):
        maskedattention_output = self.maskedattention.forward(embedding_output, embedding_output, embedding_output, decoder_padding_mask, lookahead_mask=lookahead_mask, n_heads=self.n_heads)
        input = maskedattention_output + embedding_output
        value = self.layernorm1.forward(input)
 
        attention_output = self.cross_attention.forward(query, query, value, encoder_padding_mask, lookahead_mask=None, n_heads=self.n_heads)
        attention_output = attention_output + value
        attention_output = self.layernorm2.forward(attention_output)

        output = self.fnn(attention_output)
        output = output + attention_output
        output = self.layernorm3.forward(output)

        return output  
