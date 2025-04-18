import torch.nn as nn
from Attention import MultiHeadAttention

class Decoder(nn.Module): 
    def __init__(self, d_model, n_heads, fnn_hidden_dim):
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

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)

    def forward(self, embedding_output, key, decoder_padding_mask, encoder_padding_mask, lookahead_mask):
        x = self.layernorm1(embedding_output)
        maskedattention_output = self.maskedattention(x, x, x, decoder_padding_mask, lookahead_mask=lookahead_mask, n_heads=self.n_heads)
        attention_output = maskedattention_output + embedding_output
        
        query = self.layernorm2(attention_output)
        cross_attention_output = self.cross_attention(query, key, key, encoder_padding_mask, lookahead_mask=None, n_heads=self.n_heads)
        attention_output = cross_attention_output + attention_output

        output = self.fnn(self.layernorm3(attention_output))
        output = output + attention_output

        return output   
