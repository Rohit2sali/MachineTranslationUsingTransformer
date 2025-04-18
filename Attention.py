import torch
import numpy as np
import torch.nn as nn
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model): # tokens should be of shape (batch_size, max_seq_len), the one before embedding
        super(MultiHeadAttention, self).__init__()
        self.w_q = torch.nn.Linear(d_model, d_model)
        self.w_k = torch.nn.Linear(d_model, d_model)
        self.w_v = torch.nn.Linear(d_model, d_model)
        self.w_out = torch.nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value, padding_mask, lookahead_mask, n_heads):
        batch_size, max_seq_len, d_model = query.shape
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)
        
        d_k = d_model // n_heads
        q = q.view(batch_size, max_seq_len, n_heads, d_k).permute(0, 2, 1, 3)
        k = k.view(batch_size, max_seq_len, n_heads, d_k).permute(0, 2, 1, 3)
        v = v.view(batch_size, max_seq_len, n_heads, d_k).permute(0, 2, 1, 3)
       
        attention_scores = torch.matmul(q, k.transpose(-2, -1))/ np.sqrt(d_k) # (-2, -1) will swap the last two dimension of the matrix
        padding_mask = padding_mask.to(attention_scores.device)
        
        if(lookahead_mask is not None):
            lookahead_mask = lookahead_mask.to(attention_scores.device)
            attention_scores = attention_scores + padding_mask + lookahead_mask
        else:
            attention_scores = attention_scores + padding_mask

        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention = torch.matmul(attention_weights, v)
        attention_output = attention.permute(0, 2, 1, 3).contiguous()
        attention_output = attention_output.view(batch_size, max_seq_len, d_model)
        output = self.w_out(attention_output)
        return output
