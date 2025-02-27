import torch
import torch.nn as nn


class MaskedAttention:
    def __init__(self, d_model): # input should be of shape (batch_size, maxz_seq_len), one before embedding
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)

    def forward(self, input, lookahead_mask, n_heads):
        # print("input", input)
        batch_size, max_seq_len, d_model = input.shape
        q = self.w_q(input)
        k = self.w_k(input)
        v = self.w_v(input)
        d_k = d_model // n_heads
        q = q.view(batch_size, max_seq_len, n_heads, d_k).permute(0, 2, 1, 3)
        k = k.view(batch_size, max_seq_len, n_heads, d_k).permute(0, 2, 1, 3)
        v = v.view(batch_size, max_seq_len, n_heads, d_k).permute(0, 2, 1, 3)
        

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_model, dtype=torch.float32))
        
        attention_scores = attention_scores + lookahead_mask

        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

        attention_output = torch.matmul(attention_weights, v)

        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()

        attention_output = attention_output.reshape(batch_size, max_seq_len, d_model)

        output = self.w_out(attention_output)

        return output


# a  = torch.tensor([[[1, 2, 3], [2, 3, 4], [5, 6, 4]],
#                   [[3, 2, 1], [2, 4, 1] ,[9, 8, 5]],
#                   [[6, 5, 2], [4, 7, 9], [7, 5, 4]]], dtype=torch.float32)

# b = torch.tensor([[1, 2, 3], 
#                   [2, 3, 4], 
#                   [5, 6, 4]])
# # print(b.shape)

# obj = MaskedAttention(b, 0, 0)
# output = obj.forward(a, 1)

# print(output.shape)

# from tokenizer import Tokenization
# from embedding import word_embedding

# a = ["hey whats up", "hey man what about you", "i am fine", "look at that"]
# b = "i am this"
# c = ["i am", " het what", "see that"]
# input = Tokenization(c)
# tokens, vocab = input.tokenize(c, 512)
# # print(input.decode(tokens))
# # print(tokens)
# obj = word_embedding(tokens, vocab, 512, 512)
# input = obj.get_embedding()


# atteniton = MaskedAttention(tokens, 0, 512)

# output = atteniton.forward(input, 8)
# print(type(output))
# print(output.shape)