import torch
from Transfomer import Transformer
from tokenizer import Tokenization
from get_data import data
import pickle
max_seq_len = 64
n_heads = 8
d_model = 128
fnn_hidden_dim = 2048
n_layers = 4
eps = 1e-9

with open("vocab", "rb") as f:
  vocab = pickle.load(f)
  
tokenizer = Tokenization(vocab, max_seq_len)

model = Transformer(vocab, max_seq_len, d_model, n_heads, n_layers, fnn_hidden_dim, eps)

model.load_state_dict(torch.load('model.pth'))
model.eval()

input = tokenizer.tokenize("hey whats up how are you")

prediction, _, _ = model(input, tokenizer)
print(prediction)
