import torch
from Transfomer import Transformer
from get_data import data

max_seq_len = 10
n_heads = 8
d_model = 128
fnn_hidden_dim = 2048
n_layers = 2
eps = 1e-9
n_epoch = 5
batch_size = 5

a, b, train_data, val_data, test_data, target_train_data, target_val_data, target_test_data = data().get_data(4000)

model = Transformer(a, b, max_seq_len, d_model, n_heads, n_layers, fnn_hidden_dim, eps)

model.load_state_dict(torch.load('model.pth'))

model.eval()

prediction, decoder_tokens, _, _ = model('hey whats up how are you', '', generate=True)
print(prediction)
