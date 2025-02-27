max_seq_len = 12
n_heads = 8
d_model = 512
fnn_hidden_dim = 2048
n_layers = 2
eps = 1e-9
batch_size = 4

input = ["i am a boy", "i live in india", "i am here doing this", "hey whats up"]
target = ["how are you", "how about you", "i am fine", "hey where are you going"]

transformer = Transformer(input, target, max_seq_len, d_model, n_heads, n_layers, fnn_hidden_dim, eps)
output, target_tokens, _ = transformer.forward(input, target, generate=False)