from datasets import load_dataset
from transformer import Transformer
from tokenization import Tokenization
from sacrebleu import corpus_bleu
import torch

max_seq_len = 90
n_heads = 8
d_model = 512
fnn_hidden_dim = 2048
n_layers = 4
batch_size = 128

tokenizer = Tokenization(max_seq_len)
vocab_len = len(tokenizer)

model = Transformer(vocab_len, max_seq_len, d_model, n_heads, n_layers, fnn_hidden_dim)
model_checkpoint = torch.load(r"model.pth")
# Remove 'module.' prefix
new_state_dict = {}
for key, value in model_checkpoint.items():
    new_key = key.replace("module.", "")  # Remove 'module.' from the keys
    new_state_dict[new_key] = value

# Load into the model
model.load_state_dict(new_state_dict)    


dataset = load_dataset("wmt14", "fr-en")
test_set = dataset["test"]

src_sentences = [sample["translation"]["en"] for sample in test_set]
ref_sentences = [sample["translation"]["fr"] for sample in test_set]

predictions = [] 
references = []  
model.eval()

for i in range(len(src_sentences)):
    prediction = model.module.generate(tokenizer.tokenize(src_sentences[i], layer=None))
    references.append([ref_sentences[i]])
    prediction = prediction[0]
    predictions.append((prediction[0]))


bleu = corpus_bleu(predictions, references)
print(bleu)
