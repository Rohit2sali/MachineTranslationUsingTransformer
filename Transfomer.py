from tokenizer import Tokenization
from embedding import word_embedding
from encoder import Encoder
from decoder import Decoder
import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, input, target, max_seq_len, d_model, n_heads, n_layers, fnn_hidden_dim, eps):
        super(Transformer, self).__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.tokenizer = Tokenization(input + target)
        vocab_len = self.tokenizer.get_vocab_len()

        self.embedding = word_embedding(vocab_len, max_seq_len, d_model)

        self.encoders = nn.ModuleList([Encoder(d_model, n_heads, fnn_hidden_dim, eps) for _ in range(n_layers)])

        self.decoders = nn.ModuleList([Decoder(d_model, n_heads, fnn_hidden_dim, eps) for _ in range(n_layers)])

        self.linear = nn.Linear(self.d_model, vocab_len)

    def padding_mask(self, tokens, padding_token):
        batch_size, max_seq_len = tokens.shape
        padding_mask = (tokens != padding_token)
        padding_mask = padding_mask.view(batch_size, 1, 1, max_seq_len)
        padding_mask = padding_mask.expand(-1, -1, max_seq_len, -1).float()
        mask = padding_mask.masked_fill(padding_mask == 0, float('-inf')).masked_fill(padding_mask == 1, 0.0)
        return mask

    def lookahead_mask(self, batch_size, max_seq_len):
        lookahead_mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1) # this creates an upper triangular matrix
        lookahead_mask[lookahead_mask == 1] = float('-inf') # makes all the upper triangular values to -inf
        lookahead_mask[lookahead_mask == 0] = 0
        lookahead_mask = lookahead_mask.view(1, 1, max_seq_len, max_seq_len)
        lookahead_mask = lookahead_mask.expand(batch_size, -1, -1, -1)
        return lookahead_mask
    
    def decode(self, predictions):
        output = self.tokenizer.decode(predictions)
        return output

    def forward(self, input, target, generate=False):
        tokens, vocab = self.tokenizer.tokenize(input, self.max_seq_len, generate)
        encoder_input = self.embedding.get_embedding(tokens)
        batch_size = len(tokens)
        eos_token = vocab["<eos>"]
        padding_token = vocab["<pad>"]
        padding_mask = self.padding_mask(tokens, padding_token)
        lookahead_mask = self.lookahead_mask(batch_size, self.max_seq_len) 

        encoder_output = self.encoders[0].forward(encoder_input, padding_mask, lookahead_mask)
        for encoder in self.encoders[1:]:
            encoder_output = encoder.forward(encoder_output, padding_mask, lookahead_mask)
         
        if(generate == True):
            decoder_tokens = "<sos>"
            for i in range(self.max_seq_len-1):
                target_tokens, vocab = self.tokenizer.tokenize(decoder_tokens, self.max_seq_len, generate)
                decoder_input = self.embedding.get_embedding(target_tokens)
                decoder_output = self.decoders[0].forward(decoder_input, encoder_output, padding_mask, lookahead_mask)
                for decoder in self.decoders[1:]:
                    decoder_output = decoder.forward(decoder_output, encoder_output, padding_mask, lookahead_mask)

                output = self.linear(decoder_output)
                prediction = torch.nn.functional.softmax(output, dim=-1)
                _, prediction = torch.max(prediction, dim=-1)
                prediction = self.decode(prediction[0])
                next_token  = prediction[i+1]
                decoder_tokens = decoder_tokens + ' ' + next_token
                if next_token == "<eos>":
                    break
            return prediction, decoder_tokens, _, _
        else:
            target_tokens, vocab = self.tokenizer.tokenize(target, self.max_seq_len, generate)
            decoder_input = self.embedding.get_embedding(target_tokens)

            decoder_output = self.decoders[0].forward(decoder_input, encoder_output, padding_mask, lookahead_mask)
            for decoder in self.decoders[1:]:
                decoder_output = decoder.forward(decoder_output, encoder_output, padding_mask, lookahead_mask)
        
        output = self.linear(decoder_output)
        output = output.permute(0, 2, 1)
        return output, target_tokens, eos_token, padding_token
