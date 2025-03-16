from tokenizer import Tokenization
from embedding import word_embedding
from encoder import Encoder
from decoder import Decoder
import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, vocab, max_seq_len, d_model, n_heads, n_layers, fnn_hidden_dim, eps):
        super(Transformer, self).__init__() 
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.vocab = vocab
        self.embedding = word_embedding(len(self.vocab), max_seq_len, d_model)
        self.encoders = nn.ModuleList([Encoder(d_model, n_heads, fnn_hidden_dim, eps) for _ in range(n_layers)])
        self.decoders = nn.ModuleList([Decoder(d_model, n_heads, fnn_hidden_dim, eps) for _ in range(n_layers)])
        self.linear = nn.Linear(self.d_model, len(self.vocab))

    def padding_mask(self, tokens):
        padding_token = self.vocab["<pad>"]
        batch_size, max_seq_len = tokens.shape
        padding_mask = (tokens != padding_token)
        padding_mask = padding_mask.view(batch_size, 1, 1, max_seq_len)
        padding_mask = padding_mask.expand(-1, -1, max_seq_len, -1).float()
        mask = padding_mask.masked_fill(padding_mask == 0, float('-inf')).masked_fill(padding_mask == 1, 0.0)
        return mask

    def lookahead_mask(self, batch_size, max_seq_len):
        lookahead_mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1) # this creates an upper triangular matrix
        lookahead_mask[lookahead_mask == 1] = float('-inf') # makes all the upper triangular values to -inf
        lookahead_mask = lookahead_mask.view(1, 1, max_seq_len, max_seq_len)
        lookahead_mask = lookahead_mask.expand(batch_size, -1, -1, -1)
        return lookahead_mask

    def encoder_output(self, input_tokens):
        encoder_input = self.embedding.get_embedding(input_tokens)
        padding_mask = self.padding_mask(input_tokens)
        encoder_output = self.encoders[0].forward(encoder_input, padding_mask)
        for encoder in self.encoders[1:]:
            encoder_output = encoder.forward(encoder_output, padding_mask)
        return encoder_output, padding_mask

    def generate(self, input_tokens, tokenizer):# input_tokens must be single sentence 
        encoder_output, encoder_padding_mask = self.encoder_output(input_tokens)
        batch_size, max_seq_len = input_tokens.shape
        lookahead_mask = self.lookahead_mask(batch_size, max_seq_len)
        decoder_tokens = "<sos>"
        target_tokens = tokenizer.tokenize(decoder_tokens, generate=True)
        for i in range(self.max_seq_len-1):
            decoder_padding_mask = self.padding_mask(target_tokens)
            decoder_input = self.embedding.get_embedding(target_tokens)
            decoder_output = self.decoders[0].forward(decoder_input, encoder_output, decoder_padding_mask, encoder_padding_mask, lookahead_mask)
            for decoder in self.decoders[1:]:
                decoder_output = decoder.forward(decoder_output, encoder_output, decoder_padding_mask, encoder_padding_mask, lookahead_mask)
            output = self.linear(decoder_output)
            predictions = torch.nn.functional.softmax(output, dim=-1)
            _, prediction = torch.max(predictions, dim=-1)
            for i in prediction[0]:
                if i == self.vocab["<eos>"]:
                    return tokenizer.decode(prediction, mode="generate")
            target_tokens = prediction
        return tokenizer.decode(prediction, mode="generate")

    def forward(self, input_tokens, target_tokens):
        encoder_output, encoder_padding_mask = self.encoder_output(input_tokens)
         
        decoder_input = self.embedding.get_embedding(target_tokens)
        batch_size, max_seq_len = target_tokens.shape
        lookahead_mask = self.lookahead_mask(batch_size, max_seq_len)
        decoder_padding_mask = self.padding_mask(target_tokens)

        decoder_output = self.decoders[0].forward(decoder_input, encoder_output, decoder_padding_mask, encoder_padding_mask, lookahead_mask)
        for decoder in self.decoders[1:]:
            decoder_output = decoder.forward(decoder_output, encoder_output, decoder_padding_mask, encoder_padding_mask, lookahead_mask)
        
        output = self.linear(decoder_output)
        output = output.permute(0, 2, 1)
        
        return output, self.vocab["<eos>"], self.vocab["<pad>"]
