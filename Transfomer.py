import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from embedding import word_embedding
from tokenization import Tokenization

class Transformer(nn.Module):
    def __init__(self, vocab_len, max_seq_len, d_model, n_heads, n_layers, fnn_hidden_dim):
        super(Transformer, self).__init__() 
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.embedding = word_embedding(vocab_len, max_seq_len, d_model)
        self.encoders = nn.ModuleList([Encoder(d_model, n_heads, fnn_hidden_dim) for _ in range(n_layers)])
        self.decoders = nn.ModuleList([Decoder(d_model, n_heads, fnn_hidden_dim) for _ in range(n_layers)])
        self.final_layernorm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(self.d_model, vocab_len)
        self.tokenizer = Tokenization(max_seq_len).get_tokenizer()
    
    def padding_mask(self, tokens):
        batch_size, max_seq_len = tokens.shape
        padding_mask = (tokens != self.tokenizer.pad_token_id)
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

    def generate(self, input_tokens):# input_tokens must be single sentence 
        encoder_output, encoder_padding_mask = self.encoder_output(input_tokens)
        batch_size, max_seq_len = input_tokens.shape
        lookahead_mask = self.lookahead_mask(batch_size, max_seq_len)
        target_tokens = torch.full((batch_size, max_seq_len), self.tokenizer.pad_token_id, device=input_tokens.device)
        target_tokens[:, 0] = self.tokenizer.bos_token_id
        for i in range(self.max_seq_len-1):
            decoder_padding_mask = self.padding_mask(target_tokens)
            decoder_input = self.embedding.get_embedding(target_tokens)
            decoder_output = self.decoders[0].forward(decoder_input, encoder_output, decoder_padding_mask, encoder_padding_mask, lookahead_mask)
            for decoder in self.decoders[1:]:
                decoder_output = decoder.forward(decoder_output, encoder_output, decoder_padding_mask, encoder_padding_mask, lookahead_mask)
            output = self.linear(self.final_layernorm(decoder_output)) # (batch_size, max_seq_len, d_model)
            next_token = torch.argmax(output[:, i, :], dim=-1)
            target_tokens[0][i+1] = next_token
            if (next_token == self.tokenizer.eos_token_id).any():
                return self.tokenizer.decode(target_tokens, skip_special_tokens=True)

        return self.tokenizer.decode(target_tokens, skip_special_tokens=True)

    def forward(self, input_tokens, target_tokens):
        encoder_output, encoder_padding_mask = self.encoder_output(input_tokens)
        target_tokens = torch.where(target_tokens == self.tokenizer.eos_token_id, torch.tensor(self.tokenizer.pad_token_id), target_tokens)
         
        decoder_input = self.embedding.get_embedding(target_tokens)
        batch_size, max_seq_len = target_tokens.shape
        lookahead_mask = self.lookahead_mask(batch_size, max_seq_len)
        decoder_padding_mask = self.padding_mask(target_tokens)

        decoder_output = self.decoders[0](decoder_input, encoder_output, decoder_padding_mask, encoder_padding_mask, lookahead_mask)
        for decoder in self.decoders[1:]:
            decoder_output = decoder(decoder_output, encoder_output, decoder_padding_mask, encoder_padding_mask, lookahead_mask)
       
        output = self.linear(self.final_layernorm(decoder_output))
        output = output.permute(0, 2, 1)
        
        return output
