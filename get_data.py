import torch
english = "path\english"
french = "path\french"

class get_data:
    def __init__(self, max_seq_len):
        tokenizer = Tokenization(max_seq_len)
        train_data = tokenizer.tokenize(english, layer=None)
        target_train_data = tokenizer.tokenize(french, layer="decoder")
       
        self.val_data = train_data[:5000].detach().clone().cpu()
        self.target_val_data = target_train_data[:5000].detach().clone().cpu()
        self.test_data = train_data[5000:10000].detach().clone().cpu()
        self.target_test_data = target_train_data[5000:10000].detach().clone().cpu()

        self.train_data = train_data[10000:]
        self.target_train_data = target_train_data[10000:]

    def forward(self):
        return self.train_data, self.target_train_data, self.val_data, self.target_val_data, self.test_data, self.target_test_data
