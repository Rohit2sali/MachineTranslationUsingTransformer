!wget -O dataset_v1.zip "https://object.pouta.csc.fi/OPUS-ParaCrawl/v4/moses/en-fr.txt.zip"
!unzip -o dataset_v1.zip -d custom_folder_v1


en = "/parav5/ParaCrawl.en-fr.en"
fr = "/parav5/ParaCrawl.en-fr.fr"


from transformers import AutoTokenizer

tokenizer_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
tokenizer.add_special_tokens({'bos_token': '<sos>'})

print("\nSuccessfully loaded tokenizer.")
print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")


input_tokens = []
output_tokens = []
tokens_count = 0

for i in range(len(en)):
    a = tokenizer.encode(en[i])
    b = tokenizer.encode(fr[i])
    b = [tokenizer.bos_token_id] + b
    x, y = len(a), len(b)
    r = max(x / y, y / x)
    if((15 < x < 64) and (15 < y < 64) and r <= 2):
        c = x + y
        tokens_count += c
        a = a + [tokenizer.pad_token_id] * (64 - x)
        b = b + [tokenizer.pad_token_id] * (64 - y)
        input_tokens.append(a)
        output_tokens.append(b)

print("the no of tokens in both languages are : ",tokens_count)

# while training the model, data shuffling is very important, that is the reason we are saving the pairs in set of 1 million, so that 
# we will be able to select different sets from other datasets and whole training data will be well shuffled.
cnt = 0
till = 1000000
import torch
while(cnt < len(input_tokens)):
    a = input_tokens[cnt:till]
    b = output_tokens[cnt:till]
    cnt = till
    till += 1000000

    a = torch.tensor(a)
    b = torch.tensor(b)
    
    torch.save(a, f"parainput{cnt}.pt")
    torch.save(b, f"paraoutput{cnt}.pt")
