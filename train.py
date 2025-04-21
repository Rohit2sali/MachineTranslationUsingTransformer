import torch
import math
import torch.nn as nn
from transformer import Transformer
from tokenization import Tokenization
from get_data import get_data
from torch.optim.lr_scheduler import LambdaLR

scaler = torch.cuda.amp.GradScaler()

def generate_text(predictions):
    if isinstance(predictions, torch.nn.parallel.DistributedDataParallel) or isinstance(predictions, torch.nn.DataParallel):
        predictions = predictions.module
    predictions = predictions.permute(0, 2, 1)
    predictions = torch.nn.functional.softmax(predictions, dim=-1)
    _, predictions = torch.max(predictions, dim=-1) 
    output = tokenizer.decode(predictions, skip_special_tokens=True)
    print(output[0])
    print(output[-1])

def accuracy_fn(prediction, target): # prediction : (batch_size, vocab, seq_len)
    prediction = prediction.permute(0, 2, 1)
    _, prediction = torch.max(prediction, dim=-1)
    dummy_col = torch.full((target.shape[0], 1), tokenizer.pad_token_id, device=target.device)
    target = target[:, 1:]
    target = torch.cat((target, dummy_col), dim=1)
    mask = target != tokenizer.pad_token_id
    correct = (target == prediction) & mask
    accuracy = correct.sum().item() / mask.sum().item()
    return accuracy

def calculate_loss(prediction, target_tokens):
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    dummy_col = torch.full((target_tokens.shape[0], 1), tokenizer.pad_token_id, device=target_tokens.device)
    target_tokens = target_tokens[:, 1:]
    target_tokens = torch.cat((target_tokens, dummy_col), dim=1)
    loss = criterion(prediction, target_tokens)
    return loss

def eval(val_input_tokens, val_target_tokens):
    model.eval()   
    with torch.no_grad():
        prediction = model(val_input_tokens, val_target_tokens)
        loss = calculate_loss(prediction, val_target_tokens)
        accuracy = accuracy_fn(prediction, val_target_tokens)
    return prediction, loss, accuracy

def get_scheduler(optimizer, warmup_steps, total_steps, base_lr=5e-4, min_lr=5e-6):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))  # Linear warmup to 1.0
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # in [0,1]
        scaled = cosine * (1.0 - min_lr / base_lr) + (min_lr / base_lr)  # in [min_lr/base_lr, 1.0]
        return scaled
    return LambdaLR(optimizer, lr_lambda)

def train(input_tokens, target_tokens):
    model.train()
    with torch.cuda.amp.autocast():
        prediction = model(input_tokens, target_tokens)
        loss = calculate_loss(prediction, target_tokens)
        accuracy = accuracy_fn(prediction, target_tokens)
    optimizer.zero_grad()
    scaler.scale(loss).backward()  # Scale loss to prevent underflow
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    scaler.step(optimizer)  # Update model weights
    scaler.update()
    scheduler.step()
    return loss, accuracy


def test(test_data, target_test_data):
    with torch.no_grad():
        prediction = model(test_data, target_test_data)
        loss = calculate_loss(prediction, target_test_data)
        acc = accuracy_fn(prediction, target_test_data)
    return prediction, loss, acc

if __name__ == "__main__":
    max_seq_len = 90
    n_heads = 8
    d_model = 512
    fnn_hidden_dim = 2048
    n_layers = 4
    n_epoch = 4
    batch_size = 128

    train_data, target_train_data, val_data, target_val_data, test_data, target_test_data = get_data().forward()

    indices = torch.randperm(train_data.shape[0])
    train_data = train_data[indices]
    target_train_data = target_train_data[indices]

    tokenizer = Tokenization(max_seq_len).get_tokenizer()
    vocab_len = len(tokenizer)

    model = Transformer(vocab_len, max_seq_len, d_model, n_heads, n_layers, fnn_hidden_dim)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.98), eps=1e-9)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
        
    model = model.to(device)

    # for 4 epochs 265625 steps
    scheduler = get_scheduler(optimizer, warmup_steps=3125, total_steps=265625, base_lr=5e-4, min_lr=1e-5)
    
    for epoch in range(n_epoch):
        train_loss = 0
        train_acc = 0
        for i in range(0, len(train_data), batch_size):
            input_tokens = train_data[i:i+batch_size].to(device)
            target_tokens = target_train_data[i:i+batch_size].to(device)
            loss, acc= train(input_tokens, target_tokens)
            train_loss += loss.item() 
            train_acc += acc
        
        torch.save(model.state_dict(), 'transformer.pth')
        torch.save(optimizer.state_dict(), 'optimizer.pth')
        torch.save(scheduler.state_dict(), "scheduler.pth")
        train_loss = train_loss / (len(train_data) / batch_size)
        train_acc = train_acc/(len(train_data) / batch_size)
        
        val_loss = 0
        val_acc = 0
        for i in range(0, len(val_data), batch_size):
            predictions, loss , acc = eval(val_data[i:i+batch_size].to(device), target_val_data[i:i+batch_size].to(device))
            val_loss += loss.item()
            val_acc += acc
        
        print(f"at epoch : {epoch}, train_loss : {train_loss}, train_acc : {train_acc}")
        print(f"at epoch :  val_loss : {val_loss / (len(val_data) / batch_size)}, val_acc : {val_acc / (len(val_data) / batch_size)}")

    test_loss = 0
    test_acc = 0
    for i in range(0, len(test_data), batch_size):
        predictions, loss, acc = test(test_data[i:i+batch_size].to(device), target_test_data[i:i+batch_size].to(device))
        test_loss += loss
        test_acc += acc
    print(f"at epoch : test_loss : {test_loss / (len(test_data)/batch_size)}, test_acc : {test_acc / (len(test_data)/batch_size)}")
