from Transfomer import Transformer
from get_data import data
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
scaler = torch.cuda.amp.GradScaler()

def generate_text(predictions):
    predictions = predictions.permute(0, 2, 1)
    predictions = torch.nn.functional.softmax(predictions, dim=-1)
    _, predictions = torch.max(predictions, dim=-1) 
    output = tokenizer.decode(predictions, mode="test")
    print(output[0])
    print(output[100])
    print(output[250])
    
def accuracy_fn(prediction, target, eos_token):   
    prediction = prediction.permute(0, 2, 1)
    prediction = torch.nn.functional.softmax(prediction, dim=-1)
    _, prediction = torch.max(prediction, dim=-1)
    m = 0
    for i in range(len(target)):
        for j in range(len(target[0])):
            if(target[i][j] == eos_token and j > m):
                    m = j
                    break

    target = target[:, :m+1]
    prediction = prediction[:, :m+1]
    correct = (target == prediction)
    accuracy = correct.sum().item() / torch.numel(target)
    return accuracy

def calculate_loss(prediction, target_tokens, pading_token):
    criterion = nn.CrossEntropyLoss(ignore_index=pading_token)
    loss = criterion(prediction, target_tokens)
    return loss

def eval(val_input_tokens, val_target_tokens):
    model.eval()    
    
    with torch.no_grad():
        prediction, eos_token, padding_token = model(val_input_tokens, val_target_tokens)
        loss = calculate_loss(prediction, val_target_tokens, padding_token)
        accuracy = accuracy_fn(prediction, val_target_tokens, eos_token)
    return prediction, loss, accuracy

def train(input_tokens, target_tokens):
    model.train()
    with torch.cuda.amp.autocast():
        prediction, eos_token, padding_token = model(input_tokens, target_tokens)
        loss = calculate_loss(prediction, target_tokens, padding_token)
        accuracy = accuracy_fn(prediction, target_tokens, eos_token)
    optimizer.zero_grad()
    scaler.scale(loss).backward()  # Scale loss to prevent underflow
    scaler.step(optimizer)  # Update model weights
    scaler.update()
    return loss, accuracy

def test(test_data, target_test_data):
    with torch.no_grad():
        prediction, eos_token, padding_token = model(test_data, target_test_data)
        
        test_loss = calculate_loss(prediction, target_test_data, padding_token)
        test_acc = accuracy_fn(prediction, target_test_data, eos_token)
        print("test_loss :",test_loss, "test_acc :", test_acc)
    return prediction

if __name__ == "__main__":
    max_seq_len = 64
    n_heads = 8
    d_model = 256
    fnn_hidden_dim = 2024
    n_layers = 4
    eps = 1e-9
    n_epoch = 10
    batch_size = 32

    with open("/kaggle/input/datasetfortranslation/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    train_data, val_data, test_data, target_train_data, target_val_data, target_test_data = data().get_data()
    
    tokenizer = Tokenization(vocab, max_seq_len)
    train_data = tokenizer.tokenize(train_data, generate=False)
    val_data = tokenizer.tokenize(val_data, generate=False)
    test_data = tokenizer.tokenize(test_data, generate=False)
    target_train_data = tokenizer.tokenize(target_train_data, generate=False)
    target_val_data = tokenizer.tokenize(target_val_data, generate=False)
    target_test_data = tokenizer.tokenize(target_test_data, generate=False)
    
    
    model = Transformer(vocab, max_seq_len, d_model, n_heads, n_layers, fnn_hidden_dim, eps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    best_val_loss = float("inf")
    n_batches = len(train_data) // batch_size
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for epoch in range(n_epoch):
        train_loss = 0
        train_acc = 0
        for i in range(0, len(train_data), batch_size):
            input_tokens = train_data[i:i+batch_size].to(device)
            target_tokens = target_train_data[i:i+batch_size].to(device)
            loss, acc= train(input_tokens, target_tokens)
            train_loss += loss.item() 
            train_acc += acc
            if(i % 1000 == 0):
                print("at this i :",i)

        train_loss = train_loss / (len(train_data) / batch_size)
        train_acc = train_acc/(len(train_data) / batch_size)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        predictions, val_loss , val_acc = eval(val_data.to(device), target_val_data.to(device))
        val_losses.append(val_loss.item())
        val_accuracies.append(val_acc)
        print(f"at epoch : {epoch}, train_loss : {train_loss}, train_acc : {train_acc}")
        print(f"at epoch : {epoch}, val_loss : {val_loss}, val_acc : {val_acc}")
        generate_text(predictions)
        torch.cuda.empty_cache()
        if (epoch == 1):
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'transformer_epoch{epoch + 1}.pth')
    predictions = test(test_data.to(device), target_test_data.to(device))
    generate_text(predictions)
    torch.save(model, 'model.pth')
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, n_epoch + 1), train_losses, label="Traing Loss", marker='o')
    plt.plot(range(1, n_epoch + 1), val_losses, label="Validation Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Graph")
    plt.legend()
    plt.grid()
    plt.show()

    plt.subplot(1, 2, 1)
    plt.plot(range(1, n_epoch + 1), train_accuracies, label="Traning Accu", marker='o')
    plt.plot(range(1, n_epoch + 1), val_accuracies, label="Validation Accu", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Graph")
    plt.legend()
    plt.grid()
    plt.show()
