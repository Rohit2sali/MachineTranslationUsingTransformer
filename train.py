from Transfomer import Transformer
from get_data import data
import torch
import torch.nn as nn

def generate_text(predictions):
    predictions = predictions.permute(0, 2, 1)
    
    predictions = torch.nn.functional.softmax(predictions, dim=-1)
   
    _, predictions = torch.max(predictions, dim=-1)
    output = model.decode(predictions)
    print(output[0])


def accuracy_fn(target, prediction, eos_token):
    prediction = prediction.permute(0, 2, 1)
    prediction = torch.nn.functional.softmax(prediction, dim=-1)
    prediction = torch.max(prediction, dim=-1)
    prediction = prediction[1]
    m = len(prediction[1])
    for i in range(len(target)):
        for j in range(len(target[0])):
            if(target[i][j] == eos_token):
                if(j>m):
                    m = j
                    break

    target = target[:, :m+1]
    prediction = prediction[:, :m+1]
    correct = (target == prediction)
    accuracy = correct.sum().item() / torch.numel(target)
    return accuracy

def calculate_loss(prediction, target_tokens, pading_token):
    criterion = nn.CrossEntropyLoss(ignore_index=pading_token)
    return criterion(prediction, target_tokens)

def train(input, target, val_input, val_target):
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    model.train()
    prediction, target_tokens, eos_token, padding_token = model(input, target, generate=False)
    loss = calculate_loss(prediction, target_tokens, padding_token)
    accuracy = accuracy_fn(target_tokens, prediction, eos_token)
    train_loss.append(loss)
    train_acc.append(accuracy)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    model.eval()
    with torch.no_grad():
        prediction, target_tokens, eos_token, padding_token = model(val_input, val_target, generate=False)
        valid_loss = calculate_loss(prediction, target_tokens, padding_token)
        val_accuracy = accuracy_fn(target_tokens, prediction, eos_token)
        val_loss.append(valid_loss)
        val_acc.append(val_accuracy)
        
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_acc) / len(train_acc)
    val_loss = sum(val_loss) / len(val_loss)
    val_acc = sum(val_acc) / len(val_acc)
    return train_loss, train_acc, val_loss, val_acc

def test(test_data, target_test_data):
    # test_data = test_data[:batch_size]
    # target_test_data = target_test_data[:batch_size]
    with torch.no_grad():
        prediction, target_tokens, eos_token, padding_token = model(test_data, target_test_data, generate=False)
        test_loss = calculate_loss(prediction, target_tokens, padding_token)
        test_acc = accuracy_fn(target_tokens, prediction, eos_token)
        print("test_loss :",test_loss, "test_acc :", test_acc)
    return prediction, eos_token


if __name__ == "__main__":
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # criterion = nn.CrossEntropyLoss(ignore_index=pad_token)

    best_val_loss = float("inf")
    n_batches = len(train_data) // batch_size
    for epoch in range(n_epoch):
        for i in range(0, len(train_data), batch_size):
            input = train_data[0:batch_size]
            target = val_data[0:batch_size]
            val_input = val_data[0:batch_size]
            val_target = target_val_data[0:batch_size]
            train_loss, train_acc, val_loss, val_acc = train(input, target, val_input, val_target)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'transformer_epoch{epoch + 1}.pth')
        print(f"at epoch :{epoch}, loss :{train_loss}, acc :{train_acc}, val_loss{val_loss}, val_acc :{val_acc}")
           
        predictions, eos_token = test(test_data, target_test_data)
        generate_text(predictions)
    torch.save(model.state_dict(), 'model.pth')


