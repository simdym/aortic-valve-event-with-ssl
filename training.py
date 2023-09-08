import torch

def train_epoch(model, dataloader, criterion, optimizer, device, zero_grad=True):
    model.train()
    model.to(device)
    
    loss = 0
    batches = 0

    for data, labels in dataloader:
        data = data.to(device)
        labels = labels.to(device)

        if zero_grad:
            optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, labels)
        print(loss.item())

        loss.backward()
        optimizer.step()

        loss += loss.item()
        batches += 1
    return loss / batches

def validate(model, dataloader, criterion, device):
    model.eval()
    model.to(device)

    loss = 0
    batches = 0

    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            labels = labels.to(device)

            output = model(data)

            loss += criterion(output, labels).item()
            batches += 1
    return loss / len(dataloader)   