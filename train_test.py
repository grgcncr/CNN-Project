import torch
from functions import confusion_matrix

def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch in dataloader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)  

    avg_loss = total_loss / total_samples
    avg_accuracy = correct_predictions / total_samples

    return avg_loss, avg_accuracy

def test(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    cm = torch.zeros((4, 4), dtype=torch.int64)

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            cm += confusion_matrix(labels, predicted, 4)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total_samples

    return avg_loss, accuracy, cm

