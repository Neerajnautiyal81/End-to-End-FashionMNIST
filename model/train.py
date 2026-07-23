import torch
import torch.nn as nn
import torch.optim as optim

from model import CNN
from dataset import load_data

from utils import plot_loss, plot_accuracy, plot_confusion_matrix, save_classification_report

import os

os.makedirs("saved_models", exist_ok=True)
os.makedirs("images", exist_ok=True)

# Load Dataset
trainloader, valloader, testloader = load_data(batch_size=64)


# Build CNN
model = CNN()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training Variables
epochs = 10

train_losses = []
val_losses = []

train_accuracy = []
val_accuracy = []

best_val_loss = float("inf")


# Training Loop
for epoch in range(epochs):

    # Training
    model.train()

    epoch_training_loss = 0.0

    correct = 0
    total = 0

    for images, labels in trainloader:

        optimizer.zero_grad()

        output = model(images)

        loss = criterion(output, labels)

        loss.backward()

        optimizer.step()

        epoch_training_loss += loss.item()

        _, predicted = torch.max(output, 1)

        correct += (predicted == labels).sum().item()

        total += labels.size(0)

    train_loss = epoch_training_loss / len(trainloader)

    train_losses.append(train_loss)

    train_acc = 100 * correct / total

    train_accuracy.append(train_acc)


    # Validation
    model.eval()

    epoch_validation_loss = 0.0

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in valloader:

            output = model(images)

            loss = criterion(output, labels)

            epoch_validation_loss += loss.item()

            _, predicted = torch.max(output, 1)

            correct += (predicted == labels).sum().item()

            total += labels.size(0)

    val_loss = epoch_validation_loss / len(valloader)

    val_losses.append(val_loss)

    val_acc = 100 * correct / total

    val_accuracy.append(val_acc)


    # Save Best Model
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(model.state_dict(), "saved_models/fashion_model.pth")


    print(f"Epoch {epoch+1}/{epochs}")

    print(f"Train Loss      : {train_loss:.4f}")

    print(f"Validation Loss : {val_loss:.4f}")

    print(f"Train Accuracy  : {train_acc:.2f}%")

    print(f"Validation Accuracy : {val_acc:.2f}%")

    print("-" * 50)
    
    # Test the Model

true_labels = []
predicted_labels = []

correct = 0
total = 0

model.load_state_dict(
    torch.load("saved_models/fashion_model.pth", weights_only=True)
)

model.eval()

with torch.no_grad():

    for images, labels in testloader:

        output = model(images)

        _, predicted = torch.max(output, 1)

        correct += (predicted == labels).sum().item()

        total += labels.size(0)

        true_labels.extend(labels.numpy())

        predicted_labels.extend(predicted.numpy())


test_accuracy = 100 * correct / total

print("=" * 50)
print(f"Test Accuracy : {test_accuracy:.2f}%")
print("=" * 50)

# Generate Graphs

plot_loss(train_losses, val_losses)

plot_accuracy(train_accuracy, val_accuracy)

plot_confusion_matrix(true_labels, predicted_labels)

save_classification_report(true_labels, predicted_labels)

print("Graphs and Reports Saved Successfully.")