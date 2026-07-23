import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report


# Plot Training and Validation Loss
def plot_loss(train_losses, val_losses):

    plt.figure(figsize=(8,5))

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title("Training and Validation Loss")

    plt.legend()

    plt.grid(True)

    plt.savefig("images/loss_graph.png")

    plt.close()


# Plot Training and Validation Accuracy
def plot_accuracy(train_accuracy, val_accuracy):

    plt.figure(figsize=(8,5))

    plt.plot(train_accuracy, label="Train Accuracy")
    plt.plot(val_accuracy, label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")

    plt.title("Training and Validation Accuracy")

    plt.legend()

    plt.grid(True)

    plt.savefig("images/accuracy_graph.png")

    plt.close()


# Plot Confusion Matrix
def plot_confusion_matrix(true_labels, predicted_labels):

    cm = confusion_matrix(true_labels, predicted_labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot(cmap="Blues")

    plt.savefig("images/confusion_matrix.png")

    plt.close()


# Save Classification Report
def save_classification_report(true_labels, predicted_labels):

    report = classification_report(true_labels, predicted_labels)

    with open("images/classification_report.txt", "w") as file:

        file.write(report)