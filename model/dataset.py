import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


def load_data(batch_size=64):

    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Download FashionMNIST Dataset
    trainset = torchvision.datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    testset = torchvision.datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    # Train and Validation Split
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size

    trainset, valset = random_split(trainset, [train_size, val_size])

    # DataLoaders
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True
    )

    valloader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False
    )

    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False
    )

    return trainloader, valloader, testloader