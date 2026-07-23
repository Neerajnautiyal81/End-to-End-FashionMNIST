import torch
from PIL import Image
import torchvision.transforms as transforms

from model.model import CNN


# FashionMNIST Class Names
class_names = [
    "T-shirt/Top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"
]


# Load Trained Model
model = CNN()

model.load_state_dict(
    torch.load("saved_models/fashion_model.pth", map_location=torch.device("cpu"), weights_only=True)
)

model.eval()


# Image Preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def predict_image(image):

    image = transform(image)

    image = image.unsqueeze(0)

    with torch.no_grad():

        output = model(image)

        probabilities = torch.softmax(output, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = class_names[predicted.item()]

    confidence_score = confidence.item() * 100

    return predicted_class, confidence_score