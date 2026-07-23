from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from backend.predict import predict_image


app = FastAPI(title="FashionMNIST Classifier API")


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():

    return {
        "message": "FashionMNIST Classifier API is Running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image = Image.open(io.BytesIO(image_bytes))

    predicted_class, confidence = predict_image(image)

    return {
        "prediction": predicted_class,
        "confidence": round(confidence, 2)
    }