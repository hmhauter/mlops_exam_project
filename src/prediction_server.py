import io
import os
from fastapi import FastAPI, UploadFile
from PIL import Image
import torch
from google.cloud import storage
from src.models.model import CustomModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Load the model and the dataset when the server starts
BUCKET_NAME = "test-model-server"
MODEL_FILE_NAME = "model.ckpt"

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = storage.Client("mlops-exam-project")
# Download the model from GCS using the bucket and blob
bucket = client.get_bucket(BUCKET_NAME)
blob = bucket.get_blob(MODEL_FILE_NAME)
model_bytes = blob.download_as_bytes()

# Load the model using torch.load
model = CustomModel.load_from_checkpoint(io.BytesIO(model_bytes))
model.freeze()
model.to('cuda' if torch.cuda.is_available() else 'cpu')  # Move model to GPU if available

# Hard coded classes - what would be the best way to do this?
idx_to_class = {0: 'Badminton', 1: 'Cricket', 2: 'Tennis', 3: 'Swimming', 4: 'Soccer', 5: 'Wrestling', 6: 'Karate'}

@app.post("/predict")
async def predict(photo: UploadFile):
    # Read the image file
    contents = await photo.read()
    image = Image.open(io.BytesIO(contents))
    outputs = model.predict(image)
    _, predicted = torch.max(outputs, 1)

    # Map the prediction to a class label
    predicted_label = idx_to_class[predicted.item()]

    # Return the prediction
    return {"prediction": predicted_label}
