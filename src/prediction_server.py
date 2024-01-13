import io
from fastapi import FastAPI, UploadFile
from PIL import Image
import torch
import pickle
from google.cloud import storage
from src.models.model import CustomModel

app = FastAPI()

# Load the model and the dataset when the server starts
BUCKET_NAME = "models_mlops"
MODEL_FILE_NAME = "model.ckpt"

client = storage.Client()

# Hard coded classes - what would be the best way to do this?
idx_to_class = {0: 'Badminton', 1: 'Cricket', 2: 'Tennis', 3: 'Swimming', 4: 'Soccer', 5: 'Wrestling', 6: 'Karate'}

@app.post("/predict")
async def predict(photo: UploadFile):
    # Download the model from GCS using the bucket and blob
    print("DOWNLOAD MODEL")
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.get_blob(MODEL_FILE_NAME)
    model_bytes = blob.download_as_bytes()
    print("GOT MODEL BYTES")
    # Load the model using torch.load
    model = CustomModel.load_from_checkpoint(io.BytesIO(model_bytes))
    model.freeze()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')  # Move model to GPU if available
    print("YES _ GIT MODEL")
    # Read the image file
    contents = await photo.read()
    image = Image.open(io.BytesIO(contents))
    outputs = model.predict(image)
    _, predicted = torch.max(outputs, 1)

    # Map the prediction to a class label
    predicted_label = idx_to_class[predicted.item()]

    # Return the prediction
    return {"prediction": predicted_label}
