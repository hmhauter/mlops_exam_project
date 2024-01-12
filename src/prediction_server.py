import io
from fastapi import FastAPI, UploadFile
from PIL import Image
import torch
from torchvision import transforms
from src.data.DataSet import SportDataset
from src.models.model import CustomModel

app = FastAPI()

# Load the model and the dataset when the server starts
model = CustomModel.load_from_checkpoint("lightning_logs/version_2/checkpoints/epoch=0-step=180.ckpt")
model.freeze()
model.to('cuda' if torch.cuda.is_available() else 'cpu')  # Move model to GPU if available
dataset = SportDataset(csv_file="train.csv", data_dir="data/")
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}  # Create a mapping from class indices to class labels

@app.post("/predict")
async def predict(photo: UploadFile):
    # Read the image file
    contents = await photo.read()
    image = Image.open(io.BytesIO(contents))

    # Transform the image to tensor
    transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    image = transform(image).unsqueeze(0)

    device = next(model.parameters()).device
    image = image.to(device)

    # Make a prediction
    with torch.no_grad():
        model.eval()
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    # Map the prediction to a class label
    predicted_label = idx_to_class[predicted.item()]

    # Return the prediction
    return {"prediction": predicted_label}