import io
import os
from fastapi import FastAPI, UploadFile, BackgroundTasks
from PIL import Image
import torch
import csv
from google.cloud import storage
from src.models.model import CustomModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import CLIPProcessor, CLIPModel
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestValueRange,
    TestMeanInNSigmas,
    TestNumberOfColumns,
    TestNumberOfRows,
    TestColumnValueMin,
    TestColumnValueMax,
    TestColumnValueMean,
    TestColumnValueMedian,
    TestColumnValueStd,
)
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
import pandas as pd
from fastapi.responses import HTMLResponse


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
model.to("cuda" if torch.cuda.is_available() else "cpu")  # Move model to GPU if available

# Hard coded classes - what would be the best way to do this?
idx_to_class = {0: "Badminton", 1: "Cricket", 2: "Tennis", 3: "Swimming", 4: "Soccer", 5: "Wrestling", 6: "Karate"}

# To be discussed: Do this here or in function?
# PROBLEM: This will slow down the start up process about a minute what is really bad if the user wants to use the app
model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor_clip = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


@app.post("/predict")
async def predict(photo: UploadFile, background_tasks: BackgroundTasks):
    # Read the image file
    contents = await photo.read()
    image = Image.open(io.BytesIO(contents))

    # spawn a background task to write to the bucket (for now our wip database)
    background_tasks.add_task(extract_image_data, image=image)

    outputs = model.predict(image)
    _, predicted = torch.max(outputs, 1)

    # Map the prediction to a class label
    predicted_label = idx_to_class[predicted.item()]

    # Return the prediction
    return {"prediction": predicted_label}


# log data for detecting image drift
def extract_image_data(image: Image):
    print("CALL BACKGROUND TASK")
    # extract image data with help of awesome OpenAI Model CLIP
    inputs = processor_clip(text=None, images=image, return_tensors="pt", padding=True)
    img_features = model_clip.get_image_features(inputs["pixel_values"])
    print(img_features)
    csv_file_path = "new_data.csv"
    # Write the NumPy array to a CSV file
    df = pd.DataFrame(img_features.detach().numpy())
    df.to_csv(csv_file_path, header=False, index=False, mode="a")


@app.get("/doDataset")
async def generate_csv():
    print("GENERATE CSV")
    folder_path = "/home/hhauter/Documents/W23/MLOps/mlops_exam_project/data/test"
    csv_file_path = "/home/hhauter/Documents/W23/MLOps/mlops_exam_project/src/data_drift.csv"
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Open CSV file for writing
    with open(csv_file_path, "w", newline="") as csvfile:
        # Create a CSV writer object
        csv_writer = csv.writer(csvfile)
        image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        # Iterate through each image file
        for image_file in image_files:
            print(image_file)
            # Construct the full path to the image file
            image_path = os.path.join(folder_path, image_file)

            # Open the image using PIL
            image = Image.open(image_path)

            inputs = processor_clip(text=None, images=image, return_tensors="pt", padding=True)
            img_features = model_clip.get_image_features(inputs["pixel_values"])
            img_features_np = img_features.detach().numpy()

            # Write image details to CSV file
            flattened_data = [item for sublist in img_features_np for item in sublist]
            csv_writer.writerow(flattened_data)

            # Close the image
            image.close()
    return {"message": "CSV generated"}


@app.get("/runDataDriftTests", response_class=HTMLResponse)
async def run_data_test():
    csv_file_path = "data_drift.csv"
    # Write the NumPy array to a CSV file
    reference = pd.read_csv(csv_file_path)

    csv_new_data = "new_data.csv"
    new_data = pd.read_csv(csv_new_data)

    data_integrity_dataset_tests = TestSuite(
        tests=[
            TestValueRange(),
            TestMeanInNSigmas(),
            TestNumberOfColumns(),
            TestNumberOfRows(),
            TestColumnValueMin(),
            TestColumnValueMax(),
            TestColumnValueMean(),
            TestColumnValueMedian(),
            TestColumnValueStd(),
        ]
    )
    data_integrity_dataset_tests.run(reference_data=reference, current_data=new_data)
    data_integrity_dataset_tests.save_html("data_integrity_dataset_tests.html")
    with open("data_integrity_dataset_tests.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/runDataDriftReport", response_class=HTMLResponse)
async def get_report():
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    csv_file_path = "data_drift.csv"
    # Write the NumPy array to a CSV file
    reference = pd.read_csv(csv_file_path)

    csv_new_data = "new_data.csv"
    new_data = pd.read_csv(csv_new_data)

    report.run(reference_data=reference, current_data=new_data)
    report.save_html("report.html")

    with open("report.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)
