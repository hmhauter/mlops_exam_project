import io
import os
from fastapi import FastAPI, UploadFile, BackgroundTasks
from PIL import Image
import torch
import csv
from google.cloud import storage
from src.models.model import CustomModel
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from transformers import CLIPProcessor, CLIPModel
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfColumns,
    TestNumberOfRows,
)
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
import pandas as pd
from fastapi.responses import HTMLResponse
import logging

app = FastAPI()
logging.Logger("app", level=logging.INFO)

# Load the model and the dataset when the server starts
BUCKET_MODEL = "test-model-server"
BUCKET_DATA_DRIFT = "mlops-data-drift"
MODEL_FILE_NAME = "model.ckpt"
CSV_FILE_NAME = "data_drift.csv"
CSV_FILE_NAME_NEW = "new_data.csv"

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
bucket = client.get_bucket(BUCKET_MODEL)
blob = bucket.get_blob(MODEL_FILE_NAME)
model_bytes = blob.download_as_bytes()

# Download the data drift csv from GCS using the bucket and blob
bucket_csv = client.get_bucket(BUCKET_DATA_DRIFT)
blob_csv = bucket_csv.get_blob(CSV_FILE_NAME)
csv_bytes = blob_csv.download_as_bytes()
df_data_drift = pd.read_csv(io.BytesIO(csv_bytes), encoding="utf8")


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
    image = image.convert("RGB")

    outputs = model.predict(image)
    _, predicted = torch.max(outputs, 1)

    # Map the prediction to a class label
    predicted_label = idx_to_class[predicted.item()]

    # spawn a background task to write to the bucket (for now our wip database)
    background_tasks.add_task(extract_image_data, image=image, predictionLabel=predicted_label)

    # Return the prediction
    return {"prediction": predicted_label}


# log data for detecting image drift
def extract_image_data(image: Image, predictionLabel: str):
    # extract image data with help of awesome OpenAI Model CLIP
    inputs = processor_clip(text=None, images=image, return_tensors="pt", padding=True)
    img_features = model_clip.get_image_features(inputs["pixel_values"])

    bucket_csv = client.get_bucket(BUCKET_DATA_DRIFT)
    blob_csv_new = bucket_csv.get_blob(CSV_FILE_NAME_NEW)
    csv_bytes_new = blob_csv_new.download_as_bytes()
    df_data_drift_new = pd.read_csv(io.BytesIO(csv_bytes_new), encoding="utf8")

    column_names = ["label"] + [str(i) for i in range(1, 513)]
    np_array = img_features.detach().numpy()
    df = pd.DataFrame(np_array)
    df.insert(0, "label", predictionLabel)
    df.columns = column_names
    _df_data_drift_new = pd.concat([df_data_drift_new, df], ignore_index=True)

    _csv = _df_data_drift_new.to_csv(header=True, index=False, mode="a")
    _blob = bucket_csv.blob(CSV_FILE_NAME_NEW)
    _blob.upload_from_string(_csv)


# this is a helper function to generate the base csv file for the data drift test
def __generate_csv():
    print("GENERATE CSV")
    folder_path = "./data/train"
    csv_file_path = "./data_drift.csv"
    label_file_path = "./data/train.csv"
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    # read labels in to pd
    df = pd.read_csv(label_file_path)

    # Open CSV file for writing
    with open(csv_file_path, "w", newline="") as csvfile:
        # Create a CSV writer object
        csv_writer = csv.writer(csvfile)
        image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        # Iterate through each image file
        for image_file in image_files:
            # Construct the full path to the image file
            image_path = os.path.join(folder_path, image_file)

            # Open the image using PIL
            image = Image.open(image_path)

            inputs = processor_clip(text=None, images=image, return_tensors="pt", padding=True)
            img_features = model_clip.get_image_features(inputs["pixel_values"])
            img_features_np = img_features.detach().numpy()
            label_for_image = df.loc[df["image_ID"].str.contains(image_file), "label"].values
            label = label_for_image[0]

            # Write image details to CSV file
            flattened_data = [item for sublist in img_features_np for item in sublist]
            row_data = [f"{label}"] + flattened_data

            # Write the concatenated data to CSV file
            csv_writer.writerow(row_data)
            # Close the image
            image.close()
    return {"message": "CSV generated"}


# DOES NOT MAKE SENSE FOR IMAGES AND UNSTRCTURED DATA
@app.get("/runDataDriftTests", response_class=HTMLResponse)
async def run_data_test():
    csv_file_path = "data_drift.csv"
    # Write the NumPy array to a CSV file
    reference = pd.read_csv(csv_file_path)

    csv_new_data = "new_data.csv"
    new_data = pd.read_csv(csv_new_data)

    data_integrity_dataset_tests = TestSuite(
        tests=[
            TestNumberOfColumns(),
            TestNumberOfRows(),
        ]
    )
    data_integrity_dataset_tests.run(reference_data=reference, current_data=new_data)
    data_integrity_dataset_tests.save_html("data_integrity_dataset_tests.html")
    with open("data_integrity_dataset_tests.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/runDataDriftReport", response_class=HTMLResponse)
async def get_report():
    bucket_csv = client.get_bucket(BUCKET_DATA_DRIFT)
    blob_csv_new = bucket_csv.get_blob(CSV_FILE_NAME_NEW)
    csv_bytes_new = blob_csv_new.download_as_bytes()
    df_data_drift_new = pd.read_csv(io.BytesIO(csv_bytes_new), encoding="utf8")

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset(), TargetDriftPreset()])

    report.run(reference_data=df_data_drift, current_data=df_data_drift_new)
    report.save_html("report.html")

    with open("report.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)


Instrumentator().instrument(app).expose(app)
