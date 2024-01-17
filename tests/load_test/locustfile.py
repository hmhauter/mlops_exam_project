import glob
from locust import HttpUser, task, between
import random


class APIUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def upload_and_predict(self):
        # Replace this with the path to a valid image file for prediction
        image_path = random.choice(glob.glob("data/test/*"))
        with open(image_path, "rb") as image:
            self.client.post("/predict", files={"photo": image})
