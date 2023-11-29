import tensorflow as tf
import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager



@asynccontextmanager
async def load_clf(app: FastAPI):
    # Load classifier from file
    model_dir = os.path.join('/home/dilshod/Documents/cnn_data/inception_model', '1')
    inception_model = tf.keras.models.load_model(model_dir, compile=True)
    yield
    return


app = FastAPI(title="Predicting Image Class", lifespan=load_clf)
inception_model = None


# Represents a particular image (or datapoint)
class Img(BaseModel):
    content: list = []


@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Now head over to http://websiteaddress:80/predict to get prediction for your image"


@app.post("/predict")
def predict(wine: Img):
    data_point = np.array(wine.list)

    pred = inception_model.predict(data_point).tolist()
    pred = pred[0]
    print(pred)
    return {"Prediction": pred}
