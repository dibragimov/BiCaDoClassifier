# from dynaconf import settings
import io
from dynaconf import Dynaconf
import tensorflow as tf
import base64
from PIL import Image
import numpy as np
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager


@asynccontextmanager
async def load_clf(app: FastAPI):
    # Load classifier from filefilepath = settings['MODEL_PATH']
    settings = Dynaconf(settings_files=["config/settings.toml"])
    print(settings)
    model_dir = settings['MODEL_PATH']
    # os.path.join('/home/dilshod/Documents/cnn_data/inception_model', '1')
    global inception_model
    inception_model = tf.keras.models.load_model(model_dir, compile=True)
    yield
    return


app = FastAPI(title="Predicting Image Class", lifespan=load_clf)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True,
                   allow_methods=['*'], allow_headers=['*'])
inception_model = None
class_labels = {0: 'birds', 1: 'cats', 2: 'dogs'}


# Represents a particular image (or datapoint)
class Img(BaseModel):
    content: list = []


@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Now head over to http://websiteaddress:80/predict to get prediction for your image"


@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    data_image = data['image'][0]
    data_image = str(data_image)
    if 'base64,' in data_image:  # remove this first part containing image info
        metadata, data_image = data_image.split('base64,', 1)
        print('metadata', metadata)
    image_bytes = base64.b64decode(data_image)
    # print(image_bytes)
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('RGB')
    image = image.resize((150,150), Image.NEAREST)
    image = tf.keras.preprocessing.image.img_to_array(image)
    data_point = np.expand_dims(image, axis=0)

    pred = inception_model.predict(data_point/255.0).tolist()
    pred = pred[0]
    classid = np.argmax(pred, axis=0)
    print('predictions', pred, max(pred), classid, class_labels[classid])
    return JSONResponse(jsonable_encoder({
        "probability": str(max(pred)), 'class': class_labels[classid],
        'class_id': str(classid)})
                        )
