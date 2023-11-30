import io
from dynaconf import Dynaconf
import tensorflow as tf
import base64
from PIL import Image
import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
# from pydantic import BaseModel
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
# allow traffic from everywhere
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True,
                   allow_methods=['*'], allow_headers=['*'])
inception_model = None
class_labels = {0: 'birds', 1: 'cats', 2: 'dogs'}


@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Now head over to http://websiteaddress:80/predict to get prediction for your image"


@app.post("/predict")
async def predict(request: Request):
    # read json
    data = await request.json()
    # get the content from list (for now it contains one item)
    data_image = data['image'][0]
    data_image = str(data_image)
    # if has metadata - remove it
    if 'base64,' in data_image:  # remove this first part containing image info
        metadata, data_image = data_image.split('base64,', 1)
        print('metadata', metadata)
    image_bytes = base64.b64decode(data_image)
    # print(image_bytes)
    # convert to image
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('RGB')
    image = image.resize((150,150), Image.NEAREST)
    # preprocess with Keras
    image = tf.keras.preprocessing.image.img_to_array(image)
    data_point = np.expand_dims(image, axis=0)
    # do prediction
    pred = inception_model.predict(data_point/255.0).tolist()
    pred = pred[0]  # first value as we have only 1 image
    classid = np.argmax(pred, axis=0)
    print('predictions', pred, max(pred), classid, class_labels[classid])
    return JSONResponse(jsonable_encoder({
        "probability": str(max(pred)), 'class': class_labels[classid],
        'class_id': str(classid)})
                        )
