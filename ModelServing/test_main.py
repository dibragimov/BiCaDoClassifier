import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
from .main import app
import tensorflow as tf
import numpy as np
import json
import base64
from dynaconf import Dynaconf


client = TestClient(app)
settings = Dynaconf(settings_files=["config/settings.toml"])
print(settings)
image_path = settings['TEST_IMAGE']


def test_main():
    response = client.get('/')
    assert response.status_code == 200
    resp = response.content.decode()
    assert resp == '"Congratulations! Your API is working as expected. Now head over to http://websiteaddress:80/predict to get prediction for your image"'


@pytest.mark.anyio
async def test_predict():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        img = tf.keras.utils.load_img(image_path)
        img = tf.keras.utils.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        with open(image_path, "rb") as image_file:
            encoded_img = base64.b64encode(image_file.read())
        response = await ac.post('/predict', json={"image": [encoded_img.decode("utf-8")]})  # img.tolist()})
        assert response.status_code == 200
