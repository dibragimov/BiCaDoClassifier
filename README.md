# BiCaDoClassifier
Simple Birds/Cats/Dogs classifier using FastAPI, Docker, and Tensorflow.
The model is trained using images from Kaggle's Cats and Dogs competition and CalTech's birds dataset. 
Inception_V3 model is used for transfer learning.

## Run the website
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

