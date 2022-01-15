from fastapi import FastAPI, Request
from tensorflow.keras.models import load_model
import numpy as np


model_path = 'model/generator.h5'
generator = load_model(model_path)
latent_dims = 100


app = FastAPI()


@app.get('/')
def index(request: Request):
    noise = np.random.normal(0, 1, (1, latent_dims))
    image = generator.predict(noise)
    image_list = image.tolist()
    return {'Image': image_list}