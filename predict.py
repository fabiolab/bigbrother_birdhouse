import json

import PIL
import click
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from keras.preprocessing.image import img_to_array


@click.command()
@click.option("--image_path", type=click.Path(exists=True))
@click.option("--model_path", type=click.Path(exists=True))
@click.option("--mapper_path", type=click.Path(exists=True))
def predict(image_path: str, model_path: str, mapper_path: str):
    model = load_model(model_path)

    picture = PIL.Image.open(image_path)
    picture = picture.resize(size=(224, 224))
    picture_array = img_to_array(img=picture)
    picture_array = np.expand_dims(picture_array, axis=0)
    prediction = model.predict(preprocess_input(picture_array))

    with open(mapper_path, 'r') as fp:
        mapper = json.load(fp)

    best_proba_index = np.argmax(prediction, axis=1)[0]
    print("{} ({:.1f}%)".format(mapper.get(str(best_proba_index)), prediction[0][best_proba_index] * 100))


if __name__ == "__main__":
    predict()
