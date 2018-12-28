import os

import numpy as np
from keras import applications
from keras.callbacks import ModelCheckpoint
from keras.layers import Flatten, Dropout, Dense
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
IMG_ROWS = 150
IMG_COLS = 150
EPOCHS = 10  # Number of times each sample is given to the network
BATCH_SIZE = 32  # Numer of samples given to the network before updating the model
NUM_OF_TRAIN_SAMPLES = 3000
NUM_OF_TEST_SAMPLES = 600
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'data')
MODEL_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'models')


def get_human_prediction(model: Model, input_picture_array: np.array, mapper: dict):
    return mapper.get(
        [
            idx[0]
            for idx, item in np.ndenumerate(model.predict(input_picture_array)[0])
            if item == 1.0
        ][0]
    )


def build_network(num_of_classes: int):
    # Load VGG16 model from keras
    base_model = applications.VGG16(
        weights="imagenet", include_top=False, input_shape=(IMG_ROWS, IMG_COLS, 3)
    )

    # Prevent the VGG16 layers to be modified by our custom training
    for layer in base_model.layers:
        layer.trainable = False

    # Build our own top layers for prediction
    model_ft_top = Sequential()
    model_ft_top.add(Flatten())
    model_ft_top.add(Dense(1024, activation="relu"))
    model_ft_top.add(Dropout(0.5))
    model_ft_top.add(Dense(num_of_classes, activation="softmax"))

    # Merge the VGG16 and our custom models
    model_ft = Model(input=base_model.input, output=model_ft_top(base_model.output))

    # model_ft = Model(inputs=Tensor(base_model.input), outputs=Tensor(model_ft_top(base_model.output)))

    model_ft.compile(
        optimizer=SGD(lr=1e-4, momentum=0.9),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model_ft


if __name__ == "__main__":
    # Generator for train
    train_image_generator = ImageDataGenerator()
    train_iterator = train_image_generator.flow_from_directory(
        os.path.join(DATA_DIR, "train"),  # Root directory
        target_size=(IMG_ROWS, IMG_COLS),  # Images will be processed to this size
        batch_size=BATCH_SIZE,  # How many data are processed at the same time ?
        class_mode="categorical",
    )  # Each subdir is a category

    # Generator for validation
    valid_image_generator = ImageDataGenerator()
    valid_iterator = valid_image_generator.flow_from_directory(
        os.path.join(DATA_DIR, "validation"),
        target_size=(IMG_ROWS, IMG_COLS),  # Images will be processed to this size
        batch_size=BATCH_SIZE,  # How many data are processed at the same time ?
        class_mode="categorical",
    )

    NUM_OF_CLASSES = len(train_iterator.class_indices)

    # This Keras Callbak saves the best model according to the accuracy metric
    filepath = os.path.join(MODEL_DIR, "{epoch:02d}-{val_acc:.2f}.hdf5")
    checkpoint = ModelCheckpoint(
        filepath, monitor="val_acc", verbose=1, save_best_only=True, mode="max"
    )

    # Map a prediction indice to a label
    mapper = {v: k for k, v in train_iterator.class_indices.items()}

    model = build_network(NUM_OF_CLASSES)

    model.fit_generator(
        generator=train_iterator,
        steps_per_epoch=NUM_OF_TRAIN_SAMPLES // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=valid_iterator,
        validation_steps=NUM_OF_TEST_SAMPLES // BATCH_SIZE,
        callbacks=[checkpoint],
    )
