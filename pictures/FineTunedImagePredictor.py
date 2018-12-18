import os
import shutil
from keras import applications
from keras.engine import Model
from keras.layers import Dropout, Flatten, Dense, np
from keras.models import Sequential
from keras.preprocessing import image as image_utils

from commons.logger import logger, configure


class FineTunedImagePredictor:
    # dimensions of images
    img_width = 224
    img_height = 224

    def predict_pictures_and_sort(self, path_to_pictures: str, output_dir: str):
        labels = {}

        for root, _, filenames in os.walk(path_to_pictures):
            index = 0
            nb_photos = len(filenames)

            # remove '.DS_Store' files
            if '.DS_Store' in filenames:
                filenames.remove('.DS_Store')
            for file in filenames:
                index += 1
                if index % 1000 == 0:
                    logger.info("Checking image %s/%s", index, nb_photos)

                file_path = os.path.join(root, file)
                try:
                    label = self.predict_picture(file_path)
                except OSError:
                    pass
                labels[file_path] = label

                if output_dir:
                    label_dir = os.path.join(output_dir, label)
                    if not os.path.exists(label_dir):
                        os.makedirs(label_dir)

                    dest_file_path = os.path.join(label_dir, file)
                    shutil.copy2(file_path, dest_file_path)

        return labels

    def predict_pictures(self, path_to_pictures: str):
        labels = {}

        for root, _, filenames in os.walk(path_to_pictures):
            index = 0
            nb_photos = len(filenames)

            # remove '.DS_Store' files
            if '.DS_Store' in filenames:
                filenames.remove('.DS_Store')
            for file in filenames:
                index += 1
                if index % 1000 == 0:
                    logger.info("Checking image %s/%s", index, nb_photos)

                file_path = os.path.join(root, file)
                try:
                    label = self.predict_picture(file_path)
                except OSError:
                    pass
                labels[file_path] = label

        return labels

    def predict_picture(self, file_path):
        # load the input image using the Keras helper utility while ensuring
        # that the image is resized to img_widthximg_height pixels, the required input
        # dimensions for the network -- then convert the PIL image to a NumPy array
        image = image_utils.load_img(file_path, target_size=(self.img_width, self.img_height))
        image = image_utils.img_to_array(image)

        # important! otherwise the predictions will be '0'
        image /= 255
        image = np.expand_dims(image, axis=0)

        proba = self.model.predict(image, verbose=0)[0]

        if proba.shape[-1] > 1:
            class_predicted = proba.argmax(axis=-1)
        else:
            class_predicted = (proba > 0.5).astype('int32')

        label = self.inv_map[class_predicted]
        logger.info("for {} prediction is : {}".format(file_path, label))

        return label

    def __init__(self, fine_tuned_model_weights_path: str, class_indices_path: str):
        configure('INFO')

        # load the class_indices saved in the earlier step
        class_dictionary = np.load(class_indices_path).item()
        self.inv_map = {v: k for k, v in class_dictionary.items()}
        num_classes = len(class_dictionary)

        # build the VGG16 network
        base_model = applications.VGG16(weights='imagenet',
                                        include_top=False,
                                        input_shape=(self.img_width, self.img_height, 3))
        logger.info('VGG16 Model loaded.')

        # build a classifier model to put on top of the convolutional model
        top_model = Sequential()
        top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(num_classes, activation='softmax'))

        # add the model on top of the convolutional base
        self.model = Model(input=base_model.input, output=top_model(base_model.output))
        self.model.load_weights(fine_tuned_model_weights_path)
