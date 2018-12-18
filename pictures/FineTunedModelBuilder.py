import time

import keras
import matplotlib.pyplot as plt
import os
from keras import applications
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine import Model
from keras.layers import Dropout, Flatten, Dense, np
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from numpy import argmax
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from commons.logger import logger, configure
from pictures import Util


class FineTunedModelBuilder:
    # dimensions of images
    img_width = 224
    img_height = 224

    batch_size = 32

    epochs = 200

    @staticmethod
    def plot_history(history):
        plt.figure(1)

        # summarize history for accuracy
        plt.subplot(211)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        img_path = 'accuracy.jpg'
        plt.savefig(img_path % 'accuracy')

        # summarize history for loss
        plt.subplot(212)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        plt.show()

    def train_fine_tuning_top_model(self):
        # build the VGG16 network
        base_model = applications.VGG16(weights='imagenet',
                                        include_top=False,
                                        input_shape=(self.img_width, self.img_height, 3))
        logger.info('VGG16 Model loaded.')

        # build a classifier model to put on top of the convolutional model
        top_model = self.build_top_model(base_model.output_shape[1:])

        # note that it is necessary to start with a fully-trained
        # classifier, including the top classifier,
        # in order to successfully do fine-tuning
        top_model.load_weights(self.top_model_weights_path)

        # add the model on top of the convolutional base
        model = Model(input=base_model.input, output=top_model(base_model.output))

        # set the first 15 layers (up to the last conv block)
        # to non-trainable (weights will not be updated)
        for layer in model.layers[:15]:
            layer.trainable = False

        # compile the model with a SGD/momentum optimizer
        # and a very slow learning rate.
        model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(lr=1e-4, momentum=0.9),
                      metrics=['accuracy'])

        model_checkpoint = ModelCheckpoint(filepath=self.fine_tuned_model_weights_path,
                                           save_best_only=True,
                                           mode='max',
                                           monitor='acc',
                                           verbose=0)

        logger.info("Start training model ...")
        start_time = time.time()

        # prepare data augmentation configuration
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=False)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        dataset_dir = os.path.join(self.root_dir, 'dataset')

        train_data_dir = os.path.join(dataset_dir, 'train')
        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

        validation_data_dir = os.path.join(dataset_dir, 'validation')
        validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

        # fine-tune the model
        onlyfiles = []
        for dirpath, dirnames, filenames in os.walk(train_data_dir):
            for filename in [f for f in filenames]:
                onlyfiles.append(os.path.join(dirpath, filename))

        nb_train_samples = len(onlyfiles)

        # Total number of steps (batches of samples) to yield from `generator` before declaring one
        # epoch finished and starting the next epoch.
        steps_per_epoch = nb_train_samples // self.batch_size

        # Total number of steps (batches of samples) to yield from `generator` before stopping
        validation_steps = 482

        history = model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=self.epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=[model_checkpoint])

        logger.info("Done in : %f seconds", (time.time() - start_time))

        logger.info("Evaluate fine tuned model")
        self.evaluate_model(model)

        FineTunedModelBuilder.plot_history(history)

    def train_top_model(self):
        # load the bottleneck features saved earlier
        train_data = np.load(os.path.join(self.root_dir, 'bottleneck_features_train.npy'))
        train_labels = np.load(os.path.join(self.root_dir, 'bottleneck_labels_train.npy'))
        validation_data = np.load(os.path.join(self.root_dir, 'bottleneck_features_validation.npy'))
        validation_labels = np.load(os.path.join(self.root_dir, 'bottleneck_labels_validation.npy'))

        top_model = self.build_top_model(train_data.shape[1:])

        early_stopping = EarlyStopping(verbose=1, patience=40, monitor='acc')
        model_checkpoint = ModelCheckpoint(filepath=self.top_model_weights_path,
                                           save_best_only=True,
                                           mode='max',
                                           monitor='acc',
                                           verbose=0)

        logger.info("Start training model ...")
        start_time = time.time()

        history = top_model.fit(train_data, train_labels,
                                epochs=self.epochs,
                                batch_size=self.batch_size,
                                validation_data=(validation_data, validation_labels),
                                callbacks=[model_checkpoint, early_stopping])

        logger.info("Done in : %f seconds", (time.time() - start_time))

        logger.info("Evaluate top model")
        self.evaluate_model(top_model)

        FineTunedModelBuilder.plot_history(history)

    def display_model_evaluation(self, model, data, labels):
        """
        Quantitative evaluation of the model quality on the test set
        """
        y_test = argmax(labels, axis=1)
        y_pred = model.predict_classes(data)

        target_names = [key for (key, value) in sorted(self.class_dictionary.items())]

        print()
        print(classification_report(y_test, y_pred, target_names=target_names))

        cnf_matrix = confusion_matrix(y_test, y_pred)
        Util.plot_confusion_matrix(cnf_matrix, target_names, normalize=False)

        scores = model.evaluate(data, labels,
                                batch_size=self.batch_size,
                                verbose=0)
 #       for i in range(1, len(model.metrics_names)):
 #           logger.info("%s: %.2f%%" % (model.metrics_names[i], scores[i] * 100))

        logger.info(model.metrics_names)
        logger.info(scores)

    def evaluate_model(self, model):
        bottleneck_features_test = os.path.join(self.root_dir, 'bottleneck_features_test.npy')
        bottleneck_labels_test = os.path.join(self.root_dir, 'bottleneck_labels_test.npy')

        if os.path.isfile(bottleneck_features_test) and os.path.isfile(bottleneck_labels_test):
            # load the bottleneck features saved earlier
            test_data = np.load(bottleneck_features_test)
            test_labels = np.load(bottleneck_labels_test)

            self.display_model_evaluation(model, test_data, test_labels)
        else:
            logger.warn("Unable to find test dataset")

    def build_top_model(self, input_shape=(7, 7, 512)):
        top_model = Sequential()
        top_model.add(Flatten(input_shape=input_shape))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.4))
        top_model.add(Dense(self.num_classes, activation='softmax'))
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        # adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        top_model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

        return top_model

    def __init__(self, root: str, top_model_weights_path: str, fine_tuned_model_weights_path:str, class_indices_path: str):
        configure('INFO')

        logger.info('Keras version : {}'.format(keras.__version__))

        self.root_dir = os.path.expanduser(root)

        # load the class_indices saved in the earlier step
        self.class_dictionary = np.load(class_indices_path).item()

        self.num_classes = len(self.class_dictionary)

        self.top_model_weights_path = top_model_weights_path
        self.fine_tuned_model_weights_path = fine_tuned_model_weights_path

