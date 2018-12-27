import math
import sys

import os
import os.path
import shutil
from PIL import Image
from PIL.ExifTags import TAGS
from keras import applications
from keras.layers import np
from keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array,
    array_to_img,
)
from keras.utils import to_categorical
from os.path import expanduser
from sklearn.model_selection import StratifiedShuffleSplit

from scripts.commons import configuration
from scripts.commons.logger import logger, configure

conf = configuration.load()


class DataProcessor:
    """
    First and foremost we need convert the data into a format that is needed by the Keras function.
    """

    @staticmethod
    def verify_picture_size(path_picture: str):
        try:
            image = Image.open(path_picture)
            w, h = image.size
            if w <= 1 and h <= 1:
                print(path_picture, " => bad format : 1x1")
                return False
            else:
                return True
        except:
            e = sys.exc_info()[0]
            print("Bad picture : ", path_picture)
            print("Bad picture : %s" % e)
            return False

    @staticmethod
    def init_classes(root_dir, list_classes):
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        for a_class in list_classes:
            class_dir = os.path.join(root_dir, a_class)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

    def gererate_augmented_dataset(
        self,
        pictures_source_dir: str,
        pictures_destination_dir: str,
        nb_images_whished: int,
    ):

        image_generator = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
        )

        for root, _, filenames in os.walk(pictures_source_dir):
            if ".DS_Store" in filenames:
                filenames.remove(".DS_Store")
            nb_images = len(filenames)
            nb_images_to_generate = nb_images_whished - len(filenames)
            nb_iterations_per_image = math.ceil(nb_images_to_generate / nb_images)

            logger.info("")
            logger.info(
                "nb pictures wished : {}, nb pictures found : {}, nb pictures to generate : {}".format(
                    nb_images_whished, nb_images, nb_images_to_generate
                )
            )

            for file in filenames:
                if nb_images_to_generate <= 0:
                    break
                file_path = os.path.join(root, file)
                img = load_img(file_path)  # this is a PIL image

                orientation = None
                if hasattr(img, "_getexif") and callable(getattr(img, "_getexif")):
                    exif = img._getexif()
                    if exif is not None:
                        for tag, value in exif.items():
                            decoded = TAGS.get(tag, tag)
                            if decoded == "Orientation":
                                orientation = value
                                break

                    # We rotate regarding to the EXIF orientation information
                    if orientation is 6:
                        img = img.rotate(-90, expand=True)
                    elif orientation is 8:
                        img = img.rotate(90, expand=True)
                    elif orientation is 3:
                        img = img.rotate(180, expand=True)
                    elif orientation is 2:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    elif orientation is 5:
                        img = img.rotate(-90, expand=True).transpose(
                            Image.FLIP_LEFT_RIGHT
                        )
                    elif orientation is 7:
                        img = img.rotate(90, expand=True).transpose(
                            Image.FLIP_LEFT_RIGHT
                        )
                    elif orientation is 4:
                        img = img.rotate(180, expand=True).transpose(
                            Image.FLIP_LEFT_RIGHT
                        )

                x = img_to_array(img)  # this is a Numpy array
                x = x.reshape((1,) + x.shape)  # this is a Numpy array

                i = 0
                for batch in image_generator.flow(x, batch_size=1):
                    img = array_to_img(batch[0], "channels_last", scale=True)
                    fname = "generated_{index}_{file}".format(index=i, file=file)
                    img.save(os.path.join(pictures_destination_dir, fname))

                    i += 1
                    nb_images_to_generate -= 1
                    print(
                        "\rnb pitures to generates {}".format(nb_images_to_generate),
                        end="",
                    )
                    if i >= nb_iterations_per_image or nb_images_to_generate <= 0:
                        print("\n")
                        break  # otherwise the generator would loop indefinitely

    def balanced_pictures_set(self):

        list = self.list_classes

        sizes = []
        # train
        for a_class in list:
            path = os.path.join(self.train_dir, a_class)
            sizes.append(len(os.listdir(path)))

        nb_pictures = max(sizes)

        # generate pictures
        for a_class in self.list_classes:
            path = os.path.join(self.train_dir, a_class)
            self.gererate_augmented_dataset(path, path, nb_pictures)

    def build_pictures_set(self):
        logger.info("Building pictures sets ...")

        contents = os.listdir(os.path.join(self.collect_dir))
        classes = [
            each
            for each in contents
            if os.path.isdir(os.path.join(self.collect_dir, each))
        ]

        file_paths = []
        labels = []
        for each in classes:
            class_path = os.path.join(self.collect_dir, each)
            files = os.listdir(class_path)
            if ".DS_Store" in files:
                files.remove(".DS_Store")
            # The minimum number of groups for any class cannot be less than 2.
            if len(files) >= 2:
                for file in files:
                    file_paths.append(os.path.join(class_path, file))
                    labels.append(each)
            else:
                logger.info("not enough samples for class : %s", each)

        logger.info("Nb samples : %s", len(file_paths))
        logger.info("Nb labels : %s", len(labels))

        stratified_shuffle_split = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
        train_idx, val_idx = next(stratified_shuffle_split.split(file_paths, labels))

        if self.with_test_set:
            half_val_len = len(val_idx) // 2
            val_idx, test_idx = val_idx[:half_val_len], val_idx[half_val_len:]

        logger.info("Train : %i", len(train_idx))
        path_train = os.path.join(self.dataset_dir, "train")
        for i in train_idx:
            if DataProcessor.verify_picture_size(file_paths[i]):
                shutil.copy2(file_paths[i], os.path.join(path_train, labels[i]))
            else:
                logger.info("Bad picture : %s", file_paths[i])

        logger.info("Validation : %i", len(val_idx))
        path_validation = os.path.join(self.dataset_dir, "validation")
        for j in val_idx:
            if DataProcessor.verify_picture_size(file_paths[j]):
                shutil.copy2(file_paths[j], os.path.join(path_validation, labels[j]))
            else:
                logger.info("Bad picture : %s", file_paths[j])

        if self.with_test_set:
            logger.info("Test : %i", len(test_idx))
            path_test = os.path.join(self.dataset_dir, "test")
            for k in test_idx:
                if DataProcessor.verify_picture_size(file_paths[k]):
                    shutil.copy2(file_paths[k], os.path.join(path_test, labels[k]))
                else:
                    logger.info("Bad picture : %s", file_paths[k])

        logger.info("Data converted into train, test and validation")

    def save_bottlenecks(self, class_indices_path: str):

        batch_size = 16

        # dimensions of images
        img_width = 224
        img_height = 224

        # build the VGG16 network
        model = applications.VGG16(include_top=False, weights="imagenet")

        traingen = ImageDataGenerator(rescale=1.0 / 255)

        # train
        generator = traingen.flow_from_directory(
            self.train_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False,
        )

        # train_labels = generator.classes
        # num_classes = len(generator.class_indices)
        # train_labels = to_categorical(train_labels, num_classes=num_classes)

        class_indices_path = os.path.join(self.root_dir, "class_indices.npy")

        # save the class indices to use use later
        np.save(class_indices_path, generator.class_indices)

        nb_train_samples = len(generator.filenames)
        logger.info("nb train samples : {}".format(nb_train_samples))

        # num_classes = len(generator.class_indices)
        predict_size_train = int(math.ceil(nb_train_samples / batch_size))

        bottleneck_features_train = model.predict_generator(
            generator, predict_size_train, verbose=1
        )

        logger.info("saving bottleneck_features_train.npy")
        np.save(
            os.path.join(self.root_dir, "bottleneck_features_train.npy"),
            bottleneck_features_train,
        )
        logger.info("done.")

        train_labels = generator.classes
        num_classes = len(generator.class_indices)
        train_labels = to_categorical(train_labels, num_classes=num_classes)

        logger.info("saving bottleneck_labels_train.npy")
        np.save(
            os.path.join(self.root_dir, "bottleneck_labels_train.npy"), train_labels
        )
        logger.info("done.")

        # Train dataset
        add_all = sum(list(train_labels))
        inv_map = {v: k for k, v in generator.class_indices.items()}
        for x in range(0, num_classes):
            logger.info(" {} -> {}".format(inv_map[x], add_all[x]))

        # validation
        datagen = ImageDataGenerator(rescale=1.0 / 255)
        generator = datagen.flow_from_directory(
            self.validation_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False,
        )

        nb_validation_samples = len(generator.filenames)
        logger.info("nb validation samples : {}".format(nb_validation_samples))

        predict_size_validation = int(math.ceil(nb_validation_samples / batch_size))

        bottleneck_features_validation = model.predict_generator(
            generator, predict_size_validation, verbose=1
        )

        logger.info("saving bottleneck_features_validation.npy")
        np.save(
            os.path.join(self.root_dir, "bottleneck_features_validation.npy"),
            bottleneck_features_validation,
        )
        logger.info("done.")

        validation_labels = generator.classes
        num_classes = len(generator.class_indices)
        validation_labels = to_categorical(validation_labels, num_classes=num_classes)

        logger.info("saving bottleneck_labels_validation.npy")
        np.save(
            os.path.join(self.root_dir, "bottleneck_labels_validation.npy"),
            validation_labels,
        )
        logger.info("done.")

        # Validation dataset
        add_all = sum(list(validation_labels))
        inv_map = {v: k for k, v in generator.class_indices.items()}
        for x in range(0, num_classes):
            logger.info(" {} -> {}".format(inv_map[x], add_all[x]))

        # test
        if self.with_test_set:
            datagen = ImageDataGenerator(rescale=1.0 / 255)
            generator = datagen.flow_from_directory(
                self.test_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size,
                class_mode=None,
                shuffle=False,
            )

            nb_test_samples = len(generator.filenames)
            logger.info("nb test samples : {}".format(nb_test_samples))

            predict_size_test = int(math.ceil(nb_test_samples / batch_size))

            bottleneck_features_test = model.predict_generator(
                generator, predict_size_test, verbose=1
            )

            logger.info("saving bottleneck_features_test.npy")
            np.save(
                os.path.join(self.root_dir, "bottleneck_features_test.npy"),
                bottleneck_features_test,
            )
            logger.info("done.")

            test_labels = generator.classes
            num_classes = len(generator.class_indices)
            test_labels = to_categorical(test_labels, num_classes=num_classes)

            logger.info("saving bottleneck_labels_test.npy")
            np.save(
                os.path.join(self.root_dir, "bottleneck_labels_test.npy"), test_labels
            )
            logger.info("done.")

            # Test dataset
            add_all = sum(list(test_labels))
            inv_map = {v: k for k, v in generator.class_indices.items()}
            for x in range(0, num_classes):
                logger.info(" {} -> {}".format(inv_map[x], add_all[x]))

    def __init__(
        self, root: str, with_test_set: bool = False, start_from_scratch: bool = True
    ):
        configure("INFO")
        self.with_test_set = with_test_set

        self.root_dir = expanduser(root)
        self.collect_dir = os.path.join(self.root_dir, "photos")
        if os.path.isdir(self.collect_dir):
            self.list_classes = os.listdir(self.collect_dir)
            self.list_classes.remove(".DS_Store")
            logger.info(
                "Found {} classes in collect directory : {} ".format(
                    len(self.list_classes), self.list_classes
                )
            )

            # Initialise dataset
            logger.info("Initialise dataset ...")
            self.dataset_dir = os.path.join(self.root_dir, "dataset")
            if start_from_scratch:
                if os.path.exists(self.dataset_dir) and os.path.isdir(self.dataset_dir):
                    shutil.rmtree(self.dataset_dir)
                os.makedirs(self.dataset_dir)

            self.train_dir = os.path.join(self.dataset_dir, "train")
            DataProcessor.init_classes(self.train_dir, self.list_classes)
            self.validation_dir = os.path.join(self.dataset_dir, "validation")
            DataProcessor.init_classes(self.validation_dir, self.list_classes)
            if with_test_set:
                self.test_dir = os.path.join(self.dataset_dir, "test")
                DataProcessor.init_classes(self.test_dir, self.list_classes)
        else:
            logger.error("No collect directory")
