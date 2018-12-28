import logging
import math
from os import walk, path
from typing import Iterator

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

LOGGER = logging.getLogger(__name__)


class PictureGenerator:
    def __init__(self):
        self.image_generator = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
        )

    def generate_pictures(
            self, source_dir: str, destination_dir: str, target_number_of_pictures: int
    ) -> None:
        for base_path, _, files in walk(source_dir):
            nb_images = len(files)
            nb_images_to_generate = target_number_of_pictures - nb_images
            nb_iterations_per_image = math.ceil(nb_images_to_generate / nb_images)

            LOGGER.info(
                f"{base_path} contains {nb_images} : {nb_images_to_generate} will be generated to get {target_number_of_pictures}")
            nb_images_generated = 0
            for filename in files:
                if not nb_iterations_per_image or nb_images_generated >= nb_images_to_generate:
                    break

                nb_images_generated += self._generate_pictures_from_file(
                    filename, base_path, nb_iterations_per_image, destination_dir
                )

            LOGGER.info(f"{nb_images_generated} for {base_path} ({nb_images_to_generate} planned)")

    def _get_pictures(self, filename: str, base_path: str, nb_pictures_out: int) -> Iterator:
        try:
            current_image = load_img(path.join(base_path, filename))
        except OSError as e:
            LOGGER.error("This is not an image ? ")
            LOGGER.error(e)
            return
        current_image_array = img_to_array(current_image)
        current_image_array = current_image_array.reshape((1,) + current_image_array.shape)

        for _ in range(nb_pictures_out):
            yield next(self.image_generator.flow(current_image_array, batch_size=1))

    def _generate_pictures_from_file(
            self, filename: str, base_path: str, nb_pictures_out: int, destination_dir: str
    ) -> int:
        index = 0
        for batch in self._get_pictures(filename, base_path, nb_pictures_out):
            current_image = array_to_img(batch[0], "channels_last", scale=True)
            current_image.save(path.join(destination_dir, f"generated_{index:02d}_{filename}"))
            LOGGER.debug(path.join(destination_dir, f"generated_{index:02d}_{filename}"))
            index += 1

        return index
