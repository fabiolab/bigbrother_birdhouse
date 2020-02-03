import os

from loguru import logger

from wtb.dataprocess.picture_generator import PictureGenerator


class PictureBalancer:

    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        self.picture_generator = PictureGenerator()
        self._init_targets()

    def _init_targets(self) -> None:
        # The first iteration lists root subdirs
        _, self.subdirs, _ = next(os.walk(self.base_dir))

        self.target_number_of_pictures = max([len(files) for _, _, files in os.walk(self.base_dir)])

    def _balance_subdir(self, source_dir: str) -> None:
        self.picture_generator.generate_pictures(
            source_dir, source_dir, self.target_number_of_pictures
        )

    def balance(self):
        logger.info(f"Generation for having {self.target_number_of_pictures} files")
        for directory in self.subdirs:
            self._balance_subdir(os.path.join(self.base_dir, directory))
