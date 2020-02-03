import shutil
from os import path, walk, makedirs
from random import shuffle
from typing import List, Tuple


class FileSplitter:
    TRAIN_DIR = "train"
    VALID_DIR = "valid"
    TEST_DIR = "test"

    def __init__(self, source_dir_path: str, destination_dir_path: str = None) -> None:
        if not (path.isdir(source_dir_path)):
            raise AttributeError

        if not destination_dir_path:
            destination_dir_path = source_dir_path

        self.destination_dir_path = destination_dir_path

        self.source_dir_path = source_dir_path
        self.source_dir_name = path.basename(source_dir_path)

        self.train_dir_path = path.join(
            destination_dir_path, f"{self.source_dir_name}_{self.TRAIN_DIR}"
        )
        self.valid_dir_path = path.join(
            destination_dir_path, f"{self.source_dir_name}_{self.VALID_DIR}"
        )
        self.test_dir_path = path.join(
            destination_dir_path, f"{self.source_dir_name}_{self.TEST_DIR}"
        )

        self.valid_percent = 0.2
        self.test_percent = 0.1

    def split(self, valid_percent: float = 0.2, test_percent: float = 0.1) -> None:
        self.valid_percent = valid_percent
        self.test_percent = test_percent

        # Get list of subdirectories from source directory
        subdir_list = next(walk(self.source_dir_path))[1]
        for subdir in subdir_list:
            self._split_dir(subdir, path.join(self.source_dir_path, subdir))

    def _split_dir(self, dirname: str, dirpath: str) -> None:
        target_directories = [
            path.join(self.train_dir_path, dirname),
            path.join(self.valid_dir_path, dirname),
            path.join(self.test_dir_path, dirname),
        ]

        # Create the subdir in each 'split' directory
        for directory in target_directories:
            makedirs(directory, exist_ok=True)

        file_list = next(walk(dirpath))[2]
        shuffle(file_list)
        files = self._split_list(file_list)

        # zip associates each set of file with a target directory
        for file_zip in zip(files, target_directories):
            FileSplitter._copy_files(dirpath, file_zip[0], file_zip[1])

    def _split_list(self, file_list: List[str]) -> Tuple[List[str], List[str], List[str]]:
        # [-, -, -, -, -, -, -, -, -, -]
        #         |     |
        #         v     v
        # [ valid ][test][    train    ]

        validation_index = int(len(file_list) * self.valid_percent)
        test_index = validation_index + int(len(file_list) * self.test_percent)

        validation_files = file_list[:validation_index]
        test_files = file_list[validation_index:test_index]
        train_files = file_list[test_index:]

        return train_files, validation_files, test_files

    @staticmethod
    def _copy_files(source_dir: str, file_names: List[str], target_dir: str) -> None:
        for file in file_names:
            shutil.copy2(path.join(source_dir, file), target_dir)
