import os

import pytest

from wtb.dataprocess.file_splitter import FileSplitter

BASE_DIR_PATH = '/tmp'
DATA_DIR_NAME = 'test_fs'
DATA_DIR_PATH = os.path.join(BASE_DIR_PATH, DATA_DIR_NAME)
DEST_DIR_NAME = 'test_fs_data'
DEST_DIR_PATH = os.path.join(BASE_DIR_PATH, DEST_DIR_NAME)

FAKE_FILES = [f"{idx}_file.txt" for idx in range(10)]
FAKE_DIRS = [f"{idx}_dir" for idx in range(3)]

TRAIN_DIR = os.path.join(DEST_DIR_PATH, f"{DATA_DIR_NAME}_{FileSplitter.TRAIN_DIR}")
TEST_DIR = os.path.join(DEST_DIR_PATH, f"{DATA_DIR_NAME}_{FileSplitter.TEST_DIR}")
VALID_DIR = os.path.join(DEST_DIR_PATH, f"{DATA_DIR_NAME}_{FileSplitter.VALID_DIR}")


def _create_dir_tree(root_dir):
    print(f"Create {root_dir}")
    os.makedirs(root_dir)
    for directory in FAKE_DIRS:
        path_dir = os.path.join(root_dir, directory)
        print(f"Create {path_dir}")
        os.makedirs(path_dir)
        for file in FAKE_FILES:
            open(os.path.join(path_dir, file), 'a').close()


def _clean_dir_tree(root_dir):
    for directory in FAKE_DIRS:
        path_dir = os.path.join(root_dir, directory)
        for file in FAKE_FILES:
            try:
                os.remove(os.path.join(path_dir, file))
            except FileNotFoundError:
                pass
        print(f"Remove {path_dir}")
        os.removedirs(path_dir)


@pytest.fixture(scope="module")
def init_tmp_dir():
    _create_dir_tree(DATA_DIR_PATH)

    yield

    _clean_dir_tree(DATA_DIR_PATH)
    _clean_dir_tree(TRAIN_DIR)
    _clean_dir_tree(TEST_DIR)
    _clean_dir_tree(VALID_DIR)


def test_init_not_dir():
    with pytest.raises(AttributeError):
        _ = FileSplitter('/gfddfgdf')


def test_init(init_tmp_dir):
    fs = FileSplitter(DATA_DIR_PATH)

    assert fs.source_dir_path == DATA_DIR_PATH
    assert fs.source_dir_name == DATA_DIR_NAME
    assert fs.destination_dir_path == DATA_DIR_PATH
    assert fs.valid_percent == 0.2
    assert fs.test_percent == 0.1


def test_split(init_tmp_dir):
    fs = FileSplitter(DATA_DIR_PATH, DEST_DIR_PATH)
    fs.split(valid_percent=0.2, test_percent=0.1)

    assert os.path.isdir(TRAIN_DIR)
    assert os.path.isdir(TEST_DIR)
    assert os.path.isdir(VALID_DIR)

    for path, dirs, files in os.walk(DEST_DIR_PATH):
        if os.path.basename(path) in FAKE_DIRS and TRAIN_DIR in path:
            assert len(files) == 7
        if os.path.basename(path) in FAKE_DIRS and TEST_DIR in path:
            assert len(files) == 1
        if os.path.basename(path) in FAKE_DIRS and VALID_DIR in path:
            assert len(files) == 2
