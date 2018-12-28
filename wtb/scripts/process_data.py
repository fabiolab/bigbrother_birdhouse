import logging
from os import path

from wtb.dataprocess.file_splitter import FileSplitter
from wtb.dataprocess.picture_balancer import PictureBalancer

SCRIPT_DIR = path.dirname(path.realpath(__file__))
if __name__ == "__main__":

    logger = logging.getLogger('wtb')

    image_dir = path.join(SCRIPT_DIR, "..", "..", "data")

    logger.info(f"Image preprocessing from {image_dir}")
    splitter = FileSplitter(image_dir)

    splitter.split()

    for directory in [
        splitter.train_dir_path,
        splitter.valid_dir_path,
        splitter.test_dir_path,
    ]:
        pic_balancer = PictureBalancer(directory)
        pic_balancer.balance()
