from os import path

from FileSplitter import FileSplitter
from PictureBalancer import PictureBalancer

if __name__ == "__main__":
    script_dir = path.dirname(path.realpath(__file__))
    splitter = FileSplitter(path.join(script_dir, "data"))

    splitter.split()

    for directory in [
        splitter.train_dir_path,
        splitter.valid_dir_path,
        splitter.test_dir_path,
    ]:
        pic_balancer = PictureBalancer(directory)
        pic_balancer.balance()
