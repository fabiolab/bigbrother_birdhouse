from wtb.dataprocess.file_splitter import FileSplitter
from wtb.dataprocess.picture_balancer import PictureBalancer


def process_data(root_dir):
    splitter = FileSplitter(root_dir)

    splitter.split()

    for directory in [
        splitter.train_dir_path,
        splitter.valid_dir_path,
        splitter.test_dir_path,
    ]:
        pic_balancer = PictureBalancer(directory)
        pic_balancer.balance()
