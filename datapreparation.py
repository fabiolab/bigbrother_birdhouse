"""
Preparation du dataset d'images

Usage:
    datapreparation.py --pictures_dir=pictures_dir

Options:
    --help                                      Affiche l'aide
    --version                                   Affiche le numero de version
    --pictures_dir=pictures_dir                 Repertoire contenant les photos
"""
import os
from docopt import docopt

from pictures.DataPreprocessing import DataProcessor

if __name__ == '__main__':
    arguments = docopt(__doc__, version=1)
    class_indices_path = os.path.join('pictures', 'model', 'class_indices.npy')

    data_processor = DataProcessor(arguments['--pictures_dir'], with_test_set=True, start_from_scratch=True)
    data_processor.build_pictures_set()
    # generated pictures to get a balanced datset
    data_processor.balanced_pictures_set()
    data_processor.save_bottlenecks(class_indices_path)
