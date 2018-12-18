"""
Build du modele

Usage:
    buildmodel.py --pictures_dir=pictures_dir

Options:
    --help                       Affiche l'aide
    --version                    Affiche le numero de version
    --pictures_dir=pictures_dir  Repertoire contenant les photos
"""
import os
from docopt import docopt

from pictures.FineTunedModelBuilder import FineTunedModelBuilder

if __name__ == '__main__':
    arguments = docopt(__doc__, version=1)

    model_weights_path = os.path.join('pictures', 'model', 'model_weights.h5')
    fine_tuned_model_weights_path = os.path.join('pictures', 'model', 'fine_tuned_model_weights.h5')
    class_indices_path = os.path.join('pictures', 'model', 'class_indices.npy')

    model_builder = FineTunedModelBuilder(arguments['--pictures_dir'], model_weights_path, fine_tuned_model_weights_path, class_indices_path)
    top_model_history = model_builder.train_top_model()
    top_model_history = model_builder.train_fine_tuning_top_model()


