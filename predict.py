"""
Classification des images à partir du modele pre-entraine

Usage:
    predict.py --pictures_dir=pictures_dir

Options:
    --help                                      Affiche l'aide
    --version                                   Affiche le numero de version
    --pictures_dir=pictures_dir                 Repertoire contenant les photos
"""
import os
from docopt import docopt

from pictures.FineTunedImagePredictor import FineTunedImagePredictor

if __name__ == '__main__':
    arguments = docopt(__doc__, version=1)

    model_weights_path = os.path.join('pictures', 'model', 'model_weights.h5')
    class_indices_path = os.path.join('pictures', 'model', 'class_indices.npy')

    image_predictor = FineTunedImagePredictor(model_weights_path, class_indices_path)
    predicted_pictures = image_predictor.predict_pictures(arguments['--pictures_dir'])

    print(predicted_pictures)
