"""
Classification des images à partir du modele pre-entraine

Usage:
    predict.py --pictures_dir=pictures_dir --output_dir=output_dir

Options:
    --help                                      Affiche l'aide
    --version                                   Affiche le numero de version
    --pictures_dir=pictures_dir                 Repertoire contenant les photos
    --output_dir=output_dir                   Repertoire contenant les triées

"""
import os
from docopt import docopt

from pictures.FineTunedImagePredictor import FineTunedImagePredictor

if __name__ == '__main__':
    arguments = docopt(__doc__, version=1)

    fine_tuned_model_weights_path = os.path.join('pictures', 'model', 'fine_tuned_model_weights.h5')
    class_indices_path = os.path.join('pictures', 'model', 'class_indices.npy')

    image_sorter = FineTunedImagePredictor(fine_tuned_model_weights_path, class_indices_path)
    predicted_pictures = image_sorter.predict_pictures_and_sort(arguments['--pictures_dir'], arguments['--output_dir'])

    print(predicted_pictures)
