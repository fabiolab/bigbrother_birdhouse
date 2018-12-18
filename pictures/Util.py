import matplotlib

#matplotlib.use('Agg')  # fixes issue if no GUI provided
import matplotlib.pyplot as plt

import numpy as np
import os
import itertools


def plot_confusion_matrix(cnf_matrix, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    confusion_matrix_dir = './confusion_matrix_plots'
    if not os.path.exists(confusion_matrix_dir):
        os.mkdir(confusion_matrix_dir)

    plt.cla()
    plt.figure()
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cnf_matrix)

    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, cnf_matrix[i, j],
                 horizontalalignment="center",
                 color="#BFD1D4" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if normalize:
        plt.savefig(os.path.join(confusion_matrix_dir, 'normalized.jpg'))
    else:
        plt.savefig(os.path.join(confusion_matrix_dir, 'without_normalization.jpg'))
