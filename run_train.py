"""Train a model from directories containing pictures

Usage:
    run_train.py --train_dir=<train_dir> --validation_dir=<validation_dir> --model_dir=<model_dir>

Options:
    --help                              Display this message
    --train_dir=<train_dir>             Train dir : contains subdirs of pictures
    --validation_dir=<validation_dir>   Validation dir : contains subdirs of pictures
    --model_dir=<model_dir>             Model dir : output dir where the generated models are stored

Example:
    run_train.py --train_dir=data_train --validation_dir=data_valid
"""

from docopt import docopt

from wtb.scripts.train import train

if __name__ == "__main__":
    # Command line args
    # __doc__ contains the module docstring
    arguments = docopt(__doc__)

    train(arguments["--train_dir"], arguments["--validation_dir"], arguments["--model_dir"])
