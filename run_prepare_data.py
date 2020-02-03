"""Process data
    - split a data set into three corpus : train, test and validation
    - balance the data

    Note that the script will create three directories at the source directory level. For a directory named "source",
    you will get :
    - source_train
    - source_test
    - source_valid
    "test" and "valid" will have both 10% of the whole dataset. Each subdir will be balanced, based on the most
    populated one.

Usage:
    run_prepare_data.py --source=<root_dir>

Options:
    --help                  Display this message
    --source=<root_dir>     Root dir : contains subdirs of pictures

Example:
    run_prepare_data.py --source=data
"""

from docopt import docopt

from wtb.scripts.process_data import process_data

if __name__ == "__main__":
    arguments = docopt(__doc__)

    process_data(arguments["--source"])
