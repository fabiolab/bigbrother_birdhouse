# WTB
Automatic classification of birds for this project : http://www.cabane-oiseaux.org/

## Installation

This project requires python 3.6+

Then, just clone this project and run this command from the main dir :

```
pip install -r requirements.txt
```

## Tensorflow optimization

Depending on your hardware configuration, you can get an optimized version of tensorflow.
I used a pre-compiled version for Intel CPU :

```
pip install https://storage.googleapis.com/intel-optimized-tensorflow/tensorflow-1.11.0-cp36-cp36m-linux_x86_64.whl
```

But if are lucky enough to own a NVIDIA GPU card, you will get best performances with a specific version.

## Run the notebook

Then, run the notebook with :

```
python jupyter notebook
```

## Run the tests
```
python run_tests.py
```

## Train the model from command line
```
python run_train.py --train_dir=data_train --validation_dir=data_valid --debug
```