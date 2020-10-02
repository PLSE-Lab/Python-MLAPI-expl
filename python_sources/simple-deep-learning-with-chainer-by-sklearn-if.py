"""
[Introduction]
1. chainer is deep learning framework written in python.
It supports dynamic graph construction (define-by-run).
Chainer is used in research field as well, since it is quite flexible library.
https://github.com/chainer/chainer

2. chainer_sklearn is extension module (library) to support sklearn-like interface
on top of chainer.
https://github.com/corochann/chainer_sklearn

This script provides quick & easy example for deep learning classification
using chainer & chainer_sklearn.
I hope the code is useful for Deep learning beginners to start with!


[Quick explanation of the code]
The neural network model can be defined as class (here, `MLP` class) in chainer.

When the mlp model class is defined, it can be instantiated and wrapped by
`SklearnWrapperClassifier` to solve classification task.

After that, training and predict interface is quite same with sklearn model,
you can just call `fit` to train the model, and call `predict_proba` for predict
probability for test data.


[How to install module]
Please install following module to run this script.
$ pip install chainer
$ pip install chainer_sklearn

If you want to utilize GPU, please install cupy as well.
$ pip install cupy
"""
import os
import numpy as np
import pandas as pd

import chainer
from chainer import optimizers, serializers
import chainer.links as L
import chainer.functions as F
from chainer_sklearn.links import SklearnWrapperClassifier


# --- Define Multi Layer Perceptron (MLP) Network ---
class MLP(chainer.Chain):
    """
    This is 3-layer MLP model definition
    where ReLU is used for nonlinear activation.
    
    Try changing non-linear activation (F.relu to F.sigmoid etc), 
    or adding new Linear layer to make deeper network to see the performance!
    """

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # input size of each layer will be inferred when set to None
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        return self.l3(h)


# --- Load & build data ---
print('Loading data...')
DATA_DIR = '../input'
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

train_x = train.iloc[:, 1:].values.astype('float32')
train_y = train.iloc[:, 0].values.astype('int32')
test_x = test.values.astype('float32')

print('train_x shape {}'.format(train_x.shape))
print('train_y shape {}'.format(train_y.shape))
print('test_x shape {}'.format(test_x.shape))

# --- Construct a model ---
print('Constructing model...')
hidden_dim = 32  # Hidden dim for neural network
out_dim = 10     # Number of labels to classify, it is 10 for MNIST task.
# Note that input dimension is not necessary to set.
# Chainer model automatically infer input dimension.
mlp_model = MLP(hidden_dim, out_dim)

# `chainer_sklearn` library is extension library for chainer to support
# sklearn interface.
# `SklearnWrapperClassifier` can be used for classification task

# You may set device=-1 to use CPU,
# or GPU device id (positive number) to use GPU.
model = SklearnWrapperClassifier(mlp_model, device=-1)

# --- Training ---
print('Training start...')
model.fit(train_x, train_y,
          batchsize=16,
          epoch=20,
          progress_report=True,  # Set to False if you use PyCharm
          # You may use other optimizers, for example
          # optimizer=optimizers.MomentumSGD(lr=0.0001))
          optimizer=optimizers.Adam())


# Now model training has finished.
# Trained model can be saved as follows,
out_dir = '.'
serializers.save_npz(os.path.join(out_dir, 'mlp_model.npz'), model)

# Saved model can be loaded as follows,
# serializers.load_npz('result/mlp_model.npz', model)

# --- Predict ---
print('Predicting...')
test_y = model.predict(test_x)
test_y = test_y.ravel()

# --- Save to submission file ---
submission_filepath = os.path.join(out_dir,
                                   'submission_mlp_chainer_sklearn.csv')
print('Saving submission file to {}...'.format(submission_filepath))
result_dict = {
    'ImageId': np.arange(1, len(test_y) + 1),
    'Label': test_y
}
df = pd.DataFrame(result_dict)
df.to_csv(submission_filepath, index_label=False, index=False)
