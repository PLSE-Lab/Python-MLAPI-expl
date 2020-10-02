"""
An autoencoder is an artificial neural network used for unsupervised learning of efficient codings.

The aim of an autoencoder is to learn a representation (encoding) for a set of data, typically for the purpose of dimensionality reduction.
If linear activations are used, or only a single sigmoid hidden layer, then the optimal solution to an autoencoder is strongly related to principal component analysis (PCA).

With appropriate dimensionality and sparsity constraints, autoencoders can learn data projections that are more interesting than PCA or other basic techniques.
It turns out that PCA only allows linear transformation of a data vectors. Autoencoders and RBMs, on other hand, are non-linear by the nature, and thus, 
they can learn more complicated relations between visible and hidden units. Moreover, they can be stacked, which makes them even more powerful.
"""

from __future__ import division, print_function, absolute_import
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import batch_normalization, local_response_normalization
from tflearn.layers.estimator import regression

import tensorflow as tf

import numpy as np
import pandas as pd

from sklearn import preprocessing

from sklearn.decomposition import PCA

from scipy.stats import skew
from scipy.stats.stats import pearsonr

df = pd.read_csv('../input/train.csv')
dt = pd.read_csv('../input/test.csv')

#Preprocessing
#dt.replace(['NA', ''], [0., 0.])
#dt.fillna(0., inplace=True)

all_data = pd.concat((df.loc[:,'MSSubClass':'SaleCondition'], dt.loc[:,'MSSubClass':'SaleCondition']))

#log transform the target:
#df["SalePrice"] = np.log1p(df["SalePrice"])

""" #log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats]) """

all_data = pd.get_dummies(all_data)

#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())

#creating matrices for sklearn:
X_train = all_data[:df.shape[0]]
X_test = all_data[df.shape[0]:]
y = df.SalePrice

X = np.array(X_train)
testX = np.array(X_test)
Y = np.array(y).reshape([-1, 1])


std_scale = preprocessing.StandardScaler().fit(X)
X = std_scale.fit_transform(X)
testX = std_scale.fit_transform(testX)

def pca_transform(data, n_components, whiten):
    pca = PCA(n_components=n_components, whiten=whiten)
    pca.fit(data)
    return pca.fit_transform(data)

#X = pca_transform(X, 200, False)
#testX = pca_transform(testX, 200, False)


# Building the encoder.
encoder = tflearn.input_data(shape=[None, 288])
encoder = tflearn.fully_connected(encoder, 128)
encoder = tflearn.fully_connected(encoder, 64)

# Building the decoder.
decoder = tflearn.fully_connected(encoder, 128)
decoder = tflearn.fully_connected(decoder, 288)

# Regression, with mean square error.
net = tflearn.regression(decoder, optimizer='adam', learning_rate=0.001, loss='mean_square', metric=None)

# Training the auto encoder.
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(X, X, n_epoch=100, validation_set=(testX, testX), run_id="auto_encoder", batch_size=256)

#encode_decode = np.array(model.predict(testX))

# New model, re-using the same session, for weights sharing.
encoding_model = tflearn.DNN(encoder, session=model.session)
#new_X = np.array(encoding_model.predict(X))
#new_testX = np.array(encoding_model.predict(testX))
new_X = np.array(model.predict(X))
new_testX = np.array(model.predict(testX))

#np.savetxt("preprocess_train.csv", np.hstack((new_X, Y)), delimiter=",", comments="")
#np.savetxt("preprocess_test.csv", new_testX, delimiter=",", comments="")
np.save("preprocess_train.npy", np.hstack((new_X, Y)))
np.save("preprocess_test.npy", new_testX)