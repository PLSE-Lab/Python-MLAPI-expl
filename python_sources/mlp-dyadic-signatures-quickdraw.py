#!/usr/bin/env python
# coding: utf-8

# # Quickdraw MLP using dyadic signature features
# In this notebook, we investigate the effectiveness of a simple MLP neural network using dyadic signature features on the quickdraw dataset.
# 
# ## Approach
# 1. We use all 340 categories of the `train_simplified` dataset and take 200 samples of each category, 90% for training and 10% for testing
# 2. For each drawing, we compute the stroke embedding (pen-on pen-off) and then compute the dyadic signature features
# 3. Use a simple 3 layer MLP network to classify

# ## Installation

# In[ ]:


get_ipython().system('pip install -r ../input/quickdraw-requirements/requirements.txt')


# In[ ]:


import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from iisignature import sig, logsig, prepare

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import train_test_split, cross_val_score


# ## Definitions and implementation
# 
# We use `sklearn`'s pipeline functionality to do preprocessing. 

# In[ ]:


class SigFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, level=3):
        self.level = level

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([sig(x, self.level) for x in X])


class DyadicSigFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, sig_level=3, d_level=3):
        self.sig_level = sig_level
        self.d_level = d_level

    def fit(self, X, y=None):
        return self

    def transform_instance(self, X):
        T = len(X)-1
        current_times = np.arange(T+1)
        X_fct = interp1d(current_times, X, axis=0)
        features = []
        for n in range(self.d_level+1):
            N = 2**n
            for i in range(N):
                a = i*T/N
                b = (i+1)*T/N
                times = np.concatenate(([a], current_times[int(np.ceil(a)):int(np.ceil(b))], [b]))
                path = X_fct(times)
                features.append(sig(path, self.sig_level))
        return np.concatenate(features)

    def transform(self, X):
        return [self.transform_instance(x) for x in X]
    

class PenOnOff(BaseEstimator, TransformerMixin):
    """3D embedding as specified in http://discovery.ucl.ac.uk/10066168/1/arabic_handwriting_asar2018.pdf"""
    def transform(self, X):
        return [self.transform_instance(x) for x in X]
    
    def fit(self, X, y=None):
        return self
    
    def transform_instance(self, data):
        X = []
        for index, stroke in enumerate(data):
            embedded = np.transpose(stroke + [[2*index]*len(stroke[0])]).tolist()
            if index >= 1:
                X += [[stroke[0][0], stroke[1][0], 2*index-1]]
            X += embedded
            if index < len(data)-1:
                X += [[stroke[0][-1], stroke[1][-1], 2*index+1]]
        return X


# ## Data loading
# 
# We load data from the CSV files.

# In[ ]:


from ast import literal_eval

def load_data(path, nrows=100):
    data = pd.read_csv(path, index_col='key_id', nrows=nrows)
    data['word'] = data['word'].replace(' ', '_', regex=True)
    data['drawing'] = data['drawing'].apply(literal_eval)
    return data


def load_multiple(filenames, size=400, folder='../input/quickdraw-doodle-recognition/train_simplified/'):
    return pd.concat([load_data(folder+fname, nrows=size)
                      for fname in filenames])


# We load the first all 340 categories of the `train_simplified` dataset, and 200 samples of each category.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'categories = !ls ../input/quickdraw-doodle-recognition/train_simplified/\ncategories = categories[0:340]\ndf = load_multiple(categories, size=200)')


# Let's plot a few of these drawings.

# In[ ]:


def plot_drawing(X):
    """X is a collection of strokes"""
    for x,y in X:
        plt.plot(x, y, marker='.')
    plt.gca().invert_yaxis()
    plt.axis('equal')

plt.figure(0)
plot_drawing(df.drawing.values[12002])
plt.figure(1)
plot_drawing(df.drawing.values[15010])
plt.figure(2)
plot_drawing(df.drawing.values[42446])


# ## Features preprocessing
# 
# We set the dyadic level to 4 and signature truncation level to 3. See [this paper](https://ora.ox.ac.uk/objects/uuid:dd1ec888-c558-4385-8f48-4efcb867b682/download_file?file_format=pdf&safe_filename=Lyons%2Bet%2Bal%252C%2BRotation-free%2Bonline%2Bhandwritten%2Bcharacter%2Brecognition%2Busing%2Bdyadic%2Bpath%2Bsignature%2Bfeatures%252C%2Bhanging%2Bnormal.pdf&type_of_work=Conference+item) for details. These correspond to `n` and `m` in the paper respectively.

# In[ ]:


get_ipython().run_cell_magic('time', '', "d_level = 4 # dyadic level\nsig_level = 3 # signature truncation level\ndsigmodel = Pipeline([\n    ('penonoff',   PenOnOff()),\n    ('dsignature', DyadicSigFeatures(sig_level=sig_level, d_level=d_level)),\n    ('scale',      StandardScaler()),\n])\n\nX = dsigmodel.fit_transform(df.drawing.values)\ny = LabelBinarizer().fit_transform(df.word.values)")


# We'll use 90% of the data for training and 10% for testing.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# In[ ]:


X_train.shape


# ## Neural network setup

# In[ ]:


from keras.layers import Dense, Dropout
from keras.models import Sequential


# In[ ]:


num_features = X_train.shape[1]
num_classes = len(categories)

model = Sequential()
model.add(Dense(units=2048, activation='relu', input_shape=(num_features,)))
model.add(Dense(units=2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=2048, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=num_classes, activation='softmax'))
model.summary()


# We will use the same SGD optimizer as in the paper on [dyadic signatures](https://ora.ox.ac.uk/objects/uuid:dd1ec888-c558-4385-8f48-4efcb867b682/download_file?file_format=pdf&safe_filename=Lyons%2Bet%2Bal%252C%2BRotation-free%2Bonline%2Bhandwritten%2Bcharacter%2Brecognition%2Busing%2Bdyadic%2Bpath%2Bsignature%2Bfeatures%252C%2Bhanging%2Bnormal.pdf&type_of_work=Conference+item). In addition to the accuracy metric, we use the top 3 categorial accuracy (used by this Kaggle competition).

# In[ ]:


from keras.optimizers import SGD

opt = SGD(lr=0.02, decay=5e-4, momentum=0.9)
def top_3(y_true, y_pred): 
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', top_3])


# ## Training & Performance
# 
# We train for 20 epochs using batch sizes of 100.

# In[ ]:


history = model.fit(X_train, y_train, batch_size=100, epochs=20, validation_split=.1)


# In[ ]:


plt.plot(history.history['acc']) # blue
plt.plot(history.history['val_acc']) # orange
plt.plot(history.history['val_top_3']) # green


# Our final score on the testing set is (loss, accuracy, top 3 accuracy):

# In[ ]:


model.evaluate(X_test, y_test)

