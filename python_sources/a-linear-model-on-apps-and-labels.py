#!/usr/bin/env python
# coding: utf-8

# original:  This notebook shows how to build a linear model on features from apps, app labels, phone brands and device models. It uses LogisticRegression classifier from sklearn.   It also shows an efficient way of constructing bag-of-apps and bag-of-labels features without concatenating a bunch of strings.
# 
# modified:  I grafted on the popular Keras script by Samarth Agarwal to the matrices DuneDweller created here, and then made a 2D autoencoder, which allows me to plot each class (gender+age) at the end of this.

# In[ ]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss


# ## Load data

# In[ ]:


datadir = '../input'
gatrain = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'),
                      index_col='device_id')
gatest = pd.read_csv(os.path.join(datadir,'gender_age_test.csv'),
                     index_col = 'device_id')
phone = pd.read_csv(os.path.join(datadir,'phone_brand_device_model.csv'))
# Get rid of duplicate device ids in phone
phone = phone.drop_duplicates('device_id',keep='first').set_index('device_id')
events = pd.read_csv(os.path.join(datadir,'events.csv'),
                     parse_dates=['timestamp'], index_col='event_id')
appevents = pd.read_csv(os.path.join(datadir,'app_events.csv'), 
                        usecols=['event_id','app_id','is_active'],
                        dtype={'is_active':bool})
applabels = pd.read_csv(os.path.join(datadir,'app_labels.csv'))


# ## Feature engineering
# 
# The features I'm going to use include:
# 
# * phone brand
# * device model
# * installed apps
# * app labels
# 
# I'm going to one-hot encode everything and sparse matrices will help deal with a very large number of features.
# 
# ### Phone brand
# 
# As preparation I create two columns that show which train or test set row a particular device_id belongs to.

# In[ ]:


gatrain['trainrow'] = np.arange(gatrain.shape[0])
gatest['testrow'] = np.arange(gatest.shape[0])


# A sparse matrix of features can be constructed in various ways. I use this constructor:
# 
#     csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
#     where ``data``, ``row_ind`` and ``col_ind`` satisfy the
#     relationship ``a[row_ind[k], col_ind[k]] = data[k]``
#     
# It lets me specify which values to put into which places in a sparse matrix. For phone brand data the `data` array will be all ones, `row_ind` will be the row number of a device and `col_ind` will be the number of brand.

# In[ ]:


brandencoder = LabelEncoder().fit(phone.phone_brand)
phone['brand'] = brandencoder.transform(phone['phone_brand'])
gatrain['brand'] = phone['brand']
gatest['brand'] = phone['brand']
Xtr_brand = csr_matrix((np.ones(gatrain.shape[0]), 
                       (gatrain.trainrow, gatrain.brand)))
Xte_brand = csr_matrix((np.ones(gatest.shape[0]), 
                       (gatest.testrow, gatest.brand)))
print('Brand features: train shape {}, test shape {}'.format(Xtr_brand.shape, Xte_brand.shape))


# ### Device model

# In[ ]:


m = phone.phone_brand.str.cat(phone.device_model)
modelencoder = LabelEncoder().fit(m)
phone['model'] = modelencoder.transform(m)
gatrain['model'] = phone['model']
gatest['model'] = phone['model']
Xtr_model = csr_matrix((np.ones(gatrain.shape[0]), 
                       (gatrain.trainrow, gatrain.model)))
Xte_model = csr_matrix((np.ones(gatest.shape[0]), 
                       (gatest.testrow, gatest.model)))
print('Model features: train shape {}, test shape {}'.format(Xtr_model.shape, Xte_model.shape))


# ### Installed apps features
# 
# For each device I want to mark which apps it has installed. So I'll have as many feature columns as there are distinct apps.
# 
# Apps are linked to devices through events. So I do the following:
# 
# - merge `device_id` column from `events` table to `app_events`
# - group the resulting dataframe by `device_id` and `app` and aggregate
# - merge in `trainrow` and `testrow` columns to know at which row to put each device in the features matrix

# In[ ]:


appencoder = LabelEncoder().fit(appevents.app_id)
appevents['app'] = appencoder.transform(appevents.app_id)
napps = len(appencoder.classes_)
deviceapps = (appevents.merge(events[['device_id']], how='left',left_on='event_id',right_index=True)
                       .groupby(['device_id','app'])['app'].agg(['size'])
                       .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                       .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                       .reset_index())
deviceapps.head()


# Now I can build a feature matrix where the `data` is all ones, `row_ind` comes from `trainrow` or `testrow` and `col_ind` is the label-encoded `app_id`.

# In[ ]:


d = deviceapps.dropna(subset=['trainrow'])
Xtr_app = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.app)), 
                      shape=(gatrain.shape[0],napps))
d = deviceapps.dropna(subset=['testrow'])
Xte_app = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.app)), 
                      shape=(gatest.shape[0],napps))
print('Apps data: train shape {}, test shape {}'.format(Xtr_app.shape, Xte_app.shape))


# ### App labels features
# 
# These are constructed in a way similar to apps features by merging `app_labels` with the `deviceapps` dataframe we created above.

# In[ ]:


applabels = applabels.loc[applabels.app_id.isin(appevents.app_id.unique())]
applabels['app'] = appencoder.transform(applabels.app_id)
labelencoder = LabelEncoder().fit(applabels.label_id)
applabels['label'] = labelencoder.transform(applabels.label_id)
nlabels = len(labelencoder.classes_)


# In[ ]:


devicelabels = (deviceapps[['device_id','app']]
                .merge(applabels[['app','label']])
                .groupby(['device_id','label'])['app'].agg(['size'])
                .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                .reset_index())
devicelabels.head()


# In[ ]:


d = devicelabels.dropna(subset=['trainrow'])
Xtr_label = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.label)), 
                      shape=(gatrain.shape[0],nlabels))
d = devicelabels.dropna(subset=['testrow'])
Xte_label = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.label)), 
                      shape=(gatest.shape[0],nlabels))
print('Labels data: train shape {}, test shape {}'.format(Xtr_label.shape, Xte_label.shape))


# ### Concatenate all features

# In[ ]:


Xtrain = hstack((Xtr_brand, Xtr_model, Xtr_app, Xtr_label), format='csr')
Xtest =  hstack((Xte_brand, Xte_model, Xte_app, Xte_label), format='csr')
print('All features: train shape {}, test shape {}'.format(Xtrain.shape, Xtest.shape))


# ## Neural network setup

# happycube:  Now to pull in keras and friends - most of this code is based directly from https://www.kaggle.com/samarthagarwal23/talkingdata-mobile-user-demographics/bag-of-apps-keras-11-08-16/run/327685

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.optimizers import SGD

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss

from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import GaussianDropout, GaussianNoise
from keras.layers.advanced_activations import PReLU
from keras.callbacks import EarlyStopping
#from keras.callbacks import ModelCheckpoint


# In[ ]:


def rstr(df): return df.dtypes, df.head(3) ,df.apply(lambda x: [x.unique()]), df.apply(lambda x: [len(x.unique())]),df.shape

def batch_generator(X, y, batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0


# In[ ]:


targetencoder = LabelEncoder().fit(gatrain.group)
y = targetencoder.transform(gatrain.group)
nclasses = len(targetencoder.classes_)

X_train, X_val, y_train, y_val = train_test_split(
    Xtrain, y, train_size=.98, random_state=10)


# This is where the modifications start - a 2-neuron dense network with PReLU activation is added, with the PReLU layer broken out into it's own variable, so that we can hook into it later

# In[ ]:


def baseline_model():
    # create model
    model = Sequential()

    model.add(Dense(150, input_dim=X_train.shape[1], init='glorot_normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    
    model.add(Dense(2, init='glorot_normal'))
    pre = PReLU()
    model.add(pre)

    model.add(Dense(50, init='glorot_normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    
    model.add(Dense(12, init='glorot_normal', activation='softmax'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])  #logloss
    return model,pre

model,pre=baseline_model()

early_stopping = EarlyStopping(monitor='val_loss', patience=2)

#checkpoint = ModelCheckpoint("weights.best.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')

fit= model.fit_generator(generator=batch_generator(X_train, y_train, 400, True),
                         nb_epoch=1000000,
                         samples_per_epoch=X_train.shape[0],
                         validation_data=(X_val.todense(), y_val), verbose=2,
                         callbacks=[early_stopping])


# From: https://github.com/julienr/ipynb_playground/blob/master/keras/convmnist/keras_cnn_mnist_v1.ipynb
# 
# And Keras issue #431 - near the end where the more recent backend api is used

# In[ ]:


from keras import backend as K

inputs = [K.learning_phase()] + model.inputs

fpre = K.function(inputs, [pre.output])


# This function processes the matrix in chunks - converting to a dense matrix takes too much memory for a 4GB GPU to handle... let alone Kaggle's kernel engine!

# In[ ]:


def tocoords(inp):
    coords = np.zeros((inp.shape[0], 2), dtype=np.float32)
    
    for i in range(0, inp.shape[0], 400):
        #print(i, inp.shape[0])
        if i + 400 > inp.shape[0]:
            j = inp.shape[0]
        else:
            j = i + 400

        y = fpre([0] + [inp[i:j].todense()])   
        #print(y[0])
        coords[i:j] = y[0]

    return coords

coords_train = tocoords(X_train)
coords_valid = tocoords(X_val)


# Now that we have the coordinates, build the lines for each gender...

# In[ ]:


fline = []
for c in range(0, 6):
    mask = y_train == c
    coords_masked = coords_train[np.where(mask)]
    
    fline.append([np.mean(coords_masked[:,0]), np.mean(coords_masked[:,1])])
    #print(c, min(coords_masked[:,0]), min(coords_masked[:,1]), max(coords_masked[:,0]), max(coords_masked[:,1]))
    print(targetencoder.classes_[c], np.mean(coords_masked[:,0]), np.mean(coords_masked[:,1]), np.std(coords_masked[:,0]), np.std(coords_masked[:,1]))
    
mline = []
for c in range(6, 12):
    mask = y_train == c
    coords_masked = coords_train[np.where(mask)]
    
    mline.append([np.mean(coords_masked[:,0]), np.mean(coords_masked[:,1])])
    #print(c, min(coords_masked[:,0]), min(coords_masked[:,1]), max(coords_masked[:,0]), max(coords_masked[:,1]))
    print(targetencoder.classes_[c], np.mean(coords_masked[:,0]), np.mean(coords_masked[:,1]), np.std(coords_masked[:,0]), np.std(coords_masked[:,1]))


# ... and now draw them

# In[ ]:


fline = np.array(fline)
plt.plot(fline[:,0], fline[:,1], 'g')

mline = np.array(mline)
plt.plot(mline[:,0], mline[:,1], 'r'
)


# In[ ]:




