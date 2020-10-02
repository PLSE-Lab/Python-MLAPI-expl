#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn import preprocessing

import gc
from keras.models import Model
from keras.layers import Input, Dense

from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers import Input, BatchNormalization, Dense, Reshape, Lambda, Dropout
from keras import metrics
#from . import backend as K
from keras import backend as K


# In[ ]:





# In[ ]:





# In[ ]:





# # Keras Model

# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:





# In[ ]:



train = pd.read_csv('../input/train_2016_v2.csv', nrows=13200) 
prop = pd.read_csv('../input/properties_2016.csv', nrows=13200) 
sample = pd.read_csv('../input/sample_submission.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


prop['propertycountylandusecode'] = prop['propertycountylandusecode'].apply(lambda x: str(x))
encoder = preprocessing.LabelEncoder()
encoder.fit(prop['propertycountylandusecode'])
prop['propertycountylandusecode'] = encoder.transform(prop['propertycountylandusecode'])

prop['propertyzoningdesc'] = prop['propertyzoningdesc'].apply(lambda x: str(x))
encoder2 = preprocessing.LabelEncoder()
encoder2.fit(prop['propertyzoningdesc'])
prop['propertyzoningdesc'] = encoder2.transform(prop['propertyzoningdesc'])


# In[ ]:





# In[ ]:





# In[ ]:


# Discard all non-numeric data
prop = prop.select_dtypes([np.number])
train = train.select_dtypes([np.number])
sample = sample.select_dtypes([np.number])

gc.collect()


x_train = prop.drop(['parcelid'], axis=1)

gc.collect()


# In[ ]:





# In[ ]:





# In[ ]:


train_columns = x_train.columns
temp = pd.merge(left=train, right=prop, on=('parcelid'), how='outer')
temp = temp.fillna(0)
x_train = temp.drop(['parcelid', 'logerror'], axis=1).values
y_train = temp['logerror'].values
gc.collect()


# In[ ]:





# In[ ]:





# In[ ]:


scaler = preprocessing.StandardScaler()
# x_train.apply(lambda x: (x - x.mean()) / (x.max() - x.min()))
x_train = scaler.fit_transform(x_train)

# Normalize (across the whole dataframe cos we dun care)
mean_x = x_train.mean().astype(np.float32)
std_x = x_train.std().astype(np.float32)

mean_y = y_train.mean().astype(np.float32)
std_y = y_train.std().astype(np.float32)


# In[ ]:


def normalize(x):
    return (x-mean_x)/std_x

def normalize_y(y):
    return (y-mean_y)/std_y

def de_normalize_y(y):


    return (y*std_y) + mean_y

y_train = normalize(y_train)


# In[ ]:





# In[ ]:





# ***Dense***  implements the operation: output = activation(dot(input, kernel) + bias) where activation is the element-wise activation function passed as the activation argument, kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer (only applicable if use_bias is True).
# 
# ***Dropout*** consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.
# 
# **lambda** Wraps arbitrary expression as a Layer object.
# 
# **Batch normalization layer** (Ioffe and Szegedy, 2014).
# Normalize the activations of the previous layer at each batch, i.e. applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.
# 
# 
# 
# **ReLU** The Rectified Linear Unit computes the function f(x)=max(0,x)f(x)=max(0,x). In other words, the activation is simply thresholded at zero (see image above on the left). 
# 

# In[ ]:





# In[ ]:


# Build a simple model
model = Sequential([
    #Lambda(normalize,input_shape=(52, )),
	Dense(60,input_shape=(54, )),
    BatchNormalization(),
    Dropout(0.08),
	Dense(160, activation='relu'),
	BatchNormalization(),
    Dropout(0.38),
    Dense(20, activation='relu'),
    BatchNormalization(),
    Dense(1, activation='sigmoid')
])


# In[ ]:





# In[ ]:





# In[ ]:


model.compile(loss='mean_absolute_error', optimizer='adam')
model.fit(x_train, y_train, batch_size=24, epochs=15)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, batch_size=32, epochs=25)


# **fit**     Trains the model for a fixed number of epochs (iterations on a dataset).
# **compile**  Configures the model for training.

# In[ ]:





# In[ ]:


# Prepare the submission data
sample['parcelid'] = sample['ParcelId']
del sample['ParcelId']

df_test = pd.merge(sample, prop, on='parcelid', how='left')
df_test = df_test.fillna(0)

x_test = df_test[train_columns]
#predict(self, x, batch_size=None, verbose=0, steps=None)
p_test = model.predict(x_test.values)
p_test = de_normalize_y(p_test)


# In[ ]:


p_test


# Returns the loss value & metrics values for the model in test mode.
# 
# Computation is done in batches.

# In[ ]:


# evaluate(self, x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None)
q_test=model.evaluate(x_train,y_train)
#q_test= de_normalize(q_test)


# In[ ]:





# In[ ]:


q_test


# In[ ]:


q_test= de_normalize_y(q_test)


# In[ ]:


q_test


# Runs a single gradient update on a single batch of data.
# 
# 

# In[ ]:


#train_on_batch(self, x, y, sample_weight=None, class_weight=None)
model.train_on_batch(x_train,y_train)


# Test the model on a single batch of samples.
# 
# 

# In[ ]:


#test_on_batch(self, x, y, sample_weight=None)
model.test_on_batch(x_train,y_train, sample_weight=None)


# Returns predictions for a single batch of samples.
# 
# 

# In[ ]:


#predict_on_batch(self, x)
model.predict_on_batch(x_train)


# In[ ]:


#model.evaluate_on_batch(x_train,y_train)


# **generator**
# Fits the model on data yielded batch-by-batch by a Python generator.
# The generator is run in parallel to the model, for efficiency. For instance, this allows you to do real-time data augmentation on images on CPU in parallel to training your model on GPU.

# In[ ]:


###fit_generator(self, generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)

#model.fit_generator(x_train, generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)


# In[ ]:


#evaluate_generator(self, generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False)
#model.evaluate_generator(self, generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False)


# In[ ]:


#predict_generator(self, generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
#model.predict_generator(x_train, generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)


# Retrieves a layer based on either its name (unique) or index.
# 
# Indices are based on order of horizontal graph traversal (bottom-up).

# In[ ]:


#get_layer(self, name=None, index=None)
#lay=model.get_layer(x_train.all())


# In[ ]:


print(lay)


# **generate output result to output.csv**

# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = p_test

sub.to_csv('output2.csv', index=False, float_format='%.5f')


# In[ ]:





# In[ ]:


sub.head()


# In[ ]:





# In[ ]:





# In[ ]:





# ## Support vector Machine(optional)
# not working 

# In[ ]:





# In[ ]:




from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import utils
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot



fig, subaxes = plt.subplots(1, 1, figsize=(7, 5))
this_C = 1.0
clf = SVC(kernel = 'linear', C=this_C).fit(x_train, y_train)
title = 'Linear SVC, C = {:.3f}'.format(this_C)
plot_class_regions_for_classifier_subplot(clf, X_train, y_train, None, None, title, subaxes)
# In[ ]:




from sklearn.svm import LinearSVC


clf = LinearSVC().fit(x_train, y_train)

print('Accuracy of Linear SVC classifier on training set: {:.2f}'
     .format(clf.score(x_train, y_train)))
print('Accuracy of Linear SVC classifier on test set: {:.2f}'
     .format(clf.score(x_test, y_test)))
# In[ ]:





# In[ ]:





# **Accuracy of Support Vector Machine classifier**

from sklearn.svm import SVC

svm = SVC(kernel='rbf', C=1).fit(X_train, y_train)
svm.score(X_test, y_test)
# In[ ]:




