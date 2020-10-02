#!/usr/bin/env python
# coding: utf-8

# # What is it
# 
# The idea is to try to compare NN performance with one emedding applied to each categorical feature as 
# opposed to one embedding applied to all features at once. 
# The hope is that if there are feature interactions using all features at the same time
# may provide embeddings that give us better predictions in models.
# 
# The results below show that there is no difference. The ROC AUC values from both models are the same.
# This result may be specific to this data set, though. 
# In artificial data set there may be no feature interactions at all.
# 
# Still, there may be other advantages of using one embedding for all. Simpler model, simpler inputs, for example.
# 
# ## TODO
# 
# * Different models, try adding max pooling for example
# * Different data set, same approach applied to different data set may produce different results. But than again, maybe not. Maybe NN learns all interactions it can learn in both cases.
# 

# In[ ]:


import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding, MaxPooling1D, Concatenate
from keras.layers import Input
from keras import backend as K 

import tensorflow as tf
from sklearn.metrics import roc_auc_score


# # Read data

# In[ ]:


base_data_dir = '../input/cat-in-the-dat/'
dtr = pd.read_csv(base_data_dir + "train.csv")
dts = pd.read_csv(base_data_dir + "test.csv")
dts.target = np.NaN
d = pd.concat([dtr, dts], sort=False)
train_set = dtr.shape[0]
del(dtr, dts)


# In[ ]:


d.columns


# In[ ]:


cat_features = [i for i in d.columns if not i in ("id","target")]
print(cat_features)


# In[ ]:


for c in cat_features:
    d[c] = d[c].astype("category")


# # Map categorical feature into vectors and matrix

# First prepare a list of vectors with numeric representation for each categorical value.
# Note that if there is a NA in the category it is mapped into -1 code by pandas.
# In this case there are no NAs. 
# If there are NAs we should be using code values + 1.

# In[ ]:


cat_vectors = [d[c].cat.codes.to_numpy() for c in cat_features]


# Now, make one single merged matrix with all categorical codes.
# Each categorical code must be unique. 
# To achieve that values in each subsequent categorical vector is increased by the number of levels in previous vectors.
# Then all these vectors are merged into a single matrix.

# In[ ]:


cat_size = [len(d[c].cat.categories) for c in cat_features]
cat_offset = np.cumsum([0] + cat_size[:-1])
cat_vectors2 = [cat_vectors[i] + cat_offset[i] for i in range(len(cat_vectors))]
cat_matrix = np.concatenate([np.reshape(np.ravel(c),(-1,1)) for c in cat_vectors2], axis=1)


# In[ ]:


print(cat_matrix.shape)
print(cat_matrix[0:2,])


# # Prepare train and test set split

# In[ ]:


from sklearn.model_selection import train_test_split
train_idx, test_idx = train_test_split(range(train_set), test_size=0.2)


# # Try NN model using emedding across all features

# In[ ]:


X_train = cat_matrix[train_idx,:]
X_test = cat_matrix[test_idx,:]
y_train = d.target.iloc[train_idx]
y_test = d.target.iloc[test_idx]
# X_train, X_test, y_train, y_test = train_test_split(cat_matrix[0:train_set,:], d.target[0:train_set], test_size=0.2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[ ]:


max_features = np.max(cat_matrix)+1
maxlen = cat_matrix.shape[1]
print(max_features, maxlen)


# In[ ]:


# from https://stackoverflow.com/questions/41032551/how-to-compute-receiving-operating-characteristic-roc-and-auc-in-keras
def auc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

embedding_size = 10

model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(10, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(10, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auc])


# In[ ]:


print(model.summary())


# In[ ]:


hist = model.fit(X_train, y_train,validation_data=(X_test, y_test),
          batch_size=100, epochs=3, shuffle=True)


# Look at the model convergence graphs.

# In[ ]:


import matplotlib.pyplot as plt
plt.plot(hist.history['auc'])
plt.plot(hist.history['val_auc'])
plt.title('model ROC AUC')
plt.ylabel('AUC')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Look at the ROC graph.

# In[ ]:


y_pred = model.predict(X_test)
K.clear_session()


# In[ ]:


from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr)
plt.xlabel('False positive')
plt.ylabel('True positive')
plt.title('ROC auc='+str(auc(fpr, tpr)))
plt.show()


# # Try model with each feature having it's own embedding

# In[ ]:


# from https://stackoverflow.com/questions/41032551/how-to-compute-receiving-operating-characteristic-roc-and-auc-in-keras
def auc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

def prepare_one_cat_layer(c):
    inp = Input(shape=(1,))
    es = int(round(np.log(np.max(c))))+2 # just a guess
    emb = Embedding(np.max(c)+1, es, input_length=1)(inp)
    return (inp,emb)

cat_layers = [prepare_one_cat_layer(c) for c in cat_vectors]

x = Concatenate()([c[1] for c in cat_layers])
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(10, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(10, activation="relu")(x)
x = Dropout(0.2)(x)
final_layer = Dense(1, activation="sigmoid")(x)

model = Model(inputs=[c[0] for c in cat_layers], outputs=[final_layer])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=[auc])


# In[ ]:


Xs_train = [c[train_idx] for c in cat_vectors]
Xs_test = [c[test_idx] for c in cat_vectors]


# In[ ]:


hist = model.fit(Xs_train, y_train, validation_data=(Xs_test, y_test),
          batch_size=100, epochs=3, shuffle=True)


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(hist.history['auc'])
plt.plot(hist.history['val_auc'])
plt.title('model ROC AUC')
plt.ylabel('AUC')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


y_pred = model.predict(Xs_test)
K.clear_session()


# In[ ]:


from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr)
plt.xlabel('False positive')
plt.ylabel('True positive')
plt.title('ROC auc='+str(auc(fpr, tpr)))
plt.show()


# In[ ]:




