#!/usr/bin/env python
# coding: utf-8

# This kernel uses 1D convolutions on signals from power lines to identify partial faults

# In[ ]:


import pandas as pd
import pyarrow.parquet as pq
import os

os.listdir('../input')


# Read the parquet file. The full length of each signal is 800000. We will halve it to 400000 readings to create the pipeline.

# In[ ]:


subset_train = pq.read_pandas('../input/train.parquet').to_pandas() #, columns=[str(i) for i in range(10)]).to_pandas()


# In[ ]:


subset_train = subset_train.iloc[:400000,:]
subset_train.info()


# Now read the metadata file.

# In[ ]:


metadata_train = pd.read_csv('../input/metadata_train.csv')
metadata_train.info()


# In[ ]:


metadata_train.head()


# Import plotting libraries and create some basic plots.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#plt.hist(metadata_train['target'])
sns.countplot(metadata_train['target'])


# This is a plot of the target values. As expected, a faulty power line is a kind of rare event. Let's visualize some negative and positive (faulty) signals.

# In[ ]:


fig = plt.figure(figsize=(10,8))

plt.subplot(431)
plt.plot(subset_train['0'])
plt.subplot(432)
plt.boxplot(subset_train['0'])
plt.subplot(433)
plt.hist(subset_train['0'])
    
plt.subplot(434)
plt.plot(subset_train['1'])
plt.subplot(435)
plt.boxplot(subset_train['1'])
plt.subplot(436)
plt.hist(subset_train['1'])

plt.subplot(437)
plt.plot(subset_train['3'])
plt.subplot(438)
plt.boxplot(subset_train['3'])
plt.subplot(439)
plt.hist(subset_train['3'])

plt.subplot(4,3,10)
plt.plot(subset_train['4'])
plt.subplot(4,3,11)
plt.boxplot(subset_train['4'])
plt.subplot(4,3,12)
plt.hist(subset_train['4'])


# At least from these couple of plots, we notice that the faulty signals (last 2) have relatively more outliers than the non-faulty ones. We will analyze this further with more data.
# 
# Let's separate the positive and negative signals for further analysis. I'm going to reduce the sample sizes to make sure we don't run out of memory limits.

# In[ ]:


import numpy as np

# Temporarily reduce data size to build the pipeline
small_subset_train = subset_train.iloc[:25000,:]
small_subset_train = small_subset_train.transpose()
small_subset_train.index = small_subset_train.index.astype(np.int32)
train_dataset = metadata_train.join(small_subset_train, how='right')

# Uncomment the following to train on the full dataset
#subset_train = subset_train.transpose()
#subset_train.index = subset_train.index.astype(np.int32)
#train_dataset = metadata_train.join(subset_train, how='right')


# In[ ]:


positive_samples = train_dataset[train_dataset['target']==1]
positive_samples = positive_samples.iloc[:,3:]
positive_samples.info()


# In[ ]:


positive_samples.head()


# Now let's visualize the positive (faulty) signals using a boxplot for several of them.

# In[ ]:


plt.figure(figsize=(10,4))
plt.subplot(151)
plt.boxplot(positive_samples.iloc[3,1:])
plt.subplot(152)
plt.boxplot(positive_samples.iloc[4,1:])
plt.subplot(153)
plt.boxplot(positive_samples.iloc[5,1:])
plt.subplot(154)
plt.boxplot(positive_samples.iloc[201,1:])
plt.subplot(155)
plt.boxplot(positive_samples.iloc[202,1:])


# We see that the data values differ a lot. Let's normalize the data first, this will also be needed for training some type of models later.

# In[ ]:


# Normalize the data set
from sklearn.preprocessing import StandardScaler
y_train_pos = positive_samples.iloc[:, 0]
X_train_pos = positive_samples.iloc[:, 1:]
scaler = StandardScaler()
scaler.fit(X_train_pos.T)
X_train_pos = scaler.transform(X_train_pos.T).T


# Let's visualize the boxplots again using this normalized data.

# In[ ]:


plt.figure(figsize=(10,4))
plt.subplot(151)
plt.boxplot(X_train_pos[0,:])
plt.subplot(152)
plt.boxplot(X_train_pos[1,:])
plt.subplot(153)
plt.boxplot(X_train_pos[2,:])
plt.subplot(154)
plt.boxplot(X_train_pos[3,:])
plt.subplot(155)
plt.boxplot(X_train_pos[4,:])


# Again we notice that there are a lot of outliers in the positive (faulty) signals.
# 
# Now let's extract the negative (non-faulty) samples and visualize the same boxplots, and see if we can notice any apparent difference.

# In[ ]:


negative_samples = train_dataset[train_dataset['target']==0]
negative_samples = negative_samples.iloc[:,3:]
negative_samples.info(), negative_samples.head()

y_train_neg = negative_samples.iloc[:, 0]
X_train_neg = negative_samples.iloc[:, 1:]
scaler.fit(X_train_neg.T)
X_train_neg = scaler.transform(X_train_neg.T).T

plt.figure(figsize=(10,4))
plt.subplot(151)
plt.boxplot(X_train_neg[0,:])
plt.subplot(152)
plt.boxplot(X_train_neg[1,:])
plt.subplot(153)
plt.boxplot(X_train_neg[2,:])
plt.subplot(154)
plt.boxplot(X_train_neg[3,:])
plt.subplot(155)
plt.boxplot(X_train_neg[4,:])


# The negative (non-faulty) signals have much fewer outliers, and their magnitudes also seem to be very low. Seems like the number of outliers could be a promising feature.
# 
# Now let's create the test/train split for training a Conv1D model.

# In[ ]:


from sklearn.model_selection import train_test_split

X_train_pos, X_valid_pos, y_train_pos, y_valid_pos = train_test_split(X_train_pos, y_train_pos, 
                                                                    test_size=0.2,
                                                                    random_state = 0,
                                                                    shuffle=True)

X_train_neg, X_valid_neg, y_train_neg, y_valid_neg = train_test_split(X_train_neg, y_train_neg, 
                                                                    test_size=0.2,
                                                                    random_state = 0,
                                                                    shuffle=True)


# In[ ]:


X_train_pos.shape, X_train_neg.shape


# As we know, the positive samples are fewer, so we will only select a subset of negative samples for training.

# In[ ]:


# Combine positive and negative samples for training...
def combine_positive_and_negative_samples(pos_samples, neg_samples, y_pos, y_neg):
    X_combined = np.concatenate((pos_samples, neg_samples)) 
                                                    # don't select all negative samples, to
                                                    # keep the samples balanced
    y_combined = np.concatenate((y_pos, y_neg))
    #X_train_combined.shape, y_train_combined.shape
    combined_samples = np.hstack((X_combined, y_combined.reshape(y_combined.shape[0],1)))
    np.random.shuffle(combined_samples)
    return combined_samples

# Only use 500 negative samples, to create a balanced dataset with the positive samples...
train_samples = combine_positive_and_negative_samples(X_train_pos, X_train_neg[:500, :], y_train_pos, y_train_neg[:500])
X_train = train_samples[:,:-1]
y_train = train_samples[:,-1]
X_train.shape, y_train.shape


# In[ ]:


# Create the validation set
#X_valid_combined = np.concatenate((X_valid_pos, X_valid_neg[:500,:])) # don't select all negative samples, to
                                                  # keep the samples balanced
#y_valid_combined = np.concatenate((y_valid_pos, y_valid_neg[:500]))
#X_valid_combined.shape, y_valid_combined.shape
#validation_samples = np.hstack((X_valid_combined, y_valid_combined.reshape(y_valid_combined.shape[0],1)))
#np.random.shuffle(validation_samples)

validation_samples = combine_positive_and_negative_samples(X_valid_pos, X_valid_neg[:500,:], y_valid_pos, y_valid_neg[:500])
X_valid = validation_samples[:,:-1]
y_valid = validation_samples[:,-1]
X_valid.shape, y_valid.shape


# A 1-D ConvNet would be an interesting model to try out on this signal. Earlier we saw that there are a lot of outliers in fauty signals. Since the actual signal value differs at different times, the outliers are relative to this mean signal value. A 1-D ConvNet can analyze the signal in various windows of increasing lengths and create high-level features out of that to classify on.

# In[ ]:


# Reshape training and validation data for keras input layer
X_train = X_train.reshape(-1, X_train.shape[1], 1)
X_valid = X_valid.reshape(-1, X_valid.shape[1], 1)

X_train.shape, X_valid.shape


# In[ ]:


import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import Input, layers
from tensorflow.keras import backend as K

#Conv1D Model
drop_out_rate = 0.2
lr = 0.0005
input_tensor = Input(shape=([X_train_pos.shape[1],1]))

x = layers.Conv1D(8, 11, padding='valid', activation='relu', strides=1)(input_tensor)
x = layers.MaxPooling1D(2)(x)
x = layers.Dropout(drop_out_rate)(x)
x = layers.Conv1D(16, 7, padding='valid', activation='relu', strides=1)(x)
x = layers.MaxPooling1D(2)(x)
x = layers.Dropout(drop_out_rate)(x)
x = layers.Conv1D(32, 5, padding='valid', activation='relu', strides=1)(x)
x = layers.MaxPooling1D(2)(x)
x = layers.Dropout(drop_out_rate)(x)
x = layers.Conv1D(64, 5, padding='valid', activation='relu', strides=1)(x)
x = layers.MaxPooling1D(2)(x)
x = layers.Dropout(drop_out_rate)(x)
x = layers.Conv1D(128, 3, padding='valid', activation='relu', strides=1)(x)
x = layers.MaxPooling1D(2)(x)
x = layers.Flatten()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(drop_out_rate)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(drop_out_rate)(x)
output_tensor = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(input_tensor, output_tensor)

model.compile(loss=keras.losses.binary_crossentropy,
             optimizer=keras.optimizers.Adam(lr = lr),
             metrics=['accuracy'])

model.summary()


# In[ ]:


from keras.callbacks import ModelCheckpoint

weights_file="best_weights.hdf5"
checkpoint = ModelCheckpoint(weights_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks = [checkpoint]


# In[ ]:


batch_size = 100
history = model.fit(X_train, y_train, validation_data=[X_valid, y_valid],
          batch_size=batch_size, 
          epochs=25,
          verbose=1, callbacks=callbacks)


# In[ ]:


# plot history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# plot history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


os.listdir('./')


# In[ ]:


model.load_weights("best_weights.hdf5")


# In[ ]:


from sklearn.metrics import roc_curve
from sklearn.metrics import auc

y_pred = model.predict(X_valid).ravel()
fpr, tpr, thresholds = roc_curve(y_valid, y_pred)
auc = auc(fpr, tpr)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Conv1D (area = {:.3f})'.format(auc))
#plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
# Zoom in view of the upper left corner.
plt.figure(2)
plt.xlim(0, 0.5)
plt.ylim(0.6, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Conv1D (area = {:.3f})'.format(auc))
#plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
plt.show()

