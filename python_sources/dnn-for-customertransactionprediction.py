#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

# Visualization
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
sns.set_style("darkgrid")


# In[ ]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# In[ ]:


train_data.head()


# # Understanding the dataset

# In[ ]:


print(train_data.target.value_counts())


# In[ ]:


print(train_data.shape, test_data.shape)


# In[ ]:


train_data.describe()


# In[ ]:


# Lets see the std distribution of the data
sns.distplot(train_data[train_data.columns[2:]].std(), bins=30)
plt.title('Stds distribution of all columns');


# In[ ]:


# Lets see the mean distribution of the data
sns.distplot(train_data[train_data.columns[2:]].mean(), bins=30)
plt.title('Mean distribution of all columns');


# In[ ]:


# Check wether we have missing values
train_data.isnull().any().any()


# In[ ]:


test_data.isnull().any().any()


# #### Plot the first 50 visualizations for feature distribution in space.

# In[ ]:


sns.set(rc={'figure.figsize':(10,7)})
colours = ["goldenrod","purple","darkgreen","maroon","aqua","olive","coral","darkorchid","darkviolet","saddlebrown"]
index = -1
for i in train_data.columns[2:12]:
    index = index + 1
    fig = sns.kdeplot(train_data[i] , shade=True, color=colours[index])
plt.xlabel("Features")
plt.ylabel("Density")
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig)


# In[ ]:


sns.set(rc={'figure.figsize':(10,7)})
colours = ["goldenrod","purple","darkgreen","maroon","aqua","olive","coral","darkorchid","darkviolet","saddlebrown"]
index = -1
for i in train_data.columns[12:22]:
    index = index + 1
    fig = sns.kdeplot(train_data[i] , shade=True, color=colours[index])
plt.xlabel("Features")
plt.ylabel("Density")
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig)


# In[ ]:


sns.set(rc={'figure.figsize':(10,7)})
colours = ["goldenrod","purple","darkgreen","maroon","aqua","olive","coral","darkorchid","darkviolet","saddlebrown"]
index = -1
for i in train_data.columns[22:32]:
    index = index + 1
    fig = sns.kdeplot(train_data[i] , shade=True, color=colours[index])
plt.xlabel("Features")
plt.ylabel("Density")
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig)


# In[ ]:


sns.set(rc={'figure.figsize':(10,7)})
colours = ["goldenrod","purple","darkgreen","maroon","aqua","olive","coral","darkorchid","darkviolet","saddlebrown"]
index = -1
for i in train_data.columns[32:42]:
    index = index + 1
    fig = sns.kdeplot(train_data[i] , shade=True, color=colours[index])
plt.xlabel("Features")
plt.ylabel("Density")
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig)


# In[ ]:


sns.set(rc={'figure.figsize':(10,7)})
colours = ["goldenrod","purple","darkgreen","maroon","aqua","olive","coral","darkorchid","darkviolet","saddlebrown"]
index = -1
for i in train_data.columns[42:52]:
    index = index + 1
    fig = sns.kdeplot(train_data[i] , shade=True, color=colours[index])
plt.xlabel("Features")
plt.ylabel("Density")
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig)


# ### Joint distrubutions of some variable

# In[ ]:


sns.jointplot(data=train_data, x='var_0', y='var_1', kind='hex')


# In[ ]:


print('Distributions of second 20 columns after the first 50')
plt.figure(figsize=(28, 26))
for i, col in enumerate(list(train_data.columns)[52:72]):
    plt.subplot(5, 4, i + 1)
    sns.distplot(train_data[col])
    plt.title(col)


# In[ ]:


# Since is a binary classifacation, lets check for balance in the train dataset
train_data['target'].value_counts(normalize=True)


# In[ ]:


# Check for correlations
data_cor = train_data.corr().abs().unstack().sort_values(kind="quicksort").reset_index()
data_cor = data_cor[data_cor['level_0'] != data_cor['level_1']]


# In[ ]:


data_cor.head(10)


# In[ ]:


data_cor.tail(10)


# ### All features have a low correlation with target, hence no dealing with highly correlated features.

# # Part 2. Data Preprocessing

# #### Pre-processing and data preparation to feed Network.

# In[ ]:


trian_X = train_data.drop(['ID_code', 'target'], axis = 1)
train_y = train_data['target']


# In[ ]:


print (trian_X.shape, train_y.shape)


# In[ ]:


test_X = test_data.drop(['ID_code'], axis = 1)
id_test = test_data['ID_code']


# In[ ]:


print (test_X.shape, id_test.shape)


# In[ ]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
trian_X = sc.fit_transform(trian_X)
test_X = sc.fit_transform(test_X)


# In[ ]:


# Splitting the data into training set and test set
from sklearn.model_selection import train_test_split
X_train, dev_X, Y_train, dev_Y = train_test_split(trian_X, train_y, test_size=0.30, random_state=101)


# # Part 3. Building machine learning model

# In[ ]:


# import the Keras libraries and packages
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.optimizers import Adam
from keras import regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score


# In[ ]:


# Initialize the ANN
model = Sequential()


# In[ ]:


model.add(Dense(64, input_dim=X_train.shape[1] , activation='relu',kernel_regularizer=regularizers.l1_l2(0.001)))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(196, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# ###The  function auroc prints the roc auc score  as part of the metric to judge the performance of your model.

# In[ ]:


def auroc(dev_Y, y_score):
    return tf.py_func(roc_auc_score, (dev_Y, y_score), tf.double)


# In[ ]:


metrics_list = ['accuracy', auroc]


# In[ ]:


model.compile(loss='binary_crossentropy', optimizer=Adam(lr = 0.01, decay=0.01/50), metrics=metrics_list)


# In[ ]:


model.summary()


# In[ ]:


# define learning rate schedule
rlrp = ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=0.0001, patience=5, verbose=1)

# patient early stopping
# stop when the validation loss has not improved for 10 training epochs.
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

callbacks_list = [rlrp,es]


# In[ ]:


# fit model
history = model.fit(X_train, Y_train, batch_size = 25800, epochs=50, validation_data=(dev_X, dev_Y), callbacks=callbacks_list)


# ### How training and validation tries to mimic each other on Training loss and accuracy  pass epoch 7 and stagnation of the model past epoch 10 shows that the model didnt suffer overfitting. 
# 

# In[ ]:


# Visualise report to check for overfitting
from pylab import rcParams
rcParams['figure.figsize'] = 10, 4
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


# Predicting the test set results
y_score = model.predict_proba(dev_X)
y_pred = model.predict(test_X)
# y_pred = np.argmax(y_pred, axis = 1) 


# In[ ]:


threshold = 0.5
# y_pred_ = (y_pred > threshold)
y_pred = (y_pred > threshold).astype(int)


# In[ ]:


# calculate AUC
auc = roc_auc_score(dev_Y, y_score)
print('AUC: %.2f' % auc)


# # Submission

# Our solution is successfully submitted.

# In[ ]:


pd.DataFrame({"ID_code":id_test,"target":y_pred[:,0]}).to_csv('Customer_Transaction.csv',
                                                                                     index=False,header=True)


# In[ ]:




