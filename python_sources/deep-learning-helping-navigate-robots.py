#!/usr/bin/env python
# coding: utf-8

# <h1><center>Deep Learning Helping Navigate Robots</center></h1>
# <img src="https://storage.googleapis.com/kaggle-competitions/kaggle/13242/logos/thumb76_76.png?t=2019-03-12-23-33-31" width="300"></img>
# 
# ### Dependencies

# In[ ]:


import warnings
import cufflinks
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras import optimizers
from keras.layers import Dense
from keras.utils import to_categorical
from keras.models import Sequential, Model
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore")
cufflinks.go_offline(connected=True)

# Set seeds to make the experiment more reproducible.
from tensorflow import set_random_seed
from numpy.random import seed
set_random_seed(0)
seed(0)


# ### Load data

# In[ ]:


train = pd.read_csv('../input/X_train.csv')
labels = pd.read_csv('../input/y_train.csv')
test = pd.read_csv('../input/X_test.csv')

print('Train features shape', train.shape)
display(train.head())
print('Train labels shape', labels.shape)
display(labels.head())
print('Test shape', test.shape)
display(test.head())


# ### Join train features with labels

# In[ ]:


train = train.join(labels, on='series_id', rsuffix='_')
train.drop('series_id_', axis=1, inplace=True)
print(train.shape)
display(train.head())


# ### Plotly graphs may take a while to load.
# 
# # EDA
# 
# ## Surface distribution
# - Let's see what's the label distribution of our data

# In[ ]:


f, ax = plt.subplots(figsize=(12, 8))
ax = sns.countplot(y='surface', data=train, palette="rocket", order=reversed(train['surface'].value_counts().index))
ax.set_ylabel("Surface type")
plt.show()


# ### Surface distribution by "group_id"

# In[ ]:


group_df = train.groupby(['group_id', 'surface'])['surface'].agg({'surface':['count']}).reset_index()
group_df.columns = ['group_id', 'surface', 'count']
f, ax = plt.subplots(figsize=(18, 8))
ax = sns.barplot(x="group_id", y="count", data=group_df, palette="GnBu_d")

for index, row in group_df.iterrows():
    ax.text(row.name, row['count'], row['surface'], color='black', ha="center", rotation=60)
    
plt.show()


# ## Features distribution
# - Now would be a good idea to see how each other type of features behavior
# 
# ### Orientation distribution

# In[ ]:


orientation_features = ['orientation_X', 'orientation_Y', 'orientation_Z', 'orientation_W']
train[orientation_features].iplot(kind='histogram', bins=200, subplots=True, shape=(len(orientation_features), 1))
train[orientation_features].iplot(kind='histogram', barmode='overlay', bins=200)
train[orientation_features].iplot(kind='box')


# The interesting part here is that "orientation_Y" and "orientation_X" are far more spread than the other two.

# ### Angular velocity distribution

# In[ ]:


velocity_features = ['angular_velocity_X', 'angular_velocity_Y', 'angular_velocity_Z']
train[velocity_features].iplot(kind='histogram', bins=200, subplots=True, shape=(len(velocity_features), 1))
train[velocity_features].iplot(kind='histogram', barmode='overlay', bins=200)
train[velocity_features].iplot(kind='box')


# Here all the angular velocity features seem to be centered around 0, but "angular_velocity_Y" is less spread than the others.

# ### Linear acceleration distribution

# In[ ]:


acceleration_features = ['linear_acceleration_X', 'linear_acceleration_Y', 'linear_acceleration_Z']
train[acceleration_features].iplot(kind='histogram', bins=200, subplots=True, shape=(len(acceleration_features), 1))
train[acceleration_features].iplot(kind='histogram', barmode='overlay', bins=200)
train[acceleration_features].iplot(kind='box')


# The linear acceleration features seem to be the most different between itself, all 3 features have different mean and spread.

# ### Preprocess the labels

# In[ ]:


target = train['surface']
n_labels = target.nunique()
labels_names = target.unique()
le = LabelEncoder()
target = le.fit_transform(target.values)
target = to_categorical(target)
train.drop('surface', axis=1, inplace=True)


# ### Train/validation split

# In[ ]:


features = ['orientation_X', 'orientation_Y', 'orientation_Z', 'orientation_W', 
            'angular_velocity_X', 'angular_velocity_Y', 'angular_velocity_Z', 
            'linear_acceleration_X', 'linear_acceleration_Y', 'linear_acceleration_Z']

X_train, X_val, Y_train, Y_val = train_test_split(train[features], target, test_size=0.2, random_state=0)
print('Train shape', X_train.shape)
print('Validation shape', X_val.shape)
display(X_train.head())


# ### Model

# In[ ]:


epochs = 70
batch = 128
lr = 0.001
adam = optimizers.Adam(lr)


# In[ ]:


model = Sequential()
model.add(Dense(20, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(20, activation='relu'))
model.add(Dense(n_labels, activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer=adam)
model.summary()


# In[ ]:


history = model.fit(X_train.values, Y_train, validation_data=(X_val.values, Y_val), epochs=epochs, verbose=2)


# #### Model loss plot

# In[ ]:


history_pd = pd.DataFrame.from_dict(history.history)
history_pd.iplot(kind='line')


# #### Model confusion matrix

# In[ ]:


cnf_matrix = confusion_matrix(np.argmax(Y_train, axis=1), model.predict_classes(X_train))
cnf_matrix_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
df_cm = pd.DataFrame(cnf_matrix_norm, index=labels_names, columns=labels_names)

plt.figure(figsize=(20, 7))
ax = plt.axes()
ax.set_title('Train')
sns.heatmap(df_cm, annot=True, fmt='.2f', cmap="Blues", ax=ax)
plt.show()

cnf_matrix = confusion_matrix(np.argmax(Y_val, axis=1), model.predict_classes(X_val))
cnf_matrix_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
df_cm = pd.DataFrame(cnf_matrix_norm, index=labels_names, columns=labels_names)

plt.figure(figsize=(20, 7))
ax = plt.axes()
ax.set_title('Validation')
sns.heatmap(df_cm, annot=True, fmt='.2f', cmap="Blues", ax=ax)
plt.show()


# ### Test predictions

# In[ ]:


predictions = model.predict_classes(test[features].values)
test['surface'] = le.inverse_transform(predictions)
df = test[['series_id', 'surface']]
df = df.groupby('series_id', as_index=False).agg(lambda x:x.value_counts().index[0])
df.to_csv('submission.csv', index=False)
df.head(10)

