#!/usr/bin/env python
# coding: utf-8

# A neural network to predict whether a patient has diabetes.

# ## Import the necessary libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import Normalizer
from keras.layers import Activation, Dense, Dropout, BatchNormalization, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')


# ## Load the dataset into a Pandas dataframe

# In[3]:


diabetes_df = pd.read_csv('../input/diabetes.csv', delimiter=',')
diabetes_df.head()


# In[4]:


diabetes_df.info()


# * The dataset does not have any missing data.

# In[5]:


diabetes_df.describe()


# In[6]:


diabetes_df.corr()


# ## Basic Visualization

# In[7]:


f, ax = plt.subplots(1, figsize=(10,8))
sns.heatmap(diabetes_df.corr(), annot=True, ax=ax)


# * Glucose levels had the highest effect on the outcome.
# * As expected, pregnancies were correlated to age.

# In[8]:


sns.countplot(x=diabetes_df.Outcome)


# In[9]:


f, axes = plt.subplots(4,2, figsize=(20,20))
sns.violinplot(x=diabetes_df.Outcome ,y=diabetes_df.Pregnancies, ax=axes[0,0])
sns.violinplot(x=diabetes_df.Outcome ,y=diabetes_df.Glucose, ax=axes[0,1])
sns.violinplot(x=diabetes_df.Outcome ,y=diabetes_df.BloodPressure, ax=axes[1,0])
sns.violinplot(x=diabetes_df.Outcome ,y=diabetes_df.SkinThickness, ax=axes[1,1])
sns.violinplot(x=diabetes_df.Outcome ,y=diabetes_df.Insulin, ax=axes[2,0])
sns.violinplot(x=diabetes_df.Outcome ,y=diabetes_df.BMI, ax=axes[2,1])
sns.violinplot(x=diabetes_df.Outcome ,y=diabetes_df.DiabetesPedigreeFunction, ax=axes[3,0])
sns.violinplot(x=diabetes_df.Outcome ,y=diabetes_df.Age, ax=axes[3,1])


# In[10]:


column_names = diabetes_df.columns
column_names = column_names.drop('Outcome')
for name in column_names:
    print('{}\n'.format(name))
    print(diabetes_df.groupby(['Outcome'])[name].mean())
    print('*'*50)
    print()


# * In the dataset, people diagnosed with diabetes had higher values for every attribute (mean value).

# In[11]:


f, axes = plt.subplots(4,2, figsize=(20,20))
sns.distplot(diabetes_df.Pregnancies, ax=axes[0,0])
sns.distplot(diabetes_df.Glucose, ax=axes[0,1])
sns.distplot(diabetes_df.BloodPressure, ax=axes[1,0])
sns.distplot(diabetes_df.SkinThickness, ax=axes[1,1])
sns.distplot(diabetes_df.Insulin, ax=axes[2,0])
sns.distplot(diabetes_df.BMI, ax=axes[2,1])
sns.distplot(diabetes_df.DiabetesPedigreeFunction, ax=axes[3,0])
sns.distplot(diabetes_df.Age, ax=axes[3,1])


# * *Skin Thickness, Insulin, Blood Pressure, Glucose and BMI* had no NaNs but some of the data was encoded as 0s.
# * This data was replaced using the median value.

# In[12]:


diabetes_df.SkinThickness.replace(0, diabetes_df.SkinThickness.median(), inplace=True)
diabetes_df.Insulin.replace(0, diabetes_df.Insulin.median(), inplace=True)
diabetes_df.Glucose.replace(0, diabetes_df.Glucose.median(), inplace=True)
diabetes_df.BloodPressure.replace(0, diabetes_df.BloodPressure.median(), inplace=True)
diabetes_df.BMI.replace(0, diabetes_df.BMI.median(), inplace=True)


# In[13]:


f, axes = plt.subplots(4,2, figsize=(20,20))
sns.distplot(diabetes_df.Pregnancies, ax=axes[0,0])
sns.distplot(diabetes_df.Glucose, ax=axes[0,1])
sns.distplot(diabetes_df.BloodPressure, ax=axes[1,0])
sns.distplot(diabetes_df.SkinThickness, ax=axes[1,1])
sns.distplot(diabetes_df.Insulin, ax=axes[2,0])
sns.distplot(diabetes_df.BMI, ax=axes[2,1])
sns.distplot(diabetes_df.DiabetesPedigreeFunction, ax=axes[3,0])
sns.distplot(diabetes_df.Age, ax=axes[3,1])


# ## Neural Network

# ### Preparation of the data
# * Split the data into a training set, dev set and test set.
# * Normalize the data.

# In[14]:


X = diabetes_df.drop('Outcome', axis =1).values
y = diabetes_df.Outcome.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
nl = Normalizer()
nl.fit(X_train)
X_train = nl.transform(X_train)
X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=2)
X_dev = nl.transform(X_dev)
X_test = nl.transform(X_test)


# ### Network arhitecture
# * 3 hidden layers.
# * Output layer with sigmoid activation.

# In[15]:


def nn():
    inputs = Input(name='inputs', shape=[X_train.shape[1],])
    layer = Dense(128, name='FC1')(inputs)
    layer = BatchNormalization(name='BC1')(layer)
    layer = Activation('relu', name='Activation1')(layer)
    layer = Dropout(0.3, name='Dropout1')(layer)
    layer = Dense(128, name='FC2')(layer)
    layer = BatchNormalization(name='BC2')(layer)
    layer = Activation('relu', name='Activation2')(layer)
    layer = Dropout(0.3, name='Dropout2')(layer)
    layer = Dense(128, name='FC3')(layer)
    layer = BatchNormalization(name='BC3')(layer)
    layer = Dropout(0.3, name='Dropout3')(layer)
    layer = Dense(1, name='OutLayer')(layer)
    layer = Activation('sigmoid', name='sigmoid')(layer)
    model = Model(inputs=inputs, outputs=layer)
    return model


# In[16]:


model = nn()
model.summary()


# ### Compile and fit the model

# In[17]:


model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])


# Define callbacks.

# In[18]:


reduce_lr = ReduceLROnPlateau()
early_stopping = EarlyStopping(patience=20, min_delta=0.0001)


# In[19]:


model.fit(x=X_train, y=y_train, epochs=200, validation_data=(X_dev, y_dev), callbacks=[reduce_lr, early_stopping], verbose=0)


# ### Metrics

# In[21]:


x_lst = [X_train, X_dev, X_test]
y_lst = [y_train, y_dev, y_test]
for i,(x,y) in enumerate(zip(x_lst, y_lst)):
    y_pred = model.predict(x)
    y_pred = np.around(y_pred)
    y_pred = np.asarray(y_pred)
    if i == 0:
        print('Training set:')
        print('\tAccuracy:{:0.3f}\n\tClassification Report\n{}'.format(accuracy_score(y, y_pred), 
                                                                  classification_report(y, y_pred)))
    elif i == 1:
        print('Dev set:')
        print('\tAccuracy:{:0.3f}\n\tClassification Report\n{}'.format(accuracy_score(y, y_pred), 
                                                                  classification_report(y, y_pred)))
    else:
        print('Test set:')
        print('\tAccuracy:{:0.3f}\n\tClassification Report\n{}'.format(accuracy_score(y, y_pred), 
                                                                  classification_report(y, y_pred)))
    


# * The classifier showed poor performance while predicting the positive cases in both validation and test set.
# * A different architecture or a completely different class of algorithms could be used to gain a performance boost.
