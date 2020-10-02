#!/usr/bin/env python
# coding: utf-8

# ### We will use a Keras deep learning model to predict churn for bank customers.
# 
# ### First, let's set up our environment.

# In[ ]:


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import keras
from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


df = pd.read_csv('Churn_Modelling.csv')


# In[ ]:


df.head(3)


# In[ ]:


def create_enc(df, columns):
    '''
    The following will create encoded columns based on a list of columns as an argument. The original column
    will stay intact.  Ex: create_enc(df, ['sport', 'nationality'])
    '''
    for col in columns:
        df[col+'_enc'] = df[col]
        encoder = LabelEncoder()
        encoder.fit(df[col])
        df[col+'_enc'] = encoder.transform(df[col])
    return df
df = create_enc(df, ['Geography', 'Gender'])
df = df[['CreditScore', 'Geography_enc',
       'Gender_enc', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary', 'Exited']]


# In[ ]:


df.head(3)


# ### For X, we drop the first few columns that are not needed.  We also encoded Geograpy and Gender to be able used as features.  We'll also need to standardize a few columns to better be used in the Keras model.

# In[ ]:


X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


# In[ ]:


X.shape, y.shape


# In[ ]:


X[:5]


# In[ ]:


y[:5]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# ### We'll scale the columns so the model can perform better.  We'll scale on the train and use that object to scale the unseen/test set.

# In[ ]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


cf = Sequential()
cf.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=10))
cf.add(Dense(output_dim=6, init='uniform', activation='relu'))
cf.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))


# In[ ]:


cf.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


cf.fit(X_train, y_train, nb_epoch=50, batch_size=20)


# In[ ]:


scores = cf.evaluate(X, y)
print('%s: %.2f%%' % (cf.metrics_names[1], scores[1]*100))


# In[ ]:


y_pred = cf.predict(X_test)
y_pred = (y_pred > 0.5)


# In[ ]:


cm = confusion_matrix(y_test, y_pred)
cm


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




