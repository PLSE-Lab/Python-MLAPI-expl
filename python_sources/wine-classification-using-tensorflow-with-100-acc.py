#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import pandas as pd
import numpy as np
import keras.utils as ku
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout


# In[ ]:


df = pd.read_csv('../input/winecsv/Wine.csv')

df.isnull().sum()
df.columns = [  'name'
                 ,'alcohol'
             	,'malicAcid'
             	,'ash'
            	,'ashalcalinity'
             	,'magnesium'
            	,'totalPhenols'
             	,'flavanoids'
             	,'nonFlavanoidPhenols'
             	,'proanthocyanins'
            	,'colorIntensity'
             	,'hue'
             	,'od280_od315'
             	,'proline'
                ]


# In[ ]:


df


# In[ ]:


import seaborn as sns
correlations = df[df.columns].corr(method='pearson')
sns.heatmap(correlations, cmap="YlGnBu", annot = True)


# In[ ]:


import heapq

print('Absolute overall correlations')
print('-' * 30)
correlations_abs_sum = correlations[correlations.columns].abs().sum()
print(correlations_abs_sum, '\n')

print('Weakest correlations')
print('-' * 30)
print(correlations_abs_sum.nsmallest(3))


# In[ ]:


df = df.drop(columns=['ash','magnesium', 'colorIntensity'], axis =1)


# In[ ]:


#Selecting dependent and independent variables
y = df.iloc[: ,0 ].values
X = df.iloc[:, 1:15].values
X[1]


# In[ ]:


df_hotencoded = pd.get_dummies(y)
df_hotencoded.head()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, df_hotencoded, test_size = 0.20)
y_test.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


import tensorflow as tf


# In[ ]:


ann = tf.keras.models.Sequential()


# In[ ]:


ann.add(tf.keras.layers.Dense(units = 32, input_dim = 10, activation='relu'))
ann.add(tf.keras.layers.Dense(units = 64, activation='relu'))

ann.add(tf.keras.layers.Dense(units = 128, activation='relu'))
ann.add(tf.keras.layers.Dense(units = 3, activation='softmax'))


# In[ ]:





# In[ ]:


ann.compile(optimizer= 'adam',  loss = 'categorical_crossentropy', metrics= ['accuracy'])


# In[ ]:


ann.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32 , epochs=100 )


# In[ ]:


score = ann.evaluate(X_test, y_test)
score[1]*100

