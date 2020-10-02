#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

import math

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/HR_comma_sep.csv')
len(df)


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


df.corr()


# In[ ]:


df.info()


# In[ ]:


df['sales'].unique()


# In[ ]:


df['salary'].unique()


# In[ ]:


y = df['left'].values
x = df.drop('left', axis=1).values


# In[ ]:


le1 = LabelEncoder()
x[:, 7] = le1.fit_transform(x[:, 7])

le2 = LabelEncoder()
x[:, 8] = le2.fit_transform(x[:, 8])

ohe1 = OneHotEncoder(categorical_features = [7, 8])
x = ohe1.fit_transform(x).toarray()


# In[ ]:


sc_x = StandardScaler()
x_std = sc_x.fit_transform(x)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
print('%s , %s , %s , %s' % (len(x_train), len(y_train), len(x_test), len(y_test)))


# In[ ]:


tsne = TSNE(perplexity=40)
x_tsne = tsne.fit_transform(x_std)


# In[ ]:


color_map = {0:'red', 1:'blue'}
plt.figure()

for idx, c1 in enumerate(np.unique(y)):
    plt.scatter(x = x_tsne[y == c1, 0],                y = x_tsne[y == c1, 1],                 c = color_map[idx],                 label = c1)
plt.show()


# In[ ]:


neuralnetwork = Sequential()

neuralnetwork.add(Dense(units=10, kernel_initializer='uniform', activation='relu', input_dim=20))

neuralnetwork.add(Dense(units=10, kernel_initializer='uniform', activation='relu'))

neuralnetwork.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

neuralnetwork.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

neuralnetwork.fit(x_train, y_train, batch_size=10, epochs=50)

# neuralnetwork = load_model('hr_neuralnetwork.h5')


# In[ ]:


y_pred_temp = neuralnetwork.predict(x_test)


# In[ ]:


# for i in y_pred_temp:
#     print('%s -> %s' % (i[0], round(i[0])))
    
y_pred = [round(i[0]) for i in y_pred_temp]


# In[ ]:


from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)


# In[ ]:


neuralnetwork.save('hr_neuralnetwork.h5')


# In[ ]:




