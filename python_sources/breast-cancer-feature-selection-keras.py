#!/usr/bin/env python
# coding: utf-8

# Wisconsin Breast Cancer Dataset - 
# 
# Recursive Feature Elimination (RFE) with Scikit-Learn. 
# 
# Neural Network Classification with Keras.
# 
# Feedback welcome! Interested in adding visualization as well as grid-search for RFE. 

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


one = pd.read_csv('../input/data.csv')
one = one.drop(['id', 'Unnamed: 32'], axis=1)
one.head()


# In[ ]:


one['Target'] = np.where(one['diagnosis'] == 'M', 1, 0)
one.head()


# In[ ]:


one = one.drop(['diagnosis'], axis=1)


# In[ ]:


X = one.iloc[:, 0:30]
X.head()


# In[ ]:


Y = one.iloc[:, 30]
Y.head()


# In[ ]:


feature_n = 10


# In[ ]:


model = RandomForestClassifier()
rfe = RFE(model, feature_n)
fit = rfe.fit(X, Y)


# In[ ]:


X = X[X.columns[fit.support_]]
X.head()


# In[ ]:


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# load pima indians dataset
# split into input (X) and output (Y) variables
X = X.as_matrix()
Y = Y.as_matrix()
# create model
model = Sequential()
model.add(Dense(50, input_dim=feature_n, init='uniform', activation='relu'))
model.add(Dense(25, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, nb_epoch=500, batch_size=5)
# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[ ]:


predictions = model.predict(X)
rounded = [round(x[0]) for x in predictions]


# In[ ]:


one['predictions'] = rounded


# In[ ]:


one[['Target', 'predictions']].head(15)

