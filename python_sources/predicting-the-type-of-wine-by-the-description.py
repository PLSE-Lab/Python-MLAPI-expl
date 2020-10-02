#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
sw = stopwords.words('english')


# In[ ]:


dataset = pd.read_csv('../input/winemag-data_first150k.csv')


# In[ ]:


dataset.head()


# In[ ]:


dataset.info()


# In[ ]:


dataset.dropna(axis=0)


# In[ ]:


input_data = dataset['description']
output_data = dataset['variety']


# How many varieties of wine do we have in this dataset?

# In[ ]:


print ('There are %d varieties of wines in this dataset' % len(set(output_data)))


# Since we have 632 varieties of wines in the dataset, the enconder is going from 0 to 631 

# In[ ]:


labelEncoder = LabelEncoder()
output_data = labelEncoder.fit_transform(output_data)
output_data


# Now it is time to clean the description data

# In[ ]:


input_data = input_data.str.lower()


# In[ ]:


list_aux = []
for phase_word in input_data:
    list_aux.append(' '.join([re.sub('[0-9\W_]', '', word) for word in phase_word.split() if not word in sw]))
input_data = list_aux


# In[ ]:


countVectorizer = CountVectorizer()
input_data = countVectorizer.fit_transform(input_data)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2) 


# In[ ]:


model = Sequential()
model.add(Dense(100, activation='relu', input_dim=len(countVectorizer.get_feature_names())))
model.add(Dense(units=output_data.max()+1, activation='sigmoid'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=2, verbose=1)


# In[ ]:


scores = model.evaluate(X_test, y_test, verbose=1)
print ('The accuracy of the model is %s' % scores[1])

