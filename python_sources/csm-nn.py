#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


#fix random seed for reproducibility
np.random.seed(7)


# In[ ]:


#import dataset
dataframe = pd.read_csv('../input/meningitis_dataset.csv', delimiter=',')


# In[ ]:


dataframe.describe()


# In[ ]:


dataframe.info()


# In[ ]:


#non_categorical
#dataframe.dtypes.sample(50)
dataframe.dtypes.sample(20)


# In[ ]:


non_categorical = dataframe.select_dtypes(include='int64')


# In[ ]:


#rearrange noncategorical colums
#non_categorical = non_categorical[
non_categorical.head()
non_categorical.info()


# In[ ]:


#non_categorical = non_categorical['gender_male', 'gender_female', 'rural_settlement', 'urban_settlement', 'report_year', 'age', 'child_group', 'adult_group', 'cholra', 'diarrhoea', 'measles', 'viral_haemmorrhaphic_fever', 'ebola', 'marburg_virus', 'yellow_fever', 'rubella_mars', 'malaria', 'serotype', 'NmA', 'NmC', 'NmW', 'alive', 'dead', 'unconfirmed', 'confirmed', 'null_serotype', 'meningitis']


# In[ ]:


#split dataset into input(X) and output(Y) values
#X = non_categorical.iloc[:, 0,25]
#Y = non_categorical.iloc[:, 25]
#X = non_categorical.iloc[:, 1,26]
#Y = non_categorical.iloc[:, 26]
#X = non_categorical.ix[:, 1,26]
#Y = non_categorical.ix[:, 26]
#X = non_categorical.iloc[:, 1,26]
#X = non_catergorical['gender_male']
#X = non_categorical['gender_male','gender_female','rural_settlement','urban_settlement','report_year','age','child_group','adult_group','cholera','diarrhoea','measles']
X = non_categorical
Y = non_categorical.iloc[:, 26]


# In[ ]:


#create model
model = Sequential()
#model.add(Dense(12, input_dim=25, activation='relu'))
model.add(Dense(12, input_dim=27, activation='relu'))
#model.add(Dense(25, activation='relu'))
model.add(Dense(27, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[ ]:


#compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


#fit themodel
#the model
model.fit(X, Y, epochs=150, batch_size=10)


# In[ ]:


#evaluate the model
scores = model.evaluate(X,Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)


# In[ ]:


#calculate prediction
predictions = model.predict(X)


# In[ ]:


print(predictions)


# In[ ]:


#round predictions
rounded_predictions = [round(X[0]) for X in predictions]
print(rounded_predictions)

