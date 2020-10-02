#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/big-five-personality-test/IPIP-FFM-data-8Nov2018'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **Analysing AGR4 based on response latency**
# 
# In this notebook, the goal is to create a ANN model, that predicts how much one sympathizes with other's feelings based on the response latency of all 50 statements. So based on how much you think of choosing a number from 1 to 5 for each statement, you can predict different psychological characteristics.
# 
# For this analysis I used only 60000 samples from the dataset, because of my CPU limitations. First we need to clean the rows that hold any NaN values. Also I removed the rows that had a 0 value in the **y** column, because the values should only range from 1 to 5.

# In[ ]:


import pandas as pd
import csv
import numpy as np

df = pd.read_csv('/kaggle/input/big-five-personality-test/IPIP-FFM-data-8Nov2018/data-final.csv',sep='\t')
df =df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
df.dropna(axis = 0, how= 'any')

df = df[~(df == 0).any(axis=1)]

x = df.iloc[0:60000,:].values

x_str = str(x)
x_str = x_str.replace('[','')
x_str = x_str.replace(']','')

count = len(x_str.split('\n'))
a_split = np.array_split(x, count)

with open('/kaggle/working/OCEAN_60000.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    for i in a_split:
         writer.writerows(i)


# Here we take all the data samples for response latency for each statement, so it is 50 columns in total. The column that we are trying to predict is the 24th, which stands for AGR4 "I sympathize with others' feelings". 
# 
# Since this is a multiclass analysis we have to keep in mind that for ANN we have to use Softmax activation layer, but first we need to preprocess the dataset. We have to use One Hot Encoding on the y column before we add it to the train & test split.
# 
# After splitting the data for training and testing, we scale the data and than we get into the ANN model. I have used two hidden layers in this case, with dropout at 0.1 because we don't want our model to overfit.
# 
# In this case I have used a batch size of 150 and 100 epochs to train the model. The accuracy of the model is at 80.5 % with a test score at 0.42. 
# 
# Ofcourse this is far from the ideal accuracy, but I uploaded this notebook to give an analysis idea on this interesting dataset. It can further improve with more epochs and with the full dataset.

# In[ ]:


# Importing data
data = pd.read_csv('/kaggle/working/OCEAN_60000.csv')

X = data.iloc[:, 49:99].values
y = data.iloc[:, 23].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lable_y = LabelEncoder()
y = lable_y.fit_transform(y)

onehot = OneHotEncoder()
y = onehot.fit_transform(y.reshape(-1,1)).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim=60, init='uniform', activation='relu', input_dim=50))
classifier.add(Dropout(p = 0.1))

classifier.add(Dense(output_dim=30, init='uniform', activation='relu'))
classifier.add(Dropout(p = 0.1))

classifier.add(Dense(output_dim=5, init='uniform', activation='softmax'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=150, nb_epoch=100)

score = classifier.evaluate(X_test, y_test, batch_size = 150)
print("Test score: ", score[0])
print("Test accuracy: ", score[1]*100, "%")

