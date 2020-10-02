#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/land classification challenge/socialcops_challenge"))

# Any results you write to the current directory are saved as output.


# **Importing the test data and the train data**

# In[ ]:


dataset = pd.read_csv('../input/land classification challenge/socialcops_challenge/land_train.csv')
test = pd.read_csv('../input/land classification challenge/socialcops_challenge/land_test.csv')


# identifying the training features and the dependent variable (target)

# In[ ]:


print(dataset.columns)
print(test.columns)


# **First of all checking the dataset for missing values**

# In[ ]:


dataset.isna().sum()


# In[ ]:


test.isna().sum()


# GREAT BOTH THE TRAINING AND THE TESTING DATA ARE FREE FREOM MISSING VALUES :)

# **AS WE HAVE 4 POSSIBLE VALUES FOR THE 'TARGET' VARIABLE, HENCE CHECKING WHETHER THE DATA CORRESPONDING TO THESE 4 CATEGORIES IS BALANCED OR NOT **

# In[ ]:


dataset['target'].hist()


# As we can see with the above histogram, the data is imbalanced. Therefore training on this data may lead to results that are biased towards the categories that have higher occurence in training data

# **ANALYZING THE X1 - X6 FEATURES AGAINST THE TARGET VARIABLE**

# In[ ]:


print(dataset[['target' , 'X1']].groupby(['target']).mean())
plt.plot(dataset[['target' , 'X1']].groupby(['target']).mean())
sns.catplot(x='target', y='X1',  kind='bar', data=dataset)


# In[ ]:


print(dataset[['target' , 'X2']].groupby(['target']).mean())
plt.plot(dataset[['target' , 'X2']].groupby(['target']).mean())
sns.catplot(x='target', y='X2',  kind='bar', data=dataset)


# In[ ]:


print(dataset[['target' , 'X3']].groupby(['target']).mean())
plt.plot(dataset[['target' , 'X3']].groupby(['target']).mean())
sns.catplot(x='target', y='X3',  kind='bar', data=dataset)


# In[ ]:


print(dataset[['target' , 'X4']].groupby(['target']).mean())
plt.plot(dataset[['target' , 'X4']].groupby(['target']).mean())
sns.catplot(x='target', y='X4',  kind='bar', data=dataset)


# In[ ]:


print(dataset[['target' , 'X5']].groupby(['target']).mean())
plt.plot(dataset[['target' , 'X5']].groupby(['target']).mean())
sns.catplot(x='target', y='X5',  kind='bar', data=dataset)


# In[ ]:


print(dataset[['target' , 'X6']].groupby(['target']).mean())
plt.plot(dataset[['target' , 'X6']].groupby(['target']).mean())
sns.catplot(x='target', y='X6',  kind='bar', data=dataset)


# In[ ]:


print(dataset[['target' , 'I1']].groupby(['target']).mean())
plt.plot(dataset[['target' , 'I1']].groupby(['target']).mean())
sns.catplot(x='target', y='I1',  kind='bar', data=dataset)


# In[ ]:


print(dataset[['target' , 'I2']].groupby(['target']).mean())
plt.plot(dataset[['target' , 'I2']].groupby(['target']).mean())
sns.catplot(x='target', y='I2',  kind='bar', data=dataset)


# In[ ]:


print(dataset[['target' , 'I3']].groupby(['target']).mean())
plt.plot(dataset[['target' , 'I3']].groupby(['target']).mean())
sns.catplot(x='target', y='I3',  kind='bar', data=dataset)


# In[ ]:


print(dataset[['target' , 'I4']].groupby(['target']).mean())
plt.plot(dataset[['target' , 'I4']].groupby(['target']).mean())
sns.catplot(x='target', y='I4',  kind='bar', data=dataset)


# In[ ]:


print(dataset[['target' , 'I5']].groupby(['target']).mean())
plt.plot(dataset[['target' , 'I5']].groupby(['target']).mean())
sns.catplot(x='target', y='I5',  kind='bar', data=dataset)


# In[ ]:


print(dataset[['target' , 'I6']].groupby(['target']).mean())
plt.plot(dataset[['target' , 'I6']].groupby(['target']).mean())
sns.catplot(x='target', y='I6',  kind='bar', data=dataset)


# FROM THE ABOVE PLOTS WE HAVE OBSERVED THAT, CATEGORY 3 AND 4, GENERALLY HAVE THE HIGHER VALUE FOR X1-X6 FEATURES
# 
# AND THE MAGNITUDE OF FEATURE VALUES I1-U6 IS GREATER FOR TARGET VALUE 1 AND 2

# In[ ]:


sns.heatmap(dataset.corr(), annot=True, fmt = ".2f", cmap = "coolwarm")


# WE CAN OBSERVE THAT TARGET VARIABLE IS HIGHLY CORRELATED WITH X1,X2,X3,X5,X6,I2 FEATURES

# In[ ]:


print(dataset.min())
print(dataset.max())


# as we can notice X1 - X6 features are all positive integers spanning from 0 - 10000 and I1-I6 
# 
# so to simplify the training process and making our model better,lets transform the I1-I6 features to positive values by adding a bias

# In[ ]:


dataset.iloc[: , 6:-1] = dataset.iloc[: , 6:-1] + 5
test.iloc[: , 6:-1] = test.iloc[: , 6:] + 5


# AS THE DATASET SET PROVIDED ID NOT SHUFFLED ON THE DEPENDANT VARIABLE. HENCE SHUFFLIING IT.

# In[ ]:


from sklearn.utils import shuffle
dataset = shuffle(dataset)
dataset = shuffle(dataset).reset_index()
dataset.drop('index' , axis = 1 , inplace = True)


# In[ ]:


dataset


# In[ ]:


print(dataset.min())
print(dataset.max())


# In[ ]:


y_train = dataset.iloc[: , -1:]
dataset.drop('target' , axis =1 , inplace = True)


# **DEALING WITH THE IMBALANCED DATA USING OVER SAMPLING**

# In[ ]:


from imblearn.over_sampling import SMOTE


# In[ ]:


sm = SMOTE(random_state= 2)
x_train , y = sm.fit_sample (dataset , y_train)


# In[ ]:


pd.DataFrame(y).hist()


# HENCE WE HAVE REMOVED THE IMBALANCED DATA PROBLEM

# **Now analyzing the significanve level of each feature in predicting the target variable**

# In[ ]:


x_train = pd.DataFrame(x_train , columns= ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6'])


# In[ ]:


from sklearn.decomposition import PCA
pca_x = PCA()
x_train.iloc[: ,:6] = pca_x.fit_transform(x_train.iloc[: ,:6])
test.iloc[: , :6] = pca_x.transform(test.iloc[: , :6])


# In[ ]:


pca_x.explained_variance_ratio_


# In[ ]:


pca_i = PCA()
x_train.iloc[: ,6:] = pca_i.fit_transform(x_train.iloc[: ,6:])
test.iloc[: , 6:] = pca_x.transform(test.iloc[: , 6:])


# In[ ]:


pca_i.explained_variance_ratio_


# HENCE WE CAN DROP X5, X6 ,I5,I6 BASED ON THEIR LOW EIGEN VALUE 

# In[ ]:


x_train.columns


# In[ ]:


x_train.drop([ 'I5' , 'I6'] , axis =1 ,inplace =True)
test.drop(['I5' , 'I6'] , axis =1 ,inplace =True)


# In[ ]:


x_train


# **NOW NORMALIZING THE DATA**

# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


normalizer_x = MinMaxScaler()  #for normalizing x1-x6
normalizer_i = MinMaxScaler()  #for normalizing i1-i6


# In[ ]:


test.shape


# In[ ]:


x_train.iloc[: , :6] = normalizer_x.fit_transform(x_train.iloc[: , :6])
x_train.iloc[: , 4:] = normalizer_i.fit_transform(x_train.iloc[: , 4:])

test.iloc[: , :6] = normalizer_x.fit_transform(test.iloc[: , :6])
test.iloc[: , 4:] = normalizer_i.fit_transform(test.iloc[: , 4:])


# In[ ]:


print(x_train.min())
print(x_train.max())


# In[ ]:


y = y.reshape((len(y) , 1))


# In[ ]:


#now as our dependant variable is categorical. therefore preprocessing it
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()


# In[ ]:


y = ohe.fit_transform(y)


# **NOW THE TRAINING PART BEGINS**

# In[ ]:


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Adding the output layer
classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(x_train ,y, batch_size = 100, epochs = 25 , validation_split = 0.1)


# HENCE WITH A SIMPLE 3 LAYER NN WE ARE ABLE ACHIEVE ABOVE 85% VALIDATION ACCURACY. WHICH IS DESCENT !
# HENCE OUR MODEL IS ABLE TO GENERALIZE.

# ***NOW MAKING PREDICTIONS .................................................................***

# In[ ]:


# storing the results
y_pred_1 = np.argmax( classifier.predict(test) , axis = 1)


# In[ ]:


y_pred_1 = y_pred_1 + 1


# In[ ]:


y_pred_1 = pd.DataFrame(y_pred_1)


# In[ ]:


y_pred_1.hist()


# In[ ]:


final = pd.read_csv('../input/land classification challenge/socialcops_challenge/land_test.csv')


# In[ ]:


final['target'] = np.array(y_pred_1)


# In[ ]:


final


# In[ ]:


final['target'].value_counts()


# In[ ]:


final.to_csv('result_2.csv' , index=False)


# In[ ]:




