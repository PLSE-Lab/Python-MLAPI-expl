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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df=pd.read_csv('../input/titanic/train.csv')


# In[ ]:





# In[ ]:





# In[ ]:


df.head()


# In[ ]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False)


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())


# In[ ]:


df.shape


# In[ ]:


df.drop(['Id'],axis=1,inplace=True)


# In[ ]:


df.isnull().sum()


# In[ ]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')


# In[ ]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='YlGnBu')


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


len(columns)


# In[ ]:


main_df=df.copy()


# In[ ]:


test_df=pd.read_csv('formulatedtest.csv')


# In[ ]:


final_df


# In[ ]:


import xgboost
regressor=xgboost.XGBRegressor()


# In[ ]:


booster=['gbtree','gblinear']
base_score=[0.25,0.5,0.75,1]


# In[ ]:


n_estimators = [100, 500, 900, 1100, 1500]
max_depth = [2, 3, 5, 10, 15]
booster=['gbtree','gblinear']
learning_rate=[0.05,0.1,0.15,0.20]
min_child_weight=[1,2,3,4]

# Define the grid of hyperparameters to search
hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'booster':booster,
    'base_score':base_score
    }


# In[ ]:


# Set up the random search with 4-fold cross validation
random_cv = RandomizedSearchCV(estimator=regressor,
            param_distributions=hyperparameter_grid,
            cv=5, n_iter=50,
            scoring = 'neg_mean_absolute_error',n_jobs = 4,
            verbose = 5, 
            return_train_score = True,
            random_state=42)


# In[ ]:


ann_pred=classifier.predict(df_Test.drop(['SalePrice'],axis=1).values)


# In[ ]:


from keras import backend as K
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))


# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 50, init = 'he_uniform',activation='relu',input_dim = 174))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 25, init = 'he_uniform',activation='relu'))

# Adding the third hidden layer
classifier.add(Dense(output_dim = 50, init = 'he_uniform',activation='relu'))
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'he_uniform'))

# Compiling the ANN
classifier.compile(loss=root_mean_squared_error, optimizer='Adamax')

# Fitting the ANN to the Training set
model_history=classifier.fit(X_train.values, y_train.values,validation_split=0.20, batch_size = 10, nb_epoch = 1000)


# In[ ]:




