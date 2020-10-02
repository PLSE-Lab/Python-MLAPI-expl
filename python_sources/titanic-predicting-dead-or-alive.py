#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['dark_background','ggplot'])


# In[3]:


train_df=pd.read_csv('../input/train.csv')
test_df=pd.read_csv('../input/test.csv')


# In[ ]:


print(train_df.info())
print('--'*30+'\n'+'--'*30)
print(test_df.info())
train_df.head(10)


# In[ ]:


print('Training columns with Null Values \n',train_df.isna().sum())
print('-'*20+'\n'+'-'*20)
print('Test columns with Null or NaN Values \n',test_df.isnull().sum()) 
#isna(),isnull() does the same thing


# # Cleaning Data 

# In[4]:


df_clean=[train_df,test_df]

drop_column=['PassengerId','Cabin','Ticket','Name']

for dataset in df_clean:
    dataset["Age"].fillna(dataset['Age'].median(), inplace = True)
    
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)
    
    dataset.drop(drop_column,axis=1,inplace=True)

print('Training columns with Null Values \n',train_df.isna().sum())
print('-'*20+'\n'+'-'*20)
print('Test columns with Null or NaN Values \n',test_df.isnull().sum()) 
    


# In[5]:


train_df['Embarked'].value_counts().plot(kind='bar')


# ### Since 'S' is more Na values be replaced with 'S' in 'Embarked' column

# In[6]:


train_df['Embarked'].fillna(train_df['Embarked'].mode()[0],inplace=True)


# In[ ]:


print('Training columns with Null Values \n',train_df.isna().sum())


# In[7]:


train_df = pd.get_dummies(columns=['Embarked','Sex'],data=train_df,drop_first=True)
test_df = pd.get_dummies(columns=['Embarked','Sex'],data=test_df,drop_first=True)


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[9]:


X=train_df.drop('Survived',axis=1)
y=train_df['Survived']


# ### Importing libraries and compiling different models with different units in hidden layer 

# In[ ]:


import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
early_stopping_monitor=EarlyStopping(patience=3)
#from keras.utils import to_categorical
#Setting up the model
model_1=Sequential()
model_2=Sequential()
#Add first layer
model_1.add(Dense(50,activation='relu',input_shape=(X.shape[1],)))
model_2.add(Dense(100,activation='relu',input_shape=(X.shape[1],)))
#Add second layer
model_1.add(Dense(32,activation='relu'))
model_2.add(Dense(50,activation='relu'))
#Add output layer
model_1.add(Dense(1,activation='sigmoid'))
model_2.add(Dense(1,activation='sigmoid'))
#Compile the model
model_1.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
model_2.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])


# ### Fitting models 

# In[ ]:


# Fit model_1
model_1_training = model_1.fit(X, y, epochs=30, validation_split=0.2, callbacks=[early_stopping_monitor])

# Fit model_2
model_2_training = model_2.fit(X, y, epochs=30, validation_split=0.2, callbacks=[early_stopping_monitor])


# In[ ]:


plt.plot(model_1_training.history['val_loss'], 'r', model_2_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()


# ### So model_1 is good 

# In[ ]:


y_pred = [x[0] for x in model_2.predict(test_df)]
df = pd.DataFrame({'PassengerId':pd.read_csv('../input/test.csv')['PassengerId'].values,'Survived':y_pred},dtype=int)
#df.to_csv('submission.csv', index=False)
#It gave only around 62% accuracy on test set on submission,


# ### Now using Regularization along with Hyperparameter Tuning and GridSearchCV 

# In[13]:


# Import necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Setup the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}

# Instantiate a logistic regression classifier: logreg
logreg = LogisticRegression()

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the data
logreg_cv.fit(X,y)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 
print("Best score is {}".format(logreg_cv.best_score_))


# In[14]:


#Now Using 'C'=0.4393970560760795 from above Result
logre=LogisticRegression(C=0.4393970560760795)
logre.fit(X,y)
y_pred=logre.predict(test_df)
df = pd.DataFrame({'PassengerId':pd.read_csv('../input/test.csv')['PassengerId'].values,'Survived':y_pred},dtype=int)
df.to_csv('submission.csv', index=False)

