#!/usr/bin/env python
# coding: utf-8

# # I tried to keep this as simple as possible, no preprocessing nothing, grab the data , split it and apply the algorithm.
# 
# # If you have any questions you can always ask in the comment's section or you can just google it.
# 
# # So Let's begin

# In[ ]:


# this cell is just to access the csv files from Kaggle so don't worry about it

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# reading the data into the dataframe
df=pd.read_csv('../input/mushroom-classification/mushrooms.csv')
# let's take a peek into our data
df.head()


# In[ ]:


# let's check how many records we have and how many columns / features are there in our data

df.shape

# So we have 8124 records/rows and 23 columns/features


# In[ ]:


# let's check our target columns/feature
df


# In[ ]:


# let's look at our target varibale
df.columns

# Here class is our target varibale so let's start by separating it 


# In[ ]:


# let's move our target into y ( just a convetion for better readbality)
y=df['class']
y


# In[ ]:


# let's define our x with all other columns except the target columns
x=df.iloc[:,1:]
x


# In[ ]:


# let's check if there are any missing values in our  dataset

x.isna().sum()
# as we can see there are no missing values so we can start creating our model


# # There are different Classification algorithms available which can be applied to this dataset , However because we want to keep things simple we will start with a Simple apporach, a model which requires minimal preprocessing  

# In[ ]:


from catboost import CatBoostClassifier

model=CatBoostClassifier()

# we have just defined our model with default parameters 
# Cat boost Classifier is a boosting based appraoch and can work with categorical variables which spare us from the 
# gard work of converting our non numeric columns into numeric using ( LabelEncoder , get_dummies etc)


# In[ ]:


# now let's make our train and test datasets
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=.25,random_state=3)

# we have split our x , y into x_train, x_test and y_train,y_test respectively with 25% as our test size and 75% train size 


# In[ ]:


# let's convert our target to numeric value 
from sklearn.preprocessing import LabelEncoder
y_train=pd.get_dummies(y_train,drop_first=True)


# In[ ]:


# let's tell our model which columns are of object type
a=x_train.select_dtypes(include='object')
li=a.columns.to_list()
# now let's fit our model 
model.fit(x_train,y_train,cat_features=li)


# In[ ]:


# let's check the training accuracy
model.score(x_train,y_train)

# model gives us a 100% accuracy ( looks like the model might be overfitting)


# In[ ]:


# let's see how our model performs on Test Data

model.score(x_test,y_test)

# model gives us a 100% accuracy 


# In[ ]:


# let's look at a classification report

from sklearn.metrics import classification_report
classification_report(model.predict(x_test),y_test)


# # As we can see the model performs exceptionally well and can handle categorical variable as well , and gives us a 100% accuracy on this dataset.
# 
# 
# # Catboost algorithm is an excellent algorithm to be used for classification problems as it can handle categorical variables as well and saves our time and effort and is extremly simple to apply

# # If you like this notebook please check out my other noteboks and if you can please upvote so that even others can see it.

# 
