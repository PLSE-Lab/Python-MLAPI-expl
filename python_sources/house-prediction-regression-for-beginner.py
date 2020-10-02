#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
from sklearn.preprocessing import LabelEncoder  ###for encode a categorical values
from sklearn.model_selection import train_test_split  ## for spliting the data
from lightgbm import LGBMRegressor    ## for import our model
from sklearn.preprocessing import LabelEncoder
print(os.listdir("../input"))


# ## Let's import our data

# In[ ]:


train_dataset = pd.read_csv('../input/train.csv')


# In[ ]:


train_dataset.head()


# In[ ]:


train_dataset.shape


# ## Let's split our data into X and Y

# In[ ]:


x = train_dataset.iloc[:,1:-1]
y = train_dataset.iloc[:,-1] 


# ## Now let's check for a NULL values

# In[ ]:


x.isnull().sum()


# ** There are null values in many column in our X  ** 

# ** So we have to find the columns that contain missing values than we will fill with missing values filling techniques **

# ** We have fill numerical columns misssing values with median **
# ** We will fill character missing values with most used value count **

# In[ ]:


col_miss_val = [col for col in train_dataset.columns if train_dataset[col].isnull().any()]
print(col_miss_val)


# In[ ]:


for col in col_miss_val:
    if(x[col].dtype == np.dtype('O')):
         x[col]=x[col].fillna(x[col].value_counts().index[0])    #replace nan with most frequent
    else:
        x[col] = train_dataset[col].fillna(x[col].median()) 


# In[ ]:


x.isnull().sum()


# ## Now we have to encode a categorical values

# In[ ]:


##So first we will find a columns thats contain characters value 
x.select_dtypes(include=['object'])


# ** So there are many columns that contains character values **

# In[ ]:


LE = LabelEncoder()
for col in x.select_dtypes(include=['object']):
    x[col] = LE.fit_transform(x[col])


# In[ ]:


x.head()


# ## Let's check the missing values in NULL

# In[ ]:


y.isnull().sum()


# ## Let's split the data into Training & testing

# In[ ]:


from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y ,test_size = 0.1,random_state = 0)


# ## Let's create our LGBMRegressor Model

# In[ ]:


lightgbm = LGBMRegressor(objective='regression', 
                                       num_leaves=8,
                                       learning_rate=0.0385, 
                                       n_estimators=3500,
                                       max_bin=200, 
                                       bagging_fraction=0.75,
                                       bagging_freq=5, 
                                       bagging_seed=7,
                                       feature_fraction=0.2,
                                       feature_fraction_seed=7,
                                       verbose= 0,
                                       )


# ## Let's fit our training data into Model

# In[ ]:


lightgbm.fit(x_train,y_train)


# In[ ]:


lightgbm.score(x_test,y_test)


# ## Our model created now we have to import test dataset

# In[ ]:


test_dataset = pd.read_csv('../input/test.csv')


# In[ ]:


test_dataset.isnull().sum()
test_dataset = test_dataset.iloc[:,1:]


# In[ ]:


test_dataset.isnull().sum()


# ** There are many missing values in test dataset **

# In[ ]:


test_col_miss_val = [col for col in test_dataset.columns if test_dataset[col].isnull().any()]
print(test_col_miss_val)


# In[ ]:


for col in test_col_miss_val:
    if(test_dataset[col].dtype == np.dtype('O')):
        test_dataset[col] = test_dataset[col].fillna(test_dataset[col].value_counts().index[0])    #replace nan with most frequent
        
    else:
        test_dataset[col] = test_dataset[col].fillna(test_dataset[col].median()) 


# ** Let's enode a categorical data **

# In[ ]:


for col in test_dataset.select_dtypes(include=['object']):
    test_dataset[col] = LE.fit_transform(test_dataset[col])   


# In[ ]:


test_dataset.head()


# ## Let's predict the test dataset

# In[ ]:


prediction = lightgbm.predict(test_dataset)


# In[ ]:


print(prediction)


# ## Let's create a submission.csv of prediction

# In[ ]:


ss = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


output = pd.DataFrame({'Id': ss.Id,'SalePrice': prediction})
output.to_csv('submission.csv', index=False)
output.head()


# In[ ]:




