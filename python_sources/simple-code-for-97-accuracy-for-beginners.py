#!/usr/bin/env python
# coding: utf-8

# let`s import the important libraries
# 

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # matplotlib is used for plotting
import pandas as pd # pandas is used for analysis ,cleaning, manipulate and some plotting techniques  
from sklearn.model_selection import train_test_split # train_test_split used to seperate data to train & test sets 
from sklearn.metrics import classification_report # used for text report showing the main classification metrics
from xgboost import XGBClassifier # the model we will used


# import the data
# 

# In[ ]:


train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')
kaggle = pd.read_csv('../input/digit-recognizer/sample_submission.csv')


# take a peek at the data
# 

# In[ ]:


print(train.head())


# In[ ]:


print(train.info())


# the data is already **cleaned**.  
# 
# let`s separate the data in X & y:

# In[ ]:


X = train.drop('label', axis=1)
y = train.label


# splitting the data into training sets and testing set
# 

# In[ ]:


# splitting the data into training sets and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=99)


# now let`s use XGBoost Classifier
# 

# In[ ]:


model = XGBClassifier(max_depth=30, n_estimators=300, gamma=0.5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# let`s check the acccuracy of the model

# In[ ]:


print(classification_report(y_test, y_pred))


# now let`s submit to kaggle competition
# 

# as you can see we get 0.97 accuracy. it is awesome result for simple code(with the help of the great XGBoost library)!
# 
# but of curse you can get **higher result** using **grid search** to get the best parameters for your model or **combining** another data,
# and think **PCA**(dimensionality reduction) will be improve the performance of the model.
# 
# I hope this was helpful for the beginner and it gave you some idea about how the game runs.
# please **upvote my work** if you like it (;
# 
