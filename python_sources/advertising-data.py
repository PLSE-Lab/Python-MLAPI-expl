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


# In this project I will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement.


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import pandas as pd
ad_data = pd.read_csv("../input/advertising.csv")


# In[ ]:


ad_data.head()


# In[ ]:


# to get all the columns information...
# we dont have nul values...
ad_data.info()


# In[ ]:


# STatistical Information...
ad_data.describe()


# In[ ]:


#sns.distplot(ad_data['Age'],bins=40)
sns.distplot(ad_data['Age'],kde=False,bins=40)


# In[ ]:


import matplotlib.style

import matplotlib as mpl

mpl.style.use('classic')


# In[ ]:


sns.jointplot(ad_data['Age'],ad_data['Area Income'],data=ad_data,kind='scatter')


# In[ ]:


# Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age.


# In[ ]:


sns.jointplot(ad_data['Age'],ad_data['Daily Time Spent on Site'],data=ad_data,kind='kde')


# In[ ]:


# Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'**


# In[ ]:


sns.jointplot(ad_data['Daily Time Spent on Site'],ad_data['Daily Internet Usage'],data=ad_data)


# In[ ]:


ad_data.info()


# In[ ]:


import matplotlib as mpl
mpl.rcParams['patch.force_edgecolor'] = True


# In[ ]:


sns.pairplot(data=ad_data,hue='Clicked on Ad',palette='bwr')


# In[ ]:


#       Logistic Regression
#       train test split, and train our model!


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
ad_data.columns


# In[ ]:


X = ad_data.drop(['Ad Topic Line', 'City','Country','Timestamp', 'Clicked on Ad'],axis=1)
y = ad_data['Clicked on Ad']


# In[ ]:


X.head()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[ ]:


logr=LogisticRegression()


# In[ ]:


logr.fit(X_train,y_train)


# In[ ]:


predictions = logr.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[ ]:


print(classification_report(y_test,predictions))


# In[ ]:


print(confusion_matrix(y_test,predictions))


# In[ ]:


# Predictions and Evaluations
#  ** Create a classification report for the model.**


# In[ ]:


print(classification_report(y_test,predictions))


# In[ ]:


# In this project I will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement. 
#  We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.


# In[ ]:





# In[ ]:




