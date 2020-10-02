#!/usr/bin/env python
# coding: utf-8

# <center><h1>Introduction to Python - Titanic Baseline </h1></center>
# <center><h3><a href = 'http://www.analyticsdojo.com'>www.analyticsdojo.com</a></h3></center>

# ## Running Code using Kaggle Notebooks
# - Kaggle utilizes Docker to create a fully functional environment for hosting competitions in data science.
# - You could download/run kaggle/python docker image from [GitHub](https://github.com/kaggle/docker-python) and run it as an alternative to the standard Jupyter Stack for Data Science we have been using.
# - Kaggle has created an incredible resource for learning analytics.  You can view a number of *toy* examples that can be used to understand data science and also compete in real problems faced by top companies. 

# In[ ]:


import numpy as np 
import pandas as pd 

# Input data files are available in the "../input/" directory.
# Let's input them into a Pandas DataFrame
train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')


# ## `train` and `test` set on Kaggle
# - The `train` file contains a wide variety of information that might be useful in understanding whether they survived or not. It also includes a record as to whether they survived or not.
# - The `test` file contains all of the columns of the first file except whether they survived. Our goal is to predict whether the individuals survived.

# In[ ]:


train.head()


# In[ ]:


test.head()


# ## Baseline Model: No Survivors
# - The Titanic problem is one of classification, and often the simplest baseline of all 0/1 is an appropriate baseline.
# - Even if you aren't familiar with the history of the tragedy, by checking out the [Wikipedia Page](https://en.wikipedia.org/wiki/RMS_Titanic) we can quickly see that the majority of people (68%) died.
# - As a result, our baseline model will be for no survivors.

# In[ ]:


test["Survived"] = 0


# In[ ]:


submission = test.loc[:,["PassengerId", "Survived"]]


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('everyoneDies.csv', index=False)


# ## The First Rule of Shipwrecks
# - You may have seen it in a movie or read it in a novel, but [women and children first](https://en.wikipedia.org/wiki/Women_and_children_first) has at it's roots something that could provide our first model.
# - Now let's recode the `Survived` column based on whether was a man or a woman.  
# - We are using conditionals to *select* rows of interest (for example, where test['Sex'] == 'male') and recoding appropriate columns.

# In[ ]:


#Here we can code it as Survived, but if we do so we will overwrite our other prediction. 
#Instead, let's code it as PredGender

test.loc[test['Sex'] == 'male', 'PredGender'] = 0
test.loc[test['Sex'] == 'female', 'PredGender'] = 1
test.PredGender.astype(int)


# In[ ]:


submission = test.loc[:,['PassengerId', 'PredGender']]
# But we have to change the column name.
# Option 1: submission.columns = ['PassengerId', 'Survived']
# Option 2: Rename command.
submission.rename(columns={'PredGender': 'Survived'}, inplace=True)


# In[ ]:


submission.to_csv('womenSurvive.csv', index=False)

