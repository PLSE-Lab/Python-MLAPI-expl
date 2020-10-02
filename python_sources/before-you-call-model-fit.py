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


train = pd.read_csv('/kaggle/input/data-science-nigeria-ai-in-citie/train.csv')
test = pd.read_csv('/kaggle/input/data-science-nigeria-ai-in-citie/test.csv')


# As a tutor for Port Harcourt City, I decided to take a look at the competition and achieve what I believed should be a baseline model for the competition. Here are my key findings from domain knowledge:
# - Predicting that all the males died, you get a public and private leaderboard score of around 0.78 and 0.77 respectively
# 
# - Predicting that only the adult males died (by simple boolean indexing into the dataframe) gets you a public leaderboard score of 0.81 but a private leaderboard score of 0.77 
# 
# Kindly note that I achieved these two results by just indexing into the dataframe without any machine learning models using domain knowledge. This is important as in one's journey as a data scientist or machine learning engineer, one should not be too quick to use machine learning before setting non-ML baselines. Not doing this is a recipe for disaster. 
# 
# I decided to do a mini exploratory data analysis for the dataset to see if I can set what I called an "EDA + Domain Knowledge" baseline and here was when I found out something interesting. Having read what the features were all supposed to represent from the competition page (something beginners dont usually do), I decided to create a new feature called "no_medboat" to represent those that did not have access to a MedBoat as shown by NaN in the MedBoat feature. Kindly note that NaN may have more meaning than just no data and as a data sceintist, you have to be aware of that. Not all NaNs should be dropped or Filled!

# In[ ]:


train['no_medboat'] = train['MedBoat'].isnull()
test['no_medboat'] = test['MedBoat'].isnull()


# After creating this feature, I decided to check the correlation of the dataset and see what I found:

# In[ ]:


train.corr()


# Almost everyone who did not have access to a med boat died as shown by the whooping correlation value of -0.95. Using this knowledge, I created a new submission

# In[ ]:


predictions = test['no_medboat'].map({True:0,False:1})


# In[ ]:


submission = pd.read_csv('/kaggle/input/data-science-nigeria-ai-in-citie/sample_submission.csv')


# In[ ]:


submission['Survived'] = predictions


# In[ ]:


submission.to_csv('sub.csv', index=False)


# This simple submission gets a public and private leadboard score of around 0.97 which ended up 5th on the private leaderboard, not too bad for a five line non-machine learning solution in a machine learning competition.
# ![image.png](attachment:image.png)
# 
# The main reason why I share this is because as a beginner about three years ago, I was so fascinated with building machine learning models that I didn't bother too much about data cleaning, exploratory data analysis, domain knowledge and feature engineering. I only came to the realization at the Data Science Bootcamp of 2018 which I was privileged to qualify for. So no matter where you ended up on the leaderboard, I want you to understand that proper machine learning modelling is about the entire pipeline and not just model.fit
# 
# Before you call model.fit, you must have:
# - understood what each feature represent and how certain values could mean certain things using domain knowledge,
# - properly cleaned your data,
# - come up with hypothesis and test them using the data during exploratory data analysis,
# - engineer relevant features using the knowledge you now have from the domain and EDA
# - set up good non-machine learning/ simple machine learning baselines.
# 
# I hope this advice can help one or two people kick on in their data science careers like some of us did. Remember that the goal is 1 million AI talents in 10 years and you also have a big part to play in this dream. Cheers!

# In[ ]:




