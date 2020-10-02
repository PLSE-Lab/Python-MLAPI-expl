#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Intro
# 
# EDA notebook after day 1 of the [Practical Model Evaluation workshop](https://www.kaggle.com/rtatman/practical-model-evaluation-day-1/) from Kaggle.
# 
# The project uses data from the 2018 & 2019 Kaggle Data Science Survey and the final goal will be to predict job titles from data about role responsibilities. 

# In[ ]:


# Look at the questions only
df_questions_only = pd.read_csv(os.path.join(dirname, 'questions_only.csv'))
df_questions_only.T


# In[ ]:


# Look at the responses
df_mcq_responses = pd.read_csv(os.path.join(dirname, 'multiple_choice_responses.csv'), low_memory=False)
df_mcq_responses.head()


# In[ ]:


#Length of the file
len(df_mcq_responses)


# In[ ]:


df_mcq_responses.T


# In[ ]:


#Look at answers to one choice questions only
all_that_apply = ['Q13', 'Q21', 'Q16', 'Q9', 'Q30', 'Q31', 'Q29', 'Q17', 'Q24', 'Q28', 'Q25', 'Q20', 'Q32', 'Q27', 'Q26', 'Q12', 'Q33', 'Q18']
q_columns = ['Q'+str(x) for x in range(1,34)]
one_choice = q_columns
for x in all_that_apply:
    one_choice.remove(x)
df_one_choice = df_mcq_responses[one_choice]
df_one_choice.head()


# In[ ]:


len(one_choice)


# In[ ]:


fig, ax = plt.subplots(5, 3, figsize=(15, 25))
for variable, subplot in zip(one_choice, ax.flatten()):
    sns.countplot(df_one_choice[1:][variable], 
                  ax=subplot, 
                  order=df_one_choice[1:][variable].value_counts().index.tolist(), 
                  palette='PuBu_r')
    for label in subplot.get_xticklabels():
        label.set_rotation(90)


# ## Note:
# Some of the questions were not answered by all the respondents (see survey schema), so that might bias the analysis. Next steps could include analyzing the remaining questions (those with multiple answers allowed) and to perform more analysis on the subsets of questions answered by a limited number of respondents.
