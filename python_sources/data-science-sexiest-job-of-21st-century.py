#!/usr/bin/env python
# coding: utf-8

# This is a work in process.I will be updating the kernel in the coming days.If you like my work please do vote by cliking on vote at the top of the page

# In[ ]:


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


# # Importing Python Modules

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Importing the data will nneed ISO-8859-1 encoding

# In[ ]:


mcr = pd.read_csv('../input/multipleChoiceResponses.csv',encoding='ISO-8859-1')


# In[ ]:


mcr.head()


# # Finding out the top 15 countries which have maximum participation in survey

# In[ ]:


part=mcr['Country'].value_counts()[:15].to_frame()
sns.barplot(part['Country'],part.index,palette='spring')
plt.title('Top 15 Countries by number of respondents')
plt.xlabel('Number of people participated')
fig=plt.gcf()
fig.set_size_inches(10,5)
plt.show()


# US and India had the highest participants in the survey.Interestingly China which leads in AI research doesnt feature on the top 15 list.

# # Education qualification of participants in the survey

# In[ ]:


sns.countplot(y='FormalEducation', data=mcr)


# It seems that majority of the people working in area of datascience have a master degree.

# In[ ]:


sns.countplot(y='MajorSelect', data=mcr)


# People with computer science,Maths and Electrical engineeing are more in the survey.

# # Programming language most used for datascience

# In[ ]:


sns.countplot(y='LanguageRecommendationSelect',data=mcr)


# Most people use Python and R comes in second place

# # Finding out most popular tools

# In[ ]:


tools=mcr['MLToolNextYearSelect'].value_counts().head(10)
sns.barplot(y=tools.index,x=tools)


# Tensor flow is most used tool

# In[ ]:




