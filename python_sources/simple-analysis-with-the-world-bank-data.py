#!/usr/bin/env python
# coding: utf-8

# Hello Kagglers! It's my first time to write a kaggle kernel. **I analyzed the survey data with the World Bank Data.** <br>
# My kernel is very simple and short. So you can easily enjoy my analysis!

# In[ ]:


# Import Library!
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import os
import warnings
from matplotlib import style

sns.set_style('whitegrid')
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


survey_df = pd.read_csv('../input/kaggle-survey-2018/multipleChoiceResponses.csv')
ques_list = survey_df.iloc[0,:].values.tolist()
survey_df = survey_df.iloc[1:, ]

survey_df['Q3'].value_counts()[:32]


# - Import Basic Data and pick only the top 30 countries (Based on the number of answers)

# In[ ]:


worldbank = pd.read_excel('../input/world-bank-datakaggle-survey/WorldBank_Data.xlsx')
worldbank.head()


# - All these data were based on World Bank data. (The latest data are for 2017)
# - I only brought data about the top 30 countries.
# - There is a lot of data about the economy. I want you to refer to it.  https://www.worldbank.org/

# In[ ]:


plt.figure(figsize=(13,7))
sns.regplot(worldbank['GDP_Per_Capita'], worldbank['Count/POP'])
plt.ylabel('Ans_Count/Total Popluation')
for i, v, s in worldbank[['GDP_Per_Capita', 'Count/POP', 'Country Code']].values :
    plt.text(i,v,s)


# - Wow! There is a very strong correlation between 'Ans_Count/Total Population' & 'GDP_Per_Capita'.

# - I think connecting and analyzing economic data will extract more insights!
# - I hope this analysis will help you analyze your data!

# # It's not my Final version! I'll update!
