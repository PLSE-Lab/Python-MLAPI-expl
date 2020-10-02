#!/usr/bin/env python
# coding: utf-8

# # Work-Life Balance survey, an Exploratory Data Analysis of global best practices to re-balance our lives
# 
# 
# # Introduction
# ### Dataset
# The dataset analyzed in this kernel contains10,000+ responses to Authentic-Happiness.com global work-life survey.
# This [online survey](http://www.authentic-happiness.com/your-life-satisfaction-score) includes 23 questions about the way we design our lifestyle and achieve work-life balance.
# 
# ### Objectives
# The objective of this notebook is to conduct an Exploratory Data Analysis of the survey responses and advance the understanding of work-life balance and its major influencers are.
# 
# ### Table of content
# There are two main sections:
# 
# 1. Data extraction from Google Sheet and preparation
# 
# 2. Exploratory Analysis
#     - Healthy body
#     - Healthy mind
#     - Expertise
#     - Connection
#     - Meaning
# 
# ### In summary
# The key take-away for each of the five areas are the following:
# 1. Our **BMI** is most influenced by a quality nutrition (fruits/Vegetables) and physical activity (daily steps)
# 2. Key influencers of our **stress level** are our ability to concentrate on our work during flow sessions, daily meditation and the sufficiency of our income.
# 3. Those of us who **achieve the most remarkable things**, have also maximized our productivity (completing our daily todo list), focus on our activities (flow sessions) and have earnt multiple personal awards and recognition.
# 4. Having a **core circle of family members and close friends** influences the amount of new places we visit (discovery), reduces our daily stress and improve our social connections outside of this core circle.
# 5. We find **more time for passion** when we complete well our daily todo list (personal productivity), flow through the day and have obtained many personal awards and recognition.
# 
# ### Check other Kaggle notebooks from [Yvon Dalat](https://www.kaggle.com/ydalat):
# * [Titanic, a step-by-step intro to Machine Learning](https://www.kaggle.com/ydalat/titanic-a-step-by-step-intro-to-machine-learning): **a practice run ar EDA and ML-classification**
# * [HappyDB, a step-by-step application of Natural Language Processing](https://www.kaggle.com/ydalat/happydb-what-100-000-happy-moments-are-telling-us): **find out what 100,000 happy moments are telling us**
# * [Work-Life Balance survey, an Exploratory Data Analysis of lifestyle best practices](https://www.kaggle.com/ydalat/work-life-balance-best-practices-eda): **key insights into the factors affecting our work-life balance**
# *  [Work-Life Balance survey, a Machine-Learning analysis of best practices to rebalance our lives](https://www.kaggle.com/ydalat/work-life-balance-predictors-and-clustering): **discover the strongest predictors of work-life balance**
# 
# **Interested in more facts and data to balance your life, check the [360 Living guide](https://amzn.to/2MFO6Iy) ![360 Living: Practical guidance for a balanced life](https://images-na.ssl-images-amazon.com/images/I/61EhntLIyBL.jpg)**
# 
# # 1. Data Import and Preparation

# In[ ]:


import pandas as pd # collection of functions for data processing and analysis modeled after R dataframes with SQL like features
import numpy as np  # foundational package for scientific computing
import re           # Regular expression operations
import matplotlib.pyplot as plt # Collection of functions for scientific and publication-ready visualization
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py     # Open source library for composing, editing, and sharing interactive data visualization 
from matplotlib import pyplot as pp
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

import plotly.tools as tls
import seaborn as sns  # Visualization library based on matplotlib, provides interface for drawing attractive statistical graphics

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Import dataset
df = pd.read_csv('../input/lifestyle-and-wellbeing-data/Wellbeing_and_lifestyle_data.csv')


# In[ ]:


df.head(2)


# In[ ]:


df['AGE']=df['AGE'].replace('Less than 20', '20 or less')


# In[ ]:


colomn = df.columns
colomn


# ### Descriptive Summary

# In[ ]:


def descriptive(df):
    desc=df.describe().round(1).drop({'count', 'std', '50%'}, axis=0)
    i=-0.1
    j=0
    Row = int(round(len(desc.columns.tolist())/2+0.1))
    f,ax = plt.subplots(Row,2, figsize=(28,18))
    for name in desc.columns.tolist():
        desc[name].plot(kind='barh', figsize=(14,24), title=name, ax=ax[round(i), j], fontsize=14)
        for k, v in enumerate(desc[name].tolist()):
            ax[round(i), j].text(v, k-0.1, str(v), color='black', size = 14)
        i +=0.5
        if j==0: j=1
        else: j=0
    f.tight_layout()
descriptive(df)


# In[ ]:


df['MONTH'] = pd.DatetimeIndex(df['Timestamp']).month
#df.head(3)


# # 2. Exploratory Data Analysis (EDA)
# 
# ## 2.1 Healthy body

# In[ ]:


df2 = df.pivot_table(values='BMI_RANGE', index=['AGE'], columns=['GENDER'], )
df2.head()


# In[ ]:


# HEALTHY BODY
f,ax = plt.subplots(2,3,figsize=(16,10))
ax[0,0].set_title('BODY_MASS_INDEX BY GENDER')
ax[0,1].set_title('BODY_MASS_INDEX BY GENDER & AGE')
ax[0,2].set_title('BODY_MASS_INDEX BY GENDER & AGE')
ax[1,0].set_title('BODY_MASS_INDEX & SLEEP HOURS')
ax[1,1].set_title('BODY_MASS_INDEX & SERVINGS OF FRUITS/VEGGIES')
ax[1,2].set_title('BODY_MASS_INDEX & DAILY STEPS')

sns.pointplot(x = 'GENDER', y = 'BMI_RANGE',  data=df, ax = ax[0,0])
sns.violinplot(x = 'AGE', y = 'BMI_RANGE', hue = 'GENDER', data = df, palette='coolwarm_r',
               order=['20 or less', '21 to 35', '36 to 50', '51 or more'], split = True, ax = ax[0,1])

ax[0,2].set_ylim([1, 1.6])
df2.plot(kind='bar', color=('darksalmon', 'cornflowerblue'), alpha=0.7, ax = ax[0,2])
ax[0,2].tick_params(axis='x', rotation=0)

sns.pointplot(x = 'SLEEP_HOURS', y = 'BMI_RANGE',  data=df, ax = ax[1,0])
sns.pointplot(x = 'FRUITS_VEGGIES', y = 'BMI_RANGE', data=df, ax = ax[1,1])
sns.pointplot(x = 'DAILY_STEPS', y = 'BMI_RANGE',  data=df, ax = ax[1,2])

f.suptitle('HEALTHY BODY\nHOW TO KEEP OUR BMI BELOW 25 (IN THE GRAPHS BELOW, 1 IS FOR BMI<25; 2 FOR BMI>25', fontsize=20)
plt.show()


# ### Observations
# * The body max index data in this study were collected as 1 = a BMI below 25, 2 = a BMI above 25
# * BMI is strongly correlated to daily steps and servings of fruits & vegetables (negative correlastions)
#     * Both show a 15% impact on BMI when wakling 5,000 steps daily (versus less than 1,000), and eating 5 servings (versus less than 1).
#     * A rather intuitive outcome: physical activity and an healthy diet contribute to a lower BMI.
# * What is more interesting is that
#     * The BMI for men and women average to very close values for age groups "less than 20" and "51 or more".
#     * But stronger differences are found for the following age ranges:
#         * 21 to 35: women's BMI are higher
#         * 36 to 50: men's BMI are higher
# 
# ## 2.2 Healthy mind

# In[ ]:


df['DAILY_STRESS']=pd.to_numeric(df['DAILY_STRESS'],errors = 'coerce')


# In[ ]:


df3 = df.pivot_table(values='DAILY_STRESS', index=['AGE'], columns=['GENDER'], )
df3.head()


# In[ ]:


df3 = df.pivot_table(values='DAILY_STRESS', index=['AGE'], columns=['GENDER'], )
df3.head()


# In[ ]:


# HEALTHY MIND
f,ax = plt.subplots(2,3,figsize=(16,10))
ax[0,0].set_title('AVERAGE DAILY_STRESS BY AGE GROUP')
ax[0,1].set_title('DAILY_STRESS BY GENDER')
ax[0,2].set_title('DAILY_STRESS BY AGE & GENDER')
ax[1,0].set_title('DAILY_STRESS & DAILY HOURS OF FLOW')
ax[1,1].set_title('DAILY_STRESS & DAILY HOURS OF MEDITATION')
ax[1,2].set_title('DAILY_STRESS & SUFFICIENT INCOME:2=sufficient,1=not')

ax[0,0].set_ylim([2, 3.5])
df3.plot(kind='bar', color=('darksalmon', 'cornflowerblue'), alpha=0.7, ax = ax[0,0])
ax[0,0].tick_params(axis='x', rotation=0)

sns.violinplot(x= 'GENDER',y='DAILY_STRESS', palette='coolwarm_r', data=df, ax = ax[0,1])
sns.violinplot(x = 'AGE', y = 'DAILY_STRESS', hue = 'GENDER', palette='coolwarm_r', data = df,
               order=['20 or less', '21 to 35', '36 to 50', '51 or more'], split = True, ax = ax[0,2])
sns.pointplot(x = 'FLOW', y = 'DAILY_STRESS',  data=df, ax = ax[1,0])
sns.pointplot(x = 'DAILY_STRESS', y = 'DAILY_STRESS', data=df, ax = ax[1,1])
sns.pointplot(x = 'SUFFICIENT_INCOME', y = 'DAILY_STRESS',  data=df, ax = ax[1,2])

f.suptitle('HEALTHY MIND\nWHAT DRIVES OUR DAILY_STRESS?', fontsize=20)
plt.show()


# ### Observations
# * The overall stress level for women peaks in their younger years, and, while slowly going down remains higher than the male counterparts in all age groups.
#     * The American psychology association came to the same conclusion, see more explanations and background in their study published in the [link](https://www.apa.org/news/press/releases/stress/2010/gender-stress) 
#     * See also this [article](https://www.stress.org/why-do-women-suffer-more-from-depression-and-stress) from the American Institute of Stress
# * How ability to "flow" during the day, daily meditation, and an income sufficient to cover basic needs, all contribute to 30% lower levels of stress.
# 
# ## 2.3 Expertise

# In[ ]:


df4 = df.pivot_table(values='ACHIEVEMENT', index=['AGE'], columns=['GENDER'], )
df4.head()


# In[ ]:


# EXPERTISE
f,ax = plt.subplots(2,3,figsize=(16,10))
ax[0,0].set_title('AVERAGE ACHIEVEMENTS BY AGE')
ax[0,1].set_title('ACHIEVEMENTS BY GENDER')
ax[0,2].set_title('ACHIEVEMENTS BY AGE & GENDER')
ax[1,0].set_title('ACHIEVEMENTS & PERSONAL PRODUCTIVITY')
ax[1,1].set_title('ACHIEVEMENTS & DAILY HOURS OF FLOW')
ax[1,2].set_title('ACHIEVEMENTS & PERSONAL AWARDS RECEIVED')

ax[0,0].set_ylim([3.5, 4.5])
df4.plot(kind='bar', color=('darksalmon', 'cornflowerblue'), alpha=0.7, ax = ax[0,0])
ax[0,0].tick_params(axis='x', rotation=0)

sns.violinplot(x= 'GENDER',y='ACHIEVEMENT', palette='coolwarm_r', data=df, ax = ax[0,1])
sns.violinplot(x = 'AGE', y = 'ACHIEVEMENT', palette='coolwarm_r', hue = 'GENDER', data = df,
               order=['20 or less', '21 to 35', '36 to 50', '51 or more'], split = True, ax = ax[0,2])
sns.pointplot(x = 'TODO_COMPLETED',  y = 'ACHIEVEMENT',  data=df, ax = ax[1,0])
sns.pointplot(x = 'FLOW',  y = 'ACHIEVEMENT',  data=df, ax = ax[1,1])
sns.pointplot(x = 'PERSONAL_AWARDS', y = 'ACHIEVEMENT',  data=df, ax = ax[1,2])

f.suptitle('PERSONAL ACHIEVEMENTS\nWHAT DRIVE US TO ACHIEVE REMARKABLE THINGS?', fontsize=20)
plt.show()


# ### Observations
# * Woman reports slightly more personal achievements in their early age while men report more after age 36.
# * Our daily productivity,  the ability to flow hroughtout the day and personal awards such as diploma and other certificates all contribute to higher levels of personal achievements.
# 
# ## 2.4 Connection

# In[ ]:


df5 = df.pivot_table(values='CORE_CIRCLE', index=['AGE'], columns=['GENDER'], )
df5.head()


# In[ ]:


# CONNECTION
f,ax = plt.subplots(2,3,figsize=(16,10))
ax[0,0].set_title('CORE  CIRCLE BY GENDER')
ax[0,1].set_title('CORE_CIRCLE BY GENDER')
ax[0,2].set_title('LOST_VACATION BY AGE GROUP')
ax[1,0].set_title('PLACES & CORE_CIRCLE')
ax[1,1].set_title('LOST VACATION & DAILY_STRESS')
ax[1,2].set_title('FRIENDS & CORE_CIRCLE')

ax[0,0].set_ylim([4.5, 6])
df5.plot(kind='bar', color=('darksalmon', 'cornflowerblue'), alpha=0.7, ax = ax[0,0])
ax[0,0].tick_params(axis='x', rotation=0)

sns.violinplot(x= 'GENDER',y='CORE_CIRCLE', palette='coolwarm_r', data=df, ax = ax[0,1])
sns.pointplot(x = 'AGE', y = 'LOST_VACATION',order=['20 or less', '21 to 35', '36 to 50', '51 or more'], data = df, ax = ax[0,2])
sns.pointplot(x = 'CORE_CIRCLE',  y = 'PLACES_VISITED',    data=df, ax = ax[1,0])
sns.pointplot(x = 'LOST_VACATION',  y = 'DAILY_STRESS',    data=df, ax = ax[1,1])
sns.pointplot(x = 'CORE_CIRCLE',  y = 'SOCIAL_NETWORK',    data=df, ax = ax[1,2])

f.suptitle('CONNECTION\nHOW OUR CORE CIRCLE OF FRIENDS AND FAMILY STRENGTHENS OUR CONNECTION TO THE WORLD?', fontsize=20)
plt.show()


# ### Observations
# * Womem appear to have a stronger circle of friends and family than men.
# * People in the age group 21 to 35 forfeit a maximum of vacation days, when compared to other age groups.
# * Overall, the level of their sress increase as  we lose more vacation days. But there is a slight dip between 7 and 9 days for lost vacations, as if losing six or many more vacation days does not have any impact anymore on the stress level.
# 
# ## 2.5 Passion

# In[ ]:


df6 = df.pivot_table(values='TIME_FOR_PASSION', index=['AGE'], columns=['GENDER'], )
df6.head()


# In[ ]:


# PASSION
f,ax = plt.subplots(2,3,figsize=(16,10))
ax[0,0].set_title('AVERAGE TIME_FOR_PASSION BY GENDER')
ax[0,1].set_title('TIME_FOR_PASSION BY GENDER')
ax[0,2].set_title('TIME_FOR_PASSION BY AGE GROUP')
ax[1,0].set_title('TIME_FOR_PASSION & PERSONAL PRODUCTIVITY')
ax[1,1].set_title('TIME_FOR_PASSION & DAILY HOURS OF FLOW')
ax[1,2].set_title('TIME_FOR_PASSION & PERSONAL AWARDS RECEIVED')

ax[0,0].set_ylim([3, 4])
df6.plot(kind='bar', color=('darksalmon', 'cornflowerblue'), alpha=0.7, ax = ax[0,0]) 
ax[0,0].tick_params(axis='x', rotation=0)

sns.violinplot(x= 'GENDER',y='TIME_FOR_PASSION', palette='coolwarm_r', data=df, ax = ax[0,1])
sns.violinplot(x = 'AGE', y = 'TIME_FOR_PASSION', palette='coolwarm_r', hue = 'GENDER', data = df,
               order=['20 or less', '21 to 35', '36 to 50', '51 or more'], split = True, ax = ax[0,2])
sns.pointplot(x = 'TODO_COMPLETED',  y = 'TIME_FOR_PASSION',  data=df, ax = ax[1,0])
sns.pointplot(x = 'FLOW',  y = 'TIME_FOR_PASSION',  data=df, ax = ax[1,1])
sns.pointplot(x = 'PERSONAL_AWARDS', y = 'TIME_FOR_PASSION',  data=df, ax = ax[1,2])

f.suptitle('MEANING\nHOW DO FIND WE MORE TIME FOR OUR PASSIONS?', fontsize=20)
plt.show()


# ### Observations
# * Men appear to find more time for their passion, especially in their younger and older ages.
# * The three factors correlating the most with our ability to find time for our passions are:
#     * Our daily personal productivity
#     * Daily flow
#     * The personal awards we received
#   
