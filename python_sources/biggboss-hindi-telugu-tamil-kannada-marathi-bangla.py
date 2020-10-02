#!/usr/bin/env python
# coding: utf-8

# ## Bigg Boss Hindi/Kannada/Tamil/Telugu/Malayalam/Marathi/Bangla Data sets and Data Analysis
# 
# ## https://satya-python.blogspot.com/
# 
# ### Importing Required Python Libraries

# In[ ]:


import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 50)

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# ### Import dataset

# In[ ]:


bigg_boss = pd.read_csv('/kaggle/input/Bigg_Boss_India.csv', encoding = "ISO-8859-1")
nRow, nCol = bigg_boss.shape
print(f'There are {nRow} rows and {nCol} columns')


# ## Exploratory Data Analysis (EDA)

# In[ ]:


bigg_boss.head(5)


# In[ ]:


bigg_boss.tail(10).T


# In[ ]:


bigg_boss.sample(10)


# In[ ]:


bigg_boss.info()


# In[ ]:


bigg_boss.describe()


# In[ ]:


# Unique values in each column
for col in bigg_boss.columns:
    print("Number of unique values in", col,"-", bigg_boss[col].nunique())


# In[ ]:


# Number of seasons in all Indian languages
print(bigg_boss.groupby('Language')['Season Number'].nunique().sum())

# 32 seasons happened (including current seasons)


# ## https://www.kaggle.com/thirumani/bigg-boss-india-hindi-telugu-tamil-kannada

# ## Bigg Boss Hindi has many seasons compared to other Indian languages. So, number of housemates are more in Hindi.

# In[ ]:


# Number of seasons in each Indian language
print(bigg_boss.groupby('Language')['Season Number'].nunique().nlargest(10))


# In[ ]:


# Total number of Bigg Boss housemates
fig = plt.figure(figsize=(10,4))
ax = sns.countplot(x='Language', data=bigg_boss)
ax.set_title('Bigg Boss Series - Indian Language')
for t in ax.patches:
    if (np.isnan(float(t.get_height()))):
        ax.annotate(0, (t.get_x(), 0))
    else:
        ax.annotate(str(format(int(t.get_height()), ',d')), (t.get_x(), t.get_height()*1.01))


# ## In Bigg Boss India seasons, most of the housemates entered in first day/week

# In[ ]:


# Number of normal entries and wild card entries
print(bigg_boss['Wild Card'].value_counts(), "\n")
print(round(bigg_boss['Wild Card'].value_counts(normalize=True)*100))
sns.countplot(x='Wild Card', data=bigg_boss)


# In[ ]:


# Common people has many professions, so clubbing them into one category
bigg_boss.loc[bigg_boss['Profession'].str.contains('Commoner'),'Profession']='Commoner'


# ## Number of film actress entered into the Bigg Boss houses, are more when compared to other professions 

# In[ ]:


# Participant's Profession
print(bigg_boss['Profession'].value_counts())
fig = plt.figure(figsize=(25,8))
sns.countplot(x='Profession', data=bigg_boss)
plt.xticks(rotation=90)


# In[ ]:


# Broadcastor
fig = plt.figure(figsize=(20,5))
ax = sns.countplot(x='Broadcasted By', data=bigg_boss, palette='RdBu')
ax.set_title('Bigg Boss Series - Indian Broadcastor & Total Number of Housemates')
for t in ax.patches:
    if (np.isnan(float(t.get_height()))):
        ax.annotate(0, (t.get_x(), 0))
    else:
        ax.annotate(str(format(int(t.get_height()), ',d')), (t.get_x(), t.get_height()*1.01))


# ## Salman Khan hosted most number of seasons (in Bigg Boss Hindi), Sudeep is next in the list

# In[ ]:


bigg_boss.groupby('Host Name')['Season Number'].nunique().nlargest(25)


# ## In all Bigg Boss languages, and in all seasons, Female contestants are more

# In[ ]:


# Housemate's Gender
print(bigg_boss['Gender'].value_counts())


# ### 5 Transgenders participated in all Indian languages

# In[ ]:


# Maximum TRP of Bigg Boss Hindi/India seasons
print("Maximum TRP",bigg_boss['Average TRP'].max(), "\n")
print(bigg_boss.loc[bigg_boss['Average TRP']==bigg_boss['Average TRP'].max()][["Language","Season Number"]].head(1).to_string(index=False))


# In[ ]:


# Longest season of Bigg Boss Hindi/India seasons
print("Longest season",bigg_boss['Season Length'].max(), "days \n")
print(bigg_boss.loc[bigg_boss['Season Length']==bigg_boss['Season Length'].max()][["Language","Season Number"]].head(1).to_string(index=False))


# ## https://satya-data.blogspot.com/2018/01/bigg-boss-data-set-bigg-boss.html

# In[ ]:


# All BB Winners
bigg_boss.loc[bigg_boss.Winner==1]


# In[ ]:


# Profession of BB Season Winners
bigg_boss.loc[bigg_boss.Winner==1,'Profession'].value_counts()


# In[ ]:


# Gender of Season title Winners
bigg_boss.loc[bigg_boss.Winner==1,'Gender'].value_counts()


# ## No wild card entry housemate won the Bigg Boss competition.

# In[ ]:


# Entry type of the Season Winners
bigg_boss.loc[bigg_boss.Winner==1,'Wild Card'].value_counts()


# In[ ]:


# No re-entered contestant won Bigg Boss title
bigg_boss.loc[bigg_boss.Winner==1,'Number of re-entries'].value_counts()


# In[ ]:


# Number of eliminations or evictions faced by the Bigg Boss competition winners
bigg_boss.loc[bigg_boss.Winner==1,'Number of Evictions Faced'].value_counts().sort_index()

# Number of eliminations faced - Number of Winners


# In[ ]:


# Bigg Boss winners Number of times elected as Captain
bigg_boss.loc[bigg_boss.Winner==1,'Number of times elected as Captain'].value_counts().sort_index()

# Number of times elected as Captain   - Number of winners


# In[ ]:


import pandas_profiling
pandas_profiling.ProfileReport(bigg_boss)


# In[ ]:





# ## Machine Learning Models to predict Indian Bigg Boss season Winners

# ## Telugu Bigg Boss Season3
# ## https://www.kaggle.com/thirumani/predicting-bigg-boss-telugu-season-3-winner

# ## Kannada Bigg Boss Season7
# ## https://www.kaggle.com/thirumani/predicting-bigg-boss-kannada-season-7-winner

# ## Hindi Bigg Boss Season13
# ## https://www.kaggle.com/thirumani/predicting-bigg-boss-hindi-season-13-winner

# ## Malayalam Bigg Boss Season2
# ## https://www.kaggle.com/thirumani/predicting-bigg-boss-malayalam-season-2-winner

# In[ ]:




