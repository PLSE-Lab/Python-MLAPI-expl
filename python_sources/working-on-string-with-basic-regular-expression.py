#!/usr/bin/env python
# coding: utf-8

# # Working on string with basic Regular Expression
#   In This Notebook I'll working on string type dataset using World_dev.csv with basoc Regular Expression in panas

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


happiness = pd.read_csv("/kaggle/input/World_Happiness_2015.csv")
world = pd.read_csv("/kaggle/input/World_dev.csv")


# In[ ]:


happiness.head()


# In[ ]:


world.head()


# In[ ]:


world.info()


# In[ ]:


happiness.info()


# # Merging two data set one is World_Happiness_2015.csv & World_dev.csv
# 

# In[ ]:


merged = pd.merge(happiness, world, how = "left",left_on ="Country", right_on = "ShortName")
merged.head()


# ## Remove the Elipces in merged  

# In[ ]:


pd.options.display.max_columns  = None
merged.head()


# In[ ]:


merged.columns


# In[ ]:


merged["SpecialNotes"].head(10)


# In[ ]:


# in sprcialNotes Column we can get national accounts.
merged["SpecialNotes"].loc[4][-27:-10]


# # Regular Expression

# In[ ]:


pattern= R"[Nn]ational accounts"
National_account = merged["SpecialNotes"].str.contains(pattern, na =False )


# In[ ]:


National_account.head(20)


# In[ ]:


merged[National_account]['SpecialNotes'].head()


# In[ ]:


merged[National_account]['SpecialNotes'].tail()


# In[ ]:


# in sprcialNotes Column we can get years(1000-2050) of whole columns
year_pat = r"[1-2][0-9][0-9][0-9]"


# In[ ]:


years = merged["SpecialNotes"].str.contains(year_pat,na = False)
years.head(10)


# In[ ]:


year_pat = r"([1-2][0-9][0-9][0-9])"
a = merged["SpecialNotes"].str.extractall(year_pat)
a.rename({0:"Years"}, axis = 1, inplace = True)


# In[ ]:


a.head(10)


# In[ ]:


a.tail(10)


# In[ ]:


merged.columns


# In[ ]:


merged.rename({'SourceOfMostRecentIncomeAndExpenditureData':"IESurvey"}, axis = 1, inplace = True)


# In[ ]:


merged["IESurvey"]


# ### In this columns we can sprate year in IESurvey 
# Creat a pattern of year

# In[ ]:


pattern = r'(?P<First_Year>[1-2][0-9]{3})/?(?P<Second_Year>[0-9]{2})?'


# In[ ]:


df = merged["IESurvey"].str.extract(pattern)
df


# In[ ]:


merged["IESurvey"].str.extract(pattern).info()


# ### **In First  year column sprate the 20 and shift into second year column**

# In[ ]:


twos = df["First_Year"].str[:2]
twos.head()


# In[ ]:


# df["Second_Year"] = twos + df["Second_Year"]
# df["Second_Year"]


# In[ ]:


df["Second_Year"] = twos.str.cat(df["Second_Year"])


# In[ ]:


df


# In[ ]:


merged.columns


# In[ ]:


merged['IncomeGroup'].head(20)


# In[ ]:


merged['IncomeGroup'].unique()


# In IncomeGroup column I can changing  points

# In[ ]:


dic = {'High income: OECD':"HIGH OECD", 'Upper middle income':"UPPER MIDDLE", 
 'High income: nonOECD': "HIGH NONOECD",
       np.nan:np.nan, 'Lower middle income':"LOWER MIDDLE", 'Low income':"LOW"}
dic


# In[ ]:


merged['IncomeGroup'] = merged['IncomeGroup'].map(dic)


# In[ ]:


merged['IncomeGroup'].unique()


# In[ ]:


IG = merged.pivot_table(index = 'IncomeGroup', values = 'Happiness Score')
IG


# # Visualaiztion 

# In[ ]:


IG.plot.bar()

