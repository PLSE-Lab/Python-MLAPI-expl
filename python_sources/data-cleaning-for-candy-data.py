#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Part 1 - Data Cleaning

# ## 1 Importing packages

# In[ ]:


import numpy as np
import pandas as pd


# ## 2 Read file and explore

# About Data:
# 
# * Data is about a 12 questions survey for candies during Halloween.
# * Q1 is if the surveyee is going to trick or treat himself
# * Q6 is specific to a type of candy and how the surveyee finds it (JOY, DESPAIT, NEUTRAL)
# * Q7 is other candy thats gives JOY to surveyee not mentioned in Q6 list
# * Q8 is other candy thats gives DESPAIR to surveyee not mentioned in Q6 list
# * Q10 is what color of the dress does the surveyee sees in first glance (black-blue OR white/golden)
# * Q11 is what day surveyee likes? Friday OR Sunday
# * Q12 is a Media question of a website that the suveyee would most likely check followed by co-ordinates of click.
# 
# *The data is taken from https://www.scq.ubc.ca/so-much-candy-data-seriously/*

# In[ ]:


candy = pd.read_excel('/kaggle/input/candy-data/candyhierarchy2017.xlsx')


# In[ ]:


candy.head()


# ## 3 Partial columns renaming

# In[ ]:


candy.columns


# In[ ]:


candy = candy.rename(columns = {'Q1: GOING OUT?' : 'going_out', 'Q2: GENDER' : 'gender', 'Q3: AGE': 'age', 'Q4: COUNTRY' : 'country',
       'Q5: STATE, PROVINCE, COUNTY, ETC' : 'area', 'Q10: DRESS' : 'dress', 'Q11: DAY': 'day',
       'Q12: MEDIA [Daily Dish]' : 'media_DailyDish', 'Q12: MEDIA [Science]': 'media_Science', 'Q12: MEDIA [ESPN]' : 'media_ESPN',
       'Q12: MEDIA [Yahoo]': 'media_Yahoo'})


# In[ ]:


candy.columns


# In[ ]:


candy['Unnamed: 113'].unique()


# In[ ]:


candy.drop(columns = ['Internal ID','Unnamed: 113', 'Click Coordinates (x, y)'], inplace = True)


# In[ ]:


candy.shape


# ## 4 Handling null values

# I found a better way of handling nulls later. (Patched that code and commenting this one out)
# 
# '''
# candy.shape
# 
# candy = candy.dropna(how = 'all')
# 
# candy = candy.dropna(thresh = 5, axis = 0) # I want at least 5 non-NaN values.
# 
# candy.drop(['Unnamed: 113','Click Coordinates (x, y)'], axis = 1, inplace = True)
# 
# '''

# In[ ]:


candy.dropna(subset = ['going_out', 'gender', 'age', 'country', 'area'], how = 'all', inplace = True)
candy.reset_index(drop = True, inplace = True)


# In[ ]:


candy.shape


# ## 5 Formating columns

# ### going out column

# In[ ]:


candy.going_out = candy.going_out.fillna('Not Sure')
candy.going_out.unique()


# ### gender column

# In[ ]:


candy.gender.value_counts()


# In[ ]:


# Adding 11 NaN genders to type 3 - I'd rather not say seems to be the closest to unknown
candy[candy.gender == "I'd rather not say"].shape  #checking for spaces in text - found none


# In[ ]:


candy.gender = candy.gender.fillna("I'd rather not say")
candy.gender.value_counts()


# ### country column

# In[ ]:


candy.country.unique()


# People are so creative !!

# In[ ]:


candy.country.value_counts(dropna = False).sort_values(ascending = False)


# In[ ]:


candy.country.isna().sum()


# In[ ]:


candy.country = candy.country.fillna('Unknown')


# In[ ]:


set([x for x in candy.country if 'u' in str(x)])  # unique values with 'u'


# In[ ]:


USA = [x for x in candy.country if (('u' in str(x) or 'U' in str(x)) and 'ingdom' not in str(x)     and 'urope' not in str(x) and 'stralia' not in str(x) and 'South Korea' not in str(x) and 'South africa' not in str(x) and 'uk' not in str(x))]


# In[ ]:


candy.country = candy.country.replace(to_replace = USA, value = 'USA')


# In[ ]:


candy.country.unique()


# In[ ]:


candy.country = candy.country.replace(to_replace = ['america','Ahem....Amerca',"'merica",'North Carolina ','cascadia',                                                   'New York','A','California','New Jersey','America','Alaska',                                                    'N. America'], value = 'USA')


# In[ ]:


canada = [x for x in candy.country if 'anada' in str(x).strip() or 'ANADA' in str(x) or 'Can' in str(x)]


# In[ ]:


candy.country = candy.country.replace(to_replace = canada, value = 'Canada')


# In[ ]:


candy.country.value_counts()


# In[ ]:


other = [x for x in candy.country.unique()]


# In[ ]:


other.remove('USA')
other.remove('Canada')


# In[ ]:


other


# In[ ]:


candy.country = candy.country.replace(to_replace = other, value = 'Other')


# In[ ]:


candy.country.value_counts()


# Country column got messy and I got so less done :( I am going to ignore area column, will check it out if my analysis wants me to

# In[ ]:


candy.columns


# ## 6 Datatype conversion

# In[ ]:


candy = candy.astype({'going_out':'category', 'gender':'category', 'country':'category', 'dress':'category', 'day':'category'})


# In[ ]:


candy.describe(include = 'category')


# ## 7 Wrting a function to convert 4 columns into 1 column

# In[ ]:


def melt1(row):
    for c in data.columns:
        if row[c] == 1:
            return c


# In[ ]:


data = candy[candy.columns[-4:]]


# In[ ]:


data


# In[ ]:


new_col = data.apply(melt1, axis = 1)


# In[ ]:


candy['media_preference'] = new_col


# In[ ]:


candy.drop(columns = ['media_DailyDish','media_Science','media_ESPN','media_Yahoo'], inplace = True)


# In[ ]:


candy.media_preference.value_counts(dropna = False)


# In[ ]:


#Dividing questions and other columns
'''
candy_options = [i for i in candy.columns if 'Q6' in i or 'Q7' in i or 'Q8' in i or 'Q9' in i]
other_columns = [i for i in candy.columns if 'Q6' not in i and 'Q7' not in i and 'Q8' not in i and 'Q9' not in i]
'''

personal_info_cols = candy.columns[:6]
questionare_cols = candy.columns[5:]


# In[ ]:


candy.columns


# In[ ]:


responses = len(questionare_cols) - candy[questionare_cols].isna().sum(axis = 1)


# In[ ]:


candy['responses'] = responses


# In[ ]:


candy.head(3)


# This concludes the data cleaning part of the dataset

# # Part 2 - EDA

# In[ ]:


candy_questions = [x for x in candy.columns if 'Q6' in str(x)]


# In[ ]:


candy_questions


# In[ ]:


data = pd.DataFrame(candy[candy_questions])


# In[ ]:


data.shape


# In[ ]:


re = ['type_'+ str(x) for x in range(1,104)]

dic = {}
for i in range(len(data.columns)):
    dic[data.columns[i]] = re[i]


# In[ ]:


candy = candy.rename(columns = dic)
data = data.rename(columns = dic)


# ## Delete rows that have all NaNs

# In[ ]:


data = data.dropna(axis = 0, how = 'all')
data = data.reset_index(drop = True)


# In[ ]:


data.shape


# In[ ]:


data.head(4)


# In[ ]:


d = data.melt()


# In[ ]:


d.head(5)


# In[ ]:


import seaborn as sns

sns.countplot(data = d[:4000], x = 'variable', hue = 'value')


# I want help to convert d to a form:
# 
# 
# |Response| MEH  | JOY  | DESPAIR|
# |------|------|------|--------|
# |type_1|count |count |count   |
# |type_2|count |count |count   |
# |type_3|count |count |count   |
