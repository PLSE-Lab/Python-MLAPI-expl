#!/usr/bin/env python
# coding: utf-8

# # Data Scientists in Poland
# ## 1. Imports

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', None) 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## 2. Read data and show column descriptions

# In[ ]:


df = pd.read_csv('/kaggle/input/kaggle-survey-2018/multipleChoiceResponses.csv')
df[:2]


# ## 3. Code for simplifying DataFrame and extracting significant data about Data Scientists
# 
# The following columns are extracted here:
# - `education_degree` - the highest level of education,
# - `current_role` - the current role of a person,
# - `experience_years` - years of experience in the current role,
# - `salary_range_usd` - current yearly compensation in USD,
# - `apply_ml_in_new_areas` - True if a person applies Machine Learning in new areas,
# - `do_research_in_ml` - True if a person is doing a research in Machine Learning,
# - `lower_bound_salary_usd` - lower bound of current yearly compensation in USD,
# - `upper_bound_salary_usd` - upper bound of current yearly compensation in USD.

# In[ ]:


def salary_range_str_to_ranges(salary):
    if '-' not in str(salary):
        return None, None
    lower, upper = salary.split(',')[0].split('-')
    return int(lower) * 1000, int(upper) * 1000

def simplify_data_scientist_data(df):
    df = df[['Q4', 'Q6', 'Q8', 'Q9', 'Q11_Part_4', 'Q11_Part_5', 'Q16_Part_1']].rename(columns=dict(
        Q4='education_degree', 
        Q6='current_role', 
        Q8='experience_years', 
        Q9='salary_range_usd', 
        Q11_Part_4='apply_ml_in_new_areas', 
        Q11_Part_5='do_research_in_ml', 
        Q16_Part_1='uses_python'
    ))
    df['lower_bound_salary_usd'], df['upper_bound_salary_usd'] = zip(*df['salary_range_usd'].map(salary_range_str_to_ranges))
    df['lower_bound_salary_usd'] = df['lower_bound_salary_usd'].astype(float)
    df['upper_bound_salary_usd'] = df['upper_bound_salary_usd'].astype(float)
    df['apply_ml_in_new_areas'] = df['apply_ml_in_new_areas'].map(pd.notnull)
    df['do_research_in_ml'] = df['do_research_in_ml'].map(pd.notnull)
    df = df[pd.notnull(df.lower_bound_salary_usd)]
    return df[df.current_role == 'Data Scientist']


# ## 4. Read data about Data Scientist from Poland, US and the whole world

# In[ ]:


pl_df = simplify_data_scientist_data(df[df.Q3 == 'Poland'])
pl_df


# In[ ]:


us_df = simplify_data_scientist_data(df[df.Q3 == 'United States of America'])
us_df


# In[ ]:


world_df = simplify_data_scientist_data(df)
world_df


# ## 5. Check salary bounds for Data Scientist in Poland with experience in ranges 4-5, 5-10 years
# 
# We take here into account people which are working on ML (not only on data analysis) and they are applying ML in a new areas or/and doing research in ML.
# 
# The range for salary is 40-70K USD, but we have only for 3 responses, so it's hard to derive any salary distribution based on it:

# In[ ]:


pl_df[
    (pl_df.experience_years == '4-5') | (pl_df.experience_years == '5-10')
]


# Because of a small coverage for Poland, we are going to check salaries in US where there are 118 responses with such experience:

# In[ ]:


len(us_df[
    (us_df.apply_ml_in_new_areas == True) | (us_df.do_research_in_ml == True)
][
    (us_df.experience_years == '4-5') | (us_df.experience_years == '5-10')
])


# The salary ranges in US for DS with such experience is 132-167K USD. 

# In[ ]:


us_df[
        (us_df.apply_ml_in_new_areas == True) | (us_df.do_research_in_ml == True)
    ][
        (us_df.experience_years == '4-5') | (us_df.experience_years == '5-10')
    ][['lower_bound_salary_usd', 'upper_bound_salary_usd']].mean()


# Taking into account the cost of living indexes for US and Poland (100 vs 50.9 respectively), we can derive that DS with the same experience in Poland may expect 67-85K.

# ## 6. Check distributions of experience in Poland and in the world
# 
# According to the below chart it seems there are less people in Poland which have at least 4 years of experience as Data Scientist.

# In[ ]:


pd_stats = pd.DataFrame(dict(
    poland=pl_df.groupby('experience_years').size(),
    world=world_df.groupby('experience_years').size(),
)).fillna(0)
pd_stats = pd_stats.iloc[pd_stats.index.str.extract('(\d+)', expand=False).astype(int).argsort()]
(pd_stats / pd_stats.sum(axis=0) * 100).plot(kind='bar')

