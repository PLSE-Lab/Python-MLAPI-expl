#!/usr/bin/env python
# coding: utf-8

# # Questions that I will be solving:
# 1. Which company has received the most awards overall in most of the years?
# 2. Which company has received the most awards in a single year?
# 3. Which game has received the most awards in an event?
# 4. Which country received the most awards at an event?
# 
# # Libraries

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Importing and Selecting Data

# In[ ]:


df = pd.read_csv('../input/the-game-awards/the_game_awards.csv', encoding='UTF-8')
winners = df.loc[df['winner'] == 1]


# In[ ]:


df.head(2)


# # Solving ...
# ## 1. Which company has received the most awards overall in most of the years?
# To solve that question, using winners dataframe and ignoring any NaN value present, I will look for the company that got many awards as possible.

# In[ ]:


df_winners_company = pd.DataFrame(df['company'].value_counts())
df_winners_company = df_winners_company.reset_index()
df_winners_company.columns = ['Company', 'Awards']


# To see the results, I will be plotting the first 10 companies:

# In[ ]:


sns.set(style="whitegrid")
plot1 = sns.barplot(x='Company', y='Awards', data=df_winners_company[0:9]);
plot1.set_xticklabels(plot1.get_xticklabels(), rotation=45, ha='right');


# In[ ]:


df_winners_company[0:9]


# So here you can see that **Valve** won most of the awards from 2014 to 2019.
# 
# ## 2. Which company has received the most awards in a single year?

# In[ ]:


for year in winners['year'].unique():
    # Get the data only for one year in specific
    winners_in_year = winners.loc[winners['year'] == year]
    # Get the value count for the companies in that year
    winners_companies = pd.DataFrame(winners_in_year['company'].value_counts())
    winners_companies = winners_companies.reset_index()
    winners_companies.columns = ['Company', 'Awards']
    # LOOP WITH CONDITIONAL - There is the same quantity of awards in the same year?
    for i in range(0,len(winners_companies)):
        # BREAK CONDITION - If we have one winner and the others companies has 
        # less than your awards then break break the loop
        if i != 0 and (winners_companies.loc[0,'Awards'] > winners_companies.loc[i,'Awards']):
            break
        # Print results
        print('In ',year,'the company that got most awards was ',winners_companies.loc[i,'Company'],' with ',winners_companies.loc[i,'Awards'],' awards.')


# Analyzing the text above, it is possible to see that ZA/UM is the company with more awards in a year with **4 awards in 2019**.
# 
# ## 3. Which game has received the most awards in an event?
# Using the same loop that I presented in the previous question, now for I will do the Nominee column. 
# 
# **Note**: I'm considering an event equal to a year.

# In[ ]:


for year in winners['year'].unique():
    # Get the data only for one year in specific
    winners_in_year = winners.loc[winners['year'] == year]
    # Get the value count for the companies in that year
    winners_nominee = pd.DataFrame(winners_in_year['nominee'].value_counts())
    winners_nominee = winners_nominee.reset_index()
    winners_nominee.columns = ['Nominee', 'Awards']
    # LOOP WITH CONDITIONAL - There is the same quantity of awards in the same year?
    for i in range(0,len(winners_nominee)):
        # BREAK CONDITION - If we have one winner and the others companies has less than your awards then break break the loop
        if i != 0 and (winners_nominee.loc[0,'Awards'] > winners_nominee.loc[i,'Awards']):
            break
        # Print results
        print('In ',year,'the game that got most awards was ',winners_nominee.loc[i,'Nominee'],' with ',winners_nominee.loc[i,'Awards'],' awards.')


# So by this analysis, it is possible to see that **Disco Elysium** got most awards in an event in 2019.
# 
# **NOTE**: I ignored in my code the categories related to companies/gamers most because the code itself would take out from the account.
# 
# ## 4. Which country received the most awards at an event?
# Considering that an **event** is equal to **year** and the answer for question 2, I will create a dataset specific for this question where I put **manually** the country for that specific companies, showing the answer for this question.

# In[ ]:


# Define previously the coluns manually
event = [2014,2014,2014,2015,2016,2017,2017,2018,2018,2019]
company = ['Ubisoft Montpellier','Nintendo EAD','BioWare','Nintendo','Blizzard Entertainment','StudioMDHR','Nintendo','Rockstar Games',
           'SIE Santa Monica Studio/Sony Interactive Entertainment','ZA/UM']
country = ['France','Japan','Canada','Japan','USA','Canada','Japan','USA','USA','England']
awards = [2,2,2,3,3,3,3,3,3,4]

# Create the dataframe
data = pd.DataFrame(list(zip(event,company,country,awards)),columns=['event','company','country','awards'])


# In[ ]:


data.groupby(['event', 'country']).size()


# By this line code above and previous knowledge, we can see that the only way to a country beat England is **USA in 2018** that was the only country with more than one company to win a award. Looking for the question 2 answer we can see that USA in 2018 had **6 Awards**, meaning that it is the country with more awards in an event! 
# 
# ## Conclusions
# After an analysis using *Pandas* library in Python it is possible to see that we can retrieve a lot of knowledge from a dataset before we create models or statistic analysis, meaning that it is important to know more about this library and their tools.
