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


import pandas as pd
import numpy as np


# **Load Fighter details**

# In[ ]:


fighter_details = pd.read_csv("//kaggle//input//ufcdata//raw_fighter_details.csv")


# In[ ]:


fighter_details.head()


# **Load Fight details**

# In[ ]:


fight_data = pd.read_csv("//kaggle//input//ufcdata//preprocessed_data.csv")


# In[ ]:


fight_data.head()


# **To display all columns =. In the baove output some columns are replaced by .... This can be avoided by sing the beow setting max_columns, similarly for rows max_rows.**

# In[ ]:


pd.set_option('display.max_columns',None)


# In[ ]:


fight_data.head()


# **The above two dataframes give data separately about fighters and fights. Here I'm trying to get the complete data that is in association with both the above dataframes.**

# In[ ]:


ufc_data = pd.read_csv("//kaggle//input//ufcdata//data.csv")


# In[ ]:


ufc_data.head()


# **List down the datatypes of all the columns**

# In[ ]:


ufc_data.dtypes


# **Set max row dispay to none, to see all the rows.**

# In[ ]:


pd.set_option('display.max_rows',None)


# In[ ]:


ufc_data.dtypes


# According to the above output, the data is of type object (str), bool, and float64.

# Now lets get all the column names to get more insight into the dataframe.

# In[ ]:


ufc_data.columns.tolist()


# The above output gives the details about location where the fight is held, winner of the fight, title bout, the weight class to whihc the competition belongs to and number of rounds in the competition. The players are categorized into red and blue, the features of players categorized as red, is mentioned with R_ in column name and those with blue is mentioned in B_ column name.

# In[ ]:


ufc_data.index


# From the above output, the index of the data is plain numbers.There is no identifier as game Id.

# Just trying to play around the pandas functions.

# In[ ]:


ufc_data.to_numpy()


# In[ ]:


ufc_data.size


# In[ ]:


ufc_data.describe()


# Describe functions gives an overview of the data by giving the parameters like mean, count, std, min, ma and other percentiles. However, it performs calculation only on numerical data by default, we have to explicitly say 'all' to include the analysis on categorical values.

# In[ ]:


ufc_data.describe(include='all')


# The operations like mean, min, max etc.can't be applied to object data type, hence it shown NaN. Another point to notice is that the above calculations exclude NaN, we have to explicitly include if we want it to access even NaNs.

# **Below are the list of question I'm trying to answer.**

# 1. Find out the fighter who won highest number of fights including both red and blue category.

# In[ ]:


fight_winner_df = ufc_data[['R_fighter','B_fighter','Winner']]


# In[ ]:


fight_winner_df.head()


# Number of games won by red category

# In[ ]:


len(fight_winner_df[fight_winner_df['Winner']=='Red'])


# Number of games won by blue category

# In[ ]:


len(fight_winner_df[fight_winner_df['Winner']=='Blue'])


# Overall, the players in red category won more games than those in blue category.

# Find out the player who won more games in each category

# In[ ]:


fight_winner_df[fight_winner_df['Winner']=='Red']['R_fighter'].value_counts()


# Georges St-Pierre won more number of games in Red category i.e., 19.

# In[ ]:


fight_winner_df[fight_winner_df['Winner']=='Red']['B_fighter'].value_counts()


# Jeremy Stephens won highest number of games in Blue category players and the number of games is 12.

# 2. The weight class in which red won more games that blue and viceversa.

# In[ ]:


ufc_data.head()


# In[ ]:


weight_classes = ufc_data['weight_class'].unique()


# In[ ]:


weight_classes


# In[ ]:


weight_class_winner = ufc_data[['weight_class','Winner']]


# In[ ]:


weight_class_winner.head()


# All different weight classes.

# In[ ]:


weight_class_winner['weight_class'].unique()
#len(weight_class_winner.groupby(['weight_class','Winner']))


# Using isin to check how many games happened in a weigh_class Bantamweight

# In[ ]:


len(weight_class_winner[weight_class_winner['weight_class'].isin(['Bantamweight'])])


# In[ ]:


len(weight_class_winner)


# **Checking the conditions if winner is red and weight class is Bantamweight**

# In[ ]:


weight_class_winner[((weight_class_winner['weight_class']=='Bantamweight') & (weight_class_winner['Winner'] == 'Red'))].count()


# **Get the details of games won by Red category players.**

# In[ ]:


weight_class_red=weight_class_winner[weight_class_winner['Winner'] == 'Red']


# **Get the details of games won by Blue category players.**

# In[ ]:


weight_class_blue=weight_class_winner[weight_class_winner['Winner'] == 'Blue']


# **Performing group by weight_class to get the count of games won by a particular team.**

# In[ ]:


weight_class_red = weight_class_red.groupby('weight_class',as_index=False).count()


# In[ ]:


weight_class_red.head()


# In[ ]:


weight_class_red.rename(columns={'Winner':'win_count'},inplace=True)


# In[ ]:


weight_class_red


# In[ ]:


weight_class_blue


# In[ ]:


weight_class_blue = weight_class_blue.groupby('weight_class',as_index=False).count()


# In[ ]:


weight_class_blue


# In[ ]:


weight_class_blue.rename(columns={'Winner':'win_count'},inplace=True)


# **Comparing the number of matches won by each team in all the categories.**

# In[ ]:


weight_class_winners = pd.DataFrame(columns =['weight_class','winner'])
for weight_class in weight_classes:
    red_count = weight_class_red[(weight_class_red['weight_class'] == weight_class)].win_count.values
    blue_count = weight_class_blue[(weight_class_blue['weight_class'] == weight_class)].win_count.values
    winner = ""
    if red_count > blue_count:
        winner = 'Red'
    else:
        winner = 'Blue'
    weight_class_winners.loc[len(weight_class_winners)] = [weight_class,winner]


# In[ ]:


weight_class_winners


# ****The conclusion that can be drawn from this is that, overall red performs better than blue****

# In[ ]:




