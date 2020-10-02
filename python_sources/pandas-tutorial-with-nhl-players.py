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


# # Pandas: Import CSV

# Import a csv using Pandas.

# In[ ]:


player = pd.read_csv("../input/predict-nhl-player-salaries/train.csv",encoding = "ISO-8859-1").fillna(0)


# - The (encoding = "ISO-8859-1").fillna(0) aspect takes empty data fields and sets them to zero.
# - Without this, player would return an error, as it can't parse without defining the empty spaces as 3 commas.
# - Pandas parses through the csv file to determine headers. Now You are ready to utilize your dataset.

# In[ ]:


player


# To see the first or last # of values, use .head(#) or .tail(#).
# This allows you to see the first or last # of results in the DataFrame.

# In[ ]:


player.head(10)


# # Pandas: Display tools and choosing data

# Being able to diplay specific data is essential to be able to fully utilize DataFrames.

# - top10 returns the top 10 players when the dataset is sorted by Salary.
# - Since .sort_values('Salary', ascending=False) is the first function, it sorts the entire dataset.
# - Next, its organized in decsending order based on the header called 'Salary'

# In[ ]:


player = player.sort_values('Salary', ascending=False)
top10 = player.head(10)
print(top10)


# When you look at the output, the data is organized the correct way, but some important columns, such as First Name and Last Name are not visible.

# # Pandas: Indexing and locating data

# To have the Output include the Player's Name:
# - Set the desired column headers as the index, which is the Primary, and in this case a Composite Key.

# In[ ]:


player = player.set_index(['Last Name','First Name'])
print(player)


# Use the .shape function to determine the number of arguments (Players) as well as attributes (Stats).

# In[ ]:


size = player.shape
print(size)


# Use the .columns to see what all the headers are.
# Since there are so many headers, iterate with a for loop to view in the most readible format.

# In[ ]:


columns = player.columns
for i in columns:
    print(i)


# When you have a large dataset like this, you must be able to locate specific values.

# In[ ]:


print(player.loc['Kane'])


# - This returns all players with the last name Kane (2 Players).
# - If you want to only select one of them, input the desired First_Name into the index.
# - To only return vital information([0:5]), type the indecies of those columns.

# In[ ]:


print(player.loc['Kane','Patrick'][0:5])


#  If you want to select one of them, you can also alter the output by index using .iloc(#)

# In[ ]:


print(player.loc['Kane'].iloc[:1])


# # Pandas: Aggregation and manipulation of data

# When you have a dataset with numbers it's helpful to be able to aggregate data to make decisions.
# - Lets take the sum of all the Goals('G') scored by the 10 highest paid players.

# In[ ]:


scorers = player.head(10).G.sum()
print("The 10 Highest Paid Players Scored",scorers,"total goals last season")


# - To find the highest or lowest values of a column use .max() or .min()
# - To find the index of the highest or lowest value, use .idxmax() or .idxmin()
# - Applying Sort filters like these are useful in Hockey to know which player had the most or the least of a statistic.

# In[ ]:


best_scorer = player.G.idxmax()
best_score = player.G.max()
print(best_scorer,"was the best goal scorer by scoring",best_score,"goals")


# This next example finds the average Salary of players born in Canada

# In[ ]:


can_salaries = player[player.Nat == "CAN"].Salary.mean()


# - This iterates through each Player's nationality('Nat') and only selects Canadian ('CAN') players.
# - After returning all the Canadian players, next, this code looks under the 'Salary' column and returns the mean().

# In[ ]:


print("The Average Salary for a Canadian is $",can_salaries)


# Manipulating fields of datasets together can be incredibly helpful in analyzing the data.
# The .groupby() function allows the user to select an aggregator and return results based of it

# In[ ]:


draft_results = player.groupby("DftRd").G.mean()


#  This code sets the round each player was drafted in ('DrtfRD') as the aggregator,then returns the average goals ('G') scored for all players drafted in each round.

# In[ ]:


print(draft_results)


# When manipulating DataFrames, the rows and columns are constantly changing.
#  - .size() allows you to use the count of outputted rows to make deducations from the data.

# In[ ]:


nationalities = player.groupby("Nat").size()


# - nationalities returns a dataframe grouped by Nationality('Nat').
# - the output is a count of each player in their grouped nationalities.

# In[ ]:


print(nationalities)


# - nationalities would be a helpful data piece for the NHL to show their diveristy.
# - Insead of raw numbers, a pie chart would represent this data well in a readable format.
# - This makes the data useable and presentable.

# In[ ]:


nationalities.plot.pie()

