#!/usr/bin/env python
# coding: utf-8

# <br><br><center><h1 style="font-size:4em;color:#2467C0">Soccer Data Analysis</h1></center>
# <br>

# - All the details about dataset -<br>
# - +25,000 matches<br>
# - +10,000 players<br>
# - 11 European Countries with their lead championship<br>
# - Seasons 2008 to 2016<br>
# - Players and Teams' attributes* sourced from EA Sports' FIFA video game series, including the weekly updates<br>
# - Team line up with squad formation (X, Y coordinates)<br>
# - Betting odds from up to 10 providers<br>
# - Detailed match events (goal types, possession, corner, cross, fouls, cards etc...) for +10,000 matches<br>

# ### Problem Statement:
# ## <center><font color=green>Find 10 most Important Skills/feature that contribute to the overall rating of the Player ?</font></center>

# ### Importing Libraries

# In[ ]:


import sqlite3 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ### Reading Dataset
# Our data is in the form of SQL database. We need to extract the required data from SQLite Database. For this we are using sqlite package from python

# In[ ]:


c = sqlite3.connect('../input/database.sqlite')
data = pd.read_sql_query("SELECT * FROM Player_Attributes",c)


# ### Exploring Dataset
# 
# Now lets explore our data.<br>
# Lets get some basic information about data

# In[ ]:


# Checking first five columns of the dataset
data.head()


# In[ ]:


# Total No of Columns
data.columns


# In[ ]:


len(data.columns)


# Our dataset has 42 columns.<br>
# lets check the desciption of all numerical values in the dataset

# In[ ]:


data.describe().T


# In[ ]:


# Check for the NULL values in the dataset
data.isnull().sum()


# - As we can see only 4 columns in the dataset don't have NULL values

# ### Deleting NULL records from the dataset

# In[ ]:


# Number of rows before deleting 
original_rows = data.shape[0]
print("Number of rows befor deletion: {}".format(original_rows))

# Deleting rows with NULL values
data = data.dropna()

#Number of rows after deletion
new_rows = data.shape[0]
print("Number of rows befor deletion: {}".format(new_rows))


# In[ ]:


data.dtypes


# - Only four feaures are categorical features
# - We will try to convert them into numerical values

# In[ ]:


object_dtype = list(data.select_dtypes(include=['object']).columns)


# In[ ]:


print(object_dtype)


# In[ ]:


# Lets try to check correlation between different features
corr_data = data.corr()


# In[ ]:


# Plotting these values using seaborns Heatmap
plt.figure(figsize=(12,10))
sns.heatmap(corr_data,cmap='viridis')


# In[ ]:


plt.figure(figsize=(12,12),dpi=100)
sns.clustermap(corr_data,cmap='viridis')


# - From this Clustermap we can the clustering of different features which are highly correlated to each other
# - Now we want to check the correlation of these features with Overall Rating of the player

# In[ ]:


#Extracting Highly Related features to Overall Rating
cols = corr_data.nlargest(11,columns='overall_rating')['overall_rating'].index
corr_matrix2 = np.corrcoef(data[cols].values.T)
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix2, cbar=True,cmap='viridis', annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)


# <br><br><center><h4 style="font-size:3em;color:#800000">Outcome</h4></center>
# <br>
# 
# 
# <center><h2 style = "font-size:2em;color:#397fef">Top 10 Features that are contributing to the Overall Rating of the Player</h2></center>

# <left> <h4 style ="font-size:1.3em" >1. overall_rating<br>
# 2. reactions<br>
# 3. potential<br>
# 4. short_passing<br>
# 5. ball_control<br>
# 6. long_passing<br>
# 7. vision<br>
# 8. shot_power<br>
# 9. penalties<br>
# 10. long_shots<br>
# 11. positioning</h4></left>
