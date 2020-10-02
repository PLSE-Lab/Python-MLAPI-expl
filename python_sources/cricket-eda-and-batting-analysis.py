#!/usr/bin/env python
# coding: utf-8

# Importing the numpy pandas and seabord needed to create graphs.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt


# Read the CSV File for ODI on which has both batting and balling statistics

# In[ ]:


df = pd.read_csv("../input/cricinfo-statsguru-data/ODI Player Innings Stats - All Teams.csv")


# In[ ]:


df.head(5)


# All the coloums have 85984 non null objects. From this data it can be conclued that there are 50% of balling and 50% of batting rows in the CSV. Total number of rows are 171968.

# In[ ]:


df.info()


# Taking all the batting stats required in one dataframe **df_batting**

# In[ ]:


df_batting = df[['Innings Player','Innings Runs Scored Num','Innings Minutes Batted','Innings Batted Flag','Innings Balls Faced','Innings Boundary Fours','Innings Boundary Sixes','Innings Batting Strike Rate']]


# In[ ]:


df_batting.head(5)


# As we know half of the columns are batting statistics we can see there are 85984 coloumns which are blank.

# In[ ]:


df_batting.isnull().sum()


# Remove the null and the rows containing '-', And convert the str values to int to plot graphs and pairplots.

# In[ ]:


df_batting = df_batting[~df_batting['Innings Runs Scored Num'].isin(['-'])]
df_batting = df_batting[~df_batting['Innings Minutes Batted'].isin(['-'])]
df_batting = df_batting[~df_batting['Innings Batted Flag'].isin(['-'])]
df_batting = df_batting[~df_batting['Innings Balls Faced'].isin(['-'])]
df_batting = df_batting[~df_batting['Innings Boundary Fours'].isin(['-'])]
df_batting = df_batting[~df_batting['Innings Boundary Sixes'].isin(['-'])]
df_batting = df_batting[~df_batting['Innings Batting Strike Rate'].isin(['-'])]


# In[ ]:


df_batting = df_batting.dropna()


# In[ ]:


df_batting.columns.values


# In[ ]:


df_batting['Innings Runs Scored Num'] = df_batting['Innings Runs Scored Num'].astype(int)
df_batting['Innings Minutes Batted'] = df_batting['Innings Minutes Batted'].astype(int)
df_batting['Innings Balls Faced'] = df_batting['Innings Balls Faced'].astype(int)
df_batting['Innings Boundary Fours'] = df_batting['Innings Boundary Fours'].astype(int)
df_batting['Innings Boundary Sixes'] = df_batting['Innings Boundary Sixes'].astype(int)
df_batting['Innings Batting Strike Rate'] = df_batting['Innings Batting Strike Rate'].astype(float)


# In[ ]:


df_compare = df_batting[(df_batting['Innings Player'] == 'RG Sharma') | (df_batting['Innings Player'] == 'SR Tendulkar') | (df_batting['Innings Player'] == 'V Kohli')]


# In[ ]:


df_compare


# In[ ]:


df_compare.info()


# In[ ]:


df_compare.isnull().sum()


# Below are the analysis graph of all three top players of India : Sachin, Rohit Sharma and Virat Kohli. 
# 
# Various observations can be plotted using below graphs :
# 
# -- Heatmap shows the correlation between various coloums for all the three batsmen. Out of which Balls faced is highly corelated to Innings Runs Scored Num and Innings balls Faced. We can use features like Balls faced, with fours and sixes to predict the number of runs. We will try and apply linear regression algorithm to predict the score and get the runs and check the accuracy of the model.
# 
# Below heatmap we have the pairplot which shows the number of features on scatter plot. We can see many of All of the features are positively skewed.
# 

# In[ ]:


import seaborn as sns
df_compare = df_compare.drop('Innings Batted Flag', axis=1)
sns.heatmap(df_compare.corr(), annot=True)


# In[ ]:


b = df_compare.drop(['Innings Balls Faced','Innings Minutes Batted'],axis=1)
sns.pairplot(b,hue='Innings Player')


# In[ ]:




