#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


james = pd.read_csv('../input/lebron_career.csv')


# In[ ]:


james.info()


# In[ ]:


james.head(5)


# Some modifications to prepare the data

# In[ ]:


def Prepare_james(df):
    #Convert the date column into the datetime column (from object column)
    df['date'] = pd.to_datetime(df['date'])
    #Extracting year, month, and the day of the play date from the column
    df['play_year'], df['play_month'], df['play_day'] = df['date'].dt.year, df['date'].dt.month, df['date'].dt.day
    
    #Separate the age column with age and age subdays column, and combine them to make real-time age
    df[['age','age_subdays']]=df['age'].str.split('-',expand=True).replace(np.nan, 0).astype(int)
    df['age'] = df['age'] + df['age_subdays'] / 365
    
    #Deal with the minutes player column
    df['mp'] = pd.to_datetime(df['mp'], format = '%M:%S').dt.minute
    
    #Calculating overall shooting percentage
    df['overall_pct'] = (df['fg'] + df['three'] + df['ft']) / (df['fga'] + df['threeatt'] + df['fta'])
    
    #game rating from: https://www.basketball-reference.com/about/glossary.html#pf
    df['game_rating'] = 0.7 * df['orb'] + 0.3 * df['drb'] + 0.7 * df['ast'] + 0.7 * df['blk'] + df['stl'] - df['tov']
    
    return df


# Process the data

# In[ ]:


Prepare_james(james)


# Now visualizing the data

# In[ ]:


fig, ax = plt.subplots(figsize = (8, 6))
ax.plot(james.groupby(['age'])['pts'].mean())
plt.xlabel('Age')
plt.ylabel('Points per game')
plt.title('Lebron points per game by age')


# Checking distribution of pts

# In[ ]:


sns.distplot(james.pts)


# relatively normal which is good for prediction and does not require transformation

# In[ ]:


sns.jointplot(x = 'game_rating', y = 'pts', data = james, height = 8, ratio=4, color = "r")
plt.show()


# In[ ]:


sns.jointplot(x = 'threep', y = 'pts', data = james, height = 8, ratio=4, color = "b")
plt.show()


# In[ ]:


sns.jointplot(x = 'mp', y = 'pts', data = james, height = 8, ratio=4, color = "g")
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize = (8, 6))
ax.plot(james.groupby(['play_year'])['pts'].mean())
plt.xlabel('Year')
plt.ylabel('Points per game')
plt.title('Lebron points per game by year')


# In[ ]:


fig, ax = plt.subplots(figsize = (16, 6))
mean_pts = james['pts'].mean()
james['mean_row'] = mean_pts
ax.scatter(james['opp'], james['pts'])
mean_line = ax.plot(james['opp'], james['mean_row'] , label='Mean', color = 'r', linestyle='-')
ax.legend()
plt.title("Lebron Points vs each team in the league")


# In[ ]:


james_pred = james[['pts','mp', 'age', 'overall_pct', 'game_rating', 'minus_plus']]

colormap = plt.cm.RdBu
plt.figure(figsize=(14, 12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(james_pred.corr(), linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


# 

# In[ ]:





# In[ ]:




