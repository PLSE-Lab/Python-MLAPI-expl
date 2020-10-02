#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import minmax_scale
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# **Preparing dataset**

# In[ ]:


attendance_valuation_elo_df = pd.read_csv("../input/nba_2017_att_val_elo.csv")
attendance_valuation_elo_df.head()


# In[ ]:


salary_df = pd.read_csv("../input/nba_2017_salary.csv")
salary_df.rename(columns={'NAME': 'PLAYER'}, inplace=True)
salary_df = salary_df.merge(attendance_valuation_elo_df, how = 'left', on = 'TEAM')
salary_df = salary_df.drop(['Unnamed: 0',], axis = 1)
salary_df.head()


# In[ ]:


pie_df = pd.read_csv("../input/nba_2017_pie.csv")
pie_df = pie_df[['PLAYER','AGE','GP','W','L']]
pie_df['winning_rate'] = pie_df['W'] / pie_df['GP']
pie_df = pie_df[['PLAYER','AGE','winning_rate']]
pie_df.head()


# In[ ]:


nba_2017_br = pd.read_csv("../input/nba_2017_br.csv")
nba_2017_br = nba_2017_br[['Player','Pos','MP']]
nba_2017_br = nba_2017_br.rename(columns = {'Player':'PLAYER'})
nba_2017_br = nba_2017_br.merge(pie_df, how = 'inner', on = 'PLAYER')
nba_2017_br = nba_2017_br.merge(salary_df, how = 'inner', on = 'PLAYER')
nba_2017_br.head()


# In[ ]:


nba_2017_twitter_players = pd.read_csv("../input/nba_2017_twitter_players.csv")
nba_2017_br = nba_2017_br.merge(nba_2017_twitter_players, how = 'inner', on = 'PLAYER')
nba_2017_br.head()


# **Correlation heatmap**

# In[ ]:


plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("NBA Player Correlation Heatmap")
corr = nba_2017_br.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,cmap="Greens")


# **Prepossing the data**

# In[ ]:


#Remove missing values
nba_2017_br = nba_2017_br.dropna()

cat_cols = ['PLAYER','Pos','TEAM','CONF','POSITION']

# Rescale numetic columns(Expecting target column i.e. winning_rate)
num_cols = list(nba_2017_br.drop(cat_cols,axis = 1).columns.drop('winning_rate'))
for col in num_cols:
    nba_2017_br[col + 'scaled'] = minmax_scale(nba_2017_br[col])
    nba_2017_br = nba_2017_br.drop(col,axis = 1)
    
    
#Create dummy variables
for col in cat_cols:
    dummies = pd.get_dummies(nba_2017_br[col], prefix = col)
    nba_2017_br = pd.concat([nba_2017_br,dummies], axis = 1)
    nba_2017_br = nba_2017_br.drop(col,axis = 1)

nba_2017_br.head()


# **Assign dependent and independent variables**

# In[ ]:


all_X = nba_2017_br.drop('winning_rate', axis = 1)
y = nba_2017_br['winning_rate']


# **Apply LinearRegression Model**

# In[ ]:


lr = LinearRegression()
results = lr.fit(all_X,y)
predictions = lr.predict(nba_2017_br[all_X.columns])
mse = mean_squared_error(predictions, nba_2017_br['winning_rate'])
mse


# In[ ]:


model = sm.OLS(y, all_X).fit()
model.summary()


# Mean_sqaured_error is slim to none and R-squared = 1.000 prove the linear regression model is a fit in this case.

# In[ ]:




