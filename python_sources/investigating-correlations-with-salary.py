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


# In this notebook, my objective is to see if we can develop a linear predictor for data scientist salary, using the dataset provided by @stephenofarrell for Cost of Living in various cities around the world.  One of the features I took on was including salary information for data scientists, based on data available via Glassdoor.  As can be seen in the dataset, there were several cities where salaries were missing (and others were Glassdoor rates the information as low confidence).  
# 
# I wanted to see if I could use the cost of living information, along with the salaries already included to develop a linear regressor for predicting data scientist salaries to "fill in the gaps" and for other cities not included in the data set.
# 
# To start, I imported the data and set the index to the cities, to make eliminate the numerical index and replace with categorical indices.

# In[ ]:


import pandas as pd
cost_of_living_rev1_df = pd.read_csv(r"../input/revised-cost-of-living-v1/cost-of-living_rev1(25JAN2020).csv")


# In[ ]:


cost_of_living_rev1_df.set_index('Unnamed: 0', inplace=True)
cost_of_living_rev1_df.head()


# Now, the dataset is column oriented by city, which is the field we are trying to backfill.  So, let's transpose the dataframe and inspect.

# In[ ]:


trans_col_df = cost_of_living_rev1_df.T
trans_col_df.head()


# Alright, now we've got a dataset ready for investigation.  We'll start by plotting the data correlations in sets of five.

# In[ ]:


import matplotlib
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.style.use('fivethirtyeight')
import seaborn as sns

row_labels = list(trans_col_df.index)
cols = list(trans_col_df.columns)[:-1]
sal_data = list(trans_col_df['Avg Data Scientist Salary (USD/annum)'])
i=0
while i < len(cols)-5:
    corr_data_1 = list(trans_col_df[cols[i]])
    corr_data_2 = list(trans_col_df[cols[i+1]])
    corr_data_3 = list(trans_col_df[cols[i+2]])
    corr_data_4 = list(trans_col_df[cols[i+3]])
    corr_data_5 = list(trans_col_df[cols[i+4]])
    corr_df = pd.DataFrame(zip(corr_data_1,corr_data_2,corr_data_3,corr_data_4,corr_data_5,sal_data), columns = [str(cols[i]),str(cols[i+1]),str(cols[i+2]),str(cols[i+3]),str(cols[i+4]),str('annual salary')])
    sns.pairplot(corr_df)
    plt.show()
    
    i+=5


# Next, for ease of building the model, let's print the column titles.

# In[ ]:


cols


# Last, we'll build our regressor and data model.  For this, I used Bread, Rice, and 1 Bedroom Apartment as features, against the salary.  Note that we could have selected any of the apartment prices, but should only choose one, as they are all highly correlated.  As an interesting history aside, this approach is used by militaries around the world to monitor the stability of the local economy when they are conducting operations. 

# In[ ]:


from sklearn.linear_model import LinearRegression
fit_df = pd.DataFrame(zip(trans_col_df['Loaf of Fresh White Bread (500g)'],                          trans_col_df['Rice (white), (1kg)'],                          trans_col_df['Apartment (1 bedroom) in City Centre'],                          sal_data),                      columns = ['bread_price','rice_price','apartment_price','salary'])
fit_df.dropna(inplace=True)
X = fit_df[['bread_price','rice_price','apartment_price']]
y = fit_df['salary']
reg = LinearRegression(fit_intercept=False).fit(X,y)
print(reg.score(X,y), reg.coef_, reg.intercept_)


# So, we have a model with a score around 0.695, which isn't too bad for using bread, rice, and single bed apartment prices.  Can you make it better?
