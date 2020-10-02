#!/usr/bin/env python
# coding: utf-8

# In[6]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Explanatory Data Analysis on kiva data
# In the first part, I briefly analyzed the loan amount by coutry and by sector and found a large gap between countries with higher loans and lower loans. Assuming that the amount of loans will have relationship with poverty level, the second part follows analysis on loan amount in disaggregated level.
# 
# 0. Setup the data
# 
# 1. EDA on kiva_loans.csv
# 
#     1. Number of loans by country
#     2. Top 5 countries and bottom 5 countries on mean of loan amount
#     3. Number of loans by sector
#     
# 2. Analysis on loan amount by disaggregating into 4 levels
# 
#     1. 4 levels of loan amount
#     2. Countries with higher rate of level 1 (smallest) loans
#     3. Countries with higher rate of level 4 (laergest) loans
#     4. Correlation between the levels of loan amount and MPI
#     
# This is my first kernel. Any comments are appreciated!
# 

# ### 1 . Setup the data
# Import libraries and data

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.cm as cm


# In[7]:


df = pd.read_csv('../input/kiva_loans.csv')
df.head(3)


# ### 2. EDA on kiva_loans.csv

# #### A. Number of loans by country
# - There are 87 countries
# - Top 5 countries share 50 % of the total number of loans

# In[23]:


country = pd.DataFrame(df.country.value_counts())
aa = country.iloc[:10, :]
sss = aa.country
orders = np.argsort(sss)

fig = plt.figure(figsize = (15, 6))

plt.barh(range(len(orders)), sss[orders], color='skyblue', align='center')
plt.yticks(range(len(orders)), aa.index[orders])
plt.title('Top 10 countries - Number of loans')
plt.show()


# In[18]:


labels = country.index.tolist()
sizes = country.country.tolist()
cmap = plt.cm.BuPu
colors = cmap(np.linspace(0., 1., len(labels)))
fig = plt.figure(figsize=(10,10))

plt.pie(sizes,  labels=labels, colors = colors, autopct='%1.1f%%', startangle=360)
plt.title('Number of loans by coutny - %')
plt.axis('equal')
plt.show()


# #### B. Top 5 countries and bottom 5 countries on mean of loan amount
# - While most of top 5 countries have mean higher than 10,000 USD, the mean of bottom 5 countries are all below 500 USD

# In[19]:


# Top 5 countries
df.groupby(['country'])['loan_amount'].mean().sort_values().tail()


# In[20]:


# Bottom 5 countries
df.groupby(['country'])['loan_amount'].mean().sort_values().head()


# #### C. Number of loans by sector
# - There are 15 types of sectors
# - Top 3 sectors are Agriculture, Food, and Retail

# In[25]:


sector = pd.DataFrame(df.sector.value_counts())
sector.plot(kind = 'barh', color = 'skyblue', figsize = (15, 6))
plt.title('Number of loans by sector')
plt.show()


# ### 2. Analysis on loan amount by disaggregating loan amount into 4 levels

# #### A. 4 levels of loan amount
# - Loan amount is devided into 4 levels by the value of each quantile of loan amount
#     - level1: below 275 USD
#     - level2: 276 to 500 USD
#     - level3: 501 to 1000 USD
#     - level4: above 1000 USD

# In[27]:


# Determine the loan level by values of each quantile
df.loan_amount.describe()


# In[28]:


# Create levels(%)
def newvar(df):
    if df.loan_amount <= 275:
        var = 'level1'
    elif df.loan_amount > 275 and df.loan_amount <= 500 :
        var = 'level2'
    elif df.loan_amount > 500 and df.loan_amount <= 1000 :
        var = 'level3'
    elif df.loan_amount > 1000:
        var = 'level4'
    return var
df['loan_amount_level'] = df.apply(newvar, axis=1)
df2 = df.groupby(['country', 'loan_amount_level'])['id'].agg('count').unstack()
df2['sum'] = df2.sum(axis=1)
df2['lev1'] = (df2['level1']/df2['sum'])*100
df2['lev2'] = (df2['level2']/df2['sum'])*100
df2['lev3'] = (df2['level3']/df2['sum'])*100
df2['lev4'] = (df2['level4']/df2['sum'])*100
df3 = df2.fillna(0)
df4 = df3.iloc[:, 5:]
df4.head()


# #### B. Countries with higher rate of level1 (less than 275 USD) loans
# The pink bar shows the rate of level 1 (less than 275 USD), the smallset size of loans among total number of loans by country. The bar chart describes that Nigeria, Togo, Madagascar, Liberia, Phillipines, Cambodia, and SouthSudan are the countries mostly borrowing the smallest size of loans.

# In[34]:


df5 = df4.sort_values(by = ['lev1'])
df5.plot(kind = 'bar', stacked = True, colormap = cm.Pastel1, figsize = (23, 8))
plt.title('Rate of loan size by country - pink is the rate of the smallest size of loan')
plt.show()


# #### C. Countries with higher rate of level 4 (more than 1000 USD) loans
# The grey bar shows the rate of level 4 (more than 1000 USD), the largest size of loans to the total number of loans by country. The bar chart shows that some countries only borrow level 4 loans. 

# In[33]:


df6 = df4.sort_values(by = ['lev4'], ascending=False)
df6.plot(kind = 'bar', stacked = True, colormap = cm.Pastel1, figsize = (23, 8))
plt.title('Rate of loan size by country - grey is the rate of the largest size of loan')
plt.show()


# #### D. Correlation between the levels of loan amount and MPI
# Let's take the mean MPI by country and find the correlation between level 1 and level 4 loans. The hypothesis is the higher the MPI is, the more level 1 loan the country has. Also the lower the MPI is, the more the level 4 loan the country has. 

# In[36]:


mpi = pd.read_csv('../input/kiva_mpi_region_locations.csv')
mpi2 = pd.DataFrame(mpi.groupby(['country'])['MPI'].agg('mean'))
new = pd.merge(df4, mpi2, left_index = True, right_index = True)


# #### Correlation between mean MPI and the rate of leel 1 laon
# The chart shows weak positive correlation between mean MPI and the rate of level 1 loan

# In[41]:


sns.regplot(x = new.lev1, y = new.MPI, fit_reg = True)
plt.xlabel('Rate of level 1 loan')
plt.ylabel('Mean MPI')
plt.show()


# - Let's check the siginificance of the slope of the chart!
# - The anaysis shows that the coefficient is siginificant in 2.5%. 

# In[44]:


new2 = new.dropna()
x = new2.lev1.reshape(-1, 1)
y = new2.MPI


# In[45]:


from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats


X2 = sm.add_constant(x)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())


# #### Correlation between mean MPI and the rate of level 4 loan
# The chart shows very weak neative correaltion between two, however, the linear regression result shows that there is no significance in the slope of the line. 
# 

# In[42]:


sns.regplot(x = new.lev4, y = new.MPI, fit_reg = True)
plt.xlabel('Rate of level 4 loan')
plt.ylabel('Mean MPI')
plt.show()


# In[46]:


xx = new2.lev4.reshape(-1, 1)
X2 = sm.add_constant(xx)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())


# 
