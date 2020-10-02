#!/usr/bin/env python
# coding: utf-8

# # Introduction:
# 
# Analizing data from the Ethereum Blockchain

# # Importing Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import glob
import os
get_ipython().run_line_magic('matplotlib', 'inline')


# # Reading and uploading multiple '.csv' files into one main Data Frame

# In[ ]:


path = '/kaggle/input'
all_files = glob.glob(os.path.join(path, "*.csv")) 
df = pd.concat([pd.read_csv(f) for f in all_files], axis=1) #concatenates Verticaly by columns


# In[ ]:


df.shape


# In[ ]:


# There will be repeated columns as each csv file had a date and unixtimestamp column
df.head()


# In[ ]:


#delete the duplicated columns (in this case it was Date(UTC) and UnixTimeStamp)
df = df.loc[:,~df.columns.duplicated()]
df


# In[ ]:


df.shape


# # Preprocessing

# In[ ]:


def data_inv(df):
    print('dataframe: ',df.shape[0])
    print('dataset variables: ',df.shape[1])
    print('-'*10)
    print('dateset columns: \n')
    print(df.columns)
    print('-'*10)
    print('data-type of each column: \n')
    print(df.dtypes)
    print('-'*10)
    print('missing rows in each column: \n')
    c=df.isnull().sum()
    print(c[c>0])
data_inv(df)


# In[ ]:


# Price and transac_value are objects, probably because python cannot read '1,006.41' as a string or float because there is both
# a ',' and a '.'   so we need to replace the ','
df['Price'] = df['Price'].str.replace(',','')
# after that it removes the ',' but remains an object, so we change it to float
df['Price'] = df['Price'].astype(float)


# In[ ]:


df['transac_fee'] = df['transac_fee'].astype(float)
display(df['transac_fee'])


# In[ ]:


#Create a datatime type of column from the unixtimestamp
df['date'] = pd.to_datetime(df['UnixTimeStamp'], unit='s')
df


# In[ ]:


# Price is the same as price_history_usd so we are gonna drop the later and we are also going to drop Date(UTC) since we 
# now have a proper datetime column called 'date'. We are, however, keep unixtimestamp to use in correlation, because it wont read 'date'
df = df.drop(['price_history_usd', 'Date(UTC)', 'supply_growth', 'gas_limit', 'data_size' ], axis=1)
df.describe()


# # EDA & Graphs

# ## Unique addresses and Price

# In[ ]:


f, ax = plt.subplots(figsize=(20, 6))
ax = sns.lineplot(x="date", y="Price", data=df)
#ax.set(yscale="log")
ax2 = ax.twinx()
ax2 = sns.lineplot(x="date", y="unique_adresses", data=df, color='r')
ax2.grid(b=False)
plt.show()


# ## Transactions and price

# (Logarithmic Scale)

# In[ ]:


f, ax = plt.subplots(figsize=(20, 6))
ax = sns.lineplot(x="date", y="Price", data=df)
ax.set(yscale="log")
ax2 = ax.twinx()
ax2 = sns.lineplot(x="date", y="transac_value", data=df, color='r')
ax2.set(yscale="log")
ax2.grid(b=False)
plt.show()


# ## Correlation between inputs

# In[ ]:


#finding correlation between variables
corr_inputs= df.loc[:, df.columns != 'Price']
corr = corr_inputs.corr()


# In[ ]:


plt.figure(figsize=(16,12))
sns.heatmap(corr, annot =  True, cmap="BrBG", fmt=".2f")


# # Regression

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


df.columns


# In[ ]:


inputs = df[['UnixTimeStamp','hx_rate','transac_value','unique_adresses']]
target= df['Price']


# In[ ]:


reg = LinearRegression()
reg.fit(inputs, target)


# ### R-Square

# In[ ]:


reg.score(inputs,target)


# ### Adjusted R-square

# In[ ]:


def adj_r2(x,y):
    r2 = reg.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2


# In[ ]:


adj_r2(inputs,target)


# In[ ]:


from sklearn.feature_selection import f_regression


# In[ ]:


f_regression(inputs,target)


# In[ ]:


p_values = f_regression(inputs,target)[1]
p_values.round(3)


# ### Summary Table

# In[ ]:


reg_summary = pd.DataFrame(data = inputs.columns.values, columns=['Features'])
reg_summary ['Coefficients'] = reg.coef_
reg_summary ['p-values'] = p_values
reg_summary


# In[ ]:




