#!/usr/bin/env python
# coding: utf-8

# # Sales simulation

# During preparation for migration from one system to another it was decided to prepare a disaster recovery plan. One of the items in that list is preparation to data losses. The most critical data is related to sales. So, we need to create a model which will allow to regenerate sales, at least for the future statistical analysis.

# Initializing libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Loading and preparing data. We will check our solution for 3 years and for different products.

# In[ ]:


#full data set
csv_set_full = pd.read_csv('../input/sales_data_sample.csv', encoding = 'unicode_escape')

#create data frame
df = pd.DataFrame(csv_set_full)

#initialize list of data frames
df_init = df[(df['PRODUCTLINE']=='None')]
df_c = [df_init]
df_c.clear()

#define list of years and products
prod_lst = ['Classic Cars','Ships','Planes']
year_lst = [2003,2004,2005]

#populate list of data frames for specific products, years and columns
for i in prod_lst:
    for j in year_lst:
        df_var = df[(df['PRODUCTLINE']==i) & (df['YEAR_ID']==j)]
        df_var = df_var.loc[:,['SALES']]
        df_c.append(df_var)


# Let's see KDE for all data frames

# In[ ]:


for i in df_c:
    i.plot.kde()


# Generate list of data sets for normal distribution, corresponding to initial KDE

# In[ ]:


#initialize list of data frames
s_init = df_c[0]
s_c = [s_init]
s_c.clear()

for i in df_c:
    mu = i.mean(axis=0)

    sigma = i.std(axis=0)

    num = i.count()

    s_c.append(np.random.normal(mu, sigma, num))


# Now we will compare existing sales with simulated ones. Will take different products for different years.

# Classic Cars, 2003 year

# In[ ]:


sns.distplot(df_c[0])
    
sns.distplot(s_c[0])


# Ships, 2004 year

# In[ ]:


sns.distplot(df_c[4])
    
sns.distplot(s_c[4])


# Planes, 2005 year

# In[ ]:


sns.distplot(df_c[8])
    
sns.distplot(s_c[8])


# Conclusion:
# 
# 1. According to the results, we can use his model to simulate sales.
# 2. It shows good results for Classic Cars. Ships and Planes more or less fine, except of peak values. However, we didn't use any specific logic for Classic Cars. Hence, we can get opposite results if we continue to simulate sales.
# 3. We used various products and years, to be sure that we covered all cases.
# 4. This approach can be used for statistical purposes only, because numbers are random.
# 5. This approach doesn't take into consideration any side factors which can impact sales (seasonality, natural disasters, etc.) and works with historical data only.
