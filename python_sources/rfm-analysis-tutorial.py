#!/usr/bin/env python
# coding: utf-8

# ## RFM Analysis

# Let's conduct an RFM Analysis using a retail data. It contains customer level data on transactions
# by date. It also got response information to a promotion campaign conducted by the
# organization.
# 
# I am thankful to many published materials especially by Correia given below.
# https://github.com/joaolcorreia/RFM-analysis/blob/master/RFM%20Analysis.ipynb
# 
# 

# In[ ]:


import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('../input/Retail_Data_Transactions.csv', parse_dates=['trans_date'])


# In[ ]:


df.head(3)


# This data is at transaction level. ie one row for each transaction made by a customer (note that it is not at item level).

# In[ ]:


df.info()


# Let's assume that the study is being done as of 01/Apr/2015. We will identify the earliest and latest dates of transaction.

# In[ ]:


print(df['trans_date'].min(), df['trans_date'].max())


# Number of days from the study date is calculated as below.

# In[ ]:


sd = dt.datetime(2015,4,1)
df['hist']=sd - df['trans_date']
df['hist'].astype('timedelta64[D]')
df['hist']=df['hist'] / np.timedelta64(1, 'D')
df.head()


# Only the transactions made in the last 2 years are considered for analysis.

# In[ ]:


df=df[df['hist'] < 730]
df.info()


# The data will be summarized at customer level by taking *number of days to the latest transaction*, *sum of all transction amount* and *total number of transaction*.

# In[ ]:


rfmTable = df.groupby('customer_id').agg({'hist': lambda x:x.min(), # Recency
                                        'customer_id': lambda x: len(x),               # Frequency
                                        'tran_amount': lambda x: x.sum()})          # Monetary Value

rfmTable.rename(columns={'hist': 'recency', 
                         'customer_id': 'frequency', 
                         'tran_amount': 'monetary_value'}, inplace=True)


# In[ ]:


rfmTable.head()


# We will cross check details of one customer 'CS1112'. Looks like calcultion is correct (latest transaction is 77 days back/ Total number of transaction is 6/ Total amount is 358.

# In[ ]:


df[df['customer_id']=='CS1112']


# RFM analysis involves categorising R,F and M into 3 or more categories. For convenience, let's create 4 categories based on quartiles (quartiles roughly divide the sample into 4 segments equal proportion).

# In[ ]:


quartiles = rfmTable.quantile(q=[0.25,0.50,0.75])
print(quartiles, type(quartiles))


# let's convert quartile information into a dictionary so that cutoffs can be picked up.

# In[ ]:


quartiles=quartiles.to_dict()
quartiles


# In the case of receny, lower is better and hence our categorising scheme need to be reverse.

# In[ ]:


## for Recency 

def RClass(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4
    
## for Frequency and Monetary value 

def FMClass(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1    
    


# In[ ]:


rfmSeg = rfmTable
rfmSeg['R_Quartile'] = rfmSeg['recency'].apply(RClass, args=('recency',quartiles,))
rfmSeg['F_Quartile'] = rfmSeg['frequency'].apply(FMClass, args=('frequency',quartiles,))
rfmSeg['M_Quartile'] = rfmSeg['monetary_value'].apply(FMClass, args=('monetary_value',quartiles,))


# For analysis it is critical to combine the scores to create a single score. There are few approaches. One approach is to just concatenate the scores to create a 3 digit number between 111 and 444. Here the drawback is too many categories (4x4x4). Also, not easy prioritise scores like 421 vs 412. 

# In[ ]:


rfmSeg['RFMClass'] = rfmSeg.R_Quartile.map(str)                             + rfmSeg.F_Quartile.map(str)                             + rfmSeg.M_Quartile.map(str)


# In[ ]:


rfmSeg.head()


# In[ ]:


rfmSeg.sort_values(by=['RFMClass', 'monetary_value'], ascending=[True, False])


# In[ ]:


rfmSeg.groupby('RFMClass').agg('monetary_value').mean()


# Another possibility is to combine the scores to create one score (eg. 4+1+1).  This will create a score between 3 and 12. Here the sdvantage is that each of the scores got same importance. However some scores will have many sgements as constituents (eg - 413 ad 431)

# In[ ]:


rfmSeg['Total Score'] = rfmSeg['R_Quartile'] + rfmSeg['F_Quartile'] +rfmSeg['M_Quartile']
print(rfmSeg.head(), rfmSeg.info())


# In[ ]:


rfmSeg.groupby('Total Score').agg('monetary_value').mean()


# Let's check how the combined score arrange R,F and M

# In[ ]:


rfmSeg.groupby('Total Score').agg('monetary_value').mean().plot(kind='bar', colormap='Blues_r')


# In[ ]:


rfmSeg.groupby('Total Score').agg('frequency').mean().plot(kind='bar', colormap='Blues_r')


# In[ ]:


rfmSeg.groupby('Total Score').agg('recency').mean().plot(kind='bar', colormap='Blues_r')


# We note that combined score is consistent in ordering R,F and M

# Ultimate test of RFM score is the impact on any consumer behaviour. Let's check its impact on the response of customers to a promotion campaign. 

# In[ ]:


res = pd.read_csv('../input/Retail_Data_Response.csv')
res.sort_values('customer_id', inplace=True)

print(res.head(), res.info())


# In[ ]:


rfmSeg.reset_index(inplace=True)
rfmSeg.head()


# In[ ]:


rfmSeg.sort_values('customer_id', inplace=True)
rfm2=pd.merge(rfmSeg, res, on='customer_id')


# In[ ]:


rfm2.info()


# In[ ]:


ax=rfm2.groupby('Total Score').agg('response').mean().plot(kind='bar', colormap='copper_r')
ax.set_xlabel("Total Score")
ax.set_ylabel("Proportion of Responders")


# The chart shows that response behaviour is strongly related to combined score. However, there is not much difference between the scores of 3,4,5, and 6. While the performance of the scores is much lower for scores > 6.

# In[ ]:




