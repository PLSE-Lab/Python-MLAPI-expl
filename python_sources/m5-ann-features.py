#!/usr/bin/env python
# coding: utf-8

# ## M5 Encoder Decoder model - Generating ANN features
# This is with reference to the Notebook M5 Encoder Decoder model with attention in which I feed a hidden state/ embedding from an ANN to the Encoder
# 
# Link : https://www.kaggle.com/josephjosenumpeli/m5-forecasting-encoder-decoder-with-attention
# The features generated are fairly simple. Before generating the features we scaled the data using a transformer function called MinMaxTransformer, this is optional and features can be fed without scaling too. And the scaling is done individually for each time series.
# 
# This Notebook contains the basic approach and is just the start

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_evaluation.csv')


# In[ ]:


df.head()


# In[ ]:


df['id'] = df['id'].str[:-11]


# In[ ]:


class MinMaxtransformer():
    ''' A class to scale the time series data for each item_id'''
    def __init__(self,d_x,d_y, info = None):
        self.d_x = d_x
        self.d_y = d_y
        if info is None :
            self.info = pd.DataFrame({'id': [],'min':[],'max':[]})
        else :
            self.info = info
    
    def fit(self, df):
        '''Will store in min and max values of the rows in a info dataframe'''
        self.info['id'] = df['id']
        self.info['max']= df.loc[:,self.d_x:self.d_y].max(axis=1)
        self.info['min']= df.loc[:,self.d_x:self.d_y].min(axis=1)
        self.info['maxdiffmin'] = self.info['max'] - self.info['min']
    
    def transform(self , df, d_x = None ,d_y = None):
        if d_x == None or d_y == None :
            d_x = self.d_x
            d_y = self.d_y
        filt = self.info['id'].isin(df['id'].tolist())
        info = self.info.loc[filt,:]
        for col in df.loc[:,d_x:d_y].columns:
            df[col] = (df[col] - info['min'])/(info['maxdiffmin'])
        return df
    
    def reverse_transform(self, df, d_x =None,d_y = None):
        
        filt = self.info['id'].isin(df['id'].tolist())
        info = self.info.loc[filt,:]
        if d_x == None or d_y == None :
            d_x = self.d_x
            d_y = self.d_y
        for col in df.loc[:,d_x:d_y].columns:
            df[col] = round(df[col] * info['maxdiffmin'] + info['min'])
        
        return df


# In[ ]:


mmt = MinMaxtransformer('d_1','d_1913')
mmt.fit(df)
df = mmt.transform(df,'d_1','d_1941') # this takes a little time


# **Individual time series Features**

# In[ ]:


df.set_index('id', inplace = True)
df['std'] = df.loc[:,'d_1':'d_1913'].std(axis =1 )
df['mean'] = df.loc[:,'d_1':'d_1913'].mean(axis =1 )
df['median'] = df.loc[:,'d_1':'d_1913'].median(axis =1 )
df['skew'] = df.loc[:,'d_1':'d_1913'].skew(axis =1 )


# **Item level time series Features** 
# If you are familiar with mean encodings this approach is simmilar to that in which we are capturing the properties of the aggregated series at an item level and merging into our individual time series features.
# 
# A similar approach can be adopted for store and category level.

# In[ ]:


item_df = df.groupby('item_id').agg('mean').loc[:,'d_1':'d_1913']
item_df['std'] = item_df.loc[:,'d_1':'d_1913'].std(axis =1 )
item_df['mean'] = item_df.loc[:,'d_1':'d_1913'].mean(axis =1 )
item_df['median'] = item_df.loc[:,'d_1':'d_1913'].median(axis =1 )
item_df['skew'] = item_df.loc[:,'d_1':'d_1913'].skew(axis =1 )


# In[ ]:


df = df.loc[:,['item_id','dept_id','store_id','std','mean','median','skew']]
df.reset_index(inplace = True)
item_df = item_df.loc[:,['std','mean','median','skew']]
item_df.reset_index(inplace = True)


# In[ ]:


df2 = pd.merge( df, item_df, how = 'left' , on = 'item_id') 


# **One Hot encoding** We did it on store_id and dept_id. It could have been done on state and category too but the former felt more convenient. As both variables include state and category information.

# In[ ]:


Ann_features = pd.concat([df2 , pd.get_dummies(df.loc[:,['store_id','dept_id']])],axis =1 ).drop(['item_id','dept_id','store_id'],axis =1)


# In[ ]:


Ann_features


# In[ ]:


Ann_features.to_csv('ann_features.csv')

