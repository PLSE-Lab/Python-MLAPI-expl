#!/usr/bin/env python
# coding: utf-8

# ###Just tried out prediction without model with score of 15.74640 since the dataset is clean we can play with different approach 
# if you liked it - upvote ! - you could share suggestions/comments for improving the score without using model too - CHEERS !!!

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train['date']=pd.to_datetime(train['date'])
test['date']=pd.to_datetime(test['date'])


# In[ ]:


print(train.dtypes,'\n',test.dtypes)


# In[ ]:


def features(df):
    df['year']=df['date'].dt.year
    df['month']=df['date'].dt.month
    df['week']=df['date'].dt.weekday
    df['weeknum']=df['date'].dt.week
    return df


# In[ ]:


features(train)
features(test)
print(train.head(10))


# In[ ]:


#Data visualization in tableau showed that weekday have more importance, and day have least importance so we are ignoring day !
#Weekday 6- is sunday so we take sales value on sunday for week number 1 (1 of 52) , item number 1 store 1 month 1 and for different year( 5 years from 2013-2017)
x=train[(train['item']==1) & (train['store']==1) & (train['week']==6) & (train['weeknum']==1) & (train['month']==1) ]
x


# ###Just finding the average past sales and divide by total number of years (5) to get overall increase or decrease trend per item/store in past years and multiply with "past years +1"  (6 i.e prediction for 2018)

# In[ ]:


def fun(item,store,week,weeknum,month):
    x=train[(train['item']==item) & (train['store']==store) & (train['week']==week)& (train['weeknum']==weeknum) & (train['month']==month) ]
    return np.ceil(6*(x.sales.mean()/5)).astype(int)-1


# In[ ]:


#test function
y=[]
y.append(fun(1,1,6,1,1))
y.append(fun(1,1,1,1,1))
y.append(fun(1,1,1,1,1))
y


# * ###below row will take 7 to 15 minutes to complete if CPU is 100 %

# In[ ]:


get_ipython().run_cell_magic('time', '', "y=[]\nfor index, i in test.iterrows():\n    y.append(fun(i['item'],i['store'],i['week'],i['weeknum'],i['month']))")


# In[ ]:


id1=pd.read_csv("../input/test.csv",usecols=['id'])


# In[ ]:


sub=pd.DataFrame({'id':id1.id,'sales':y})
sub.head(5)


# In[ ]:


sub.to_csv("NO_MODEL.csv",index=False)


# In[ ]:




