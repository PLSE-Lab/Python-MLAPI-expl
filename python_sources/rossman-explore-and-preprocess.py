#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# In[ ]:


os.chdir("../input/")

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
store = pd.read_csv("../input/store.csv")
sample = pd.read_csv("../input/sample_submission.csv")


# In[ ]:


train.shape, test.shape, store.shape, sample.shape


# In[ ]:


def unique(a):                    
    return len(a.unique())

def unique_value(a):           
    result = []
    for i in a.unique():
        result.append(i)
    return result


# In[ ]:


def data_preprocess(df):
    
    if df["StateHoliday"].dtypes.name == 'object':
        df["StateHoliday"] = df["StateHoliday"].replace({"a":1,"b": 2,"c": 3}).astype('int')

    df["Year"] = df["Date"].apply(lambda x: int(x.split('-')[0]))
    df["Month"] = df['Date'].apply(lambda x: int(x.split('-')[1]))
    df["Day"] = df['Date'].apply(lambda x: int(x.split('-')[2]))
    df["Week"] = df["Month"] * 7 + df["Day"]
    
    #df.drop("Date",axis=1,inplace=True)
       
    return df


# # Train

# In[ ]:


train.info()


# In[ ]:


train = data_preprocess(train)


# - ### Date

# In[ ]:


DIU = train.set_index("Date").groupby("Date").agg(unique)
DIU.shape


# In[ ]:


display(DIU.Store.head())
DIU.Store.value_counts()


# #### => Some store have Blank Date

# - ### Store

# In[ ]:


store_cnt = train.Store.unique()
Blank_date = DIU[DIU.Store.values == 935].index

store_all_day = train.loc[train.Date.isin(Blank_date),"Store"].unique()
store_blank_day = list(set(store_cnt) - set(store_all_day))
store_blank_day = np.array(store_blank_day)

Store_Date_cnt = train.groupby("Store").count().reset_index().iloc[:,[0,2]]
Store_by_Date = Store_Date_cnt.groupby("Date").agg(unique_value)
Store_by_Date["count"] = Store_Date_cnt.groupby("Date").agg(unique)
Store_by_Date


# In[ ]:


t_sales = train[train.Sales > 0]
fig, ax = plt.subplots()
t_sales[t_sales.Store.isin(store_blank_day)].groupby("Date").mean().reset_index().plot(x="Date",y="Sales",ax=ax)
t_sales[t_sales.Store.isin(store_all_day)].groupby("Date").mean().reset_index().plot(x="Date",y="Sales",ax=ax)
fig.set_size_inches(16,4)
plt.show()


# - #### Open

# In[ ]:


train.loc[train.Open == 0,["Sales","Customers"]].sum()


# In[ ]:


op = pd.DataFrame(train.Open.value_counts())
for i in range(1,len(store_cnt)+1):
    op[i] = train.loc[train.Store == i,"Open"].value_counts()
    
op.iloc[:,1:].mean(axis=1)


# - #### Promo 

# In[ ]:


train.Promo.value_counts()


# In[ ]:


promo = pd.DataFrame(train.Promo.value_counts())
for i in range(1,len(store_cnt)+1):
    promo[i] = train.loc[train.Store == i,"Promo"].value_counts(normalize=1)


# In[ ]:


promo.iloc[:,1:].mean(axis=1)


# In[ ]:


pvalue = []
t_sales = train[train.Sales > 0]
store_cnt = train.Store.unique()

for i in range(1,len(store_cnt)+1):
    t = t_sales[t_sales.Store==i]
    ttest = stats.ttest_ind(t.loc[t.Promo == 0,"Sales"].values, t.loc[t.Promo == 1,"Sales"].values)
    p = ttest[1]
    pvalue.append(p)


# In[ ]:


(np.array(pvalue) <= 0.05).sum()


# In[ ]:


np.max(pvalue), np.mean(pvalue)


# In[ ]:


pvalue.index(np.min(pvalue))


# In[ ]:


fig, ax = plt.subplots()
t_sales = train.copy()
t_sales = t_sales[t_sales.Sales > 0]

t=t_sales[t_sales.Store == 335]

t[t.Promo == 0].plot(x="Date",y="Sales",ax=ax,label="Promo_X")
#ax.plot(train.loc[train.Store == ad,"Date"],train.loc[train.Store == ad,"Sales"])
t[t.Promo == 1].plot(x="Date",y="Sales",ax=ax,label="Promo_O")
fig.set_size_inches(16,4)
plt.show()


# - #### DayOfWeek 

# In[ ]:


train.groupby("DayOfWeek")[["DayOfWeek","Date"]].agg(unique)


# In[ ]:


avg_Week = pd.DataFrame([[134,135,135,135,135,134,134],[134,135,135,135,135,134,134]],columns=[1,2,3,4,5,6,7])

display(pd.crosstab(train.Open,train.DayOfWeek)/avg_Week)
display(pd.crosstab(train.SchoolHoliday,train.DayOfWeek)/avg_Week)
display(pd.crosstab(train.StateHoliday,train.DayOfWeek))


# - #### SchoolHoliday 

# In[ ]:


train.SchoolHoliday.value_counts()


# In[ ]:


sch = pd.DataFrame(train.SchoolHoliday.value_counts())
for i in range(1,len(store_cnt)+1):
    sch[i] = train.loc[train.Store == i,"SchoolHoliday"].value_counts(normalize=1)
sch.iloc[:,:6]


# In[ ]:


# SchoolHoliday Affect Store

sch.iloc[:,1:].mean(axis=1)


# - #### StateHoliday

# In[ ]:


train.StateHoliday.value_counts()


# In[ ]:


state = train.groupby("Date").agg(unique)
state.StateHoliday.value_counts()


# In[ ]:


state_date = state[state.StateHoliday == 2].index

state_date


# In[ ]:


st = pd.DataFrame(train.StateHoliday.value_counts())
for i in state_date:
    compar = train.loc[train.Date == i,["Store","StateHoliday","Sales"]]
    st[i] = compar.StateHoliday.value_counts()

st.loc["sum",:] = st.sum()
st


# - #### Customers

# In[ ]:


corr = []
for i in range(1,len(store_cnt)+1):
    corr.append(train.loc[train.Store == i,"Sales"].corr(
        train.loc[train.Store == i,"Customers"]))


# In[ ]:


np.mean(corr), np.std(corr)


# In[ ]:


#### All train feature relate with Sales & Date 


# # Store

# In[ ]:


store.info()


# In[ ]:


def df_merge(df, store):  
    
    store.CompetitionDistance.fillna(store.CompetitionDistance.median(),inplace=True)
    for i in store.columns:
        if "Since" in i :
            store[i].fillna(store[i].median(),inplace=True)

    df = df.merge(store)
    
    for x in ['StoreType', 'Assortment', 'StateHoliday']:
        labels = df[x].unique()
        map_labels = dict(zip(labels, range(1,len(labels)+1)))
        df[x] = df[x].map(map_labels)
    
    df.loc[df["Year"] < df["Promo2SinceYear"],"Promo2"] = 0
    df.loc[(df["Year"] == df["Promo2SinceYear"]) & (df["Week"]<df["Promo2SinceWeek"]),"Promo2"] = 0
    
    df.loc[df["Year"] < df["CompetitionOpenSinceYear"],"CompetitionDistance"] = 0
    df.loc[(df["Year"] == df["CompetitionOpenSinceYear"])&(df["Month"] < df["CompetitionOpenSinceMonth"]),"CompetitionDistance"] = 0
    
    df.drop(["Promo2SinceYear","Promo2SinceWeek","CompetitionOpenSinceYear",
            "CompetitionOpenSinceMonth","PromoInterval"],axis=1,inplace=True)
    
    return df


# In[ ]:


train = data_preprocess(train)
train = df_merge(train,store)

test.fillna(1,inplace=True)
test = data_preprocess(test)
test = df_merge(test,store)


# In[ ]:


train.columns 


# In[ ]:


test.columns


# ---

# ## Model
# - seq_feature => RNN   
#                            -- Merge
# - Others => DNN        

# In[ ]:


seq_feature = ["Year","Month","Day","DayOfWeek"]


# In[ ]:


from keras.layers import Input, Dense, Multiply
from keras.layers import BatchNormalization, Activation
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import regularizers
from keras import backend as K

import tensorflow as tf


# In[ ]:


# model = Sequential()
# model.add(Dense)
# model.add()

