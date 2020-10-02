#!/usr/bin/env python
# coding: utf-8

# <a id="0"></a> <br>
# ## Kernel Headlines
# 1. [Features at a Glance](#1)
#     1. [Train.csv](#2)
#     
#         1. [first_active_month](#3)
#         2. [card_id](#4)
#         3. [features_#](#5)
#         4. [target](#6)
#         
# 	2. [merchants.csv](#7)
# 	
# 	    1. [merchant_group_id & merchant_category_id & subsector_id](#8)
# 		2. [numerical_1 & numerical_2](#9)
# 		3. [category_1 & category_2 & category_4 ](#10)
# 		4. [most_recent_sales_range & most_recent_purchases_range](#11)
#         5. [city_id & state_id](#12)
#         6. [avg_sales_lag3 & avg_sales_lag6 & avg_sales_lag12](#13)
#         7. [avg_purchases_lag3 & avg_purchases_lag6 & avg_purchases_lag12](#14)
# 	
#     3. [Historical_transactions_df.csv](#15)
# 	
# 		1. [authorized_flag & catgory_1 & category_2 & category_3](#16)
# 		2. [purchase_amount](#17)
# 		3. [installments](#18)
# 		4. [churn rate using merchant_id](#19)
# 		5. [churn rate using card_id](#20)
# 		6. [nested_pie_chart historical state_id & city_id](#21)
# 	
# 	4. [Answering Basic Question about Datasets](#22)
# 		1. [Which features could be helpful ?! ](#23)
# 		2. [Status of Test Set](#24)
# 		3. [Who are disloyal merchants and who are affordable merchants](#25)
# 		4. [Who are disloyal cards and who are affordable cards](#26)
#        
# 2. [Training](#27)
#       1. [Preprocessing ](#28)
#       2. [SimpleTraining ](#29)
#       3. [Feature Importances](#30)
#       4. [Improving accuracy using boosting methods](#31)

# <a id="1"></a> <br>
# #  1-FEATURES AT A GLANCE
# 
# In the first paragraph we will take a glance to our feature spaces.  We will try to data feature by feature.
# 
# Lets import necessary packages ;-).

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime,timedelta
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# Considering the memory limitations, we need to reduce the size of historical_transactions. It is the biggest dataset existed in this kernel. 
# Inspiring form [Fabien Kernel](https://www.kaggle.com/fabiendaniel/elo-world), we decrease the size of our data to prevent probable mermory overflow.

# In[ ]:



def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[ ]:


train_df = reduce_mem_usage(pd.read_csv("../input/train.csv",parse_dates=["first_active_month"]))
test_df = reduce_mem_usage(pd.read_csv("../input/test.csv",parse_dates=["first_active_month"]))
historical_df = reduce_mem_usage(pd.read_csv("../input/historical_transactions.csv",parse_dates=["purchase_date"]))
new_merchants_df = reduce_mem_usage(pd.read_csv("../input/new_merchant_transactions.csv",parse_dates=["purchase_date"]))
merchants_df = reduce_mem_usage(pd.read_csv("../input/merchants.csv"))
# tmp_train_df = pd.merge(train_df,pd.concat([new_merchants_df,historical_df]),on="card_id")
# tmp_test_df =  pd.merge(test_df,pd.concat([new_merchants_df,historical_df]),on="card_id")


# <a id="2"></a> <br>
# **A. Train.csv**

# In[ ]:


train_df.head()


# In[ ]:


train_df.first_active_month.describe()


# <a id="3"></a> <br>
# 1. FIRST_ACTIVE_MONTH

# In[ ]:


fig, axes = plt.subplots(figsize=(15,10))
axes.set_title("First Active Month")
axes.set_ylabel("#")
axes.set_xlabel("years")
train_df.first_active_month.value_counts().plot()


# <a id="4"></a> <br>
# 2. CARD_ID

# In[ ]:


train_df.card_id.describe()


# <a id="5"></a> <br>
# 3. FEATURES_1 & FEATURES_2 & FEATURES_3

# In[ ]:


fig,axes = plt.subplots(1,3,figsize=(15,8),sharey=True)
axes[0].set_ylabel("#")
axes[1].set_ylabel("#")
axes[2].set_ylabel("#")
# axes.set_xlabel("feature")
train_df["feature_1"].value_counts().plot(kind="bar",ax = axes[0],title="feature_1",rot=0,color="tan")
train_df["feature_2"].value_counts().plot(kind="bar",ax=axes[1],title="feature_2",rot=0,color="teal")
train_df["feature_3"].value_counts().plot(kind="bar",ax=axes[2],title="feature_3",rot=0,color="gray")


# <a id="4"></a> <br>
# 4. TARGET

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,10))
axes.set_title("target boxplot")
axes.set_ylabel("target")
axes.boxplot(list(train_df["target"].values),showmeans=True)


# <a id="7"></a> <br>
# **B. Merchants.csv**

# In[ ]:


merchants_df.head()


# In[ ]:


merchants_df.columns


# <a id="8"></a> <br>
# 1. MERCHANT_GROUP_ID & MERCHANT_CATEGORY_ID & SUBSCTOR_ID
# 
# *merchant_group_id: Merchant group (anonymized )*
# 
# *merchant_category_id: Unique identifier for merchant category (anonymized )*
# 
# *subsector_id: Merchant category group (anonymized )*

# In[ ]:


fig,axes = plt.subplots(1,3,figsize=(15,8))
np.log(merchants_df["merchant_group_id"].value_counts()[:10]).plot(kind="bar",ax = axes[0],title="Log(#(merchant_group_id))",rot=0,color="tan")
merchants_df["merchant_category_id"].value_counts()[:10].plot(kind="bar",ax=axes[1],title="#(merchant_category_id)",rot=0,color="teal")
merchants_df["subsector_id"].value_counts()[:10].plot(kind="bar",ax=axes[2],title="#(subsector_id)",rot=0,color="blue")


# <a id="9"></a> <br>
# 2. NUMERICAL_1 & NUMERICAL_2
# 
# *numerical_1: anonymized measure*
# 
# *numerical_2: anonymized measure*

# In[ ]:


pd.concat([merchants_df["numerical_1"],merchants_df["numerical_2"]],axis=1).describe()


# <a id="10"></a> <br>
# 3. CATEGORY_1 & CATEGORY_2 & CATEGORY_4
# 
# *category_1: anonymized category*
# 
# *category_2: anonymized category*
# 
# *category_4: anonymized category*

# In[ ]:


fig,axes = plt.subplots(1,3,figsize=(21,7))
merchants_df["category_1"].value_counts().plot(kind="pie",explode=(0,0.1),autopct='%1.1f%%',shadow=False, startangle=90,ax=axes[0])
merchants_df["category_2"].value_counts().plot(kind="pie",explode=(0,0.01,0.02, 0.03, 0.04),autopct='%1.1f%%',shadow=False, startangle=90,ax=axes[1])
merchants_df["category_4"].value_counts().plot(kind="pie",explode=(0,0.1),autopct='%1.1f%%',shadow=False, startangle=90,ax=axes[2])


# <a id="11"></a> <br>
# 4. MOST_RECENT_SALES_RANGE & MOST_RECENT_PURCHASE_RANGE
# 
# *most_recent_sales_range: Range of revenue (monetary units) in last active month --> A > B > C > D > E *
# 
# *most_recent_purchases_range: Range of quantity of transactions in last active month --> A > B > C > D > E*

# In[ ]:


fig,axes = plt.subplots(1,2,figsize=(16,8))
merchants_df["most_recent_purchases_range"].value_counts().plot(kind="pie",autopct='%1.1f%%',shadow=False, startangle=90,ax=axes[0])
merchants_df["most_recent_sales_range"].value_counts().plot(kind="pie",autopct='%1.1f%%',shadow=False, startangle=90,ax=axes[1])


# <a id="12"></a> <br>
# 5. CITY_ID & STATE_ID
# 
# *city_id: City identifier (anonymized )*
# 
# *state_id: State identifier (anonymized)*

# In[ ]:


#city_id && #state_id
fig,axes = plt.subplots(2,1,figsize=(16,8))
np.log(merchants_df["city_id"].value_counts())[:20].plot(kind="bar",title="Log(#(city_id))",rot=0,ax=axes[0],color="b")
np.log(merchants_df["state_id"].value_counts()).plot(kind="bar",title="Log(#(state_id))",rot=0,ax=axes[1],color="b")


# <a id="13"></a> <br>
# 6. AVG_SALES_LAG3 & AVG_SALES_LAG6 & AVG_SALES_LAG12
# 
# *avg_sales_lag3: Monthly average of revenue in last 3 months divided by revenue in last active month*
# 
# *avg_sales_lag6: Monthly average of revenue in last 6 months divided by revenue in last active month*
# 
# *avg_sales_lag12: Monthly average of revenue in last 12 months divided by revenue in last active month*

# In[ ]:


merchants_df[["avg_sales_lag3","avg_sales_lag6","avg_sales_lag12"]].describe()


# <a id="14"></a> <br>
# 7. AVG_PURCHASES_LAG3 & AVG_PURCHASES_LAG6 & AVG_PURCHASES_LAG12
# 
# *avg_purchases_lag3: Monthly average of transactions in last 3 months divided by transactions in last active month*
# 
# *avg_purchases_lag6: Monthly average of transactions in last 6 months divided by transactions in last active month*
# 
# *avg_purchases_lag12: Monthly average of transactions in last 12 months divided by transactions in last active month*

# In[ ]:


merchants_df[["avg_purchases_lag3","avg_purchases_lag6","avg_purchases_lag12"]].describe()


# <a id="15"></a> <br>
# **C. History_transactions.csv**

# In[ ]:


historical_df.head()


# In[ ]:


historical_df.columns


# <a id="16"></a> <br>
# 1. AUTHORIZED_FLAG & CATEGORY_1 & CATEGORY_2 & CATEGORY_3
# 

# In[ ]:


fig,axes = plt.subplots(2,2,figsize=(8,8))
historical_df["authorized_flag"].value_counts().plot(kind="pie",explode=(0,0.1),autopct='%1.1f%%',shadow=False, startangle=90,ax=axes[0][0])
historical_df["category_1"].value_counts().plot(kind="pie",explode=(0,0.1),autopct='%1.1f%%',shadow=False, startangle=90,ax=axes[0][1])
historical_df["category_3"].value_counts().plot(kind="pie",explode=(0,0.1,0.1),autopct='%1.1f%%',shadow=False, startangle=90,ax=axes[1][0])
historical_df["category_2"].value_counts().plot(kind="pie",explode=(0,0.1,0.1,0.1,0.1),autopct='%1.1f%%',shadow=False, startangle=90,ax=axes[1][1])


# <a id="17"></a> <br>
# 2.  PURCHASE_AMOUNT
# 

# In[ ]:


historical_df["purchase_amount"].describe()


# There is a huge difference between 75% percentile and maximum. You can conclude that there is a peak which may not be related to normal activeities. Lets get 99% percentile to check this. 

# In[ ]:


np.percentile(historical_df["purchase_amount"].values,q=99)


# 99 percent of purchase amounts are less than 1.22. Lets assume the remaining 1 percent is outlier and move to visualize it.

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
axes.set_title("purchase amount violin plot (removing top 1%)")
axes.set_ylabel("purchase_amount")
ax = sns.violinplot(y=list(historical_df[historical_df["purchase_amount"] < np.percentile(historical_df["purchase_amount"],99)]["purchase_amount"]),showmeans=True,showmedians=True, palette="muted")


# <a id="18"></a> <br>
# 3.  INSTALLMENTS
# 

# In[ ]:


fig,axes = plt.subplots(1,1,figsize=(8,8))
axes.set_title("log(#(installments))")
axes.set_ylabel("#")
axes.set_xlabel("records")
np.log(historical_df.installments.value_counts()).plot(kind="bar",color="b")


# <a id="19"></a> <br>
# 4.  CHURN_RATE WITH MERCHANT_ID
# 

# In[ ]:


historical_df["purchase_date"].head(5)


# In[ ]:


"first_day:'{}' and last_day:'{}'".format(np.min(historical_df["purchase_date"]), np.max(historical_df["purchase_date"]))


# In[ ]:


start_date = datetime(2017, 1, 1)
end_date = datetime(2018, 2, 1)
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)
dates_month = []
for single_date in daterange(start_date, end_date):
    dates_month.append(single_date.strftime("%Y-%m"))
dates_month = list(set(dates_month))

tmp_churn_df = pd.DataFrame()
tmp_churn_df["date"] = historical_df["purchase_date"]
tmp_churn_df["yaer"] = pd.DatetimeIndex(tmp_churn_df["date"]).year
tmp_churn_df["month"] =pd.DatetimeIndex(tmp_churn_df["date"]).month
tmp_churn_df["merchant_id"] = historical_df["merchant_id"]
tmp_churn_df.head()


# In[ ]:


"distinct merchants who have bought on the website on 2017-01 are:'{}'merchants".format(len(set(historical_df[(tmp_churn_df.yaer == 2017) & (tmp_churn_df.month == 1) ]["merchant_id"])))


# In[ ]:


target_intervals_list = [(2017,1),(2017,2),(2017,3),(2017,4),(2017,5),(2017,6),(2017,7),(2017,8),(2017,9),(2017,10),(2017,11),(2017,12),(2018,1),(2018,2)]
intervals_visitors = []
for tmp_tuple in target_intervals_list:
    intervals_visitors.append(tmp_churn_df[(tmp_churn_df.yaer == tmp_tuple[0]) & (tmp_churn_df.month == tmp_tuple[1]) ]["merchant_id"])
"Size of intervals_visitors:{} ".format(len(intervals_visitors))

tmp_matrix = np.zeros((14,14))

for i in range(0,14):
    k = False
    tmp_set = []
    for j in range(i,14): 
        if k:
            tmp_set = tmp_set & set(intervals_visitors[j])
        else:
            tmp_set = set(intervals_visitors[i]) & set(intervals_visitors[j])
        tmp_matrix[i][j] = len(list(tmp_set))
        k = True
xticklabels = ["interval 1","interval 2","interval 3","interval 4","interval 5","interval 6","interval 7","interval 8",
              "interval 9","interval 10","interval 11", "interval 12","interval 13","interval 14"]
yticklabels = [(2017,1),(2017,2),(2017,3),(2017,4),(2017,5),(2017,6),(2017,7),(2017,8),(2017,9),(2017,10),(2017,11),(2017,12),(2018,1),(2018,2)]
fig, ax = plt.subplots(figsize=(14,14))
ax = sns.heatmap(np.array(tmp_matrix,dtype=int), annot=True, cmap="RdBu_r",xticklabels=xticklabels,fmt="d",yticklabels=yticklabels)
ax.set_title("Churn-rate using merchant_id")
ax.set_xlabel("intervals")
ax.set_ylabel("months")


# In[ ]:


import gc;gc.collect()


# <a id="20"></a> <br>
# 5.  CHURN_RATE WITH CARD_ID
# 

# In[ ]:


start_date = datetime(2017, 1, 1)
end_date = datetime(2018, 2, 1)
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)
dates_month = []
for single_date in daterange(start_date, end_date):
    dates_month.append(single_date.strftime("%Y-%m"))
dates_month = list(set(dates_month))

tmp_churn_df = pd.DataFrame()
tmp_churn_df["date"] = historical_df["purchase_date"]
tmp_churn_df["yaer"] = pd.DatetimeIndex(tmp_churn_df["date"]).year
tmp_churn_df["month"] =pd.DatetimeIndex(tmp_churn_df["date"]).month
tmp_churn_df["card_id"] = historical_df["card_id"]
tmp_churn_df.head()

"distinct cards who have bounght on the website on 2017-01 are:'{}'merchants".format(len(set(historical_df[(tmp_churn_df.yaer == 2017) & (tmp_churn_df.month == 1) ]["merchant_id"])))
target_intervals_list = [(2017,1),(2017,2),(2017,3),(2017,4),(2017,5),(2017,6),(2017,7),(2017,8),(2017,9),(2017,10),(2017,11),(2017,12),(2018,1),(2018,2)]
intervals_visitors = []
for tmp_tuple in target_intervals_list:
    intervals_visitors.append(tmp_churn_df[(tmp_churn_df.yaer == tmp_tuple[0]) & (tmp_churn_df.month == tmp_tuple[1]) ]["card_id"])
"Size of intervals_visitors:{} ".format(len(intervals_visitors))

tmp_matrix = np.zeros((14,14))

for i in range(0,14):
    k = False
    tmp_set = []
    for j in range(i,14): 
        if k:
            tmp_set = tmp_set & set(intervals_visitors[j])
        else:
            tmp_set = set(intervals_visitors[i]) & set(intervals_visitors[j])
        tmp_matrix[i][j] = len(list(tmp_set))
        k = True
xticklabels = ["interval 1","interval 2","interval 3","interval 4","interval 5","interval 6","interval 7","interval 8",
              "interval 9","interval 10","interval 11", "interval 12","interval 13","interval 14"]
yticklabels = [(2017,1),(2017,2),(2017,3),(2017,4),(2017,5),(2017,6),(2017,7),(2017,8),(2017,9),(2017,10),(2017,11),(2017,12),(2018,1),(2018,2)]
fig, ax = plt.subplots(figsize=(14,14))
ax = sns.heatmap(np.array(tmp_matrix,dtype=int), annot=True, cmap="RdBu_r",xticklabels=xticklabels,fmt="d",yticklabels=yticklabels)
ax.set_title("Churn-rate using card_ids")
ax.set_xlabel("intervals")
ax.set_ylabel("months")


# Comparing two diagram represents that in overall number of buys have been increased and the system have satisfied churn rate. We will go deeper to churn rates and conversion rates in next sections.

# In[ ]:


del tmp_churn_df
import gc;gc.collect()


# <a id="21"></a> <br>
# 5.  NESTED PIE CHART FOR STATE_ID AND CITY_ID
# 

# In[ ]:


city_state_df = historical_df[["city_id" , "state_id"]]

tmp_df = city_state_df.groupby(by="city_id").count().reset_index()
tmp_df = tmp_df.rename(index=str,columns={"state_id" : "count"})
tmp_df.sort_values(ascending=False,by="count",inplace=True)
# tmp_df = tmp_df.head(100)
lables=list(tmp_df.city_id)
sizes=list(tmp_df["count"])

tmp_df=city_state_df.groupby(by="state_id").count().reset_index()
tmp_df = tmp_df.rename(index=str,columns={"city_id" : "count"})
tmp_df.sort_values(ascending=False,by="count",inplace=True)
labels_gender = list(tmp_df.state_id)
sizes_gender = list(tmp_df["count"])

fig, ax = plt.subplots(figsize=(15,15))
plt.pie(sizes, labels=lables, startangle=90,frame=True)
plt.pie(sizes_gender,radius=0.75,startangle=90)
centre_circle = plt.Circle((0,0),0.5,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.show()
print()


# In[ ]:


del city_state_df
del tmp_df


# states -1, 1 and 2 have more than half of buy records. 

# <a id="22"></a> <br>
# **D. ANSWERING BASIC QUESTIONS ABOUT DATASETS. **
# 
# <a id="23"></a> <br>
# 1.  WHICH FEATURES COULD BE HELPFUL
# 

# In[ ]:


historical_df["year"] = historical_df["purchase_date"].dt.year
historical_df["month"] = historical_df["purchase_date"].dt.month
historical_df["day"] = historical_df["purchase_date"].dt.day
historical_df["hour"] = historical_df["purchase_date"].dt.hour
#  you can also use single line commands
# historical_df = pd.concat(
#     [
#         historical_df,
#         historical_df["purchase_date"].apply(lambda x:
#                                              pd.Series({
#                                                  'year':x.year,
#                                                  'month':x.month,
#                                                  'day':x.day
#                                              }))
#     ]
#     ,axis=1) 


# In[ ]:


fig,axes = plt.subplots(4,2,figsize=(10,30))
axes[0][0].set_title("yearly histogram")
axes[0][1].set_title("monthly histogram")
axes[1][0].set_title("daily histogram")
axes[1][1].set_title("hourly histogram")
axes[2][0].set_title("state_id histogram")
axes[2][1].set_title("city_id histogram")
axes[3][0].set_title("subsector_id histogram")
axes[3][1].remove()

historical_df.year.hist(ax=axes[0][0],normed=True,bins=2)
historical_df.month.hist(ax=axes[0][1],normed=True,bins=12)
historical_df.day.hist(ax=axes[1][0],normed=True,bins=30)
historical_df.hour.hist(ax=axes[1][1],normed=True,bins=24)
historical_df["state_id"].hist( bins=30,normed=True,ax=axes[2][0])
historical_df["city_id"].hist( bins=30,normed=True,ax=axes[2][1])
historical_df["subsector_id"].hist( bins=30,normed=True,ax=axes[3][0])


# Although daily histogram reveals that there is a uniform distribution for historical data but hourly histogram represents that it could be usefull data for our competition goal. city id, state id and also subsector id have acceptable entropy and they may have considerable potential for enhacing the accuracy of model.

# In[ ]:


import gc; gc.collect()


# <a id="24"></a> <br>
# 2.  STATUS OF TEST SET ?!

# In[ ]:


test_df.describe()


# In[ ]:


test_df.shape


# In[ ]:


# test_df = test_df.dropna()


# test_df contains a null in first_active_month in record number #11578

# In[ ]:


datetimes_df = pd.concat([train_df[["first_active_month"]].groupby(by=["first_active_month"],axis=0).size(),test_df[["first_active_month"]].groupby(by=["first_active_month"],axis=0).size()],axis=1)

fig, ax1 = plt.subplots(figsize=(20,10))
t = datetimes_df.index
s1 = datetimes_df.iloc[:,0]
ax1.plot(t, s1, 'b-')
ax1.set_xlabel('date')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('train', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
s2 = datetimes_df.iloc[:,1]
ax2.plot(t, s2, 'r--')
ax2.set_ylabel('test', color='r')
ax2.tick_params('y', colors='r')
fig.tight_layout()


# Test sets are completeltely in the same distribution the train data are.

# In[ ]:


fig,axes = plt.subplots(1,2,figsize=(10,5))
axes[0].set_title("train card_id distru")
axes[1].set_title("test card_id distru")
axes[0].hist(train_df[["card_id"]].groupby(by=["card_id"]).size(),bins=3)
axes[1].hist(test_df[["card_id"]].groupby(by=["card_id"]).size(),bins=3)


#    There is only one card_id in both of the train and test datasets.

# <a id="25"></a> <br>
# 3.  WHO ARE DISLOYAL MERCHANTS & WHO ARE AFFORDABLE MERCHANTS  ?!

# In[ ]:


# calling gargage collector
del datetimes_df
import gc; gc.collect()


# In[ ]:


card_merchant_df = historical_df[["card_id","merchant_id","purchase_amount"]]
card_merchant_df.head()


# In[ ]:


merchants_loyality_and_revenue = card_merchant_df.groupby(by=["merchant_id"]).agg({"card_id":"count","purchase_amount":"sum"})


# In[ ]:


merchants_loyality_and_revenue = merchants_loyality_and_revenue.reset_index().rename(index=str , columns={"purchase_amount" : "amount" , "card_id" : "count"})
merchants_loyality_and_revenue["avg"] = merchants_loyality_and_revenue["amount"] / merchants_loyality_and_revenue["count"]
merchants_loyality_and_revenue.head()


# In[ ]:


# who are disloyal merchants
merchants_loyality_and_revenue.sort_values(by="count",ascending=False).head(5)


# As you can see merchant_id 'M_ID_00a6ca8a8a' has the most interactions with card_ids.

# In[ ]:


#who are the most affordable merchants
merchants_loyality_and_revenue.sort_values(by="avg",ascending=False).head(5)


# <a id="26"></a> <br>
# 4.  WHO ARE DISLOYAL CARDS & WHO ARE AFFORDABLE CARDS  ?!

# In[ ]:


del merchants_loyality_and_revenue


# In[ ]:


cards_loyality_and_revenue = card_merchant_df.groupby(by=["card_id"]).agg({"merchant_id":"count","purchase_amount":"sum"})
cards_loyality_and_revenue = cards_loyality_and_revenue.reset_index().rename(index=str , columns={"purchase_amount" : "amount" , "merchant_id" : "count"})
cards_loyality_and_revenue["avg"] = cards_loyality_and_revenue["amount"] / cards_loyality_and_revenue["count"]


# In[ ]:


cards_loyality_and_revenue.head()


# In[ ]:


# who are disloyal cards
cards_loyality_and_revenue.sort_values(by="count",ascending=False).head(5)


# In[ ]:


#who are the most affordable cards
cards_loyality_and_revenue.sort_values(by="avg",ascending=False).head(5)


# In[ ]:


import gc; gc.collect()


# <a id="27"></a> <br>
# #  2-TRAINING
# 
# Now, lets move on training steps. In the first poit we will try to do simple regression on our data.

# In[ ]:


historical_df.head(1)


# <a id="28"></a> <br>
# **A. PREPROCESSING**

# In[ ]:


#extracting information from historical_df
import gc;gc.collect()
historical_df['authorized_flag'] = historical_df['authorized_flag'].map({'Y':1, 'N':0})
def aggregate_historical_transactions(history):
    history.loc[:, 'purchase_date'] = pd.DatetimeIndex(history['purchase_date']).astype(np.int64) * 1e-9
    
    agg_func = {
        'authorized_flag': ['sum', 'mean'],
        'merchant_id': ['nunique'],
        'city_id': ['nunique'],
        'state_id': ['nunique'],
        'purchase_amount': ['sum', 'median', 'max', 'min', 'std'],
        'installments': ['sum', 'median', 'max', 'min', 'std'],
        'purchase_date': [np.ptp],
        'month_lag': ['min', 'max'],
        'year': ['nunique'],
        'month': ['nunique'],
        'day': ['nunique'],
        'hour': ['nunique'],
        'merchant_category_id': ['nunique'],
        }
    agg_history = history.groupby(['card_id']).agg(agg_func)
    agg_history.columns = ['hist_' + '_'.join(col).strip() 
                           for col in agg_history.columns.values]
    agg_history.reset_index(inplace=True)
    
    df = (history.groupby('card_id')
          .size()
          .reset_index(name='hist_transactions_count'))
    
    agg_history = pd.merge(df, agg_history, on='card_id', how='left')
    
    return agg_history

new_history = aggregate_historical_transactions(historical_df)


# In[ ]:


new_history.head(1)


# In[ ]:


new_merchants_df.head(1)


# In[ ]:


new_merchants_df['authorized_flag'] = new_merchants_df['authorized_flag'].map({'Y':1, 'N':0})
def aggregate_new_transactions(new_trans):    
    agg_func = {
        'authorized_flag': ['sum', 'mean'],
        'merchant_id': ['nunique'],
        'city_id': ['nunique'],
        'purchase_amount': ['sum', 'median', 'max', 'min', 'std'],
        'installments': ['sum', 'median', 'max', 'min', 'std'],
        'month_lag': ['min', 'max'],
        'subsector_id':['nunique'],
        'state_id':['nunique']
        }
    agg_new_trans = new_trans.groupby(['card_id']).agg(agg_func)
    agg_new_trans.columns = ['new_' + '_'.join(col).strip() 
                           for col in agg_new_trans.columns.values]
    agg_new_trans.reset_index(inplace=True)
    
    df = (new_trans.groupby('card_id')
          .size()
          .reset_index(name='new_transactions_count'))
    
    agg_new_trans = pd.merge(df, agg_new_trans, on='card_id', how='left')
    
    return agg_new_trans

new_merchants = aggregate_new_transactions(new_merchants_df)


# In[ ]:


new_merchants.head(1)


# In[ ]:


train_df['elapsed_time'] = (datetime(2018, 2, 1) - train_df['first_active_month']).dt.days
test_df['elapsed_time'] = (datetime(2018, 2, 1) - test_df['first_active_month']).dt.days


# In[ ]:


train_df = pd.merge(train_df, new_history, on='card_id', how='left')
test_df = pd.merge(test_df, new_history, on='card_id', how='left')

train_df = pd.merge(train_df, new_merchants, on='card_id', how='left')
test_df = pd.merge(test_df, new_merchants, on='card_id', how='left')


# In[ ]:


train_df.shape, test_df.shape


# In[ ]:


set(train_df.columns)-set(test_df.columns)


# In[ ]:


# categorizing feature_1 & feature_2 & feature_3
from sklearn.preprocessing import LabelEncoder
for col in ["feature_1", "feature_2","feature_3"]:
#     print(col)
    lbl = LabelEncoder()
    lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
    train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
    test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))


# In[ ]:


exclude_features = ['card_id', 'first_active_month']


# In[ ]:


train_df = train_df.loc[:,train_df.columns[~train_df.columns.isin(exclude_features)]]


# In[ ]:


from sklearn.model_selection import train_test_split
_train, _eval = train_test_split(train_df, test_size=0.2, random_state=42)


# In[ ]:


_train.shape


# <a id="29"></a> <br>
# **B. SIMPLE TRAINING**

# In[ ]:


import lightgbm as lgb
params = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 30,
    "min_child_samples": 100,
    "learning_rate": 0.1,
    "bagging_fraction": 0.7,
    "feature_fraction": 0.5,
    "bagging_frequency": 5,
    "bagging_seed": 2018,
    "verbosity": -1
}
lgb_train = lgb.Dataset(_train.loc[:,_train.columns[~_train.columns.isin(["target"])]], _train["target"])
lgb_eval = lgb.Dataset(_eval.loc[:,_eval.columns[~_eval.columns.isin(["target"])]], _eval["target"], reference=lgb_train)
gbm = lgb.train(params, lgb_train, num_boost_round=2000, valid_sets=[lgb_eval], early_stopping_rounds=100,verbose_eval=100)


# In[ ]:


exclude_features = ["first_active_month","card_id"]
test_df = test_df.loc[:,test_df.columns[~test_df.columns.isin(exclude_features)]]

predicted_target = gbm.predict(test_df, num_iteration=gbm.best_iteration)
predicted_target[predicted_target < 0] = 0 

sub_df = pd.read_csv('../input/sample_submission.csv')
sub_df["target"] = predicted_target

sub_df.to_csv("submission_raw.csv", index=False)


# You will get 3.92 score if use the output for submission.
# 
# In next steps we will use some methods for improving the performance of the regressioner. 

# In[ ]:


test_df.shape, sub_df.shape


# <a id="30"></a> <br>
# **C. FEATURE IMPORTANCE**

# In[ ]:


fig, ax = plt.subplots(figsize=(12,18))
lgb.plot_importance(gbm, max_num_features=50, height=0.8, ax=ax)
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=15)
plt.show()


# <a id="31"></a> <br>
# **D. IMPROVING PERFORMANCE USING BOOSTING METHODS**
# 
# 
# Now, lets improve our accuracy by using boosting approaches. We have used [This kernel](https://www.kaggle.com/youhanlee/hello-elo-ensemble-will-help-you) because of easy and straight forward road map it has. Thanks to @youhanlee.

# In[ ]:


target = train_df['target']
del train_df['target']


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


lgb_params = {"objective" : "regression", "metric" : "rmse", 
               "max_depth": 7, "min_child_samples": 20, 
               "reg_alpha": 1, "reg_lambda": 1,
               "num_leaves" : 64, "learning_rate" : 0.001, 
               "subsample" : 0.8, "colsample_bytree" : 0.8, 
               "verbosity": -1}

FOLDs = KFold(n_splits=5, shuffle=True, random_state=1989)

oof_lgb = np.zeros(len(train_df))
predictions_lgb = np.zeros(len(test_df))

features_lgb = list(train_df.columns)
feature_importance_df_lgb = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(FOLDs.split(train_df)):
    trn_data = lgb.Dataset(train_df.iloc[trn_idx], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train_df.iloc[val_idx], label=target.iloc[val_idx])

    print("LGB " + str(fold_) + "-" * 50)
    num_round = 2000
    clf = lgb.train(lgb_params, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=500, early_stopping_rounds = 2000)
    oof_lgb[val_idx] = clf.predict(train_df.iloc[val_idx], num_iteration=clf.best_iteration)

    fold_importance_df_lgb = pd.DataFrame()
    fold_importance_df_lgb["feature"] = features_lgb
    fold_importance_df_lgb["importance"] = clf.feature_importance()
    fold_importance_df_lgb["fold"] = fold_ + 1
    feature_importance_df_lgb = pd.concat([feature_importance_df_lgb, fold_importance_df_lgb], axis=0)
    predictions_lgb += clf.predict(test_df, num_iteration=clf.best_iteration) / FOLDs.n_splits
    

print(np.sqrt(mean_squared_error(oof_lgb, target)))


# In[ ]:


import xgboost as xgb

xgb_params = {'eta': 0.001, 'max_depth': 7, 'subsample': 0.8, 'colsample_bytree': 0.8, 
          'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True}

FOLDs = KFold(n_splits=5, shuffle=True, random_state=1989)

oof_xgb = np.zeros(len(train_df))
predictions_xgb = np.zeros(len(test_df))


for fold_, (trn_idx, val_idx) in enumerate(FOLDs.split(train_df)):
    trn_data = xgb.DMatrix(data=train_df.iloc[trn_idx], label=target.iloc[trn_idx])
    val_data = xgb.DMatrix(data=train_df.iloc[val_idx], label=target.iloc[val_idx])
    watchlist = [(trn_data, 'train'), (val_data, 'valid')]
    print("xgb " + str(fold_) + "-" * 50)
    num_round = 2000
    xgb_model = xgb.train(xgb_params, trn_data, num_round, watchlist, early_stopping_rounds=100, verbose_eval=500)
    oof_xgb[val_idx] = xgb_model.predict(xgb.DMatrix(train_df.iloc[val_idx]), ntree_limit=xgb_model.best_ntree_limit+50)

    predictions_xgb += xgb_model.predict(xgb.DMatrix(test_df), ntree_limit=xgb_model.best_ntree_limit+50) / FOLDs.n_splits

np.sqrt(mean_squared_error(oof_xgb, target))


# In[ ]:


print('lgb', np.sqrt(mean_squared_error(oof_lgb, target)))
print('xgb', np.sqrt(mean_squared_error(oof_xgb, target)))


# In[ ]:


total_sum = 0.5 * oof_lgb + 0.5 * oof_xgb
print("CV score: {:<8.5f}".format(mean_squared_error(total_sum, target)**0.5))


# In[ ]:


cols = (feature_importance_df_lgb[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df_lgb.loc[feature_importance_df_lgb.feature.isin(cols)]

plt.figure(figsize=(14,14))
sns.barplot(x="importance",
            y="feature",
            data=best_features.sort_values(by="importance",
                                           ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')


# In[ ]:


sub_df = pd.read_csv('../input/sample_submission.csv')
sub_df["target"] = 0.5 * predictions_lgb + 0.5 * predictions_xgb
sub_df.to_csv("submission.csv", index=False)


# Now, the accuracy have been increased thanks to boosting methods ;-).

# 
# **In progress ...**
# 
# **Be in touch to get last commits ...**
# 
# **I'll try to complete it as soon as possible**
# 
# **Your upvote will be motivation for me for continuing the kernel ;-)**
