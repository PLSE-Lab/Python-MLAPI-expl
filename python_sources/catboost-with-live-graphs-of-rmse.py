#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from catboost import Pool, CatBoostRegressor


# In[ ]:


raw_df = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")


# In[ ]:


raw_df.sort_values(by="first_active_month", inplace=True)


# In[ ]:


raw_df.head()


# In[ ]:


print(raw_df.nunique(), raw_df["first_active_month"].unique())


# ### About data
# Data seems to be of 75 months, starting 2011-11 to 2018-02
# * *feature_1* has 5 categories
# * *feature_2* has 3 categories
# * *feature_3* has 2 categories
# 

# ### Let's train very basic baseline model with given features

# ##### Convert first_active_month to datetime

# In[ ]:


raw_df["first_active_month"] = pd.to_datetime(raw_df["first_active_month"], format="%Y-%m")


# In[ ]:


raw_df.sort_values(by="first_active_month", inplace=True)


# In[ ]:


raw_df.tail()


# #### Train test split

# In[ ]:


# raw_df["first_active_month"].sort_values().value_counts()


# In[ ]:


def train_test_split(df):
    X_train = df[df["first_active_month"] < "2017-11-01" ]
    X_val =  df[df["first_active_month"] >= "2017-11-01" ]
    y_train = X_train["target"]
    y_val = X_val["target"]
    X_train.drop(columns="target", axis=1, inplace=True)
    X_val.drop(columns="target", axis=1, inplace=True)
    X_train["first_active_month"] = X_train["first_active_month"].dt.strftime('%Y-%m-%d')
    X_val["first_active_month"] = X_val["first_active_month"].dt.strftime('%Y-%m-%d')
    return X_train, X_val, y_train, y_val
    


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(raw_df)


# In[ ]:


print(X_train.shape, X_val.shape,y_train.shape, y_val.shape)


# In[ ]:


X_train.fillna(0, inplace=True)
X_val.fillna(0, inplace=True)


# In[ ]:


X_train.info()


# In[ ]:


train_pool = Pool(X_train, y_train,cat_features=[0,1])
test_pool = Pool(X_val, cat_features=[0,1])


# In[ ]:


model = CatBoostRegressor(depth=6, learning_rate=0.0001, iterations=500, loss_function='RMSE')


# In[ ]:


model.fit(train_pool,plot=True)


# In[ ]:


preds = model.predict(test_pool)
print(preds)


# In[ ]:


df = pd.DataFrame()
df["date"] = X_val["first_active_month"]
df["card_id"] = X_val["card_id"]
df["actual"] = y_val
df["pred"] = preds


# In[ ]:


def plot(df):
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m")
    plt.plot(df["date"], df["actual"], label="actual")
    plt.plot(df["date"], df["pred"], label="predicted")
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Prediction')
    plt.xticks(rotation=30)
    plt.yticks([x * 0.1 for x in range(-1, 1)])
#     plt.axis(ymax=1.1)
    plt.title("Compare")
    plt.show()


# In[ ]:


def plot_cards():
    x = 0
    for d in df['date'].unique(): 
        temp_df = df[df["date"] == d]
#         print(temp_df.shape)
        plot(temp_df)
#         if x == 10:
#             break
#         x = x + 1
    


# In[ ]:


plot_cards()

