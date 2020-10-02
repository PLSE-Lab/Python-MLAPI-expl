#!/usr/bin/env python
# coding: utf-8

# In this notebook, we will have a look at each columns and filling their missing values

# In[ ]:


import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as mplt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv("../input/train.csv",
                    parse_dates=["timestamp"],
                   date_parser=lambda x: pd.datetime.strptime(x, "%Y-%m-%d"))


# In[ ]:


train.shape


# In[ ]:


train.id.min(), train.id.max(), train.id.nunique()


# All the id's are unique. Seems all right.

# In[ ]:


train.timestamp.min(), train.timestamp.max(), train.timestamp.nunique()


# In[ ]:


train["year_month"] = train.timestamp.map(lambda x: x.year * 100 + x.month)
train["month"] = train.timestamp.dt.month
train["year"] = train.timestamp.dt.year
train["weekday"] = train.timestamp.dt.weekday


# In[ ]:


ts_df = train[["id", "year_month", "year", "month", "weekday"]]


# In[ ]:


y_df = ts_df.groupby("year").count().reset_index()
m_df = ts_df.groupby("month").count().reset_index()
wd_df = ts_df.groupby("weekday").count().reset_index()

fig, ax = mplt.subplots(ncols=2)
fig.set_size_inches(13, 3)

sns.barplot(data=y_df, x="year", y="id", ax=ax[0])
ax[0].set_title("Transactions over the years")

sns.barplot(data=wd_df, x="weekday", y="id", ax=ax[1])
ax[1].set_title("Transactions over weekdays")

fig, axs = mplt.subplots()
fig.set_size_inches(13, 3)

sns.barplot(data=m_df, x="month", y="id", ax=axs)
axs.set_title("Transaction over months")


# So, thers are less transaction on weekends. 
# That's expected.
# 
# Also we can notice  
# 1. there is a deep decline of transactions from 2014 to 2015 and  
# 2. the rise in transactions from 2013 to 2014

# Columns: full_sq and life_sq

# Let's have a look at how they have grown over these years

# In[ ]:


yr_grp = train.groupby("year").mean().reset_index()
fig, ax = mplt.subplots(ncols=2)
fig.set_size_inches(10, 3)

sns.barplot(data=yr_grp, x="year", y="full_sq", orient="v", ax=ax[0])
ax[0].set_title("full_sq over the years")

sns.barplot(data=yr_grp, x="year", y="life_sq", orient="v", ax=ax[1])
ax[1].set_title("life_sq over the years")


# It looks like people start preferring larger houses.  
# Living area includes rooms in the houses. So, we now will have a look at the number of rooms(num_room) feature

# In[ ]:


sns.heatmap(train[["full_sq", "life_sq", "num_room", "price_doc"]].corr(), annot=True)


# Lets have a look at the year difference between the year of transaction and the year built

# In[ ]:


mode_by_own = train.loc[train.product_type == "OwnerOccupier", "build_year"].mode()[0]
mode_by_invest = train.loc[train.product_type == "Investment", "build_year"].mode()[0]
(mode_by_own, mode_by_invest)


# In[ ]:


train.loc[(train.product_type == "OwnerOccupier") & (train.build_year.isnull()), "build_year"] = mode_by_own
train.loc[(train.product_type == "Investment") & (train.build_year.isnull()), "build_year"] = mode_by_invest


# In[ ]:


train["year_difference"] = train.year - train.build_year


# In[ ]:


inv_val = train.loc[train.product_type == "Investment", "year_difference"].values
own_val = train.loc[train.product_type == "OwnerOccupier", "year_difference"].values


# In[ ]:


fig, ax = mplt.subplots(nrows=2)
fig.set_size_inches(15, 10)
ax[0].hist(inv_val)
ax[0].set_title("Year difference for investment buildings")
sns.countplot(own_val, ax=ax[1])
ax[1].set_title("Year difference for owner occupied buildings")


# In[ ]:


train.loc[train.full_sq < 10, :].shape


# Seems, there are too small houses

# In[ ]:


train.loc[train.full_sq < train.life_sq,:].shape


# These 37 records have living area(without ballconies and non-residential areas) greater than its total area.
# 
# So, we can impute their total area with their living area

# In[ ]:


train.loc[train.full_sq < train.life_sq, "full_sq"] = train.life_sq


# In[ ]:


train.loc[train.floor > train.max_floor, :].shape


# There are also records with floor values greater than the maximum floors in the building

# In[ ]:


train.loc[train.kitch_sq > train.full_sq, :].shape


# These have their kitchen area larger than the total area of the house

# In[ ]:


rooms = train[["num_room", "price_doc"]].groupby("num_room").aggregate(np.mean).reset_index()
mplt.scatter(x=rooms.num_room, y=rooms.price_doc)
mplt.xlabel("Num rooms")
mplt.ylabel('Mean Price')


# 8 rooms seems plausible. 
# 
# But then, a house with many rooms but with a price more or less to a house with an average of 5 or 6 rooms seems like we have an erroneous entry

# In[ ]:


population_errors = train.full_all - (train.male_f + train.female_f)
sns.countplot(population_errors)


# It looks like full_all is not the exact sum of male and female population. We can eliminate this column and use the ratio of male and female population as a feature

# In[ ]:


train.loc[(train.state < 1) ^ (train.state > 4), "state"] = np.nan
inv_counts = train[train.product_type == "Investment"]["state"].value_counts()
own_counts = train[train.product_type == "OwnerOccupier"]["state"].value_counts()
product_category = pd.DataFrame([inv_counts, own_counts])
product_category.index = ["Investment", "OwnerOccupier"]
product_category.plot(kind="bar", stacked=True)


# https://www.kaggle.com/c/sberbank-russian-housing-market/discussion/32247#180000
# 
# According to this thread, it seems most owner occupied houses are of best quality

# Will add more...
# Upvote if you find it useful
