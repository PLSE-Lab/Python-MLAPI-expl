#!/usr/bin/env python
# coding: utf-8

# # Relation between sales and evnet/SNAP (EDA calendar.csv)
# 
# calendar.csv has event/SNAP information for each day.  
# SNAP is a Supplemental Nutrition Assistance Program which is already discussed [here](https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/133614)
# 
# Intuitively, sales relates to a day which is held a specific event or SNAP.  
# This notebook shows the investigation result of this assumption.
# 
# 
# ## Import Packages

# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


sns.set()


# ## Load data

# In[ ]:


input_data_df = {}

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        base = os.path.splitext(filename)[0]
        input_data_df[base] = pd.read_csv(os.path.join(dirname, filename))


# calendar.csv includes following event related features.
# 
# * `event_name_1`: Event name
# * `event_type_1`: Event type
# * `event_name_2`: Event name
# * `event_type_2`: Event type
# * `snap_CA`: SNAP in CA (If SNAP is conducted on CA, this value becomes 1)
# * `snap_TX`: SNAP in TX (If SNAP is conducted on TX, this value becomes 1)
# * `snap_WI`: SNAP in WI (If SNAP is conducted on WI, this value becomes 1)

# In[ ]:


calendar_df = input_data_df["calendar"].copy()
calendar_df.head(5)


# ## Sales against SNAP
# 
# sales_train_validation.csv includes number of sales for each day.  
# So, we can investigate the sales effect of SNAP with using calendar.csv and sales_train_validation.csv.

# In[ ]:


def get_snap_means(filter):
    snap_days = calendar_df[filter]["d"].values
    data_df = input_data_df["sales_train_validation"].copy()
    feats = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"] + list(snap_days)
    data_df = data_df.loc[:, feats]
    data_df.dropna(how="all", axis=1, inplace=True)

    snap_days_in_data = []
    for c in data_df.columns:
        if c.startswith("d_"):
            snap_days_in_data.append(c)

    agg_dicts = {}
    for d in snap_days_in_data:
        agg_dicts[d] = np.sum

    snap_means_df = data_df.groupby(["cat_id", "state_id"]).agg(agg_dicts)
    snap_means_df = pd.DataFrame(snap_means_df.T.mean())
    
    return snap_means_df

snap_means_df = get_snap_means(calendar_df["snap_TX"] == 0).drop([0], axis=1)
for category in ["CA", "TX", "WI"]:
    for w in [0, 1]:
        snap_means_df = pd.merge(snap_means_df, get_snap_means(calendar_df["snap_{}".format(category)] == w), left_index=True, right_index=True, how="left")
        snap_means_df.rename(columns={0: "{}/ snap_{}".format("w" if w else "wo", category)}, inplace=True)


# In[ ]:


snap_means_df


# At first glance, we can conifirm that SNAP increases the sale...

# ## Plot
# 
# Let's plot them.

# In[ ]:


fig = plt.figure(figsize=(12.0, 12.0))

for i, category in enumerate(["FOODS", "HOBBIES", "HOUSEHOLD"]):
    for j, state in enumerate(["CA", "TX", "WI"]):
        ax = fig.add_subplot(33*10 + i * 3 + j + 1)
        index = np.arange(3)
        bar_width = 0.35
        labels = snap_means_df.loc[category].index
        ax.bar(index, snap_means_df.loc[category]["wo/ snap_{}".format(state)], bar_width,
               color="b", label="wo/ snap_{}".format(state))
        ax.bar(index + bar_width, snap_means_df.loc[category]["w/ snap_{}".format(state)], bar_width,
               color="r", label="w/ snap_{}".format(state))
        ax.set_title("Number of {} (snap_{})".format(category, state))
        ax.set_xlabel("state_id")
        ax.set_ylabel("Average Number of Items")
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(labels)
        ax.legend()

fig.tight_layout()


# As we expected first, sales are increased by SNAP.  
# And we noticed the interesting features from this plot.
# 
# * SNAP increases sales not only "FOODS", but also "HOBBIES" and "HOUSEHOLD".
# * If SNAP is held on other state, sales tend to be increased.
#    * But this comes from the fact that date of SNAP is almost overlapped described [here](https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/133614#764158).

# ## Sold Items against Event
# 
# This is now in progress...

# In[ ]:




