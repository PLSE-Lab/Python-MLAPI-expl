#!/usr/bin/env python
# coding: utf-8

# The introduction to the dataset of this competition says:
# 
# "This dataset contains anonymized features pertaining to a time-varying value for a financial instrument. Each instrument has an id. Time is represented by the 'timestamp' feature and the variable to predict is 'y'.[...]"
# 
# In this notebook I try to look at trades. The first timestamp, where an asset (=a new id) appears, is an entry. The first timestamp after an entry, where the asset is no longer contained in the data, is an exit.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

with pd.HDFStore("../input/train.h5", "r") as hfdata:
    data = hfdata.get("train")
        
get_ipython().run_line_magic('matplotlib', 'inline')


# First we create a DataFrame, which holds each trade in a line. A trade consists of the asset id, the entry of the trade and the exit of the trade. The DataFrame is sorted by entry resp. exit, and named "trades_df".

# In[ ]:


ids = data[["id", "timestamp"]]
id_timestamp_ct = pd.crosstab(index=ids.id, columns=ids.timestamp)
id_timestamp_ct.insert(1813, 1813, 0)
id_timestamp_ct.insert(0, -1, 0)
id_timestamp_ct_diff = id_timestamp_ct.diff(axis=1).abs().fillna(0)
transaction_indexes = np.where(id_timestamp_ct_diff)

tmp = [(id_timestamp_ct_diff.index[x], x, y) for x,y in zip(transaction_indexes[0], transaction_indexes[1])]
trades = [[id, entry-1, exit-1] for ((id, _, entry), (_, _, exit)) in list(zip(tmp[0::2], tmp[1::2]))]
trades_df = pd.DataFrame(data=trades, columns=["id","entry","exit"])
trades_df = trades_df.sort_values(by=["entry", "exit"]).reset_index(drop=True)
trades_df.head(20)


# So it does not look like classic Pairs Trading, where two assets are held during exactly the same time period, one long, one short. But I have seen good analyses by others indicating that there is some kind of "advanced" Pair Traiding, or long short strategy.
# (If a single timeperiod stands for a very short time, e.g. minutes, it would not always be possible to open or close two assets of a pair at exactly the same time. So in this case it might still be classic Pairs Trading.)

# In[ ]:


print("Number of assets: {}".format(len(trades_df.id.unique())))
print("Number of trades: {}".format(len(trades_df.id)))


# In[ ]:


_ = trades_df[["entry","exit"]].plot(linestyle="none", marker=".")


# Looks like most trades are open until the last timeperiod of the training data.

# In[ ]:


trades_df[["id", "exit"]].groupby("exit").count().sort_values(by="id", ascending=False).head().rename(columns = {"id": "count"})


# Indeed 1086 out of 1466 trades are open until the last timeperiod.

# In[ ]:


cnt_trades = trades_df[["id","entry"]].groupby("id").count()
cnt_trades[cnt_trades.entry > 1].sort_values(by="entry", ascending=False)


# So 1414 out of 1424 assets are traded only once, the remaining 10 are traded at least twice. Asset with id=1178 is even traded 19 times.
