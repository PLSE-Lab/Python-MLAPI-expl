#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd, numpy as np
from matplotlib import pyplot as plt

import scipy.stats  as stats


# This notebook will help you switching easily from **M5 accuracy** prediction to **uncertainty** . We just use a multiplier scheme under the hood.

# In[ ]:


pd.options.display.max_columns = 50


# In[ ]:


best = pd.read_csv("../input/accuracy-best-public-lbs/kkiller_first_public_notebook_under050_v5.csv")
best.head()


# In[ ]:


sales = pd.read_csv("../input/m5-forecasting-uncertainty/sales_train_validation.csv")
sales.head()


# In[ ]:


sub = best.merge(sales[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]], on = "id")
sub["_all_"] = "Total"
sub.shape


# In[ ]:


sub.head()


# In[ ]:


qs = np.array([0.005,0.025,0.165,0.25, 0.5, 0.75, 0.835, 0.975, 0.995])
qs.shape


# In[ ]:


qs2 = np.log(qs/(1-qs))*.065

ratios = stats.norm.cdf(qs2)
ratios /= ratios[4]
ratios = pd.Series(ratios, index=qs)
ratios.round(3)


# In[ ]:





# In[ ]:


def quantile_coefs(q):
    return ratios.loc[q].values


# In[ ]:


def get_group_preds(pred, level):
    df = pred.groupby(level)[cols].sum()
    q = np.repeat(qs, len(df))
    df = pd.concat([df]*9, axis=0, sort=False)
    df.reset_index(inplace = True)
    df[cols] *= quantile_coefs(q)[:, None]
    if level != "id":
        df["id"] = [f"{lev}_X_{q:.3f}_validation" for lev, q in zip(df[level].values, q)]
    else:
        df["id"] = [f"{lev.replace('_validation', '')}_{q:.3f}_validation" for lev, q in zip(df[level].values, q)]
    df = df[["id"]+list(cols)]
    return df


# In[ ]:


def get_couple_group_preds(pred, level1, level2):
    df = pred.groupby([level1, level2])[cols].sum()
    q = np.repeat(qs, len(df))
    df = pd.concat([df]*9, axis=0, sort=False)
    df.reset_index(inplace = True)
    df[cols] *= quantile_coefs(q)[:, None]
    df["id"] = [f"{lev1}_{lev2}_{q:.3f}_validation" for lev1,lev2, q in 
                zip(df[level1].values,df[level2].values, q)]
    df = df[["id"]+list(cols)]
    return df


# In[ ]:


levels = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "_all_"]
couples = [("state_id", "item_id"),  ("state_id", "dept_id"),("store_id","dept_id"),
                            ("state_id", "cat_id"),("store_id","cat_id")]
cols = [f"F{i}" for i in range(1, 29)]


# In[ ]:


df = []
for level in levels :
    df.append(get_group_preds(sub, level))
for level1,level2 in couples:
    df.append(get_couple_group_preds(sub, level1, level2))
df = pd.concat(df, axis=0, sort=False)
df.reset_index(drop=True, inplace=True)
df = pd.concat([df,df] , axis=0, sort=False)
df.reset_index(drop=True, inplace=True)
df.loc[df.index >= len(df.index)//2, "id"] = df.loc[df.index >= len(df.index)//2, "id"].str.replace(
                                    "_validation$", "_evaluation")

df.shape


# In[ ]:


df.head()


# In[ ]:


df.to_csv("submission.csv", index = False)
