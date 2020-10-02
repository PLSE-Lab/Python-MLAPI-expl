#!/usr/bin/env python
# coding: utf-8

# As the column "id" refers to the id of a financial instrument, I tried to explore these instruments in isolation. Moreover I assume that column "y" refers to returns of the instrument.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

with pd.HDFStore("../input/train.h5", "r") as train:
    df = train.get("train")


# First I try to find instruments, that neither start at timestamp 0 nor end at timestamp 1812 (=the last one). This is because I want to see if there are lags in the calculation of fundamental, derived or technical columns.

# In[ ]:


ids = df["id"].unique()
ids_in = {}
for x in ids:
    time = df[df["id"] == x].timestamp
    if time.min() > 100 and time.max() < 1812:
        ids_in[x] = (time.min(), time.max())

for k, v in sorted(ids_in.items())[:10]:
    print("id {} in [{},{}]".format(k,v[0],v[1]))


# I chose the instrument with id = 52 as an example. The first chart shows column "y", which i suppose is the return of instrument, calculated as return[t] = value[t] / value[t-1] - 1.

# In[ ]:


instrument = 52
dfi = df[df["id"] == instrument]
plt.figure(figsize=(8,4))
plt.plot(dfi["timestamp"], dfi["y"], linestyle="none", marker=".")
plt.xlabel('timestamp')
plt.ylabel('returns')
_ = plt.title('returns for id {}'.format(instrument))


# Here I calculate the cumulative returns. So basically it is the line chart of the value of the instrument (stock?).

# In[ ]:


pd.set_option('mode.chained_assignment',None)
dfi.loc[:,"cumprod"] = (1+dfi["y"]).cumprod()
plt.figure(figsize=(8,4))
plt.plot(dfi["timestamp"], dfi["cumprod"], linestyle="none", marker=".")
plt.xlabel('timestamp')
plt.ylabel('value')
_ = plt.title('compound returns for id {}'.format(instrument))


# Now i plot all columns / features of the chosen instrument usable for prediction.

# In[ ]:


cols = [x for x in dfi.columns.values if x not in ["id", "timestamp","y","cumprod"]]
l = len(cols)
f, ax = plt.subplots(int(l/3) + (1 if l%3 > 0 else 0), 3, figsize=(12,int(1.5*l)))
cnt = 0
for col in cols:
    fig = ax[int(cnt/3),cnt%3]
    fig.plot(dfi["timestamp"], dfi[col], linestyle="none", marker=".")
    fig.set_title("{} for id {}".format(col,instrument))
    fig.set_xlim([0,2000])
    fig.axvline(x=ids_in[instrument][0],color="r",linewidth=1)
    fig.axvline(x=ids_in[instrument][1],color="r",linewidth=1)
    cnt += 1


# The structures of the different features are very different. So to my humble opinion this difference in structure has to be reflected somehow in the model. I would not expect a simple linear model to perform / generalize very well.

# In[ ]:




