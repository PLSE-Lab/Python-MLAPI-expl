#!/usr/bin/env python
# coding: utf-8

# The MEP, Maximum Entropy Production Principle helps predict the future performance of open system reaction chambers not at equilibrium. One such system is an ad marketplace.
# 
# ![Open Systen not at Equilibrium](https://elmtreegarden.com/wp-content/uploads/2019/03/boiler2.jpg)
# 
# Let use this concept to find the maximum possible Equity given a price vector.

# In[17]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


df = pd.DataFrame({"T":[18.00,12.93,12.84,11.29,10.48,9.76],
                   "n":[166.67,166.67,166.67,166.67,166.67,166.67]},
                 index=["A","B","C","D","E","F"])

df["sales"] = df["T"] * df["n"]
df["p"]= df["sales"]/df.sales.sum()
df["s"] = -df["p"]*np.log2(df["p"])
eq = df.sales.sum()*(1+df.s.sum())
print("Equity = ",eq," Sales = ", df.sales.sum(), "caliber =", df.s.sum())
df.head(10)


# In[18]:


df["sales"] = df["T"] * df["n"]
df["p"]= df["sales"]/df.sales.sum()
df["s"] = -df["p"]*np.log2(df["p"])
eq = df.sales.sum()*(1+df.s.sum())
print("Equity = ",eq," Sales = ", df.sales.sum(), "caliber =", df.s.sum())
df.head(10)


# Begin loop here - or repeat execution from this cell down. To reduce added noise, reduce min_n which is the amplitude of the added noise.

# In[19]:


min_n = df.n.min()
#min_n = min_n/10. uncomment for second cycle for improved results

dn = pd.DataFrame((min_n)*np.random.random_sample((1000000, 6))-(min_n/2), columns=list('ABCDEF'))
dn["tot_r"]= dn["A"] + dn["B"] + dn["C"] + dn["D"] + dn["E"] + dn["F"]


dn["A"] += df.n.iloc[0]
dn["B"] += df.n.iloc[1]
dn["C"] += df.n.iloc[2]
dn["D"] += df.n.iloc[3]
dn["E"] += df.n.iloc[4]
dn["F"] += df.n.iloc[5]

dn["A"] = dn["A"]*(1000/(dn["tot_r"]+1000.))
dn["B"] = dn["B"]*(1000/(dn["tot_r"]+1000.))
dn["C"] = dn["C"]*(1000/(dn["tot_r"]+1000.))
dn["D"] = dn["D"]*(1000/(dn["tot_r"]+1000.))
dn["E"] = dn["E"]*(1000/(dn["tot_r"]+1000.))
dn["F"] = dn["F"]*(1000/(dn["tot_r"]+1000.))
dn["tot"]= dn["A"] + dn["B"] + dn["C"] + dn["D"] + dn["E"] + dn["F"]

dn["sales"] = dn["A"] * df.loc["A","T"] + dn["B"]* df.loc["B","T"] + dn["C"]* df.loc["C","T"]    + dn["D"]* df.loc["D","T"] + dn["E"]* df.loc["E","T"] + dn["F"]* df.loc["F","T"]

dn["caliber"] = -(dn["A"] * df.loc["A","T"] / dn["sales"]) * np.log2(dn["A"] * df.loc["A","T"] / dn["sales"])    -(dn["B"] * df.loc["B","T"] / dn["sales"]) * np.log2(dn["B"] * df.loc["B","T"] / dn["sales"])    -(dn["C"] * df.loc["C","T"] / dn["sales"]) * np.log2(dn["C"] * df.loc["C","T"] / dn["sales"])    -(dn["D"] * df.loc["D","T"] / dn["sales"]) * np.log2(dn["D"] * df.loc["D","T"] / dn["sales"])    -(dn["E"] * df.loc["E","T"] / dn["sales"]) * np.log2(dn["E"] * df.loc["E","T"] / dn["sales"])    -(dn["F"] * df.loc["F","T"] / dn["sales"]) * np.log2(dn["F"] * df.loc["F","T"] / dn["sales"])
dn["equity"] = dn["sales"]*(1 + dn["caliber"])
    
dn = dn.sort_values("equity", ascending=False)
dn.reset_index(inplace = True)
dn.head()


# In[20]:



plt.scatter(dn['caliber'], dn['equity'], s=2)


# In[21]:


plt.scatter(dn['sales'], dn['equity'], s=2)


# In[22]:


import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D

#%% Generate mock data
number_of_datapoints = 1200
dns = dn.sample(n=5000)
x = dns.sales
y = dns.caliber
z = dns.equity.values


#%% Create Color Map
colormap = plt.get_cmap("YlOrRd")
norm = matplotlib.colors.Normalize(vmin=min(z), vmax=max(z))

#%% 3D Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, s=2, c=colormap(norm(z)), marker='o')
ax.set_xlabel('Sales')
ax.set_ylabel('Entropy')
ax.set_zlabel('Equity')
plt.show()


# In[23]:


df2 = df[["T"]]
df3 = dn.nlargest(1,"equity")
df2["n"] = 0.
for i in df.index:
    c = df3.loc[0,i]
    df2.n.loc[i] = c

df2["sales"] = df2["T"] * df2["n"]
df2["p"]= df2["sales"]/df2.sales.sum()
df2["s"] = -df2["p"]*np.log2(df2["p"])
equity = df2.sales.sum()*(1+df2.s.sum())
print("Equity = ",equity," Sales = ", df2.sales.sum(),"caliber = ",df2.s.sum())

df2.head(6)


# In[25]:


df = df2

