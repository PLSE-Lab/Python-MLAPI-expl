#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv("../input/haberman.csv",names=['age','year','nodes','survival'])


# In[ ]:


print (data.columns)


# In[ ]:


print(data.shape)


# In[ ]:


data["survival"].value_counts()


# In[ ]:


data["survival"]=data["survival"].map({1:"alive",2:"dead"})
data["survival"]=data["survival"].astype('category')
data.head()


# In[ ]:


data.plot(kind="scatter",x="age",y="nodes")


# most of the patients have nodes between 0 to 10

# In[ ]:


sns.set_style("whitegrid");
sns.FacetGrid(data, hue="survival", size=4)    .map(plt.scatter, "age", "nodes")    .add_legend();
plt.show();
plt.close();


# We cannot distinguish people who died and people didnt not died

# In[ ]:


plt.close();
sns.set_style("whitegrid");
sns.pairplot(data, hue="survival", size=3);
plt.show()


# We cannot distinguish people who died and people didnt not died

# In[ ]:


import numpy as np
data_died = data.loc[data["survival"] == "dead"];
data_notdied = data.loc[data["survival"] == "alive"];
data_notdied
plt.plot(data_died["nodes"], np.zeros_like(data_died["nodes"]), '*')
plt.plot(data_notdied["nodes"], np.zeros_like(data_notdied["nodes"]), '*')
plt.show()


# people who died are with nodes 10 to 23 and people who didnt not died are noded above 23

# In[ ]:


sns.FacetGrid(data,hue="survival",size=5)   .map(sns.distplot,"nodes")   .add_legend()
plt.show();


# In[ ]:


sns.FacetGrid(data,hue="survival",size=5)   .map(sns.distplot,"year")   .add_legend()
plt.show();


# In[ ]:


counts, bin_edges = np.histogram(data_notdied['nodes'], bins=20, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend(['Pdf for the patients who survive more than 5 years',
            'Cdf for the patients who survive more than 5 years'])
plt.show()


# In[ ]:


counts, bin_edges = np.histogram(data_died['nodes'], bins=20, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.legend(['Pdf for the patients who died within 5 years',
            'Cdf for the patients who  died within 5 years'])
plt.show()


# In[ ]:


sns.boxplot(x="survival",y="nodes",data=data)

