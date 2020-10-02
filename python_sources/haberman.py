#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

hb=pd.read_csv("../input/haberman.csv",)
print(hb.shape)


# In[ ]:


print(hb.columns)


# In[ ]:


hb["status"].value_counts()


# In[ ]:


hb.head()


# In[ ]:


print("total people survied more than 5 years= ",f'{(225/306)*100}')


# In[ ]:


surv_more = hb[hb['status']==1]
surv_more.describe()


# In[ ]:


surv_less = hb[hb['status']==2]
surv_less.describe()


# In[ ]:


sns.FacetGrid(hb, hue = 'status' , height = 5).map(sns.distplot , 'operation_year').add_legend();
plt.show();


# In[ ]:


sns.FacetGrid(hb , hue = 'status' , height=5).map(sns.distplot , 'age').add_legend();
plt.show()


# In[ ]:


sns.FacetGrid(hb, hue = 'status' , size =5).map(sns.distplot , 'axil_nodes').add_legend();
plt.show()


# In[ ]:


sns.set_style("whitegrid");
sns.FacetGrid(hb, hue = 'status' , size =5).map(plt.scatter , 'operation_year','axil_nodes').add_legend();
plt.show()


# In[ ]:


plt.close();
sns.set_style("whitegrid");
sns.pairplot(hb,hue='status',height=3);
plt.show()


# In[ ]:


sns.boxplot(x = 'status',y = 'operation_year',data = hb)
plt.show()


# In[ ]:


sns.boxplot(x = 'status',y = 'age',data = hb)
plt.show()


# In[ ]:


sns.boxplot(x = 'status',y = 'axil_nodes',data = hb)
plt.show()


# In[ ]:


sns.violinplot(x="status", y="operation_year", data=hb, size=8)
plt.show()


# In[ ]:


sns.violinplot(x="status", y="axil_nodes", data=hb, size=8)
plt.show()


# In[ ]:


sns.jointplot(x='axil_nodes' , y = 'age' , data = hb , kind = 'kde')
plt.show()


# In[ ]:


plt.close()
sns.jointplot(x='operation_year' , y='axil_nodes' , data = hb , kind='kde')
plt.show()


# **Final Conclusion**
# 1. we can use axil_node to identify the class
# 2. survived patients mostly have lesser value of axil_node
# 3. Younger people has more chance of survival, age  between 30-35 have survived more than 5 years.
# 4.the age between 77 to 83 are not survived.
