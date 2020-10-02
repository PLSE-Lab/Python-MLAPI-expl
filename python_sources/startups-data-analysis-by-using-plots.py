#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook", style="darkgrid", palette="Paired", font="sans- serif", font_scale=1, color_codes=True)
import warnings
import plotly.express as px
import plotly.graph_objects as go


# In[ ]:


investments_VC = pd.read_csv("../input/startup-investments-crunchbase/investments_VC.csv",encoding= 'unicode_escape')


# In[ ]:


investments_VC.head()


# In[ ]:


investments_VC.columns


# In[ ]:


investments_VC.info()


# In[ ]:


print(investments_VC.isnull().sum())
sns.heatmap(investments_VC.isnull(),cmap="viridis")


# In[ ]:


investments_VC.shape


# In[ ]:


data_new = investments_VC.dropna()


# In[ ]:


fig=px.scatter_3d(data_new,x=' funding_total_usd ',y=' market ',z='region',color='status',size='funding_rounds',hover_data=['name'])
fig.show()


# In[ ]:


figpie=px.pie(data_new,values='seed',names=' market ',title="seed funding versus type of company",hover_data=['name'])
figpie.show()


# In[ ]:


X = investments_VC["status"].value_counts()
plt.pie(X,labels=X.index,startangle=90,autopct='%1.1f%%',explode = (0,0.1,0.3))
plt.show()     


# In[ ]:


investments_VC.describe()


# In[ ]:


Found_2006 =  investments_VC[investments_VC["founded_year"]>=2006]


# In[ ]:


Found_2006.shape


# In[ ]:


sns.countplot(x="founded_year",data=Found_2006,order=Found_2006.founded_year.value_counts().index)
plt.title("Startups since 2006")
plt.xticks(rotation=90)


# In[ ]:


sns.countplot(data=investments_VC,x="country_code",order=investments_VC["country_code"].value_counts()[:20].index)
plt.xticks(rotation=90)


# In[ ]:


IndianStartup =  investments_VC[investments_VC["country_code"]=="IND"]


# In[ ]:


figin = go.Figure()

figin.add_trace(go.Scatter(
                x=IndianStartup['name'],
                y=IndianStartup[' funding_total_usd '],
                name="",
                line_color='orange'))
figin.update_layout(title_text="funding status in india")
figin.show()

