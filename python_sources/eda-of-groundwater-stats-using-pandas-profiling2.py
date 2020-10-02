#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pyforest')


# In[ ]:


from pyforest import *


# In[ ]:


df= pd.read_csv("../input/districtwise-ground-water-resources-by-july-2017/Dynamic_2017_2_0.csv")


# In[ ]:


df.head()


# In[ ]:


df=df.drop("S.no.", axis=1)


# In[ ]:


df.head()


# In[ ]:


df.isnull().sum()


# In[ ]:


df= df.dropna(axis=0)


# In[ ]:


df.dtypes


# # Counting the current ground water left for different districts

# In[ ]:


df["net available water present"]= df["Total Annual Ground Water Recharge"] - df["Total Current Annual Ground Water Extraction"]


# In[ ]:


df.head()


# # Plotting a barplot to see the current availability of water v/s different states

# In[ ]:


ax=sns.barplot(x="net available water present", y="Name of State", data=df)

ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right')


# ### From above we can see that there is very less availability of water for Punjab, Rajasthan, Delhi and Haryana and very high availability for Sikkim, West Bengal, Andhra Pradesh and Arunachal Pradesh

# In[ ]:


get_ipython().system('pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip')


# In[ ]:


from pandas_profiling import ProfileReport


# In[ ]:


profile = ProfileReport(df, title='District wise ground water Stats', html={'style':{'full_width':False}})


# In[ ]:


profile


# ## By looking at the above statistical analysis one can easily find out the relation between different features. (Go to interactions heading and you fill find the relationships)
