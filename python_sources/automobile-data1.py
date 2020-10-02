#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings("ignore")


# In[ ]:


df=pd.read_csv("../input/Automobile_data.csv")


# In[ ]:


pd.set_option("max_columns",40)


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.tail()


# In[ ]:


df.describe()


# In[ ]:


df.describe(include="object")


# In[ ]:


df.isnull().sum()


# In[ ]:


nul=df["price"].isnull()==True


# In[ ]:


df.loc[nul]


# Here are three NaN values in Price , but we don't remove it and don't replace it with none(like mean , median or mode),
# because there are only 57 total values and all are unique 
# 

# In[ ]:


df["price"].nunique()


# In[ ]:


df.loc[df["company"]=="isuzu"]


# In[ ]:


df.loc[df["company"]=="porsche"]


# In[ ]:


df["price"].dtype


# In[ ]:


df.loc[df["company"]=="toyota"]


# In[ ]:


df.company.value_counts()


# In[ ]:


df.groupby("company")["price"].max().sort_values(ascending=False)


# In[ ]:


df.groupby("company")["average-mileage"].mean().sort_values(ascending=False)


# In[ ]:


df.sort_values("price",ascending=False)


# In[ ]:


sns.heatmap(df.corr(),annot=True)


# In[ ]:


plt.plot(df["price"],"ro--",c="green")
plt.legend()


# In[ ]:


np.log(df["horsepower"]).hist()


# In[ ]:


plt.scatter(df["horsepower"],df["price"])
plt.subplot()


# In[ ]:


df["horsepower"].plot()


# In[ ]:


for j in df:
    if df[j].dtype!="O":
        sns.barplot(df[j],df["price"])
        plt.title(j)
        plt.show()
        


# After visualize from Heatmap and from Bar plot , we can easily check that Horse-Power is Increasing and Mileage is Decreasing 
# while the values of Price are increasing 

# In[ ]:


for col in df.columns:
    if df[col].dtype=="O":
        sns.scatterplot(df["price"],df[col])
        plt.title(col)
        plt.show()


# In[ ]:


df.loc[df["price"]==df["price"].max()]


# ### At last we can only say that mercedes-benz has high value of price because it's  horse-Power is almost high and Mileage is almost low and both factors are already visualize above as well.
