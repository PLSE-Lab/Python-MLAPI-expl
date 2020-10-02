#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Data Reading

# In[ ]:


data=pd.read_csv("../input/final-dataset/all_data.csv")
data.head()


# ## Data Cleaning and modifying 

# In[ ]:


nan_df=data[data.isna().any(axis=1)]
nan_df.head()


# In[ ]:


data.info()


# In[ ]:


data=data.dropna(how="all")
data.head()


# In[ ]:


data.tail()


# In[ ]:


data_null=data[data.isnull().any(axis=1)]


# In[ ]:


data["Product"].value_counts()


# In[ ]:


data = data[data["Order Date"].str[0:2]!="Or"]
data.head()


# In[ ]:


data["Month"] = pd.to_datetime(data["Order Date"]).dt.month


# In[ ]:


data.head()


# In[ ]:


def get_city(address):
    return address.split(",")[1].strip()

def get_state(address):
    return address.split(",")[2].strip(" ")[0:2]

data['all_cities']= data["Purchase Address"].apply(lambda x: f"{get_city(x)}({get_state(x)})")


# In[ ]:


data['Quantity Ordered'] = pd.to_numeric(data['Quantity Ordered'])
data['Price Each'] = pd.to_numeric(data['Price Each'])


# In[ ]:


data.head()


# ### Data Exploration

# In[ ]:


data["Month"].value_counts()


# In[ ]:


data["Sales"] = data["Quantity Ordered"]*data["Price Each"]


# In[ ]:


month_wise_sale=data.groupby(["Month"]).sum()


# In[ ]:


month_wise_sale


# In[ ]:


import matplotlib.pyplot as plt
months=range(1,13)

plt.figure(figsize=(12,8))
plt.bar(months,month_wise_sale['Sales'])
plt.xticks(months)
plt.xlabel("Month Number")
plt.ylabel("Sales in USD")
plt.show()


# In[ ]:


city_wise_sale=data.groupby(["all_cities"]).sum()


# In[ ]:


city_wise_sale


# In[ ]:


keys = [city for city, df in data.groupby(['all_cities'])]

plt.figure(figsize=(12,8))
plt.bar(keys,city_wise_sale['Sales'])
plt.xticks(keys,rotation="vertical",size=15)
plt.xlabel("All Cities")
plt.ylabel("Sales in USD")
plt.show()


# In[ ]:


data["Hour"] = pd.to_datetime(data["Order Date"]).dt.hour
data["Minute"] = pd.to_datetime(data["Order Date"]).dt.minute
data["Count"]=1


# In[ ]:


data.head()


# In[ ]:


keys=range(24)


# In[ ]:


plt.plot(keys, data.groupby(['Hour']).count()['Count'])
plt.xticks(keys)
plt.grid()
plt.show()

# My recommendation is slightly before 11am or 7pm


# In[ ]:




