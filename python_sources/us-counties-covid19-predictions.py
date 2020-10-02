#!/usr/bin/env python
# coding: utf-8

# # US, per county, COVID-19 data

# In[ ]:


import pandas as pd

url = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv"
df = pd.read_csv(url)

print(df.head())


# In[ ]:


df.describe()


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


df.corr().style.background_gradient(cmap='coolwarm')


# In[ ]:


df.corr().plot.bar()


# In[ ]:


df.plot(kind='density', subplots=True, layout=(2,2), sharex=False)
plt.show()


# In[ ]:


sns.regplot(x='cases', y='deaths', data=df, logistic=False, color='green')


# In[ ]:


plt.figure(figsize=(16,9)) # Figure size
sns.lineplot(x='date', y='cases', data=df, marker='o', color='red') 
plt.title('Cases per day in the US') # Title
plt.xticks(df.date.unique(), rotation=90) # All values in x-axis; rotate 90 degrees
plt.show()

