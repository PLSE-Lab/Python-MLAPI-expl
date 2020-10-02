#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 

from datetime import datetime
from sklearn.linear_model import LinearRegression

import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# # Loading data

# In[ ]:


# import Dataset
df = pd.read_csv('../input/temperature-readings-iot-devices/IOT-temp.csv', parse_dates=['noted_date'])
df.head()


# # Data processing
# ## Removing useless columns

# In[ ]:


df['room_id/id'].value_counts()


# In[ ]:


# dropping columns
cols_drop = ['id', 'room_id/id']
df = df.drop(cols_drop, axis=1)


# ## Table reshape

# In[ ]:


print("the dataset has shape = {}".format(df.shape))


# In[ ]:


df.describe()
# duplicate rows have been dropped


# ### Time features
# Here I want to round time of measurement up to hours. After I will get information about mean temperature for each measure hour for inside and outside. It will allow to know about relation between Inside and outside in the moment.

# In[ ]:


# building new features for time stamp.
df['measure_hour'] = df.noted_date.apply(lambda x:datetime.strftime(x,'%Y-%m-%d %H:00:00'))


# ### Getting the new data table
# Here I will get table with hour of measurement as index, and mean valuse of inside and outside temperatures in this hour. After I will drop all rows with at leas one NaN value.

# In[ ]:


data = df.groupby(['measure_hour','out/in']).temp.mean().reset_index()
data = data.pivot(index = 'measure_hour',columns = 'out/in', values = 'temp').reset_index().dropna()
data.head()


# ## Visualization 
# Let's see at distributions of temperatures for all measurements:

# In[ ]:


fig, ax = plt.subplots(figsize = (6,4))
g = sns.distplot(data.In, label = 'In')
g = sns.distplot(data.Out, label = 'Out')
plt.legend()
g.set_xlabel('Temperature')


# And our goal - relationship between inside and outside temperatures

# In[ ]:


sns.scatterplot(x =data.Out, y = data.In)


# We can make a guess that people turn on the conditioner if outside temperature in higher than 35. So, can say, that inside temperature is in interval [28;35] if outside temperature is higher than 35. Lets take a look on piece of data with outside temperature above 35:

# In[ ]:


sns.scatterplot(x =data[data.Out>35].Out, y = data[data.Out>35].In)


# But we can see some obvious dependense inside of outside for outisde in interval [25;35]. We defenetly can say that there is no obvious dependence:

# In[ ]:


sns.scatterplot(x =data[data.Out<=35].Out, y = data[data.Out<=35].In)


# So, after removing from set values with outside temperature above 35, we can see linear dependence. And we can use linear regression to predict inside  temperature with known outside value.

# In[ ]:


linear = data[(pd.notna(data.Out))&(pd.notna(data.In)) & (data.Out<35)]


# I'll remove some outliers here:

# In[ ]:


#removing oultliers
linear = linear.drop(index = linear[((linear.Out>32)&(linear.In<30)|(linear.Out<25))].index, axis = 0)
sns.scatterplot(x = linear.Out, y = linear.In)


# # Linear regression

# In[ ]:


#Linear regression building
model = LinearRegression()
model.fit(linear[['Out']],linear.In)


# In[ ]:


k,b = model.coef_[0],model.intercept_
print(k,b)
sns.scatterplot(x = linear.Out, y = linear.In)
reg_line = np.linspace(25,35,100)
plt.plot(reg_line, reg_line*k + b)


# # Conclusion
# So, I would describe the relationship between inside and outside temperatures like this:
# 
# $$
# Inside(Outside) = 0.89*Outside+3,\ {if}\  Outside \in [25,35], \\
# Inside(Outside) \in [28;35], {if}\ Outside \in (35;\infty)
# $$
