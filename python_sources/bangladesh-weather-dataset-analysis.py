#!/usr/bin/env python
# coding: utf-8

# **Import Libraries**

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# **Read Dataset**

# In[ ]:


data_bdwd = pd.read_csv('../input/bangladesh-weather-dataset/Temp_and_rain.csv')


# **Check Dataset**

# In[ ]:


data_bdwd


# In[ ]:


data_bdwd.head(8)


# In[ ]:


data_bdwd.columns


# In[ ]:


data_bdwd.describe()


# In[ ]:


print('Total Entries: ', str(len(data_bdwd)))


# In[ ]:


data_bdwd['tem'].hist().plot()


# In[ ]:


data_bdwd['rain'].hist().plot()


# In[ ]:


data_bdwd.plot('Year', 'tem')


# In[ ]:


data_bdwd.plot('Year', 'rain')


# In[ ]:


print(max(data_bdwd['tem']))


# In[ ]:


print(min(data_bdwd['tem']))


# In[ ]:


data_bdwd.loc[data_bdwd.loc[:, 'tem'] == 29.526, :]


# **May 1979(05/1979)**,
# **Temperature: 29.526**,
# **Which is the max temperature in this dataset.**

# In[ ]:


data_bdwd.loc[data_bdwd.loc[:, 'tem'] == 16.8006, :]


# **January 1978(01/1978)**,
# **Temperature: 16.8006**,
# **Which is the min temperature in this dataset.**

# In[ ]:


print(max(data_bdwd['rain']))


# In[ ]:


data_bdwd.loc[data_bdwd.loc[:, 'rain'] == 1012.02, :]


# **August 2011(08/2011) Rain 1012.02, which is the max rain in the data set**

# In[ ]:


print(min(data_bdwd['rain']))


# In[ ]:


data_bdwd.loc[data_bdwd.loc[:, 'rain'] == 0.0, :]


# **This is not useable!!!!!!!!!!! As there were many days, where no rain happens**

# In[ ]:


sns.countplot(x = 'tem', data = data_bdwd)


# In[ ]:


sns.countplot(x = 'rain', data = data_bdwd)


# In[ ]:


data_bdwd.info()


# In[ ]:


data_bdwd.isnull()


# In[ ]:


data_bdwd.isnull().sum()


# In[ ]:


sns.heatmap(data_bdwd.isnull(), cmap = 'viridis')


# **No null data in this dataset**

# In[ ]:


data_bdwd.plot.scatter(x = 'Year', y = 'rain', c = 'tem', colormap='viridis')


# In[ ]:


data_bdwd.plot.scatter(x = 'Year', y = 'rain', c ='red')


# In[ ]:


data_bdwd.plot.scatter(x = 'Year', y = 'tem', c ='blue')

