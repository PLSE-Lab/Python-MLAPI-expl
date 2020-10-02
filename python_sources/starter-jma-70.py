#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import os 
import pandas as pd

dat = pd.read_csv('../input/arduinosensorvalue/ArduinoSensorValues.csv')


# In[ ]:


dat.head()


# In[ ]:


dat.describe()


# In[ ]:


plt.matshow(dat.corr())
plt.colorbar()
plt.show()


# In[ ]:


plt.scatter(dat['gravity_x'], dat['gravity_y'])


# In[ ]:


plt.scatter(dat['gravity_x'], dat['gravity_z'])


# In[ ]:


plt.scatter(dat['gravity_y'], dat['gravity_z'])


# In[ ]:


sns.regplot(x=dat['gravity_x'], y=dat['gravity_y'])


# In[ ]:


sns.regplot(x=dat['gravity_x'], y=dat['gravity_z'])


# In[ ]:


sns.regplot(x=dat['gravity_y'], y=dat['gravity_z'])


# In[ ]:


plt.scatter(dat['accelerometer_x'], dat['accelerometer_y'])


# In[ ]:


plt.scatter(dat['accelerometer_x'], dat['accelerometer_y'])


# In[ ]:


plt.scatter(dat['accelerometer_y'], dat['accelerometer_z'])


# In[ ]:


sns.regplot(x=dat['accelerometer_x'], y=dat['accelerometer_y'])


# In[ ]:


sns.regplot(x=dat['accelerometer_x'], y=dat['accelerometer_z'])


# In[ ]:


sns.regplot(x=dat['accelerometer_y'], y=dat['accelerometer_z'])


# In[ ]:


p = dat.hist(figsize = (20,20))


# In[ ]:


sns.set(style="whitegrid")
ax = sns.swarmplot(x="accelerometer_x", y="accelerometer_y", data=dat)

