#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pylab 
import scipy.stats as stats


# In[ ]:


fire=pd.read_csv("../input/forestfires.csv")


# In[ ]:


fire.head()


# # A
# ### Month, Day are Categorical (Ordinal ) Attribute
# ### X, Y, FFMC, DMC, DC, ISI, Temp, RH, wind, rain and are are continous(interval-scalled) attribute

# # B
# ### Five number Summary for Continous Variable

# In[ ]:


fire['X'].describe()


# In[ ]:


fire['Y'].describe()


# In[ ]:


fire['FFMC'].describe()


# In[ ]:


fire['DMC'].describe()


# In[ ]:


fire['DC'].describe()


# In[ ]:


fire['ISI'].describe()


# In[ ]:


fire['temp'].describe()


# In[ ]:


fire['RH'].describe()


# In[ ]:


fire['wind'].describe()


# In[ ]:


fire['rain'].describe()


# # Mode for categorical attribute

# In[ ]:


fire['day'].mode()


# In[ ]:


fire['month'].mode()


# # C
# ### Mean snd standard deviation for Continous attribute

# In[ ]:


fire['X'].mean()


# In[ ]:


fire['X'].std()


# In[ ]:


fire['Y'].mean()


# In[ ]:


fire['Y'].std()


# In[ ]:


fire['FFMC'].mean()


# In[ ]:


fire['FFMC'].std()


# In[ ]:


fire['DMC'].mean()


# In[ ]:


fire['DMC'].std()


# In[ ]:


fire['DC'].mean()


# In[ ]:


fire['DC'].std()


# In[ ]:


fire['ISI'].mean()


# In[ ]:


fire['ISI'].std()


# In[ ]:


fire['temp'].mean()


# In[ ]:


fire['temp'].std()


# In[ ]:


fire['RH'].mean()


# In[ ]:


fire['RH'].std()


# In[ ]:


fire['wind'].mean()


# In[ ]:


fire['wind'].std()


# In[ ]:


fire['rain'].mean()


# In[ ]:


fire['rain'].std()


# # D
# ### Quantile plot for temp and wind

# In[ ]:


x=fire['temp']
y=fire['wind']
stats.probplot(x,dist="norm", plot=pylab)
pylab.show()
stats.probplot(y, dist="norm", plot=pylab)
pylab.show()


# # E
# ### Histogram

# In[ ]:


plt.hist2d(x,y)


# # F
# ### Scatter plot

# In[ ]:


sns.jointplot(x,y)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




