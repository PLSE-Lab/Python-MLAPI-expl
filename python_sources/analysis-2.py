#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 


# In[ ]:


data=pd.read_excel('../input/covid19-india/Complete COVID-19_Report in India.xlsx')
data.head()


# In[ ]:


data.columns


# In[ ]:


data['Date Announced'].value_counts()


# In[ ]:


x = data['Date Announced'].value_counts()


# # Graph showing rate of increase!

# In[ ]:


from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
plt.plot(x)
plt.show()


# # Alert: Men be careful. Just Work from Home! LoL!

# In[ ]:


sns.set()


# # Let's Check the Gender distribution of the patients affected

# In[ ]:


gender_info = data['Gender'].value_counts()
gender_info.plot(kind='bar') #Lots of missing values.


# # Graph showing the Current state of patients in India

# In[ ]:


x1 = data['Current Status'].value_counts()
x1.plot(kind='bar')


# In[ ]:


x1 #24 Deceased or Dead


# In[ ]:


x2=data['Nationality'].value_counts() #Affected patients in India 
x2.plot(kind='bar')


# In[ ]:


x2


# In[ ]:


df = data
plot_df = df.dropna()
plt.hist(plot_df['Age Bracket'])


# In[ ]:


plot_df.head()


# In[ ]:


plot_df.shape


# OMG! This is disastrous! If any row with atleast one missing value or NAN was present, the entire column will be eliminated. However, in our usecase, only 10 patients data was completely filled, with no single missing value. 
