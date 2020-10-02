#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
print("Setup Complete")


# In[ ]:


my_filepath = "../input/arctic-sea-ice-19792015/SeaIce.csv"
my_data = pd.read_csv(my_filepath)


# In[ ]:


my_data.head()


# In[ ]:


sns.scatterplot(x=my_data['Year'], y=my_data['Extent'], hue=my_data['Area'])

