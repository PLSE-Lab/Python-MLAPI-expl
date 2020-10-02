#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv("../input/crisis-data.csv")


# In[ ]:


data.head()


# In[ ]:


#checking how many registered incidents we have in this dataset
final_call_counts = data.groupby(['Final Call Type']).count()
print(final_call_counts)


# In[ ]:


#what's the most commons dispositions in this dataset?
data['Disposition'].value_counts().plot(kind='pie', figsize=(10, 10));


# In[ ]:


#checking if the index 'Use of Force' is well distributed
data['Use of Force Indicator'].value_counts().plot(kind='pie', figsize=(10, 10));


# In[ ]:


#checking the precinct with most occurrences
data['Precinct'].value_counts().plot(kind='bar')


# In[ ]:


#creating a bar plot of some occurrences types
plt.bar([1, 2, 3, 4, 5], [69, 3, 8, 8, 5], tick_label=['suicide', 'rape', 'animal complaints', 'traffic', 'shoplift - theft'])
plt.title('Some Occurrences Types')
plt.xlabel('Types')
plt.ylabel('Quantity');

