#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb   #seaborn to enhance the visualizations of matplotlib
sb.set()


# In[ ]:


cereals = pd.read_csv("../input/cereal.csv")


# ### Numerical data analysed

# In[ ]:


cereals.describe()


# In[ ]:


cereals.head()


# In[ ]:


mfr_counts = cereals['mfr'].value_counts()  #product counts for each unique company


# # Companies included in data and their number of broducts

# In[ ]:


mfr_counts.plot(kind='bar')
plt.xlabel("Companies")
plt.ylabel("Products")
plt.title("PRODUCTS DIVERSITY")
#plt.xticks([])  #remove x_labels and x_ticks (not available in seaborn.set),


# # Mean rating for all manufacturers

# In[ ]:


K_mean=cereals[cereals['mfr'] == 'K']['rating'].mean()


# In[ ]:


AHFP_mean=cereals[cereals['mfr'] == 'A']['rating'].mean()


# In[ ]:


GM_mean=cereals[cereals['mfr'] == 'G']['rating'].mean()


# In[ ]:


Nab_mean=cereals[cereals['mfr'] == 'N']['rating'].mean()


# In[ ]:


P_mean=cereals[cereals['mfr'] == 'P']['rating'].mean()


# In[ ]:


Quaker_mean=cereals[cereals['mfr'] == 'Q']['rating'].mean()


# In[ ]:


RP_mean=cereals[cereals['mfr'] == 'R']['rating'].mean()


# ### Rating Plot

# In[ ]:


plotx=[AHFP_mean, GM_mean, K_mean, Nab_mean, P_mean, Quaker_mean, RP_mean]


# In[ ]:


plt.bar(cereals.mfr.unique(), plotx)
plt.title('Average rating of products')
plt.xlabel("Company")
plt.ylabel("Mean rating")

