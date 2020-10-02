#!/usr/bin/env python
# coding: utf-8

# **Given below is an example of Seaborn's pairplot. We plan to see how other variables effect the target variable using the pairplot visualization technique**

# So we start by importing the needed libraries
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# The next step is to import the dataset.
# 
# After importing the dataset, running the code [dataframe name].head() provides us with a brief look of what the actual dataset looks like.

# In[ ]:


filename='/kaggle/input/top10dataset/top10s.xlsx'
df=pd.read_excel(filename,encoding='ISO-8859-1')
df.head()


# GIven below is a small glimpse of what the seaborn library can do. Seaborn is a powerful Data Visualization library based on its robus parent matplotlib.
# 
# Given below is 

# In[ ]:



#Visualising the variables against popularity using pairplot

plt.figure(figsize=(20,20))
sns.pairplot(df)


# The matrix above consists of various scatterplots. Since we are only concerned with the relationship of popu (popularity) with other variables,we bring our attention the bottom row.
# 
# 
# On the y axis we have popularity while rest of the variables lie on the x axis.
# 
# This helps in visually discoverin any relationship that exists among the variables.
# 
# To point out a few, there is an inverse relationship between popularity and livliness (live) while a energy(nrgy) and popularity have a direct relation.
# 
# Some initial conclusive observations can be made with this simple visualization technique!
