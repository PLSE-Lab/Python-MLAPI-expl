#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[ ]:


file_path = '../input/Iris.csv'

iris_data = pd.read_csv(file_path, index_col= 'Id')

iris_data.head()


# In[ ]:





# In[ ]:


#scatter plot
plt.figure(figsize = (10,6))
sns.scatterplot(x = iris_data['SepalLengthCm'], y= iris_data['SepalWidthCm'], data= iris_data, hue= iris_data['Species'])
plt.show()


# In[ ]:





# In[ ]:


#supress warnings
import warnings
warnings.filterwarnings('ignore')

#dist plot
plt.figure(figsize = (10,8))
sns.distplot(iris_data['SepalLengthCm'], label="Iris-setosa")

plt.legend()

plt.show()


# In[ ]:




