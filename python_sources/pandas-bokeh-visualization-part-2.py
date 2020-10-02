#!/usr/bin/env python
# coding: utf-8

# ## **Introduction:**

# ### **This Kernal is the Part 2 of the Pandas-Bokeh Visualisation done on various datasets like iris and Bird Recordings. For Better Understanding of Pandas-Bokeh Checkout my Previous Kernals .**

# ![18775489_G.webp](attachment:18775489_G.webp)

# ***Pandas-Bokeh* provides a Bokeh plotting backend for Pandas, GeoPandas and Pyspark DataFrames, similar to the already existing Visualization feature of Pandas. Importing the library adds a complementary plotting method plot_bokeh() on DataFrames and Series.**
# 
# **With *Pandas-Bokeh*, creating stunning, interactive, HTML-based visualization is as easy as calling:**
# 
# **df.plot_bokeh()**
# 
# ***Pandas-Bokeh* also provides native support as a Pandas Plotting backend for Pandas >= 0.25. When Pandas-Bokeh is installed, switchting the default Pandas plotting backend to Bokeh can be done via:**
# 
# **pd.set_option('plotting.backend', 'pandas_bokeh')**

# ## Importing Libraries

# In[ ]:


import numpy as np
import pandas as pd
import pandas_bokeh
import seaborn as sns
import matplotlib.pyplot as plt
pandas_bokeh.output_notebook()
pd.set_option('plotting.backend', 'pandas_bokeh')
# Create Bokeh-Table with DataFrame:
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.models import ColumnDataSource


# In[ ]:


df = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")


# In[ ]:


df.head()


# ### **Scatter Plot - Petal_Length Vs Sepal_Width**

# In[ ]:


data_table = DataTable( columns=[TableColumn(field=Ci, title=Ci) for Ci in df.columns],source=ColumnDataSource(df),height=300) # Setting Up Data Table

scatter = df.plot_bokeh.scatter(x="petal_length", y="sepal_width", category="species", title="Iris Dataset Visualization", show_figure=True) # Scatter Plot

pandas_bokeh.plot_grid([[data_table, scatter]], plot_width=400, plot_height=350)# Scatter Plot + Data Table Visuals


# ## **Bar Plot**

# In[ ]:


df.plot_bokeh.bar(xlabel="petal_length",ylabel="sepal_width",alpha=0.6,figsize=(2000,800),title="petal_length Vs Sepal_length")


# ### **Stacked Bar Plotting**

# In[ ]:


df.plot_bokeh.bar(xlabel="petal_length",ylabel="sepal_width",alpha=0.6,figsize=(2000,800),title="petal_length Vs Sepal_length",stacked=True)


# ## **Now let us look at a more practical example of housing prices problem to understand it better.** 

# In[ ]:


df = pd.read_csv("../input/house-prices-advanced-regression-techniques/housetrain.csv",index_col='SalePrice')# Loading Dataset.....
df.head()


# In[ ]:


df.describe()


# In[ ]:


numeric_features = df.select_dtypes(include=[np.number])

p_bar = numeric_features.plot_bokeh.bar(ylabel="Sale Price", figsize=(1000,800),title="Housing Prices", alpha=0.6)# Ploting the Bar Plot


# In[ ]:


df=pd.read_csv("../input/xeno-canto-bird-recordings-extended-a-m/train_extended.csv")


# In[ ]:


df.head()


# In[ ]:


plt.figure(figsize=(16,8))
sns.scatterplot(x='longitude', y='latitude', data=df)
plt.grid()
plt.show()


# ## **Do Upvote my kernel if you find my insights helpful.**
# 
# **I would covering these approaches in future:**
# 
# 1. GeoViews and HoloViews
# 2. PieChart Visuals
