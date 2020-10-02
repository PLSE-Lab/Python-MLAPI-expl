#!/usr/bin/env python
# coding: utf-8

# # Graphical representation of missing values with missingno

# ## Contents 
# 
# 1.  [Matrix Plotting](#matrixPlotting)
# 2. [Dendrograms](#dendrograms) 
# 3. [Heat Map](#heatMap)
# 4. [Bar Graph](#barGraph) 
# 

# ### I came across an interesting module recently which visualizes the missing values and wanted to see how it works and its beautiful! So I went and made a small kernel about it. I took the wikipedia movie plots to work on. It has 34886 rows and 8 columns.

# #### Click [here](https://github.com/ResidentMario/missingno) to go to its github repo. 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import missingno as msno


# ### Importing and reading the dataset

# In[ ]:


wiki_df = pd.read_csv("../input/wiki_movie_plots_deduped.csv")


# In[ ]:


wiki_df.head()


# In[ ]:


wiki_df.shape


# In[ ]:


wiki_df.describe()


# In[ ]:


wiki_df.info()


# ### Replacing Unknown, unknown and NaN values with np.NaN

# In[ ]:


wiki_df["Director"] = wiki_df["Director"].apply(lambda x: np.NaN if x == "Unknown" else x)


# In[ ]:


wiki_df["Cast"] = wiki_df["Cast"].apply(lambda x: np.NaN if x == "NaN" else x)


# In[ ]:


wiki_df["Genre"] = wiki_df["Genre"].apply(lambda x: np.NaN if x == "unknown" else x)


# In[ ]:


wiki_df.info()


# <h3 id="matrixPlotting">Plotting with missingno matrix</h3>

# #### See [more](https://github.com/ResidentMario/missingno#matrix). 

# In[ ]:


msno.matrix(wiki_df)


# In[ ]:


# wiki_df.Cast


# In[ ]:


wiki_df[wiki_df["Title"].duplicated(keep="first")==True].head()


# In[ ]:


wiki_df[wiki_df["Wiki Page"].duplicated(keep="first")==True].head()


# <h3 id="dendrograms"> Plotting with dendrograms</h3> 

# ####  It splits using the minimum distance between the clusters that are created using hierarchial clustering algorithm. See [more](https://github.com/ResidentMario/missingno#dendrogram).

# In[ ]:


msno.dendrogram(wiki_df)


# <h3 id="heatMap">Heat Map</h3>

# #### See [more](https://github.com/ResidentMario/missingno#heatmap).

# In[ ]:


msno.heatmap(wiki_df)


# <h3 id="barGraph">Bar Graph</h3>

# #### See [more](https://github.com/ResidentMario/missingno#bar-chart).

# In[ ]:


msno.bar(wiki_df)

