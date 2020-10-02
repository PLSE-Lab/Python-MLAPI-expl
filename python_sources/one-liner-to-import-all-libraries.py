#!/usr/bin/env python
# coding: utf-8

# # Documentation:
# 

# The library can be found at: [https://github.com/8080labs/pyforest]()

# Over the course of time, the need for importing mulitple libraries has become the norm. I, myself, have been through all those tasks, especially the pesky errors of a library not being included in the project. 
# 
# This brought out a thought of finding a way to get all of the libraries imported in one go. After a lot of scouring of the web, I found this library which does it all, yet, it lacked a little bit of documentation and details about which all libraries had been imported. With this, I intend to make a note about all the libraries present in the library. 
# 
# This is basically to unify the experience one might go through while locating this library/ importing all of the libraries when an error throws up. 
# 
# 

# ### Installation:

# In[ ]:


get_ipython().system('pip install pyforest')


# ### Loading the library

# In[ ]:


from pyforest import *


# ## active_imports()

# This feature of the library gives an insight to all the libraries imported currently. 
# Pyforest uses the method of _Lazy Imports_, which signifies that a library is _only_ imported when is has been called into use. 
# 
# 
# LazyImport Example: 
# 
# - As this library removes the use of _import pandas as pd_ {Or any other library}, the pandas library can directly be called with the _pd_ name. 
# - When pd is used once in the program, the Pyforest library calls the _Pandas_ library from itself and imports it automatically. 
# 
# This signifies, that during the beginning of the program, the active_imports() would be an empty list. 
# As soon as any other function/library is used, the library associated with it would import itself within the program. 

# #### Code Snippets:

# In[ ]:


active_imports()


# This is the empty list, signifying that no library has been imported currently, for the program. 
# For brevity, I'll be using the titanic database, as it is rather concise, with clean and formatted data. 

# In[ ]:


df = pd.DataFrame(pd.read_csv('../input/train.csv'))


# As the dataframe is loaded, the pandas library hsa been added to the active_imports() module, as follows;

# In[ ]:


active_imports()


# As *pd* can also be useful for using pandas profiling, that has been called out too. 

# ## List of libraries in the Pyforest library and their calls: 

# ### Legend: 
#  Name - **Call**

# ### Data Exploration
# 
# - Pandas - **pd**
# - Matplotlib.pyplot - **plt**
# - Numpy - **np**
# - Dataframe -  **dd**
# - Seaborn - **sns**
# - Pyplot - **py**
# - Plotly.Express - **px**
# - Plotly.graph_objs - **go**
# - Dash - **dash**
# - Bokeh - **bokeh**
# - Altair - **alt**
# - Pydot - **pydot**
# 

# ### Machine Learning
# - Sklearn - **sklearn**
# - OneHotEncoder - **OneHotEncoder**
# 
# (All the other libraries can be added to the library for further use)
# 
# ### Deep Learning
# 
# - TensorFlow - **tf**
# - Keras - **keras**
# 
# ### NLP
# 
# - NLTK - **nltk**
# - Gensim - **gensim**
# - Spacy - **spacy**
# 
# ### Helper libraries:
# - Sys - **sys**
# - Os - **os**
# - re - **re**
# - glob - **glob**
# - Path - **Path**
# - Pickle - **pickle**
# - DateTime - **dt**
# - TQDM - **tqdm**
# 

# #### Some examples of use: 

# #### Distplot of Age of Survivors:

# In[ ]:


df['Age'].dropna(inplace= True)
sns.distplot(df['Age'])


# #### Pandas Profiling
# 

# A one shot way for basic EDA

# In[ ]:


df.profile_report()


# **To be continued later, for even more extensive documentation**
