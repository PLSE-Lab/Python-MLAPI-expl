#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# Greetings from the Kaggle bot! This is an automatically-generated kernel with starter code demonstrating how to read in the data and begin exploring. If you're inspired to dig deeper, click the blue "Fork Notebook" button at the top of this kernel to begin editing.

# ## Exploratory Analysis
# To begin this exploratory analysis, first import libraries and define functions for plotting the data using `matplotlib`. Depending on the data, not all plots will be made. (Hey, I'm just a simple kerneling bot, not a Kaggle Competitions Grandmaster!)

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from plotly.offline import init_notebook_mode, iplot 
import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px
import pycountry
py.init_notebook_mode(connected=True)

# Graphics in retina format 
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

# Increase the default plot size and set the color scheme
plt.rcParams['figure.figsize'] = 8, 5

# Disable warnings in Anaconda
import warnings
warnings.filterwarnings('ignore')


# There is 1 csv file in the current version of the dataset:
# 

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# The next hidden code cells define functions for plotting data. Click on the "Code" button in the published kernel to reveal the hidden code.

# Now you're ready to read in the data and use the plotting functions to visualize the data.

# In[ ]:


nRowsRead = 1000 # specify 'None' if want to read whole file
# Coursera AI GSI Percentile and Category.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df = pd.read_csv('/kaggle/input/Coursera AI GSI Percentile and Category.csv', delimiter=',', nrows = nRowsRead)
df.dataframeName = 'Coursera AI GSI Percentile and Category.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')


# Let's take a quick look at what the data looks like:

# In[ ]:


df.head()


# In[ ]:


df.info()


# Distribution graphs (histogram/bar graph) of sampled columns:

# In[ ]:


plt.figure(figsize=(20,12))

sns.countplot(x="region", data=df)


# In[ ]:


plt.figure(figsize=(20,12))

sns.countplot(x="incomegroup", data=df)


# In[ ]:


plt.figure(figsize=(20,12))

sns.countplot(x="region", data=df,hue='percentile_category')


# In[ ]:


plt.figure(figsize=(20,12))
sns.distplot(df['percentile_rank'], hist=True, kde=True)


# In[ ]:


df.competency_id.value_counts()


# In[ ]:


# creating different dataframes based on the competency Ids
df_AI = df[df['competency_id'] == 'artificial-intelligence']
df_Stats_prog = df[df['competency_id'] == 'statistical-programming']
df_Stats = df[df['competency_id'] == 'statistics']
df_SE = df[df['competency_id'] == 'software-engineering']
df_Math = df[df['competency_id'] == 'fields-of-mathematics']
df_ML = df[df['competency_id'] == 'machine-learning']


# In[ ]:


def map(data):
    """
    function to plot a world map of the competency ids, distributed regionwise
    """
    
    fig = go.Figure(data=go.Choropleth(
        locations = data['iso3'],
        z = data['percentile_rank'],
        text = data['percentile_category'],
        colorscale = "Rainbow",
        autocolorscale=False,
        reversescale=True,
        marker_line_color='darkgray',
        marker_line_width=0.5,
        #colorbar_tickprefix = '$',
        colorbar_title = 'Skill Index (1 is highest)'))

    fig.update_layout(
            title_text= data['competency_id'].iloc[0].title() +" "+'Skill Index in 2019',
            geo=dict(
                  showframe=False,
                  showcoastlines=False,
                  projection_type='equirectangular'))

    fig.show()


# In[ ]:


map(df_AI)


# **thanks https://www.kaggle.com/parulpandey for this ** !!

# In[ ]:




