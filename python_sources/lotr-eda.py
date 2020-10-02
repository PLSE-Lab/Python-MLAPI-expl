#!/usr/bin/env python
# coding: utf-8

# # Let's begin our adventure to **Misty Mountains** on Middle Earth!

# > **Far over the misty mountains cold **
# <br>
# > **To dungeons deep and caverns old**
# <br>
# > **We must away ere break of day**
# <br>
# > **To find our long-forgotten gold!**

# ![LOTR](https://wallpapercave.com/wp/PQVpQko.jpg)

# In[ ]:


get_ipython().system('pip install chart-studio')


# In[ ]:


# Data processing libraries
import numpy as np 
import pandas as pd 

# Visualization libraries
import datetime
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

# Plotly visualization libraries
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import chart_studio.plotly as py
from plotly.graph_objs import *
from IPython.display import Image
pd.set_option('display.max_rows', None)

import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ### Read LOTR data into dataframes

# In[ ]:


char_df = pd.read_csv('../input/lord-of-the-rings-data/lotr_characters.csv')
script_df = pd.read_csv('../input/lord-of-the-rings-data/lotr_scripts.csv')


# ### Get the metadata about the datasets

# In[ ]:


char_df.head()


# In[ ]:


script_df.head()


# ### Group the characters from the scripts by movies
# In order to scale the data, bring the data in log scale

# In[ ]:


records = script_df.groupby(['movie']).size()
records = records.sort_values()

grouped_df = pd.DataFrame(records)

grouped_df['Count'] = pd.Series(records).values
grouped_df['Movies'] = grouped_df.index
grouped_df['Log Count'] = np.log(grouped_df['Count'])
grouped_df.head()


# **Bar Chart**

# In[ ]:


fig = go.Figure(go.Bar(
    x = grouped_df['Movies'],
    y = grouped_df['Log Count'],
    text=['Bar Chart'],
    name='LOTR Movies',
    marker_color=grouped_df['Count']
))

fig.update_layout(
    height=800,
    title_text='Movies distribution in the LOTR Trilogy',
    showlegend=True
)

fig.show()


# ### **Visualize the distribution of characters on Pie Chart**

# In[ ]:


char_df.head()


# ### Group by gender

# In[ ]:


gender_df = char_df[['gender','name', 'spouse']]
gender_df.head()


# In[ ]:


gen_df = gender_df.groupby('gender')['name'].value_counts().reset_index(name='count')
gen_df['count'] = gender_df.groupby('gender')['name'].transform('size')
gen_df.head()


# ### Count the characters present across all the genders

# In[ ]:


test_df = gender_df
df = test_df.groupby(['gender'], as_index=False, sort=False)['name'].count()
df.head()


# In[ ]:


fig = px.pie(df, values='name', names='gender')
fig.show()


# ### Visualize the character composition in LOTR

# In[ ]:


tdf = char_df.groupby(['race'], as_index=False, sort=False)['name'].count()
tdf.head()


# In[ ]:


fig = px.pie(tdf, values='name', names='race')
fig.show()


# In[ ]:


char_df.head()


# ### Analyze the scripts for the triology

# In[ ]:


script_df.head()


# ### Count the number of occurences of each character in the dialogues across triology

# In[ ]:


sdf = script_df.groupby('char')['movie'].value_counts().reset_index(name='count')
sdf['count'] = script_df.groupby('char')['movie'].transform('size')

sdf.head()


# In[ ]:


fig = px.pie(sdf, values='count', names='char')
fig.show()

