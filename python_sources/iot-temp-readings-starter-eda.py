#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ![Meme](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTYTun5NvzTs9qTAy5zuN13vyyv8lwQNPMqxACQWeDcY8GUG7S0)

# In[ ]:


#import packages
import pandas as pd
from pandas import datetime
import numpy as np


#to plot within notebook
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')
# Above is a special style template for matplotlib, highly useful for visualizing time series data

import seaborn as sns


#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10,10


# In[ ]:


# import Dataset

df = pd.read_csv('../input/temperature-readings-iot-devices/IOT-temp.csv', parse_dates=['noted_date'])

df.name = 'IOT'


# In[ ]:


# converting index to datetime format.

df["Date"] = pd.to_datetime(df["noted_date"])


# In[ ]:


df.head(10)
# useless features encountered which can be dropped.


# In[ ]:


cols_drop = ['id', 'noted_date', 'room_id/id']


# In[ ]:


# dropping columns

df = df.drop(cols_drop, axis=1)


# In[ ]:


df.head(10)

# duplicate rows encountered in the dataset


# In[ ]:


df.dtypes


# In[ ]:


print("the dataset has shape = {}".format(df.shape))


# In[ ]:


rows_drop = ['temp', 'out/in', 'Date'] 

# dropping ALL duplicte rows with all same values.

df.drop_duplicates(subset = rows_drop, 
                     keep = False, inplace = True)


# In[ ]:


df.describe()

# duplicate rows have been dropped


# In[ ]:


# building new features for time stamp.

def features_build(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df.Date.dt.month
    df['Day'] = df.Date.dt.day
    df['WeekOfYear'] = df.Date.dt.weekofyear
features_build(df)


# In[ ]:


ax = sns.scatterplot(x="Month", y="temp", hue="out/in", data=df)

# plotting discrete tempt values for month time stamp.


# ## other presentations and trends in the dataset are below:

# In[ ]:


get_ipython().run_cell_magic('HTML', '', "<div class='tableauPlaceholder' id='viz1575283714956' style='position: relative'>\n<noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Te&#47;TemperaturePlotsfromIOTData&#47;Dashboard1&#47;1_rss.png' style='border: none' />\n</a></noscript>\n<object class='tableauViz'  style='display:none;'>\n<param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />\n<param name='embed_code_version' value='3' /> \n<param name='site_root' value='' />\n<param name='name' value='TemperaturePlotsfromIOTData&#47;Dashboard1' />\n<param name='tabs' value='yes' /><param name='toolbar' value='no' />\n<param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Te&#47;TemperaturePlotsfromIOTData&#47;Dashboard1&#47;1.png' />\n<param name='animate_transition' value='yes' />\n<param name='display_static_image' value='yes' />\n<param name='display_spinner' value='yes' />\n<param name='display_overlay' value='yes' />\n<param name='display_count' value='yes' />\n</object>\n</div>                \n<script type='text/javascript'>\nvar divElement = document.getElementById('viz1575283714956');\nvar vizElement = divElement.getElementsByTagName('object')[0];\nvizElement.style.minWidth='420px';vizElement.style.maxWidth='1150px';vizElement.style.width='100%';\nvizElement.style.minHeight='583px';vizElement.style.maxHeight='883px';\nvizElement.style.height=(divElement.offsetWidth*0.75)+'px';\nvar scriptElement = document.createElement('script');                    \nscriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';     \nvizElement.parentNode.insertBefore(scriptElement, vizElement);          \n</script>")


# ### You can check this Tableau presentation in detail from this link: [here](https://public.tableau.com/views/TemperaturePlotsfromIOTData/Dashboard2?:display_count=y&:toolbar=n&:origin=viz_share_link)
# 
# ### P.S. *You can also visit my tableau [profile](https://public.tableau.com/profile/atul.anand3150#!/) to check my other VIZs*.
# 

# #### Thus, these presentations also verify our intuition observed from the Tableau observation. We already plotted similar plots on tableau.
# 
# **Datas verified:**
# 
# 1. max tempt: 51*C
# 2. min tempt: 21*C
# 3. Highest Tempt(month): September
# 4. outside tempt. > inside tempt. (anomalies excluded)
# 5. tempt gap(between outside and inside) converges in middle of year.

# #### Please, comment below how much you liked this  kernel. feedbacks are heartily welcomed!
# 
# **P.S.** *This is just a starter kernel. So, I haven't put much effort and tried to keep it short and simple.*
# 
# ## PLease UPVOTE if you liked the kernel. 
# 
# # THANKS! :-)
