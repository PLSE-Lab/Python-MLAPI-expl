#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph.
import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import warnings
warnings.filterwarnings('ignore')
from pylab import rcParams
# figure size in inches
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('/kaggle/input/mobile-data-speeds-of-all-india-during-march-2018/march18_myspeed.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.rename(columns={'Service Provider': 'Operator','Data Speed(Mbps)': 'Throughput'},inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.groupby('Technology')['Throughput'].max().sort_index()


# In[ ]:


#reference from https://www.kaggle.com/anuragk240/visualising-data-speeds
columns = ['Technology', 'Test_type', 'Operator', 'LSA']
for c in columns:
    v = df[c].unique()
    g = df.groupby(by=c)[c].count().sort_values(ascending=True)
    r = np.arange(len(v))
    print(g.head)
    plt.figure(figsize = (6, len(v)/2 +1))
    plt.barh(y = r, width = g.head(len(v)))
    total = sum(g.head(len(v)))
    print(total)
    for (i, u) in enumerate(g.head(len(v))):
        plt.text(x = u + 0.2, y = i - 0.08, s = str(round(u/total*100, 2))+'%', color = 'green', fontweight = 'bold')
    plt.margins(x = 0.2)
    plt.yticks(r, g.index)
    plt.show()


# # Throughput Analysis across the Telecom Circles
# 
# * Maximum upload and download throughput recorded was at Himachal Pradesh circle
# * Minimum upload and download throughput recorded is 0 Mbps
# * Delhi has very low average throughput in upload
# * North East has very low average throughput in download

# In[ ]:


rcParams['figure.figsize'] = 16, 8
width = 0.25 
# Plotting the bars
x = df.groupby('LSA')['Throughput'].mean().sort_values()
x_indexes = np.arange(len(x.index))
y = df[df["Test_type"]=="Upload"].groupby('LSA')['Throughput'].mean().sort_values()
z = df[df["Test_type"]=="Download"].groupby('LSA')['Throughput'].mean().sort_values()
plt.bar(x_indexes-width,y, width,label="Average Upload") 
plt.bar(y.index,z, width,label="Average Download") 
plt.bar(x_indexes+width,x, width,label="Average Combined") 
plt.title("Throughput accross Circles")
plt.ylabel('Throughput in Mbps')
plt.style.use('seaborn-pastel')
plt.xticks(rotation = 90)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


fig = plt.figure()
rcParams['figure.figsize'] = 16, 8
plt.subplot(2, 2, 1)
width = 0.25 
# Plotting the bars
x = df[df["Test_type"]=="Download"].groupby('LSA')['Throughput'].max().sort_values()
x_indexes = np.arange(len(x.index))
y = df[df["Test_type"]=="Download"].groupby('LSA')['Throughput'].mean().sort_values()
plt.title("Download Throughput accross country")
plt.bar(y.index,y, width,label="Average") 
plt.bar(x_indexes+width,x, width,label="Maximum") 
plt.ylabel('Throughput in Mbps')
plt.style.use('seaborn-pastel')
plt.xticks(rotation = 90)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.subplot(2, 2, 2)
u = df[df["Test_type"]=="Upload"].groupby('LSA')['Throughput'].max().sort_values()
x_indexes = np.arange(len(u.index))
v = df[df["Test_type"]=="Upload"].groupby('LSA')['Throughput'].mean().sort_values()
plt.title("Upload Throughput accross country")
plt.bar(v.index,v, width,label="Average") 
plt.bar(x_indexes+width,u, width,label="Maximum") 
plt.ylabel('Throughput in Mbps')
plt.style.use('seaborn-pastel')
plt.xticks(rotation = 90)
plt.grid(True)
plt.legend()
plt.tight_layout()


# # Data across the Telecom circles

# In[ ]:


#download data
x,y


# In[ ]:


#upload data
u,v


# In[ ]:


fig = plt.figure()
rcParams['figure.figsize'] = 16, 8
width = 0.25 
# Plotting the bars

y = df[df["Test_type"]=="Download"].groupby('Operator')['Throughput'].max().sort_values()
z = df[df["Test_type"]=="Upload"].groupby('Operator')['Throughput'].max().sort_values()
x_indexes = np.arange(len(y.index))
plt.title("Throughput accross country")
plt.bar(y.index,y, width,label="Maximum Download") 
plt.bar(x_indexes+width,z, width,label="Maximum Upload") 
plt.ylabel('Throughput in Mbps')
plt.style.use('seaborn-pastel')
plt.xticks(rotation = 90)
plt.grid(True)
plt.legend()
plt.tight_layout()


# # Operator throughput analysis
# 
# * Maximum throughput accross all the circles is maintained by Jio
# * JIO has highest throughput for 4G networks as well

# In[ ]:


fig = plt.figure()
rcParams['figure.figsize'] = 16, 8
width = 0.25 
# Plotting the bars
y = df[df["Technology"]=="4G"].groupby('Operator')['Throughput'].max().sort_values()
x_indexes = np.arange(len(y.index))
z = df[df["Technology"]=="4G"].groupby('Operator')['Throughput'].mean().sort_values()
plt.title("4G Throughput accross country / Operator")
plt.bar(y.index,y, width,label="Maximum ") 
plt.bar(x_indexes+width,z, width,label="Average") 
plt.ylabel('Throughput in Mbps')
plt.style.use('seaborn-pastel')
plt.xticks(rotation = 90)
plt.grid(True)
plt.legend()
plt.tight_layout()


# In[ ]:


def float_signal_strength(x):
    if x == "na":
        return np.NaN
    else:
        return float(x)
df["Signal_strength"] = df["Signal_strength"].apply(lambda x: float_signal_strength(x))


# In[ ]:


df.info()


# In[ ]:


df["Signal_strength"].fillna(df["Signal_strength"].mean(),inplace=True)


# # Signal strength vs Throughput

# In[ ]:


#reference from https://www.kaggle.com/anuragk240/visualising-data-speeds

import matplotlib.colors as colors

fig = plt.figure()
rcParams['figure.figsize'] = 16, 8
x = df['Signal_strength']
y = df['Throughput']
plt.hist2d(x, y, bins = 40, norm=colors.LogNorm())
plt.ylabel('Data Speed(Mbps)')
plt.xlabel('Signal_strength')
plt.style.use('seaborn-pastel')
plt.xticks(rotation = 90)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

