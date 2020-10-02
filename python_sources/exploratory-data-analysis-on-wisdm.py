#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import re
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

plt.rcParams["figure.figsize"] = (20,10)

RANDOM_SEED = 42


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Exploratory Data Analysis (EDA) on WIreless Sensor Data Mining Dataset.

# On this notebook, we gonna try to visualize WISDM (WIreless Sensor Data Mining) and get insights from it.
# 
# This data has been released by the Wireless Sensor Data Mining
# (WISDM) Lab. <http://www.cis.fordham.edu/wisdm/>
# 

# **About the Dataset:**
# 
# The "WISDM Smartphone and Smartwatch Activity
# and Biometrics Dataset" includes data collected from
# 51 subjects, each of whom were asked to perform 18
# tasks for 3 minutes each. Each subject had a smartwatch
# placed on his/her dominant hand and a
# smartphone in their pocket. The data collection was
# controlled by a custom-made app that ran on the
# smartphone and smartwatch.
# 
# The attributes collected are : x,y and z.
# these attributes are the sensors data collected from mobile phones.
# 

# <img src="https://www.mathworks.com/help/supportpkg/android/ref/simulinkandroidsupportpackage_galaxys4_accelerometer.png" width="240" height="240" align="center"/>
# 

# Let's import the data and start.

# In[ ]:


columns = ['user','activity','timestamp', 'x-axis', 'y-axis', 'z-axis']
df = pd.read_csv('/kaggle/input/wisdmdata/WISDM_ar_v1.1_raw.txt', header = None, names = columns)


# In[ ]:


df=df.dropna()
df.head()


# In[ ]:


df.info()


# The z axies column looks like its being identified by pandas as an object, let's transform it to float!

# In[ ]:


df['z-axis'] = df['z-axis'].map(lambda x: str(re.findall("\d+\.\d+", str(x))))
df['z-axis'] = df['z-axis'].map(lambda x: x[2:-2])
df['z-axis'] = pd.to_numeric(df['z-axis'],errors='coerce')


# In[ ]:


df.info()


# Now our data is good, let's start discovering.
# First lets see how balance is our data.

# In[ ]:


df['activity'].value_counts().plot(kind='bar', title='Number of Examples by Activities',color=['b','r','g','y','k','r']);


# It looks like we have a lot of walking and jogging data, while a little bit of sitting and standing data,
# The data is unbalance, but let's see what gonna happend.

# Let's see how much data is given for each user.

# In[ ]:


df['user'].value_counts().plot(kind='bar', title='Number of Examples by User',color=['r','y','g','b']);


# The examples for each user is balanced somehow.

# I guess not much can be extracted from this plots, let's check the activities signlas and see how it works.

# In[ ]:


def plot_activity(activity, df):
    data = df[df['activity'] == activity][['x-axis', 'y-axis', 'z-axis']][:200]
    axis = data["x-axis"].plot(subplots=True, 
                     title=activity,color="b")
    axis = data["y-axis"].plot(subplots=True, 
                 title=activity,color="r")
    axis = data["z-axis"].plot(subplots=True, 
             title=activity,color="g")
    for ax in axis:
        ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))


# In[ ]:


plot_activity("Sitting", df)


# For Sitting Activity, We notice that there is no intersection between the axies, due the less frequency of the the signals and also when someone is setting, the sensors tend to be more stable than other activities.

# In[ ]:


plot_activity("Standing", df)


# The x and z axis are intersecting in standing, that's a good sign to differenciate between standing and sitting, since in both activities, the sensors are tend to be stable, but for standing, we can notice that the x & z axis are more sensitive.

# In[ ]:


plot_activity("Walking", df)


# whoaa, he's just walking and there's a chaos in the plot already, it means that the sensors are so sensetive, which is a good thing, the more sensitive, the more accurate to describe an activity.
# for the y axis, the wavelength is a bit wider, while the ampliture is not that big.

# In[ ]:


plot_activity("Jogging", df)


# Unlike walking, the ampliture is maximal, while the wave length is too small for jogging.

# Let's see how it looks like on the Stairs.

# In[ ]:


plot_activity("Upstairs", df)


# In[ ]:


plot_activity("Downstairs", df)


# When we're going upstairs, we can notice that the y axis signal is a bit away from the other 2, while in downstairs the 3 axis signals are intersecting in some period of time.

# Let's check how the values are correlated for some activities.

# In[ ]:


plt.rcParams["figure.figsize"] = (15,7)


# In[ ]:


def plot_corr(activity, df):
    corr = df[df["activity"]==activity].corr()
    corr = corr[["x-axis","y-axis","z-axis"]][2:5]
    ax = sns.heatmap(
        corr, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=100)    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    );


# In[ ]:


plot_corr("Walking",df)


# The y and x are negativly correlated, and y and z are positivly correlated, while z and x axis are not correlated

# In[ ]:


plot_corr("Jogging",df)


# we can notice that z and x axis are not correlated at all, which is a good thing for training our model, also in Jogging activity, the x and y and y and z are positivly correlated.

# # Conclusion

# We applied EDA Techniques on the WISDM Dataset to get insights that may help us on the creation of the model, and the parameters tuning.

# In[ ]:




