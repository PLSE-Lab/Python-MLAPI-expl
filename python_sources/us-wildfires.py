#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt

import sqlite3 as sql
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_profiling as pp
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix, f1_score, classification_report, accuracy_score, roc_auc_score
pd.set_option('display.max_columns',100)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

get_ipython().run_line_magic('matplotlib', 'inline')


# Any results you write to the current directory are saved as output.


# # Predicting WildFire Cause

# **Create Connection**

# In[ ]:


conn = sql.connect("../input/FPA_FOD_20170508.sqlite")


# **Read from Connection and Load it in a dataframe**

# In[ ]:


wfdf = pd.read_sql("Select * from fires",con= conn)


# In[ ]:


wfdf.shape


# In[ ]:


wfdf.head()


# In[ ]:


wfdf.columns


# ## Profile Report

# In[ ]:


pp.ProfileReport(wfdf)


# **Converting date on which fire was controlled and discovery date from julian date to regular format**

# In[ ]:


epoch = pd.to_datetime(0, unit='s').to_julian_date()
wfdf['CONT_DATE'] = pd.to_datetime(wfdf['CONT_DATE'] - epoch, unit='D')
wfdf['DISCOVERY_DATE'] = pd.to_datetime(wfdf['DISCOVERY_DATE'] - epoch, unit = 'D')
wfdf['FIRE_DURATION'] = wfdf['CONT_DATE'] - wfdf['DISCOVERY_DATE']
wfdf['FIRE_DURATION'] = wfdf['FIRE_DURATION'].dt.days
wfdf['FIRE_DURATION'].fillna(wfdf['FIRE_DURATION'].mean(),inplace = True)
wfdf['FIRE_DURATION'] = wfdf['FIRE_DURATION'].apply(lambda x : int(x))


# ## Data analysis

# ### **Fire occurence over a period of time**

# In[ ]:


plt.figure(figsize=(15, 10))
ax = sns.countplot(x="FIRE_YEAR", data = wfdf, palette="YlOrRd", color='white')
ax.set_title("Frequency of wildfire over time", fontdict = {'fontsize':20, 'fontweight':'bold','color':'white'})
ax.set_xlabel("Year", fontdict = {'fontsize':18, 'fontweight': 'medium','color':'white'})
ax.set_ylabel("Frequency", fontdict = {'fontsize':18, 'fontweight': 'medium','color':'white'})
ax.grid(which = 'major',color = 'black', linewidth = 1)


# **As it can be seen most wildfires occured in the year 2006 with count of more than 100,000 widlfires in a single year!**

# ### **Fires by Size**

# In[ ]:


labels = {'B':'0.26-9.9',
          'A':'0-2.5', 
          'C':'10.0-99.9', 
          'D':'100-299', 
          'E':'300-999', 
          'F':'1000-4999', 
          'G': '5000+'}
plt.figure(figsize=(10,6))
ax= sns.countplot(x = 'FIRE_SIZE_CLASS', data = wfdf, palette = 'YlOrRd_r', order = wfdf['FIRE_SIZE_CLASS'].value_counts().index)
ax.set_xlabel('Classes of Wildfires', fontdict = {'fontsize':18, 'fontweight': 'medium'})
ax.set_ylabel('Count of Wildfires', fontdict = {'fontsize':18, 'fontweight': 'medium'})
ax.set_title('Count of fire by size', fontdict = {'fontsize':20, 'fontweight': 'bold'})
ax.grid(linewidth = 1, color = 'black')
ax.set_xticklabels(labels.values())


# **It can be seen from the counplot that most of the wildfires spread from 0.26 to 9.9 acres of land**

# ### **Cause of Wild Fire**

# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.countplot(y = 'STAT_CAUSE_DESCR', data = wfdf, palette = 'YlOrRd_r', order=wfdf['STAT_CAUSE_DESCR'].value_counts().index)
ax.set_ylabel('Cause of Wildfires', fontdict = {'fontsize':18, 'fontweight': 'medium'})
ax.set_xlabel('Frequency of wildfires because of the cause', fontdict = {'fontsize':18, 'fontweight': 'medium'})
ax.set_title('Causes of wildfire', fontdict = {'fontsize':20, 'fontweight': 'bold'})
ax.grid()


# **In most cases the cause of fire is Debris Building**

# ### **Countries which are most and least fire prone**

# In[ ]:


plt.figure(figsize = (15,10))
ax = sns.countplot(x = 'STATE' , data = wfdf, palette = 'YlOrRd_r', order = wfdf['STATE'].value_counts().index)
ax.set_ylabel('Frequency of wildfires', fontdict = {'fontsize':18, 'fontweight': 'medium'})
ax.set_xlabel('States', fontdict = {'fontsize':18, 'fontweight': 'medium'})
ax.set_title('Most and least fire prone states', fontdict = {'fontsize':20, 'fontweight': 'bold'})
ax.grid()
plt.tight_layout()


# **California is most wildfire prone area**

# In[ ]:


grid = sns.FacetGrid(data= wfdf , col = 'FIRE_YEAR', height = 4, aspect = 2, col_wrap=3)
grid.map(sns.countplot, 'STATE', order = wfdf['STATE'].unique(), palette = 'YlOrRd_r')
grid.set_xticklabels(rotation = 90, fontdict = {'fontsize':10})
grid.set(ylim = (0,12000))
for ax in grid.axes.flat:
    ax.set_title(ax.get_title(), fontsize='xx-large')
    ax.set_ylabel(ax.get_ylabel(), fontsize='xx-large')
    ax.set_xlabel(ax.get_xlabel(),fontsize = 'xx-large')
# plt.tight_layout()


# **As I live in California I am interested to check which city exactly is fire prone**

# In[ ]:


ca_data = wfdf[wfdf['STATE'] == 'CA']
plt.figure(figsize=(15,10))
plt.scatter(x = ca_data['LONGITUDE'],y = ca_data['LATITUDE'],sizes = ca_data['FIRE_SIZE']/100,c = ca_data['FIRE_DURATION']/100, cmap='YlOrRd', vmin=0.0, vmax=1.0)
plt.colorbar()
plt.xlabel('LONGITUDE', fontdict={'fontsize' : 18, 'fontweight' : 'medium'})
plt.ylabel('LATITUDE', fontdict={'fontsize' : 18, 'fontweight' : 'medium'})
plt.title("California's fire prone areas ",fontdict={'fontsize' : 20, 'fontweight' : 'bold'})
plt.show()


# **The circular spots represent the firesize that is the spread of fire in acres and the color internsity that is as we move from 
# yellow to red scale the duration to bring the fire under control increases, which can be also looked as the number days required to bring the fire under control is directly proportional to how intense the fire was**

# **As it can be seen on 2nd of Feb the main reason for wildfire at South Dakota is Arson**

# In[ ]:


X = wfdf[['STATE', 'FIPS_NAME', 'FIRE_DURATION', 'FIRE_SIZE_CLASS']]
y = wfdf['STAT_CAUSE_DESCR']


# **Since the chosen feature 'STATE' is categorical variable I need to convert it to numerical variable. So I'll be performing one hot encoding to convert 'STATE' to numerical variable**

# In[ ]:


label_encoder = LabelEncoder()


# In[ ]:


X['STATE'] = label_encoder.fit_transform(X['STATE'])


# In[ ]:


X['FIPS_NAME'] = label_encoder.fit_transform(X['FIPS_NAME'].astype('str'))


# In[ ]:


X['FIRE_SIZE_CLASS'] = label_encoder.fit_transform(X['FIRE_SIZE_CLASS'])


# In[ ]:


X.head()


# In[ ]:


y = X['FIRE_SIZE_CLASS']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.20, random_state= 0)


# In[ ]:


logclf = LogisticRegression(random_state=0, solver='lbfgs',multi_class= 'multinomial', max_iter=500).fit(X_train, y_train)


# In[ ]:


y_pred = logclf.predict(X_test)


# In[ ]:


accuracy_score(y_test, y_pred)


# In[ ]:


print(classification_report(y_test, y_pred))


# In[ ]:


y_pred = label_encoder.inverse_transform(y_pred)

