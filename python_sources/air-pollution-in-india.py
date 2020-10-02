#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# To take care of warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the pd.readinput directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/data.csv", encoding = "ISO-8859-1")
data.head()


# In[ ]:


state_replace_dict = {
    'state' : {
        r'Uttaranchal': 'Uttarakhand',
    }
}


# In[ ]:


data.replace(state_replace_dict,regex=True, inplace=True)
data['state'].unique()


# In[ ]:


# To check null values we can use heatmap
sns.heatmap(data.isnull(),yticklabels=False,cbar=False, cmap='viridis')


# In[ ]:


data['type'].unique()


# In[ ]:


data.info()


# In[ ]:


# As stn_code and agency are having so many null values and sampling date is also not that 
# useful as we are having date field. And, as pm2_5 is one of the most important field in this dataset but it is having 
# so many null values, so has to remove it.
# So, we can remove these columns.
data.drop(['stn_code','agency','sampling_date','pm2_5','location_monitoring_station'], inplace = True, axis=1)
data.head()


# In[ ]:


# Now we will work on missing values, i.e NaN values.
from sklearn.impute import SimpleImputer

# For so2 and no2 we are taking median of all values by state.
data['so2'] = data.groupby('state')['so2'].transform(lambda x: x.fillna(x.median()))
data['no2'] = data.groupby('state')['no2'].transform(lambda x: x.fillna(x.median()))


# In[ ]:


# For rspm and spm we are taking mean of all values.
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(data.iloc[:,5:7].values)
#imputer = imputer.fit(X[['Age', 'Salary']])
data.iloc[:,5:7] = imputer.transform(data.iloc[:,5:7].values)
data.info()


# In[ ]:


# Now for 'type'
# 'type' is having very few missing values so we can fill the nan values
# with most common occuring,i.e, Residential, Rural and other Areas.

data['type'] = data['type'].fillna('Residential, Rural and other Areas')
data.info()


# In[ ]:


data.head()


# **Year wise Analysis of Pollutants**

# In[ ]:


# Adding new column for year.
data['date'] = pd.to_datetime(data['date'])
data['year'] = data['date'].dt.year
data.head()


# In[ ]:


import matplotlib.pyplot as plt
#import plotly.express as px
plt.rcParams['figure.figsize'] = (10, 5)

# SO2 Trend in India by year
sns.lineplot(x='year', y='so2', data=data).set_title("SO2 Trend in India by year")


# In[ ]:


# NO2 Trend in India by year
sns.lineplot(x='year', y='no2', data=data).set_title("NO2 Trend in India by year")


# In[ ]:


# RSPM Trend in India by year
sns.lineplot(x='year', y='rspm', data=data).set_title("RSPM Trend in India by year")


# In[ ]:


# SPM Trend in India by year
sns.lineplot(x='year', y='spm', data=data).set_title("SPM Trend in India by year")


# We can observe considerable sudden increase in the concentration of RSPM and NO2 from year 2003 to 2004.
# But we can observe slight decrease in the concentration of all the pollutants from 2 years.

# **Statewise Analysis of Pollutant**

# In[ ]:


pollutant_statewise = data.groupby('state').mean()[['so2', 'no2', 'rspm', 'spm']]
pollutant_statewise.head()


# In[ ]:


# For NO2
NO2_statewise = pollutant_statewise.sort_values(by = 'no2', ascending=False)
NO2_statewise_Top10 = NO2_statewise['no2'].head(10)
NO2_statewise_Bottom10 = NO2_statewise['no2'].tail(10)
NO2_statewise_Top10.plot.bar(color='m')
plt.show()
NO2_statewise_Bottom10.plot.bar(color='b')
plt.show()


# In[ ]:


# For SO2
NO2_statewise = pollutant_statewise.sort_values(by = 'so2', ascending=False)
NO2_statewise_Top10 = NO2_statewise['so2'].head(10)
NO2_statewise_Bottom10 = NO2_statewise['so2'].tail(10)
NO2_statewise_Top10.plot.bar(color='m')
plt.show()
NO2_statewise_Bottom10.plot.bar(color='b')
plt.show()


# In[ ]:


# For RSPM
NO2_statewise = pollutant_statewise.sort_values(by = 'rspm', ascending=False)
NO2_statewise_Top10 = NO2_statewise['rspm'].head(10)
NO2_statewise_Bottom10 = NO2_statewise['rspm'].tail(10)
NO2_statewise_Top10.plot.bar(color='m')
plt.show()
NO2_statewise_Bottom10.plot.bar(color='b')
plt.show()


# In[ ]:


# For SPM
NO2_statewise = pollutant_statewise.sort_values(by = 'spm', ascending=False)
NO2_statewise_Top10 = NO2_statewise['spm'].head(10)
NO2_statewise_Bottom10 = NO2_statewise['spm'].tail(10)
NO2_statewise_Top10.plot.bar(color='m')
plt.show()
NO2_statewise_Bottom10.plot.bar(color='m')
plt.show()


# From the statewise analysis we can observe that industrialization and huge amount of vehicles are giving a greater threat to our environment.
# All the big and developed states which are having industrialization at it's boom are surrounded by these deadly pollutants, like our capital( or should I say pollution capital) Delhi, West Bengal, Haryana, Uttra Pradesh.
# On the other side, parts of North,North-East, and some states of south are having really less amount of pollutants as these states are still very close to nature.

# **Areawise Analysis of Pollutant**

# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(30,14))
ax = sns.barplot("so2", y="type",
                 data=data,
                 ax=axes[0,0]
                )
ax = sns.barplot("no2", y="type",
                 data=data,
                 ax=axes[0,1]
                )
ax = sns.barplot("rspm", y="type",
                 data=data,
                 ax=axes[1,0]
                )
ax = sns.barplot("spm", y="type",
                 data=data,
                 ax=axes[1,1]
                )


# **Analysis on the basis of year and state**

# In[ ]:


# For SO2
f, ax = plt.subplots(figsize=(15,15))
sns.heatmap(data.pivot_table('so2', index='state', columns=['year'],aggfunc='median',margins=True),
           annot=True,cmap="YlGnBu").set_title('SO2 for Year and State')


# In[ ]:


# For NO2
f, ax = plt.subplots(figsize=(15,15))
sns.heatmap(data.pivot_table('no2', index='state', columns=['year'],aggfunc='median',margins=True),
           annot=True,cmap="YlGnBu").set_title('NO2 for Year and State')


# From above analysis we can observe year by year increase in pollution for each state.
# Delhi having large numbers for NO2 but very low for SO2, which tells us that main reason for pollution in our capital is road traffic and fossil fuels combustion.

# In[ ]:




