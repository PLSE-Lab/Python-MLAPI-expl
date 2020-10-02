#!/usr/bin/env python
# coding: utf-8

# # London Police Records (Descriptive Analysis)
# 
# The goal of this notebook is to provide a descriptive analysis about the crimes reported to the London Police
# 
# **Developed by**: Jhonnatan Torres
# 
# **Note**: English is not my primary language, I apologize in advance for any typo or grammar mistake
# 
# ---

# The 1st step is to load some required libraries and look into the datasets available in the directory, the file "london-street.csv" was chosen for this analysis

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


sns.set_style('darkgrid')
plt.style.use('seaborn-darkgrid')


# In[ ]:


from dask.distributed import Client
client = Client()


# In[ ]:


import dask.dataframe as dd


# Reading the data using Dask

# In[ ]:


df = dd.read_csv('/kaggle/input/london-police-records/london-street.csv')


# In[ ]:


df.count().compute()


# There are more than 2.9M reports to the police in the file, based in the column names, there is information about the location of the crime or report, the type of crime, and the month of the report. Let's take a look to the first 5 rows in the dataset

# In[ ]:


df.head()


# In[ ]:


df['COUNTER']=1


# An additional column with a "1" is included in order to facilitate some operations and calculations

# In[ ]:


df['YEAR'] = df['Month'].apply(lambda x: str(x).split('-')[0],meta=('Month', 'object'))


# In[ ]:


df.columns


# The **year** was extracted from the **Month** column, however, this new variable it is not going to be used in this analysis because the information by year is not complete, the reports started on June 2014 (midyear) 

# In[ ]:


df.tail()


# In[ ]:


plt.figure(figsize=(12,6))
df.groupby('Month').count().compute()['COUNTER'].plot.line(marker='o')
plt.title("Total Number of Reports by Year and Month")


# The monthly volume of reports was fluctuating, however, it looks like there was an upward trend since May 2015

# In[ ]:


df['Month'].nunique().compute()


# In[ ]:


total_reports_crimes=df.groupby('Crime type').count().compute()['COUNTER'].sort_values(ascending=False)
total_reports_crimes = total_reports_crimes.reset_index()


# In[ ]:


total_reports_crimes['%Rep'] = total_reports_crimes['COUNTER']/np.sum(total_reports_crimes['COUNTER'])
total_reports_crimes['%Pareto'] = np.cumsum(total_reports_crimes['%Rep'])
total_reports_crimes


# Based in the table above, **Anti-social behaviour**, **Violence and sexual offences** and **Other theft** are accountable for **55%** of the crimes reported to the London Police

# In[ ]:


crimes_df=df.groupby(['Crime type','Month']).count().compute()['COUNTER'].reset_index()


# In[ ]:


crimes_df.head()


# In[ ]:


chart = sns.catplot(x="Month",y='COUNTER',col="Crime type",col_wrap=4,kind='bar',data=crimes_df,palette='winter')
chart.set_xticklabels(rotation=90)


# There are two points to highlight based in the chart above:
# 1. There is a seasonal effect in the number of reports related to **Anti-social behaviour**
# 2. There is an upward trend in the number of reports related to **Violence and Sexual Offences**

# In[ ]:


df = df.categorize('Crime type')
c_matrix=df.pivot_table(index='Month',columns='Crime type',values='COUNTER',aggfunc='sum').compute()
c_matrix.head()


# In[ ]:


crime_vso_locations=df[df['Crime type']=="Violence and sexual offences"].groupby(['Latitude','Longitude']).sum().compute()['COUNTER'].reset_index()
crime_vso_locations.sort_values(by='COUNTER',ascending=False,inplace=True)
crime_vso_locations=crime_vso_locations.head(25)


# In[ ]:


import plotly.express as px
fig=px.scatter_mapbox(crime_vso_locations, lat="Latitude", lon="Longitude",color="COUNTER",size="COUNTER",
                      title="Top 25 locations with the highest volume of reports related to Violence and Sexual Offences",
                      color_continuous_scale=px.colors.sequential.thermal)
fig.update_layout(mapbox_style="carto-positron")
fig.show()
#another mapbox styles available:
#open-street-map
#carto-darkmatter


# The scatter map above is showing the Top 25 locations with the highest volume of **Violence and Sexual Offence** reports to the London Police. Unfortunately, I am not familiar with the London Geography, but based in the data, the coordinates linked with the highest number of reports are near to *Southhall* and *Stratford*

# In[ ]:


crime_asb_locations=df[df['Crime type']=="Anti-social behaviour"].groupby(['Latitude','Longitude']).sum().compute()['COUNTER'].reset_index()
crime_asb_locations.sort_values(by='COUNTER',ascending=False,inplace=True)
crime_asb_locations=crime_asb_locations.head(25)


# In[ ]:


import plotly.express as px
fig=px.scatter_mapbox(crime_asb_locations, lat="Latitude", lon="Longitude",color="COUNTER",size="COUNTER",
                      title="Top 25 locations with the highest volume of reports related to Antisocial Behaviour",
                      color_continuous_scale=px.colors.sequential.Plasma)
fig.update_layout(mapbox_style="carto-positron")
fig.show()


# The scatter map above is showing the Top 25 locations with the highest volume of **Antisocial Behaviour** reports to the London Police. Based in the data, the coordinates linked with the highest number of reports are near to *Covent Garden* and *Ilford*
# 
# It was noticed a seasonal behaviour in the number of reports, with a spike on June, hence, could be assumed this increase is related to a sport or public event

# ## Final Comments
# 
# * Using DASK, it was possible to read and analyze around 3M records available in the dataset, the most common crime types were found and even, the locations with the highest volume of reports were shown in a map
# 
# * This information could be used by the Police in order to allocate more Officers / Security Cameras or by Politicians in order to define the basis for a new campaign
# 
