#!/usr/bin/env python
# coding: utf-8

# # COVID19 India - Bar Chart Race

# ### Create a bar chart race for the confirmed COVID19 cases by states in India.

# ## Install Bar Chart Race

# In[ ]:


# Install package in the current Jupyter kernel
import sys
get_ipython().system('{sys.executable} -m pip install bar_chart_race')


# ## 1. Import Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bar_chart_race as bcr
from IPython.display import HTML
import warnings
warnings.filterwarnings("ignore")
from IPython.display import Video


# ## 2. Load Data

# In[ ]:


covid_data_complete = pd.read_csv('../input/covid19-in-india/covid_19_india.csv', index_col=False)
covid_data_complete.head()


# ## 3. Inspect the dataframe

# In[ ]:


covid_data_complete.info()


# In[ ]:


covid_data_complete.isnull().sum()


# ## 4. Drop Columns and modify dataframe

# In[ ]:


# change datatype of date to a pandas datetime format
covid_data_complete['Date'] = pd.to_datetime(covid_data_complete['Date'], dayfirst=True)
#covid_data_complete["Date"] = covid_data_complete["Date"].apply(pd.to_datetime)

#drop columns other than total_cases
drop_cols = ['Sno', 'Time', 'ConfirmedIndianNational', 'ConfirmedForeignNational', 'Cured', 'Deaths']
covid_data_complete.drop(covid_data_complete[drop_cols],axis=1,inplace=True)


covid_data_complete = covid_data_complete[covid_data_complete['State/UnionTerritory'] 
                                          != 'Cases being reassigned to states']

#dropping data before Feb 29 2020 as only Kerala had 3 cases till then in a span of one month
covid_data_complete = covid_data_complete[covid_data_complete['Date'] >
                                          pd.to_datetime(pd.DataFrame({'year': [2020],'month': [2],'day': [28]}))[0]]

covid_data_complete.head()


# In[ ]:


covid_data = covid_data_complete.copy() #make a copy for analysis
covid_data.columns = ['Date', 'States', 'Cases'] #rename columns
covid_data.head(10)


# In[ ]:


total_countries = covid_data['States'].nunique()
total_countries


# ## 5. Arrange the dataframe for creating bar chart race

# In[ ]:


# set countries and date as index and find cases
# transpose the dataframe to have countries as columns and dates as rows
covid_data_by_date = covid_data.set_index(['States','Date']).unstack()['Cases'].T.reset_index()

covid_data_by_date = covid_data_by_date.set_index('Date') #make date as index - desired by barchartrace

covid_data_by_date = covid_data_by_date.fillna(0) #fill na with 0

covid_data_by_date


# ## 6. Create the BarChartRace and save file as mp4

# In[ ]:


#make the mp4 file with the BarChartRace and save it
df = covid_data_by_date
bcr.bar_chart_race(
    df=df,
    filename='India_Covid19_BarChartRace.mp4',
    orientation='h',
    sort='desc',
    n_bars=10,
    fixed_order=False,
    fixed_max=False,
    steps_per_period=10,
    interpolate_period=False,
    label_bars=True,
    bar_size=.95,
    period_label={'x': .99, 'y': .25, 'ha': 'right', 'va': 'center'},
    period_fmt='%B %d, %Y',
    period_summary_func=lambda v, r: {'x': .99, 'y': .05,
                                      's': f'Total cases: {v.nlargest(total_countries).sum():,.0f}\n\nDeveloper: Akhil James',
                                      #'s': '',
                                      'ha': 'right', 'size': 10, 'family': 'Courier New'},
    perpendicular_bar_func='median',
    period_length=1000,
    figsize=(5, 3),
    dpi=500,
    cmap='dark24',
    title='COVID-19 cases in India',
    title_size=10,
    bar_label_size=10,
    tick_label_size=10,
    shared_fontdict={'color' : '.1'},
    scale='linear',
    writer=None,
    fig=None,
    bar_kwargs={'alpha': .7},
    filter_column_colors=True)  


# ## Video file is in the output section.
