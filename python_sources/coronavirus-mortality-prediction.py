#!/usr/bin/env python
# coding: utf-8

# # Tracking the mortality of 2019 Coronavirus

# In[ ]:


get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.prototype._should_scroll = function(lines) {\n    return false;\n}')


# In[ ]:


import datetime as dt
dt_string = dt.datetime.now().strftime("%d/%m/%Y")
print(f"Kernel last updated: {dt_string}")
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns 
import datetime as dt
import folium
from folium.plugins import HeatMap, HeatMapWithTime
get_ipython().run_line_magic('matplotlib', 'inline')
print(os.listdir('/kaggle/input'))


# In[ ]:


data_df = pd.read_csv("../input/coronavirus-2019ncov/covid-19-all.csv")
world_population = pd.read_csv('../input/population-by-country-2020/population_by_country_2020.csv')


# In[ ]:


print(f"Rows: {data_df.shape[0]}, Columns: {data_df.shape[1]}")


# In[ ]:


data_df.head()


# In[ ]:


data_df.tail()


# In[ ]:


world_population.head()


# In[ ]:


for column in data_df.columns:
    print(f"{column}:{data_df[column].dtype}")


# In[ ]:


print(f"Date - unique values: {data_df['Date'].nunique()} ({min(data_df['Date'])} - {max(data_df['Date'])})")


# In[ ]:


data_df['Date'] = pd.to_datetime(data_df['Date'])


# In[ ]:


for column in data_df.columns:
    print(f"{column}:{data_df[column].dtype}")


# In[ ]:


print(f"Date - unique values: {data_df['Date'].nunique()} ({min(data_df['Date'])} - {max(data_df['Date'])})")


# In[ ]:


def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))


# In[ ]:


missing_data(data_df)


# In[ ]:


all_countries = data_df['Country/Region'].unique()
print(f"Countries/Regions:{all_countries}")


# In[ ]:


data_all_wd = pd.DataFrame(data_df.groupby(['Country/Region', 'Date'])['Confirmed',  'Recovered', 'Deaths'].sum()).reset_index()
data_all_wd.columns = ['Country', 'Date', 'Confirmed', 'Recovered', 'Deaths' ]
data_all_wd = data_all_wd.sort_values(by = ['Country','Date'], ascending=False)


# In[ ]:


def logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0))) + 1


# In[ ]:


import datetime
import scipy
def plot_logistic_fit_data(d_df, title, population):
    d_df = d_df.sort_values(by=['Date'], ascending=True)
    print(d_df.tail())
    print()
    d_df['x'] = np.arange(len(d_df)) + 1
    d_df['y'] = d_df['Deaths']

    x = d_df['x']
    y = d_df['y']

    c2 = scipy.optimize.curve_fit(logistic,  x,  y,  p0=(100000, 0.2, 70) )
    #y = logistic(x, L, k, x0)
    popt, pcov = c2
    print("Predicted L (the maximum number of confirmed deaths): " + str(int(popt[0])))
    print("Predicted k (growth rate): " + str(float(popt[1])))
    print("Predicted x0 (the day of the inflexion): " + str(int(popt[2])) + "")
#     print(*popt)
    x = range(1,d_df.shape[0] + int(popt[2]))
    
    y_fit = logistic(x, *popt)
    size = 3 
    f, ax = plt.subplots(1,1, figsize=(4*size,2*size))
    g = sns.scatterplot(x=d_df['x'], y=(d_df['y'] / population), label='Confirmed deaths (included for fit)', color='red')
    g = sns.lineplot(x=x, y=(y_fit / population), label='Predicted values', color='green')
    plt.xlabel('Days since first death')
    plt.ylabel(f'deaths per million of population')
    plt.title(f'Confirmed deaths & predicted evolution (logistic curve) per million population: {title}')
    plt.xticks(rotation=90)
    ax.grid(color='black', linestyle='dotted', linewidth=0.75)
    plt.show()


# In[ ]:


pop = world_population['Population (2020)'].sum()
population = float(pop) / 1e6
print(f"")
print(f"World Population {population} million")
print(f"")

data_xx = data_all_wd

d_df = data_xx.copy()
d_df = d_df.resample('D', on='Date').sum()

try:
    plot_logistic_fit_data(d_df, 'World', population)
except Exception as e:
    print(e)


# In[ ]:


countries = ['US', 'Ireland', 'United Kingdom', 'France', 'Spain', 'Italy', 'Germany', 'Denmark',  'New Zealand', 'Turkey', 'Iceland', 'Norway', 
             'Sweden', 'Finland', 'Israel', 'Taiwan', 'South Korea', 'Japan', 'Russia', 'China', 'Iran', 'Portugal', 'Brazil'  ]

list.sort(countries)
pop_country_replacements = {'US': 'United States' }

for country in countries:
    pop_country = country if country not in pop_country_replacements else pop_country_replacements[country]
    pop = world_population[world_population['Country (or dependency)']==pop_country]['Population (2020)']
    population = float(pop) / 1e6
    print(f"")
    print(f"{country} Population {population} million")
    print(f"")
          
    data_xx = data_all_wd[data_all_wd['Country']==country]

    d_df = data_xx.copy()
    try:
        plot_logistic_fit_data(d_df, country, population)
    except Exception as e:
        print(e)

