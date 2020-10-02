#!/usr/bin/env python
# coding: utf-8

# <h1>Tracking the spread of 2019 Coronavirus</h1>
# 
# <img src="https://storage.googleapis.com/kaggle-datasets-images/544069/992803/500beb47c451ac68fae29a8eb95ae45c/dataset-card.jpg" width=400></img>
# 
# # Introduction
# 
# The 2019-nCoV is a highly contagious coronavirus that originated from Wuhan (Hubei province), Mainland China. This new strain of virus has striked fear in many countries as cities are quarantined and hospitals are overcrowded.
# 
# We are using here a Kaggle Dataset [Coronavirus 2019-nCoV](https://www.kaggle.com/gpreda/coronavirus-2019ncov) updated daily, based on [John Hopkins data](https://github.com/CSSEGISandData/COVID-19/). 
# 
# The Kernel will be rerun frequently to reflect the daily evolution of the cited dataset.
# 
# We start by analyzing the data for Mainland China, where the pandemic originated. We show time evolutions and snapshots of Confirmed, Recovered cases as well as Deaths. Then we move to explore the evolution of the pandemics in the rest of the World.
# 
# For both Mainland China and the rest of the World we are also showing the snapshot and time evolution of mortality, calculated in two ways: as Deaths / Confirmed cases (most probably a underestimate) and as Deaths / Recovered cases (most probably an overestimate).
# 

# In[ ]:


import datetime as dt
dt_string = dt.datetime.now().strftime("%d/%m/%Y")
print(f"Kernel last updated: {dt_string}")


# # Analysis preparation
# 
# ## Load packages

# In[ ]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns 
import datetime as dt
import folium
from folium.plugins import HeatMap, HeatMapWithTime
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load the data
# 
# There are multiple files in the coronavirus data folder, we will take the last updated one.
# We also include GeoJSON data for China and for World.

# In[ ]:


print(os.listdir('/kaggle/input'))
DATA_FOLDER = "/kaggle/input/coronavirus-2019ncov"
print(os.listdir(DATA_FOLDER))
GEO_DATA = "/kaggle/input/china-regions-map"
print(os.listdir(GEO_DATA))
WD_GEO_DATA = '/kaggle/input/python-folio-country-boundaries'
print(os.listdir(WD_GEO_DATA))


# In[ ]:


data_df = pd.read_csv(os.path.join(DATA_FOLDER, "covid-19-all.csv"))
cn_geo_data = os.path.join(GEO_DATA, "china.json")
wd_geo_data = os.path.join(WD_GEO_DATA, "world-countries.json")


# # Preliminary data exploration

# ## Glimpse the data
# 
# We check data shape, we look to few rows of the data, we check for missing data.

# In[ ]:


print(f"Rows: {data_df.shape[0]}, Columns: {data_df.shape[1]}")


# In[ ]:


data_df.head()


# In[ ]:


data_df.tail()


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


# Let's look to the missing values.

# In[ ]:


missing_data(data_df)


# Let's check the spread of the 2019-nCoV in various Regions/Countries and Provinces/States.

# In[ ]:


print(f"Countries/Regions:{data_df['Country/Region'].nunique()}")
print(f"Province/State:{data_df['Province/State'].nunique()}")


# ## Load geo data
# 
# Let's check the GeoJSON data first.

# In[ ]:


ch_map = folium.Map(location=[35, 100], zoom_start=4)

folium.GeoJson(
    cn_geo_data,
    name='geojson'
).add_to(ch_map)

folium.LayerControl().add_to(ch_map)

ch_map


# In[ ]:


wd_map = folium.Map(location=[0,0], zoom_start=2)

folium.GeoJson(
    wd_geo_data,
    name='geojson'
).add_to(wd_map)

folium.LayerControl().add_to(wd_map)

wd_map


# # Mainland China
# 
# We start by exploring the data in Mainland China, where the epidemics first apeared.   
# 
# Let's group the data from China on `Province/State`.

# In[ ]:


data_cn = data_df.loc[data_df['Country/Region']=="China"]
data_cn = data_cn.sort_values(by = ['Province/State','Date'], ascending=False)


# We will show the last updated values for confirmed cases, deaths and recovered cases, grouped by province/state in Mainland China.

# In[ ]:


filtered_data_last = data_cn.drop_duplicates(subset = ['Province/State'],keep='first')


# In[ ]:


def plot_count(feature, value, title, df, size=1):
    f, ax = plt.subplots(1,1, figsize=(4*size,4))
    df = df.sort_values([value], ascending=False).reset_index(drop=True)
    g = sns.barplot(df[feature][0:20], df[value][0:20], palette='Set3')
    g.set_title("Number of {} - first 20 by number".format(title))
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    plt.show()    


# ## Mainland China - total cases

# In[ ]:


plot_count('Province/State', 'Confirmed', 'Confirmed cases (last updated)', filtered_data_last, size=4)


# In[ ]:


plot_count('Province/State', 'Recovered', 'Recovered cases (last updated)', filtered_data_last, size=4)


# In[ ]:


plot_count('Province/State', 'Deaths', 'Death cases (last updated)', filtered_data_last, size=4)


# Now we will show again the confirmed cases, deaths and recovered cases, grouped by province/state in Mainland China, as evolved in time.

# In[ ]:


def plot_time_variation(df, y='Confirmed', hue='Province/State', size=1, is_log=False):
    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))
    g = sns.lineplot(x="Date", y=y, hue=hue, data=df)
    plt.xticks(rotation=90)
    plt.title(f'{y} cases grouped by {hue}')
    if(is_log):
        ax.set(yscale="log")
    ax.grid(color='black', linestyle='dotted', linewidth=0.75)
    plt.show()  


# ## Mainland China - time evolution

# In[ ]:


plot_time_variation(data_cn, size=4, is_log=True)


# In[ ]:


plot_time_variation(data_cn, y='Recovered', size=4, is_log=True)


# ## Mainland China - overall
# 
# Let's compare overall values for Mainland China (Confirmed, Recovered, Deaths).
# 

# In[ ]:


def plot_time_variation_all(df, title='Mainland China', size=1):
    f, ax = plt.subplots(1,1, figsize=(4*size,2*size))
    g = sns.lineplot(x="Date", y='Confirmed', data=df, color='blue', label='Confirmed')
    g = sns.lineplot(x="Date", y='Recovered', data=df, color='green', label='Recovered')
    g = sns.lineplot(x="Date", y='Deaths', data=df, color = 'red', label = 'Deaths')
    plt.xlabel('Date')
    plt.ylabel(f'Total {title} cases')
    plt.xticks(rotation=90)
    plt.title(f'Total {title} cases')
    ax.grid(color='black', linestyle='dotted', linewidth=0.75)
    plt.show()  


# In[ ]:


data_cn = data_cn.loc[data_cn['Date']<'2020-03-10']
data_cn_agg = data_cn.groupby(['Date']).sum().reset_index()
plot_time_variation_all(data_cn_agg, size=3)


# ## Mainland China (except Hubei) - time evolution

# In[ ]:


filtered_data_last = filtered_data_last.reset_index()
plot_time_variation(data_cn.loc[~(data_cn['Province/State']=='Hubei')],y='Recovered', size=4, is_log=True)


# The following Folium group using **CircleMarker** is created using the inspiration from: https://www.kaggle.com/grebublin/coronavirus-propagation-visualization-forecast Kernel.

# In[ ]:


m = folium.Map(location=[30, 100], zoom_start=4)

folium.Choropleth(
    geo_data=cn_geo_data,
    name='Confirmed cases - regions',
    key_on='feature.properties.name',
    fill_color='YlGn',
    fill_opacity=0.05,
    line_opacity=0.3,
).add_to(m)

radius_min = 2
radius_max = 40
weight = 1
fill_opacity = 0.2

_color_conf = 'red'
group0 = folium.FeatureGroup(name='<span style=\\"color: #EFEFE8FF;\\">Confirmed cases</span>')
for i in range(len(filtered_data_last)):
    lat = filtered_data_last.loc[i, 'Latitude']
    lon = filtered_data_last.loc[i, 'Longitude']
    province = filtered_data_last.loc[i, 'Province/State']
    recovered = filtered_data_last.loc[i, 'Recovered']
    death = filtered_data_last.loc[i, 'Deaths']

    _radius_conf = np.sqrt(filtered_data_last.loc[i, 'Confirmed'])
    if _radius_conf < radius_min:
        _radius_conf = radius_min

    if _radius_conf > radius_max:
        _radius_conf = radius_max

    _popup_conf = str(province) + '\n(Confirmed='+str(filtered_data_last.loc[i, 'Confirmed']) + '\nDeaths=' + str(death) + '\nRecovered=' + str(recovered) + ')'
    folium.CircleMarker(location = [lat,lon], 
                        radius = _radius_conf, 
                        popup = _popup_conf, 
                        color = _color_conf, 
                        fill_opacity = fill_opacity,
                        weight = weight, 
                        fill = True, 
                        fillColor = _color_conf).add_to(group0)

group0.add_to(m)
folium.LayerControl().add_to(m)
m


# ## Mainland China - mortality
# 
# 
# Let's plot the mortality in two ways. We will calculate the mortality as percent of Deaths from the Confirmed cases and also will calculate the mortality as a percent from the Recovered cases. The first one is an underestimate, since death will happen typically at least few days after a case is confirmed. The second one is an overestimate, since death will most probably occur much faster than a recovery. But we plot both to have both estimation and most probably the real mortality is in between. 

# In[ ]:


def plot_time_variation_mortality(df, title='Mainland China', size=1):
    f, ax = plt.subplots(1,1, figsize=(4*size,2*size))
    g = sns.lineplot(x="Date", y='Mortality (D/C)', data=df, color='blue', label='Mortality (Deaths / Confirmed)')
    g = sns.lineplot(x="Date", y='Mortality (D/R)', data=df, color='green', label='Mortality (Death / Recovered)')
    plt.xlabel('Date')
    ax.set_yscale('log')
    plt.ylabel(f'Mortality {title} [%]')
    plt.xticks(rotation=90)
    plt.title(f'Mortality percent {title}\nCalculated as Deaths/Confirmed cases and as Death / Recovered cases')
    ax.grid(color='black', linestyle='dashed', linewidth=1)
    plt.show()  


# In[ ]:


data_cn_agg['Mortality (D/C)'] = data_cn_agg['Deaths'] / data_cn_agg['Confirmed'] * 100
data_cn_agg['Mortality (D/R)'] = data_cn_agg['Deaths'] / data_cn_agg['Recovered'] * 100
plot_time_variation_mortality(data_cn_agg, size = 5)


# We can observe that mortality calculated as Deaths / Confirmed cases oscilated constantly around 3% for Mainland China. The mortality calculated as Deaths / Recovered cases started high (since deaths were more prevalent that recoveries at start, especially because cases were discovered late and mostly the very severe). As the epidemics progressed and more and more measures to contains the spread were put in place, the percent of Deaths / Recovered droped. In the final, both calculations will converge to something that looks like around 3% for Mainland China.

# # All World
# 
# Let's check now the status in the whole World.

# In[ ]:


data_wd = data_df.copy() #data_df.loc[~(data_df['Country/Region'].isin(["Mainland China", "China"]))]
data_wd = pd.DataFrame(data_wd.groupby(['Country/Region', 'Date'])['Confirmed', 'Recovered', 'Deaths'].sum()).reset_index()
data_wd.columns = ['Country', 'Date', 'Confirmed', 'Recovered', 'Deaths' ]
data_wd = data_wd.sort_values(by = ['Country','Date'], ascending=False)


# In[ ]:


data_ct = data_wd.sort_values(by = ['Country','Date'], ascending=False)
filtered_data_ct_last = data_wd.drop_duplicates(subset = ['Country'], keep='first')
data_ct_agg = data_ct.groupby(['Date']).sum().reset_index()


# In[ ]:


plot_count('Country', 'Confirmed', 'Confirmed cases - all World excepting China', filtered_data_ct_last, size=4)


# In[ ]:


plot_count('Country', 'Recovered', 'Recovered - all World excepting China', filtered_data_ct_last, size=4)


# In[ ]:


plot_count('Country', 'Deaths', 'Deaths - all World excepting China', filtered_data_ct_last, size=4)


# ## All World - time variation
# 
# We show the time variation of the whole World cases (Confirmed, Recovered, Deaths).

# In[ ]:


plot_time_variation_all(data_ct_agg, 'All World', size=4)


# Let's look separatelly to the dynamic of few countries.

# In[ ]:


data_select_agg = data_ct.groupby(['Country', 'Date']).sum().reset_index()


# In[ ]:


def plot_time_variation_countries(df, countries, case_type='Confirmed', size=3, is_log=False):
    f, ax = plt.subplots(1,1, figsize=(4*size,4*size))
    for country in countries:
        df_ = df[(df['Country']==country) & (df['Date'] > '2020-02-01')] 
        g = sns.lineplot(x="Date", y=case_type, data=df_,  label=country)  
        ax.text(max(df_['Date']), max(df_[case_type]), str(country))
    plt.xlabel('Date')
    plt.ylabel(f'Total  {case_type} cases')
    plt.title(f'Total {case_type} cases')
    plt.xticks(rotation=90)
    if(is_log):
        ax.set(yscale="log")
    ax.grid(color='black', linestyle='dotted', linewidth=0.75)
    plt.show()  


# In[ ]:


countries = ['China', 'Italy', 'Iran', 'Spain', 'Germany', 'Switzerland', 'US', 'South Korea', 'United Kingdom', 'France', 'Netherlands', 'Japan']
plot_time_variation_countries(data_select_agg, countries, size=4)


# **27-03-2020** update: as of today, first three countries as number of Confirmed cases are US, China & Italy.

# In[ ]:


countries = ['China', 'Italy', 'Iran', 'Spain', 'Germany', 'Switzerland', 'US', 'South Korea', 'United Kingdom', 'France', 'Netherlands', 'Japan']
plot_time_variation_countries(data_select_agg, countries,case_type = 'Deaths', size=4)


# We can observe the very high dynamic in countries like Italy or Spain and the almost saturated curve in the case of South Korea or China.

# ## Time variation - logarithmic scale
# 
# 
# Let's represent the same time variation but on a logarithmic scale. This will enable easy comparison between infection rates of countries that are now in different stages of epidemics propagation and with different number of confirmed cases.

# In[ ]:


countries = ['China','Italy', 'Iran', 'Spain', 'Germany', 'Switzerland', 'US', 'South Korea', 'United Kingdom', 'France', 'Netherlands', 'Japan', 'Romania']
plot_time_variation_countries(data_select_agg, countries, size=4, is_log=True)


# We can observe on the logarithmic scale the very high morbidity in countries like Italy, Spain, US, Switzerland, Netherlands or France, while in countries like Japan, South Korea and even Iran we can see a smaller rate of new infections. In the logarithmic plot, the slope of the curve allows to compare infection rates between countries with very different number of total infections.  
# 
# US shows (for few last days) the sharpest log-curve slope, with the sharpest increase of number of cases per day. With this trend, US will surpass Italy as number of total confirmed infections in very short time.
# 
# I include here for demonstration the case of Romania (my country). Although the number of current infections is relatively low, by comparing the slope of the curve with Italy, Spain or UK, we can see that Romania has a comparable slope (in the logarithmic scale), therefore this should be an alarm sign for the local authorities, that were until recently slow on enforcing decisive measures for ensuring social distance.
# 
# **27-03-2020** update: US leads as number of Confirmed cases, with a very sharp rise in the last days.
# 

# In[ ]:


countries = ['China','Italy', 'Iran', 'Spain', 'Germany', 'Switzerland', 'US', 'South Korea', 'United Kingdom', 'France', 'Netherlands', 'Japan']
plot_time_variation_countries(data_select_agg, countries,case_type = 'Deaths', size=4, is_log=True)


# # Heatmap with cases in the World
# 
# 
# ## Confirmed cases in the World
# 
# 
# Let's see a heatmap with cases distribution (as of last update) in the World (including Mainland China).

# In[ ]:


data_ps = data_df.sort_values(by = ['Province/State','Date'], ascending=False)
filtered_data_ps = data_ps.drop_duplicates(subset = ['Province/State'],keep='first').reset_index()

data_cr = data_df.sort_values(by = ['Country/Region','Date'], ascending=False)
filtered_data_cr = data_cr.drop_duplicates(subset = ['Country/Region'],keep='first').reset_index()

filtered_data_cr = filtered_data_cr.loc[~filtered_data_cr.Latitude.isna()]
filtered_data_cr = filtered_data_cr.loc[~filtered_data_cr.Longitude.isna()]
filtered_data = pd.concat([filtered_data_cr, filtered_data_ps], axis=0).reset_index()


# In[ ]:


m = folium.Map(location=[0,0], zoom_start=2)
filtered_data['Cnf'] = np.sqrt(filtered_data['Confirmed'])
HeatMap(data=filtered_data[['Latitude', 'Longitude', 'Cnf']].groupby(['Latitude', 'Longitude']).sum().reset_index().values.tolist(),        radius=18, max_zoom=12).add_to(m)
m


# ## Recovered cases in the World
# 
# 
# Let's see a heatmap with Recovered in the World (including China).

# In[ ]:


m = folium.Map(location=[0,0], zoom_start=2)
filtered_data['R'] = np.sqrt(filtered_data['Recovered'])
HeatMap(data=filtered_data[['Latitude', 'Longitude', 'R']].groupby(['Latitude', 'Longitude']).sum().reset_index().values.tolist(),        radius=15, max_zoom=12).add_to(m)
m


# In[ ]:


data_all_wd = pd.DataFrame(data_df.groupby(['Country/Region', 'Date'])['Confirmed',  'Recovered', 'Deaths'].sum()).reset_index()
data_all_wd.columns = ['Country', 'Date', 'Confirmed', 'Recovered', 'Deaths' ]
data_all_wd = data_all_wd.sort_values(by = ['Country','Date'], ascending=False)
filtered_all_wd_data_last = data_all_wd.drop_duplicates(subset = ['Country'],keep='first')
filtered_all_wd_data_last.loc[filtered_all_wd_data_last['Country']=='Mainland China', 'Country'] = 'China'


# ## Al World mortality

# In[ ]:


data_ct_agg = data_all_wd.groupby(['Date']).sum().reset_index()

data_ct_agg['Mortality (D/C)'] = data_ct_agg['Deaths'] / data_ct_agg['Confirmed'] * 100
data_ct_agg['Mortality (D/R)'] = data_ct_agg['Deaths'] / data_ct_agg['Recovered'] * 100
plot_time_variation_mortality(data_ct_agg, title = ' - Rest of the World (not Mainland China)', size = 3)


# Mortality calculated as Deaths / Recovered is still increasing, as well as Deaths / Confirmed cases. This is mainly due to drastic increase in the number of Deaths in countries like Italy and Iran. Let's look separatelly to those countries.

# In[ ]:


data_all_wd = pd.DataFrame(data_df.groupby(['Country/Region', 'Date'])['Confirmed',  'Recovered', 'Deaths'].sum()).reset_index()
data_all_wd.columns = ['Country', 'Date', 'Confirmed', 'Recovered', 'Deaths' ]
data_all_wd = data_all_wd.sort_values(by = ['Country','Date'], ascending=False)
data_italy = data_all_wd[data_all_wd['Country']=='Italy']
data_it_agg = data_italy.groupby(['Date']).sum().reset_index()

data_it_agg['Mortality (D/C)'] = data_it_agg['Deaths'] / data_it_agg['Confirmed'] * 100
data_it_agg['Mortality (D/R)'] = data_it_agg['Deaths'] / data_it_agg['Recovered'] * 100

plot_time_variation_mortality(data_it_agg, title = ' - Italy', size = 3)


# The high mortality (and still raising) in Italy is most probably the result of delayed social isolation measures and also reluctance of the population to observe the rules. We see a raising pattern in both these ratios starting March 1st and it is likely for this pattern to continue since social isolation rules are slow to be enforced in this highly populated, with a large density of senior citizens, European country.

# In[ ]:


data_iran = data_all_wd[data_all_wd['Country']=='Iran']
data_ir_agg = data_iran.groupby(['Date']).sum().reset_index()

data_ir_agg['Mortality (D/C)'] = data_ir_agg['Deaths'] / data_ir_agg['Confirmed'] * 100
data_ir_agg['Mortality (D/R)'] = data_ir_agg['Deaths'] / data_ir_agg['Recovered'] * 100

plot_time_variation_mortality(data_ir_agg, title = ' - Iran', size = 3)


# We can observe a strange pattern starting from March 7th, that seems to indicate a certain correlation between Confirmed and Recovered cases. Also in this case we can observe to increasing mortality rate by both metrics.
# 
# 

# In[ ]:


data_sk = data_all_wd[data_all_wd['Country']=='South Korea']
data_sk_agg = data_sk.groupby(['Date']).sum().reset_index()

data_sk_agg['Mortality (D/C)'] = data_sk_agg['Deaths'] / data_sk_agg['Confirmed'] * 100
data_sk_agg['Mortality (D/R)'] = data_sk_agg['Deaths'] / data_sk_agg['Recovered'] * 100

plot_time_variation_mortality(data_sk_agg, title = ' - South Korea', size = 3)


# In South Korea preventive measures made possible initially to keep the ratio of Deaths to new Confirmed cases low (below 1%), while ratio of Deaths to Recovered cases is approaching now 10%. In recent days the number of death ratio to confirmed cases started to raise, approaching 1%.

# In[ ]:


data_sp = data_all_wd[data_all_wd['Country']=='Spain']
data_sp_agg = data_sp.groupby(['Date']).sum().reset_index()

data_sp_agg['Mortality (D/C)'] = data_sp_agg['Deaths'] / data_sp_agg['Confirmed'] * 100
data_sp_agg['Mortality (D/R)'] = data_sp_agg['Deaths'] / data_sp_agg['Recovered'] * 100

plot_time_variation_mortality(data_sp_agg, title = ' - Spain', size = 3)


# In Spain death ratio (calculated as number of deaths / confirmed cases) is above 2% is raising. Spain might be more than a week behind Italy in terms of evolution of epidemic.

# In[ ]:


data_de = data_all_wd[data_all_wd['Country']=='Germany']
data_de_agg = data_de.groupby(['Date']).sum().reset_index()

data_de_agg['Mortality (D/C)'] = data_de_agg['Deaths'] / data_de_agg['Confirmed'] * 100
data_de_agg['Mortality (D/R)'] = data_de_agg['Deaths'] / data_de_agg['Recovered'] * 100

plot_time_variation_mortality(data_ir_agg, title = ' - Germany', size = 3)


# We can observe that in Germany the mortality calculated as number of Deaths over number of confirmed is rather small (less than 0.6%) while the percent of Deaths / Recovered is around 3%. This is due to the large number of testes performed as well as due to medical system preparadness and capacity thus reflecting better the real morbidity of the virus. 

# In[ ]:


data_us = data_all_wd[data_all_wd['Country']=='US']
data_us_agg = data_us.groupby(['Date']).sum().reset_index()

data_us_agg['Mortality (D/C)'] = data_us_agg['Deaths'] / data_us_agg['Confirmed'] * 100
data_us_agg['Mortality (D/R)'] = data_us_agg['Deaths'] / data_us_agg['Recovered'] * 100

plot_time_variation_mortality(data_us_agg, title = ' - US', size = 3)


# 
# 
