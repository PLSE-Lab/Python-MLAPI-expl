#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns; sns.set;
from datetime import timedelta
#pd.options.display.float_format = '{:, .0f}'.format
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
plt.style.use('fivethirtyeight') 


# In[ ]:


train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv').rename(columns = {"Province/State": "State", "Country/Region":"country",
                                                                                                   "ConfirmedCases":"cases", "Fatalities":"deaths"})
train['Date']=pd.to_datetime(train.Date)
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')
sub = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv')


# In[ ]:


countries = pd.read_csv('/kaggle/input/undata-country-profiles/country_profile_variables.csv')
countries['country'] = countries.country.str.strip()
#rename columns for easier coding
col_dict = {"Region":"Region",'Surface area (km2)':"surface_area", 'Population in thousands (2017)':'pop_in_thou', 
            'Population density (per km2, 2017)':'pop_density','Sex ratio (m per 100 f, 2017)':"sex_ratio",
            'GDP: Gross domestic product (million current US$)':'gdp','GDP growth rate (annual %, const. 2005 prices)':"gdp_growth", 
            'Urban population (% of total population)':'urban_pop','Urban population growth rate (average annual %)':'urban_growth_rate',
            'Fertility rate, total (live births per woman)':'fertility_rate','Life expectancy at birth (females/males, years)':'life_expectancy',
            'Population age distribution (0-14 / 60+ years, %)':'pop_age_distribution', 'International migrant stock (000/% of total pop.)':'imigrant_pop',
            'Infant mortality rate (per 1000 live births':'infant_mortality', 'Health: Total expenditure (% of GDP)':'healthcare_expenditure',
            'Health: Physicians (per 1000 pop.)':'physicians_per_1000'}
countries = countries.rename(columns = col_dict)
countries = countries[['country', 'surface_area', 'pop_in_thou','pop_density','sex_ratio','gdp','gdp_growth','urban_pop','urban_growth_rate', 'imigrant_pop',
           'fertility_rate','life_expectancy','pop_age_distribution','infant_mortality','healthcare_expenditure','physicians_per_1000']]
#Get the country names from the country data in-line with country nemes in COVID-19 Dataset
countries = countries.replace({'Bolivia (Plurinational State of)':'Bolivia', 'Congo':'Congo (Kinshasa)', 
                               'United States of America':'US', 'Iran (Islamic Republic of)':'Iran', 'Bosnia & Herzegovina': 'Bosnia and Herzegovina', 
                               'Venezuela (Bolivarian Republic of)':'Venezuela', "Viet Nam": "Vietnam", 'Trinidad & Tobago':'Trinidad and Tobago', 
                               'Republic of Korea':'Korea, South', 'Russian Federation':"Russia",'Brunei Darussalam':'Brunei',
                               'United Republic of Tanzania': 'Tanzania', 'Bahamas': 'The Bahamas', 'Gambia': 'Gambia, The',
                               'The former Yugoslav Republic of Macedonia': 'North Macedonia','Democratic Republic of the Congo': 'Republic of the Congo','Republic of Moldova':"Moldova",
                               'Taiwan': 'Taiwan*', 'Central African Rep.': 'Central African Republic', 'Czech Republic': 'Czechia'})
age_dist = countries.pop_age_distribution.str.split("/", expand = True)
countries['age_dist_0-14'] = age_dist[0].replace({"-99":np.nan, "...":np.nan}).astype(float)
countries['age_dist_60p'] = age_dist[1].replace({"-99":np.nan, "...":np.nan}).astype(float)
countries['age_dist_14-60'] = 100 - countries['age_dist_0-14']-countries['age_dist_60p']
countries = countries.drop(["pop_age_distribution", "surface_area", 'gdp_growth', 'urban_growth_rate'], axis = 1)
countries['gdp_per_cap']=(countries.gdp*1000) / (countries.pop_in_thou)
countries[countries.gdp_per_cap<0] = np.nan
life_ex = countries.life_expectancy.str.split("/", expand = True)
life_ex[0] = life_ex[0].replace({"-99":np.nan, "...":np.nan}).astype(float)
life_ex[1] = life_ex[1].replace({"-99":np.nan, "...":np.nan}).astype(float)
countries['life_expectancy'] = (life_ex[0]+life_ex[1])/2
countries['immigrand_pct'] = countries.imigrant_pop.str.split("/", expand = True, n=1)[1]
countries = countries.drop(['imigrant_pop', 'life_expectancy'], axis = 1)
countries = countries[~pd.isna(countries.country)]
countries['population'] = countries.pop_in_thou * 1000
#countries = countries.merge(train[['country', 'Lat', "Long"]].drop_duplicates(), how = 'left', on = 'country')


# In[ ]:


#There is a lot of 'state' data by aggregating it you reduce the number of observations from 17k to 6.5k (on 3/23)
data = train.drop(["Lat","Long", "Id"], axis =1).groupby(["country", "Date"]).sum().reset_index()
date_past_100 = data[data.cases > 100].groupby('country')[['Date']].min().reset_index()[['country', 'Date']].rename(columns = {"Date":"date_past_100"})
date_past_100
data = data.merge(date_past_100, how = 'left', on = 'country')
data['date_past_100'] = (data.date_past_100.fillna(pd.to_datetime(-999999)))
data['days_since_100'] = ((data.Date - data.date_past_100)/(8.64e+13)).astype(int)
data.loc[(data['days_since_100'] < 0) | (data['days_since_100'] > 10000), 'days_since_100'] = 0
data = data.drop('date_past_100', axis = 1)


# In[ ]:


data_c = data.merge(countries[['country', 'population']], on = 'country', how = 'left')
data_c['cases_p_1000'] = data_c.cases / (data_c.population / 1000)
data_c['deaths_p_1000'] = data_c.deaths / (data_c.population / 1000)


# In[ ]:


max_cases = data_c.groupby('country')[['cases_p_1000', 'deaths_p_1000', 'cases', 'deaths']].max().sort_values('cases_p_1000')
top_10 = max_cases[~pd.isna(max_cases.cases_p_1000)].sort_values('cases', ascending = False).head(10).reset_index()
top_10_countries = top_10.country.tolist()


# In[ ]:


train_clean= pd.DataFrame([])
for country in data_c.country.unique():
    cou_dat = data_c[data_c.country == country]
    cou_dat['pct_chg_cases']=(cou_dat.cases.pct_change()).fillna(0).tolist()
    cou_dat['days_to_double']=(1/cou_dat.cases.pct_change()).fillna(0).tolist()
    cou_dat['DtD_sma5'] = cou_dat.iloc[:,9].rolling(window=5).mean()
    cou_dat['pct_c_cases'] = cou_dat.iloc[:,8].rolling(window=5).mean()
    cou_dat['cases_10_days_prior'] = cou_dat[['cases']].shift(periods = 5).fillna(0)
    cou_dat['deaths_15_days_prior'] = cou_dat[['deaths']].shift(periods = 15).fillna(0)
    cou_dat['new_cases'] = cou_dat['cases'].diff()
    train_clean = train_clean.append(cou_dat)
train_clean = train_clean.replace([np.inf, -np.inf], 0)


# In[ ]:


plot_dat = train_clean[train_clean.country.isin(top_10_countries)]
last_obs=plot_dat.groupby('country', as_index = False).max()


# In[ ]:


plot_dat = train_clean[train_clean.country.isin(top_10_countries)]
#ax[0].set(yscale = 'log')
ax[0].set_ylim([100,20000])
#ax.set_xlim([0,30])
sns.lineplot(x="days_since_100", y="new_cases", hue = "country", data = plot_dat, markers = True, ax=ax[0])
ax[0].set(xlabel='Days Since 100th Case', ylabel = "New Cases", title = "New Daily Corona Cases")
#plt.xticks(rotation=30)

ax[1].set(yscale = 'log'); ax[1].set_ylim([100,100000])
ax[1].set_xlim([0,30])
sns.lineplot(x="days_since_100", y="cases", hue = "country", data = plot_dat, markers = True, ax=ax[1])
ax[1].set(xlabel='Days Since 100th Case', ylabel = "Cases (log-scale)", title = "Total Cases")
#plt.xticks(rotation=30)


# In[ ]:





# In[ ]:


f,ax = plt.subplots(figsize=(15,7))
#ax.set(yscale = 'log')
ax.set_ylim([0,20])
ax.set_xlim([0,30])
p1=sns.lineplot(x="days_since_100", y="DtD_sma5", hue = "country", data = plot_dat, markers = True)
ax.set(xlabel='Days Since 100th Case', ylabel = "Days to Double", title = "Days to Double Cases")
#plt.xticks(rotation=30)
for item, color in zip(plot_dat.groupby(country).max().Date)


# In[ ]:





# In[ ]:


data = train.drop(["Lat","Long", "Id"], axis =1).groupby(["country", "Date"]).sum().reset_index()
data= data.merge(countries[['country','population', 'urban_pop']], on = 'country', how = 'left')
data['cases_p_pop'] = data.cases / data.population 
data['deaths_p_pop'] = data.deaths / data.population


# In[ ]:


data


# In[ ]:


data.groupby('country', 'Date').


# In[ ]:


data.head()


# Id - observation ID
# 
# Country - Country name (does this correlate to UN data?)
# 
# other columns are self explanitory

# In[ ]:


countries.country.unique()


# In[ ]:


train.country.unique()


# In[ ]:


mer= train.merge(countries[['country','pop_in_thou']], on = 'country', how = 'left')
mer[pd.isna(mer.pop_in_thou)].country.unique()


# In[ ]:


data_country_totals = train.groupby(['country', 'Date'])[['cases', 'deaths']].sum().reset_index()
data_country_totals = data_country_totals.merge(countries[['country','Population', 'pop_density']], on = 'country', how = 'left')
pd.options.display.float_format = '{:,.0f}'.format
ag_f = {'cases':'sum', 'deaths': 'sum', 'Population':'mean', 'pop_density':'mean'}
data_country_totals.groupby("country")[['cases','deaths']].agg(ag_f)


# In[ ]:





# In[ ]:


#train_pop[pd.isna(train_pop.Population)]


# In[ ]:


data_country_totals

