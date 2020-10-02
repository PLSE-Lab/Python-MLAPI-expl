#!/usr/bin/env python
# coding: utf-8

# # Covid-19 Analysis in Czech Republic

# Today I am going to take a look on Covid-19 pandemic evolution in Czech Republic. Data we are using are provided by Ministry of Czech Republic and are composed of multiple files and merged all together, so we need to load just 1 file at all. Data are from end of January 2020 up to end of June 2020.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# # Load dataset
# Keep in mind dataset contains data on 3 different levels of granularity and must be separated into 3 dataset

# In[ ]:


# convert date to python date format right when loading data
df = pd.read_csv('/kaggle/input/covid19-czech-republic/covid19-czech.csv', parse_dates=['date'])


# Quick look on first few records in dataset. Dataset contains almost 18k rows and 19 columns.

# In[ ]:


print('Size of dataset: {}'.format(df.shape))
df.head()


# Just quickly fill missing values with empty string, as data are on different granularity level, there are going to be few missing values that we can simply impute.

# In[ ]:


df = df.fillna('')


# # Split data
# Now we create 3 datasets, each will be used in different analyze

# In[ ]:


# daily/date data level
df_daily = df[df['date'] < '2020-07-02'].groupby(['date']).max()[['daily_tested','daily_infected','daily_cured','daily_deaths','daily_cum_tested','daily_cum_infected','daily_cum_cured','daily_cum_deaths']].reset_index()

# regional data level
df_region = df[df['region'] != ''].groupby(['region']).agg(
    region_accessories_qty=pd.NamedAgg(column='region_accessories_qty', aggfunc='max'), 
    infected=pd.NamedAgg(column='infected', aggfunc='sum'),
    cured=pd.NamedAgg(column='cured', aggfunc='sum'),
    death=pd.NamedAgg(column='death', aggfunc='sum')
).reset_index()

# detail data level
df_detail = df[['date','region','sub_region','age','sex','infected','cured','death','infected_abroad','infected_in_country']].reset_index(drop=True)


# # Feature engineering
# We will create new calculation **daily_active** and **daily_cum_active** that calculates what are active cases. It's pretty good to see how many population is infected at the time and consider if health care can handle it.

# In[ ]:


df_daily['daily_active'] = df_daily['daily_infected'] - df_daily['daily_cured'] - df_daily['daily_deaths']
df_daily['daily_cum_active'] = df_daily['daily_cum_infected'] - df_daily['daily_cum_cured'] - df_daily['daily_cum_deaths']
#df_daily = df_daily.groupby(df_daily['date'].dt.to_period('W').dt.start_time).sum().reset_index()


# # Exploratory Data Analysis
# Now let's draw several charts to see details about pandemic in Czech Republic. I know there could be much much more, but there are different kernels with tens of different visualizations :)

# ## Daily test performed
# It seems maximum number of tests has been done in May and was decreasing from that time, it has stabilized at start of June.

# In[ ]:


plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
plt.fill_between(df_daily.date, df_daily.daily_tested, color="purple", label='active',alpha=0.5)
plt.xticks(rotation=45)
plt.title('Daily tests performed')

plt.subplot(1,2,2)
plt.fill_between(df_daily.date, df_daily.daily_cum_tested, color="purple", label='active',alpha=0.5)
plt.xticks(rotation=45)
plt.title('Daily tests performed- cumulative sum')

plt.show()


# ## Daily cured / death / infected / active
# It may be interesting to compare all these statistics

# In[ ]:


plt.figure(figsize=(20,8))
sns.lineplot(df_daily.date, df_daily.daily_cured, color="green", label='cured')
sns.lineplot(df_daily.date, df_daily.daily_deaths, color="red", label='death')
sns.lineplot(df_daily.date, df_daily.daily_infected, color="blue", label='infected')
sns.lineplot(df_daily.date, df_daily.daily_active, color="orange", label='active')
plt.xticks(rotation=45)
plt.title('Daily cured / death / infected / active')
plt.show()


# Right it's not as easy to read it, especially to compare deaths as there is much less of them. I am going to separate charts now. Can you see **increase of daily active** cases from April (when hit bottom) up to end of June?

# In[ ]:


plt.figure(figsize=(20,12))
plt.subplot(4,1,1)
sns.lineplot(df_daily.date, df_daily.daily_cured, color="green", label='Daily new cured cases')
plt.xlabel('')

plt.subplot(4,1,2)
sns.lineplot(df_daily.date, df_daily.daily_deaths, color="red", label='Daily new death cases')
plt.xlabel('')

plt.subplot(4,1,3)
sns.lineplot(df_daily.date, df_daily.daily_infected, color="blue", label='Daily new infected cases')
plt.xlabel('')

plt.subplot(4,1,4)
sns.lineplot(df_daily.date, df_daily.daily_active, color="orange", label='Daily new active cases')
plt.xlabel('')

plt.show()


# Another chart to compare all 4 stats

# In[ ]:


plt.figure(figsize=(20,8))

plt.subplot(2,2,1)
plt.fill_between(df_daily.date, df_daily.daily_cured, color="green", label='Daily new cured cases', alpha=0.5)
plt.legend(loc='best')
plt.xlabel('')

plt.subplot(2,2,2)
plt.fill_between(df_daily.date, df_daily.daily_deaths, color="red", label='Daily new death cases',  alpha=0.5)
plt.legend(loc='best')
plt.xlabel('')

plt.subplot(2,2,3)
plt.fill_between(df_daily.date, df_daily.daily_infected, color="blue", label='Daily new infected cases',  alpha=0.5)
plt.legend(loc='best')
plt.xlabel('')

plt.subplot(2,2,4)
plt.fill_between(df_daily.date, df_daily.daily_active, color="orange", label='Daily new active cases',  alpha=0.5)
plt.legend(loc='best')
plt.xlabel('')

plt.show()


# ## Daily cumulative sum of cured / death / infected / active
# Cumulative sums are also nice to see. Do you think 2nd wave of Covid-19 has started in Czech Republic?

# In[ ]:


plt.figure(figsize=(20,5))

plt.subplot(1,2,1)
sns.lineplot(df_daily.date, df_daily.daily_cum_cured, color="green", label='cured')
sns.lineplot(df_daily.date, df_daily.daily_cum_deaths, color="red", label='death')
sns.lineplot(df_daily.date, df_daily.daily_cum_infected, color="blue", label='infected')
sns.lineplot(df_daily.date, df_daily.daily_cum_active, color="orange", label='infected')
plt.legend(loc='best')

plt.subplot(1,2,2)
plt.fill_between(df_daily.date, df_daily.daily_cum_cured, color="green", label='cured', alpha=0.5)
plt.fill_between(df_daily.date, df_daily.daily_cum_deaths, color="red", label='death',  alpha=0.5)
plt.fill_between(df_daily.date, df_daily.daily_cum_infected, color="blue", label='infected',  alpha=0.5)
plt.fill_between(df_daily.date, df_daily.daily_cum_active, color="orange", label='active',  alpha=0.5)
plt.legend(loc='best')

plt.show()


# ## Regional statistics
# What are differences between regions? Are some doing better than the other one? I will also compare accessories delivered to region with other metrics if health care accessories can have impact on evolution of pandemic (it should, or not?)

# What I could find is that **Moravskoslezsky kraj** had a lot of infected people (more than capital Prague), but got very poor health care accessories delivery. On the other hand, Prague got much more accessories, had less infected, but still more deaths... maybe how accessories sums were calculated was not correct.

# In[ ]:


plt.figure(figsize=(20, 15))
plt.subplot(2,2,1)
sns.barplot(x=df_region.infected, y=df_region.region)
plt.title('Number of infected')
plt.subplot(2,2,2)
sns.barplot(x=df_region.cured, y=df_region.region)
plt.title('Number of cured')
plt.subplot(2,2,3)
sns.barplot(x=df_region.death, y=df_region.region)
plt.title('Number of death')
plt.subplot(2,2,4)
sns.barplot(x=df_region.region_accessories_qty, y=df_region.region)
plt.title('Health care accessories delivered')

plt.show()


# In[ ]:


# quick fix for sex & age for further analysis
df_detail['sex'] = df_detail['sex'].replace('', np.NaN)
df_detail = df_detail.dropna()
df_detail['age'] = df_detail['age'].astype('int64')


# ## Sex comparison
# Male or female has more deaths? What you think, is it better to be male or female at these days? :)

# It seems that male & female has pretty much same chance to be infected, but male dies much more likely than female... if I could switch now!

# In[ ]:


plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
sns.barplot(x=df_detail.sex, y=df_detail.infected, estimator=np.sum)
plt.title('Infected by Sex')

plt.subplot(1,3,2)
sns.barplot(x=df_detail.sex, y=df_detail.cured, estimator=np.sum)
plt.title('Cured by Sex')

plt.subplot(1,3,3)
sns.barplot(x=df_detail.sex, y=df_detail.death, estimator=np.sum)
plt.title('Death by Sex')

plt.show()


# ## Age comparison
# Experts are talking that old people are risky group of people in pandemic... can we prove it?

# Right, you will find out that mortality on Covid-19 is starting from 40 years and dramatically increase with age!

# In[ ]:


plt.figure(figsize=(20,15))
plt.subplot(3,1,1)
sns.barplot(x=df_detail.age, y=df_detail.infected, estimator=np.sum)
plt.title('Infected by age')
plt.xticks(rotation=90)

plt.subplot(3,1,2)
sns.barplot(x=df_detail.age, y=df_detail.cured, estimator=np.sum)
plt.title('Cured by age')
plt.xticks(rotation=90)

plt.subplot(3,1,3)
sns.barplot(x=df_detail.age, y=df_detail.death, estimator=np.sum)
plt.title('Death by age')
plt.xticks(rotation=90)

plt.show()


# ## Sub-region statistics
# Is there anything interesting when looking on data from sub-region point of view?

# I could see that there is unusually high number of infected cases in Karvina, unusually high number of deaths in Chleb and Ostrava.

# In[ ]:


plt.figure(figsize=(20, 20))
plt.subplot(1,3,1)
sns.barplot(x=df_detail.infected, y=df_detail.sub_region, estimator=np.sum)
plt.title('Number of infected')
plt.subplot(1,3,2)
sns.barplot(x=df_detail.cured, y=df_detail.sub_region, estimator=np.sum)
plt.title('Number of cured')
plt.subplot(1,3,3)
sns.barplot(x=df_detail.death, y=df_detail.sub_region, estimator=np.sum)
plt.title('Number of death')

plt.show()


# ## Imported cases from abroad
# It is known that a lot of Covid-19 cases are imported from abroad. People are coming back to Czech Republic and bringing also unwanted guest with them.

# Most of imports seems to be from Italy (where pandemic has started in Europe), Austria and Ukraine. We know just about 1 person that has died (when Covid-19 was imported) and this one was from Ukraine.

# In[ ]:


df_import = df_detail[df_detail['infected_abroad'] == 'Y'].groupby(['date', 'infected_abroad','infected_in_country']).sum()[['infected', 'cured', 'death']].reset_index()

plt.figure(figsize=(20,15))
plt.subplot(3,1,1)
sns.barplot(x=df_import['infected_in_country'], y=df_import['infected'], estimator=np.sum)
plt.title('Infected by country of import')

plt.subplot(3,1,2)
sns.barplot(x=df_import['infected_in_country'], y=df_import['cured'], estimator=np.sum)
plt.title('Cured by country of import')

plt.subplot(3,1,3)
sns.barplot(x=df_import['infected_in_country'], y=df_import['death'], estimator=np.sum)
plt.title('Died by country of import')

plt.show()


# That's all from my analysis, hope you liked it. Feel free to use dataset and create your own visualizations, let me know then ;)

# ### Thanks for checking my notebook, if you liked it, make sure to vote for for this notebook!
