#!/usr/bin/env python
# coding: utf-8

# # Historical Analysis of Cholera (1949-2016)
# 
# ![](https://www.who.int/images/default-source/imported/cholera.tmb-1200v.jpg?Culture=en&sfvrsn=76f17511_32)
# *Picture from the World Health Organization website about [cholera](https://www.who.int/news-room/fact-sheets/detail/cholera).*
# 
# Cholera is an endemic disease caused by the bacteria *Vibrio cholerae* that infects 2.86 million people every year, according to a [research](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4455997/) conducted in 2015. In 2017, an outbreak in the Democratic Republic of the Congo, Yemen and Somalia caused 1.227.391 cases, which is 929% higher than 2016 and doubled the previous record of cases. This shows that cholera is still a dangerous disease to a large part of the population who doesn't have access to water sanitation and clean food. **In this kernel I will make an historical analysis of the disease progression and inspect which development and public health indexes are more correlated with the cholera incidence. My goal is to shed a light on this thorny issue. If you like it, please consider giving it an upvote.** 
# 
# In order to do so, I utilized data from the World Health Organization, which anually collect reports from its members and compiles them in their website. About three milion cases occur every year, but, as we will see, the number of reported ones is way lower than that. This happens because some of the most affected countries doesn't have enough money and capacity to make a good cholera surveillance. **Therefore, it is necessary to make a disclaimer: this analysis is limited and we are only seeing part of the problem.**
# 
# Before we start, let's understand what Cholera is. It can affects both children and adults and for most of them it won't cause any symptom. However, the bacteria will be present in their faeces for up to 10 days and will possibly infect other people if sanitation is not present. A minority of people will develop acute watery diarrhoea that, if not treated, can kill within hours. Besides that, most cases of cholera can be easily treated with oral rehydration solution. According to WHO, "a multifaceted approach is key to control cholera, and to reduce deaths. A combination of surveillance, water, sanitation and hygiene, social mobilisation, treatment, and oral cholera vaccines are used". Since 1990, three oral vaccines that cost at most $4 have been developed.
# 
# ## Table of contents
# 
# - [Importing data and libs](#importing-data)
# - [Data preparation](#data-prep)
#     * [First look at the data](#prep-first)
#     * [Inspecting supposedly numeric columns](#prep-insp)
#     * [Missing values](#prep-miss)
# - [Exploratory analysis](#eda)
#     * [How is the disease spreading globally over time?](#eda-global-time)
#     * [What are the regions most affected by Cholera?](#eda-region-time)
#     * [Visualizing the disease progression with an animated world map](#eda-country-time)
#     * [How many countries report Cholera cases every year?](#eda-report-time)
#     * [Which country is more affected?](#eda-bar-country)
#     * [Where is the death rate high?](#eda-bar-death-rate)
#     * [Is chorela incidence correlated with human development indexes?](#eda-corr)
#     * [What is the impact of water treatment in the spread of cholera?](#eda-water)
# 
# ## Importing data and libs <a id ='importing-data'></a>

# In[ ]:


# Standard packages
import json

# Libs to deal with tabular data
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

# Plotting packages
import seaborn as sns
sns.axes_style("darkgrid")
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import plotly.express as px

# To display stuff in notebook
from IPython.display import display, Markdown


# In[ ]:


df = pd.read_csv('../input/cholera-dataset/data.csv')


# ## Data preparation <a id ='data-prep'></a>
# 
# ### First look at the data <a id ='prep-first'></a>

# In[ ]:


df.head()


# In[ ]:


df.dtypes


# In[ ]:


df = df.rename(columns={
    'Number of reported cases of cholera':'Cases',
    'Number of reported deaths from cholera':'Deaths',
    'Cholera case fatality rate':'Death rate',
    'WHO Region':'Region'
})


# ### Inspecting supposedly numeric columns <a id ='prep-insp'></a>
# 
# Since the columns cases, deaths and death rate should be numeric, I tried to look for non-numeric characters to figure out problems.

# In[ ]:


df[df['Cases'].str.contains('[^0-9\.]').astype(bool) & df['Cases'].notnull()]


# In[ ]:


df[df['Deaths'].str.contains('[^0-9\.]').astype(bool) & df['Deaths'].notnull()]


# In[ ]:


df[df['Death rate'].str.contains('[^0-9\.]').astype(bool) & df['Death rate'].notnull()]


# In the next cells I will fix the dataset in order to keep only numeric data in the cases, deaths and death rate columns. Notice that Iran has duplicated values and in fact, according to the WHO website, two values were reported in that year. I decided to keep the higher number.

# In[ ]:


df.loc[1059, 'Cases'] = '5'
df.loc[1059, 'Deaths'] = '0'
df.loc[1059, 'Death rate'] = '0.0'


# In[ ]:


for column in ['Cases', 'Deaths', 'Death rate']:
    df[column] = df[column].replace('Unknown', np.nan).str.replace(' ', '')
    df[column] = pd.to_numeric(df[column])


# ### Missing values <a id ='prep-miss'></a>

# In[ ]:


df.isnull().sum()


# In[ ]:


df[df.isnull().any(1)]


# ## Exploratory Analysis <a id ='eda'></a>
# 
# Now I will try to answer some questions that I find interesting. Before we get started, it is necessary to show the countries of each region so that we can have a more accurate analysis.
# 
# ![](http://origin.who.int/about/regions/en/WHO_Regions.gif)
# *World map divided in 6 regions according to the World Health Organization. Picture found [here](http://origin.who.int/about/regions/en/).* 
# 
# ### How has the disease been spreading globally over time? <a id ='eda-global-time'></a>

# In[ ]:


global_year = df.groupby('Year').sum().loc[:, ['Cases', 'Deaths']]

ax = sns.lineplot(data=global_year)
plt.xlabel('Year', fontsize=15)
plt.title('Cholera evolution in the last 70 years', fontsize=16, fontweight='bold')
plt.show()


# With this graph, we can say that:
# 
# - There are 4 crisis (peaks) in the last 70 years, in the 50s, 70s, 90s, 10s. Interesting enough, these peaks have a difference of 20 years between each other, so we might expect a new cholera outbreak around 2030, kidding. 
# - The last two peaks had about 400.000 more cases than the first two.
# - Since 1990, we observe that a new plateau has been defined, with every year having at least 100.000 cases.
# - Until 1960, the number of deaths followed closely the number of cases, that is, the death rate at that time was very high when compared to nowadays. With the ease of getting a cholera vaccine and the economic improvement in several countries, the number of deaths has been decreasing since then.
# 
# ### What are the regions most affected by Cholera? <a id ='eda-region-time'></a>

# In[ ]:


region_year = df.groupby(['Year', 'Region']).sum()['Cases'].reset_index()
region_year = region_year.pivot(index = 'Year', columns = 'Region', values = 'Cases').fillna(0.0)

region_year.plot.area()
plt.title('Number of reported cases by region', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=15)
plt.show()


# - Notice that until mid 70s, south-east Asia was the region that had the highest number of cases. According to the data we have, the mid 50s crisis, for instance, had a strong participation of the countries from south-east Asia and western Pacific.
# - South-east Asian countries, together with African countries, had a big impact on the 70s peak.
# - In the last two outbreaks, American countries were responsible for the huge increase in the number of cases.
# - African countries has been responsible for new plateau in the number of cases since 90s.
# - Since 2010, eastern mediterranean countries have been demonstrating an increase in cases.
# 
# ### Visualizing the disease progression with an animated world map <a id ='eda-country-time'></a>

# In[ ]:


codes = pd.read_csv('../input/alpha3-country-codes/alpha3.csv', names=['Code', 'Country']).set_index('Country')

country_year = df.groupby(['Year', 'Country']).sum()['Cases'].reset_index()
country_year = country_year.join(codes, how='left', on='Country')


# In[ ]:


fig = px.choropleth(
    country_year, 
    locations = "Code",
    color = "Cases",
    hover_name = "Country",
    color_continuous_scale = px.colors.sequential.Plasma,
    animation_frame = 'Year',
    animation_group = 'Country',
    range_color = [0, 100000]
)
fig.show()


# This animation is very nice and if we pause it in the years corresponding to the peak of cases, we can see the countries where the disease was more present.
# 
# - Until 1964, India was the country that had the highest number of reported cases. After that, Philipines and Indonesia also had a considerable number of cases.
# - In 1970, India, Philipines, Nigeria and Niger had a large participation in the peak.
# - In 1991, the first reported American outbreak had started in Peru and spread to the rest of the continent in the next few years. Brazil, Bolivia and Equador also had a lot of cases.
# - Since 1990, we can see that at least one African country per year had a considerable number of cases. That explains again the plateau and confirms that cholera seems to be an endemic disease in the African continent. Global health authorities should look at this issue in a more careful way because there is a vaccine being produced since the beggining of 19th centure and it is cheap.
# - In 2010, Haiti and Dominican Republic had a huge number of cases. It probably happened due to 2010 Haiti earthquake, which unfortunately left a large part of the population in poor conditions and without acess to clean water.
# 
# ### How many countries report Cholera cases every year? <a id ='eda-report-time'></a>

# In[ ]:


n_countries_year = df.groupby(['Year', 'Region']).count()['Country'].rename('Number of countries').reset_index()
n_countries_year = n_countries_year.pivot(index = 'Year', columns = 'Region', values = 'Number of countries')
n_countries_year = n_countries_year.fillna(0.0)

n_countries_year.plot.area()
plt.title('Number of countries reporting data by region', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=15)
plt.show()


# Regarding the number of countries reporting cholera data:
# 
# - After 1970 the number of countries reporting cases to WHO increased from about 20 to 40. 
# - In the 90s, American and European countries report more causes causing a peak of over 80 countries probably in 1994.
# - As we have seen in the world map animation, cholera is consistently being reported in a number of African countries.
# 
# ### Which country is more affected? <a id ='eda-bar-country'></a>

# In[ ]:


country_agg = df.groupby('Country').sum().loc[:, ['Cases', 'Deaths']]
country_agg['Death rate'] = country_agg['Deaths'] * 100 / country_agg['Cases']
country_agg = country_agg.sort_values('Cases', ascending=False).head(10)

country_agg.loc[:, ['Deaths', 'Cases']].iloc[::-1].plot(kind='barh', figsize=(8,6), rot=0, colormap='coolwarm_r')
plt.title('Contries most affected by cholera', fontsize=16, fontweight='bold')
plt.ylabel('Country', fontsize=15)
plt.show()


# Overall, the countries with high number of cases are those that had already faced an outbreak: India in 50s, Haiti in 10s, Peru in 90s and so on.
# 
# ### Where is the death rate high? <a id ='eda-bar-death-rate'></a>
# 
# Before showing the plot, it is valid to remember that until 1970 the death rate was considerable higher, so countries that faced outbreaks before that year are more likely to appear in the top.

# In[ ]:


country_agg = df.groupby('Country').sum().loc[:, ['Cases', 'Deaths']]
country_agg['Death rate'] = country_agg['Deaths'] * 100 / country_agg['Cases']
country_agg = country_agg.sort_values('Death rate', ascending=False).head(10)

sns.barplot(country_agg['Death rate'].values, country_agg.index, palette='Reds_r')
plt.title('Highest death rates overall', fontsize=16, fontweight='bold')
plt.xlabel('Percentage (%)', fontsize=15)
plt.ylabel('Country', fontsize=15)
plt.show()


# According to the graph, Bangladesh and Oman have death rates above 40%, followed closely by India with around 38%.
# 
# Now, let's examine which countries have the highest rates in the last decade.

# In[ ]:


last_decade = df.loc[df['Year'] >= 2007, :]
country_agg = last_decade.groupby('Country').sum().loc[:, ['Cases', 'Deaths']]
country_agg['Death rate'] = country_agg['Deaths'] * 100 / country_agg['Cases']
country_agg = country_agg.sort_values('Death rate', ascending=False).head(10)

sns.barplot(country_agg['Death rate'].values, country_agg.index, palette='Reds_r')
plt.title('Highest death rates since 2007', fontsize=16, fontweight='bold')
plt.xlabel('Percentage (%)', fontsize=15)
plt.ylabel('Country', fontsize=15)
plt.show()


# Notice that the situation changed, with African and Middle-east countries being in the top.
# 
# ### Is chorela incidence correlated with human development indexes? <a id ='eda-corr'></a>
# 
# For this task I'm going to use the human development metrics from 2015 reported by the United Nations Development Program. I'm also going to only use cholera data from 2015.

# In[ ]:


hdi = pd.read_csv('../input/human-development/human_development.csv')
hdi['Country'] = hdi['Country'].replace({
    'Tanzania (United Republic of)':'United Republic of Tanzania',
    'United Kingdom':'United Kingdom of Great Britain and Northern Ireland',
    'Congo (Democratic Republic of the)':'Democratic Republic of the Congo',
    'United States':'United States of America'
})
hdi['Gross National Income (GNI) per Capita'] = pd.to_numeric(hdi['Gross National Income (GNI) per Capita'].str.replace(',', ''))
hdi = hdi.set_index('Country')
hdi = hdi.iloc[:, 1:6]

data_2015 = df.loc[df['Year'] == 2015, ['Country', 'Cases', 'Deaths', 'Death rate']].set_index('Country')
data_2015 = data_2015.join(hdi, how='left')

print('In 2015, {} countries reported cholera data'.format(data_2015.shape[0]))

correlations = data_2015.corr(method='pearson')
correlations = correlations.iloc[:3, 3:]

sns.heatmap(data=correlations.T, annot=True, color=sns.color_palette("coolwarm", 7))
plt.title('Correlation between development indexes and cholera metrics', fontsize=16, fontweight='bold')
plt.xticks(rotation=0) 
plt.yticks(rotation=0) 
plt.show()


# We can see low correlations between the metrics. The lowest correlations is between death rate and expected years of education. In other words, it means that countries with higher death rates have lower expected years of education.
# 
# Now, let's analyze the scatterplots of these variables. 

# In[ ]:


fig, ax = plt.subplots(5, 3, figsize=(12,16))

for i, col_i in enumerate([
    'Human Development Index (HDI)', 'Life Expectancy at Birth',
    'Expected Years of Education', 'Mean Years of Education',
    'Gross National Income (GNI) per Capita']):
    for j, col_j in enumerate(['Cases', 'Deaths', 'Death rate']):        
        sns.regplot(x = data_2015[col_j], y = data_2015[col_i], ci=None, ax=ax[i, j])
        if(i != 4):
            #ax[i, j].get_xaxis().set_ticks([])
            ax[i, j].set_xlabel('')
        if(j != 0):
            #ax[i, j].get_yaxis().set_ticks([])
            ax[i, j].set_ylabel('')
        

plt.suptitle('Relationship between indexes', fontsize=16, fontweight='bold')
plt.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()


# Scatterplots whose data follow a kind of linear relationship have higher correlations. This is the case of the plots in the third column, for example.
# 
# ### What is the impact of water treatment in the spread of cholera? <a id ='eda-water'></a>
# 
# To answer this question, I'll use a dataset that I created and made available [here](https://www.kaggle.com/mateuscco/sanitation-and-water-global-indexes) at Kaggle. It has six indexes reported to the World Health Organization by countries every year since 2000. 
# 
# - Population using safely managed drinking-water services (%)
# - Population using at least basic drinking-water services (%)
# - Population using safely managed sanitation services (%)
# - Population using at least basic sanitation services (%)
# - Population with basic handwashing facilities at home (%)
# - Population practicing open defecation (%) 
# 
# They are divided by the type of residence (urban and rural) and also have an agreggated total. In this analysis I will only use the total columns and focus on the data from 2015.

# In[ ]:


health_indexes = pd.read_csv('../input/sanitation-and-water-global-indexes/SanitationAndWaterGlobalIndexes/indexes.csv')

health_indexes = health_indexes.loc[health_indexes['Year'] == 2015, [
    'Country',
    'Population using at least basic drinking-water services (%) - Total',
    'Population using safely managed drinking-water services (%) - Total',
    'Population using at least basic sanitation services (%) - Total',
    'Population using safely managed sanitation services (%) - Total',
    'Population with basic handwashing facilities at home (%) - Total',
    'Population practising open defecation (%) - Total'
]]

health_indexes = health_indexes.set_index('Country').rename(columns={
    'Population using at least basic drinking-water services (%) - Total':'Basic drinking-water services (%)',
    'Population using safely managed drinking-water services (%) - Total':'Safe drinking-water services (%)',
    'Population using at least basic sanitation services (%) - Total':'Basic sanitation services (%)',
    'Population using safely managed sanitation services (%) - Total':'Safe sanitation services (%)',
    'Population with basic handwashing facilities at home (%) - Total':'Basic handwashing facilities (%)',
    'Population practising open defecation (%) - Total':'Open defecation (%)'
})


# In[ ]:


print('In 2015, {} countries at least one of the above indexes.'.format(health_indexes.shape[0]))

health_indexes.isnull().sum() * 100 / health_indexes.shape[0]


# Above we can see that some metrics, like access to basic handwashing facilities, have a high percentage of missing values.

# In[ ]:


health_data_2015 = data_2015.loc[:, ['Cases', 'Deaths', 'Death rate']].join(health_indexes, how='left')

correlations = health_data_2015.corr(method='pearson')
correlations = correlations.iloc[:3, 3:]

sns.heatmap(data=correlations.T, annot=True, color=sns.color_palette("coolwarm", 7))
plt.title('Correlation between public health indexes and cholera metrics', fontsize=16, fontweight='bold')
plt.xticks(rotation=0) 
plt.yticks(rotation=0) 
plt.show()


# Once again we have correlations that shows meaningful insights.
# 
# - First, the percentages of population with access to safe sanitation and water-drinking services are the variables with the lowest correlations between death rate. In other words, countries where these metrics are low have a high death rate. As we already know, public water sanitation is a key strategy to prevent new cases and this chart shows why.
# - Surprisingly, easy access to basic handwashing facilities isn't much correlated with cholera incidence.
# - It's worth noticing that there is a considerable positive correlation between the percentage of population practicing open defecation and death rates. That is, countries with high percentages of open defecation usually have higher cholera death rates.

# In[ ]:




