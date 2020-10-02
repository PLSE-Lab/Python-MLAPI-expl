#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings 
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.cm as cm
import seaborn as sns

import collections
from wordcloud import WordCloud, STOPWORDS

import os
print(os.listdir("../input"))


# # Data

# In[ ]:


migrants = pd.read_csv('../input/MissingMigrants-Global-2019-03-29T18-36-07.csv')
migrants.drop(['Web ID', 'URL'], axis = 1, inplace=True)
migrants.head()


# In[ ]:


migrants[migrants['Region of Incident'] == 'Mediterranean'].groupby(['Cause of Death'])['Total Dead and Missing'].sum().sort_values(ascending=False)[:15]


# In[ ]:


migrants.describe(exclude='O')


# In[ ]:


migrants.describe(exclude='number')


# In[ ]:


migrants.info()


# In[ ]:


# Convert string month into numerical one
migrants['Reported Month(Number)'] = pd.to_datetime(migrants['Reported Month'], format='%b').apply(lambda x: x.month)

migrants[migrants['Reported Year'] == 2014]['Reported Month(Number)'].min(), migrants[migrants['Reported Year'] == 2019]['Reported Month(Number)'].max()


# - Data from Jan.2014 to March.2015

# In[ ]:


migrants.loc[:, 'Minimum Estimated Number of Missing'].sum(), migrants.loc[:, 'Number of Survivors'].sum(), 


# In[ ]:


print(migrants.loc[:, 'Number of Children'].sum())
migrants.loc[:, 'Number of Children'].plot.box()


# In[ ]:


print(migrants.loc[:, 'Number of Males'].sum())
migrants.loc[:, 'Number of Males'].plot.box()


# In[ ]:


print(migrants.loc[:, 'Number of Females'].sum())
migrants.loc[:, 'Number of Females'].plot.box()


# In[ ]:


migrants.loc[:, 'Number of Survivors'].plot.box()


# ### Number of missing values

# In[ ]:


na_sum = []
for col in migrants.columns:
    na_sum.append(migrants[col].isna().sum())

migrants_df = pd.DataFrame({'cols':migrants.columns,
                            'total_na' : na_sum})

migrants_df = migrants_df.sort_values(by='total_na', ascending=False).drop(list(migrants_df[migrants_df.total_na == 0].index), axis=0)
migrants_df.plot.bar(x = 'cols', y = 'total_na', rot=85, fontsize=18)
del migrants_df


# # HeatmapWithTimestamp by using folium
# - See below animation with leftmost below date

# In[ ]:


import folium
from folium.plugins import HeatMapWithTime


migrants['Location Coordinates'].fillna('0, 0', inplace = True) # initialize missing value into 0, 0  location
migrants['lat'] = migrants['Location Coordinates'].apply(lambda x: float(str(x).split(', ')[0]))
migrants['lon'] = migrants['Location Coordinates'].apply(lambda x: float(str(x).split(', ')[1]))

basemap = folium.folium.Map(location = [migrants['lat'].median(), migrants['lon'].median()], zoom_start = 2)

indexes = ['{}/{}'.format(month, year) for year in migrants['Reported Year'].unique()[::-1] for month in range(1, 13)]

heat_data = [[[row['lat'], row['lon'], row['Total Dead and Missing']] for _, row in migrants[migrants['Reported Year'] == year][migrants['Reported Month(Number)'] == month].iterrows()]
             for year in migrants['Reported Year'].unique()[::-1] for month in range(1, 13)]

HeatMapWithTime(heat_data, auto_play = True, index = indexes, display_index = indexes).add_to(basemap)
basemap.save('Animated heatmap of migrants death or missing from 2014 to 2019 by month')
basemap


# - Most of incidents occur in Mexico-US border, Mediterranean, North Africa and Horn of Africa

# ### Number of incidents by region, migration route and UNSD grouping

# In[ ]:


migrants['Region of Incident'].value_counts().plot.bar(rot=80, fontsize=18)


# In[ ]:


migrants['Migration Route'].value_counts().plot.bar(rot=80, fontsize=18)


# In[ ]:


migrants['UNSD Geographical Grouping'].value_counts().plot.bar(rot=80, fontsize=18)


# # Finding reason of death using wordcloud

# In[ ]:


all_cause_death          = ' '.join(migrants['Cause of Death'].str.lower())
all_location_description = ' '.join(migrants['Location Description'].str.lower().fillna(' '))
all_information_source   = ' '.join(migrants['Information Source'].str.lower().fillna(' '))


# In[ ]:


def words_frequency(corpus):
    stopwords = STOPWORDS
    
    wordcloud = WordCloud(stopwords=stopwords, background_color="white", max_words=150).generate(corpus) 
    rcParams['figure.figsize'] = 10, 20
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    
    # Split corpus into each words
    filtered_words = [word for word in corpus.split() if word not in stopwords]
    
    # Make counter object that have each count of word
    counted_words = collections.Counter(filtered_words)
    
    # Store most common words
    words = []
    counts = []
    for letter, count in counted_words.most_common(10):
        words.append(letter)
        counts.append(count)

    rcParams['figure.figsize'] = 20, 10        # set figure size

    plt.title('Top words in the corpus vs their count')
    plt.xlabel('Count')
    plt.ylabel('Words')
    plt.barh(words, counts, color=cm.rainbow(np.linspace(0, 1, 10)))


# In[ ]:


words_frequency(all_cause_death)


# In[ ]:


words_frequency(all_location_description)


# In[ ]:


words_frequency(all_information_source)


# In[ ]:


migrants.loc[:, ['Total Dead and Missing', 'Number Dead', 'Number of Survivors']].plot.kde()
migrants.loc[:, ['Number of Males', 'Number of Females', 'Number of Children']].plot.kde()
migrants.loc[:, ['Total Dead and Missing']].plot.kde()


# In[ ]:


migrants['Region of Incident'].value_counts()[:10], migrants['Migration Route'].value_counts()[:10], migrants['UNSD Geographical Grouping'].value_counts()[:10]


# In[ ]:


def col_frequency_with_df(df, col):
    corpus = ' '.join(df[col].str.lower())
    return words_frequency(corpus)


# In[ ]:


col_frequency_with_df(migrants.loc[migrants['UNSD Geographical Grouping'] == 'Northern Africa',  :], 'Cause of Death')


# In[ ]:


col_frequency_with_df(migrants.loc[migrants['UNSD Geographical Grouping'] == 'Northern America', :], 'Cause of Death')


# In[ ]:


col_frequency_with_df(migrants.loc[migrants['UNSD Geographical Grouping'] == 'Uncategorized',  :], 'Cause of Death')


# - Dead migrants at North Africa mostly died cause lack of medicine, vehicle accidents
# - Dead migrants at North America mostly found as skeleton and reason of death are mostly unknown
# - Dead migrants at uncategorized Mostly drowned, I think uncategorized just indicate Mediterranean

# In[ ]:


col_frequency_with_df(migrants.loc[migrants['Region of Incident'] == 'US-Mexico Border',    :], 'Cause of Death')


# In[ ]:


col_frequency_with_df(migrants.loc[migrants['Region of Incident'] == 'North Africa',    :], 'Cause of Death')


# In[ ]:


col_frequency_with_df(migrants.loc[migrants['Region of Incident'] == 'Mediterranean',    :], 'Cause of Death')


# - Result is similar to previous 'UNSD Geographical Grouping' part

# In[ ]:


col_frequency_with_df(migrants.loc[migrants['Migration Route'] == 'Central America to US',    :], 'Cause of Death')


# In[ ]:


col_frequency_with_df(migrants.loc[migrants['Migration Route'] == 'Central Mediterranean',    :], 'Cause of Death')


# In[ ]:


col_frequency_with_df(migrants.loc[migrants['Migration Route'] == 'Western Mediterranean',    :], 'Cause of Death')


# In[ ]:


col_frequency_with_df(migrants.loc[migrants['Migration Route'] == 'Eastern Mediterranean',    :], 'Cause of Death')


# - Similar to previous parts

# In[ ]:


migrants['Reported Year'].value_counts().sort_index().plot.bar()


# - Incidents are increasing from 2014 to 2018

# In[ ]:


migrants['Migration Route'].value_counts()


# In[ ]:


col_frequency_with_df(migrants.loc[migrants['Reported Year'] == 2014,    :], 'Cause of Death')


# In[ ]:


col_frequency_with_df(migrants.loc[migrants['Reported Year'] == 2015,    :], 'Cause of Death')


# In[ ]:


col_frequency_with_df(migrants.loc[migrants['Reported Year'] == 2016,    :], 'Cause of Death')


# In[ ]:


col_frequency_with_df(migrants.loc[migrants['Reported Year'] == 2017,    :], 'Cause of Death')


# In[ ]:


col_frequency_with_df(migrants.loc[migrants['Reported Year'] == 2018,    :], 'Cause of Death')


# - What about reason of death on specific year on specific region or migration route?

# In[ ]:


col_frequency_with_df(migrants[migrants['Reported Year'] == 2014][migrants['Region of Incident'] == 'US-Mexico Border'], 'Cause of Death')


# In[ ]:


col_frequency_with_df(migrants[migrants['Reported Year'] == 2015][migrants['Region of Incident'] == 'US-Mexico Border'], 'Cause of Death')


# In[ ]:


col_frequency_with_df(migrants[migrants['Reported Year'] == 2016][migrants['Region of Incident'] == 'US-Mexico Border'], 'Cause of Death')


# In[ ]:


col_frequency_with_df(migrants[migrants['Reported Year'] == 2017][migrants['Region of Incident'] == 'US-Mexico Border'], 'Cause of Death')


# In[ ]:


col_frequency_with_df(migrants[migrants['Reported Year'] == 2018][migrants['Region of Incident'] == 'US-Mexico Border'], 'Cause of Death')


# In[ ]:


def unique_year(df, location_col, location_value, value_counts_col):
    for year in list(df['Reported Year'].unique())[::-1]:
        print(year)
        counts = df[df['Reported Year'] == year][df[location_col] == location_value][value_counts_col].value_counts()
        print(counts[:5])
        print('Total : {}'.format(counts.sum()))
        print('-' * 30)
        
def n_deadmissing_by_year(df, sum_col, location_col, location_value):
    """Return sum of values of some column using grouped values"""
    
    for year in list(df['Reported Year'].unique())[::-1]:
        print(year)
        Sum = df[df['Reported Year'] == year][df[location_col] == location_value].groupby(['Cause of Death'])[sum_col].sum().sort_values(ascending=False) 
        print(Sum[:15])
        print('Total : {}'.format(Sum.sum()))
        print('-' * 30)


# In[ ]:


unique_year(migrants, 'Region of Incident', 'US-Mexico Border', 'Cause of Death')


# - Reason of migrants death from 2014 to 2018 in US-mexico border is mostly unknown, drowning and hyperthemia.
# - Also let's see about number of dead or missing migrants grouped by their reason of death not only about number of incidents.

# In[ ]:


# Number of deaths by reason of death from 2014 to 2019
print(migrants[migrants['Region of Incident'] == 'US-Mexico Border'].groupby(['Cause of Death'])['Total Dead and Missing'].sum().sort_values(ascending=False)[:15])
print('@' * 30)

n_deadmissing_by_year(migrants, 'Total Dead and Missing', 'Region of Incident', 'US-Mexico Border')


# - Most of case migrants died for reason such as mixed,unknown(skeleton remains) and drowning etc.

# In[ ]:


col_frequency_with_df(migrants[migrants['Reported Year'] == 2014][migrants['Region of Incident'] == 'North Africa'], 'Cause of Death')


# In[ ]:


col_frequency_with_df(migrants[migrants['Reported Year'] == 2015][migrants['Region of Incident'] == 'North Africa'], 'Cause of Death')


# In[ ]:


col_frequency_with_df(migrants[migrants['Reported Year'] == 2016][migrants['Region of Incident'] == 'North Africa'], 'Cause of Death')


# In[ ]:


col_frequency_with_df(migrants[migrants['Reported Year'] == 2017][migrants['Region of Incident'] == 'North Africa'], 'Cause of Death')


# In[ ]:


col_frequency_with_df(migrants[migrants['Reported Year'] == 2018][migrants['Region of Incident'] == 'North Africa'], 'Cause of Death')


# In[ ]:


col_frequency_with_df(migrants[migrants['Reported Year'] == 2019][migrants['Region of Incident'] == 'North Africa'], 'Cause of Death')


# In[ ]:


unique_year(migrants, 'Region of Incident', 'North Africa', 'Cause of Death')


# - In North Africa case, most of incidnets imply the reason of death is lack of food/water/medicine and vehicle accident.

# In[ ]:


# Number of deaths by reason of death from 2014 to 2019 
print(migrants[migrants['Region of Incident'] == 'North Africa'].groupby(['Cause of Death'])['Total Dead and Missing'].sum().sort_values(ascending=False)[:15])
print('@' * 30)

# Number of deaths by reason of death by year from 2014 to 2019
n_deadmissing_by_year(migrants, 'Total Dead and Missing', 'Region of Incident', 'North Africa')


# Most of migrants in North Africa died for reason such as lack of food/water/medicine/shelter, vehicle accident and violence/abuse/murder

# In[ ]:


col_frequency_with_df(migrants[migrants['Reported Year'] == 2014][migrants['Region of Incident'] == 'Mediterranean'], 'Cause of Death')


# In[ ]:


col_frequency_with_df(migrants[migrants['Reported Year'] == 2015][migrants['Region of Incident'] == 'Mediterranean'], 'Cause of Death')


# In[ ]:


col_frequency_with_df(migrants[migrants['Reported Year'] == 2016][migrants['Region of Incident'] == 'Mediterranean'], 'Cause of Death')


# In[ ]:


col_frequency_with_df(migrants[migrants['Reported Year'] == 2017][migrants['Region of Incident'] == 'Mediterranean'], 'Cause of Death')


# In[ ]:


col_frequency_with_df(migrants[migrants['Reported Year'] == 2018][migrants['Region of Incident'] == 'Mediterranean'], 'Cause of Death')


# In[ ]:


col_frequency_with_df(migrants[migrants['Reported Year'] == 2019][migrants['Region of Incident'] == 'Mediterranean'], 'Cause of Death')


# In[ ]:


unique_year(migrants, 'Region of Incident', 'Mediterranean', 'Cause of Death')


# In[ ]:


# Number of deaths by reason of death from 2014 to 2019 
print(migrants[migrants['Region of Incident'] == 'Mediterranean'].groupby(['Cause of Death'])['Total Dead and Missing'].sum().sort_values(ascending=False)[:15])
print('@' * 30)

# Number of deaths by reason of death by year from 2014 to 2019
n_deadmissing_by_year(migrants, 'Total Dead and Missing', 'Region of Incident', 'Mediterranean')


# - Migrants in Mediterranean died for reason such as drowning mostly

# # Number of incidents on different location 

# In[ ]:


import matplotlib as mpl

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : 18}

lines = {'linewidth' : 2}

mpl.rc('font', **font)
mpl.rc('lines', **lines)

f, axes = plt.subplots(6, 1, figsize=(7, 5), sharex=True)

for year, ax in zip(migrants['Reported Year'].unique(), axes):
    sns.barplot(x=list(pd.DataFrame(migrants[migrants['Reported Year'] == year]['Migration Route'].value_counts())[:5].index),  y='Migration Route',
                palette="rocket", ax=ax, data = pd.DataFrame(migrants[migrants['Reported Year'] == year]['Migration Route'].value_counts())[:5])
    ax.axhline(0, color="k", clip_on=False)
    ax.set_ylabel(year)

plt.xticks(rotation=45)
plt.rcParams.update({'font.size': 22})
plt.show()

del font, lines, f, axes


# In[ ]:


def years_value_counts(df, col):    
    
    # Store whole value_counts series with their particular year into list
    stats_of_years = [pd.DataFrame(df[df['Reported Year'] == year][col].value_counts()) for year in df['Reported Year'].unique()]
    
    # concat dfs with their corresponding column
    stats_of_years         =  pd.concat(stats_of_years, axis=1)
    stats_of_years.columns = df['Reported Year'].unique()
    stats_of_years.fillna(0, inplace=True)
    return stats_of_years


# In[ ]:


route_year   = years_value_counts(migrants, 'Migration Route')
UNSD_year    = years_value_counts(migrants, 'UNSD Geographical Grouping')
region_year  = years_value_counts(migrants, 'Region of Incident')


# In[ ]:


import plotly.graph_objs as go
import plotly            as py
from plotly.offline      import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

fig = go.Figure(data=[go.Bar(name = str(col), x = route_year.transpose().index, y = route_year.transpose()[col]) 
                      for col in route_year.transpose().columns],
                
                layout = dict(
                    xaxis = dict(
                        title = dict(text = 'Number of incidents on specific migration route by year', font = dict(size=18))),
                    barmode = 'stack'))

iplot(fig)


# - Basically number of incidents increased since 2014 until 2016 with relatively large gap and decreased from 16 to 18 slightly. 
# - Most of incidents occured at Central America to US. 
# - Central Mediterranean increase until 2017. In 2018, decrease until 1/3 amount of number of incidents on 2017.
# - Also incidents of Eastern Mediterranean decrease 2015-2017 while incidents of Western Mediterranean increase 2015-2018.
# - So most of incidents(not equivalent to number of dead or missing migrants) have occurred at borderline of US-Mexico and mediterranean.

# In[ ]:


fig = go.Figure(data=[go.Bar(name = str(col), x = region_year.transpose().index, y = region_year.transpose()[col]) 
                      for col in region_year.transpose().columns],
                
                layout = dict(
                    xaxis = dict(
                        title = dict(text = 'Number of incidents on specific region by year', font = dict(size=18))),
                    barmode = 'stack'))

iplot(fig)


# - US-Mexico border is increasing since 2014
# - Number of incidents of North Africa and Horn of Africa  fluctuate from 2015 to 2018 while number of incidents on Mediterranean keep maintained around 200

# In[ ]:


fig = go.Figure(data=[go.Bar(name = str(col), x = UNSD_year.transpose().index, y = UNSD_year.transpose()[col]) 
                      for col in UNSD_year.transpose().columns],
                
                layout = dict(
                    xaxis = dict(
                        title = dict(text = 'Number of incidents on UNSD geo grouping location by year', font = dict(size=18))),
                    barmode = 'stack'))

iplot(fig)


# In[ ]:


fig = go.Figure(data=[go.Bar(name = str(year), 
                             x = migrants[migrants['Reported Year'] == year]['Reported Month(Number)'].value_counts().sort_index().index, 
                             y = migrants[migrants['Reported Year'] == year]['Reported Month(Number)'].value_counts().sort_index()) 
                      for year in migrants['Reported Year'].unique()[::-1]],
                
                layout = dict(
                    xaxis = dict(
                        title = dict(text = 'Number of migrant incidents by month on specific year', font = dict(size=18))),
                    barmode = 'stack'))

iplot(fig)


# In[ ]:


del route_year, UNSD_year, region_year


# - Let's see number of death or missing migrants by month on each region

# In[ ]:


pd.to_datetime(migrants['Reported Month'], format='%b').apply(lambda x: x.month).value_counts().sort_index().plot.bar()


# - We can see some functuation.Number of missing or daed migrants decrease Jan-April and increase April-October, then decrease again October-April roughly.

# # Number of missing or death by year

# In[ ]:


months = [m for m in range(1,13)]
years  = migrants['Reported Year'].unique()[::-1]

fig = go.Figure(data=[go.Bar(
    name = str(year), x = months,
    
    y = [migrants[migrants['Reported Year'] == year][migrants['Reported Month(Number)'] == month]['Total Dead and Missing'].sum() 
         for month in months]) 
                      for year in years],
                
                layout = dict(
                    xaxis = dict(
                        title = dict(text = 'Number of missing or dead migrants by month on specific year', font = dict(size=18))),
                    barmode = 'group'))

iplot(fig)


# In[ ]:


fig = go.Figure(data=[go.Bar(
    name = str(year), x = [m for m in range(1,13)],
    
    # List of number of migrants missing or death of months by specific year
    y = [
        migrants[migrants['Reported Year'] == year][migrants['Reported Month(Number)'] == month]['Total Dead and Missing'].sum() 
        for month in [m for m in range(1,13)]
    ] 
) for year in migrants['Reported Year'].unique()[::-1]],
                
                layout = dict(
                    xaxis = dict(
                        title = dict(text = 'Number of missing or dead migrants by month on specific year', font = dict(size=18))),
                    barmode = 'stack'))

iplot(fig)


# ### 2014
# - Mostly die from July to September and December
# 
# ### 2015
# - There is huge amount of migrants died or disappeared on April
# 
# ### 2016
# - Number of dead or missing migrants are not constant
# 
# ### 2017
# - There is big number of died or disappeared migrants on May and June 
# 
# ### 2018
# - There is big number of died or disappeared migrants on June

# In[ ]:


def groupbar_by_month(df, region_name, barmode = 'stack'):
    months = [m for m in range(1,13)]
    years  = df['Reported Year'].unique()[::-1]

    fig = go.Figure(data=[go.Bar(
        name = str(year), x = months,

        y = [df[df['Region of Incident'] == region_name][df['Reported Year'] == year][df['Reported Month(Number)'] == month]['Total Dead and Missing'].sum() 
             for month in months]) 
                          for year in years],
            
                    layout = dict(
                        xaxis = dict(
                            title = dict(text = 'Number of missing or dead migrants by month on specific year in {}'.format(region_name), font = dict(size=18))),
                        barmode = barmode))
    
    return iplot(fig)


# In[ ]:


groupbar_by_month(migrants, 'US-Mexico Border', barmode = 'stack')
groupbar_by_month(migrants, 'US-Mexico Border', barmode = 'group')


# ## Number of missing or death on US-Mexico border by year
# - Mostly there is specific month migrants die or disappear by year
# 
# ### 2014
# - Diying or missing of migrants increase July to September.
# 
# ### 2015
# - Diying or missing of migrants occurred mostly on December
# 
# ### 2016
# - Diying or missing of migrants mostly occurred except January - May and November.
# 
# ### 2017
# - There is flunctuation on months except December. And big Diying or missing of migrants occurred on December.
# 
# ### 2018
# - Number of died or disappeared migrants are constant except June and August.
# 

# In[ ]:


groupbar_by_month(migrants, 'North Africa', barmode = 'stack')
groupbar_by_month(migrants, 'North Africa', barmode = 'group')


# ## Number of missing or death on North Africa by year
# 
# ### 2014
# - There is just small amount of dying or missing of migrants just from April to June, July(1),  August(3) and December(1). Mostly on April to June.
# 
# ### 2015
# - Most of diying or missing of migrants occurred from October to December.
# 
# ### 2016
# - There is big diying or missing of migrants occurred on February.
# 
# ### 2017
# - Diying or missing of migrants increase from April to November and decrease until April
# 
# ### 2018
# - There is just few of diying or missing of migrants from Septembe to December. And curve is more decreasing form

# In[ ]:


groupbar_by_month(migrants, 'Mediterranean', barmode = 'stack')
groupbar_by_month(migrants, 'Mediterranean', barmode = 'group')


# ## Number of missing or death on Mediterranean by year
# 
# ### 2014
# - Most of dying or missing occurred from May to September.
# 
# ### 2015
# - There is big dying or missing at April, secondly June, thirdly October, fourthly December
# 
# ### 2016
# - There is mountain from March to July and peak at May. They're increase from September to November.
# 
# ### 2017
# - Most of diying or missing of migrants occurred on first half. And they have increasing form
# 
# ### 2018
# - There is big diying or missing of migrants on June and decreasing form from September to December.

# ## Missing or death of male, female and children by year

# In[ ]:


fig = go.Figure(data=[go.Bar(
    name = col, x = migrants['Reported Year'].unique()[::-1],
    
    y = [
        migrants[migrants['Reported Year'] == year][col].sum() 
        for year in migrants['Reported Year'].unique()[::-1]
    ]
) for col in ['Number of Females', 'Number of Males', 'Number of Children']],
                
                layout = dict(
                    xaxis = dict(
                        title = dict(text = 'Number of missing or dead male, female and children migrants by year', font = dict(size=18))),
                    barmode = 'stack'))

iplot(fig)
del fig


# In[ ]:


def mfc_death_by_year(df, region, barmode = 'stack'):
    fig = go.Figure(data=[go.Bar(
        name = col, x = df['Reported Year'].unique()[::-1],

        y = [
            df[df['Region of Incident'] == region][df['Reported Year'] == year][col].sum() 
            for year in df['Reported Year'].unique()[::-1]
        ] 
    ) for col in ['Number of Females', 'Number of Males', 'Number of Children']],

                    layout = dict(
                        xaxis = dict(
                            title = dict(text = 'Number of missing or dead male, female and children migrants by year in {}'.format(region), font = dict(size=18))),
                        barmode = barmode))

    return iplot(fig)


# In[ ]:


for region in migrants['Region of Incident'].value_counts().index:
    mfc_death_by_year(migrants, region)


# In[ ]:


for region in migrants['Region of Incident'].value_counts().index:
    mfc_death_by_year(migrants, region, barmode = 'group')


#  By just seeing charts we can easily know dead migrants are mostly male, however we don't know about total migrants population and sex/children composition of population. So we don't know actually male can easily die during migration with same(or even similar) population or there is so much male migrants with same fatalities. 
