#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Libraries" data-toc-modified-id="Libraries-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Libraries</a></span></li><li><span><a href="#Functions" data-toc-modified-id="Functions-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Functions</a></span></li><li><span><a href="#Data-Prep" data-toc-modified-id="Data-Prep-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Data Prep</a></span></li><li><span><a href="#Terrorism-Around-the-World" data-toc-modified-id="Terrorism-Around-the-World-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Terrorism Around the World</a></span></li><li><span><a href="#Countries-and-Terrorism" data-toc-modified-id="Countries-and-Terrorism-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Countries and Terrorism</a></span></li><li><span><a href="#NLP-on-Incident-Summary" data-toc-modified-id="NLP-on-Incident-Summary-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>NLP on Incident Summary</a></span><ul class="toc-item"><li><span><a href="#Regular-Expressions" data-toc-modified-id="Regular-Expressions-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Regular Expressions</a></span><ul class="toc-item"><li><span><a href="#Sites-and-Hiperlinks" data-toc-modified-id="Sites-and-Hiperlinks-6.1.1"><span class="toc-item-num">6.1.1&nbsp;&nbsp;</span>Sites and Hiperlinks</a></span></li><li><span><a href="#Numbers" data-toc-modified-id="Numbers-6.1.2"><span class="toc-item-num">6.1.2&nbsp;&nbsp;</span>Numbers</a></span></li><li><span><a href="#Special-Characteres" data-toc-modified-id="Special-Characteres-6.1.3"><span class="toc-item-num">6.1.3&nbsp;&nbsp;</span>Special Characteres</a></span></li><li><span><a href="#Additional-Whitespaces" data-toc-modified-id="Additional-Whitespaces-6.1.4"><span class="toc-item-num">6.1.4&nbsp;&nbsp;</span>Additional Whitespaces</a></span></li></ul></li><li><span><a href="#Lower-Case" data-toc-modified-id="Lower-Case-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>Lower Case</a></span></li><li><span><a href="#WordCloud" data-toc-modified-id="WordCloud-6.3"><span class="toc-item-num">6.3&nbsp;&nbsp;</span>WordCloud</a></span></li></ul></li><li><span><a href="#References" data-toc-modified-id="References-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# * This kernel is a continuation of the kernel [Global Terrorism - Data Understanding](https://www.kaggle.com/thiagopanini/global-terrorism-data-understanding), where an extremely important process of understanding the data was carried out in order to go deep into the context and make useful data transformations for further analysis.
# 
# This second approach intends to tell a story about Terrorism around the world taking the data provided as a basis. Here, we will apply an exploratory data analysis, look for patterns and explanations related to the context and present the conclusions in a dynamic and visual ways. If you like this kernel, **please upvote**! This always keep me motivated to do things even better! English is not my mother language... so sorry for any mistake!
# 
# We will use libs like **Folium**, **Seaborn**, **Matplotlib** and other usefull tools to try to see:

# * A big picture of terrorism around the world and its evolution over the years;
# * Countries with most incidents recorded;
# * Countries with highest number of victims;
# * A dashboard for terrorism analysis in some countries;
# * Incidents that lasted more than 24h (extended = 1);
# * Major radical groups responsible for terrorist attacks (gname);
# * Attacks with the highest number of terrorists (nperps);
# * A WordCloud for attributes like summary corp1, target1 and motive;

# # Libraries

# In[ ]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import folium
from folium.plugins import FastMarkerCluster, Fullscreen, MiniMap, HeatMap, HeatMapWithTime
import geopandas as gpd
from branca.colormap import LinearColormap
import os
import json
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import re
from wordcloud import WordCloud, STOPWORDS


# # Functions

# In[ ]:


def style_function(feature):
    """
    Customize maps
    """
    return {
        'fillColor': '#ffaf00',
        'color': 'grey',
        'weight': 1.5,
        'dashArray': '5, 5'
    }

def highlight_function(feature):
    """
    Customize maps
    """
    return {
        'fillColor': '#ffaf00',
        'color': 'black',
        'weight': 2,
        'dashArray': '5, 5'
    }

def format_spines(ax, right_border=True):
    """
    This function sets up borders from an axis and personalize colors
    
    Input:
        Axis and a flag for deciding or not to plot the right border
    Returns:
        Plot configuration
    """    
    # Setting up colors
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['top'].set_visible(False)
    if right_border:
        ax.spines['right'].set_color('#CCCCCC')
    else:
        ax.spines['right'].set_color('#FFFFFF')
    ax.patch.set_facecolor('#FFFFFF')
    
def count_plot(feature, df, colors='Blues_d', hue=False, ax=None, title=''):
    """
    This function plots data setting up frequency and percentage in a count plot;
    This also sets up borders and personalization.
    
    Input:
        The feature to be counted and the dataframe. Other args are optional.
    Returns:
        Count plot.
    """    
    # Preparing variables
    ncount = len(df)
    if hue != False:
        ax = sns.countplot(x=feature, data=df, palette=colors, hue=hue, ax=ax, 
                           order=df[feature].value_counts().index)
    else:
        ax = sns.countplot(x=feature, data=df, palette=colors, ax=ax,
                           order=df[feature].value_counts().index)

    # Make twin axis
    ax2=ax.twinx()

    # Switch so count axis is on right, frequency on left
    ax2.yaxis.tick_left()
    ax.yaxis.tick_right()

    # Also switch the labels over
    ax.yaxis.set_label_position('right')
    ax2.yaxis.set_label_position('left')
    ax2.set_ylabel('Frequency [%]')
    frame1 = plt.gca()
    frame1.axes.get_yaxis().set_ticks([])

    # Setting up borders
    format_spines(ax)
    format_spines(ax2)

    # Setting percentage
    for p in ax.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
                ha='center', va='bottom') # set the alignment of the text
    
    # Final configuration
    if not hue:
        ax.set_title(df[feature].describe().name + ' Counting plot', size=13, pad=15)
    else:
        ax.set_title(df[feature].describe().name + ' Counting plot by ' + hue, size=13, pad=15)  
    if title != '':
        ax.set_title(title)       
    plt.tight_layout()
    
def country_analysis(country_name, data, palette, colors_plot2, color_lineplot):
    """
    This function creates a dashboard with informations of terrorism in a certain country.
    Input:
        The function receives the name of the country, the dataset and color configuration
    Output:
        It returns a 4 plot dashboard.
    """
    # Preparing
    country = data.query('country_txt == @country_name')
    if len(country) == 0:
        print('Country did not exists in dataset')
        return 
    country_cities = country.groupby(by='city', as_index=False).count().sort_values('eventid', 
                                                                                   ascending=False).iloc[:5, :2]
    suicide_size = country['suicide'].sum() / len(country)
    labels = ['Suicide', 'Not Suicide']
    colors = colors_plot2
    
    country_year = country.groupby(by='iyear', as_index=False).sum().loc[:, ['iyear', 'nkill']]
    country_weapon = country.groupby(by='weaptype1_txt', as_index=False).count().sort_values(by='eventid',
                                                                                             ascending=False).iloc[:, 
                                                                                                                   :2]
    # Dashboard
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
    
    # Plot 1 - Top 5 terrorism cities
    sns.barplot(x='eventid', y='city', data=country_cities, ci=None, palette=palette, ax=axs[0, 0])
    format_spines(axs[0, 0], right_border=False)
    axs[0, 0].set_title(f'Top 5 {country_name} Cities With Most Terrorism Occurences')
    """for p in axs[0, 0].patches:
        width = p.get_width()
        axs[0, 0].text(width-290, p.get_y() + p.get_height() / 2. + 0.10, '{}'.format(int(width)), 
                ha="center", color='white')"""
    axs[0, 0].set_ylabel('City')
    axs[0, 0].set_xlabel('Victims')
    
    # Plot 2 - Suicide Rate
    center_circle = plt.Circle((0,0), 0.75, color='white')
    axs[0, 1].pie((suicide_size, 1-suicide_size), labels=labels, colors=colors_plot2, autopct='%1.1f%%')
    axs[0, 1].add_artist(center_circle)
    format_spines(axs[0, 1], right_border=False)
    axs[0, 1].set_title(f'{country_name} Terrorism Suicide Rate')
    axs[0, 0].set_ylabel('Victims')
    
    # Plot 3 - Victims through the years
    sns.lineplot(x='iyear', y='nkill', data=country_year, ax=axs[1, 0], color=color_lineplot)
    format_spines(axs[1, 0], right_border=False)
    axs[1, 0].set_xlim([1970, 2017])
    axs[1, 0].set_title(f'{country_name} Number of Victims Over Time')
    axs[1, 0].set_ylabel('Victims')
    
    # Plot 4 - Terrorism Weapons
    sns.barplot(x='weaptype1_txt', y='eventid', data=country_weapon, ci=None, palette=palette, ax=axs[1, 1])
    axs[1, 1].set_xticklabels(axs[1, 1].get_xticklabels(), rotation=90)
    axs[1, 1].set_xlabel('')
    axs[1, 1].set_ylabel('Count')
    format_spines(axs[1, 1], right_border=False)
    axs[1, 1].set_title(f'{country_name} Weapons Used in Attacks')
    
    plt.suptitle(f'Terrorism Analysis in {country_name} between 1970 and 2017', size=16)    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()


# # Reading the Data

# In previous versions, I separated the analysis into "Data Preparation" and "Storytelling" just to keep in mind that some transformations in data would be necessary to reach the goals. But, as long as the storytelling part took the center stage and grew on large scale, I decided to make the transformations on data only where applicable.
# 
# In other words, all the steps required for the analysis will be performed on their respective topic. For now, let's just read the data, filter attributes and go through visualizations!

# In[ ]:


terr = pd.read_csv('../input/globalterrorismdb_0718dist.csv', encoding='ISO-8859-1')
attribs = ['eventid', 'iyear', 'imonth', 'iday', 'extended', 'country_txt', 'region_txt', 'city', 
                        'latitude', 'longitude', 'specificity', 'summary', 'success', 'suicide', 'attacktype1_txt', 
                        'targtype1_txt', 'copr1', 'target1', 'natlty1_txt', 'gname', 'motive', 'nperps', 
                        'weaptype1_txt', 'nkill', 'nkillter', 'nwound', 'nwoundte', 'ishostkid', 'nhostkid']
terr_data = terr.loc[:, attribs]
terr_data.head()


# Before we get started, we need to ensure the charts will be plotted the way we expect. First, we have to set the name of United States country to same string that we have in the json file used as map (United States of America). Second, let's just change the large string describing the "Vehicle" in `weaptype1_txt` attribute.

# In[ ]:


terr_data['country_txt'] = terr_data['country_txt'].apply(lambda x: x.replace('United States', 
                                                                              'United States of America'))
terr_data['weaptype1_txt'] = terr_data['weaptype1_txt'].apply(lambda x: x.split()[0] if 'Vehicle' in x.split() else x)


# If you want to see details of a full process of global terrorism data preparation, please go through the [Global Terrorism - Data Understanding](https://www.kaggle.com/thiagopanini/global-terrorism-data-understanding).

# # Terrorism Around the World

# In[ ]:


url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data'
world_geo = f'{url}/world-countries.json'
json_data = gpd.read_file(f'{url}/world-countries.json')


# In[ ]:


country_data = terr_data.groupby(by=['country_txt'], 
                                 as_index=False).count().sort_values(by='eventid', ascending=False).iloc[:, :2]
nkill_data = terr_data.groupby(by=['country_txt'], 
                                 as_index=False).sum().sort_values(by='eventid', 
                                                                   ascending=False).loc[:, ['country_txt', 'nkill']]
temp_global = json_data.merge(country_data, left_on='name', right_on='country_txt', how='left').fillna(0)
global_data = temp_global.merge(nkill_data, left_on='name', right_on='country_txt', how='left').fillna(0)

m = folium.Map(
    location=[0, 0], 
    zoom_start=1.50,
    tiles='openstreetmap'
)

folium.Choropleth(
    geo_data=json_data,
    name='Ataques Terroristas',
    data=country_data,
    columns=['country_txt', 'eventid'],
    key_on='feature.properties.name',
    fill_color='OrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    nan_fill_color='white',
    nan_fill_opacity=0.9,
    legend_name='Terrorism Recorded 1970 - 2017',
    popup_function='Teste'
).add_to(m)

Fullscreen(
    position='topright',
    title='Expand me',
    title_cancel='Exit me',
    force_separate_button=True
).add_to(m)

folium.GeoJson(
    global_data,
    style_function=style_function,
    highlight_function=highlight_function,
    tooltip=folium.GeoJsonTooltip(fields=['name', 'eventid', 'nkill'],
                                  aliases=['Country:', 'Incidents:', 'Victims'],
                                  labels=True,
                                  sticky=True)
).add_to(m)

m.save('terrorism_incidents.html')
m


# Well, here we can see clearly that Iraq is the country with the highest number of incidents recorded. The map also shows tooltips with the name of the country, number of incidents and total of victims recorded. Another thing that can be said looking at the map is that the Middle East and South Asia are the regions with the highes number of recorded attacks between 1970 and 2017.

# In[ ]:


heat_data = terr_data.groupby(by=['latitude', 'longitude'], 
                                 as_index=False).count().sort_values(by='eventid', ascending=False).iloc[:, :3]

m = folium.Map(
    location=[33.312805, 44.361488], 
    zoom_start=2.5, 
    tiles='Stamen Toner'
)

HeatMap(
    name='Mapa de Calor',
    data=heat_data,
    radius=10,
    max_zoom=13
).add_to(m)

Fullscreen(
    position='topright',
    title='Expand me',
    title_cancel='Exit me',
    force_separate_button=True
).add_to(m)

m.save('terrorism_density.html')
m


# With this Heatmap, we can see places with greater concentration of terrorism incidents from 1970 to 2017. To get more insights, let's see the"evolution" of terrorism year by year.

# In[ ]:


year_list = []
for year in terr_data['iyear'].sort_values().unique():
    data = terr_data.query('iyear == @year')
    data = data.groupby(by=['latitude', 'longitude'], 
                        as_index=False).count().sort_values(by='eventid', ascending=False).iloc[:, :3]
    year_list.append(data.values.tolist())

m = folium.Map(
    location=[0, 0], 
    zoom_start=2.0, 
    tiles='Stamen Toner'
)

HeatMapWithTime(
    name='Terrorism Heatmap',
    data=year_list,
    radius=9,
    index=list(terr_data['iyear'].sort_values().unique())
).add_to(m)

m


# Now we have a selection bar in the bottom of the map where we can select the terrorist records from a specific year between 1970 and 2017. It is important to cross this information with historical facts, wars and incidents. It's kind of interesting to see how the concentration of incidents starts at North America in the 70s and move to Europe and Middle East region as long as the time goes by.

# The most recent data we have is from 2017. Let's plot a global heatmap to see incidents among the months of 2017. Using the selection bar in the bottom, we can see the concentration of terrorism from january to december of 2017.

# In[ ]:


month_index = [
    'jan/2017',
    'feb/2017',
    'mar/2017',
    'apr/2017',
    'may/2017',
    'jun/2017',
    'jul/2017',
    'aug/2017',
    'sep/2017',
    'oct/2017',
    'nov/2017',
    'dec/2017'
]

month_list = []
for month in terr_data.query('iyear==2017')['imonth'].sort_values().unique():
    data = terr_data.query('imonth == @month')
    data = data.groupby(by=['latitude', 'longitude'], 
                        as_index=False).sum().sort_values(by='imonth', 
                                                          ascending=True).loc[:, ['latitude', 
                                                                                   'longitude', 
                                                                                   'nkill']]
    month_list.append(data.values.tolist())

m = folium.Map(
    location=[0, 0], 
    zoom_start=1.5, 
    tiles='Stamen Toner'
)

HeatMapWithTime(
    name='Mapa de Calor',
    data=month_list,
    radius=4,
    index=month_index
).add_to(m)

m


# # Countries and Terrorism

# Let's change the scenery and see the effects of terrorism in specific countries. First of all, let's take a look at the main countries affected by terrorism.

# In[ ]:


fig, ax = plt.subplots(figsize=(12, 6))
count_plot('region_txt', terr_data, ax=ax, colors='autumn')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title('Distribution of Attacks per Region (1970-2017)', size=15)
plt.show()


# As we have already seen in our first geographical plot, the highest concentration of incidentes recorded are from Middle East & North Africa. The region represents 27.8% of all records between 1970 and 2017. 
# 
# In the next plot, we will make a comparison of this historical data with 2017 data, but this time looking at the top 10 countries if highest nuber of terrorist incidents.

# In[ ]:


country_victims = terr_data.groupby(by='country_txt', as_index=False).sum().sort_values(by='nkill', 
                                                                      ascending=False).loc[:, ['country_txt', 
                                                                                               'nkill']]
country_victims = country_victims.iloc[:10, :]

terr_data_2017 = terr_data.query('iyear == 2017')
country_victims_2017 = terr_data_2017.groupby(by='country_txt', as_index=False).sum().sort_values(by='nkill', 
                                                                      ascending=False).loc[:, ['country_txt', 
                                                                                               'nkill']]
country_victims_2017 = country_victims_2017.iloc[:10, :]
country_victims_2017['country_txt'][16] = 'Central African Rep.'
country_victims_2017['country_txt'][22] = 'Dem. Rep. Congo'

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))

sns.barplot(x='nkill', y='country_txt', data=country_victims, ci=None,
                 palette='autumn', ax=axs[0])
sns.barplot(x='nkill', y='country_txt', data=country_victims_2017, ci=None,
                 palette='autumn', ax=axs[1])

format_spines(axs[0], right_border=False)
format_spines(axs[1], right_border=False)
axs[0].set_title('Top 10 - Total Victims by Country (1970-2017)')
axs[1].set_title('Top 10 - Total Victims by Country (2017)')
axs[0].set_ylabel('')
axs[1].set_ylabel('')

for p in axs[0].patches:
    width = p.get_width()
    axs[0].text(width-4000, p.get_y() + p.get_height() / 2. + 0.10, '{}'.format(int(width)), 
            ha="center", color='white')

for p in axs[1].patches:
    width = p.get_width()
    axs[1].text(width-300, p.get_y() + p.get_height() / 2. + 0.10, '{}'.format(int(width)), 
            ha="center", color='white')

plt.show()


# With the grap above we can see that Iraq and Afghanistan are the countries with most terrorism occurences in 2017 (and also in all period). Colombia, Peru and El Salvador appear in historica data but don't appear in 2017 data maybe because of past conflicts. Let's make a more specific analysis in some countries to see more details.
# 
# Now I present you the dashboard for country-terrorism relationship analysis. We will see details from Iraq, United States, Nigeria, Colombia and Egypt.

# In[ ]:


country_analysis(country_name='Iraq', data=terr_data, palette='summer', 
                 colors_plot2=['crimson', 'green'], color_lineplot='crimson')


# In[ ]:


country_analysis(country_name='United States of America', data=terr_data, palette='plasma', 
                 colors_plot2=['crimson', 'navy'], color_lineplot='navy')


# In[ ]:


country_analysis(country_name='Nigeria', data=terr_data, palette='summer', 
                 colors_plot2=['crimson', 'green'], color_lineplot='green')


# In[ ]:


country_analysis(country_name='Colombia', data=terr_data, palette='hot', 
                 colors_plot2=['crimson', 'gold'], color_lineplot='crimson')


# In[ ]:


country_analysis(country_name='Egypt', data=terr_data, palette='copper', 
                 colors_plot2=['crimson', 'brown'], color_lineplot='brown')


# # NLP on Incident Summary

# Well, in this session our goal is to apply some Natural Language Processing techniques to take a look at the words as a tool for understanding the Terrorism around the World. First of all, let's see some elements from the column _summary_

# In[ ]:


terr_data['summary'][:10]


# We can see two points by now:
# 
# * It is necessary to handle null data on this
# * Also, we have to eliminate the date information at the beginning of each instance

# In[ ]:


temp_corpus = terr_data['summary'].dropna()
corpus = temp_corpus.apply(lambda x: x.split(': ')[-1]).values
print(f'We have {len(corpus)} elements on the corpus\n\n')
print(f'Example 1: \n{corpus[1]}\n')
print(f'Example 2: \n{corpus[-1]}')


# ## Regular Expressions

# Now that we have already transformed the corpus on an array structure, let's apply some Regular Expressions to deal with non-desired elements. The analysis we will do in the following topics will cover:
# 
# * Search for Sites and Hiperlinks
# * Search for Numbers
# * Search for Special Characteres
# * Search for Additional Whitespaces

# ### Sites and Hiperlinks

# For this task, we will iterate all over the corpus applying the method `findall` using a specific Regular Expression created for searching sites and hiperlinks. Let's see what we can tell about it.

# In[ ]:


for c in corpus:
    urls = re.findall('(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', c)
    if len(urls) == 0:
        pass
    else:
        print(f'Description: {list(corpus).index(c)} - Links: {urls}')


# Well, it seems that only one attack summary have hiperlinks on it. Let's print it just to confirm.

# In[ ]:


# Example
corpus[6977]


# Ok, our method worked. So we will replace the links with the token "link".

# In[ ]:


# Replacing sites and hiperlinks
corpus_wo_hiperlinks = []
for c in corpus:
    c = re.sub(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', 'link', c)
    corpus_wo_hiperlinks.append(c)
corpus_wo_hiperlinks[6977]


# Good! Keep going.

# ### Numbers

# As we could see on some examples above, there are incidents descriptions with numbers. Here we will search for those text descriptions and replace the numbers with the token `number`.

# In[ ]:


# Example of description with number
corpus_wo_hiperlinks[399]


# In[ ]:


# Replacing numbers
corpus_wo_numbers = []
for c in corpus_wo_hiperlinks:
    c = re.sub('\d+(?:\.\d*(?:[eE]\d+))?', 'number', c)
    corpus_wo_numbers.append(c)
corpus_wo_numbers[399]


# We are one step closer to the goal.

# ### Special Characteres

# Again, as long as we are looking at incidents descriptions, we faced some special characteres that need to be replaced for further analysis. For this, let's apply another Regular Expresion to replace special characters with whitespace.

# In[ ]:


# Example with special characteres
corpus_wo_numbers[1113]


# In[ ]:


# Replacing special characteres with whitespace
corpus_text = []
for c in corpus_wo_numbers:
    c = re.sub(r'\W', ' ', c)
    corpus_text.append(c)
corpus_text[1113]


# ### Additional Whitespaces

# As we applied Regular Expressions (like this one on session 6.1.3), we generated some additional whitespaces. We can threat it with RegEx as well.

# In[ ]:


# Removing additional whitespaces
corpus_after_regex = []
for c in corpus_text:
    c = re.sub(r'\s+', ' ', c)
    corpus_after_regex.append(c)
    
corpus_after_regex[1113]


# Good! I think we are done here with RegEx

# ## Lower Case

# The next step we must on Natural Language Processing is putting all the tokens in lower case. We can do that with the method `apply` of Pandas DataFrame.

# In[ ]:


cleaned_corpus = pd.Series(corpus_after_regex).apply(lambda x: x.lower())
cleaned_corpus = list(cleaned_corpus.values)
cleaned_corpus[990]


# ## WordCloud

# In this session, we will generate a WordCloud for all descriptions.

# In[ ]:


# Genereating wordcloud
text = ' '.join(cleaned_corpus)
stopwords = set(STOPWORDS)

wordcloud = WordCloud(stopwords=stopwords, background_color="white", collocations=False).generate(text)

plt.figure(figsize=(15, 15))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# I hope you really enjoy this storytelling. Please upvote this kernel to keep me motivated to do even more!

# This is not the final version. There is much more to do:
# 
# * Create a WordCloud for attributes like corp1, target1 and motive;
# * Look for more Exploratory Data Analysis like:
#     * Incidents that lasted more than 24h (extended = 1);
#     * Major radical groups responsible for terrorist attacks (gname);
#     * Attacks with the highest number of terrorists (nperps);

# # References

# https://nbviewer.jupyter.org/gist/jtbaker/57a37a14b90feeab7c67a687c398142c?flush_cache=true
# 
# https://github.com/python-visualization/folium/issues/904
# 
# https://towardsdatascience.com/data-101s-spatial-visualizations-and-analysis-in-python-with-folium-39730da2adf
# 
# https://www.kaggle.com/rachan/how-to-folium-for-maps-heatmaps-time-analysis
# 
# https://python-visualization.github.io/folium/plugins.html

# In[ ]:




