#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# *The essence of all religions is one* - Gandhi
# 
# Religion can be a very controversial subject so let me lay down some ground-rules right off the bat and state that this notebook will try utmost not to get into philosophical/political aspects of religion nor will it be one which pits one versus the other. What the proceeding analysis will do is to take a good hard look at the data available to us, seek to uncover trends and piece together a story worth five decades and over seven continents.
# 
# Lets Go

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
from wordcloud import WordCloud, STOPWORDS
from scipy.misc import imread
import base64


# Now the dataset is split into 3 distinct parts (or .csv files to be exact). The first part contains data on a National level (i.e data specific to a country), the second part is on a Regional level (i.e data split by region such as the Middle East or Asia etc). Lastly, the third csv file is to do with data on a Global level. Therefore let us proceed by loading the data in.

# In[ ]:


# Load in the .csv files as three separate dataframes
Global = pd.read_csv('../input/global.csv') # Put to caps or else name clash
national = pd.read_csv('../input/national.csv')
regional = pd.read_csv('../input/regional.csv')


# # 1. REGIONAL LEVEL ANALYSIS
# 
# Starting off first with a high-level analysis of looking at the data on a regional level, we will later work our way down to a more granular level. So let's see what the *Regional* dataframe has in store for us. 

# In[ ]:


# Print the top 3 rows
regional.head(3)


# In[ ]:


print(regional['region'].unique())


# Since there are quite a handful of religions and their denominations (evinced from the columns), my approach will therefore be to first narrow down our choice by ignoring all denominations for now and only focusing on the data that relates to the parent religion. This is achieved by looking at columns demarcated by *_all*. The dates range from 1945 to 2010 with an interval of 5 years from one to another and therefore this nicely fits into our aim for five decades worth of analysis. 
# 
# This EDA will be split into what I think should be intuitive Religious groupings, where we first explore the Abrahamic religions, then the Eastern Asian and Indian ones.

# ### Abrahamic Religions
# 
# Plotting the five-decade regional trend for the Abrahamic faiths ( Christianity, Islam and Judaism) using stacked area plots, we get the following charts

# In[ ]:


#fig = plt.figure(figsize=(8, 5))
fig, axes = plt.subplots(nrows=1, ncols=3)
colormap = plt.cm.inferno_r
# fig = plt.figure(figsize=(20, 10))
# plt.subplot(121)
christianity_year = regional.groupby(['year','region']).christianity_all.sum()
christianity_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False,ax= axes[0],figsize=(11.5,4.5) , legend=False)
axes[0].set_title('Christianity Adherents',y=1.08,size=10)

# plt.subplot(122)
islam_year = regional.groupby(['year','region']).islam_all.sum()
islam_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False, ax= axes[1], legend= False)
axes[1].set_title('Islam Adherents',y=1.08,size=10)

judaism_year = regional.groupby(['year','region']).judaism_all.sum()
judaism_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False, ax= axes[2])
axes[2].legend(bbox_to_anchor=(-1.7, -0.3, 1.8, 0.1), loc=10,prop={'size':12},
           ncol=5, mode="expand", borderaxespad=0.)
axes[2].set_title('Judaism Adherents',y=1.08,size=10)

plt.tight_layout()
plt.show()


# In[ ]:





# **Takeaway from the plots**
# 
# One can immediately notice the upward and increasing trend in the numbers of adherents for both Christianity and Islam. With regards to Christianity, the numbers have gone from strength to strength (over the past 50 years) and grown for countries in the Western Hemisphere, Asia and Africa. However there does not seem to be much increase or decrease for Europe. 
# 
# Turning our attention to Islam, the largest contributor to the increasing numbers would be due to Asia, evinced from her more than two-fold jump just under a 20 year period (1980 to 2000). Both the Middle East and Africa also have had increases in their Islamic adherents albeit at a less rapid pace. 
# 
# Finally, Judaism has not had the drastic increase or swelling in numbers compared to the other two religions. In fact there seems to be a peak and trough in the numbers over the past 50 years, whereby the Middle East has contributed to the greatest increase while Europe on the other hand has declined. 
# 
# Finally I would like to add that one has to temper this analysis with many caveats 

# ### Indian Religions
# 
# Finally, let us plot the data for some of the major Indian religions that we have on hand, namely Hinduism, Sikhism and Jainism.

# In[ ]:


#fig = plt.figure(figsize=(8, 5))
fig, axes = plt.subplots(nrows=1, ncols=3)
colormap = plt.cm.inferno
# fig = plt.figure(figsize=(20, 10))
# plt.subplot(121)
christianity_year = regional[regional['region'] != 'Asia'].groupby(['year','region']).hinduism_all.sum()
christianity_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False,ax= axes[0],figsize=(13.5,6.5) , legend=False)
axes[0].set_title('Hindusim Adherents',y=1.08,size=12)

# plt.subplot(122)
islam_year = regional[regional['region'] != 'Asia'].groupby(['year','region']).sikhism_all.sum()
islam_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False, ax= axes[1], legend= False)
axes[1].set_title('Sikhism Adherents',y=1.08,size=12)

judaism_year = regional[regional['region'] != 'Asia'].groupby(['year','region']).jainism_all.sum()
judaism_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False, ax= axes[2])
axes[2].legend(bbox_to_anchor=(-1.7, -0.3, 2, 0.1), loc=10,prop={'size':12},
           ncol=5, mode="expand", borderaxespad=0.)
axes[2].set_title('Jainism Adherents',y=1.08,size=12)

plt.tight_layout()
plt.show()


# In[ ]:





# ### East Asian Religions
# 
# Now let's group some of the Eastern Asian religions that we have on hand with us, namely Buddhism, Taoism (mainly in South-East Asia, China) and Shinto (Japanese indigenous religion). Although to be totally accurate, Buddhism (some may think of it as a Philosophy/Way of Life instead) has roots enshrined in the Indian subcontinent. 

# In[ ]:


#fig = plt.figure(figsize=(8, 5))
fig, axes = plt.subplots(nrows=1, ncols=3)
colormap = plt.cm.Purples
# fig = plt.figure(figsize=(20, 10))
# plt.subplot(121)
christianity_year = regional[regional['region'] != 'Asia'].groupby(['year','region']).buddhism_all.sum()
christianity_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False,ax= axes[0],figsize=(13.5,6.5) , legend=False)
axes[0].set_title('Buddhist Adherents',y=1.08,size=12)

# plt.subplot(122)
islam_year = regional[regional['region'] != 'Asia'].groupby(['year','region']).taoism_all.sum()
islam_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False, ax= axes[1], legend= False)
axes[1].set_title('Taoist Adherents',y=1.08,size=12)

judaism_year = regional[regional['region'] != 'Asia'].groupby(['year','region']).shinto_all.sum()
judaism_year.unstack().plot(kind='area',stacked=True,  colormap= colormap, grid=False, ax= axes[2])
axes[2].legend(bbox_to_anchor=(-1.7, -0.3, 2, 0.1), loc=10,prop={'size':12},
           ncol=5, mode="expand", borderaxespad=0.)
axes[2].set_title('Shinto Adherents',y=1.08,size=12)

plt.tight_layout()
plt.show()


# **Takeaway from the plots**
# 
# The first and most obvious thing is that (should come as no surprise) the large majority of adherents hail from the Asian continent. Once again, all three religions are experiencing a general upward and increasing trend in their number of adherents. However what stands out is that these Eastern Asian religions have not experienced the sort of increase or spread of geographic presence outside the Asian region as compared to their Abrahamic counterparts. As observed from the stacked barplots, there are relatively very few adherents anywhere else apart from Asia.

# ### Indian Religions
# 
# Finally, let us plot the data for some of the major Indian religions that we have on hand, namely Hinduism, Sikhism and Jainism.

# **Takeaway from the plots**
# 
# Much like the East Asian religions, these three Indian religions are also experiencing a general increase in their numbers across the five decades. 

# # 2. NATIONAL LEVEL ANALYSIS
# 
# Having analysed the data at a regional level, let us now inspect the data more closely on a country level. 

# In[ ]:


national.head(3)


# ### 3D Globe Plots with Plotly
# 
# Bringing back those 3D Globe plots you may have seen in my older kernels via the Plotly library, let us take a look at the three Abrahamic religions for the most recent year of 2010 and the spread of their respective adherents globally.

# In[ ]:


# Create a dataframe with only the 2010 data
national_2010 = national[national['year'] == 2010]
# Extract only the parent religion with the "_all" and ignoring their denominations for now
religion_list = []
for col in national_2010.columns:
    if '_all' in col:
        religion_list.append(col)
metricscale1=[[0.0,"rgb(20, 40, 190)"],[0.05,"rgb(40, 60, 190)"],[0.25,"rgb(70, 100, 245)"],[0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]]
data = [ dict(
        type = 'choropleth',
        autocolorscale = False,
        colorscale = 'Viridis',
        reversescale = True,
        showscale = True,
        locations = national_2010['state'].values,
        z = national_2010['christianity_all'].values,
        locationmode = 'country names',
        text = national_2010['state'].values,
        marker = dict(
            line = dict(color = 'rgb(200,200,200)', width = 0.5)),
            colorbar = dict(autotick = True, tickprefix = '', 
            title = 'Number of Christian Adherents')
            )
       ]

layout = dict(
    title = 'Christian Adherents in 2010',
    geo = dict(
        showframe = True,
        showocean = True,
        oceancolor = 'rgb(0,0,0)',
        #oceancolor = 'rgb(222,243,246)',
        projection = dict(
        type = 'orthographic',
            rotation = dict(
                    lon = 60,
                    lat = 10),
        ),
        lonaxis =  dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
            ),
        lataxis = dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
                )
            ),
        )
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='worldmap2010')

data2 = [ dict(
        type = 'choropleth',
        autocolorscale = False,
        colorscale = 'Viridis',
        reversescale = True,
        showscale = True,
        locations = national_2010['state'].values,
        z = national_2010['islam_all'].values,
        locationmode = 'country names',
        text = national_2010['state'].values,
        marker = dict(
            line = dict(color = 'rgb(200,200,200)', width = 0.5)),
            colorbar = dict(autotick = True, tickprefix = '', 
            title = 'Number of Islamic Adherents')
            )
       ]

layout2 = dict(
    title = 'Islamic Adherents in the Year 2010',
    geo = dict(
        showframe = True,
        showocean = True,
        oceancolor = 'rgb(28,10,16)',
        #oceancolor = 'rgb(222,243,246)',
        projection = dict(
        type = 'orthographic',
            rotation = dict(
                    lon = 60,
                    lat = 10),
        ),
        lonaxis =  dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
            ),
        lataxis = dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
                )
            ),
        )
fig = dict(data=data2, layout=layout2)
py.iplot(fig, validate=False, filename='worldmap2010') 

data3 = [ dict(
        type = 'choropleth',
        autocolorscale = False,
        colorscale = 'Viridis',
        reversescale = True,
        showscale = True,
        locations = national_2010['state'].values,
        z = national_2010['judaism_all'].values,
        locationmode = 'country names',
        text = national_2010['state'].values,
        marker = dict(
            line = dict(color = 'rgb(200,200,200)', width = 0.5)),
            colorbar = dict(autotick = True, tickprefix = '', 
            title = 'Number of Judaism Adherents')
            )
       ]

layout3 = dict(
    title = 'Judaism Adherents in the Year 2010',
    geo = dict(
        showframe = True,
        showocean = True,
        oceancolor = 'rgb(28,10,16)',
        #oceancolor = 'rgb(222,243,246)',
        projection = dict(
        type = 'orthographic',
            rotation = dict(
                    lon = 60,
                    lat = 10),
        ),
        lonaxis =  dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
            ),
        lataxis = dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
                )
            ),
        )
fig = dict(data=data3, layout=layout3)
py.iplot(fig, validate=False, filename='worldmap2010') 


# *PLEASE CLICK AND SCROLL THE ABOVE. THESE GLOBE PLOTS ARE INTERACTIVE. DOUBLE-CLICK ON THE GLOBE IF YOU WANT TO GET BACK TO THE ORIGINAL VIEW*

# It is interesting to observe the geographic distribution of how the different religious adherents were spread out over the world. For Christianity, the greatest numbers of adherents in absolute terms can be found in the United States of America and Brazil as evinced by the darkest colours on the plots. For Islam, the greatest numbers can be found in countries of Indonesia, India, Pakistan and Iran. Finally as noted earlier with regards to regional trend on Judaism, it seems that most Judaism adherents are based in the United States of America, but are pretty sparsely located elsewhere.

# ### Mercator Plots with Plotly
# 
# The Mercator plot, which is one of the most famous types of map plots is essentially a projection with parallel spacings calculated to maintain conformality of the various countries in the globe. Here, we plot the numbers of adherents via a Mercator projection for some of the Asian religions and see what we get.

# In[ ]:


metricscale1=[[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],[0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]]
# Mercator plots for the Buddhism
data = [ dict(
        type = 'choropleth',
        locations = national_2010['code'],
        z = national_2010['buddhism_all'],
        text = national_2010['code'].unique(),
        colorscale = metricscale1,
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(200,200,200)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'Number of Buddhist Adherents'),
      ) ]

layout = dict(
    title = 'Spread of Buddhist adherents in 2010',
    geo = dict(
        scope = 'asia',
        showframe = False,
        showocean = True,
        oceancolor = 'rgb(0,0,50)',
#         oceancolor = 'rgb(232,243,246)',
        #oceancolor = ' rgb(28,107,160)',
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='d3-world-map' )

# Mercator plots for Hinduism
data1 = [ dict(
        type = 'choropleth',
        locations = national_2010['code'],
        z = national_2010['hinduism_all'],
        text = national_2010['code'].unique(),
        colorscale = metricscale1,
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(200,200,200)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'Number of Hinduism Adherents'),
      ) ]

layout1 = dict(
    title = 'Spread of Hinduism adherents in 2010',
    geo = dict(
        scope = 'asia',
        showframe = False,
        showocean = True,
        oceancolor = 'rgb(0,0,50)',
#         oceancolor = 'rgb(232,243,246)',
        #oceancolor = ' rgb(28,107,160)',
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data1, layout=layout1 )
py.iplot( fig, validate=False, filename='world-map' )

# Mercator plots for Shinto
data2 = [ dict(
        type = 'choropleth',
        locations = national_2010['code'],
        z = national_2010['shinto_all'],
        text = national_2010['code'].unique(),
        colorscale = metricscale1,
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(200,200,200)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'Number of Shinto Adherents'),
      ) ]

layout2 = dict(
    title = 'Spread of Shinto adherents in 2010',
    geo = dict(
        scope = 'asia',
        showframe = False,
        showocean = True,
        oceancolor = 'rgb(0,0,50)',
#         oceancolor = 'rgb(232,243,246)',
        #oceancolor = ' rgb(28,107,160)',
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data2, layout=layout2 )
py.iplot( fig, validate=False, filename='world-map2' )


# As evinced from the Mercator plots, these Indian and East Asian religions are still very much bounded and asian-centric. So it may be revealing to observe the movement and trends in the religions of the different countries. First, since the list of countries is pretty numerous, let us segregate these countries as such to tidy things up

# In[ ]:


# Although I know that Thailand, Cambodia, Lao, Vietnam, Malaysia, Singapore, Philippines, Indonesia, Brunei
# are South-East Asian countries to be exact, I decided to group these together
East_asian_countries = ['China', 'Mongolia', 'Taiwan', 'North Korea',
       'South Korea', 'Japan','Thailand', 'Cambodia',
       'Laos', 'Vietnam',  'Malaysia', 'Singapore',
       'Brunei', 'Philippines', 'Indonesia']

South_asian_countries = ['India', 'Bhutan', 'Pakistan', 'Bangladesh',
       'Sri Lanka', 'Nepal']

East_european_countries = [
    'Poland', 'Czechoslovakia', 'Czech Republic', 'Slovakia','Malta', 'Albania', 'Montenegro', 'Macedonia',
       'Croatia', 'Yugoslavia', 'Bosnia and Herzegovina', 'Kosovo',
       'Slovenia', 'Bulgaria', 'Moldova', 'Romania','Estonia', 'Latvia', 'Lithuania', 'Ukraine', 'Belarus',
       'Armenia', 'Georgia',
]

West_european_countries = [
    'United Kingdom', 'Ireland', 'Netherlands', 'Belgium', 'Luxembourg',
       'France', 'Liechtenstein', 'Switzerland', 'Spain', 'Portugal', 'Germany','Greece', 'Italy'
]

Africa = ['Mali', 'Senegal',
       'Benin', 'Mauritania', 'Niger', 'Ivory Coast', 'Guinea',
       'Burkina Faso', 'Liberia', 'Sierra Leone', 'Ghana', 'Togo',
       'Cameroon', 'Nigeria', 'Gabon', 'Central African Republic', 'Chad',
       'Congo', 'Democratic Republic of the Congo', 'Uganda', 'Kenya',
       'Tanzania', 'Burundi', 'Rwanda', 'Somalia']

South_america = ['Peru', 'Brazil',
       'Bolivia', 'Paraguay', 'Chile', 'Argentina', 'Uruguay','Colombia',
       'Venezuela']
#European_countries = ['United Kingdom', 'Ireland', 'Netherlands', 'Belgium', 'Luxembourg',
#       'France', 'Monaco', 'Liechtenstein', 'Switzerland', 'Spain',
#       'Andorra', 'Portugal', 'Germany', 'Poland', 'Austria', 'Hungary',
#       'Czechoslovakia', 'Czech Republic', 'Slovakia', 'Italy',
#       'San Marino', 'Malta', 'Albania', 'Montenegro', 'Macedonia',
#       'Croatia', 'Yugoslavia', 'Bosnia and Herzegovina', 'Kosovo',
#       'Slovenia', 'Greece', 'Cyprus', 'Bulgaria', 'Moldova', 'Romania',
#       'Russia', 'Estonia', 'Latvia', 'Lithuania', 'Ukraine', 'Belarus',
#       'Armenia', 'Georgia',  'Finland', 'Sweden', 'Norway',
#       'Denmark', 'Iceland',]


# ### Stacked Area plots 
# 
# Having now segregated our individual countries into more manageable groups so that are area plots do not get too cluttered, we shall now start delving into religious trends on a more granular level. For brevity, I shall only selectively pick interesting or telling trends.

# **Christianity Trends**
# 
# As alluded earlier where we looked at the Christianity numbers on a regional level, we expect to see a general increase in adherent numbers. Let us see which countries contribute most 

# In[ ]:


plt.figure(figsize=(10, 10))


# East Asian Numbers
christianity_year = national[ national['state'].isin(East_asian_countries) ].groupby(['year','state']).christianity_all.sum()
christianity_year.unstack().plot(kind='area',stacked=True,  colormap= 'cubehelix', grid=False,figsize=(10,8))
plt.title('Christanity in East Asia')
# Place a legend above this subplot, expanding itself to
# fully use the given bounding box.
plt.gca().legend_.remove()
plt.legend(bbox_to_anchor=(-0.2, -0.5, 1.4, .5), loc=5,
            ncol=4, mode="expand", borderaxespad=0.)
plt.ylabel('Number of Christian adherents')
plt.show()

christianity_year = national[ national['state'].isin(Africa) ].groupby(['year','state']).christianity_all.sum()
christianity_year.unstack().plot(kind='area',stacked=True,  colormap= 'cubehelix_r', grid=False,figsize=(10,8))
plt.title('Christanity in Africa', size=12)
# Place a legend above this subplot, expanding itself to
# fully use the given bounding box.
plt.gca().legend_.remove()
plt.legend(bbox_to_anchor=(-0.2, -0.5, 1.4, .5), loc=5,
            ncol=4, mode="expand", borderaxespad=0.)
plt.ylabel('Number of Christian adherents')
plt.show()


# South American Numbers
christianity_year = national[ national['state'].isin(South_america) ].groupby(['year','state']).christianity_all.sum()
christianity_year.unstack().plot(kind='area',stacked=True,  colormap= 'cubehelix_r', grid=False,figsize=(10,8))
plt.title('Christanity in South America')
# Place a legend above this subplot, expanding itself to
# fully use the given bounding box.
plt.gca().legend_.remove()
plt.legend(bbox_to_anchor=(-0.2, -0.5, 1.4, .5), loc=5,
            ncol=4, mode="expand", borderaxespad=0.)
plt.ylabel('Number of Christian adherents')
plt.show()


christianity_year = national[ national['state'].isin(West_european_countries) ].groupby(['year','state']).christianity_all.sum()
christianity_year.unstack().plot(kind='area',stacked=True,  colormap= 'cubehelix', grid=False,figsize=(10,8))
plt.title('Christanity in Western Europe')
# Place a legend above this subplot, expanding itself to
# fully use the given bounding box.
plt.gca().legend_.remove()
plt.legend(bbox_to_anchor=(-0.2, -0.5, 1.4, .5), loc=5,
            ncol=4, mode="expand", borderaxespad=0.)
plt.ylabel('Number of Christian adherents')
plt.show()


# **First Impressions from the Plots**
# 
# It seems that Christianity is spreading and gaining adherents across most countries over the past five decades. For the Asian region, the Philippines is one of the largest contributors of Christian adherents while for the South American region, this can be attributed to the country of Brazil. African countries are also showing remarkable growth in the numbers of adherents, especially in Congo and the Niger. 
# 
# Western Europe on the other hand does not have that obvious a trend. It seems that their numbers of Christian adherents are experiencing undulations of sorts. What is also interesting is that the change in adherent numbers from these Western European countries seem to be in phase with one another, dropping and increasing in sync.

# **Islam Trends**
# 
# Let us turn our attention to the religion of Islam and see what insights we can glean from plotting the data

# In[ ]:


islam_year = national[ national['state'].isin(West_european_countries) ].groupby(['year','state']).islam_all.sum()
islam_year.unstack().plot(kind='area',stacked=True,  colormap= 'PuBuGn', grid=False,figsize=(10,8))
plt.title('Islam in Western Europe')
# Place a legend above this subplot, expanding itself to
# fully use the given bounding box.
plt.gca().legend_.remove()
plt.legend(bbox_to_anchor=(-0.2, -0.5, 1.4, .5), loc=5,
            ncol=4, mode="expand", borderaxespad=0.)
plt.ylabel('Number of Islam adherents')
plt.show()

islam_year = national[ national['state'].isin(Africa) ].groupby(['year','state']).islam_all.sum()
islam_year.unstack().plot(kind='area',stacked=True,  colormap= 'PuBuGn_r', grid=False,figsize=(10,8))
plt.title('Islam in Africa')
# Place a legend above this subplot, expanding itself to
# fully use the given bounding box.
plt.gca().legend_.remove()
plt.legend(bbox_to_anchor=(-0.2, -0.5, 1.4, .5), loc=5,
            ncol=4, mode="expand", borderaxespad=0.)
plt.ylabel('Number of Islam adherents')
plt.show()

islam_year = national[ national['state'].isin(South_america) ].groupby(['year','state']).islam_all.sum()
islam_year.unstack().plot(kind='area',stacked=True,  colormap= 'PuBuGn', grid=False,figsize=(10,8))
plt.title('Islam in South America')
# Place a legend above this subplot, expanding itself to
# fully use the given bounding box.
plt.gca().legend_.remove()
plt.legend(bbox_to_anchor=(-0.2, -0.5, 1.4, .5), loc=5,
            ncol=4, mode="expand", borderaxespad=0.)
plt.ylabel('Number of Islam adherents')
plt.show()

islam_year = national[ national['state'].isin(East_asian_countries) ].groupby(['year','state']).islam_all.sum()
islam_year.unstack().plot(kind='area',stacked=True,  colormap= 'PuBuGn', grid=False,figsize=(10,8))
plt.title('Islam in Eastern Asia')
# Place a legend above this subplot, expanding itself to
# fully use the given bounding box.
plt.gca().legend_.remove()
plt.legend(bbox_to_anchor=(-0.2, -0.5, 1.4, .5), loc=5,
            ncol=4, mode="expand", borderaxespad=0.)
plt.ylabel('Number of Islam adherents')
plt.show()


# **First Impressions from the Plots**
# 
# Once again all plots convey the same story, one of an increasing number of Islam adherents as well as an increasing global reach. First off, we can observe that the greatest increase in Western Europe can be attributed to both France and Germany evinced by the biggest proportions that these countries occupy in the stacked area plots. The trend in Africa is also pretty remarkable in the sense that the increase in Islam seems to be going at almost the same rate throughout the five decades given the near homogeneous shape of the plots. In South America, Argentina experienced the greatest increase in adherents as the numbers shot up during the mid-90's.

# **Buddhism**

# In[ ]:


Budd_year = national[ national['state'].isin(South_asian_countries) ].groupby(['year','state']).buddhism_all.sum()
Budd_year.unstack().plot(kind='area',stacked=True,  colormap= 'viridis_r', grid=False,figsize=(10,8))
plt.title('Buddhism in South Asian countries')
# Place a legend above this subplot, expanding itself to
# fully use the given bounding box.
plt.gca().legend_.remove()
plt.legend(bbox_to_anchor=(-0.2, -0.7, 1.4, .5), loc=5,
            ncol=4, mode="expand", borderaxespad=0.)
plt.ylabel('Number of Buddhist adherents')
plt.show()

Budd_year = national[ national['state'].isin(West_european_countries) ].groupby(['year','state']).buddhism_all.sum()
Budd_year.unstack().plot(kind='area',stacked=True,  colormap= 'viridis_r', grid=False,figsize=(10,8))
plt.title('Buddhism in Western Europe')
# Place a legend above this subplot, expanding itself to
# fully use the given bounding box.
plt.gca().legend_.remove()
plt.legend(bbox_to_anchor=(-0.2, -0.7, 1.4, .5), loc=5,
            ncol=4, mode="expand", borderaxespad=0.)
plt.ylabel('Number of Buddhist adherents')
plt.show()

Budd_year = national[ national['state'].isin(Africa) ].groupby(['year','state']).buddhism_all.sum()
Budd_year.unstack().plot(kind='area',stacked=True,  colormap= 'viridis', grid=False,figsize=(10,8))
plt.title('Buddhism in Africa')
# Place a legend above this subplot, expanding itself to
# fully use the given bounding box.
plt.gca().legend_.remove()
plt.legend(bbox_to_anchor=(-0.2, -0.7, 1.4, .5), loc=5,
            ncol=4, mode="expand", borderaxespad=0.)
plt.ylabel('Number of Buddhist adherents')
plt.show()

Budd_year = national[ national['state'].isin(South_america) ].groupby(['year','state']).buddhism_all.sum()
Budd_year.unstack().plot(kind='area',stacked=True,  colormap= 'viridis_r', grid=False,figsize=(10,8))
plt.title('Buddhism in South America')
# Place a legend above this subplot, expanding itself to
# fully use the given bounding box.
plt.gca().legend_.remove()
plt.legend(bbox_to_anchor=(-0.2, -0.7, 1.4, .5), loc=5,
            ncol=4, mode="expand", borderaxespad=0.)
plt.ylabel('Number of Buddhist adherents')
plt.show()


# **Hinduism**

# In[ ]:


# Stacked area plot of Hinduism in West Europe
Hin_year = national[ national['state'].isin(West_european_countries) ].groupby(['year','state']).hinduism_all.sum()
Hin_year.unstack().plot(kind='area',stacked=True,  colormap= 'gist_earth', grid=False,figsize=(10,8))
plt.title('Hinduism in West Europe')
# Place a legend above this subplot, expanding itself to
# fully use the given bounding box.
plt.gca().legend_.remove()
plt.legend(bbox_to_anchor=(-0.2, -0.7, 1.4, .5), loc=5,
            ncol=4, mode="expand", borderaxespad=0.)
plt.ylabel('Number of Hindu adherents')
plt.show()

Hin_year = national[ national['state'].isin(East_european_countries) ].groupby(['year','state']).hinduism_all.sum()
Hin_year.unstack().plot(kind='area',stacked=True,  colormap= 'gist_earth', grid=False,figsize=(10,8))
plt.title('Hinduism in East Europe')
# Place a legend above this subplot, expanding itself to
# fully use the given bounding box.
plt.gca().legend_.remove()
plt.legend(bbox_to_anchor=(-0.2, -0.7, 1.4, .5), loc=5,
            ncol=4, mode="expand", borderaxespad=0.)
plt.ylabel('Number of Hindu adherents')
plt.show()

Hin_year = national[ national['state'].isin(South_asian_countries) ].groupby(['year','state']).hinduism_all.sum()
Hin_year.unstack().plot(kind='area',stacked=True,  colormap= 'gist_earth', grid=False,figsize=(10,8))
plt.title('Hinduism in South Asia')
# Place a legend above this subplot, expanding itself to
# fully use the given bounding box.
plt.gca().legend_.remove()
plt.legend(bbox_to_anchor=(-0.2, -0.7, 1.4, .5), loc=5,
            ncol=4, mode="expand", borderaxespad=0.)
plt.ylabel('Number of Hindu adherents')
plt.show()

Hin_year = national[ national['state'].isin(East_asian_countries) ].groupby(['year','state']).hinduism_all.sum()
Hin_year.unstack().plot(kind='area',stacked=True,  colormap= 'gist_earth_r', grid=False,figsize=(10,8))
plt.title('Hinduism in East Asia')
# Place a legend above this subplot, expanding itself to
# fully use the given bounding box.
plt.gca().legend_.remove()
plt.legend(bbox_to_anchor=(-0.2, -0.7, 1.4, .5), loc=5,
            ncol=4, mode="expand", borderaxespad=0.)
plt.ylabel('Number of Hindu adherents')
plt.show()


# # Other Interesting Analysis
# 
# Having focused our attentions on the major religions so far from the sections above, let us now turn our attention to some of the other lesser known ones present in the data and see what kind of story the data has in store for us.
# 
# ### A.) Confucianism - a story of North and South Korea
# 
# For those not too familiar with Confucianism , it is also sometimes described as a *tradition, philosophy* or *humanistic/rationalistic* religion (taken from Wikipedia) and was inspired and developed from the teachings of the Chinese sage Confucius who lived around 500 BC.

# In[ ]:


# Prevalence of Confucianism in Asian countries
christianity_year = national[ national['state'].isin(East_asian_countries) ].groupby(['year','state']).confucianism_all.sum()
christianity_year.unstack().plot(kind='area',stacked=True,  colormap= 'hot_r', grid=False,figsize=(7,7))

#plt.figure(figsize=(10,5))
plt.title('Stacked area plot of the trend in Confucianism over the years', y=1.09)
# Place a legend above this subplot, expanding itself to
# fully use the given bounding box.
plt.gca().legend_.remove()
plt.legend(bbox_to_anchor=(-0.5, -0.7, 1.8, .5), loc=5,
            ncol=4, mode="expand", borderaxespad=0.)
plt.ylabel('Number of Confucianism adherents')
plt.show()


# **Takeaway from the plots**
# 
# Interestingly, one can see that Confucianism was very prevalent in South Korea from 1950 all the way to 1980. From there on, there was a steep decline in the numbers through two decades till 2000, where there seemed to be a mini revival in the numbers. On the other hand, the data shows that there a relatively small number of adherents in North Korea till the late 90's, where the numbers of adherents experienced a steep increase in the numbers. 

# ### B.) Animism
# 
# Animism is the belief that natural objects such as rivers, rocks and flora possess souls and are alive and come with feelings and intentions.
# 
# **Rise in China, Fall in India**

# In[ ]:


# Plotting Animism for both East and South Asian countries
christianity_year = national[ national['state'].isin(East_asian_countries) ].groupby(['year','state']).animism_all.sum()
christianity_year.unstack().plot(kind='area',stacked=True,  colormap= 'hot', grid=False,figsize=(10,8))
plt.title('Stacked Area plot of Animism over the years for Central-East-SE Asian countries', y =1.09)
# Place a legend above this subplot, expanding itself to
# fully use the given bounding box.
plt.gca().legend_.remove()
plt.legend(bbox_to_anchor=(-0.2, -0.5, 1.4, .5), loc=5,
            ncol=4, mode="expand", borderaxespad=0.)
plt.ylabel('Number of Animism adherents')
plt.show()

christianity_year = national[ national['state'].isin(South_asian_countries) ].groupby(['year','state']).animism_all.sum()
christianity_year.unstack().plot(kind='area',stacked=True,  colormap= 'hot', grid=False,figsize=(10,8))
plt.title('Stacked Area plot of Animism over the years for South Asian countries', y=1.09)
# Place a legend above this subplot, expanding itself to
# fully use the given bounding box.
plt.gca().legend_.remove()
plt.legend(bbox_to_anchor=(-0.2, -0.5, 1.4, .5), loc=5,
            ncol=4, mode="expand", borderaxespad=0.)
plt.ylabel('Number of Animism adherents')
plt.show()


# From the plots, we see that the number of adherents to Animism has more than doubled over the five decades in China making up for the general upward trend in numbers across the whole Central/East Asian/SEA region. On the other hand, India which had a nearly threefold increase in the numbers of Animism adherents from 1950 to 1990 experienced a very steep decline in the numbers when it came to the turn of the new millennium.

# # Conclusion
# 
# As one can see, there is an increasing trend in the number of religious adherents over the five decades. Be it for the Abrahamic or Indian or Asian religions.

# # *WORK IN PROGRESS. NEED TO SLEEP NOW WILL CONTINUE LATER*

# In[ ]:


'''
print ('this is my second time kaggle code')
'''

