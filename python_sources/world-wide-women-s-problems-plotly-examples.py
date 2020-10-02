#!/usr/bin/env python
# coding: utf-8

# **WORLD WIDE WOMEN'S PROBLEMS**
# 
# In this kernel I will investigate women's problems using Human Freedom Index dataset. 
# 
# Human Freedom Index dataset is about:
# "*A central purpose of The Human Freedom Index is to paint a broad but reasonably accurate picture of the extent of overall freedom in the world. A larger purpose is to more carefully explore what we mean by freedom and to better understand its relationship to any number of other social and economic phenomena.*"
# 
# From the dataset I will focus on women related datas related to countries and regions;
#     * Missing women,
#     * Inheritance rights,
#     * Women's security,
#     * Overall security and safety,
#     * Freedom of movement,
#     * Freedom of religion,
#     * Rule of Law,
#     * Economic freedom.
# 
# I know this kernel is just a scracth on the surface of the women's problem. And is not enough the show the importance of the subject. 
# Please add your comments and write your opinions and feel free to like it.
# 
# I hope you will enjoy my analysis.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
# word cloud library
from wordcloud import WordCloud

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/hfi_cc_2018.csv')


# In[ ]:


data.head(5)


# **1. ANALYSE SECURITY AND FREEDOM OF MOVEMENT OF WOMEN IN MIDDLE EASTERN & NORTH AFRICAN COUNTRIES IN 2016**

# I start my analysis by investigating freedom of women's movement in Middle Eastern and North African countries because of the unstable conditions of the region.
# 
# First of all we need to filter the data set for the year 2016 and for the Middle East and North African regions.

# In[ ]:


# Filtering data to observe Middle Eastern and North Africaan Countries in the most recent year (2016) :
filter01 = data.year == 2016
filter02 = data.region == 'Middle East & North Africa'
data01 = data[filter01 & filter02]
data01.sample(5)


# Now we can draw our scatter graphic. 
# First trace will show us Middle East and North African women's security level in 2016. First trace of the graphic will be red.
# Second trace will show us the women's freedom of movement in the same regions in 2016. Second trace will be green.

# In[ ]:


# First trace will be security of women in 2016: 
trace01 = go.Scatter(
                     x = data01['ISO_code'],
                     y = data01['pf_ss_women'],
                     mode = 'lines+markers',
                     name = 'Security of Women',
                     marker = dict(color = 'rgba(150, 20, 20, 0.5)'),
                     text= data01['countries'])
# Second trace will show us freedom of women movement in 2016:
trace02 = go.Scatter(
                     x = data01['ISO_code'],
                     y = data01['pf_movement_women'],
                     mode = 'lines+markers',
                     name = 'Freedom of Movement',
                     marker = dict(color = 'rgba(20, 150, 20, 0.5)'),
                     text= data01['countries'])
datanew = [trace01, trace02]
layoutnew = dict(title = 'Security of Women and Freedom of Women Movement Comparison in Middle East and Northern Africa',
                 xaxis = dict(title = 'Countries', ticklen = 3, zeroline = False))
fig = dict(data=datanew, layout=layoutnew)
py.iplot(fig)


# From the graphic above one can see some country's women has less freedom of movement, even the women's security level is high in the country. 
# This might be related by different conditions like; religion, culture, women's role in the economy or the law system of the country.

# **2. ANALYSE SECURITY OF WOMEN IN LATIN AMERICA AND THE CARIBBEAN BETWEEN 2008, 2013 AND 2016**

# My second analysis will focus on women's security in Latin American and Caribbean Countries. 
# 
# I will use three plots for 2008, 2013 and 2016 years in order to see security level difference in the region.

# In[ ]:


# Filtering dataset to three different years (2008, 2013,2016)
data2008 = data[data.year == 2008]
data2008 = data2008[data2008.region == 'Latin America & the Caribbean']

data2013 = data[data.year == 2013]
data2013 = data2013[data2013.region == 'Latin America & the Caribbean']

data2016 = data[data.year == 2016]
data2016 = data2016[data2016.region == 'Latin America & the Caribbean']

# First trace is security of women in 2008:
trace2008 = go.Scatter(
                        x = data2008['ISO_code'],
                        y = data2008['pf_ss_women'],
                        mode = 'markers',
                        name = 'Women Security in 2008',
                        marker = dict(color='rgba(0,0,255,0.5)'),
                        text = data2008['countries'])

# Second trace is security of women in 2013:
trace2013 = go.Scatter(
                        x = data2013['ISO_code'],
                        y = data2013['pf_ss_women'],
                        mode = 'markers',
                        name = 'Women Security in 2013',
                        marker = dict(color='rgba(0,255,0,0.5)'),
                        text = data2013['countries'])

# Third trace is security of women in 2016:
trace2016 = go.Scatter(
                        x = data2016['ISO_code'],
                        y = data2016['pf_ss_women'],
                        mode = 'markers',
                        name = 'Women Security in 2016',
                        marker = dict(color='rgba(255,0,0,0.5)'),
                        text = data2016['countries'])
datanew = [trace2008, trace2013, trace2016]
layoutnew = dict(title = 'Security of Women in Latin America & the Caribbean Between 2008, 2013, 2016',
                 xaxis = dict(title = 'Countries', ticklen = 3, zeroline = False))
fig = dict(data=datanew, layout=layoutnew)
py.iplot(fig)


# From the graph above one can see in some countries women's security level decreases with years. This analysis can give a clue if that country is in an unstable condition or transition in their culturel/political/economic etc states.

# **3. INVESTIGATE RELATIONSHIP BETWEEN MISSING WOMEN AND SECURITY LEVEL IN MIDDLE EASTERN AND NORTH AFRICAN COUNTRIES**

# Because of conflicts, wars and immigration issues in Middle Eastern and North African countries, I wantted to investigate missing women in the region.
# 
# In order to do that, I compared missing women and security level datas of the countries.

# In[ ]:


# Missing women in Middle East and North Africa:
trace01 = go.Bar(
                 x = data01['ISO_code'],
                 y = data01['pf_ss_women_missing'],
                 name = 'Missing Women',
                 marker = dict(color='rgba(130, 35, 45, 0.5)',
                              line = dict(color='rgba(0,0,0)', width=0.5)),
                 text = data01.countries)
# Security Level of the Country:
trace02 = go.Bar(
                 x = data01['ISO_code'],
                 y = data01['pf_ss'],
                 name = 'Security of the Country',
                 marker = dict(color='rgba(45, 35, 145, 0.5)',
                              line = dict(color='rgba(0,0,0)', width=0.5)),
                 text = data01.countries)
datanew = [trace01, trace02]
layoutnew = go.Layout(barmode='group', title='Missing Women Comparison')
fig = go.Figure(data=datanew, layout=layoutnew)
py.iplot(fig)


# From the graph above one can easly see that, there is a missing women problem even the country's security level is high (like Israel, Lebanon, Morrocco, Turkey).

# **4. ANALYSE SECURITY OF WOMEN AMONG MIDDLE EASTERN AND NORTH AFRICAN COUNTRIES**

# One can see an overall comparison of the women's security level in Middle Eastern and North African countries.

# In[ ]:


# Analysing security of women among Middle East and North African Countries:
fig = {
        'data': [ 
             {
                'values' : data01['pf_ss_women'],
                'labels' : data01['countries'],
                'domain' : {'x': [0, 1]},
                'name' : 'Security for Women',
                'hoverinfo' : 'label+percent+name',
                'hole' : 0.3,
                'type' : 'pie'
              },
             ],
         'layout' : {
                     'title' : 'Security of Women Among Middle East and N.Africa',
                     'annotations' : [
                                        { 'font' : {'size' : 20},
                                          'showarrow' : False,
                                          'text' : ' ',
                                          'x' : 0.20,
                                          'y' : 1
                                         },
                                      ]    
                     }
        }
py.iplot(fig)


# **5. COMPARE RELATIONSHIP BETWEEN SECURITY OF WOMEN & RULE OF LAW & FREEDOM OF RELIGION IN EUROPEAN COUNTIRES**

# In this analysis I focues on European countries by comparing security of women, rule of law and freedom of religion datas.

# In[ ]:


# Filtering data to see only European Countries informations in 2016:
dataE = data[data.year == 2016]
dataEU = dataE[(dataE.region == 'Western Europe') + (dataE.region == 'Eastern Europe')]
dataEU = dataEU.dropna(subset=['pf_ss_women_missing'])
dataEU.sample(5)


# In[ ]:


colorEU = [float(each) for each in dataEU['pf_ss_women']]
sizeEU = [float(each) for each in dataEU['pf_ss_women']]

data05 = [
    {
        'x' : dataEU['pf_rol'],
        'y' : dataEU['pf_religion'],
        'mode' : 'markers',
        'marker' : { 'color': colorEU, 'size' : sizeEU, 'showscale' : True},
        'text' : dataEU.countries
    } 
]
layout05 = dict(title = 'Relationship Between Security of Women & Rule of Law & Freedom of Religion', 
                xaxis = dict(title = 'Rule of Law Index', ticklen = 4, zeroline = False),
                yaxis = dict(title = 'Freedom of Religion Index', ticklen = 4, zeroline = False))
fig = dict(data=data05, layout=layout05)
py.iplot(fig)


# One can see there is a positive relationship with countries' rule of law, freedom of religion and security of women levels.

# In[ ]:


colorEU = [float(each) for each in dataEU['pf_ss_women']]

trace3D = go.Scatter3d(
    x = dataEU['pf_rol'],
    y = dataEU['pf_religion'],
    z = dataEU['pf_ss_women'],
    mode = 'markers',
    marker = dict(
        size = 7,
        color=colorEU
    )
)

data06 = [trace3D]
layout06 = go.Layout(
    title = 'Relationship Between Security of Women & Rule of Law & Freedom of Religion',
)
fig = go.Figure(data=data06, layout=layout06)
py.iplot(fig)


# **6. COMPARE THE SECURITY OF WOMEN BETWEEN 2008 AND 2016 IN THE SOUTH ASIAN COUNTRIES**

# I wanted to show securtiy of women in the South Asian Countries by comparing 2008 and 2016 datas.

# In[ ]:


# Lets visualize the subject above with a histogram:
data2008 = data[(data.year == 2008) + (data.region == 'South Asia')]
data2008 = data2008.dropna(subset=['pf_ss_women'])
data2016 = data[(data.year == 2016) + (data.region == 'South Asia')]
data2016 = data2016.dropna(subset=['pf_ss_women'])

trace01 = go.Histogram(
                        x = data2008['pf_ss_women'],
                        opacity = 0.65,
                        name = 'Security of Women in 2008',
                        marker = dict(color='rgba(150,237,32,0.5)'),
                        text= data2008['countries'])
trace02 = go.Histogram(
                        x = data2016['pf_ss_women'],
                        opacity = 0.65,
                        name = 'Security of Women in 2016',
                        marker = dict(color='rgba(80,22,236,0.5)'))
dataHist = [trace01, trace02]
layoutHist = go.Layout(
                        title = 'Comparison of Security of Women in the Years 2008 and 2016',
                        xaxis = dict(title='Countries'))
fig = go.Figure(data = dataHist, layout = layoutHist)
py.iplot(fig)


# From the graph above one can see in women's security decreases in some countries (Philippines, Sri Lanka, Thailand, Vietnam, Cambodia).

# **7. SHOWING THE MOST MENTIONED REGIONS IN OUR DATASET (WORDCLOUD LIBRARY EXAMPLE)**

# In order to see most mentioned regions in our dataset we can use wordcloud library. In this graphic below I didn't filter the dataset related to women issues. 

# In[ ]:


region = data.region
plt.subplots(figsize = (8,8))
wordcloud = WordCloud(
                          background_color='white',
                          width=512,
                          height=384
                         ).generate(" ".join(region))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')
plt.show()


# **8. COMPARE ECONOMIC FREEDOM RANKS OF EASTERN AND WESTERN EUROPEAN COUNTRIES**

# In this analysis I compared economic freedom ranks of the Western and Eastern European countries.
# 
# Even this subject is not directly related to women' problems, economic problems are always the biggest source in a nation.

# In[ ]:


dataEF = data.filter(['year', 'ISO_code', 'countries', 'region', 'ef_score', 'ef_rank', 'hf_score', 'hf_rank'])
EastEU = dataEF[dataEF.region == 'Eastern Europe']
WestEU = dataEF[dataEF.region == 'Western Europe']


# Now we can make a graph consist of two traces. First trace will show us the economic freedom rank of Eastern European Countries and the second one will show us economic freedom rank of Western European Countries.

# In[ ]:


trace1 = go.Box(
                y = WestEU.ef_rank,
                name = 'Economic Freedom Rank of Western European Countries',
                marker = dict(color= 'rgb(78, 54, 23)')
                )
trace2 = go.Box(
                y = EastEU.ef_rank,
                name = 'Economic Freedom Rank of Eastern European Countries',
                marker = dict(color = 'rgb(20, 24, 79)')
                )
efdata = [trace1, trace2]
py.iplot(efdata)


# One can see economic freedom rank is higher and in a norrower trend in Western European Countries while the Eastern European Countries there is a broader variaty.

# **9. ANALYSE THE RELATIONSHIP BETWEEN SECURITY OF WOMEN, SECURITY LEVEL OF THE COUNTRY AND ECONOMIC FREEDOM IN OCEANIC COUNTRIES**

# In this section I filtered the human freedom index data to show only Oceania region's security of women, security of people and economic freedom rank columns.

# In[ ]:


dataocean = data[data.region == 'Oceania']
dataocean = dataocean.filter(['pf_ss_women', 'pf_ss', 'ef_rank'])
dataocean['index'] = np.arange(1, len(dataocean)+1)


# In[ ]:


import plotly.figure_factory as ff
fig = ff.create_scatterplotmatrix(dataocean, diag='box', index = 'index', colormap='Portland', colormap_type='cat', height=700, width=700)
py.iplot(fig)


# We can see from the graphic above that if the ''economic freedom rank'' is low, ''security of women'' and ''security of human'' levels are low as well.

# **10. SECURTIY OF WOMEN'S WORLD WIDE DISTRIBUTION IN 2016**
# 
# To conclude my analysis I compared world countries according to their women's security level in a world map.

# In[ ]:


dataset = data[data.year == 2016]
dataset = dataset.loc[:, ['year', 'countries', 'pf_ss_women']]
dataset.tail()


# In[ ]:


ssw = [dict(
    type = 'choropleth',
    locations = dataset['countries'],
    locationmode = 'country names',
    z = dataset['pf_ss_women'],
    text = dataset['countries'],
    colorscale = [[0,"rgb(5, 10, 172)"],[2,"rgb(40, 60, 190)"],[4,"rgb(70, 100, 245)"],\
            [6,"rgb(90, 120, 245)"],[8,"rgb(106, 137, 247)"],[10,"rgb(220, 220, 220)"]],
    autocolorscale = False,
    reversescale = True,
    marker = dict(line = dict(color = 'rgb(150,150,150)',width = 0.5 )),
    colorbar = dict(autotick=False, tickprefix= '', title='Security of Women'),
)]

layout = dict(
    title = 'Securtiy of Women in 2016',
    geo = dict(showframe=False, showcoastlines=True, projection=dict(type='Mercator'))
)

fig = dict(data=ssw, layout=layout)
py.iplot( fig, validate=False, filename='security-of-women')


# Hope you liked my analysis. If you like please vote my kernel and feel free to add your comments.
# 
# Melih.
