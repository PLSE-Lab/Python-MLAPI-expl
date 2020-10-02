#!/usr/bin/env python
# coding: utf-8

# <h1>Summary</h1>
# This is a detailed analysis report about the current situation in the world with COVID-19. In this notebook I am going to try to explain alot of the terms that you hear a lot in the news. I am going to talk about flattening the curve, influction point and exponential growth to explain the current situation and mathematical models to predict the situation of the recent future. However, lets get started with some of the visualizations to set the ground running. 
# 

# In[ ]:



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# Here are some of the packages that were not installed. but we can install it just like that.  
# !pip install pandas_flavor

# here's several helpful packages to load in 
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
# import pandas_flavor as pf # for pipelining functions together. 
# scraping info from webpage
import requests
import lxml.html as lh
import missingno as msno # this package checks for missing values in data. 

# plotly
import plotly.express as px 
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will 
# list all files under the input directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Any results you write to the current directory are saved as output.

## Datasets
data = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
confirmed = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
recovered = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
deaths = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

## Do some cleaning and prepare the data as I like to do usually
data['Country/Region'].replace('Mainland China', 'China', inplace = True)
data['Country/Region'].replace('Ivory Coast', "Cote d'Ivoire", inplace = True)
data['Country/Region'].replace(' Azerbaijan','Azerbaijan', inplace = True)
data['Country/Region'].replace("('St. Martin',)", "St. Martin", inplace=True)
data['Country/Region'].replace("Cape Verde", "Cabo Verde", inplace=True)
data['Country/Region'].replace('Others', 'Cruise Ship', inplace = True)
data['Country/Region'].replace('Diamond Princess', 'Cruise Ship', inplace = True)
data['Country/Region'].replace('MS Zaandam', 'Cruise Ship', inplace = True)
data.rename(columns={'Country/Region':'Country'}, inplace = True)
us_counties = pd.read_csv('../input/us-counties-covid-19-dataset/us-counties.csv')


# In[ ]:


url = 'http://statisticstimes.com/geography/countries-by-continents.php'
#Create a handle, page, to handle the contents of the website
page = requests.get(url)
#Store the contents of the website under doc
doc = lh.fromstring(page.content)
#Parse data that are stored between <tr>..</tr> of HTML
tr_elements = doc.xpath('//tr')

#Create empty list
country_iso_continent = {}
for i in range(36, 285):
    row = []
    for t in tr_elements[i]:
        row.append(t.text_content())
    country_iso_continent[row[1]] = (row[2], row[6])
    

country_iso_continent['East Timor'] =('TLS', 'Asia')
country_iso_continent['Niger'] =('NER', 'Africa')
country_iso_continent["Gambia, The"] =('GMB', 'Africa')
country_iso_continent['Gambia'] = ('GMB', 'Africa')
country_iso_continent['The Gambia'] =('GMB', 'Africa')
country_iso_continent['Bahamas, The'] =('BHS', 'North America')
country_iso_continent['The Bahamas'] =('BHS', 'North America')
country_iso_continent['Bahamas'] =('BHS', 'North America')
country_iso_continent['Syria'] = ('SYR', 'Asia')
country_iso_continent['Tanzania'] = ('TZA','Africa')
country_iso_continent['Central African Republic']=('CAF', 'Africa')
country_iso_continent['Kosovo'] =('SRB', 'Europe')
country_iso_continent['Curacao']=('CUW', 'South America')
country_iso_continent['Venezuela'] =('VEN', 'South America')
country_iso_continent['Sudan']=('SDN','Africa')
country_iso_continent['Reunion'] = ('REU', 'Africa')
country_iso_continent['Channel Islands'] = ('USA', 'North America')
country_iso_continent['Congo (Kinshasa)'] =('COD', 'Africa')
country_iso_continent["Congo (Brazzaville)"] = ('COD', 'Africa')
country_iso_continent["Republic of the Congo"]=('COD', 'Africa')
country_iso_continent['Cayman Islands'] =('CYM', 'North America')
country_iso_continent['Bolivia'] = ('BOL', 'South America')
country_iso_continent['Holy See'] = ('VAT', 'Europe')
country_iso_continent['occupied Palestinian territory'] = ('PSE', 'Asia')
country_iso_continent['Brunei'] =('BRN', 'Asia')
country_iso_continent['St. Martin'] =('MAF', 'North America')
country_iso_continent['Republic of Ireland']=('IRL', 'Europe')
country_iso_continent['Moldova']=('MDA', 'Europe')
country_iso_continent['Vatican City'] = ('VAT', 'Europe')
country_iso_continent['West Bank and Gaza'] = ('PSE', 'Asia')
country_iso_continent['Palestine'] = ('PSE', 'Asia')
country_iso_continent['Faroe Islands'] = ('FRO', 'Europe')
country_iso_continent['Saint Barthelemy'] = ('BLM', 'North America')
country_iso_continent['United Arab Emirates'] = ('ARE', 'Asia')
country_iso_continent['Macau'] = ('MAC','Asia')
country_iso_continent['Taiwan'] = ('TWN', 'Asia')
country_iso_continent['US'] = ('USA', 'North America')
country_iso_continent['Philippines'] = ('PHL', 'Asia')
country_iso_continent['South Korea'] = ('PRK', 'Asia')
country_iso_continent['Vietnam'] = ('VNM', 'Asia')
country_iso_continent["Cote d'Ivoire"] = ('CIV', 'Africa')
country_iso_continent['North Macedonia'] = ('MKD', 'Europe')
country_iso_continent['UK'] = ('GBR', 'Europe')
country_iso_continent['Russia'] = ('RUS', 'Asia')
country_iso_continent['Others'] = ('TEMP', 'NA')
country_iso_continent['Netherlands'] = ('NLD', 'Europe')
country_iso_continent['Iran'] = ('IRN', 'Asia')
country_iso_continent['Hong Kong'] = ('HKG', 'Asia')
country_iso_continent['Macau'] = ('MAC', 'Asia')
country_iso_continent['United Arab Emirates'] = ('ARE', 'Asia')
country_iso_continent['Georgia'] = ('DEU', 'Asia')
country_iso_continent['Estonia'] = ('EST', 'Europe')
country_iso_continent['San Marino'] = ('SMR', 'Europe')
country_iso_continent['Azerbaijan'] = ('AZE', 'Asia')
country_iso_continent['Belarus'] = ('BLR', 'Europe')
country_iso_continent['North Ireland'] = ('GBR', 'Europe')
country_iso_continent['Luxembourg'] = ('LUX', 'Europe')
country_iso_continent['Lithuania'] = ('LTU', 'Europe')
country_iso_continent['Czech Republic'] = ('CZE','Europe')
country_iso_continent['Dominican Republic'] =('DOM', 'North America')
country_iso_continent['Laos'] = ('LAO', 'Asia')
country_iso_continent['Cruise Ship'] = ('NA', 'NA')
country_iso_continent['Burma'] = ('MMR', 'Asia')


def checker(df, string, dic_cont):
    """
    This function checks if there are any new countries or same countries with slightly different spelled names. 
    """
    temp = []

    for i in df.loc[:,string].unique():
        if i not in dic_cont:
            temp.append(i)

    return temp


## Set the data
viz = data[['Country', 'ObservationDate', 'Confirmed', 'Deaths', 'Recovered']]
viz = viz.groupby(['Country', 'ObservationDate']).sum().reset_index()
viz.sort_values('ObservationDate', ascending=True, inplace=True)
# checker(viz, 'Country/Region', country_iso_continent)

viz['iso_alpha'] = viz['Country'].apply(lambda x: country_iso_continent[x][0])
viz['Continent'] = viz['Country'].apply(lambda x: country_iso_continent[x][1])
# checker(viz, 'Country/Region', dic_cont)


# Let's take a look at how the COVID-19 disease is spreading around the world. 

# In[ ]:


fig = px.choropleth(
    viz, 
    locations="iso_alpha",
    labels='Country',
    color="Confirmed", 
    hover_name="Country", 
    animation_frame="ObservationDate", 
    range_color=[0,viz['Confirmed'].max()], 
#     height=800,
    color_continuous_scale='Reds',
)
fig.update_layout(
    paper_bgcolor='rgb(243, 243, 200)', 
    plot_bgcolor='rgb(243, 243, 200)',
    hoverlabel=dict(
        bgcolor="white", 
        font_size=16,
        font_family="Rockwell"
    )
)
config = {
    'displaylogo': False,
    'displayModeBar': False
}
fig.update_traces(hovertemplate=None)
fig.show(config = config)


# This animated map shows all the confirmed cases of COVID-19 spreads all over the world. 

# In[ ]:



## Visualize the data
fig = px.scatter(
    viz,
    x="Confirmed",
    y="Deaths", 
    animation_frame="ObservationDate", 
    animation_group="Country", 
#     height = 800,
    size="Confirmed", 
    color="Recovered", 
    hover_name="Country", 
    color_continuous_scale='Greens',
    title = 'COVID-19 Situation So Far',
    range_color=[0,viz.Recovered.max()],
    log_x=True,    
#     text ='iso_alpha',
#     marker = dict(width=2, color = 'black'),
#     line = dict(width = 4),
    size_max=100, 
    range_x=[100,viz.Confirmed.max()+100000], 
    range_y=[-9000,viz.Deaths.max()+10000])

fig.update_layout(
    paper_bgcolor='rgb(243, 243, 200)', 
    plot_bgcolor='rgb(243, 243, 200)',
    hoverlabel=dict(
        bgcolor="white", 
        font_size=16, 
        font_family="Rockwell"
    )
)
fig.update_xaxes(showspikes=False)
fig.update_yaxes(showspikes=False)
fig.show()


# This is a simple animated bubble chart showing total comfirmed cases for each country on the x-axis, deaths on the y-axis and total recovered using colors. The bubble also expends based on total confirmed cases giving you a slightly better understanding of the situation. There is a couple of things to take note here. 
# 
# <ul>
#     <li>The x-axis is in log scale, trying to cope with the exponential rate of the spread of the disease.</li>
#     <li>China pretty much stayed consistant around ~80K total confirmed cases while swiftly recovering in a massive rate and keeping the fatalities in check. </li>
#     <li>Meanwhile Italy, Spain and United States are rapidly passing by China in every possible counts except recovery rate.</li>
# </ul>
# 
# Let's look at a bar race chart to visualize it a bit more. 

# In[ ]:


get_ipython().run_cell_magic('HTML', '', '<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/1835578" data-url="https://flo.uri.sh/visualisation/1835578/embed"><script src="https://public.flourish.studio/resources/embed.js"></script></div>')


# There were times when I used to find satisfaction while looking at bar race charts. This chart is anything but satisfactory.

# In[ ]:


top_confirmend_country = viz[viz['ObservationDate'] == viz['ObservationDate'].max()].sort_values('Confirmed', ascending = False)['Country'].head(10)

fig = go.Figure()

for i in top_confirmend_country:
    temp = viz[viz['Country'] == i]
    fig.add_trace(
        go.Scatter(
            x=temp.ObservationDate, 
            y= temp.Confirmed, 
            name=i, 
            line = dict(width=5)
        )
    )
    
for i in top_confirmend_country:
    temp = viz[viz['Country'] == i]
    fig.add_trace(
        go.Scatter(
            x=temp.ObservationDate, 
            y= temp.Recovered, 
            name=i, 
            line = dict(width=5)
        )
    )
    
for i in top_confirmend_country:
    temp = viz[viz['Country'] == i]
    fig.add_trace(
        go.Scatter(
            x=temp.ObservationDate, 
            y= temp.Deaths, 
            name=i, 
            line = dict(width=5)
        )
    )
    


# # Add Annotations and Buttons
# high_annotations = [dict(x="01/25/2020",
#                          y=df.High.mean()x,
#                          xref="x", yref="y",
#                          text="High Average:<br> %.2f" % df.High.mean(),
#                          ax=0, ay=-40),
#                     dict(x=df.High.idxmax(),
#                          y=df.High.max(),
#                          xref="x", yref="y",
#                          text="High Max:<br> %.2f" % df.High.max(),
#                          ax=0, ay=-40)]
# low_annotations = [dict(x="2015-05-01",
#                         y=df.Low.mean(),
#                         xref="x", yref="y",
#                         text="Low Average:<br> %.2f" % df.Low.mean(),
#                         ax=-40, ay=40),
#                    dict(x=df.High.idxmin(),
#                         y=df.Low.min(),
#                         xref="x", yref="y",
#                         text="Low Min:<br> %.2f" % df.Low.min(),
#                         ax=0, ay=40)]

fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            active=0,
            x=0.57,
            y=1.2,
            buttons=list(
                [
                dict(label="Confirmed",
                     method="update",
                     args=[{"visible": [True]*10 + [False]*10 + [False]*10},
                           {"title": "Confirmed:",
                            "annotations": []}]),
                dict(label="Recovered",
                     method="update",
                     args=[{"visible": [False]*10 + [True]*10+[False]*10},
                           {"title": "Recovered:",
                            "annotations": []}]),
                dict(label="Deaths",
                     method="update",
                     args=[{"visible": [False]*10 + [False]*10 + [True]*10},
                           {"title": "Fatalities:",
                            "annotations": []}]),

            ]),
        )
    ])

# Set title
fig.update_layout(
    title_text="COVID-19 So Far",
    paper_bgcolor='rgb(243, 243, 200)', 
    plot_bgcolor='rgb(243, 243, 200)',
#     height = 800,
    xaxis_domain=[0.05, 1.0],
    hoverlabel=dict(
        bgcolor="white", 
        font_size=16, 
        font_family="Rockwell"
    )
)
fig.update_xaxes(showspikes=True)
fig.update_yaxes(showspikes=True)
fig.update_traces(hovertemplate="None")
fig.show()


# This is a chart showing total confirmed, recoverd and Deaths. You can use the tabs on top to switch between different charts. Let's dig a little deeper and 

# In[ ]:


latest_day = data[data.ObservationDate == data.ObservationDate.max()]
latest_day['World'] = 'World'
latest_day['Continent'] = latest_day['Country'].apply(lambda x: country_iso_continent[x][1])
latest_day['iso_alpha'] = latest_day['Country'].apply(lambda x: country_iso_continent[x][0])
latest_day['Province/State'].fillna('', inplace = True);


# ## Total Confirmed:

# In[ ]:


fig = px.treemap(latest_day, 
                 path=['World', 'Continent', 'Country', 'Province/State'], 
                 values='Confirmed',
                 color='Confirmed',
                 color_continuous_scale='Reds')

fig.update_layout(
    hoverlabel=dict(
        bgcolor="white", 
        font_size=16, 
        font_family="Rockwell"
    )

)
fig.update_traces(hovertemplate='<b>%{label}</b><br>Confirmed: %{color:.0f}')
fig.show()


# ## Total Recovered:

# In[ ]:


fig = px.treemap(latest_day, 
                 path=['World', 'Continent', 'Country', 'Province/State'],
                 values='Recovered',
                 color='Recovered',
                 branchvalues='total',
                 color_continuous_scale='Greens')
fig.update_layout(
    hoverlabel=dict(
        bgcolor="white", 
        font_size=16, 
        font_family="Rockwell"
    )
)
fig.update_traces(hovertemplate = '<b>%{label}</b><br>Recovered: %{color:.0f}')
fig.show()


# ## Total Fatalities:

# In[ ]:


fig = px.treemap(latest_day, 
                 path=['World', 'Continent', 'Country', 'Province/State'],
                 values='Deaths',
                 labels = 'Country',
                 color='Deaths', 
                 color_continuous_scale='Reds')
fig.update_layout(
    hoverlabel=dict(
        bgcolor="white", 
        font_size=16, 
        font_family="Rockwell"
    )
)
fig.update_traces(hovertemplate = '<b>%{label}</b><br>Deaths: %{color:.0f}')
fig.show()


# ## Fatality Rate: 

# In[ ]:


df = latest_day[['Continent', 'Country','Province/State', 'Deaths','Confirmed']]
levels = [ 'Country', 'Continent']
color_columns = ['Deaths','Confirmed']
value_column = 'Deaths'

def build_hierarchical_dataframe(df, levels, value_column, color_columns=None, middle_circle_name=None):
    """
    Build a hierarchy of levels for Sunburst or Treemap charts.

    Levels are given starting from the bottom to the top of the hierarchy,
    ie the last level corresponds to the root.
    """
    df_all_trees = pd.DataFrame(columns=['id', 'parent', 'value', 'color'])
    for i, level in enumerate(levels):
        df_tree = pd.DataFrame(columns=['id', 'parent', 'value', 'color'])
        dfg = df.groupby(levels[i:]).sum()
        dfg = dfg.reset_index()
        df_tree['id'] = dfg[level].copy()
        if i < len(levels) - 1:
            df_tree['parent'] = dfg[levels[i+1]].copy()
        else:
            df_tree['parent'] = middle_circle_name
        df_tree['value'] = dfg[value_column]
        df_tree['color'] = dfg[color_columns[0]] / dfg[color_columns[1]]
        df_all_trees = df_all_trees.append(df_tree, ignore_index=True)
    total = pd.Series(dict(id=middle_circle_name, parent='',
                              value=df[value_column].sum(),
                              color=df[color_columns[0]].sum() / df[color_columns[1]].sum()))
    df_all_trees = df_all_trees.append(total, ignore_index=True)
    return df_all_trees


df_all_trees = build_hierarchical_dataframe(df, levels, value_column, color_columns, 'World')
average_score = df['Deaths'].sum() / df['Confirmed'].sum()

fig = make_subplots(1, 2, specs=[[{"type": "domain"}, {"type": "domain"}]],)

fig.add_trace(go.Sunburst(
    labels=df_all_trees['id'],
    parents=df_all_trees['parent'],
    values=df_all_trees['value'],
    branchvalues='total',
    marker=dict(
        colors=df_all_trees['color'],
        colorscale='OrRd',
        cmid=average_score),
    hovertemplate='<b>%{label} </b> <br> Fatalities: %{value}<br> Fatality rate: %{color:.2f}',
    name=''
    ), 1, 1)

fig.add_trace(go.Sunburst(
    labels=df_all_trees['id'],
    parents=df_all_trees['parent'],
    values=df_all_trees['value'],
    branchvalues='total',
    marker=dict(
        colors=df_all_trees['color'],
        colorscale='OrRd',
        cmid=average_score),
    hovertemplate='<b>%{label} </b> <br> Fatalities: %{value}<br> Fatality rate: %{color:.2f}',
    name='',
    maxdepth=2
    ), 1, 2)

fig.update_layout(
    margin=dict(t=10, b=10, r=10, l=10),
    hoverlabel=dict(
        bgcolor="white", 
        font_size=16, 
        font_family="Rockwell"
    )
)
fig.show()


# ## Recovery Rate:

# In[ ]:


df = latest_day[['Continent', 'Country','Province/State', 'Recovered','Confirmed']]

levels = [ 'Country', 'Continent']## levels should be placed in hierarchical order(example: [state, country, continent], [grandson, son, father, grandfather])
color_columns = ['Recovered','Confirmed']
value_column = 'Confirmed'

df_all_trees = build_hierarchical_dataframe(df, levels, value_column, color_columns, 'World')
average_score = df['Recovered'].sum() / df['Confirmed'].sum()

fig = make_subplots(1, 2, specs=[[{"type": "domain"}, {"type": "domain"}]],)

fig.add_trace(go.Sunburst(
    labels=df_all_trees['id'],
    parents=df_all_trees['parent'],
    values=df_all_trees['value'],
    branchvalues='total',
    marker=dict(
        colors=df_all_trees['color'],
        colorscale='RdBu',
        cmid=average_score),
    hovertemplate='<b>%{label} </b> <br> Confirmed: %{value}<br> Recovery Rate: %{color:.2f}',
    name=''
    ), 1, 1)

fig.add_trace(go.Sunburst(
    labels=df_all_trees['id'],
    parents=df_all_trees['parent'],
    values=df_all_trees['value'],
    branchvalues='total',
    marker=dict(
        colors=df_all_trees['color'],
        colorscale='RdBu',
        cmid=average_score),
    hovertemplate='<b>%{label} </b> <br> Confirmed: %{value}<br> Recovery Rate: %{color:.2f}', 
    name='',
    maxdepth = 2
    ), 1, 2)

fig.update_layout(
    margin=dict(t=10, b=10, r=10, l=10), 
    hoverlabel=dict(
        bgcolor="white", 
        font_size=16, 
        font_family="Rockwell"
    )
)
fig.show()


# The **color** in this chart represents the **recovered rate** of all the confirmed cases. As you can see China has a recovered rate of 94% giving it a dark blue color while US has a recovery rate of 0.05% which is below the average recovery rate so far; hence the reddish color. In the next few days hopefully this rate will improve. Let's look at the mortality rate now

# ## Recovery Rate in North America:

# In[ ]:


df = latest_day[['Continent', 'Country','Province/State', 'Recovered','Confirmed']]
df = df[df['Continent'] =='North America']
levels = [ 'Province/State','Country'] ## levels should be placed in hierarchical order(example: [state, country, continent], [grandson, son, father, grandfather])
color_columns = ['Recovered','Confirmed']
value_column = 'Confirmed'

df_all_trees = build_hierarchical_dataframe(df, levels, value_column, color_columns, 'North America')
average_score = df['Recovered'].sum() / df['Confirmed'].sum()

fig = make_subplots(2, 1, specs=[[{"type": "domain"}], [{"type": "domain"}]],)


fig.add_trace(go.Sunburst(
    labels=df_all_trees['id'],
    parents=df_all_trees['parent'],
    values=df_all_trees['value'],
    branchvalues='total',
    marker=dict(
        colors=df_all_trees['color'],
        colorscale='RdBu',
        cmid=average_score),
    hovertemplate='<b>%{label} </b> <br> Confirmed: %{value}<br> Recoverey Rate: %{color:.4f}',
    name=''
    ), 1, 1)


fig.add_trace(go.Treemap(
    labels=df_all_trees['id'],
    parents=df_all_trees['parent'],
    values=df_all_trees['value'],
    branchvalues='total',
    marker=dict(
        colors=df_all_trees['color'],
        colorscale='RdBu',
        cmid=average_score),
    hovertemplate='<b>%{label} </b> <br> Confirmed: %{value}<br> Recovery Rate: %{color:.2f}',
    name=''
    ), 2,1)

fig.update_layout(
    margin=dict(t=10, b=10, r=10, l=10), 
    height = 1200,
    hoverlabel=dict(
        bgcolor="white", 
        font_size=16, 
        font_family="Rockwell"
    )
)
fig.show()


# # Total Confirmed (United States and their Counties):

# In[ ]:


df = us_counties[us_counties.date == us_counties.date.max()]
df['United States'] = 'United States'

fig = px.treemap(df, 
                 path=['United States', 'state', 'county'],
                 values='cases',
                 labels = 'state',
                 color='cases', 
                 color_continuous_scale='Reds')
fig.update_layout(
    hoverlabel=dict(
        bgcolor="white", 
        font_size=16, 
        font_family="Rockwell"
    )
)
fig.update_traces(hovertemplate = '<b>%{label}</b><br>Total Cases: %{color:.0f}')
fig.show();


# In[ ]:


# df = latest_day[['Continent', 'Country','Province/State', 'Recovered','Confirmed', 'Deaths']]
# levels = ['Country', 'Continent'] ## levels should be placed in hierarchical order(example: [state, country, continent], [grandson, son, father, grandfather])
# color_columns = ['Recovered','Confirmed']
# value_column = 'Confirmed'

# df_all_trees = build_hierarchical_dataframe(df, levels, value_column, color_columns, 'World')
# average_score = df['Recovered'].sum() / df['Confirmed'].sum()


# fig = go.Figure()
# fig.add_trace(go.Treemap(
#     labels=df_all_trees['id'],
#     parents=df_all_trees['parent'],
#     values=df_all_trees['value'],
#     branchvalues='total',
#     marker=dict(
#         colors=df_all_trees['color'],
#         colorscale='RdBu',
#         cmid=average_score),
#     hovertemplate='<b>%{label} </b> <br> Confirmed: %{value}<br> Recovery Rate: %{color:.2f}',
#     name=''
#     ))
# fig.show()


# The following part is a work in progress. 
# 
# Resources:
# * How to write better codes
#     * [Six steps to more professional data science code](https://www.kaggle.com/rtatman/six-steps-to-more-professional-data-science-code)
#     * [Creating a Good Analytics Report](https://www.kaggle.com/jpmiller/creating-a-good-analytics-report)
#     * [Code Smell](https://en.wikipedia.org/wiki/Code_smell)
#     * [Python style guides](https://www.python.org/dev/peps/pep-0008/)
# 
