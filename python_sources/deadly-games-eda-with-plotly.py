#!/usr/bin/env python
# coding: utf-8

# # INTRODUCTION
# 
# This is an EDA performed on a dataset named [Sacred Games](https://www.kaggle.com/zusmani/sacred-games) about *political attacks* in Pakistan from year **1947**-**2018**. I have tried to analyze civilian and politician's casualty rate based on different factors.
# 
# **Warning: Plotly graphs may take some time to render**
# 
# <br>Content:
# 
# 1. [Attack Locations on Map](#1)
# 1. [Attack Rate w.r.t Other Factors](#2)
# 1. [Customized Functions](#3)
# 1. [Analysis on Province](#4)
# 1. [Analysis on City](#5)
# 1. [Analysis on Day and Time](#6)
# 1. [Analysis on Location](#7)
# 1. [Analysis on Attack Type](#8)
# 1. [Analysis on Party](#9)
# 1. [Trend Over The Years](#10)
# 1. [Location Word Cloud](#11)
# 1. [Observations](#12)

# ## Importing Packages

# In[ ]:


#Numpy/Pandas
import numpy as np
import pandas as pd

# plotly
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
import plotly.offline as off

#For HTML Rendering
from IPython.core.display import display, HTML

#matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#folium for Map
import folium
from folium import plugins


# word cloud
from wordcloud import WordCloud

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Setting Matplotlib Params
plt.rcParams["font.size"] = 12
plt.rcParams["font.weight"] = 'bold'
plt.rcParams["figure.figsize"] = (15,10)

#Setting Seaborn Style
sns.set_style("whitegrid")


# ## Loading Data

# In[ ]:


df = pd.read_csv('../input/Attacks on Political Leaders in Pakistan.csv', encoding='latin1')


# ## Data Cleaning

# In[ ]:


#Fixing misspelled column name
lat_col = df.columns.values
lat_col[11] = 'Longitude'
df.columns = lat_col
df.columns


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


#Dropping Irrelevant columns
df.drop('S#', axis=1, inplace=True)


# In[ ]:


#Replace NULL values in Location Category with UNKNOWN
df['Location Category'].fillna('UNKNOWN', inplace=True)

#Giving same values but with different text, the same text
df.loc[df['Location Category'] == 'Details Missing', 'Location Category'] = 'UNKNOWN'
df.loc[df['Province'] == 'Fata', 'Province'] = 'FATA'
df.loc[df['City'] == 'ATTOCK', 'City'] = 'Attock'


# In[ ]:


#Convert Categorical variables to Category type
columns = ['Target Status', 'Day', 'Day Type', 'Time', 'City', 'Location Category',
          'Province', 'Target Category', 'Space (Open/Closed)', 'Party']
df[columns] = df[columns].astype('category')


# In[ ]:


#Get Month and Year of Attack
df['month'] = pd.DatetimeIndex(df['Date']).month
df['year'] = pd.DatetimeIndex(df['Date']).year

#correct wrong interpretation of data
df['year']=df['year'].replace(2051, 1951)
df['year']=df['year'].replace(2058, 1958)


# In[ ]:


df.describe()


# In[ ]:


print(df['Target Category'].value_counts())
print(df['Target Status'].value_counts())
print(df['Space (Open/Closed)'].value_counts())


# ## Feature Engineering

# In[ ]:


df['marker_popup'] = ''
for index, row in df.iterrows():
    df.loc[index, 'marker_popup'] = df.loc[index,'City'].strip() + '(' + str(df.loc[index,'Date']) + '  |  <b>Killed</b>: ' + str(df.loc[index,'Killed']) + '  |  Injured: ' + str(df.loc[index,'Injured'])  + ')'


# In[ ]:


df['Target Attack'] = (df['Target Category'] == 'Target').astype(int)
df['Suicide Attack'] = (df['Target Category'] != 'Target').astype(int)
df['Open Space'] = (df['Space (Open/Closed)'] == 'Open').astype(int)
df['Closed Space'] = (df['Space (Open/Closed)'] != 'Open').astype(int)
df['Politician Killed'] = (df['Target Status'] == 'Killed').astype(int)
df['Politician Escaped'] = (df['Target Status'] != 'Killed').astype(int)


# In[ ]:


month_lookup = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May',
            6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

df['month'] = df['month'].apply(lambda x: month_lookup[x])


# ## EDA

# <a id="1"></a> <br>
# ### Attack Locations on Map
# 
# ***"Red"* markers show locations where politicians got killed and *"Green"* markers show locations where politicians escaped the attack. Click on markers for attack details**

# In[ ]:


pk_map = folium.Map(location=[30.3753, 69.3451],
                   zoom_start=5)
# mark each station as a point
for index, row in df.iterrows():
    folium.Marker([df.loc[index,'Latitude'], df.loc[index,'Longitude']],
                  icon=folium.Icon(color= 'red' if df.loc[index, 'Target Status'] == 'Killed' else 'green'),
                  popup=df.loc[index,'marker_popup']).add_to(pk_map)
pk_map


# <a id="2"></a> <br>
# ### Attack Rate w.r.t Other Factors
# **Let's check out on which day does most attacks happen?**

# In[ ]:


sns.countplot(y='Day', data = df)


# **At What time does most Attacks happen?**

# In[ ]:


sns.countplot(y='Time', data = df)


# **How about City? Which City suffered most attack?**

# In[ ]:


sns.countplot(y='City', data = df)


# **We could already see attack rate on Province from Map but let's still check it out**

# In[ ]:


sns.countplot(x='Province', data = df)


# <a id="a1"></a> <br>

# In[ ]:


sns.countplot(y='Location Category', data = df)


# In[ ]:


sns.countplot(y='Party', data = df)


# <a id="3"></a> <br>
# ### Customized Functions
# 
# **DRAW BAR CHART FUNCTION**

# In[ ]:


def draw_barchart(dataframe, x_col, y_cols, chart_title='', x_title='', y_title='', agg_func = 'sum', tick_angle=0):
    
    if dataframe is None:
        raise ValueError('dataframe is not Provided')
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError('dataframe should be of type Pandas Dataframe')
    if type(x_col) is not str:
        raise ValueError('x_col should be of string type')
    if not isinstance(y_cols,(list,)):
        raise ValueError('x_col should be passed as a list')
    
    Province = dataframe[x_col]
    data = []

    for i in range(len(y_cols)):
        data.append(
            dict(
            type = 'bar',
            x = Province,
            y = dataframe[y_cols[i]],
            name = y_cols[i],
            transforms = [
                dict(
                    type = 'aggregate',
                    groups = Province,
                    aggregations = [dict(
                        target = 'y', func = agg_func, enabled = True)]
                )
            ]
            )
        )


    if tick_angle > 0:
        layout = dict(
            title = '<b>' + chart_title + '</b>',
            xaxis = dict(title = x_col if len(x_title) == 0 else x_title, tickangle=tick_angle),
            yaxis = dict(title = y_title),
            barmode = 'relative'
        )
    else:
        layout = dict(
            title = '<b>' + chart_title + '</b>',
            xaxis = dict(title = x_col if len(x_title) == 0 else x_title),
            yaxis = dict(title = y_title),
            barmode = 'relative'
    )

    off.iplot({
        'data': data,
        'layout': layout
    }, validate = False)


# **DRAW BUBBLE CHART FUNCTION**

# In[ ]:


def draw_bubblechart(dataframe, x_col, y_cols):
    
    if dataframe is None:
        raise ValueError('dataframe is not Provided')
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError('dataframe should be of type Pandas Dataframe')
    if type(x_col) is not str:
        raise ValueError('x_col should be of string datatype')
    if not isinstance(y_cols,(list,)):
        raise ValueError('x_col should be passed as a list')
    
    ycol_size = len(y_cols)
    data = []
    updatemenu_list = []
    
    for i in range(ycol_size):
        visible = [True if j == i else False for j in range(ycol_size)]
        data.append(
            dict(
                type = 'scatter',
                mode = 'markers',
                x = dataframe[x_col],
                y = dataframe[y_cols[i]],
                text = dataframe[y_cols[i]],
                hoverinfo = 'text',
                name = y_cols[i],
                opacity = 0.8,
                marker = dict(
                    size = dataframe[y_cols[i]],
                    sizemode = 'area'
                ),
                transforms = [
                    dict(
                        type = 'aggregate',
                        groups = dataframe[x_col],
                        aggregations = [dict(
                            target = 'y', func = 'sum', enabled = True)]
                    )
                ]
            ))
        updatemenu_list.append(
            dict(label = y_cols[i],
                method = 'update',
                args = [{
                        'visible': visible
                    },
                    {
                        'title': y_cols[i] + ' Per Year',
                        'yaxis.title': y_cols[i]
                    }
                ])
        )

    layout = dict(
        title = '<b>Casualty Rate Per Year</b>',
        xaxis = dict(
            title = x_col,
            showgrid = False
        ),
        yaxis = dict(
            type = 'exp'
        ),
        updatemenus = list([
            dict(
                active = -1,
                buttons = updatemenu_list,
                direction = 'down',
                pad = {'r': 10, 't': 10},
                showactive = True,
                x = 0.05,
                xanchor = 'left',
                y = 1.1,
                yanchor = 'top'
            )
        ])
    )

    off.iplot({
        'data': data,
        'layout': layout
    }, validate = False)


# <a id="4"></a> <br>
# ### Analysis on Province

# <a id="a2"></a> <br>
# **Let's first check out Total Casualty Rate in each Province**

# In[ ]:


draw_barchart(df, 'Province', ['Killed', 'Injured'], 'Total Casualty Rate By Province','Province', 'Casualty Rate')


# **Let's see which Province has more Politician Casualty Rate**

# In[ ]:


draw_barchart(df, 'Province', ['Politician Killed', 'Politician Escaped'], 'Politician Casualty Rate By Province','Province', 'Politician Casualty Rate')


# **What kind of Attack has more tendency to occur in each Province?**

# In[ ]:


draw_barchart(df, 'Province', ['Target Attack', 'Suicide Attack'], 'Suicide Attack vs Target Attack By Province','Province', 'Attack Type')


# **Does more Attacks happened in Open or Closed Space?**

# In[ ]:


draw_barchart(df, 'Province', ['Open Space', 'Closed Space'], 'Open/Closed Space Attacks By Province','Province', 'Open/Closed Space')


# **What day has more casualty rate in each Province?**

# In[ ]:


#Group By Province and Day
df_province_day = df.groupby(['Province', 'Day'])['Politician Killed', 'Politician Escaped', 'Killed', 'Injured'].sum()
#Reset Index
df_province_day.reset_index(level=[0, 1], inplace=True)
#Join and Clean Columns
df_province_day['ProvinceByDay'] = df_province_day['Day'].str.cat(df_province_day['Province'],sep='-' )
df_province_day[['Politician Killed', 'Politician Escaped', 'Killed', 'Injured']] = df_province_day[['Politician Killed', 'Politician Escaped', 'Killed', 'Injured']].fillna(0)


# In[ ]:


draw_barchart(df_province_day, 'ProvinceByDay', ['Killed', 'Injured'], 'Total Casualty Rate By Province','Province', 'Casualty Rate', tick_angle=45)


# In[ ]:


draw_barchart(df_province_day, 'ProvinceByDay', ['Politician Killed', 'Politician Escaped'], 'Politician Casualty Rate By Province','Province', 'Casualty Rate', tick_angle=45)


# **What time of day has more casualty rate in each Province?**

# In[ ]:


#Group By Province and Time
df_province_time = df.groupby(['Province', 'Time'])['Politician Killed', 'Politician Escaped', 'Killed', 'Injured'].sum()
#Reset Index
df_province_time.reset_index(level=[0, 1], inplace=True)
#Join and Clean Columns
df_province_time['ProvinceByTime'] = df_province_time['Time'].str.cat(df_province_time['Province'],sep='-' )
df_province_time[['Politician Killed', 'Politician Escaped', 'Killed', 'Injured']] = df_province_time[['Politician Killed', 'Politician Escaped', 'Killed', 'Injured']].fillna(0)


# In[ ]:


draw_barchart(df_province_time, 'ProvinceByTime', ['Killed', 'Injured'], 'Total Casualty Rate By Province','Province', 'Casualty Rate', tick_angle=45)


# **At what time of day does more Politician's casualty happen in each Province?**

# In[ ]:


draw_barchart(df_province_time, 'ProvinceByTime', ['Politician Killed', 'Politician Escaped'], 'Politician Casualty Rate By Province','Province', 'Casualty Rate', tick_angle=45)


# <a id="5"></a> <br>
# ### Analysis on City

# <a id="a3"></a> <br>
# **Did City had any affect on Total Casualty?**

# In[ ]:


draw_barchart(df, 'City', ['Killed', 'Injured'], 'Total Casualty By City ','City', 'Total Casualty', tick_angle=90)


# **How about City having any affect on Politician Casualty?**

# In[ ]:


draw_barchart(df, 'City',  ['Politician Killed', 'Politician Escaped'], 'Politician Casualty By City ','City', 'Politician Casualty', tick_angle=90)


# <a id="a4"></a> <br>
# **What kind of Attack has more tendency to occur in each City?**

# In[ ]:


draw_barchart(df, 'City', ['Target Attack', 'Suicide Attack'], 'Suicide Attack vs Target Attack By City','City', 'Attack Type')


# **Does more Attacks happened in Open or Closed Space in each City?**

# In[ ]:


draw_barchart(df, 'City', ['Open Space', 'Closed Space'], 'Open/Closed Space Attacks By City','City', 'Open/Closed Space')


# **What time of day has more casualty rate in each City?**

# In[ ]:


#Group By City and Time
df_city_time = df.groupby(['City', 'Time'])['Politician Killed', 'Politician Escaped', 'Killed', 'Injured'].sum()
#Reset Index
df_city_time.reset_index(level=[0, 1], inplace=True)
#Join and Clean Columns
df_city_time['CityByTime'] = df_city_time['Time'].str.cat(df_city_time['City'],sep='-' )
df_city_time[['Politician Killed', 'Politician Escaped', 'Killed', 'Injured']] = df_city_time[['Politician Killed', 'Politician Escaped', 'Killed', 'Injured']].fillna(0)


# In[ ]:


draw_barchart(df_city_time, 'CityByTime', ['Killed', 'Injured'], 'Total Casualty Rate By City and Time','City', 'Casualty Rate', tick_angle=45)


# In[ ]:


draw_barchart(df_city_time, 'CityByTime', ['Politician Killed', 'Politician Escaped'], 'Politician Casualty Rate By City and Time','City', 'Casualty Rate', tick_angle=45)


# **Which day has more casualty rate in each City?**

# In[ ]:


#Group By City and Day
df_city_day = df.groupby(['City', 'Day'])['Politician Killed', 'Politician Escaped', 'Killed', 'Injured'].sum()
#Reset Index
df_city_day.reset_index(level=[0, 1], inplace=True)
#Join and Clean Columns
df_city_day['CityByDay'] = df_city_day['Day'].str.cat(df_city_day['City'],sep='-' )
df_city_day[['Politician Killed', 'Politician Escaped', 'Killed', 'Injured']] = df_city_day[['Politician Killed', 'Politician Escaped', 'Killed', 'Injured']].fillna(0)


# In[ ]:


draw_barchart(df_city_day, 'CityByDay', ['Killed', 'Injured'], 'Total Casualty Rate By City and Day','City', 'Casualty Rate', tick_angle=45)


# In[ ]:


draw_barchart(df_city_day, 'CityByDay', ['Politician Killed', 'Politician Escaped'], 'Politician Casualty Rate By City and Day','City', 'Casualty Rate', tick_angle=45)


# <a id="a5"></a> <br>
# **Which Days are Suicide and Target Attacks more likely to happen?**

# In[ ]:


draw_barchart(df, 'Day', ['Target Attack', 'Suicide Attack'], 'Attack Type By Days','Day', 'Attack Type')


# <a id="6"></a> <br>
# ### Analysis on Day and Time

# <a id="a6"></a> <br>
# **How about Time of the Day?**

# In[ ]:


draw_barchart(df, 'Time', ['Target Attack', 'Suicide Attack'], 'Attack Type By Time of Day','Time of Day', 'Attack Type')


# In[ ]:


draw_barchart(df, 'Location Category', ['Target Attack', 'Suicide Attack'], 'Attack Type By Time of Day','Time of Day', 'Attack Type')


# **Has Day got anything to do with Casualty?**

# In[ ]:


draw_barchart(df, 'Day', ['Politician Killed', 'Politician Escaped'], 'Politician Casualty By Day','Day', 'Politician Casualty')


# <a id="7"></a> <br>
# ### Analysis on Location Category

# **Does place of attack affect Total Casualty?**

# In[ ]:


draw_barchart(df, 'Location Category', ['Killed', 'Injured'], 'Total Casualty By Location ','Location', 'Total Casualty')


# **Does place of attack affect Politician Casualty?**

# In[ ]:


draw_barchart(df, 'Location Category', ['Politician Killed', 'Politician Escaped'], 'Politician Casualty By Location ','Location', 'Politician Casualty')


# **Does time and location has any relation with total casualty?**

# In[ ]:


#Group By Location and Time
df_loc_time = df.groupby(['Location Category', 'Time'])['Politician Killed', 'Politician Escaped', 'Killed', 'Injured'].sum()
#Reset Index
df_loc_time.reset_index(level=[0, 1], inplace=True)
#Join and Clean Columns
df_loc_time['LocationByTime'] = df_loc_time['Time'].str.cat(df_loc_time['Location Category'],sep='-' )
df_loc_time[['Politician Killed', 'Politician Escaped', 'Killed', 'Injured']] = df_loc_time[['Politician Killed', 'Politician Escaped', 'Killed', 'Injured']].fillna(0)


# In[ ]:


draw_barchart(df_loc_time, 'LocationByTime', ['Killed', 'Injured'], 'Total Casualty Rate By Location Category and Time','Location Category', 'Casualty Rate', tick_angle=45)


# **Does time and location has any relation with politician casualty?**

# In[ ]:


draw_barchart(df_loc_time, 'LocationByTime', ['Politician Killed', 'Politician Escaped'], 'Politician Casualty Rate By Location Category and Time','Location Category', 'Casualty Rate', tick_angle=45)


# **Does day and location has any relation with total casualty?**

# In[ ]:


#Group By Location and Time
df_loc_day = df.groupby(['Location Category', 'Day'])['Politician Killed', 'Politician Escaped', 'Killed', 'Injured'].sum()
#Reset Index
df_loc_day.reset_index(level=[0, 1], inplace=True)
#Join and Clean Columns
df_loc_day['LocationByDay'] = df_loc_day['Day'].str.cat(df_loc_day['Location Category'],sep='-' )
df_loc_day[['Politician Killed', 'Politician Escaped', 'Killed', 'Injured']] = df_loc_day[['Politician Killed', 'Politician Escaped', 'Killed', 'Injured']].fillna(0)


# In[ ]:


draw_barchart(df_loc_day, 'LocationByDay', ['Killed', 'Injured'], 'Total Casualty Rate By Location Category and Day','Location Category', 'Casualty Rate', tick_angle=45)


# **Does day and location has any relation with political casualty?**

# In[ ]:


draw_barchart(df_loc_day, 'LocationByDay', ['Politician Killed', 'Politician Escaped'], 'Politician Casualty Rate By Location Category and Day','Location Category', 'Casualty Rate', tick_angle=45)


# <a id="8"></a> <br>
# ### Analysis on Attack Type

# **What's the Total Casualty in both Attack Types?**

# In[ ]:


draw_barchart(df, 'Target Category', ['Killed', 'Injured'], 'Total Casualty By Attack Type','Attack Type', 'Total Casualty')


# <a id="a7"></a> <br>
# **Does Attack Type show any relation with Politician Casualty?**

# In[ ]:


draw_barchart(df, 'Target Category', ['Politician Killed', 'Politician Escaped'], 'Politician Casualty By Attack Type','Attack Type', 'Politician Casualty')


# <a id="9"></a> <br>
# ### Analysis on Party

# **How's the total casualty rate for attack on each party?**

# In[ ]:


draw_barchart(df, 'Party', ['Killed', 'Injured'], 'Total Casualty By Party ','Party', 'Total Casualty')


# **How's the politician's casualty rate for each party?**

# In[ ]:


draw_barchart(df, 'Party', ['Politician Killed', 'Politician Escaped'], 'Politician Casualty By Party ','Party', 'Politician Casualty')


# <a id="a8"></a> <br>
# **Which kind of attack each party faced?**

# In[ ]:


draw_barchart(df, 'Party', ['Target Attack', 'Suicide Attack'], 'Attack Type By Party ','Party', 'Attack Type')


# <a id="10"></a> <br>
# ### Trend Over The Years

# **Grouping Casualty Data by Year**

# In[ ]:


#Group By Year
df_year = df.groupby('year')['Injured', 'Killed'].sum()
df_year.reset_index(level=0, inplace=True)


# In[ ]:


draw_bubblechart(df_year, 'year', ['Killed', 'Injured'])


# <a id="11"></a> <br>
# ### Location Word Cloud

# **Now Let's look at Location Feature and try to see if there exists some pattern in it**

# In[ ]:


wordcloud = WordCloud(background_color='white').generate(" ".join(df['Location']))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# <a id="12"></a> <br>
# ### Observations

# 
# * [**Most Political attacks were done on road or in their residence**](#a1)
# 
# 
# * [**Low attack frequency was observed on Sunday**](#2)
# 
# 
# * [**KPK faced most attacks along with highest politician killed but most civilian casualty was observed in Sindh Province**](#a2)
# 
# 
# * [**After Karachi, most civilian casualty was observed in Quetta whereas Quetta and Peshawar had same number of Politicians Killed after Karachi**](#a3)
# 
# 
# * [**Even though Karachi had most civilian casualty but most suicide attacks were done in Peshawar whereas Karachi only had a single suicide attack**](#a4)
# 
# 
# * [**Attacks on Thursday were mostly suicide attacks with only single attack out of 7 attacks being a targeted attack**](#a5)
# 
# 
# * [**Most attacks were done on afternoon and evening but 16/23 attacks in evening were suicide attacks whereas 5/9 attacks in afternoon were suicide attacks**](#a6)
# 
# 
# * [**In a targeted attack, only 3 politicians out of 24 escaped whereas suicide attack 17 politicians out of 27 managed to escape**](#a7)
# 
# 
# * [**All attacks on ANP party were suicide attacks whereas PTI politicians only faced targeted attacks. Additionally, PPPP and PMLN were mostly victims of suicide attacks**](#a7)
# 
# 
