#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#Import the relevant modules
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import requests as r
import seaborn as sns
import re
import time
import bs4 as BeautifulSoup
import sys  
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode, plot
init_notebook_mode(connected=True)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Global Terrorism Data Analysis
# 
# ## __About Notebook__
# The contents of this notebook are supplementary to a Dash web application. The majority of the below is scratchwork that will be used to build the web application - I've included my thought process to hopefully help other "wanna-be Data Scientists" who are trying to  something similar. The notebook will enclude rudimentary analysis and aims to produce an "end product" which will be a refined dataframe that can be used as an input for the Dash app. Also, this is an on going project as I am still learning and trying new things but I thought I'd share the finer points with the Kaggle community. 
# 
# ## __Analysis Objective__
# The purpose of this analysis is to help combat the rise and spread of terrorism. The analysis hopes to achieve this objective by first providing a collection of useful graphs/charting material that wil provide a better illustration of coherent terrorism trends and eventually tools that will use supervised learning methods to predict the *odds* of an attack. While it hopefully shouldn't need any explaining, this product tries to avoid any political/religious/personal biases in order to make more accurate predictions.
# 
# #### A Note to Readers
# This notebook uses interactive plotly charts. For charts which have multiple series, you can de/select a series by clicking on it once and zoom in on just that one series by double clicking on it (and the reactivating all series by double clicking).

# In[ ]:


#Import the Datasets
gtd = pd.read_csv('../input/gtd/globalterrorismdb_0718dist.csv', encoding='ISO-8859-1',)
econ_df = pd.read_csv('../input/econset/AP_NY_MKTCAP.csv', skiprows = 4)


# ## Formating

# In[ ]:


#Evaluate the Paramaters and Rows
gtd.info() #Evaluation of the Paramaters results in a high number of features
gtd.rename(
    columns={'iyear':'year', 'imonth':'month', 'iday':'day', 'gname': 'gname1', 'claimmode': 'claimmode1', 'gsubname': 'gsubname1', 'claimmode_txt': 'claimmode1_txt'}, inplace=True)


#The below breaks the features into varioius subgroups for further study
db_vars = ['eventid', 'year', 'month', 'day','approxdate','extended','resolution']
info_vars = ['summary', 'crit1', 'crit2', 'crit3','doubtterr','alternative',
                'alternative_txt', 'multiple','related'] #Array of General Paramaters
loc_vars = ['country', 'country_txt','region', 'region_txt','provstate','city','vicinity','location',
           'latitude','longitude', 'specificity'] #Array of Location Parameters
atk_vars = ['attacktype1','attacktype1_txt', 'attacktype2','attacktype2_txt','attacktype3',
           'attacktype3_txt','success','weaptype1','weaptype1_txt','weapsubtype1','weapsubtype1_txt',
           'weaptype2','weaptype2_txt','weapsubtype2','weapsubtype2_txt','weaptype3','weaptype3_txt',
           'weapsubtype3','weapsubtype3_txt','weaptype4','weaptype4_txt','weapsubtype4',
            'weapsubtype4_txt','weapdetail'] #Array of Attack Detail Parameters
vic_vars = ['targtype1','targtype1_txt','targsubtype1','targsubtype1_txt','corp1','target1',
           'natlty1','natlty1_txt','targtype2','targtype2_txt','targsubtype2','targsubtype2_txt',
           'corp2','target2','natlty2','natlty2_txt','targtype3','targtype3_txt','targsubtype3',
           'targsubtype3_txt','corp3','target3','natlty3','natlty3_txt'] #Array of Target/Victim Parameters
perp_vars = ['gname1','gsubname1','gname2','gsubname2','gname3','gsubname3','guncertain1','guncertain2','guncertain3',
           'individual','nperps','nperpcap','claimmode1','claimmode1_txt','compclaim','claim2','claimmode2',
            'claim3','claimmode3','motive'] #Arrray of Perpetrator Parameters
cas_vars =['nkill','nkillus','nkillter','nwound','nwoundus','nwoundte','property','propextent','propextent_txt',
          'propvalue','propcomment','ishostkid','nhostkid','nhostkidus','nhours','ndays','divert','kidhijcountry',
          'ransom','ransomamt','ransomamtus','ransompaid','ransompaidus','ransomnote','hostkidoutcome',
           'hostkidoutcome_txt','nreleased'] #Array of Casualty Parameters
xtra_vars = ['addnotes','INT_LOG','INT_IDEO','INT_MISC','INT_ANY','scite1','scite2','scite3','dbsource'] #Additional Misc Parameters


# In[ ]:


#There are still too many parameters even within the sub groups - further cleaning below
pattern = r'_txt'

#Will create a function to clean up the extra columns with name "text"
group = [db_vars,info_vars,loc_vars,atk_vars,vic_vars,perp_vars,cas_vars,xtra_vars]

def txt_col(pattern,group): #CAN ONLY BE PERFORMED ONCE
    cols_dict = {} #Dictionary to upload to our rename dictionary
    toss_list = [] #List to drop and create separate DF for ML
    cat_list = [] #List to change format to categories for charts
    for var in group: #For loop to create entries in dictionary
        for param in var:
            if re.search(pattern,param):
                root = param[0:(len(param)-len(pattern))]
                cols_dict[root] = str(root)+"_cat"
                cols_dict[param] = root
                toss_list.append(cols_dict[root])
                cat_list.append(root)
                var.remove(param)
    return cols_dict, toss_list, cat_list 

cols_dict, toss_list, cat_list = txt_col(pattern,group)

#Set of operations TO BE PERFORMED ONCE 
gtd.rename(columns=cols_dict, inplace=True)
gtd_ml = gtd[toss_list].copy()
gtd.drop(columns=toss_list, inplace=True)


# In[ ]:


#Next up we are going to create a dictionary of country codes from another database
#We will map the values of the Country code to the input database so se can create Chloropleth Maps

#Ahead we saw that some Country names are different...
econ_df.loc[econ_df['Country Name'] == 'Russian Federation', 'Country Name'] = 'Russia'


#We need to build some script to add country codes to a new dictionary so we can build new DF
each_country = econ_df['Country Name'].unique()
country_dict = {}
code_dict = {}
for i, e in enumerate(each_country):
    country_dict[e] = econ_df.loc[econ_df['Country Name'] == e,'Country Code'][i]
    code_dict[econ_df.loc[econ_df['Country Name'] == e,'Country Code'][i]] = e

gtd.insert(list(gtd.columns).index('country'),'country_code',gtd.country)
gtd['country_code'] = gtd.country_code.map(country_dict)


# In[ ]:


#To create a map to better visualize the spread, we prep the data by creating a new DF

#First we create a few new dataframes to more easily extract the various countries
map_df = econ_df.copy()

#set the new DF's index to country code so you can more easily dig out the value counts later

map_df.set_index('Country Code',inplace = True)
map_df['Terror Incidents'] = gtd.country_code.value_counts()
map_df['Terror Incidents'].fillna(0, inplace=True)


# In[ ]:


"""
We write a function  to create a new valcnt column so we can make many (there are many groups)
The following function is for string outputs
We see that a number of the categories have types 1-3 so a for loop concatenates all the relevant values

The function will take the mapdf, filter column and new column values as inputs
The function will output and perform column operations which will create a new column with aggregate values
The function will also have a nested funciton to create dictionaries which will be used to later aggregate valeus
"""
def val_cnt_create(df, col_fil, n_col):
    # First for loop pulls out each country code
    for e in df.index:
        dict_list = [] #Array will be used to store dictionaries later of value counts later
        param_list = [] #Array will store all the different values for global parameters
        txt_col = [] #Array will concatenate the string values in the ned
        first_str = "<b>" + df.loc[e,'Country Name'] + '</b><br> '
        
        def dict_create(n, col_fil): #Function to create dicitonaries to append to dict_list
            unique_params = []
            spec_filter = col_fil+str(n+1) #variable to track column titles from gtd
            for param in gtd[spec_filter].unique():
                unique_params.append(param)  
            filtered_df = gtd.loc[gtd.country_code == e, spec_filter] #variable for filtered dated set
            val_cnt_dict = {filtered_df.value_counts().index[i]: a1 for i, a1 in enumerate(filtered_df.value_counts())}
            return val_cnt_dict, unique_params
    
        
        #Second for loop creates a column with dictionary of different va variables
        for n1 in range(3):
            val, unique = dict_create(n1, col_fil)
            dict_list.append(val)
            for x in unique:
                if x not in param_list:
                    param_list.append(x)
                    
        old_dict = {param: 0 for param in param_list}
        for dic in dict_list:
            for param in param_list:
                try:
                    old_dict[param] += dic[param]
                except:
                    pass
        for param in param_list:
            if old_dict[param] == 0:
                del old_dict[param]
            try:
                txt_input = "{}: {} <br> ".format(param, old_dict[param])
                txt_col.append(txt_input)
            except:
                pass
        
        second_str = "".join(txt_col)
        txt = first_str + second_str
        df.loc[e,n_col] = txt

#Whew! Not the sexiest piece of code with all those for loops, but the number of parameters is not too many
#Now we can create many many chart inputs for many different charts!


# In[ ]:


def sum_create(a, col_fil, n_col):
    for e in a.index:
        total = gtd.loc[gtd.country_code == e, col_fil].sum()
        a.loc[e, n_col] = total.astype(int)

sum_create(map_df, 'nkill', 'Total Victims')
sum_create(map_df, 'nkillter', 'Terrorist Deaths')
map_df['Total Body Count'] = map_df['Total Victims'] + map_df['Terrorist Deaths']
sum_create(map_df, 'propvalue', 'Total Property Damage')


# In[ ]:


val_cnt_create(map_df, 'gname', 'Perpetrator Text') 
val_cnt_create(map_df, 'weaptype', 'Weapon Text')
val_cnt_create(map_df, 'attacktype', 'Attack Text')
val_cnt_create(map_df, 'targtype', 'Target Text')
val_cnt_create(map_df, 'natlty', 'Target Nationality Text')
val_cnt_create(map_df, 'gsubname', 'Perpsub Text')


# In[ ]:


#Next create a lazy function to construct quick maps - the following is for the chloropleth types
#Chlorpleth maps will use the "Country Code" columns that were created earlier

#Lets give ourself options for color so we can later divide themes
color_dict = {'blue': [
        [0, "rgb(5, 10, 172)"],
        [0.35, "rgb(40, 60, 190)"],
        [0.5, "rgb(70, 100, 245)"],
        [0.6, "rgb(90, 120, 245)"],
        [0.7, "rgb(106, 137, 247)"],
        [1, "rgb(220, 220, 220)"]
    ] , 
          'green': [
        [0, "rgb(5, 172, 10)"],
        [0.35, "rgb(40, 190, 60)"],
        [0.5, "rgb(70, 245, 100)"],
        [0.6, "rgb(90, 245, 120)"],
        [0.7, "rgb(106, 247, 137)"],
        [1, "rgb(220, 220, 220)"]
    ],
          'red': [
        [0, "rgb(172, 10, 5)"],
        [0.35, "rgb(190, 60, 40)"],
        [0.5, "rgb(245, 100, 70)"],
        [0.6, "rgb(245, 120, 90)"],
        [0.7, "rgb(247, 137, 106)"],
        [1, "rgb(220, 220, 220)"]
    ]}

def button_vars(df, main_var, text_var, color='blue'):
    
    #Next create the data variable - array of graph objects 
    data = dict(type = 'choropleth',
        locations = df.index,
        z = df[main_var],
        text = [e for e in df[text_var]],
        colorscale = color_dict[color],#we implement the dicitonary constructed above
        autocolorscale = False,
        reversescale = True,
        marker = go.choropleth.Marker(
            line = go.choropleth.marker.Line(
            color = 'rgb(180,180,180)',
            width = 0.5
            )),
        colorbar = go.choropleth.ColorBar( 
            title = 'Number<br>of {}'.format(main_var),#function input
        )
    )
    
        
    layout = {
        'title': {
            'text' : '{} by Country'.format(main_var) #function input
            },
        'annotations' : [{
            'x' : 0.55,
            'y' : 0.1,
            'xref' : 'paper',
            'yref' : 'paper',
            'text' : 'Hover over countries for more details on {}s'.format(text_var.split(" ")[0]),#function input
            'showarrow': False
        }]
    }
    

    return data, layout


# In[ ]:


#First create empty dictionary for figure
vic_tgt_dat, vic_tgt_lay = button_vars(map_df, 'Total Victims', 'Target Text', 'red')
vic_perp_dat, vic_perp_lay = button_vars(map_df, 'Total Victims', 'Perpetrator Text', 'red')
vic_atk_dat, vic_atk_lay = button_vars(map_df, 'Total Victims', 'Attack Text', 'red')

tot_tgt_dat, tot_tgt_lay = button_vars(map_df, 'Total Body Count', 'Target Text', 'red')
tot_perp_dat, tot_perp_lay = button_vars(map_df, 'Total Body Count', 'Perpetrator Text', 'red')
tot_atk_dat, tot_atk_lay = button_vars(map_df, 'Total Body Count', 'Attack Text', 'red')

inc_tgt_dat, inc_tgt_lay = button_vars(map_df, 'Terror Incidents', 'Target Text', 'blue')
inc_perp_dat, inc_perp_lay = button_vars(map_df, 'Terror Incidents', 'Perpetrator Text', 'blue')
inc_atk_dat, inc_atk_lay = button_vars(map_df, 'Terror Incidents', 'Attack Text', 'blue')

prp_tgt_dat, prp_tgt_lay = button_vars(map_df, 'Total Property Damage', 'Target Text', 'green')
prp_perp_dat, prp_perp_lay = button_vars(map_df, 'Total Property Damage', 'Perpetrator Text', 'green')
prp_atk_dat, prp_atk_lay = button_vars(map_df, 'Total Property Damage', 'Attack Text', 'green')


atk_dat, atk_lay = button_vars(map_df, 'Terror Incidents', 'Attack Text', 'blue')


figure = {'data': [],
         'layout': {},
         'frames': []}

figure['data'] = [atk_dat]
figure['layout'] = go.Layout(
        title = go.layout.Title(
        text = '{} by Country'.format('Total Victims') #function input
        ),
        geo = go.layout.Geo(
            showframe = False,
            showcoastlines = False,
            projection = go.layout.geo.Projection(
                type = 'equirectangular'
            )
        ),
        annotations = [go.layout.Annotation(
            x = 0.6,
            y = 0.1,
            xanchor='right',
            yanchor='top',
            xref = 'paper',
            yref = 'paper',
            text = 'Hover over countries for more details'.format('Target Text'.split(" ")[0]),#function input
            showarrow = False
        )]
    )



figure['frames'].append({'data': [vic_tgt_dat], 'name': 'vic tgt'})
figure['frames'].append({'data': [vic_perp_dat], 'name': 'vic perp'})
figure['frames'].append({'data': [vic_atk_dat], 'name': 'vic atk'})

figure['frames'].append({'data': [tot_tgt_dat], 'name': 'tot tgt'})
figure['frames'].append({'data': [tot_perp_dat], 'name': 'tot perp'})
figure['frames'].append({'data': [tot_atk_dat], 'name': 'tot atk'})

figure['frames'].append({'data': [inc_tgt_dat], 'name': 'inc tgt'})
figure['frames'].append({'data': [inc_perp_dat], 'name': 'inc perp'})
figure['frames'].append({'data': [inc_atk_dat], 'name': 'inc atk'})

figure['frames'].append({'data': [prp_tgt_dat], 'name': 'prp tgt'})
figure['frames'].append({'data': [prp_perp_dat], 'name': 'prp perp'})
figure['frames'].append({'data': [prp_atk_dat], 'name': 'prp atk'})

figure['layout']['updatemenus'] = [
    {'buttons': [
        {'args': [['vic tgt'], {'frame': {'duration': 300, 'redraw': True},
                                    'fromcurrent': True, 
                                    'transition': {'duration': 300, 'easing': 'quadratic-in-out'},
                                    }, vic_tgt_lay],
         'label': 'Target by Total Victims',
         'method': 'animate'},
        {'args': [['tot tgt'], {'frame': {'duration':300, 'redraw': True},
                                    'fromcurrent': True, 
                                    'transition': {'duration': 300, 'easing': 'quadratic-in-out'}
                                   }, tot_tgt_lay],
         'label': 'Target by Total Body Count ',
         'method': 'animate'},
        {'args': [['inc tgt'], {'frame': {'duration':300, 'redraw': True},
                                    'fromcurrent': True, 
                                    'transition': {'duration': 300, 'easing': 'quadratic-in-out'}
                                   }, inc_tgt_lay],
         'label': 'Target by Total Incidents ',
         'method': 'animate'},
        {'args': [['prp tgt'], {'frame': {'duration':300, 'redraw': True},
                                    'fromcurrent': True, 
                                    'transition': {'duration': 300, 'easing': 'quadratic-in-out'}
                                   }, prp_tgt_lay],
         'label': 'Target by Total Cost ',
         'method': 'animate'}
        
    ],
     'direction': 'down',
     'pad': {'r': 10, 't': 87},
     'showactive': False,
     'x': 0.1,
     'xanchor': 'right',
     'y': 1,
     'yanchor': 'top'}, 
    {'buttons': [
        {'args': [['vic perp'], {'frame': {'duration': 300, 'redraw': True},
                                    'fromcurrent': True, 
                                    'transition': {'duration': 300, 'easing': 'quadratic-in-out'},
                                    }, vic_perp_lay],
         'label': 'Perpetrator by Total Victims',
         'method': 'animate'},
        {'args': [['tot perp'], {'frame': {'duration':300, 'redraw': True},
                                    'fromcurrent': True, 
                                    'transition': {'duration': 300, 'easing': 'quadratic-in-out'}
                                   }, tot_perp_lay],
         'label': 'Perpetrator by Total Body Count ',
         'method': 'animate'},
        {'args': [['inc perp'], {'frame': {'duration':300, 'redraw': True},
                                    'fromcurrent': True, 
                                    'transition': {'duration': 300, 'easing': 'quadratic-in-out'}
                                   }, inc_perp_lay],
         'label': 'Perpetrator by Total Incidents ',
         'method': 'animate'},
        {'args': [['prp perp'], {'frame': {'duration':300, 'redraw': True},
                                    'fromcurrent': True, 
                                    'transition': {'duration': 300, 'easing': 'quadratic-in-out'}
                                   }, prp_perp_lay],
         'label': 'Perpetrator by Total Cost ',
         'method': 'animate'}
        
    ],
     'direction': 'down',
     'pad': {'r': 10, 't': 87},
     'showactive': False,
     'x': 0.1,
     'xanchor': 'right',
     'y': 0.9,
     'yanchor': 'top'},
    {'buttons': [
        {'args': [['vic atk'], {'frame': {'duration': 300, 'redraw': True},
                                    'fromcurrent': True, 
                                    'transition': {'duration': 300, 'easing': 'quadratic-in-out'},
                                    }, vic_atk_lay],
         'label': 'Attack Type by Total Victims',
         'method': 'animate'},
        {'args': [['tot atk'], {'frame': {'duration':3500, 'redraw': True},
                                    'fromcurrent': True, 
                                    'transition': {'duration': 300, 'easing': 'quadratic-in-out'}
                                   }, tot_atk_lay],
         'label': 'Attack Type by Total Body Count ',
         'method': 'animate'},
        {'args': [['inc atk'], {'frame': {'duration':3500, 'redraw': True},
                                    'fromcurrent': True, 
                                    'transition': {'duration': 300, 'easing': 'quadratic-in-out'}
                                   }, inc_atk_lay],
         'label': 'Attack by Total Incidents ',
         'method': 'animate'},
        {'args': [['prp atk'], {'frame': {'duration':3500, 'redraw': True},
                                    'fromcurrent': True, 
                                    'transition': {'duration': 300, 'easing': 'quadratic-in-out'}
                                   }, prp_atk_lay],
         'label': 'Attack by Total Cost ',
         'method': 'animate'}
        
    ],
     'direction': 'down',
     'pad': {'r': 10, 't': 87},
     'showactive': False,
     'x': 0.1,
     'xanchor': 'right',
     'y': 0.8,
     'yanchor': 'top'}]


iplot(figure, filename = 'd3-world-map')


# ### What does this tell us?
# 
# #### <u>Attacks</u>:
# <p>
#     We can see from hovering over the main hot spots that bombing/explosions is the predominant method of attack
#     across every region followed usually by Armed Assault or Facillity Infrastructure Attacks. this information seems
#     relevant but should be expressed as a bar graph (horizontal) in the dashboard
# </p>
# 
# #### <u> Perpetrators/Perpetrator sub groups</u>:
# <p>
#     The perpetrator visual by group seems less helpful as the underlying data is inconsistent - we're gonna try taking a look as a scatter plot instead
#     </p>
# 
# #### <u> Weapon Visuals </u>:
# <p>
#     Also not hugely helpful as it is, we'll try some scatter plots for the weapons and see what other trends we can find
#     
# 
# ### How can we use this?
# 
# The filter view is a good idea for the dash application implementation. Initial idea is to implement the graph in the lower left majority of the dash application. 
# 
# #### <u> Specific additions</u>
# - [ ] The right panel will need to have linked windows which can give greater detail on other subtypes. 
# - [ ] We will need to implement a range slider that can show break downs for date ranges
# - [ ] Annoyingly, the documentation for the plotly offline buttons is very thin so I'm unsure on how to use the update method to adjust the data attributes (I tried dictionaries combined with lists) but have been so far unsuccessful. If anyone in the Kaggle Community has some better suggestions, would love to hear them. If I could use the updae method to change parameters in both the Layout and the Data section then I could use the animation to connect the map to a slider and  perhaps even add some scatter geos to see how different attack patterns evolve over time
# - [ ] A bar chart to see the distribution of parameters is more useful than text - something to link to the right panel if there is a hoverData output
# 
# 
# ### What should we check next
# - [X] Lets see if we can find some break down of the main perpetrator for each country and links to the acts they have commmitted (accomplished with funciton above)
# - [X] Analysis for damage dealt with total dead and wounded - include property damage (accomplishd with function above)
# - [X] Lets see breakdown of target types
# 
# Overall, I'd like to do a lot more with this chart but I wanted to show what I had so far. 

# In[ ]:


#Next lets look at a couple historical events
#Since there are too many countries and we will use 5 countries with most attack for each year

traces = []

#Since we don't want a bunch of duplicate series, we will create graphs per country
country_dictY = {} #dictionary for y values
country_dictX = {} #dictionary for xvalues
country_dictTxt = {}
for e in gtd.year.unique(): #We want to pull up top 5 for each year and add them to dictionaries
    top_5 = gtd.groupby(['year','country']).country.count()[e].sort_values(ascending=False)[0:5]
    for i, country in enumerate(top_5.index):
        if country not in country_dictX:
            country_dictX[country] = []
            country_dictY[country] = []
            country_dictTxt[country] = []
            
        country_dictX[country].append(e)
        country_dictY[country].append(top_5[i])
        text_primer = gtd.loc[(gtd.country==country) & (gtd.year == e), 'gname1'].value_counts().sort_values(ascending=False).index
        try:
            txt_input = "Three Most Active Terror Groups <br>1) {}<br>2) {}<br>3) {}".format(text_primer[0],text_primer[1],text_primer[2])
        except:
            txt_input = "Most Active Group<br> {}".format(text_primer[0])
        
        country_dictTxt[country].append(txt_input)


for country in country_dictX: #After our dictionaries are loaded, we create instances of graph object and append to list
    #text = "Top 3 Most active Terror Groups <br>1 {}<br>2 {}<br>3 {}".format(country_dictTxt[country].index[0],country_dictTxt[country].index[1],country_dictTxt[country].index[2])
    
    trace = go.Bar(x = country_dictX[country], y=country_dictY[country], name=country, text=country_dictTxt[country])
    traces.append(trace)
line_trace = go.Scatter(x = gtd.groupby('year').year.count().index, 
                        y=[total for total in gtd.groupby('year').year.count()], 
                           name='Total Terror Attacks for the year', 
                           mode='lines+markers', 
                       yaxis='y2')
traces.append(line_trace)

data = traces 
layout = go.Layout(
    autosize = False, width = 1000, height=750, 
    title = 'Terrorist Attacks by Year in Five most Affected Countries',
    xaxis = dict(
             rangeslider = dict(thickness = 0.1),
             showline = True,
             showgrid = False
         ),
    yaxis = dict(title='Terror Attack by Country',
             showline = True,
             showgrid = False),
    yaxis2 = dict(title = 'Total Terror Attack',
                 showline = True,
                 overlaying = 'y', #Allows for second axis
                 side = 'right'),
    legend=dict(x=1.1, y=1.2),
    annotations = [go.layout.Annotation(
            x = 0.01,
            y = -.1,
            xref = 'paper',
            yref = 'paper',
            text = 'Hover over countries for more details on Groups involved',
            showarrow = False
        )]#If you don't want the leged rubbing to close to the graph...
)
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='grouped-bar')


# ### About this chart
# - Hover over the data points to get more details on the main groups and the countries (the label is sometimes not so visible on the right)
# - I added a range slider as its sometimes nice to take a look at the the years upclose (thought it would be nice if someone knew how to stop the years to switch to smaller integers?
# 
# 
# ### What does this tell us?
# - This graph shows that the sharp rise in global terrorism really took off in 2011 with the rise of the Taliban in Iraq, Afghanistan and Pakistan. 
# - Things really started kicking off when ISIL entered the scene in 2013
# - Interestingly, there was a drop in the period of the 90s up to 9/11
# - The period before 9/11 was marked by largely local/smaller scale incidents in regions experiencing political unrest
# - The 80s were marked by generally elevating tensions which seem to drop shortly after cold war (conveniently during the Clinton  era - related?). The regions most affected seem to be Central/South American countries
# - The 70s seemed like they were a milder time where we go even more local (heating tensions in Ireland)
# 
# #### Things to highlight
# - The trend in this data highlihgts that we are lacking feature representation in this data as there is clearly a trend (introduction of new adversaries and political climate) so part of the project to make this into a successful predictive algorithm, we will need to extract additional features like political climate and formation of new groups 

# In[ ]:


#Lets create a visual which highlights the terror attacks by target types
#We want to create a scatter plot which will capture injuries, kills and total attempts
#The text should include (1) Most Frequent Group, (2) Most common country
scatter_traces = []
kill_dicts = []
injury_dicts = []
size_dicts = []
for n in range(3):
    kill_dict = {gtd.groupby(['targtype'+str(n+1)]).nkill.sum().index[i]: e for i, e in enumerate(gtd.groupby(['targtype'+str(n+1)]).nkill.sum())}
    injury_dict = {gtd.groupby(['targtype'+str(n+1)]).nwound.sum().index[i]: e for i, e in enumerate(gtd.groupby(['targtype'+str(n+1)]).nwound.sum())}
    size_dict = {gtd['targtype'+str(n+1)].value_counts().index[i]: e for i, e in enumerate(gtd['targtype'+str(n+1)].value_counts())}
    kill_dicts.append(kill_dict)
    injury_dicts.append(injury_dict)
    size_dicts.append(size_dict)

tkill_dict = {e: 0 for e in gtd.targtype1.unique()}
tinjury_dict = {e: 0 for e in gtd.targtype1.unique()}
tsize_dict = {e: 0 for e in gtd.targtype1.unique()}
text_dict = {e:0 for e in gtd.targtype1.unique()}

for e in gtd.targtype1.unique():
    for dic1 in kill_dicts:
        try:
            tkill_dict[e] += dic1[e]
        except:
            pass
    for dic2 in injury_dicts:
        try:
            tinjury_dict[e] += dic2[e]
        except:
            pass
    for dic3 in size_dicts:
        try:
            tsize_dict[e] += dic3[e]
        except:
            pass
    most_group = gtd.loc[gtd.targtype1==e,'gname1'].value_counts().sort_values(ascending=False).index[0]
    most_weap = gtd.loc[gtd.targtype1==e,'weaptype1'].value_counts().sort_values(ascending=False).index[0]
    most_attack = gtd.loc[gtd.targtype1==e,'attacktype1'].value_counts().sort_values(ascending=False).index[0]
    most_country = gtd.loc[gtd.targtype1==e, 'country'].value_counts().sort_values(ascending=False).index[0]
    
    try:
        detperc = round(tkill_dict[e]/(tkill_dict[e]+tinjury_dict[e])*100,2)
    except:
        detperc = 'No Attacks'


    text_dict[e] = 'Target Type <b>{cat}</b> <br>Total Injuries: <b>{inj}</b>, Total Deaths: <b>{det}</b><br>     Total Incidents: <b>{inc}</b>, Death to Death + Injury Percentage <b>    {detperc}%</b><br>The most common attack: <b>{atk}</b><br>     The most affected country: <b>{con}</b>'.format(cat=e,                                                    con=most_country,                                                     det=tkill_dict[e], inj=tinjury_dict[e],                                                    detperc=detperc, atk = most_attack,                                                    inc=tsize_dict[e])





for i, item in enumerate(tkill_dict):
    scatter_trace = go.Scatter(x =[tinjury_dict[item]], y = [tkill_dict[item]], name = item, mode = 'markers', text = text_dict[item],
        marker = dict(
            size = tsize_dict[item]/1000,
            opacity = 0.9))
    
    scatter_traces.append(scatter_trace)

data = scatter_traces 
layout = go.Layout(
    autosize = False, width = 1000, height = 750,
    title = 'Target Victim Breakdowns',
    xaxis = dict(title = "Total Injuries",
                 #type = 'log',
             showline = True,
             showgrid = False
         ),
    yaxis = dict(title='Total Kills',
                 type = 'log',
             showline = True,
             showgrid = False),
    annotations = [go.layout.Annotation(
            x = 0.55,
            y = 0.1,
            xref = 'paper',
            yref = 'paper',
            text = 'Hover over markers to see more details',
            showarrow = False
)])

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='grouped-bar')


# ### What does this tell us?
# - This chart would be better served as a select event to a country (so that you can see the break down over history). 
# - While as a whole it proves to broad to be useful (clearly Iraq is skewing the ouput of the data), it does prove useful as a starting point to do more things

# In[ ]:


# We are going to remake the above chart with a few more features - namely a slider with time and region distinction

start_yr = 1970

#First create empty dictionary for figure
figure = {
    'data': [],
    'layout': {},
    'frames': []
}

# fill in most of figure layout
figure['layout']['xaxis'] = {'range':[0,5],'title': 'Total Injuries', 'type': 'log'} #Remember log = 10^range
figure['layout']['yaxis'] = {'range':[0,5],'title': 'Total Kills', 'type': 'log'}
figure['layout']['hovermode'] = 'closest'
figure['layout']['title'] = 'Target Victim Break down by Region and Year'
#Initilialize  figure Slider
figure['layout']['sliders'] = {
    'args': [
        'transition', {
            'duration': 400,
            'easing': 'cubic-in-out'
        }
    ],
    'initialValue': str(start_yr),
    'plotlycommand': 'animate',
    'values': gtd.loc[gtd.year >= start_yr, 'year'].unique().astype(str),
    'visible': True}

# Update figure Button menus
figure['layout']['updatemenus'] = [
    {'buttons': [
        {'args': [None, {'frame': {'duration': 500, 'redraw': True},
                         'fromcurrent': True, 'transition': 
                         {'duration': 300, 'easing': 'quadratic-in-out'}}
                 ],
         'label': 'Play',
         'method': 'animate'},
        {'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                           'transition': {'duration': 0}}],
         'label': 'Pause',
         'method': 'animate'}
    ],
     'direction': 'left',
     'pad': {'r': 10, 't': 87},
    'showactive': False,
    'type': 'buttons',
    'x': 0.1,
    'xanchor': 'right',
    'y': 0,
    'yanchor': 'top'}]

#skeleton dictionary of figure slider parameters
sliders_dict = {
    'active': 0,
    'yanchor': 'top',
    'xanchor': 'left',
    'currentvalue': {
        'font': {'size': 20},
        'prefix': 'Year:',
        'visible': True,
        'xanchor': 'right'
    },
    'transition': {'duration': 300, 'easing': 'cubic-in-out'},
    'pad': {'b': 10, 't': 50},
    'len': 0.9,
    'x': 0.1,
    'y': 0,
    'steps': []
}



# In[ ]:


#We create the figure Data portion here

#create empty dictionaries first
region_kills = {region: [] for region in gtd.region.unique()}
region_injury = {region: [] for region in gtd.region.unique()}
region_size = {region: [] for region in gtd.region.unique()}
region_text = {region: [] for region in gtd.region.unique()}

for region in gtd.region.unique(): # Loop through all regions and to create each series for each continent
    kill_dicts = []
    injury_dicts = []
    size_dicts = []
    regionset = gtd[(gtd.region == region) & (gtd.year == start_yr)]
    
    if regionset.shape[0] > 0: #Make sure to take out empty regions with no incidents and maintain memory usage
        #First create empty dictionary to collect all target type sub dictionary totals
        tkill_dict = {e: 0 for e in gtd[gtd.region==region].targtype1.unique()}
        tinjury_dict = {e: 0 for e in gtd[gtd.region==region].targtype1.unique()}
        tsize_dict = {e: 0 for e in gtd[gtd.region==region].targtype1.unique()}
        text_dict = {e:0 for e in gtd[gtd.region==region].targtype1.unique()}
        
        for n in range(3): 
            #Loop to find all the target types and create dictionary of kills/injuries/sizes
            kill_dict = {regionset.groupby(['targtype'+str(n+1)]).nkill.sum().index[i]: e for i, e in enumerate(regionset.groupby(['targtype'+str(n+1)]).nkill.sum())}
            injury_dict = {regionset.groupby(['targtype'+str(n+1)]).nwound.sum().index[i]: e for i, e in enumerate(regionset.groupby(['targtype'+str(n+1)]).nwound.sum())}
            size_dict = {gtd.loc[(gtd.region == region) & (gtd.year == start_yr),'targtype'+str(n+1)].value_counts().index[i]: e for i, e in enumerate(gtd.loc[(gtd.region == region) & (gtd.year == start_yr),'targtype'+str(n+1)].value_counts())}
            #update the dictionary to appropriate lists
            kill_dicts.append(kill_dict)
            injury_dicts.append(injury_dict)
            size_dicts.append(size_dict)
        

        #Next loop through each target type and collect all the values in each dictionary/list
        #Objective here is to create a list for x, y, text descrip and size based on each target/country/year
        for target in regionset.targtype1.unique():#First pull out each target type
            #Next loop through each dictionary in the all the lists and append the total dictionary
            for dic1 in kill_dicts:
                try:
                    tkill_dict[target] += dic1[target]
                except:
                    continue
            for dic2 in injury_dicts:
                try:
                    tinjury_dict[target] += dic2[target]
                except:
                    continue
            for dic3 in size_dicts:
                try:
                    tsize_dict[target] += dic3[target]
                except:
                    continue
                    
            #For each region/year and target type there is a most attacked/country which we can add to text
            most_attack = regionset.loc[regionset.targtype1==target,'attacktype1'].value_counts().sort_values(ascending=False).index[0]
            most_country = regionset.loc[regionset.targtype1==target, 'country'].value_counts().sort_values(ascending=False).index[0]
            most_group = regionset.loc[regionset.targtype1==target,'gname1'].value_counts().sort_values(ascending=False).index[0]
            
            try: #Next we want to create a text variable for each target which shows death percentage
                detperc = round(tkill_dict[target]/(tkill_dict[target]+tinjury_dict[target])*100,2)
            except:
                detperc = 'No Attacks'
            
            #We add all the items from the dictionaries to the text dictionary
            text_dict[target] = 'Target Type <b>{cat}</b> in {reg} for {yr} <br>Total Injuries: <b>{inj}</b>, Total Deaths: <b>{det}</b><br>             Total Incidents: <b>{inc}</b>, Death to Death + Injury Percentage <b>            {detperc}%</b><br>The most common attack: <b>{atk}</b><br>             The most affected country: <b>{con}</b><br>The most active group: <b>{grp}</b>'.format(cat=target,                                                            con=most_country,                                                             det=tkill_dict[target], inj=tinjury_dict[target],                                                            detperc=detperc, atk = most_attack,                                                            inc=tsize_dict[target], reg=region, yr = start_yr, grp=most_group)


            #end of target loop, total target dictionaries should be finished now  
            
        
        #Now collect the target and total dictionaries into regional subdivisions
        region_kills[region] = tkill_dict
        region_injury[region] = tinjury_dict
        region_size[region] = tsize_dict
        region_text[region] = text_dict
        
        #Now that we have all the totals collected, create arrays for the data dictionary

#First loop pulls out each key from the dictionary (each key = region)
for region in region_kills: #Contents is a list with dictionary for each target type
    xlist = []
    ylist = []
    tlist = []
    slist = []
    for target in region_kills[region]: #This loop pulls out each target from the dictionary so we can append to each list
        if region_kills[region][target]>0 or region_injury[region][target]>0: #This way we only capture "live" incidents
            xlist.append(region_injury[region][target])
            ylist.append(region_kills[region][target])
            tlist.append(region_text[region][target])
            slist.append(region_size[region][target])
    #At at end of for loop we should have full list for each region 
    
    if bool(xlist):
        data_dict = {
        'x': xlist,
        'y': ylist,
        'mode': 'markers',
        'text': tlist,
        'marker': {
            'sizemode': 'area',
            'sizeref': 1,
            'size': slist
        },
        'name': region}
        #For each region, we append a series to data
        figure['data'].append(data_dict)


# In[ ]:


#Create the Frames and Slider step for each year
for yr in gtd.loc[gtd.year >= start_yr, 'year'].unique():
    frame = {'data':[], 'name':str(yr)} #Dictionary for each frame - end of each yr loop pass will be added here
    
    #Create a lot of empty dictionaries for each year/region split
    region_yr_kills = {region: [] for region in gtd.region.unique()}
    region_yr_injury = {region: [] for region in gtd.region.unique()}
    region_yr_size = {region: [] for region in gtd.region.unique()}
    region_yr_text = {region: [] for region in gtd.region.unique()}
    
    for region in gtd.region.unique(): # Loop through all regions and to create each series for each continent
        kill_dicts = []
        injury_dicts = []
        size_dicts = []
        regionset = gtd[(gtd.region == region) & (gtd.year == yr)]

        if regionset.shape[0] > 0: #Make sure to take out empty regions with no incidents and maintain memory usage
            #First create empty dictionary to collect all target type sub dictionary totals
            tkill_dict = {e: 0 for e in gtd[gtd.region==region].targtype1.unique()}
            tinjury_dict = {e: 0 for e in gtd[gtd.region==region].targtype1.unique()}
            tsize_dict = {e: 0 for e in gtd[gtd.region==region].targtype1.unique()}
            text_dict = {e:0 for e in gtd[gtd.region==region].targtype1.unique()}

            for n in range(3): 
                #Loop to find all the target types and create dictionary of kills/injuries/sizes
                kill_dict = {regionset.groupby(['targtype'+str(n+1)]).nkill.sum().index[i]: e for i, e in enumerate(regionset.groupby(['targtype'+str(n+1)]).nkill.sum())}
                injury_dict = {regionset.groupby(['targtype'+str(n+1)]).nwound.sum().index[i]: e for i, e in enumerate(regionset.groupby(['targtype'+str(n+1)]).nwound.sum())}
                size_dict = {gtd.loc[(gtd.region == region) & (gtd.year == yr),'targtype'+str(n+1)].value_counts().index[i]: e for i, e in enumerate(gtd.loc[(gtd.region == region) & (gtd.year == yr),'targtype'+str(n+1)].value_counts())}
                #update the dictionary to appropriate lists
                kill_dicts.append(kill_dict)
                injury_dicts.append(injury_dict)
                size_dicts.append(size_dict)


            #Next loop through each target type and collect all the values in each dictionary/list
            #Objective here is to create a list for x, y, text descrip and size based on each target/country/year
            for target in regionset.targtype1.unique():#First pull out each target type
                #Next loop through each dictionary in the all the lists and append the total dictionary
                for dic1 in kill_dicts:
                    try:
                        tkill_dict[target] += dic1[target]
                    except:
                        continue
                for dic2 in injury_dicts:
                    try:
                        tinjury_dict[target] += dic2[target]
                    except:
                        continue
                for dic3 in size_dicts:
                    try:
                        tsize_dict[target] += dic3[target]
                    except:
                        continue

                #For each region/year and target type there is a most attacked/country which we can add to text
                most_attack = regionset.loc[regionset.targtype1==target,'attacktype1'].value_counts().sort_values(ascending=False).index[0]
                most_country = regionset.loc[regionset.targtype1==target, 'country'].value_counts().sort_values(ascending=False).index[0]
                most_group = regionset.loc[regionset.targtype1==target,'gname1'].value_counts().sort_values(ascending=False).index[0]

                try: #Next we want to create a text variable for each target which shows death percentage
                    detperc = round(tkill_dict[target]/(tkill_dict[target]+tinjury_dict[target])*100,2)
                except:
                    detperc = 'No Attacks'

                #We add all the items from the dictionaries to the text dictionary
                text_dict[target] = 'Target Type <b>{cat}</b> in {reg} for {yr} <br>Total Injuries: <b>{inj}</b>, Total Deaths: <b>{det}</b><br>                 Total Incidents: <b>{inc}</b>, Death to Death + Injury Percentage <b>                {detperc}%</b><br>The most common attack: <b>{atk}</b><br>                 The most affected country: <b>{con}</b><br>The most active group: <b>{grp}</b>'.format(cat=target,                                                                con=most_country,                                                                 det=tkill_dict[target], inj=tinjury_dict[target],                                                                detperc=detperc, atk = most_attack,                                                                inc=tsize_dict[target], reg=region, yr = yr, grp = most_group)


                #end of target loop, total target dictionaries should be finished now  


            #Now collect the target and total dictionaries into regional subdivisions
            region_yr_kills[region] = tkill_dict
            region_yr_injury[region] = tinjury_dict
            region_yr_size[region] = tsize_dict
            region_yr_text[region] = text_dict

            #Now that we have all the totals collected, create arrays for the data dictionary
            #First loop pulls out each key from the dictionary (each key = region)
    for region in region_yr_kills: #Contents is a list with dictionary for each target type
        xlist = []
        ylist = []
        tlist = []
        slist = []
        for target in region_yr_kills[region]: #This loop pulls out each target from the dictionary so we can append to each list
            if region_yr_kills[region][target]>0 or region_yr_injury[region][target]>0: #This way we only capture "live" incidents
                xlist.append(region_yr_injury[region][target])
                ylist.append(region_yr_kills[region][target])
                tlist.append(region_yr_text[region][target])
                slist.append(region_yr_size[region][target])
        #At at end of for loop we should have full list for each region 

        if bool(xlist):
            data_dict = {
            'x': xlist,
            'y': ylist,
            'mode': 'markers',
            'text': tlist,
            'marker': {
                'sizemode': 'area',
                'sizeref': 1,
                'size': slist
            },
            'name': region}
            frame['data'].append(data_dict)

    figure['frames'].append(frame)
    
    slider_step = {'args': [
        [str(yr)],
        {'frame': {'duration': 300, 'redraw': True},
         'mode': 'immediate',
         'transition': {'duration': 300}}
    ],'label': str(yr),'method': 'animate'}

    sliders_dict['steps'].append(slider_step)

        

figure['layout']['sliders'] = [sliders_dict]


# In[ ]:


iplot(figure)


# ### What does this tell us?
# - The above chart is an extension of the previous scatter plot outlining Target Victim Types
# - This chart helps paint a much more useful picture on what we gathered from the earlier time series bar plot which showed the top 5 most affected countries.
# - The animation (admittedly - slider needs some more work) provides a helpful perspective in the sudden lul and jump in activity. I recommend looking at the animatino by doule clicking one region and seeing the evolution - though be careful as I need to debug the colorscheme.
#     - For example, a look into Sub Sahara Africa shows that the terrorist activity is characterized by regional political transition/instability (South Africa and Apartheid, Sierre Leone/Burundi's melt down and since the 2000's Somalia's insurgence of Al Shahaab and Nigeria's Boko Harem). Also interesting is the sharp rise in Armed Assault Activity in regions like Nigeria (which is Afria's most populous country)
#     - It's interesting how the Middle East had so little acivity for decades and became such a hotspot for terrorism only in the last decade.
#     - Also interesting to note is the consistent level of terrorism in places like the Phillippines - even though India places like India and Pakistan have drastically more aggregate terrorism,, the more violent outbursts seem to be characterized by in the Phillipines and Thailand.
#         - It would be interesitng to see if there are any underlying parameters that connect the regions

# ### That's it for now!
# 
# The data set is patchy in areas (the dates are a hot mess) but the information can be useful in being used as an output layer for deep learning frameworks (most of the parameters are split into nice categories). However, it's hard to imagine that we can find substantial correlation to predict events with this information alone. Further work will involve testing this to see if there is substantial variability between parameters to deduce a relationship through a neural network. What will be nice is more economic  and political data (though will need a crawl through central economic databases around the world) which documents historical levels and paints a better picture on general stability during the time of attacks. More work likely to follow on this but would love to hear your thought! Also, if any experts on plotly buttons who know how to adjust choropleth maps, please let me know!

# 

# 
