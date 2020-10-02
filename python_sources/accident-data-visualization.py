#!/usr/bin/env python
# coding: utf-8

# ### This kernel is designed for visualization only.
# #### if you find anything wrong in data accessment, please give a direction in comment section. Thank you and if you like this, please upvote

# ### To remenber, which column represent what field, i have rename the column. this is done only for personal resason, nothing else.

# In[ ]:


import pandas as pd
import numpy as np
import plotly
import plotly.offline as plt
import plotly.graph_objs as go
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib as mplt
get_ipython().run_line_magic('matplotlib', 'inline')
plotly.offline.init_notebook_mode(connected=True)

charactestics = pd.read_csv('../input/caracteristics.csv', encoding = "ISO-8859-1", low_memory=False)
holiday = pd.read_csv('../input/holidays.csv', low_memory=False)
places = pd.read_csv('../input/places.csv', low_memory=False)
users = pd.read_csv('../input/users.csv',low_memory=False)
vehicles = pd.read_csv('../input/vehicles.csv', low_memory=False)

Mapping = {
    'Num_Acc':'AccidentID',
    'jour':'Day',
    'mois':'Month',
    'an':'Year',
    'hrmn':'Hour',
    'lum':'LightingCondition',
    'dep':'Department',
    'com':'Municipality',
    'agg':'Localisation',
    'int':'Intersection',
    'atm':'AtmosphericCondition',
    'col':'CollisionType',
    'adr':'Address',
    'gps':'GpsCoding',
    'lat':'Latitude',
    'long':'Longitude',
    'catr':'RoadCategory',
    'voie':'RoadNumber',
    'v1':'RouteNumber',
    'v2':'RouteName',
    'circ':'TrafficType',
    'nbv':'NumberofLanes',
    'vosp':'OuterLane',
    'Prof':'RoadProfile',
    'pr':'HomePRNumber',
    'pr1':'PRDistance',
    'plan':'LaneStructure',
    'lartpc':'CentralLaneWidth',
    'larrout':'OuterLaneWidth',
    'surf':'SurfaceCondition',
    'infra':'Infrastructure',
    'situ':'SituationofAccident',
    'env1':'SchoolPoint',
    'Acc_number':'AccidentID',
    'Num_Veh':'NumberOfVehicles',
    'place':'Place',
    'catu':'UserCatagory',
    'grav':'Severity ',
    'Year_on':'UserYoB',
    'locp':'Locationofpedestrian',
    'actp':'Actionofpedestrian',
    'etatp':'PedestrianGroup',
    'sexe' : 'Sex',
    'secu':'SafetyEquipment'
}

charactestics.rename(index=str, columns=Mapping, inplace=True)
holiday.rename(index=str, columns=Mapping, inplace=True)
places.rename(index=str, columns=Mapping, inplace=True)
users.rename(index=str, columns=Mapping, inplace=True)
vehicles.rename(index=str, columns=Mapping, inplace=True)


# # Data visualization of charactestic dataframe
# ### ----------------------------------------------------------------------------------
# ### charactestics dataframe contain four columns named Year, Month,Day, Hour
# ##### lets convert them to `datetime` format and drop them.
# ###### Note: `format` used is `%Y-%m-%d` ,  hour column is not used and droped. well i didn't know how to pass hour in datetime format

# In[ ]:


charactestics['Year'] = charactestics.Year + 2000
for name in ['Year','Month','Day']:
    charactestics[name] = charactestics[name].astype('str')
    
charactestics['Date'] = charactestics.Year + '-' + charactestics.Month + '-' + charactestics.Day
charactestics['Date'] = pd.to_datetime(charactestics.Date, format='%Y-%m-%d')
charactestics.drop(['Year','Month', 'Day', 'Hour'], axis=1, inplace=True)


# #### lets find out, how much data we are missing in `charactestics`, and plot them.

# In[ ]:


missigData = charactestics.isnull().sum() / len(charactestics)
missigData = missigData[missigData > 0]
missigData.sort_values(inplace=True)
missigData = missigData.to_frame()
missigData.columns = ['count']
missigData.index.names = ['Name']
missigData['Name'] = missigData.index

trace = [go.Bar(x=missigData['Name'], y=missigData['count'])]
layout = go.Layout( title='MissingValues', xaxis=dict(tickangle=-20) )
fig = go.Figure(data=trace, layout=layout)
plt.iplot(fig)


# ### Now we know, which column is missing data.
# ##### before filling those value. lets classify column into `category` or `non category` data

# In[ ]:


for name in ['LightingCondition','Localisation','Intersection','AtmosphericCondition', 'CollisionType', 'Municipality', 'Department']:
    charactestics[name] = charactestics[name].astype('category')


# ### To fill the missing value, method is used to fill most occured category

# In[ ]:


def FillmostOccuredValueToNan(columnName, yesOrNO=True):
    data = charactestics.groupby(by=columnName, as_index=False).count()
    data = data.sort_values(by='AccidentID', ascending=False)
    data = data.head(1)
    fillValue = data[columnName]
    if yesOrNO:
        charactestics[columnName].fillna(fillValue, inplace=True)
    else:
        charactestics[columnName].fillna('UnKnown', inplace=True)
    print('Most occured value in column \'', columnName, '\' is ', fillValue.values)


# ### if the missing value percentage is more than 10%, then it is classified in new category `UnKnown`
# 
# ### From missing value graph, we know that `GpsCoding` is missing more then `50%`.  so give it unknown category
# #### Note: if you have better method, please do tell.
# 

# In[ ]:


FillmostOccuredValueToNan('Municipality')
FillmostOccuredValueToNan('AtmosphericCondition')
FillmostOccuredValueToNan('CollisionType')
FillmostOccuredValueToNan('GpsCoding', False)


# ### For now move on from missing data. we will come to that after a while.
# #### Now let plot pie chart of accident, and find out which situation in each column caused more accident

# In[ ]:


data_1 = charactestics.groupby(by='LightingCondition', as_index=False).count()
data_2 = charactestics.groupby(by='Intersection', as_index=False).count()
data_3 = charactestics.groupby(by='AtmosphericCondition', as_index=False).count()
data_4 = charactestics.groupby(by='CollisionType', as_index=False).count()
data_5 = charactestics.groupby(by='Localisation', as_index=False).count()
data_6 = charactestics.groupby(by='GpsCoding', as_index=False).count()


# In[ ]:


Title = " "
def PlotPiechart(labels, values, columnName):
    fig = {
      "data": [
        {
          "labels": labels,
          "values": values.AccidentID,
          #"domain": {"x": [0, 1]},
          "name": columnName,
          "hoverinfo":"label+percent+name",
          "hole": .6,
          "type": "pie"
        },
         ],
      "layout": {
           # "title":"Percentage of Accident happened in situations : " + columnName,
            "title":Title,
            "annotations": [
                {
                    "font": {
                        "size": 40
                    },
                    "showarrow": False,
                    "text": " ",
                    "x": 5.50,
                    "y": 0.5
                }
            ]
        }
    }
    plt.iplot(fig)


# #### lets make first the details of category in list in each situations

# In[ ]:



LightingConditionLabel = {1:'Full day', 
                          2:'Twilight or dawn', 
                          3:'Night without public lighting',
                          4:'Night with public lighting not lit',
                          5:'Night with public lighting on'}

IntersectionLabel = {0:'Unknown', 
                     1:'Out of intersection', 
                     2:'Intersection in X', 
                     3:'Intersection in T', 
                     4:'Intersection in Y', 
                     5:'Intersection with more than 4 branches',
                     6:'Giratory',
                     7:'Place', 
                     8:'Level crossing',
                     9:'Other intersection'}

AtmosphericConditionLabel = {1:'Normal', 
                             2:'Light rain', 
                             3:'Heavy rain', 
                             4:'Snow - hail', 
                             5:'Fog - smoke',
                             6:'Strong wind - storm', 
                             7:'Dazzling weather', 
                             8:' Cloudy weather', 
                             9:'Other'}

collisionLabel = {1:'Two vehicles - frontal',
                2:'Two vehicles - from the rear',
                3:'Two vehicles - by the side',
                4:'Three vehicles and more - in chain',
                5:'Three or more vehicles - multiple collisions',
                6:'Other collision',
                7:'Without collision'}

LocalisationLabel = {1:'Out of agglomeration',
                     2:'In built-up areas'}

for data in [data_1, data_2, data_3, data_4, data_5]:
    data= data.sort_values(by='AccidentID')

data_6 = data_6.sort_values(by='AccidentID')

data_1.LightingCondition = data_1.LightingCondition.map(LightingConditionLabel)
data_2.Intersection = data_2.Intersection.map(IntersectionLabel)
data_3.AtmosphericCondition = data_3.AtmosphericCondition.map(AtmosphericConditionLabel)
data_4.CollisionType = data_4.CollisionType.map(collisionLabel)
data_5.Localisation = data_5.Localisation.map(LocalisationLabel)


# In[ ]:


Title = "Percentage of Accident happened because of Lighting Condition"
PlotPiechart(data_1.LightingCondition.values, data_1, 'LightingCondition')


# In[ ]:


Title = "Which crossing caused more accident. <br> Area on which department should focus on."
PlotPiechart(data_2.Intersection.values, data_2, 'Intersection')


# In[ ]:


Title = "Percentage of Accident because of Atmosphere"
PlotPiechart(data_3.AtmosphericCondition.values, data_3, 'AtmosphericCondition')


# In[ ]:


Title = "Type of collision happens most often "
PlotPiechart(data_4.CollisionType.values, data_4, 'CollisionType')


# In[ ]:


Title = "Area in which most accident happened"
PlotPiechart(data_5.Localisation.values, data_5, 'Localisation')


# In[ ]:


Title = 'What should is say more.'
PlotPiechart(data_6.GpsCoding.values , data_6, 'GpsCoding')


# ### let see which department area has more accident. 

# In[ ]:


data_7 = charactestics.groupby(by='Department', as_index=False).count()
data_7 = data_7.sort_values(by='AccidentID')


# In[ ]:


data = [go.Bar(
            x=data_7.Department,
            y=data_7.AccidentID,
            text=data_7.AccidentID,
            #textposition = 'auto',
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color=data_7.AccidentID,
                    width=2.5),
            ),
            opacity=1
        )]
layout = go.Layout(
    title='Number of Accident Happened in each Department',
    xaxis= dict( title= 'Department Number'),
     yaxis=dict(title='Count')
)
data = go.Figure(data=data, layout=layout)


# In[ ]:


plt.iplot(data)


# ### let find out how many accident happened in each day from starting to ending date.

# In[ ]:


data_8 = charactestics.groupby(by='Date', as_index=False).count()
data_8 = data_8.sort_values(by='Date')


# In[ ]:


data_8_1 = go.Scatter(x=data_8.Date, y=data_8.AccidentID, name = "Actual Number", line = dict(color = '#17BECF'))
data_8_2 = go.Scatter(x=data_8.Date, y=data_8.AccidentID.rolling(7, min_periods=1).mean(), name = "Rolling Avg of Week", line = dict(color = '#FCEB71'))
Title = 'Time line graph how much accident happened daily.'
layout = dict(
    title=Title,
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(count=12,
                     label='1Y',
                     step='month',
                     stepmode='backward'),
                dict(count=24,
                     label='2Y',
                     step='month',
                     stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(),
        type='date'
    )
)
data = go.Figure(data=[data_8_1, data_8_2], layout=layout)


# In[ ]:


plt.iplot(data)


# ## if you want to visualize number of accident of daily basis.
# #### let make a function of that, and named it `HowMuchDepartmentWasInresponsible`

# In[ ]:


def HowMuchDepartmentWasInresponsible(departmentNumber):
    data = charactestics[charactestics.Department == departmentNumber]
    data = data.groupby(by='Date', as_index=False).count()
    data = data.sort_values(by='Date')
    data_1 = go.Scatter(x=data.Date, y=data.AccidentID, name = "Actual Number", line = dict(color = '#17BECF'))
    data_2 = go.Scatter(x=data.Date, y=data.AccidentID.rolling(7, min_periods=1).mean(), name = "Rolling Avg of Week", line = dict(color = '#FC5B71'))
    data = go.Figure(data=[data_1, data_2], layout=layout)
    plt.iplot(data)


# In[ ]:


Title = "TimeLine of Department 60 <br> Graph showing number of accident happened daily in that department"
HowMuchDepartmentWasInresponsible(60)


# ###### Let find out  weather from column name `AtmosphericCondition`. and it specify accident condition in which it occured

# In[ ]:


def PlotAtomsphericDetails(departmentNumber):
    data = charactestics[charactestics.Department == departmentNumber]
    data = data.groupby(by='Date', as_index=False).count()
    data = data.sort_values(by='Date')
    data_1 = go.Scatter(x=data.Date, y=data.AtmosphericCondition, mode='markers', name = "Actual Number", line = dict(color = '#17BECF'))
    data_2 = go.Scatter(x=data.Date, y=data.AtmosphericCondition.rolling(7, min_periods=1).mean(), name = "Rolling Avg of Week", line = dict(color = '#FC5B71'))
    data = go.Figure(data=[data_1, data_2], layout=layout)
    plt.iplot(data)


# In[ ]:


PlotAtomsphericDetails(50)


# # Now let move on to `places` data

# In[ ]:


missigData = places.isnull().sum() / len(places)
missigData = missigData[missigData > 0]
missigData.sort_values(inplace=True)
missigData = missigData.to_frame()
missigData.columns = ['count']
missigData.index.names = ['Name']
missigData['Name'] = missigData.index

trace = [go.Bar(x=missigData['Name'], y=missigData['count'])]
layout = go.Layout( title='MissingValues', xaxis=dict(tickangle=-20) )
fig = go.Figure(data=trace, layout=layout)
plt.iplot(fig)


# ### Let do the same for this data to fill value of `nan` values,
# #### for now fill values for whose missing percentage is 10%

# In[ ]:


def FillmostOccuredValueToNanValue(columnName, yesOrNO=True):
    data = places.groupby(by=columnName, as_index=False).count()
    data = data.sort_values(by='AccidentID', ascending=False)
    data = data.head(1)
    fillValue = data[columnName]
    if yesOrNO:
        places[columnName].fillna(fillValue, inplace=True)
    else:
        places[columnName].fillna('UnKnown', inplace=True)
    print('Most occured value in column \'', columnName, '\' is ', fillValue.values)


# In[ ]:


for data in ['RoadCategory', 'TrafficType', 'SituationofAccident', 'SurfaceCondition', 'prof', 'LaneStructure', 'SchoolPoint',
            'Infrastructure', 'OuterLane', 'NumberofLanes', 'OuterLaneWidth', 'CentralLaneWidth']:
    FillmostOccuredValueToNanValue(data)


# ### reading the overview of data, `category` type of datacolumn are 
# `RoadCategory`
# `TrafficType` 
# `OuterLane`
# `prof` 
# `SurfaceCondition`
# `LaneStructure` 
# `Infrastructure` 
# `SituationofAccident` 
# `SchoolPoint`

# In[ ]:


for name in ['RoadCategory',
'TrafficType' ,
'OuterLane',
'prof',
'SurfaceCondition',
'LaneStructure',
'Infrastructure', 
'SituationofAccident' ,
'SchoolPoint'] :
    places[name] = places[name].astype('category')


# In[ ]:


data_1 = places.groupby(by='RoadCategory', as_index=False).count()
data_2 = places.groupby(by='TrafficType', as_index=False).count()
data_3 = places.groupby(by='OuterLane', as_index=False).count()
data_4 = places.groupby(by='prof', as_index=False).count()
data_5 = places.groupby(by='SurfaceCondition', as_index=False).count()
data_6 = places.groupby(by='LaneStructure', as_index=False).count()
data_7 = places.groupby(by='Infrastructure', as_index=False).count()
data_8 = places.groupby(by='SituationofAccident', as_index=False).count()
data_9 = places.groupby(by='SchoolPoint', as_index=False).count()


# In[ ]:


RoadCategoryLabel = {1:'Highway',
                    2:'National Road',
                    3:'Departmental Road',
                    4:'Communal Way',
                    5:'Off public network',
                    6:'Parking lot open to public traffic',
                    9:'other'}

TrafficTypelabel = {
    1:'One way',
    2:'Bidirectional',
    3:'Separated carriageways',
    4:'With variable assignment channels'
}

OuterLaneLabel = {
    0:'FootPath',
    1:'Bike path',
    2:'Cycle Bank',
    3:'Reserved channel'
}

profLable = {
    1:'Dish',
    2:'Slope',
    3:'Hilltop',
    4:'Hill bottom'  
}

SurfaceConditionLable = {
    1 :'normal',
    2 : 'wet',
    3 : 'puddles',
    4 : 'flooded',
    5 : 'snow',
    6 : 'mud',
    7 : 'icy',
    8 : 'fat - oil',
    9 : 'other'
}

LaneStructureLabel = {
    1:'Straight part',
    2:'Curved on the left',
    3:'Curved right',
    4:'In "S"'
}

InfrastructureLabel = {
    0:'UnKnown Place',
    1 : 'Underground - tunnel',
    2: 'Bridge - autopont',
    3 : 'Exchanger or connection brace',
    4:'Railway',
    5:'Carrefour arranged',
    6 : 'Pedestrian area',
    7 : 'Toll zone',
}

SituationofAccidentLabel = {
    1 : 'On the road',
    2 : 'On emergency stop band',
    3: 'On the verge',
    4 : 'On the sidewalk',
    5 : 'On bike path',
}


# In[ ]:


data_1.RoadCategory =  data_1.RoadCategory.map(RoadCategoryLabel)
data_2.TrafficType =  data_2.TrafficType.map(TrafficTypelabel)
data_3.OuterLane = data_3.OuterLane.map(OuterLaneLabel)
data_4.prof =  data_4.prof.map(profLable)
data_5.SurfaceCondition =  data_5.SurfaceCondition.map(SurfaceConditionLable)
data_6.LaneStructure =  data_6.LaneStructure.map(LaneStructureLabel)
data_7.Infrastructure = data_7.Infrastructure.map(InfrastructureLabel)
data_8.SituationofAccident = data_8.SituationofAccident.map(SituationofAccidentLabel)


# In[ ]:


Title = "Which type of Road is more dangerous based on number of accident happened "
PlotPiechart(data_1.RoadCategory.values , data_1, 'RoadCategory')


# In[ ]:


Title = "Which Type of traffic, department should focus more. <br> based on total number of accident"
PlotPiechart(data_2.TrafficType.values , data_2, 'TrafficType')


# In[ ]:


Title = "Type of outerLane is present in data."
PlotPiechart(data_3.OuterLane.values , data_3, 'OuterLane')


# In[ ]:


Title = 'Type of road, one should drive carefully'
PlotPiechart(data_4.prof.values , data_4, 'prof')


# In[ ]:


Title = 'Condition of road distruction <br> is this a cause of accident'
PlotPiechart(data_5.SurfaceCondition.values , data_5, 'SurfaceCondition')


# In[ ]:


Title = "type of lane"
PlotPiechart(data_6.LaneStructure.values , data_6, 'LaneStructure')


# In[ ]:


Title = "Infrastructure condition in which accident happened "
PlotPiechart(data_7.Infrastructure.values , data_7, 'Infrastructure')


# In[ ]:


Title = "Where most of accident happended"
PlotPiechart(data_8.SituationofAccident.values , data_8, 'SituationofAccident')


# In[ ]:


Title = 'School point during accident happened'
PlotPiechart(data_9.SchoolPoint.values , data_9, 'SchoolPoint')


# In[ ]:


missigData = users.isnull().sum() / len(users)
missigData = missigData[missigData > 0]
missigData.sort_values(inplace=True)
missigData = missigData.to_frame()
missigData.columns = ['count']
missigData.index.names = ['Name']
missigData['Name'] = missigData.index

trace = [go.Bar(x=missigData['Name'], y=missigData['count'])]
layout = go.Layout( title='MissingValues', xaxis=dict(tickangle=-20) )
fig = go.Figure(data=trace, layout=layout)
plt.iplot(fig)


# In[ ]:


users.head()


# In[ ]:


place_1 = users.groupby('Place', as_index=False).count()
users_1 = users.groupby('UserCatagory', as_index=False).count()
#serev_1 = users.groupby('Severity', as_index=False).count()
sexes_1 = users.groupby('Sex', as_index=False).count()
safty_1 = users.groupby('SafetyEquipment', as_index=False).count()
locps_1 = users.groupby('Locationofpedestrian', as_index=False).count()
trajet_1 = users.groupby('trajet', as_index=False).count()


# In[ ]:


trace_1 = go.Bar(x=place_1.Place, y=place_1.AccidentID/place_1.AccidentID.sum(), name='Places')
trace_2 = go.Bar(x=users_1.UserCatagory, y=users_1.AccidentID/users_1.AccidentID.sum(), name='UserCategory')
trace_3 = go.Bar(x=sexes_1.Sex, y=sexes_1.AccidentID/sexes_1.AccidentID.sum(), name='Sex')
trace_4 = go.Bar(x=safty_1.SafetyEquipment, y=safty_1.AccidentID/safty_1.AccidentID.sum(), name='SafetyEquipment')
trace_5 = go.Bar(x=locps_1.Locationofpedestrian, y=locps_1.AccidentID/locps_1.AccidentID.sum(), name='LocationofPredistion')
trace_6 = go.Bar(x=trajet_1.trajet, y=trajet_1.AccidentID/trajet_1.AccidentID.sum(), name='Trajet')


# In[ ]:


from plotly import tools
fig = tools.make_subplots(rows=3, cols=2, shared_yaxes=True)

fig.append_trace(trace_1, 1,1)
fig.append_trace(trace_2, 1,2)
fig.append_trace(trace_3, 2,1)
fig.append_trace(trace_4, 2,2)
fig.append_trace(trace_5, 3,1)
fig.append_trace(trace_6, 3,2)

fig['layout']['xaxis1'].update(title='Distribution of accident based on Places')
fig['layout']['xaxis2'].update(title='Distribution of accident based on UserCatagory')
fig['layout']['xaxis3'].update(title='Distribution of accident based on Sex')
fig['layout']['xaxis4'].update(title='Distribution of accident based on SafetyEquipment')
fig['layout']['xaxis5'].update(title='Distribution of accident based on LocationofPredistion')
fig['layout']['xaxis6'].update(title='Distribution of accident based on Trajet')

fig['layout']['yaxis1'].update(title='%')
fig['layout']['yaxis2'].update(title='%')
fig['layout']['yaxis3'].update(title='%')
fig['layout']['yaxis4'].update(title='%')
fig['layout']['yaxis5'].update(title='%')
fig['layout']['yaxis6'].update(title='%')

fig['layout'].update(height=1000, width=900, title='overview of users data')
plt.iplot(fig)


# In[ ]:




