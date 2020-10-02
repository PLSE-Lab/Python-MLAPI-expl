#!/usr/bin/env python
# coding: utf-8

# Importing the Libraries

# In[ ]:


import numpy as np
import pandas as pd 


# Importing the Libraries for Data Visualization

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
from plotly.offline import iplot, plot, download_plotlyjs, init_notebook_mode
from plotly.graph_objs import graph_objs as go
import cufflinks as cf


# Initializing the parameters

# In[ ]:


init_notebook_mode(connected = True)
cf.go_offline(connected = True)


# Importing the DataSet

# In[ ]:


import os
print(os.listdir("../input/fifa19"))


# In[ ]:


fifa_data = pd.read_csv('../input/fifa19/data.csv')


# In[ ]:


fifa_data.columns


# Finding the Players with highest Growth Potential

# In[ ]:


fifa_data['Growth'] = fifa_data['Potential'] - fifa_data['Overall']


# Finding and Removing the Null Values

# In[ ]:


fifa_data.isnull().sum()[fifa_data.isnull().sum() > 8000]


# In[ ]:


fifa_data.drop('Loaned From', axis = 1, inplace = True)


# In[ ]:


fifa_data.drop('Release Clause', axis = 1, inplace = True)


# Data Cleaning

# In[ ]:


fifa_data.drop(['Club', 'Jersey Number', 'Contract Valid Until', 'Joined'], axis = 1, inplace = True)


# In[ ]:


fifa_data.drop(['Photo', 'Flag', 'Club Logo', 'Real Face'], axis = 1, inplace = True)


# In[ ]:


fifa_data.drop(index = fifa_data[fifa_data['Preferred Foot'].isna()].index, inplace = True)


# In[ ]:


fifa_data.drop(index = fifa_data[fifa_data['Position'].isna()].index, inplace = True)


# In[ ]:


fifa_data.drop(index = fifa_data[fifa_data['RB'].isna()].index, inplace = True )


# Converting the Data Points of the Value and Wage Column

# In[ ]:


def convertValue(value) :
    if value[-1] == 'M' :
        value = value[1:-1]
        value = float(value) * 1000000
        return value
    
    if value[-1] == 'K' :
        value = value[1:-1]
        value = float(value) * 1000
        return value


# In[ ]:


fifa_data['Wage'] = fifa_data['Wage'].apply(lambda x : convertValue(x))


# In[ ]:


fifa_data['Value'] = fifa_data['Value'].apply(lambda x : convertValue(x))


# Converting the Data Values into ML understandable format

# In[ ]:


fifa_data.select_dtypes(include = object).columns


# Working on the 'Body Type' Column

# In[ ]:


fifa_data['Body Type'][fifa_data['Body Type'] == 'Messi'] = 'Normal'
fifa_data['Body Type'][fifa_data['Body Type'] == 'C. Ronaldo'] = 'Normal'
fifa_data['Body Type'][fifa_data['Body Type'] == 'Neymar'] = 'Lean'
fifa_data['Body Type'][fifa_data['Body Type'] == 'PLAYER_BODY_TYPE_25'] = 'Normal'
fifa_data['Body Type'][fifa_data['Body Type'] == 'Shaqiri'] = 'Stocky'
fifa_data['Body Type'][fifa_data['Body Type'] == 'Akinfenwa'] = 'Stocky'


# Converting the Height and Weight Column

# In[ ]:


def convertWeight(weight) :
    weight = weight[0:3]
    return weight


# In[ ]:


def convertHeight(height) :
    height = height.split("'")
    height = float(height[0]) * 30.48 + float(height[1]) * 2.54 
    
    return height


# In[ ]:


fifa_data['Weight'] = fifa_data['Weight'].apply(lambda x : convertWeight(x))
fifa_data['Height'] = fifa_data['Height'].apply(lambda x : convertHeight(x))


# Converting the Position Column

# In[ ]:


def convertPosition(val) :
    
    if val == 'RF' or val == 'ST' or val == 'LW' or val == 'LF' or val == 'RS' or val == 'LS' or val == 'RM' or val == 'LM' or val == 'RW' or val == 'CF' :
        return 'Forward'
    
    elif val == 'GK' :
        return 'GoalKeeper'
    
    elif val == 'RCM' or val == 'LCM' or val == 'LDM' or val == 'CAM' or val == 'CDM' or val == 'LAM' or val == 'RDM' or val == 'CM' or val == 'RAM' :
        return 'MidFielder'
    
    return 'Defender'


# In[ ]:


fifa_data['Position'] = fifa_data['Position'].apply(lambda x : convertPosition(x))


# Converting all the Rating Columns

# In[ ]:


temp_columns =['LS', 'ST', 'RS', 'LW', 'LF', 'CF',
               'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB',
               'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']


# In[ ]:


def convertRatings(rating) :
    rating = rating.split('+')
    rating = int(rating[0]) + int(rating[1])
    return rating


# In[ ]:


for column in temp_columns :
    fifa_data[column] = fifa_data[column].apply(lambda x : convertRatings(x))


# Function to get the top 10 Players Feature Wise

# In[ ]:


def getTOP10(feature) :
   return fifa_data.sort_values(by = feature, ascending = False)[['Name', feature]].head(10)


# In[ ]:


getTOP10('RW')


# Plotting the Geographical Map for the count of players

# In[ ]:


Player_Count = fifa_data.groupby('Nationality').size().reset_index()


# In[ ]:


Player_Count.columns = ['Country', 'Count']


# In[ ]:


data1 = go.Choropleth(locationmode = 'country names', locations = Player_Count['Country'],
                     z =  Player_Count['Count'], colorscale = 'oranges', )


# In[ ]:


layout1 = go.Layout(title = 'Players Count Per Country') 


# In[ ]:


graphPlayerCountPerCountry = go.Figure(data = data1, layout = layout1)


# In[ ]:


graphPlayerCountPerCountry


# Plotting the Graph to show the Age Distribution among the Players

# In[ ]:


graphPlayerAge = fifa_data['Age'].iplot(kind = 'histogram', title = 'Player Age Distribution', xTitle = 'Age',
                                        yTitle = 'Count', theme = 'pearl', )


# Plotting the Graph to show the Height and Weight Distribution among the Players

# In[ ]:


plt.figure(figsize =(30,15))
graphPlayerHeightWeight = sns.boxplot(x = 'Weight', y = 'Height', data = fifa_data, )


# Plotting the Graph to show the Position Distribution of the Players

# In[ ]:


data2 = go.Pie(labels = fifa_data['Position'].value_counts().index.values, values = fifa_data['Position'].value_counts().values, 
               hole = 0.3)


# In[ ]:


layout2 = go.Layout(title = 'Player Position Distribution')


# In[ ]:


graphPlayerPosition = go.Figure(data = data2, layout = layout2)


# In[ ]:


graphPlayerPosition


# Plotting the Graph to show the Overall Rating Distribution among the Players

# In[ ]:


graphPlayerOverall = fifa_data['Overall'].iplot(kind = 'histogram', title = 'Player Overall Distribution', 
                                                xTitle = 'Overall Rating', yTitle ='Count')


# Plotting the Graph to show the Preferred Foot Distribution among the Players

# In[ ]:


data3 = go.Pie(labels = fifa_data['Preferred Foot'].value_counts().index.values, values = fifa_data['Preferred Foot'].value_counts().values, 
               hole = 0.3)


# In[ ]:


layout3 = go.Layout(title = 'Player Preferred Foot Distribution')


# In[ ]:


graphPlayerPreferredFoot = go.Figure(data = data3, layout = layout3)


# In[ ]:


graphPlayerPreferredFoot


# Applying Machine Learning

# Predicting the Overall of the Player Using Linear Regression Model

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


fifa_data['Value'] = fifa_data['Value'].apply(lambda x  : x/1000000)


# In[ ]:


x = fifa_data[['Potential', 'Value', 'LS', 'ST', 'RS', 'LW',
       'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM',
       'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB',
       'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',
       'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',
       'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',
       'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',
       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
       'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'Growth']]


# In[ ]:


y = fifa_data['Overall']


# In[ ]:


x.fillna(value = 0, inplace = True)
y.fillna(value = 0, inplace = True)


# Splitting the Data into Training and Test Set

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 101)


# Importing the GridSearchCV to find out the optimal value of the parameters of the model

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


paramlist = {'n_jobs' : [0.1 , 1, 10, 100 ]}


# In[ ]:


gridSearch = GridSearchCV(estimator = LinearRegression(), param_grid = paramlist, verbose = 5)


# In[ ]:


gridSearch.fit(x_train,y_train)


# In[ ]:


gridSearch.best_params_


# In[ ]:


model = LinearRegression(n_jobs = 0.1)


# Training the Model

# In[ ]:


model.fit(x_train, y_train)


# In[ ]:


predictions = model.predict(x_test)


# Import the Performance metrics

# In[ ]:


from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[ ]:


mean_absolute_error(y_true = y_test, y_pred = predictions)


# In[ ]:


mean_squared_error(y_true = y_test, y_pred = predictions)


# In[ ]:




