#!/usr/bin/env python
# coding: utf-8

# <img src='https://i0.wp.com/glitched.africa/wp-content/uploads/2019/02/blog-fifa-19-cover-new-big.jpg?fit=1920%2C1080&ssl=1' />

# FIFA 19 is a football simulation video game developed by EA Vancouver as part of Electronic Arts' FIFA series. <br><br>
# 
# It was announced on 6th June 2018 in a press conference and was released on 28th September 2018 for PlayStation 3, PlayStation 4, Xbox 360, Xbox One, Nintendo Switch, and Microsoft Windows. <br><br>
# 
# It is the 26th installment in the FIFA series. The game features the UEFA club competitions for the first time, including the UEFA Champions League. <br><br>
# 
# This notebook was created to provide information about various aspects of players using interactive plotly visualizations. It also contains machine learning models which is used to classify players based on their position and predict the Overall rating of a player based on the features present in the dataset.<br><br>
# 
# The FIFA 19 dataset contains over 18,000 rows with 89 columns. Dataset can be found here - https://www.kaggle.com/karangadiya/fifa19
# <br><br><br>
# 
# ---

# ### Import required libraries

# In[ ]:


import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


init_notebook_mode(connected=True)
cf.go_offline()


# <br><br>
# ### Read the data

# In[ ]:


data = pd.read_csv('../input/data.csv')


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


print('Number of Categorical Columns: ', len(data.select_dtypes(include=object).columns))
print('Number of Numerical Columns: ', len(data.select_dtypes(exclude=object).columns))


# We have a dataset with 18,207 rows which includes 45 categorical features and 44 numerical features.

# <br><br>
# ### Dealing with unnecessary features and missing values

# In[ ]:


#Dropping columns which are of very less significance.
data.drop(columns=['Unnamed: 0', 'ID', 'Photo', 'Flag', 'Club Logo', 'Special', 'Real Face', 'Release Clause',
                   'Joined', 'Contract Valid Until'], inplace=True)


# In[ ]:


#Check for missing values in columns where missing values is more than half of the total number of values.
data.isnull().sum()[data.isnull().sum() >= 9000]


# In[ ]:


#Dropping column based on above condition
data.drop(columns=['Loaned From'], inplace=True)


# In[ ]:


data.isnull().sum()


# Still a lot of missing values to deal with. Let us fill in these missing values appropriately and/or drop columns which are not required. <br><br>

# In[ ]:


#Players who are not part of any club.
data['Club'].fillna(value='No Club', inplace=True)


# In[ ]:


data[data['Preferred Foot'].isna()].head()


# In[ ]:


#Full of NaN values for many features, so drop.
data.drop(index=data[data['Preferred Foot'].isna()].index, inplace=True)


# In[ ]:


data[data['Position'].isna()][['Name', 'Nationality', 'LS', 'ST','RS', 'LW', 'LF', 'CF', 'RF', 'RW',
                              'LAM', 'CAM', 'RAM', 'LM', 'LCM','CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 
                              'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']].head()


# In[ ]:


#Can fill in position manually but LS, RS, CF, etc. features have no values, so drop them.
data.drop(index=data[data['Position'].isna()].index, inplace=True)


# In[ ]:


#Checking the number of missing values in the remaining columns.
data.isnull().sum()[data.isnull().sum() > 0]


# In[ ]:


len(data[data['Position'] == 'GK'])


# Looks like the above features are not set for Goalkeepers. We cannot drop them as it would remove all the goal keepers from our dataset. Instead we will fill these values with 0.

# In[ ]:


data.fillna(value=0, inplace=True)


# In[ ]:


data.isnull().sum().sum()


# Our dataset does not contain any missing values.<br><br><br>

# ### Converting categorical features to appropriate numerical features

# In[ ]:


data.select_dtypes(include=object).columns


# In[ ]:


#Function to convert value and wage of the player.
def currencyConverter(val):
    if val[-1] == 'M':
        val = val[1:-1]
        val = float(val) * 1000000
        return val
        
    elif val[-1] == 'K':
        val = val[1:-1]
        val = float(val) * 1000
        return val
    
    else:
        return 0


# In[ ]:


data['Value in Pounds'] = data['Value'].apply(currencyConverter)
data['Wage in Pounds'] = data['Wage'].apply(currencyConverter)

data.drop(columns=['Value', 'Wage'], inplace=True)

data.head()


# Value and Wage have been converted.
# <br><br><br>

# In[ ]:


data[['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM',
       'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM',
       'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']].head()


# In[ ]:


#Function to convert skill rating at each position.
def skillConverter(val):
    if type(val) == str:
        s1 = val[0:2]
        s2 = val[-1]
        val = int(s1) + int(s2)
        return val
    
    else:
        return val


# In[ ]:


skill_columns = ['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM',
       'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM',
       'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']
                      
for col in skill_columns:
    data[col] = data[col].apply(skillConverter)


# In[ ]:


data[['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM',
       'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM',
       'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']].head()


# Converted to numeric.
# <br><br>

# In[ ]:


data[['Height', 'Weight']].head()


# In[ ]:


def height_converter(val):
    f = val.split("'")[0]
    i = val.split("'")[1]
    h = (int(f) * 30.48) + (int(i)*2.54)
    return h

def weight_converter(val):
    w = int(val.split('lbs')[0])
    return w


# In[ ]:


data['Height in Cms'] = data['Height'].apply(height_converter)
data['Weight in Pounds'] = data['Weight'].apply(weight_converter)

data.drop(columns=['Height', 'Weight'], inplace=True)
data[['Height in Cms', 'Weight in Pounds']].head()


# <br><br><br>
# For the remaining columns - **Work Rate, Body Type, Position**, we will not be converting them to numerical features right now. Here, we ensure these features have appropriate values and they will be converted to numerical features when feeding this data to our machine learning models.

# In[ ]:


data['Work Rate'].unique()


# In[ ]:


data['Body Type'].unique()


# In[ ]:


data['Body Type'][data['Body Type'] == 'Messi'] = 'Lean'
data['Body Type'][data['Body Type'] == 'C. Ronaldo'] = 'Normal'
data['Body Type'][data['Body Type'] == 'Neymar'] = 'Lean'
data['Body Type'][data['Body Type'] == 'Courtois'] = 'Lean'
#PLAYER_BODY_TYPE_25 is the body type of Mohammed Salah who has a Normal body type.
data['Body Type'][data['Body Type'] == 'PLAYER_BODY_TYPE_25'] = 'Normal'
data['Body Type'][data['Body Type'] == 'Shaqiri'] = 'Stocky'
data['Body Type'][data['Body Type'] == 'Akinfenwa'] = 'Stocky'


# In[ ]:


print(data['Position'].unique())
print(data['Position'].nunique())


# Let us simplify the above positions into 4 simple categories of - **F**orwards, **M**idfielders, **D**efenders and **G**oal**K**eepers

# In[ ]:


def position_simplifier(val):
    
    if val == 'RF' or val == 'ST' or val == 'LF' or val == 'RS' or val == 'LS' or val == 'CF':
        val = 'F'
        return val
        
    elif val == 'LW' or val == 'RCM' or val == 'LCM' or val == 'LDM' or val == 'CAM' or val == 'CDM' or val == 'RM'          or val == 'LAM' or val == 'LM' or val == 'RDM' or val == 'RW' or val == 'CM' or val == 'RAM':
        val = 'M'
        return val

    
    elif val == 'RCB' or val == 'CB' or val == 'LCB' or val == 'LB' or val == 'RB' or val == 'RWB' or val == 'LWB':
        val = 'D'
        return val
    
    else:
        return val
        


# In[ ]:


data['Position'] = data['Position'].apply(position_simplifier)
data['Position'].value_counts()


# <br><br><br>

# ### Visualizations
# Plotting a few visualizations which give us more information about the dataset.

# **Player distribution across Countries**

# In[ ]:


df_nations = data.groupby(by='Nationality').size().reset_index()
df_nations.columns = ['Nation', 'Count']


# In[ ]:


df_nations[(df_nations['Nation'] == 'England') | (df_nations['Nation'] == 'Wales') 
           | (df_nations['Nation'] == 'Scotland') | (df_nations['Nation'] == 'Northern Ireland') ]


# In[ ]:


df_temp = pd.DataFrame(data= [['United Kingdom', 2148]], columns=['Nation', 'Count'])
df_nations = df_nations.append(df_temp, ignore_index=True)
df_nations.tail()


# Adding values of England, Northern Ireland, Scotland and Wales under United Kingdom as our choropleth map considers the following countries as a whole which is included in the United Kingdom.<br>
# 
# Hover over the map to confirm the same.

# In[ ]:


trace2 = dict(type='choropleth',
              locations=df_nations['Nation'],
              z=df_nations['Count'],
              locationmode='country names',
              colorscale='Portland'
             )

layout = go.Layout(title='<b>Number of Players in each Country</b>',
                   geo=dict(showocean=True,
                            oceancolor='#AEDFDF',
                            projection=dict(type='natural earth'),
                        )
                  )

fig = go.Figure(data=[trace2], layout=layout)
py.iplot(fig)


# Most players are from European and South American countries.<br><br>
# Top 5 countries -
# 1. Engalnd - 1657
# 2. Germany - 1195
# 3. Spain - 1071
# 4. Argentina - 936
# 5. France - 911
# <br><br>
# Hover over the red spot on the map to get the value of United Kingdom which includes England, Scotland, Northern Ireland and Wales.
# <br><br><br>

# <br><br>
# **Age Distribution of Players**

# In[ ]:


trace1 = go.Histogram(x=data['Age'], nbinsx=55, opacity=0.7)

layout = go.Layout(title='<b>Players Age Distribution<b>',
                   xaxis=dict(title='<b><i>Age</b></i>'),
                   yaxis=dict(title='<b><i>Count</b></i>'),
                  )

fig = go.Figure(data=[trace1], layout=layout)
py.iplot(fig)


# Most players age lie in the range of 19 to 29.<br><br><br>

# **Player Height and Weight distibution**

# In[ ]:


fig = tools.make_subplots(rows=1, cols=2)

trace7a = go.Histogram(x=data['Height in Cms'], nbinsx=25, opacity=0.7, name='Height in cms')
trace7b = go.Histogram(x=data['Weight in Pounds'], nbinsx=30, opacity=0.7, name='Weight in Pounds')

fig.append_trace(trace7a, 1,1)
fig.append_trace(trace7b, 1,2)

fig['layout'].update(title='<b>Height & Weight Distribution</b>',                      xaxis=dict(automargin=True),
                     yaxis=dict(title='<b><i>Count</b></i>')
                    )
py.iplot(fig)


# Majority of the players height lie in the range of 175cms to 190cms.<br>
# Majority of the players weight lie in the range of 150lbs to 174lbs.<br><br><br><br>

# **Number of players in each Position**

# In[ ]:


trace6 = go.Pie(values=data['Position'].value_counts().values,
                labels=data['Position'].value_counts().index.values,
                hole=0.3
               )
 

layout = go.Layout(title='<b>Distribution of Players Position-Wise</b>')

fig = go.Figure(data=[trace6], layout=layout)
py.iplot(fig)


# Number of players in each position -
# 1. Midfielders = 7589
# 2. Defenders = 5866
# 3. Forwards = 2667
# 4. Goal Keepers = 2025
# <br><br><br>

# **Are the players Right Footed  or Left Footed?**

# In[ ]:


trace3 = go.Pie(values=data['Preferred Foot'].value_counts().values,
                 labels=data['Preferred Foot'].value_counts().index.values,
                 hole=0.3
                )
 

layout = go.Layout(title='<b>Preferred Foot</b>')

fig = go.Figure(data=[trace3], layout=layout)
py.iplot(fig)


# Majority of the players prefer their Right foot.<br><br>
# Right Foot = 13,938<br>
# Left Foot = 4,209
# 
# <br><br><br><br>

# **Does Overall depend on Work Rate?**

# In[ ]:


trace4 = go.Violin(x=data['Work Rate'],
                y=data['Overall']
               )

layout = go.Layout(title='<b>Work Rate vs Overall</b>',
                   xaxis=dict(title='<b><i>Work Rate</b></i>'),
                   yaxis=dict(title='<b><i>Overall</b></i>')
                  )

fig = go.Figure(data=[trace4], layout=layout)
py.iplot(fig)


# From the above visualization, it is clear that players have almost similar Overall ratings across different Work Rates.
# <br><br><br><br>

# **Player Attributes based on Position**

# In[ ]:


#We are choosing 6 attributes here. We are grouping the data by Position and finding the average of our 6 attributes.
df_skills = data.groupby(by='Position')['Crossing', 'Finishing', 'FKAccuracy', 
                            'StandingTackle', 'Marking', 'Interceptions'].mean().reset_index()


# In[ ]:


trace5a = go.Scatterpolar(theta=['Crossing', 'Finishing', 'FKAccuracy', 
                                 'StandingTackle', 'Marking', 'Interceptions',
                                 'Crossing'
                                ],
                          r=df_skills[df_skills['Position'] == 'F'][['Crossing', 'Finishing', 'FKAccuracy', 
                                                                     'StandingTackle', 'Marking', 'Interceptions',
                                                                     'Crossing'
                                                                    ]].values[0],
                          fill='toself',
                          name='Forwards'
                         )

trace5b = go.Scatterpolar(theta=['Crossing', 'Finishing', 'FKAccuracy', 
                                 'StandingTackle', 'Marking', 'Interceptions',
                                 'Crossing'
                                ],
                          r=df_skills[df_skills['Position'] == 'M'][['Crossing', 'Finishing', 'FKAccuracy', 
                                                                     'StandingTackle', 'Marking', 'Interceptions',
                                                                     'Crossing'
                                                                    ]].values[0],
                          fill='toself',
                          name='Midfielders'
                         )

trace5c = go.Scatterpolar(theta=['Crossing', 'Finishing', 'FKAccuracy', 
                                 'StandingTackle', 'Marking', 'Interceptions',
                                 'Crossing'
                                ],
                          r=df_skills[df_skills['Position'] == 'D'][['Crossing', 'Finishing', 'FKAccuracy', 
                                                                     'StandingTackle', 'Marking', 'Interceptions',
                                                                     'Crossing'
                                                                    ]].values[0],
                          fill='toself',
                          name='Defenders'
                         )


trace5d = go.Scatterpolar(theta=['Crossing', 'Finishing', 'FKAccuracy', 
                                 'StandingTackle', 'Marking', 'Interceptions',
                                 'Crossing'
                                ],
                          r=df_skills[df_skills['Position'] == 'GK'][['Crossing', 'Finishing', 'FKAccuracy', 
                                                                     'StandingTackle', 'Marking', 'Interceptions',
                                                                     'Crossing'
                                                                    ]].values[0],
                          fill='toself',
                          name='Goal Keepers'
                         )

layout = go.Layout(polar=dict(radialaxis=dict(visible=True,
                                              range=[0, 100]
                                             )
                             
                             ),
                   showlegend=True,
                   title='<b>Attributes by Position</b>'
                  )

fig = go.Figure(data=[trace5a, trace5b, trace5c, trace5d], layout=layout)
py.iplot(fig)


# No surprise with Defenders scoring the highest in defending attributes such as StandingTackle, Marking and Interceptions.<br><br>
# Midfielders seem like all rounders. They are good at everything but excel in Crossing and FKAccuracy.<br><br>
# Forwards' main job is to score goals. Hence, it is expected they score the highest in Finishing.<br><br><br><br>
# 

# ### Classification based on Position

# **Import required libraries**

# In[ ]:


from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, r2_score


# In[ ]:


#Transforming categorical feature into numeric.
#Goal Keeper = 0
#Defender = 1
#Midfielder = 2
#Forward = 3
def pos_numeric(val):
    if val == 'GK':
        return 0
    elif val == 'D':
        return 1
    elif val == 'M':
        return 2
    else:
        return 3
    
data['Position'] = data['Position'].apply(pos_numeric)


# In[ ]:


df_pos = data.copy()

#Dropping unnecessary columns
df_pos.drop(columns=['Name', 'Nationality', 'Club'], inplace=True)


# In[ ]:


X = df_pos.drop(columns=['Position'])
X = pd.get_dummies(X)
y = df_pos['Position']


# In[ ]:


#Splitting dataset into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# **Logisitic Regression model**

# In[ ]:


logmodel = LogisticRegression()


# In[ ]:


logmodel.fit(X_train, y_train)


# In[ ]:


prediction = logmodel.predict(X_test)


# In[ ]:


print(classification_report(y_test, prediction))
print('\n')
print(confusion_matrix(y_test, prediction))
print('\n')
print('Accuracy Score: ', accuracy_score(y_test, prediction))


# Just by dropping a few columns and without any fine tuning, the Logisitic Regression model gets a pretty good accuracy score.<br><br>
# Notice the 100% accuracy while classifying Goal Keepers. This is expected as the features such as LS, RS, CF, etc. is equal to 0 for all the goal keepers in our dataset.<br><br>
# Let us try to increase the accuracy in classifying other positions.
# <br><br><br><br>

# **Correlation between Position and other features**

# In[ ]:


df_pos.corr().abs()['Position'].sort_values(ascending=False)


# Notice how similar features such as LS, RS and CAM, LAM, RAM and so on have similar correlation. Let us combine such features into individual single features.

# In[ ]:


df_pos['Frw'] = (df_pos['RF'] + df_pos['ST'] + df_pos['LF'] + df_pos['RS'] + df_pos['LS'] + df_pos['CF']) / 6

df_pos['Mid'] = (df_pos['LW'] + df_pos['RCM'] + df_pos['LCM'] + df_pos['LDM'] + df_pos['CAM'] + df_pos['CDM'] +                 df_pos['RM'] + df_pos['LAM'] + df_pos['LM'] + df_pos['RDM'] + df_pos['RW'] + df_pos['CM'] + df_pos['RAM'])                /13

df_pos['Def'] = (df_pos['RCB'] + df_pos['CB'] + df_pos['LCB'] + df_pos['LB'] + df_pos['RB'] + df_pos['RWB']                 + df_pos['LWB']) / 7

df_pos['Gk'] = (df_pos['GKDiving'] + df_pos['GKHandling'] + df_pos['GKKicking'] + df_pos['GKPositioning']               + df_pos['GKReflexes']) / 5

df_pos.drop(columns=['RF', 'ST', 'LW', 'RCM', 'LF', 'RS', 'RCB', 'LCM', 'CB', 'LDM', 'CAM', 'CDM',
                     'LS', 'LCB', 'RM', 'LAM', 'LM', 'LB', 'RDM', 'RW', 'CM', 'RB', 'RAM', 'CF', 'RWB', 'LWB',
                     'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes'
                    ], inplace=True)


# In[ ]:


print('Correlation with Position: ', df_pos.corr().abs()['Position'].sort_values(ascending=False).index, '\n')
print('Categorical columns in dataset: ', df_pos.select_dtypes(include=object).columns, '\n')
print('Number of features in dataset: ', len(df_pos.columns))


# By combining certain features, we have brought down the nubmber of features from 75 to 48.<br><br><br>

# **Dropping columns with lower correlation**

# In[ ]:


#df_pos = data.copy()
#Dropping Preferred Foot column too as it does not play a significant role in classifying the position of the player. 
df_pos.drop(columns=['StandingTackle', 'Potential', 'Age', 'Value in Pounds', 
                     'Jumping', 'Jersey Number', 'Wage in Pounds', 'Overall', 'Marking',
                     'International Reputation', 'Strength', 'Preferred Foot'], inplace=True)


# In[ ]:


X = df_pos.drop(columns=['Position'])
X = pd.get_dummies(X)
y = df_pos['Position']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# **Logistic Regression model after dropping lower correlated features**

# In[ ]:


logmodel.fit(X_train, y_train)


# In[ ]:


prediction = logmodel.predict(X_test)


# In[ ]:


print(classification_report(y_test, prediction))
print('\n')
print(confusion_matrix(y_test, prediction))
print('\n')
print('Accuracy Score: ', accuracy_score(y_test, prediction))


# The new Logisitic Regression model does a better job than the previous model. <br><br>
# It is able to better classify Defenders and Midfielders. However, accuracy of classifying Forwards goes down.
# <br><br><br><br><br>

# **Removing outliers**

# In[ ]:


sns.set_style(style='darkgrid')
plt.rcParams['figure.figsize'] = 12, 8


# In[ ]:


sns.scatterplot(data=df_pos, x='Finishing', y='Positioning', hue='Position', palette='viridis')
plt.show()


# From the above visualization, you can notice several outliers. Let us remove them.

# In[ ]:


df_pos = df_pos[~((df_pos['Position'] == 1) & (df_pos['Finishing'] > 30) & (df_pos['Positioning'] < 60))]
df_pos = df_pos[~((df_pos['Position'] == 2) & (df_pos['Finishing'] > 60) & (df_pos['Positioning'] < 80))]
df_pos = df_pos[~((df_pos['Position'] == 3) & (df_pos['Finishing'] < 45))]
df_pos = df_pos[~((df_pos['Position'] == 3) & (df_pos['Finishing'] < 60) & (df_pos['Positioning'] > 70))]
df_pos = df_pos[~((df_pos['Position'] == 2) & (df_pos['Finishing'] > 65) & (df_pos['Positioning'] > 70))]


# In[ ]:


sns.scatterplot(data=df_pos, x='Finishing', y='Positioning', hue='Position', palette='viridis')
plt.show()


# There are still several outliers present but if we try and remove them, it might lead to overfitting.<br><br>
# Let us now test our model after removing outliers.

# **Logistic Regression Model**

# In[ ]:


X = df_pos.drop(columns=['Position'])
X = pd.get_dummies(X)
y = df_pos['Position']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
logmodel.fit(X_train, y_train)


# In[ ]:


prediction = logmodel.predict(X_test)


# In[ ]:


print(classification_report(y_test, prediction))
print('\n')
print(confusion_matrix(y_test, prediction))
print('\n')
print('Accuracy Score: ', accuracy_score(y_test, prediction))


# The overall accuracy of the model increases. Our model is now able to better classify Midfielders and Forwards.<br><br>
# Let us feed the above data into a different classification model and see how it performs.<br><br><br><br>

# **Gradient Boosting Classifier**

# In[ ]:


gbclassifier = GradientBoostingClassifier()


# In[ ]:


gbclassifier.fit(X_train, y_train)


# In[ ]:


prediction = gbclassifier.predict(X_test)


# In[ ]:


print(classification_report(y_test, prediction))
print('\n')
print(confusion_matrix(y_test, prediction))
print('\n')
print('Accuracy Score: ', accuracy_score(y_test, prediction))


# The Gradient Boosting Classifier performs much better than the Logistic Regression model as expected and gets a very high accuracy score.
# <br><br><br>
# 

# <br>

# ----

# <br>
# ### Predicting *Overall* from the dataset.

# In[ ]:


df_ovr = data.copy()
df_ovr.drop(columns=['Name', 'Nationality', 'Club'], inplace=True)


# In[ ]:


X = df_ovr.drop(columns=['Overall'])
X = pd.get_dummies(X)
y = df_ovr['Overall']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# **Linear Regression Model**

# In[ ]:


linmodel = LinearRegression()


# In[ ]:


linmodel.fit(X_train, y_train)


# In[ ]:


pred = linmodel.predict(X_test)


# In[ ]:


print('RMSE:', np.sqrt(mean_squared_error(y_test, pred)))
print('r^2 score: ', r2_score(y_test, pred))


# An r2_score of 0.9364 is good where 1.0 is the best possible r2_score.<br><br>
# Let us see if we can better this.<br><br><br>

# **Correlation and Outliers**

# In[ ]:


df_ovr.corr().abs()['Overall'].sort_values(ascending=False)


# In[ ]:


#Dropping Height in Cms because of a very low correlation with Overall.
#Not dropping GK because it would be one of the features to predict Overall for goalkeepers.
df_ovr.drop(columns=['Height in Cms'], inplace=True)


# In[ ]:


sns.scatterplot(data=df_ovr, x='Reactions', y='Overall')
plt.show()


# In[ ]:


df_ovr = df_ovr[~((df_ovr['Reactions'] < 25))]
df_ovr = df_ovr[~((df_ovr['Reactions'] < 35) & (df_ovr['Overall'] > 55))]
df_ovr = df_ovr[~((df_ovr['Reactions'] < 35) & (df_ovr['Overall'] > 55))]
df_ovr = df_ovr[~((df_ovr['Reactions'] > 62) & (df_ovr['Overall'] < 55) & (df_ovr['Reactions'] < 70))]

df_ovr.drop(df_ovr[(df_ovr['Reactions'] == 73) & (df_ovr['Overall'] == 55)].index, inplace=True)
df_ovr.drop(df_ovr[(df_ovr['Reactions'] == 74) & (df_ovr['Overall'] == 59)].index, inplace=True)
df_ovr.drop(df_ovr[(df_ovr['Reactions'] == 79) & (df_ovr['Overall'] == 64)].index, inplace=True)
df_ovr.drop(df_ovr[(df_ovr['Reactions'] == 82) & (df_ovr['Overall'] == 68)].index, inplace=True)
df_ovr.drop(df_ovr[(df_ovr['Reactions'] == 83) & (df_ovr['Overall'] == 70)].index, inplace=True)
df_ovr.drop(df_ovr[(df_ovr['Reactions'] == 84) & (df_ovr['Overall'] == 69)].index, inplace=True)


# In[ ]:


sns.scatterplot(data=df_ovr, x='Reactions', y='Overall')
plt.show()


# After removing outliers.<br><br><br>

# **Linear Regression Model**

# In[ ]:


X = df_ovr.drop(columns=['Overall'])
X = pd.get_dummies(X)
y = df_ovr['Overall']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


linmodel.fit(X_train, y_train)


# In[ ]:


pred = linmodel.predict(X_test)
print('RMSE:', np.sqrt(mean_squared_error(y_test, pred)))
print('r^2 score: ', r2_score(y_test, pred))


# The Linear Regression model does slightly better than the previous model. <br><br>
# RMSE Score comes down while there is asmall increase in r2_score. <br><br><br>

# **Gradient Boosting Regressor**

# In[ ]:


X = df_ovr.drop(columns=['Overall'])
X = pd.get_dummies(X)
y = df_ovr['Overall']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


gbregressor = GradientBoostingRegressor()


# In[ ]:


gbregressor.fit(X_train, y_train)


# In[ ]:


pred = gbregressor.predict(X_test)


# In[ ]:


pred = gbregressor.predict(X_test)
print('RMSE:', np.sqrt(mean_squared_error(y_test, pred)))
print('r^2 score: ', r2_score(y_test, pred))


# A huge improvement using the Gradient Boosting Regressor.<br>
# 
# RMSE Score comes down to 0.6868, while r2_score is very close to 1.
