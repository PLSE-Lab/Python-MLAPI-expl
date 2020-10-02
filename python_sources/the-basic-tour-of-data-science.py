#!/usr/bin/env python
# coding: utf-8

# # Data science is a field where scientific methods,  process, and algorithms are used to extract insights from structured and unstructured data. 
# Data science basic structure is consist of the following features:
# 1. Problem Specification
# 2. Obtain Data
# 3. Scrub Data
# 4. Model Data
# 5. Analyzing the Data
# 6. Visualizing the Data
# 
# In this article, I will show all the features of data science techniques that have been covered in the course on a proper dataset.
# 
# Pandas
# Matplotlib
# Scipy (Regression)
# Machine Learning Techniques (Supervised and Unsupervised Learning)
# Neural Network
# 
# Dataset: I have selected FIFA 19 complete player dataset from Kaggle. 
# 
# **Task 1: First task is to exploratory analysis based on the dataset.**
# 
# **Task 2: Use a predictive model on the dataset to make player position prediction**

# # Why I chose this dataset?
# I chose FIFA 19 complete player dataset because it has more than 10k data in the dataset. This dataset contains a lot of features with a combination of structured and unstructured data. The dataset contains NaN and unnecessary value which requires a complex pre-processing of the dataset. All these challenges make me chose this dataset over other datasets. 

# # Obtain the data

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import plotly.graph_objs as ptly
from plotly.offline import init_notebook_mode, iplot
import math
from collections import Counter
from sklearn.model_selection import train_test_split


# In[ ]:


data = pd.read_csv('../input/fifa19/data.csv', header=0)


# In[ ]:


data.head(5)


# # Scrubbing and Formatting

# In[ ]:


model_data = data.head(10000)


# Now, In the selected dataset, there are many columns that are not needed for further analysis which is not relatable to my goals.

# In[ ]:


model_data.drop(columns=['Unnamed: 0', 'ID', 'Photo', 'Flag', 'Club', 'Club Logo', 'Special', 'Real Face', 'Release Clause',
                   'Joined', 'Loaned From', 'Contract Valid Until'], inplace=True)


# In[ ]:


model_data.head(2)


# Checking how many null values are in the dataset!

# In[ ]:


model_data.isnull().sum()


# In[ ]:


model_data.shape


# # Let's make the position in four categories!
# 
# 1. Goal Keeper (GK)
# 2. Midfielder (MD)
# 3. Defender (DF)
# 4. Forward (FD)
# 
# 
# 
# 

# In[ ]:


def position_conversion(value):
    
    if value == 'RF' or value == 'ST' or value == 'LF' or value == 'RS' or value == 'LS' or value == 'CF':
        return 'F'
        
    elif value == 'LW' or value == 'RCM' or value == 'LCM' or value == 'LDM' or value == 'CAM' or value == 'CDM' or value == 'RM'          or value == 'LAM' or value == 'LM' or value == 'RDM' or value == 'RW' or value == 'CM' or value == 'RAM':
        return 'M'
    
    elif value == 'RCB' or value == 'CB' or value== 'LCB' or value == 'LB' or value == 'RB' or value == 'RWB' or value == 'LWB':
        return 'D'
    
    else:
        return value
model_data['Position'] = model_data['Position'].apply(position_conversion)


# In[ ]:


model_data['Position'].unique()


# I will be setting this position into a specific number which is acting as a separate class for the model.

# In[ ]:


def position_setting(value):
  if value == 'GK':
    return 0
  elif value == 'D':
    return 1
  elif value == 'M':
    return 2
  else:
    return 3
model_data['Position'] = model_data['Position'].apply(position_setting)


# In[ ]:


model_data['Position'].unique()


# From the looks of the dataset, I need to fix these following columns:
# 1. Value
# 2. Wage
# 3. Work Rate
# 4. Body Type
# 5. Height
# 6. Weight
# 7. All the abilities that contain "+" in-between numbers.

# # Let's start with height and weight

# In[ ]:


model_data[['Height', 'Weight']].head(5)


# In[ ]:


# Height conversion to Cms
def height_convert(value):
  get_split_val = value.split("'")
  update_height = (int(get_split_val[0])*30.48) + (int(get_split_val[1])*2.54)
  return update_height
model_data['Height'] = model_data['Height'].apply(height_convert)


# In[ ]:


# Weight conversion to pounds
def weight_convert(value):
  update_weight = int(value.split('lbs')[0])
  return update_weight
model_data['Weight'] = model_data['Weight'].apply(weight_convert)


# In[ ]:


model_data[['Height', 'Weight']].head(5)


# # Now, let's work on value and wage!

# In[ ]:


def value_wage_conversion(value):
  if value[-1] == 'M':
    value = value[1:-1]
    value = float(value) * 1000000
    return value
  elif value[-1] == 'K':
    value = value[1:-1]
    value = float(value) * 1000
    return value
  else:
    return 0
model_data['Value'] = model_data['Value'].apply(value_wage_conversion)


# In[ ]:


model_data['Wage'] = model_data['Wage'].apply(value_wage_conversion)


# In[ ]:


model_data[['Value', 'Wage']].head(5)


# # Let's simplify all the skils!

# In[ ]:


def skill_conversion(value):
  if type(value) == str:
    if value == 'NaN':
      return 0
    else:
      return int(value[0:2]) + int(value[-1])
  elif math.isnan(value):
    return 0
  else:
    return value
all_skills = ['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM',
       'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM',
       'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']
for i in all_skills:
  model_data[i] = model_data[i].apply(skill_conversion)


# # Let's check about preferred foot!

# In[ ]:


model_data['Preferred Foot'].unique()


# In[ ]:


# setting left foot as 0 and right foot as 1
def foot_fix(value):
  if value == 'Left':
    return 0
  else:
    return 1
model_data['Preferred Foot'] = model_data['Preferred Foot'].apply(foot_fix)


# I don't need the Jersey number! it's not necessary to achieve my goal.

# In[ ]:


model_data.drop(columns=['Jersey Number'], inplace=True)


# # Let's check about the Body Type!

# In[ ]:


model_data['Body Type'].unique()


# In[ ]:


def body_type_fix(value):
  if value == 'Messi':
    return 'Lean'
  elif value == 'C. Ronaldo':
    return 'Normal'
  elif value == 'Neymar':
    return 'Lean'
  elif value == 'Courtois':
    return 'Normal'
  elif value == 'PLAYER_BODY_TYPE_25':
    return 'Normal'
  elif value == 'Shaqiri':
    return 'Stocky'
  elif value == 'Akinfenwa':
    return 'Normal'
  else:
    return value
model_data['Body Type'] = model_data['Body Type'].apply(body_type_fix)


# # Lets check about work rate!

# In[ ]:


model_data['Work Rate'].unique()


# Let's simplify work rate into three category:
# 
# 1. High
# 2. Medium
# 3. Low

# In[ ]:


def work_rate_conversion(value):
  if value == 'Medium/ Medium':
    return 'Medium'
  elif value == 'High/ Low':
    return 'Medium'
  elif value == 'High/ Medium':
    return 'High'
  elif value == 'High/ High':
    return 'High'
  elif value == 'Medium/ High':
    return 'Medium'
  elif value == 'Medium/ Low':
    return 'Low'
  elif value == 'Low/ High':
    return 'Medium'
  elif value == 'Low/ Medium':
    return 'Low'
  elif value == 'Low/ Low':
    return 'Low'
  else:
    return value
model_data['Work Rate'] = model_data['Work Rate'].apply(work_rate_conversion)


# In[ ]:


model_data.isnull().sum()


# The dataset is fixed with preprocessing.

# In[ ]:


model_data


# # Exploratory Data Analysis

# In[ ]:


model_data.info()


# In[ ]:


model_data.shape


# In[ ]:


model_data.describe


# In[ ]:


model_data.head(5)


# In[ ]:


model_data.tail(5)


# In[ ]:


model_data['Potential'].describe()


# # Relation between players and their potential

# In[ ]:


# Creating a new dataframe for players potential to show in a graph in regarding to player names!!
potential = pd.DataFrame({"Name": model_data.Name, "Value": model_data.Potential})
potential = potential.tail(20)
plt.figure(figsize=(6,3))
sns.barplot(x = potential['Name'],y = potential['Value'])
plt.xticks(rotation = 75)
plt.xlabel("Name")
plt.ylabel("Potential")
plt.show()


# # Player statistics in each country

# In[ ]:


total_country = Counter(model_data.Nationality)
select_country_list = total_country.most_common(25)
country, tot_num = zip(*select_country_list)
country, tot_num = list(country), list(tot_num)
plt.figure(figsize=(15,12))
sns.barplot(x = country,y = tot_num)
plt.xticks(rotation = 75)
plt.xlabel("Nationality")
plt.ylabel("Total Number of player")
plt.show()


# # Number of players in each position

# In[ ]:


# position wise total player count
plt.figure(figsize=(10,6))
sns.countplot(model_data.Position,order=model_data.Position.value_counts().index)
plt.xticks(rotation=90)
plt.title("Position",fontsize=15)


# # Players age distribution

# In[ ]:


# plotting against age distribution
make_list = ["Senior (more than 27 years)" if age > 27 else "Junior (age below 23)" if age < 23 else "Prime Age (age between 23 and 27)"  for age in model_data.Age]
make_df = pd.DataFrame({"Age": make_list})
plt.figure(figsize=(10,6))
sns.countplot(x = make_df.Age)
plt.ylabel("Number of Players Age")
plt.title("Players age",fontsize=15)


# In[ ]:


def configure_plotly_browser_state():
  import IPython
  display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              plotly: 'https://cdn.plot.ly/plotly-1.5.1.min.js?noext',
            },
          });
        </script>
        '''))


# # Number of players in each country that is showing in a world map

# In[ ]:


configure_plotly_browser_state()
init_notebook_mode(connected=False)


plt1 = dict(type='choropleth',
              locations=country,
              z=tot_num,
              locationmode='country names'
             )

layout = ptly.Layout(title=' Total Number of Players in each Country',
                   geo=dict(showocean=True,
                            projection=dict(type='natural earth'),
                        )
                  )

fig = ptly.Figure(data=[plt1], layout=layout)
iplot(fig)


# # Histogram of age distribution

# In[ ]:


configure_plotly_browser_state()
init_notebook_mode(connected=False)

x = model_data['Age']
data = [ptly.Histogram(x=x)]

iplot(data, filename='age distribution')


# # Height and weight distribution in histogram

# In[ ]:


configure_plotly_browser_state()
init_notebook_mode(connected=False)
x0 = model_data.Height
x1 = model_data.Weight

trace0 = ptly.Histogram(
    x=x0
)
trace1 = ptly.Histogram(
    x=x1
)
data = [trace0, trace1]
layout = ptly.Layout(barmode='stack')
fig = ptly.Figure(data=data, layout=layout)

iplot(fig, filename='stacked histogram')


# # Scatter plot for players acceleration distribution

# In[ ]:


configure_plotly_browser_state()
init_notebook_mode(connected=False)
trace1 = ptly.Scatter(
    y = model_data['Acceleration'].head(500),
    mode='markers',
    marker=dict(
        size=16,
        color = np.random.randn(500), 
        colorscale='Viridis',
        showscale=True
    )
)
data = [trace1]

iplot(data, filename='Acceleration')


# # Combination of skills for few players

# In[ ]:


new_data1 = model_data.head(5)
configure_plotly_browser_state()
init_notebook_mode(connected=False)
trace1 = ptly.Barpolar(
    r = new_data1['Age'],
    text=new_data1['Name'],
    name='Players age',
    marker=dict(
        color='rgb(106,81,163)'
    )
)
trace2 = ptly.Barpolar(
    r = new_data1['Potential'],
    text=new_data1['Name'],
    name='Players Potential',
    marker=dict(
        color='rgb(158,154,200)'
    )
)
trace3 = ptly.Barpolar(
    r = new_data1['Stamina'],
    text=new_data1['Name'],
    name='Players Stamina',
    marker=dict(
        color='rgb(203,201,226)'
    )
)
trace4 = ptly.Barpolar(
    r = new_data1['Strength'],
    text=new_data1['Name'],
    name='Players Strength',
    marker=dict(
        color='rgb(242,240,247)'
    )
)
data = [trace1, trace2, trace3, trace4]
layout = ptly.Layout(
    title='Players statistics',
    font=dict(
        size=16
    ),
    legend=dict(
        font=dict(
            size=16
        )
    ),
    radialaxis=dict(
        ticksuffix='%'
    ),
    orientation=270
)
fig = ptly.Figure(data=data, layout=layout)
iplot(fig, filename='Players statistics distribution')


# # Combination of players skills in a bar chart

# In[ ]:


configure_plotly_browser_state()
init_notebook_mode(connected=False)

trace1 = {
  'x': new_data1['Name'],
  'y': new_data1['Potential'],
  'name': 'Potential',
  'type': 'bar'
};
trace2 = {
  'x': new_data1['Name'],
  'y': new_data1['Overall'],
  'name': 'Overall',
  'type': 'bar'
};
trace3 = {
  'x': new_data1['Name'],
  'y': new_data1['Acceleration'],
  'name': 'Acceleration',
  'type': 'bar'
 }
 
trace4 = {
  'x': new_data1['Name'],
  'y': new_data1['Aggression'],
  'name': 'Aggression',
  'type': 'bar'
 }
 
data = [trace1, trace2, trace3, trace4];
layout = {
  'xaxis': {'title': 'Name'},
  'yaxis': {'title': 'Statistics'},
  'barmode': 'relative',
  'title': 'Player statistics'
};
iplot({'data': data, 'layout': layout}, filename='barmode-relative')


# # Players position distribution in a pie chart

# In[ ]:


new_data2 = model_data.head(200)
configure_plotly_browser_state()
init_notebook_mode(connected=False)

pos = ptly.Pie(values=new_data2['Position'].value_counts().values,
                labels=new_data2['Position'].value_counts().index.values,
                hole=0.3
               )
 

layout = ptly.Layout(title='players position distribution')

fig = ptly.Figure(data=[pos], layout=layout)
iplot(fig)


# # Making models

# The goal of these models is to predict a player position in a footbal field based on the players skills statistics!

# In[ ]:


model_data.head(2)


# For the purpose of the model, I won't be using Name, Nationality, Work Rate, Body Type, Value, Wage. I will be dropping these features from the dataset. 

# In[ ]:


model_data.drop(columns=['Name', 'Nationality', 'Value', 'Wage', 'Work Rate', 'Body Type'], inplace=True)


# In[ ]:


model_data.head(2)


# # Logistic Regression

# In[ ]:


X = model_data.drop(columns=['Position'], axis=1)
y = model_data['Position']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
prediction = model.predict(X_test)


# In[ ]:


print(classification_report(y_test, prediction), '\n')
print(confusion_matrix(y_test, prediction), '\n')
print('Accuracy Score: ', accuracy_score(y_test, prediction))


# In[ ]:


model_data.corr().abs()['Position'].sort_values(ascending=False)


# However, we can sort the skills into 4 categories also such as:
# 1. Forward
# 2. Defender
# 3. Midfielder
# 3. Goal Keeper

# In[ ]:


model_data_1 = model_data.head(5000)


# In[ ]:


model_data_1.head(2)


# In[ ]:


model_data_1['forward'] = (model_data_1['RF'] + model_data_1['ST'] + model_data_1['LF'] + model_data_1['RS'] + model_data_1['LS'] + model_data_1['CF']) / 6


# In[ ]:


model_data_1['midfielder'] = (model_data_1['LW'] + model_data_1['RCM'] + model_data_1['LCM'] + model_data_1['LDM'] + model_data_1['CAM'] + model_data_1['CDM'] +                 model_data_1['RM'] + model_data_1['LAM'] + model_data_1['LM'] + model_data_1['RDM'] + model_data_1['RW'] + model_data_1['CM'] + model_data_1['RAM'])                /13


# In[ ]:


model_data_1['defender'] = (model_data_1['RCB'] + model_data_1['CB'] + model_data_1['LCB'] + model_data_1['LB'] + model_data_1['RB'] + model_data_1['RWB']                 + model_data_1['LWB']) / 7


# In[ ]:


model_data_1['gk'] = (model_data_1['GKDiving'] + model_data_1['GKHandling'] + model_data_1['GKKicking'] + model_data_1['GKPositioning']               + model_data_1['GKReflexes']) / 5


# In[ ]:


model_data_1[['forward', 'midfielder', 'defender', 'gk']].head(5)


# # Now let's delete all the field that were merged together!!

# In[ ]:


model_data_1.drop(columns=['RF', 'ST', 'LW', 'RCM', 'LF', 'RS', 'RCB', 'LCM', 'CB', 'LDM', 'CAM', 'CDM',
                     'LS', 'LCB', 'RM', 'LAM', 'LM', 'LB', 'RDM', 'RW', 'CM', 'RB', 'RAM', 'CF', 'RWB', 'LWB',
                     'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes'
                    ], inplace=True)


# In[ ]:


model_data_1.head(2)


# In[ ]:


model_data_1.shape


# Let's run the same approach again!

# In[ ]:


X = model_data_1.drop(columns=['Position'])
y = model_data_1['Position']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
prediction = model.predict(X_test)


# In[ ]:


print(classification_report(y_test, prediction), '\n')
print(confusion_matrix(y_test, prediction), '\n')
print('Accuracy Score: ', accuracy_score(y_test, prediction))


# # Linear Regression Model

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
prediction = model.predict(X_test)


# In[ ]:


print ("Score:", model.score(X_test, y_test))


# So we can see that Logistic Regression gave us better classification result than the Linear Regression.

# In[ ]:


plt_data = model_data_1.head(500)
sns.swarmplot(data=plt_data, x='Age', y='Positioning', hue='Position', palette='viridis')
plt.figure(figsize=(20, 15))
plt.show()


# In[ ]:


sns.swarmplot(data=model_data_1, x='Age', y='Positioning', hue='Position', palette='viridis')
plt.figure(figsize=(20, 15))
plt.show()


# So, after merging those skill set together this model is getting the same results. Let's verify that the model is giving us proper accuracy. 

# In[ ]:


# However, from the dataset it appears to be an imbalanced dataset! why?
model_data_1['Position'].value_counts()


# As you can see that, there is only 455 Gk, 742 Strikers where 2233 is Midfielders and 1570 defenders.
# 
# Our last model might give us biased result due to imbalanced dataset in each class!

# In[ ]:


from imblearn.combine import SMOTEENN
from collections import Counter

smote_enn = SMOTEENN(random_state=0)
X_resampled, y_resampled = smote_enn.fit_sample(X, y)
print(sorted(Counter(y_resampled).items()))


# # SMOTEENN is used to balance the dataset!

# In[ ]:


# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.33)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)


# # Linear Regression

# In[ ]:


# fit a model
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)


# In[ ]:


plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")


# In[ ]:


print ("Score:", model.score(X_test, y_test))


# So, after balancing the dataset, we are getting better accuracy than previous. It's 95.82% now whereas it was 84.90% using linear regression!

# # Logistic Regression

# In[ ]:


# fit a model
lm = linear_model.LogisticRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)


# In[ ]:


print ("Score:", model.score(X_test, y_test))


# So, after balancing the dataset, we are getting better accuracy than previous. It's 97.20% now whereas it was 90.24% using logistic regression!

# # Neural Network Model

# In[ ]:


# A multilayer perceptron (MLP) is a feedforward artificial neural network model that maps sets of input data onto a set of appropriate outputs.
from sklearn.neural_network import MLPClassifier


# In[ ]:


X = model_data.drop('Position', axis=1) #feature
y = model_data['Position']

smote_enn = SMOTEENN(random_state=0)
X_resampled, y_resampled = smote_enn.fit_sample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.45)
mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
mlp.fit(X_train,y_train)


# In[ ]:


predictions = mlp.predict(X_test)
print(classification_report(y_test,predictions))


# In[ ]:


print ("Score:", mlp.score(X_test, y_test))


# # After using a neural network, it shows us a slight improvement in accuracy from the dataset which is 98%!

# In[ ]:


#END#

