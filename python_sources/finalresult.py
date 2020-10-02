#!/usr/bin/env python
# coding: utf-8

# **2. Data Pre-processing**

# **2.1 Importing all the required libraries - **

# In[ ]:


import pandas as pd
import regex as re
import plotly.offline as py
import plotly.graph_objs as go
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
import os
py.init_notebook_mode(connected=True)
import plotly.tools as tls

from sklearn.tree import DecisionTreeClassifier


# Reading in training and testing data -

# In[ ]:


train = pd.read_csv('../input/foundation/foundation.csv')
test = pd.read_csv('../input/testfinaldata/testdata.csv')
train.head()


# In[ ]:


test.head()


# Create a column with the name 'Year'. So that some data analytics can be done on it. 

# In[ ]:


train['year'] = -1
for i in range(train.shape[0]):
    train['year'][i] = '20' + train['Start Date'][i][7:]
train.head()


# Train data since 2016 -  

# In[ ]:


trainold = train


# In[ ]:





# #Alter train data to include only those countries which are playing in the 2019 World Cup.

# In[ ]:


# country = ['Afghanistan','Australia','Bangladesh','England','India','New Zealand','Pakistan','South Africa','Sri Lanka','West Indies']
# train = train[train.Team.isin(country)]
# train = train[train.Opposition.isin(country)]
# train.head() 


# In[ ]:





# Create a column to depict win/loss as a boolean value.
# * if winbool == 1 --> Team A Wins
# * if winbool == 0 --> Team B Wins

# In[ ]:


train['winbool'] = 0


# In[ ]:


train.head()


# Reset the index to begin at 0.

# In[ ]:


train = train.reset_index()
train = train.drop(['index'],axis=1)
train.head()


# Logic to populate winbool column.

# In[ ]:


for i in range(train.shape[0]):
    if train['Winner'][i] == train['Team'][i]:
        train['winbool'][i] = 1
train.head()
        


# Add 3 more columns:
# * Match Country (String) : Specifies which country the match is being played in. 
# * TeamA Home (Boolean) : '1' if Match Coutry is home country for Team A, else '0'.
# * TeamB Home (Boolean) : '1' if Match Coutry is home country for Team B, else '0'.

# In[ ]:


train['TeamA Home'] = 0
train['TeamB Home'] = 0
train['Match Country'] = 'qwerty'
train.head()


# Utility sets to help map Cricket Grounds to Countries.

# In[ ]:


Australia = {'Melbourne', 'Hobart', 'Sydney', 'Adelaide', 'Brisbane','Perth','Canberra','Townsville'}
Bangladesh= {'Dhaka','Chattogram','Khulna','Fatullah','Sylhet'}
Cannada = {'King City (NW)','Toronto'}
England = {'The Oval','Bristol','Birmingham', 'Leeds', 'Lord\'s','Cardiff', 'Nottingham', 'Manchester','Chester-le-Street','Taunton','Southampton'}
HongKong = {'Mong Kok'}
India = {'Dehradun','Thiruvananthapuram','Mumbai (BS)','Guwahati','Greater Noida','Chennai','Kanpur','Jaipur','Pune','Dharamsala','Kochi','Ranchi','Rajkot','Indore', 'Ahmedabad','Cuttack','Visakhapatnam', 'Nagpur', 'Delhi', 'Bengaluru','Mohali','Mumbai','Kolkata','Hyderabad (Deccan)'}
Ireland = {'Belfast','Dublin','Dublin (Malahide)'}
Kenya = {'Mombasa'}
Malaysia= {'Kuala Lumpur'}
NewZealand = {'Wellington','Mount Maunganui','Lincoln','Nelson','Whangarei', 'Queenstown', 'Christchurch', 'Napier', 'Hamilton','Auckland','Dunedin'}
Netherlands = {'The Hague','Amstelveen'}
Pakistan = {'Lahore'}
PNG = {'Port Moresby'}
Scotland = {'Aberdeen','Edinburgh','Ayr'}
SouthAfrica= {'Cape Town','Benoni','St George\'s','Potchefstroom','Kimberley','Bloemfontein', 'Durban', 'Johannesburg', 'Port Elizabeth', 'Centurion','Paarl','East London'}
SriLanka ={'Galle','Dambulla','Colombo (SSC)', 'Hambantota', 'Colombo (RPS)', 'Pallekele'}
UAE ={'Dubai (DSC)', 'Sharjah', 'Abu Dhabi', 'ICCA Dubai'}
WestIndies ={'Gros Islet','Basseterre','Bridgetown','Providence','Port of Spain','North Sound','Kingston','Kingstown'}
Zimbabwe ={'Harare','Bulawayo'}
      


# Mapping Ground to Country.

# In[ ]:


for i in range(train['Ground'].shape[0]):
    if train['Ground'][i] in Australia:
        train['Match Country'][i] = 'Australia'
    elif train['Ground'][i] in Bangladesh:
        train['Match Country'][i] = 'Bangladesh'
    elif train['Ground'][i] in Cannada:
        train['Match Country'][i] = 'Cannada'
    elif train['Ground'][i] in England:
        train['Match Country'][i] = 'England'
    elif train['Ground'][i] in HongKong:
        train['Match Country'][i] = 'Hong Kong'
    elif train['Ground'][i] in India:
        train['Match Country'][i] = 'India'
    elif train['Ground'][i] in Ireland:
        train['Match Country'][i] = 'Ireland'
    elif train['Ground'][i] in Kenya:
        train['Match Country'][i] = 'Kenya'
    elif train['Ground'][i] in Malaysia:
        train['Match Country'][i] = 'Malaysia'
    elif train['Ground'][i] in NewZealand:
        train['Match Country'][i] = 'New Zealand'
    elif train['Ground'][i] in Netherlands:
        train['Match Country'][i] = 'Netherlands'
    elif train['Ground'][i] in Pakistan:
        train['Match Country'][i] = 'Pakistan'
    elif train['Ground'][i] in PNG:
        train['Match Country'][i] = 'P.N.G.'
    elif train['Ground'][i] in Scotland:
        train['Match Country'][i] = 'Scotland'
    elif train['Ground'][i] in SouthAfrica:
        train['Match Country'][i] = 'South Africa'
    elif train['Ground'][i] in SriLanka:
        train['Match Country'][i] = 'Sri Lanka'
    elif train['Ground'][i] in UAE:
        train['Match Country'][i] = 'U.A.E.'
    elif train['Ground'][i] in WestIndies:
        train['Match Country'][i] = 'West Indies'
    elif train['Ground'][i] in Zimbabwe:
        train['Match Country'][i] = 'Zimbabwe'
    


# In[ ]:


train.head()


# Populate TeamA Home and TeamB Home, according to the Match Country.

# In[ ]:


for i in range(train.shape[0]):
    if train['Match Country'][i] == train['Team'][i]:
        train['TeamA Home'][i] = 1
    elif train['Match Country'][i] == train['Opposition'][i]:
        train['TeamB Home'][i] = 1


# In[ ]:


train.head()


# In[ ]:


train.to_csv('preprocessedData.csv', index=False)


# In[ ]:


country2019wc = ['Afghanistan','Australia','Bangladesh','England','India','New Zealand','Pakistan','South Africa','Sri Lanka','West Indies']
trainwc2019 = train[train.Team.isin(country2019wc)]
trainwc2019 = trainwc2019[trainwc2019.Opposition.isin(country2019wc)]
trainwc2019.shape[0]


# In[ ]:


totalMatches=pd.concat([train['Team'],train['Opposition']])
totalMatches=totalMatches.value_counts().reset_index()
totalMatches.head(11)
# totalMatches.columns=['Team','Total Matches']


# **Visualizations**

# In[ ]:





# In[ ]:


totalMatches=pd.concat([train['Team'],train['Opposition']])
totalMatches=totalMatches.value_counts().reset_index()
totalMatches.columns=['Team','Total Matches']
totalMatches.head(11)
# totalMatches.columns=['Team','Total Matches']


# In[ ]:


totalMatches['wins']=train['Winner'].value_counts().reset_index()['Winner']
totalMatches.set_index('Team',inplace=True)
totalMatches.head(11)


# In[ ]:


trace1 = go.Bar(
    x=totalMatches.index,
    y=totalMatches['Total Matches'].head(11),
    name='Total Matches'
)

trace2 = go.Bar(
    x=totalMatches.index,
    y=totalMatches['wins'].head(11),
    name='Wins'
)
data = [trace1, trace2]
layout = go.Layout(
    barmode='stack'
)
fig = go.Figure(data = data, layout =layout)
py.iplot(fig, filename='stacked-bar')


# In[ ]:


match_succes_rate = (totalMatches['wins']/totalMatches['Total Matches'])*100
print(match_succes_rate.head(11))


# In[ ]:


#print(match_succes_rate)

def annot_plot(ax,w,h):                                    # function to add data to plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for p in ax.patches:
        ax.annotate('{0:.1f}'.format(p.get_height()), (p.get_x()+w, p.get_height()+h))

data = match_succes_rate.head(11).sort_values(ascending = False)
plt.figure(figsize=(7,3))
ax = sns.barplot(x = data.index, y = data, palette="Set2");
plt.ylabel('Succes rate of each team')
plt.xticks(rotation=80)
annot_plot(ax,0.08,1)


# In[ ]:


#performance in england


# In[ ]:


trainEngland = train[train['Match Country'].isin(['England'])]
trainEngland.head()
#TODO: add a check to include only those countries wh have played 10 or more matches


# In[ ]:


trainEngland = trainEngland.reset_index()
trainEngland = trainEngland.drop(['index'],axis=1)
trainEngland.head()


# In[ ]:


totalMatchesInEngland=pd.concat([trainEngland['Team'],trainEngland['Opposition']])
totalMatchesInEngland=totalMatchesInEngland.value_counts().reset_index()
totalMatchesInEngland.columns=['Team','Total Matches']


# In[ ]:


totalMatchesInEngland['wins']=trainEngland['Winner'].value_counts().reset_index()['Winner']
totalMatchesInEngland.set_index('Team',inplace=True)
totalMatchesInEngland.head(10)


# In[ ]:


trace1 = go.Bar(
    x=totalMatchesInEngland.index,
    y=totalMatchesInEngland['Total Matches'].head(8),
    name='Total Matches'
)

trace2 = go.Bar(
    x=totalMatchesInEngland.index,
    y=totalMatchesInEngland['wins'].head(8),
    name='Wins'
)
data = [trace1, trace2]
layout = go.Layout(
    barmode='stack'
)
fig = go.Figure(data = data, layout =layout)
py.iplot(fig, filename='stacked-bar')


# In[ ]:


match_succes_rate = (totalMatchesInEngland['wins'].head(8)/totalMatchesInEngland['Total Matches'].head(8))*100
print(match_succes_rate.head(8))


# In[ ]:


def annot_plot(ax,w,h):                                    # function to add data to plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for p in ax.patches:
        ax.annotate('{0:.1f}'.format(p.get_height()), (p.get_x()+w, p.get_height()+h))

data = match_succes_rate.sort_values(ascending = False)
plt.figure(figsize=(7,3))
ax = sns.barplot(x = data.index, y = data, palette="Set2");
plt.ylabel('Succes rate of each team')
plt.xticks(rotation=80)
annot_plot(ax,0.08,1)


# In[ ]:


yearsSince2016 = ['2016','2017','2018','2019']
trainSince2016 = train[train.year.isin(yearsSince2016)]
trainSince2016.shape[0]


# In[ ]:


totalMatches=pd.concat([trainSince2016['Team'],trainSince2016['Opposition']])
totalMatches=totalMatches.value_counts().reset_index()
totalMatches.columns=['Team','Total Matches']


# In[ ]:


totalMatches['wins']=trainSince2016['Winner'].value_counts().reset_index()['Winner']
totalMatches.set_index('Team',inplace=True)
totalMatches.head(11)


# In[ ]:


trace1 = go.Bar(
    x=totalMatches.index,
    y=totalMatches['Total Matches'].head(11),
    name='Total Matches'
)

trace2 = go.Bar(
    x=totalMatches.index,
    y=totalMatches['wins'].head(11),
    name='Wins'
)
data = [trace1, trace2]
layout = go.Layout(
    barmode='stack'
)
fig = go.Figure(data = data, layout =layout)
py.iplot(fig, filename='stacked-bar')


# In[ ]:


match_succes_rate = (totalMatches['wins']/totalMatches['Total Matches'])*100
print(match_succes_rate.head(11))

def annot_plot(ax,w,h):                                    # function to add data to plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for p in ax.patches:
        ax.annotate('{0:.1f}'.format(p.get_height()), (p.get_x()+w, p.get_height()+h))

data = match_succes_rate.head(11).sort_values(ascending = False)
plt.figure(figsize=(7,3))
ax = sns.barplot(x = data.index, y = data, palette="Set2");
plt.ylabel('Succes rate of each team')
plt.xticks(rotation=80)
annot_plot(ax,0.08,1)


# In[ ]:


trainEngland = trainEngland[trainEngland.year.isin(yearsSince2016)]
totalMatchesInEngland=pd.concat([trainEngland['Team'],trainEngland['Opposition']])
totalMatchesInEngland=totalMatchesInEngland.value_counts().reset_index()
totalMatchesInEngland.columns=['Team','Total Matches']


# In[ ]:


totalMatchesInEngland['wins']=trainEngland['Winner'].value_counts().reset_index()['Winner']
totalMatchesInEngland.set_index('Team',inplace=True)
totalMatchesInEngland.head(8)


# In[ ]:


trace1 = go.Bar(
    x=totalMatchesInEngland.index,
    y=totalMatchesInEngland['Total Matches'].head(8),
    name='Total Matches'
)

trace2 = go.Bar(
    x=totalMatchesInEngland.index,
    y=totalMatchesInEngland['wins'].head(8),
    name='Wins'
)
data = [trace1, trace2]
layout = go.Layout(
    barmode='stack'
)
fig = go.Figure(data = data, layout =layout)
py.iplot(fig, filename='stacked-bar')


# In[ ]:


match_succes_rate = (totalMatchesInEngland['wins']/totalMatchesInEngland['Total Matches'])*100
print(match_succes_rate.head(8))

def annot_plot(ax,w,h):                                    # function to add data to plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for p in ax.patches:
        ax.annotate('{0:.1f}'.format(p.get_height()), (p.get_x()+w, p.get_height()+h))

data = match_succes_rate.head(8).sort_values(ascending = False)
plt.figure(figsize=(7,3))
ax = sns.barplot(x = data.index, y = data, palette="Set2");
plt.ylabel('Succes rate of each team')
plt.xticks(rotation=80)
annot_plot(ax,0.08,1)


# In[ ]:


yearsSince2018 = ['2018','2019']
trainSince2018 = train[train.year.isin(yearsSince2018)]
trainSince2018.shape[0]


# In[ ]:


totalMatches=pd.concat([trainSince2018['Team'],trainSince2018['Opposition']])
totalMatches=totalMatches.value_counts().reset_index()
totalMatches.columns=['Team','Total Matches']


# In[ ]:


totalMatches['wins']=trainSince2018['Winner'].value_counts().reset_index()['Winner']
totalMatches.set_index('Team',inplace=True)
totalMatches.head(11)


# In[ ]:


trace1 = go.Bar(
    x=totalMatches.index,
    y=totalMatches['Total Matches'].head(11),
    name='Total Matches'
)

trace2 = go.Bar(
    x=totalMatches.index,
    y=totalMatches['wins'].head(11),
    name='Wins'
)
data = [trace1, trace2]
layout = go.Layout(
    barmode='stack'
)
fig = go.Figure(data = data, layout =layout)
py.iplot(fig, filename='stacked-bar')


# In[ ]:


match_succes_rate = (totalMatches['wins']/totalMatches['Total Matches'])*100
print(match_succes_rate.head(11))

def annot_plot(ax,w,h):                                    # function to add data to plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for p in ax.patches:
        ax.annotate('{0:.1f}'.format(p.get_height()), (p.get_x()+w, p.get_height()+h))

data = match_succes_rate.head(11).sort_values(ascending = False)
plt.figure(figsize=(7,3))
ax = sns.barplot(x = data.index, y = data, palette="Set2");
plt.ylabel('Succes rate of each team')
plt.xticks(rotation=80)
annot_plot(ax,0.08,1)


# In[ ]:


trainEngland = trainEngland[trainEngland.year.isin(yearsSince2018)]
totalMatchesInEngland=pd.concat([trainEngland['Team'],trainEngland['Opposition']])
totalMatchesInEngland=totalMatchesInEngland.value_counts().reset_index()
totalMatchesInEngland.columns=['Team','Total Matches']


# In[ ]:


trainEngland


# In[ ]:


totalMatchesInEngland['wins']=trainEngland['Winner'].value_counts().reset_index()['Winner']
totalMatchesInEngland.set_index('Team',inplace=True)
totalMatchesInEngland


# In[ ]:


a = train[train.year.isin(yearsSince2018)]
a[a['Match Country'] == 'England']


# In[ ]:





# In[ ]:





# **3. Data Modelling**

# Create feature table with required columns.

# In[ ]:


#features = ['Team','Opposition','Ground','TeamA Rating','TeamB Rating','Match Country']
features = ['Team','Opposition','TeamA Rating','TeamB Rating','Match Country','TeamA Home','TeamB Home','Ground']
#features = ['Team','Opposition','TeamA Rating','TeamB Rating','Match Country','TeamA Home','TeamB Home']


# In[ ]:


X = train[features]
y = train.winbool
X.tail(40)


# In[ ]:





# Combine both train and test data, so as to encode all the categorical data to integer data.
# 
# * X : train data
# * X1: combined data
# * X2: test data 

# In[ ]:


X1 = X.append(test,ignore_index=True)
X2 = X1[1042:]
X2.head()


# Create a backup of original train and test data, before encoding it. 

# In[ ]:


trainBkp = X
testBkp = X2


# Import LabelEncoder from sklearn to encode the categorical data.

# In[ ]:


from sklearn.preprocessing import LabelEncoder


# Apply encoding to select columns of combined dataframe.

# In[ ]:


number = LabelEncoder()
X1['Ground'] = number.fit_transform(X1['Ground'].astype('str'))
X1['Team'] = number.fit_transform(X1['Team'].astype('str'))
X1['Opposition'] = number.fit_transform(X1['Opposition'].astype('str'))
X1['Match Country'] = number.fit_transform(X1['Match Country'].astype('str'))


# separate out encoded train and test data.

# In[ ]:


X1.head()


# In[ ]:


X = X1[:1042]
test = X1[1042:]
X.tail()


# In[ ]:


test.head()


# Import required function from sklearn to do the test/train split.

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# > > Take out actual train data from 'X' dataframe and keep the rest of data for cross-validation.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


y.shape[0]


# In[ ]:


#dt
model = DecisionTreeClassifier(random_state=1)

# Fit model
model.fit(X, y)


# In[ ]:


predictions = model.predict(X_test)
print ('Score:', model.score(X_test, y_test))


# In[ ]:


predictions = model.predict(test)
print(predictions)
testBkp['dtresult'] = predictions
testBkp


# In[ ]:





# 

# In[ ]:


def winner(x):
    if x.dtresult == 1:
        x["Winning_Team"] = x.Team
    else:
        x["Winning_Team"] = x.Opposition
    return x

a = testBkp.apply(winner, axis= 1)
b = a.groupby("Winning_Team").size()
b = b.sort_values(ascending=False)
print(b)


# In[ ]:


testBkp.drop(['dtresult'],axis=1,inplace=True)
testBkp


# In[ ]:





# In[ ]:



num_categories = 2
y_train = keras.utils.to_categorical(y_train, num_categories)
y_test= keras.utils.to_categorical(y_test, num_categories)
y_train


# In[ ]:





# Import keras for actual model building.

# In[ ]:


import keras


# In[ ]:



# Model Building
model = keras.models.Sequential()
model.add(keras.layers.Dense(50, activation="relu", input_dim = 8))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(80, activation="relu"))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(2, activation="softmax"))


# In[ ]:


# Compiling the model - adaDelta - Adaptive learning
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
#model.compile(loss=keras.losses.categorical_crossentropy, optimizer='rmsprop', metrics=['accuracy'])


# In[ ]:


# Training and evaluating
batch_size = 50
num_epoch = 1000
model_log = model.fit(X_train, y_train, batch_size = batch_size, epochs=num_epoch, verbose=1, validation_data=(X_test, y_test))


# Check the accuracy on the validation data.

# In[ ]:



train_score = model.evaluate(X_train, y_train, verbose=0)
test_score = model.evaluate(X_test, y_test, verbose=0)
print('Train accuracy:', train_score[1])
print('Test accuracy:', test_score[1])


# In[ ]:


model.summary()


# Now that the model is ready, we run the model on Test data.

# In[ ]:


prediction = model.predict_classes(test)
testBkp["Result"] = prediction
testBkp.head()


# Print the score of each team in the group matches. 

# In[ ]:


def winner(x):
    if x.Result == 1:
        x["Winning_Team"] = x.Team
    else:
        x["Winning_Team"] = x.Opposition
    return x

data_2019_final = testBkp.apply(winner, axis= 1)
results_2019 = data_2019_final.groupby("Winning_Team").size()
results_2019 = results_2019.sort_values(ascending=False)
print(results_2019)


# We have got the top 4 teams in the group stages.
# These 4 teams will proceed to the semi-finals.
# 
# The semi-finals are played according to the following rule.
# 
# * First - Fourth
# * Second - Third

# In[ ]:


def semifinal(first, second, third, fourth):
    first = 'England'
    second = 'New Zealand'
    third = 'India'
    fourth = 'Australia'
    
df = pd.DataFrame(columns=['Team','Opposition','TeamA Rating','TeamB Rating','Match Country','TeamA Home','TeamB Home','Ground'])


# In[ ]:



df.loc[-1] = ['England','Australia',1,5,'England',1,0,'Manchester']
df.loc[0] = ['Australia','England',5,1,'England',0,1,'Manchester']
df.loc[1] = ['India','New Zealand',2,3,'England',0,0,'Birmingham']
df.loc[2] = ['New Zealand','India',3,2,'England',0,0,'Birmingham']
# adding a row
df.index = df.index + 1  # shifting index
df = df.sort_index() 
df


# Encode all the categorical data to integer values

# In[ ]:


cleanup = {"Team":     {"Australia":1, "England":3, "New Zealand": 5, "India": 4},
                "Opposition": {"Australia":1, "England":3, "New Zealand": 5, "India": 4},
                "Match Country":{"England":2 },
                "Ground":{"Manchester": 46,"Birmingham":5},
               }

df1 = df.replace(cleanup)
df1.head()


# In[ ]:


df


# In[ ]:


predictionsemi = model.predict_classes(df1)
df['Result'] = -1

df["Result"] = predictionsemi
df
#df["Result"].head()


# In[ ]:


df['Result'][2].dtype


# In[ ]:


semifinalWinners = []
winloss = ['looses','wins']
for i in range(df.shape[0]):
    print('When '+df['Team'][i]+' bats first, then '+df['Team'][i] + ' '+winloss[df['Result'][i]])
    if i%2:
        print('')
    if df['Result'][i] ==1:
        semifinalWinners.append(df['Team'][i])
    


# In[ ]:


semifinalWinners


# In[ ]:


df = pd.DataFrame(columns=['Team','Opposition','TeamA Rating','TeamB Rating','Match Country','TeamA Home','TeamB Home','Ground'])
df   


# In[ ]:


# d = {'Team': [semifinalWinners[0],semifinalWinners[1]], 'Opposition': [semifinalWinners[1],semifinalWinners[0]], 
#      'TeamA Rating': [3, 4], 'TeamB Rating': [3, 4], 'Match Country': [3, 4], 
#      'TeamA Home': [3, 4],'Team': [semifinalWinners[0],semifinalWinners[1]],'Team': [semifinalWinners[0],semifinalWinners[1]],}

df.loc[-1] = ['England','India',1,2,'England',1,0,'Lord\'s']
df.loc[0] = ['India','England',2,1,'England',0,1,'Lord\'s']
# adding a row
df.index = df.index + 1  # shifting index
df = df.sort_index() 
df


# In[ ]:


cleanup = {"Team":     {"England":3,"India": 4},
                "Opposition": {"England":3, "India": 4},
                "Match Country":{"England":2 },
                "Ground":{"Lord\'s": 45},
               }

df.replace(cleanup, inplace=True)
df.head()


# In[ ]:


predictionsemi = model.predict_classes(df)
df['Result'] = -1

df["Result"] = predictionsemi
df
#df["Result"].head()


# In[ ]:




