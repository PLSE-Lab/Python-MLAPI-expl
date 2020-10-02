#!/usr/bin/env python
# coding: utf-8

# <h1><center><font size="6">Tutorial for Classification</font></center></h1>
# 
# <h2><center><font size="4">Dataset used: Titanic - Machine Learning from Disaster</font></center></h2>
# 
# <br>
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/640px-RMS_Titanic_3.jpg" width="450"></img>

# In[ ]:


import datetime
print("Last updated:")
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))


# 
# 
# # <a id='0'>Content</a>
# 
# - <a href='#1'>Introduction</a>  
# - <a href='#2'>Prepare the data analysis</a>  
#  - <a href='#21'>Load packages</a>  
#  - <a href='#22'>Load the data</a>   
# - <a href='#3'>Data exploration</a>   
#  - <a href='#31'>Check for missing data</a>  
#  - <a href='#32'>Sex, Age, SibSp, Parch</a>   
#  - <a href='#33'>Fare, Embarked, Pclass</a>  
#  - <a href='#34'>Ticket, Cabin, Name</a>   
#  - <a href='#35'>Multiple features visualization</a>   
#  - <a href='#36'>Imputation of missing data</a>   
# - <a href='#4'>Feature engineering</a>
#  - <a href='#41'>Extract Title from Name</a>
#  - <a href='#42'>Build families</a>
#  - <a href='#43'>Extract Deck from Ticket</a>  
#  - <a href='#44'>Estimate age</a>  
#  - <a href='#45'>More features engineering</a>  
# - <a href='#5'>Predictive model for survival</a>
#  - <a href='#50'>Split the data</a>  
#  - <a href='#51'>Build a baseline model</a>  
#  - <a href='#52'>Model evaluation</a>    
#  - <a href='#53'>Model refinement</a> 
#  - <a href='#54'>Submission</a>  
#  - <a href='#55'>Hyperparameters optimization</a>
#  - <a href='#56'>Submission (model with hyperparameters optimization)</a>  
# - <a href='#6'>Model ensambling</a>
#  - <a href='#61'>Create the ensamble framework</a>
#  - <a href='#62'>Create the Out-Of-Fold Predictions</a>
#  - <a href='#63'>Train the first level models</a>
#  - <a href='#64'>Correlation of the results</a>
#  - <a href='#65'>Build the second level (ensamble) model</a>
#  - <a href='#66'>Submission (ensamble)</a>
# - <a href='#7'>References</a>    

# # <a id='1'>Introduction</a>  
# 
# This Kernel will take you through the process of **analyzing the data** to understand the **predictive values** of various **features** and the possible correlation between different features, **selection of features** with predictive value, **features engineering** to create features with higher predictive value, creation of a **baseline model**, succesive **refinement** of the model (we are using **RandomForest**) through selection of features and, at the end, **submission** of the best solution found. 
# 
# Next, we take the model and define a multi-dimmensional matrix of **hyperparameters** we would like to test. We use Gradient Search and cross-validation to select the best set of hyperparameters. The best model is then used for the **second submission**.
# 
# Next, we will use **ensambling** (second level model trained with the output of first level models). We create several models (with the same set of parameters used with the previous model). We use **AdaBoost**, **CatBoost**, **ExtraTrees**, **GradientBoosting**, **RandomForest** and **SupportVectorMachines**. We are using **Out-Of-Folds** to avoid the risk that the base model predictions already having "seen" the test set and therefore overfitting when feeding these predictions. The Out-Of-Folds are concatenated and feed to the second level model (XGBoost Classifier) and the prediction using the second level model is submitted (**third submission**) as the solution.
# 
# The dataset used for this tutorial is the famous now **Titanic** dataset.
# 
# 
# <a href="#0"><font size="1">Go to top</font></a>  

# # <a id='2'>Prepare the data analysis</a>   
# 
# 
# Before starting the analysis, we need to make few preparation: load the packages, load and inspect the data.
# 

# ## <a id='21'>Load packages</a>
# 
# We load the packages used for the analysis. There are packages for data manipulation, visualization, models, hyperparameter optimization and model metrics..

# In[ ]:


import pandas as pd
import numpy as np
import sys
import os
import random
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import scipy
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
import xgboost as xgb


# <a href="#0"><font size="1">Go to top</font></a>  
# 
# 
# ## <a id='22'>Load the data</a>  
# 
# Let's see first what data files do we have in the root directory. 

# In[ ]:


IS_LOCAL = False
if(IS_LOCAL):
    PATH="../input/titanic/"
else:
    PATH="../input/"
os.listdir(PATH)


# There are **train** and **test** data as well as an example **submission** file.  
# 
# Let's load the **train** and **test** data.

# In[ ]:


train_df=pd.read_csv(PATH+'train.csv')
test_df=pd.read_csv(PATH+'test.csv')


# <a href="#0"><font size="1">Go to top</font></a>  
# 
# # <a id='3'>Data exploration</a>  
# 
# We check the shape of train and test dataframes and also show a selection of rows, to have an initial image of the data.
# 
# 

# In[ ]:


train_df.sample(5).head()


# In[ ]:


test_df.sample(5).head()


# In[ ]:


print("Train: rows:{} cols:{}".format(train_df.shape[0], train_df.shape[1]))
print("Test:  rows:{} cols:{}".format(test_df.shape[0], test_df.shape[1]))


# Both **train** and **test** files contains the following values:  
# 
# * **PassengerID** - the index of the passenger (in the dataset);  
# * **PClass** - the class of the passenger (from 1 to 3);
# * **Name** - the name of the passenger;
# * **Sex** - the sex of the passenger (female or male);  
# * **Age** - the age (where available) of the passenger;  
# * **SibSp** - the number of sibilings / spouses aboard of Titanic;  
# * **Parch** - the number of parents / children aboard of Titanic;  
# * **Ticket** - the ticket number;  
# * **Fare** - the passenger fare (ticket cost);  
# * **Cabin** - the cabin number;  
# * **Embarked** - the place of embarcation of the passenger (C = Cherbourg, Q = Queenstown, S = Southampton).  
# 
# The **train** data has as well the target value, **Survived**.
# 
# It is important, before going to create a model, to have a good understanding of the data. We will therefore explore the various features.

# 
# Let's start by checking if there are missing data, unlabeled data or data that is inconsistently labeled. 

# ## <a id='31'>Check for missing data</a>  
# 
# Let's create a function that check for missing data in the two datasets (train and test).

# In[ ]:


def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data(train_df)


# In[ ]:


missing_data(test_df)


# Both in **train** and **test** datasets, `Cabin` has more than 77% missing, `Age` more than 19%. `Embarked` is missing in 2 cases for **train** and `Fare` misses in  1 case for **test**.   
# 
# We will discuss, through this tutorial, the various possible methods to deal with missing data.

# ## <a id='32'>Sex, Age, SibSp, Parch</a>  
# 
# Let's check now the data (in **train** and **test**) for `Sex`, `Age`, `SibSp` and `Parch`.

# In[ ]:


def get_categories(data, val):
    tmp = data[val].value_counts()
    return pd.DataFrame(data={'Number': tmp.values}, index=tmp.index).reset_index()


# In[ ]:


def get_survived_categories(data, val):
    tmp = data.groupby('Survived')[val].value_counts()
    return pd.DataFrame(data={'Number': tmp.values}, index=tmp.index).reset_index()


# In[ ]:


def draw_trace_bar(data_df,color='Blue'):
    trace = go.Bar(
            x = data_df['index'],
            y = data_df['Number'],
            marker=dict(color=color),
            text=data_df['index']
        )
    return trace


def plot_bar(data_df, title, xlab, ylab,color='Blue'):
    trace = draw_trace_bar(data_df, color)
    data = [trace]
    layout = dict(title = title,
              xaxis = dict(title = xlab, showticklabels=True, tickangle=0,
                          tickfont=dict(
                            size=10,
                            color='black'),), 
              yaxis = dict(title = ylab),
              hovermode = 'closest'
             )
    fig = dict(data = data, layout = layout)
    iplot(fig, filename='draw_trace')


# In[ ]:


def plot_two_bar(data_df1, data_df2, title1, title2, xlab, ylab):
    trace1 = draw_trace_bar(data_df1, color='Blue')
    trace2 = draw_trace_bar(data_df2, color='Lightblue')
    
    fig = tools.make_subplots(rows=1,cols=2, subplot_titles=(title1,title2))
    fig.append_trace(trace1,1,1)
    fig.append_trace(trace2,1,2)
    
    fig['layout']['xaxis'].update(title = xlab)
    fig['layout']['xaxis2'].update(title = xlab)
    fig['layout']['yaxis'].update(title = ylab)
    fig['layout']['yaxis2'].update(title = ylab)
    fig['layout'].update(showlegend=False)
    

    iplot(fig, filename='draw_trace')


# In[ ]:


def plot_survived_bar(data_df, var, ytitle= 'Number of passengers',title= 'Number of survived and not survived passengers by {}'):
    dfS = data_df[data_df['Survived']==1]
    dfN = data_df[data_df['Survived']==0]

    traceS = go.Bar(
        x = dfS[var],y = dfS['Number'],
        name='Survived',
        marker=dict(color="Blue"),
        text=dfS['Number']
    )
    traceN = go.Bar(
        x = dfN[var],y = dfN['Number'],
        name='Not survived',
        marker=dict(color="Red"),
        text=dfS['Number']
    )
    
    data = [traceS, traceN]
    layout = dict(title = title.format(var),
          xaxis = dict(title = var, showticklabels=True), 
          yaxis = dict(title = ytitle),
          hovermode = 'closest'
    )
    fig = dict(data=data, layout=layout)
   
    iplot(fig, filename='draw_trace')


# In[ ]:


plot_two_bar(get_categories(train_df,'Sex'), get_categories(test_df,'Sex'), 
             'Train data', 'Test data',
             'Sex', 'Number of passengers')


# In[ ]:


plot_survived_bar(get_survived_categories(train_df,'Sex'), 'Sex')


# From the total female passengers, 74% survived.  
# In the same time, from the total male passengers, only 18% survived.

# In[ ]:


plot_two_bar(get_categories(train_df,'Age'), get_categories(test_df,'Age'), 
             'Train data', 'Test data',
             'Age', 'Number of passengers')


# In[ ]:


plot_survived_bar(get_survived_categories(train_df,'Age'), 'Age')


# Majority of the passengers were between 20 and 35 years old.
# 
# The survival rate for the age interval 15-35 years was quite small.
# 

# In[ ]:


plot_two_bar(get_categories(train_df,'SibSp'), get_categories(test_df,'SibSp'), 
             'Train data', 'Test data',
             'SibSp', 'Number of passengers')


# In[ ]:


plot_survived_bar(get_survived_categories(train_df,'SibSp'), 'SibSp')


# Most of the passengers traveled alone. From the passengers travelling alone, only 34% survived.
# The passengers with only one or two sibbilings survived in around 50% of the cases. 
# Survival rates decrease considerably for number of sibilings or spouses of 3, 4, 5 and is practically 0 for 8.

# In[ ]:


plot_two_bar(get_categories(train_df,'Parch'), get_categories(test_df,'Parch'), 
             'Train data', 'Test data',
             'Parch', 'Number of passengers')


# In[ ]:


plot_survived_bar(get_survived_categories(train_df,'Parch'), 'Parch')


# For the number of children or parents equal to 0, the survival rate is only 34%. 
# For values of 1,2 and 3, the survival rate is 50%, decreasing for larger numbers.

# <a href="#0"><font size="1">Go to top</font></a>  
# 
# 
# ## <a id='33'>Fare, Embarked, Pclass</a>  
# 
# Let's check now the data (in **train** and **test**) for  `Fare`,  `Embarked` and`Pclass`.  
# 

# In[ ]:


def draw_trace_histogram(data_df,color='Blue'):
    trace = go.Histogram(
            x = data_df['index'],
            y = data_df['Number'],
            marker=dict(color=color),
            text=data_df['index']
        )
    return trace

def plot_two_histogram(data_df1, data_df2, title1, title2, xlab, ylab):
    trace1 = draw_trace_histogram(data_df1, color='Blue')
    trace2 = draw_trace_histogram(data_df2, color='Lightblue')
    
    fig = tools.make_subplots(rows=1,cols=2, subplot_titles=(title1,title2))
    fig.append_trace(trace1,1,1)
    fig.append_trace(trace2,1,2)
    
    fig['layout']['xaxis'].update(title = xlab)
    fig['layout']['xaxis2'].update(title = xlab)
    fig['layout']['yaxis'].update(title = ylab)
    fig['layout']['yaxis2'].update(title = ylab)
    fig['layout'].update(showlegend=False)
    

    iplot(fig, filename='draw_trace')


# In[ ]:


def plot_survived_histogram(data_df, var):
    dfS = data_df[data_df['Survived']==1]
    dfN = data_df[data_df['Survived']==0]

    traceS = go.Histogram(
        x = dfS[var],y = dfS['Number'],
        name='Survived',
        marker=dict(color="Blue"),
        text=dfS['Number']
    )
    traceN = go.Histogram(
        x = dfN[var],y = dfN['Number'],
        name='Not survived',
        marker=dict(color="Red"),
        text=dfS['Number']
    )
    
    data = [traceS, traceN]
    layout = dict(title = 'Number of survived and not survived passengers by {}'.format(var),
          xaxis = dict(title = var, showticklabels=True), 
          yaxis = dict(title = 'Number of passengers'),
          hovermode = 'closest'
    )
    fig = dict(data=data, layout=layout)
   
    iplot(fig, filename='draw_trace')


# In[ ]:


plot_two_histogram(get_categories(train_df,'Fare'), get_categories(test_df,'Fare'), 
             'Train data', 'Test data',
             'Fare', 'Passengers')


# In[ ]:


plot_survived_histogram(get_survived_categories(train_df,'Fare'), 'Fare')


# Survival rate increases considerably with the fare value. This confirm the image that richer people survived better. We will validate this observation as well with the class information.

# In[ ]:


plot_two_bar(get_categories(train_df,'Embarked'), get_categories(test_df,'Embarked'), 
             'Train data', 'Test data',
             'Embarked', 'Number of passengers')


# In[ ]:


plot_survived_bar(get_survived_categories(train_df,'Embarked'), 'Embarked')


# The best survival rate is for passengers embarked in Cherbourg (more than 50%), the worst for passengers embarked in Southampton.

# In[ ]:


plot_two_bar(get_categories(train_df,'Pclass'), get_categories(test_df,'Pclass'), 
             'Train data', 'Test data',
             'Pclass', 'Number of passengers')


# In[ ]:


plot_survived_bar(get_survived_categories(train_df,'Pclass'), 'Pclass')


# First class passengers survived in a percent of 63% while less than 50% survived in 2nd class. For passengers in 3rd class, only 24% survived.

# <a href="#0"><font size="1">Go to top</font></a>  
# 
# 
# ## <a id='34'>Ticket, Cabin, Name</a>  
# 
# Let's check now the data (in **train** and **test**) for  `Ticket`,  `Cabin` and`Name`.
# 
# All these are alphanumeric (contains both letters and numbers), like `Ticket` and `Cabin` or are text fields (`Name`). 
# 
# We will have to process them in order to use as features.  
# 
# Let's look to `Ticket` first.
# 
# 

# In[ ]:


train_df['Ticket'].value_counts().head(10)


# `Ticket` has, most probably, little predictive value.  
# 
# Let's see also the `Cabin`.
# 

# In[ ]:


train_df['Cabin'].value_counts().head(10)


# `Cabin` has the first letter that is, most probably, giving the information on the deck. This might have a predictive value and we will process further in the next sections.
# 
# `Name` might contain multiple information. Let's check few of the `Name` fields.

# In[ ]:


train_df['Name'].head(10)


# We see that in the `Name` field we have the Family name, the title (which might indicate as well the social status or marital status), and the first name. So, `Name` is actually a quite rich feature column, which can be further exploited (and we will exploit in the next sections). 

# ## <a id='35'>Multiple features visualization</a>
# 
# Let's show the number of survived/not survived passengers grouped by Class and Sex.

# In[ ]:


tmp = train_df.groupby(['Pclass', 'Sex'])['Survived'].value_counts()
df = pd.DataFrame(data={'Passengers': tmp.values}, index=tmp.index).reset_index()
hover_text = []
for index, row in df.iterrows():
    hover_text.append(('Pclass: {}<br>'+
                      'Sex: {}<br>'+
                      'Survived: {}<br>'+
                      'Passengers: {}').format(row['Pclass'],
                                            row['Sex'],
                                            row['Survived'],
                                            row['Passengers']))
df['hover_text'] = hover_text


# In[ ]:


trace = go.Scatter(
        x=df['Pclass'],
        y=df['Sex'],
        text=df['hover_text'],
        mode='markers',
        marker=dict(
            sizemode='diameter',
            sizeref=3,
            size=df['Passengers'],
            color = df['Survived'],
            colorscale = 'Bluered',
        )
    )
data = [trace]

layout = dict(title = 'Number of surviving/not surviving passengers by class and sex',
          xaxis = dict(title = 'Class', showticklabels=True, type='category'), 
          yaxis = dict(title = 'Sex', type='category'),            
          hovermode = 'closest',
              height=400, width=600, 
         )
fig=go.Figure(data=data, layout=layout)
iplot(fig, filename='bubble_plot')


# 
# Let's show the number of survived/not survived passengers grouped by SibSp and Parch.

# In[ ]:


tmp = train_df.groupby(['SibSp', 'Parch'])['Survived'].value_counts()
df = pd.DataFrame(data={'Passengers': tmp.values}, index=tmp.index).reset_index()
hover_text = []
for index, row in df.iterrows():
    hover_text.append(('Sibilings: {}<br>'+
                      'Parents/Children: {}<br>'+
                      'Survived: {}<br>'+
                      'Passengers: {}').format(row['SibSp'],
                                            row['Parch'],
                                            row['Survived'],
                                            row['Passengers']))
df['hover_text'] = hover_text


# In[ ]:


trace = go.Scatter(
        x=df['SibSp'],
        y=df['Parch'],
        text=df['hover_text'],
        mode='markers',
        marker=dict(
            sizemode='diameter',
            sizeref=4,
            size=df['Passengers'],
            color = df['Survived'],
            colorscale = 'Bluered',
        )
    )
data = [trace]

layout = dict(title = 'Passengers by number of Sibilings and  parents/children',
          xaxis = dict(title = 'Sibilings', showticklabels=True, type='category'), 
          yaxis = dict(title = 'Parents/Children', type='category'),            
          hovermode = 'closest',
              height=400, width=600, 
         )
fig=go.Figure(data=data, layout=layout)
iplot(fig, filename='bubble_plot')


# ## <a id='36'>Imputation of missing data</a>  
# 
# We are creating a model for imputation of **Fare** data.  
# 
# First, let's create a list with **train** and **test** datasets, to process both in the same time.
# 

# In[ ]:


test_df['Survived'] = None
all_df = pd.concat([train_df, test_df], axis=0)


# In[ ]:


print(f'Combined data: {all_df.shape}')


# In[ ]:


all_df.head()


# Let's create a Decision Tree model to predict missing Fare value.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
def encrypt_single_column(data):
    le = LabelEncoder()
    le.fit(data.astype(str))
    return le.transform(data.astype(str))


# In[ ]:


features = ['Pclass','Sex','Embarked','SibSp','Parch']

for feature in features:
    all_df[feature] = encrypt_single_column(all_df[feature])


# In[ ]:


X = all_df.loc[~(all_df.Fare.isna())]
y = X['Fare'].values
X = X[features]
X_test = all_df.loc[all_df.Fare.isna()]    
X_test = X_test[features]
print(f'X: {X.shape} y: {y.shape}, X_text: {X_test.shape}')


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
clf = DecisionTreeRegressor()
clf.fit(X, y)
y_test = clf.predict(X_test)


# In[ ]:


print(f'Fare: {y_test}')


# We replace the predicted value in the original (combined) data.

# In[ ]:


all_df.loc[all_df.Fare.isna(), 'Fare'] = y_test


# In[ ]:


all_df.loc[all_df.Fare.isna()].shape


# <a href="#0"><font size="1">Go to top</font></a>  
# 
# 
# # <a id='4'>Features engineering</a>
# 
# From the original features, we will create new features.

# ## <a id='41'>Extract Title from Name</a>
# 
# Let's start with processing the names. We will extract the title from the names.  
# 
# We create now a list of datasets (we name it as well all_df, we reuse this name):
# 

# In[ ]:


all_df = [train_df, test_df]


# We apply the rule for extracting the title.

# In[ ]:


for dataset in all_df:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


# Let's verify the relationship between `Title` and `Sex`.

# In[ ]:


np.transpose(pd.crosstab(train_df['Title'], train_df['Sex']))


# There are male only titles: `Capt`, `Col`, `Don`, `Jonkheer`, `Major`, `Master`, `Mr`, `Rev` and `Sir`. 
# As well, there are female only titles: `Countess`, `Lady`, `Miss`,  `Mlle`,  `Mme`, `Mrs`, `Ms`.
# There is a female `Dr` (and other 6 males).
# 
# Most of these titles are quite rare. We will either group them as `Rare` or correct them (for example, to reunite all young womens with the title `Miss`.
# 
# Let's start by grouping all `Miss` and `Mrs` variations under a single name.

# In[ ]:


for dataset in all_df:
    #unify `Miss`
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    #unify `Mrs`
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


# Then, let's set the female Dr as one of the female tipical roles.

# In[ ]:


train_df[(train_df['Title'] == 'Dr') & (train_df['Sex'] == 'female')]


# She is traveling in Cabin `D17` in 1st class. Let's see if she is alone.

# In[ ]:


train_df[train_df['Cabin']=='D17']


# Because she is traveling with a friend about the same age and with a `Mrs` title, we might want to set her as well as a `Mrs`. Let's do it.

# In[ ]:


train_df.loc[train_df.PassengerId == 797, 'Title'] = 'Mrs'


# Let's check if this worked well.

# In[ ]:


train_df[train_df['Cabin']=='D17']


# We succesfully set passenger #797 as a `Mrs`.
# 
# Let's also group all the rare titles under a `Rare` title:

# In[ ]:


for dataset in all_df:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',     'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')


# Let's verify the average survival ratio for the passengers with the aggregated titles and sex.

# In[ ]:


train_df[['Title', 'Sex', 'Survived']].groupby(['Title', 'Sex'], as_index=False).mean()


# In[ ]:


plot_survived_bar(get_survived_categories(train_df,'Title'), 'Title')


#   All rare female titles were saved.   
#   Married females had the highest survival rate besides these very rare cases, of 79%. Young unmarried women followed with 70% survival rate. 
#   Lowest survival rate had the men with `Mr` title. 
#   Between men, the ones with `Master` title had a much higher survival rate than  these, with 57%.

# ## <a id='42'>Build families</a>
# 
# 
# ### Calculate Family Size
# 
# From `SibSp` and `Parch` we create a new feature, `FamilySize`.

# In[ ]:


for dataset in all_df:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1


# Let's check the correlation between family size and survival rate.

# In[ ]:


train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean()


# In[ ]:


plot_survived_bar(get_survived_categories(train_df,'FamilySize'), 'FamilySize')


# Unmarried passengers and passengers with very large families had the lowest survival rate (30% singles and less than 20% the family members with families larger than 5). This can be explained by the fact that singles were most probably mens in lower classes while families with small number of children most probably saved at least a part of them (for example the mother with the childrens) cooperating between them to ensure salvation. Large families might had lower survival rates due to various reasons, including maybe difficulties to coordinate or more difficult decision on whom to embark on the boats (if not possible for all to embark). Another reason for larger families to survive might be related with the class.

# ### Identify families by surname
# 
# Let's also try to aggregate families by surname. For this, we will extract the surname from Name.

# In[ ]:


for dataset in all_df:
    dataset['Surname'] = dataset.Name.str.extract('([A-Za-z]+)\,', expand=False)


# In[ ]:


tmp = train_df.groupby(['Surname'])['Survived'].value_counts()
df = pd.DataFrame(data={'Size of group with same Surname': tmp.values}, index=tmp.index).reset_index().sort_values(['Size of group with same Surname', 'Surname'], ascending=False)


# In[ ]:


tmp = df.groupby(['Size of group with same Surname'])['Survived'].value_counts()
df = pd.DataFrame(data={'Number': tmp.values}, index=tmp.index).reset_index().sort_values(['Size of group with same Surname', 'Survived'], ascending=False)
df


# In[ ]:


plot_survived_bar(df, 'Size of group with same Surname', ytitle= 'Number of groups', title= 'Number of survived and not survived groups by {}')


# The above graph is not showing the number of families survived or not survived. It is grouping family members that survived and family members that did not survived by Surname (family name). We did not checked if there are several different families with the same name or a family appears with a part of the family members in the survived lot and with a part of the family in the not survived lot (which we know it happens frequently, or example adult men in inferior class did not survived at all).

# ## <a id='43'>Extract Deck from Cabin</a>
# 
# We will extract the deck name from the Cabin name by separating the first character from each cabin name. 
# Unfortunatelly, a very small number of passengers have `Cabin` information therefore also the `Deck` information will be only available for a reduced number of passengers.

# In[ ]:


for dataset in all_df:
    dataset['Deck'] = dataset.Cabin.str.extract('^([A-Za-z]+)', expand=False)


# In[ ]:


train_df[['Deck', 'Survived']].groupby(['Deck'], as_index=False).mean()


# In[ ]:


plot_survived_bar(get_survived_categories(train_df,'Deck'), 'Deck')


# The plot shows just the data for the `Deck` information that could be extracted i.e. where a `Cabin` was defined.

# ## <a id='44'>Estimate age</a>
# 
# A relativelly large number of passengers have missing Age information. We will try to estimate this missing information from other available data, `Sex` and `Pclass`. Before doing this, we will also map sex values to numeric values.

# In[ ]:


for dataset in all_df:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


# In[ ]:


age_aprox = np.zeros((2,3))
for dataset in all_df:
    for i in range(0, 2):
        for j in range(0, 3):
            aprox_age = dataset[(dataset['Sex'] == i) &                                   (dataset['Pclass'] == j+1)]['Age'].dropna()
            age_aprox[i,j] = aprox_age.median()
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age'] =                     age_aprox[i,j]

    dataset['Age'] = dataset['Age'].astype(int)


# Now we replaced the missing Age values with the one obtained from the approximations.

# In[ ]:


tmp = train_df.groupby(['Title', 'Pclass'])['Survived'].value_counts()
df = pd.DataFrame(data={'Passengers': tmp.values}, index=tmp.index).reset_index()
df


# ## <a id='44'>More features engineering</a>  
# 
# 
# Let's map all titles to numeric values.

# In[ ]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in all_df:
    dataset['Title'] = dataset['Title'].map(title_mapping)   


# And let's plot the `Title` grouped by the survived and not survived.

# In[ ]:


plot_survived_bar(get_survived_categories(train_df,'Title'), 'Title')


# We also map `Fare` to 4 main fare segments and label them from 0 to 3.

# In[ ]:


for dataset in all_df:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3


# And let's plot the `Fare` clusters grouped by the survived and not survived.

# In[ ]:


plot_survived_bar(get_survived_categories(train_df,'Fare'), 'Fare')


# Similarly, we map `Age` to 5 main segments, labeled from 0 to 4. 

# In[ ]:


for dataset in all_df:
    dataset.loc[ dataset['Age'] <= 16, 'Age']  = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4


# And let's plot the `Age` clusters grouped by the survived and not survived.

# In[ ]:


plot_survived_bar(get_survived_categories(train_df,'Age'), 'Age')


# Family Size is mapped then to only 3 sizes (0 to 2), the first corresponding to the case when someone is alone.

# In[ ]:


for dataset in all_df:
    dataset.loc[ dataset['FamilySize'] <= 1, 'FamilySize'] = 0
    dataset.loc[(dataset['FamilySize'] > 1) & (dataset['FamilySize'] <= 4), 'FamilySize'] = 1
    dataset.loc[ dataset['FamilySize'] > 4, 'FamilySize'] = 2


# Let's also add one more feature, `Class*Age`, calculated as `Class` x `Age`.

# In[ ]:


for dataset in all_df:
    dataset['Class*Age'] = dataset['Pclass'] * dataset['Age']


# And let's plot the `Class*Age` grouped by the survived and not survived.

# In[ ]:


plot_survived_bar(get_survived_categories(train_df,'Class*Age'), 'Class*Age')


# Let's see now how looks like train and test set.

# In[ ]:


train_df.head(3)


# In[ ]:


test_df.head(3)


# <a href="#0"><font size="1">Go to top</font></a>  
# 
# 

# 
# # <a id='5'>Predictive model for survival</a>  
# 
# Let's start with creation of the predictive models. We will create a very simple model for starting.

# 
# ## <a id='50'>Split the data</a>  
# 
# Let's split the training and validation set. We will use a 80-20 split.
# We also set the matrices for train and validation and the vectors with the target values.

# In[ ]:


#We are using 80-20 split for train-test
VALID_SIZE = 0.2
#We also use random state for reproducibility
RANDOM_STATE = 2018

train, valid = train_test_split(train_df, test_size=VALID_SIZE, random_state=RANDOM_STATE, shuffle=True )


# 
# ## <a id='51'>Build a baseline model</a>  

# We will start with a simple model, with just few predictors.
# 
# Let's set now the predictors list and the target value. We start with two predictors, the `Sex` and `Pclass`.

# In[ ]:


predictors = ['Sex', 'Age']
target = 'Survived'


# In[ ]:


train_X = train[predictors]
train_Y = train[target].values
valid_X = valid[predictors]
valid_Y = valid[target].values


# Let's prepare a simple model, using Random Forest. We set few algorithm parameters and initialize a clasiffier.

# In[ ]:


RFC_METRIC = 'gini'  #metric used for RandomForrestClassifier
NUM_ESTIMATORS = 100 #number of estimators used for RandomForrestClassifier
NO_JOBS = 4 #number of parallel jobs used for RandomForrestClassifier


# In[ ]:


clf = RandomForestClassifier(n_jobs=NO_JOBS, 
                             random_state=RANDOM_STATE,
                             criterion=RFC_METRIC,
                             n_estimators=NUM_ESTIMATORS,
                             verbose=False)


# Then, we fit the classifier with the train data prepared before.

# In[ ]:


clf.fit(train_X, train_Y)


# In[ ]:


preds = clf.predict(valid_X)


# Let's plot the features importance. This shows the relative importance of the predictors features for the current model. With this information, we are able to select the features we will use for our gradually refined models.

# In[ ]:


def plot_feature_importance():
    tmp = pd.DataFrame({'Feature': predictors, 'Feature importance': clf.feature_importances_})
    tmp = tmp.sort_values(by='Feature importance',ascending=False)
    plt.figure(figsize = (7,4))
    plt.title('Features importance',fontsize=14)
    s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
    s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.show()   


# In[ ]:


plot_feature_importance()


# <a href="#0"><font size="1">Go to top</font></a>  
# 
# 
# ## <a id='52'>Evaluate the model</a>  
# 
# Let's evaluate the model performance. 
# 
# We are evaluating first the accuracy for the train set.
# 
# 

# In[ ]:


clf.score(train_X, train_Y)
acc = round(clf.score(train_X, train_Y) * 100, 2)
print("RandomForest accuracy (train set):", acc)


# The result means that the total number of correct predictions divided by the total number of examples in the training set is around 0.77. Because we have a binary classification, this means:
# 
# $$Accuracy =\frac{ \textrm{True Positives} + \textrm{True Negatives}}{\textrm{Number of Examples}}$$ 
# 
# 
# Then we evaluate the accuracy for the validation set. 

# In[ ]:


clf.score(valid_X, valid_Y)
acc = round(clf.score(valid_X, valid_Y) * 100, 2)
print("RandomForest accuracy (validation set):", acc)


# The accuracy for the validation is much better than the accuracy for the training set. 
# This means we have higher bias than variation. The model does not learn too well the training set, we are not overfitting (yet). The validation is better, i.e. we are generalizing well. Before improving the variation, we will try to improve the bias, while looking how this affects the variation as well.
# 
# Let's plot now the classification report for validation data.

# In[ ]:


print(metrics.classification_report(valid_Y, preds, target_names=['Not Survived', 'Survived']))


# There are few values given in this report / each category in the target (`Survived` or `Not survived`):
# 
# * **Precision**  
# * **Recall**  
# * **F1-score**  
# 
# Let's explain each of them:
# 
# * **Precision** identifies the frequency with which a model was correct when predicting the positive class. That is:
# 
# $$Precision = \frac{\textrm{True Positives}}{\textrm{True Positives} + \textrm{False Positives}}$$
# 
# * **Recall** answers the following question: Out of all the possible positive labels, how many did the model correctly identify? That is:
# 
# $$Recall = \frac{\textrm{True Positives}}{\textrm{True Positives} + \textrm{False Negatives}}$$
# 
# * **F1-score** is the harmonic mean of **precision** and **recall**.
# 
# $$\textrm{F1-score} = 2  \frac{Precision * Recall}{Precision + Recall}$$
# 
# 
# Precison is better for the `Not survived` as well as Recall and F1-score.
# 
# Let's also show the confusion matrix.

# In[ ]:


def plot_confusion_matrix():
    cm = pd.crosstab(valid_Y, preds, rownames=['Actual'], colnames=['Predicted'])
    fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))
    sns.heatmap(cm, 
                xticklabels=['Not Survived', 'Survived'],
                yticklabels=['Not Survived', 'Survived'],
                annot=True,ax=ax1,
                linewidths=.2,linecolor="Darkblue", cmap="Blues")
    plt.title('Confusion Matrix', fontsize=14)
    plt.show()


# In[ ]:


plot_confusion_matrix()


# <a href="#0"><font size="1">Go to top</font></a>  
# 
# ## <a id='53'>Model refinement</a> 
# 
# Let's rebuild the model and add more features to it.

# ### Model with {Sex, Age, Pclass, Fare} features

# In[ ]:


predictors = ['Sex', 'Age', 'Pclass', 'Fare']
target = 'Survived'


# In[ ]:


train_X = train[predictors]
train_Y = train[target].values
valid_X = valid[predictors]
valid_Y = valid[target].values


# We fit the model.

# In[ ]:


clf.fit(train_X, train_Y)


# We predict the validation set.

# In[ ]:


preds = clf.predict(valid_X)


# Let's plot feature importance.

# In[ ]:


plot_feature_importance()


# Let's see the accuracy for the training set and for the validation set.

# In[ ]:


clf.score(train_X, train_Y)
acc = round(clf.score(train_X, train_Y) * 100, 2)
print("RandomForest accuracy (train set):", acc)


# In[ ]:


clf.score(valid_X, valid_Y)
acc = round(clf.score(valid_X, valid_Y) * 100, 2)
print("RandomForest accuracy (validation set):", acc)


# Let's see also the classification report for the validation set.

# In[ ]:


print(metrics.classification_report(valid_Y, preds, target_names=['Not Survived', 'Survived']))


# Here we plot the confusion matrix.

# In[ ]:


plot_confusion_matrix()


# The model performance with train data improved, i.e. the model is now representing better the training set.  The accuracy and precision with the validation set was as well improved. The recall and f1-score are smaller for `Survived`. 
# 
# Let's repeat the experiment adding more features.

# ### Model with {Sex, Age, Pclass, Fare, Parch, SibSp} features

# In[ ]:


predictors = ['Sex', 'Age', 'Pclass', 'Fare', 'Parch', 'SibSp']
target = 'Survived'


# In[ ]:


train_X = train[predictors]
train_Y = train[target].values
valid_X = valid[predictors]
valid_Y = valid[target].values


# Let's fit the model with the new predictors.

# In[ ]:


clf.fit(train_X, train_Y)


# We predict the validation set.

# In[ ]:


preds = clf.predict(valid_X)


# We plot the feature importance.

# In[ ]:


plot_feature_importance()


# **Sex** is the dominant feature, followed by **Pclass** and **Fare**.  
# Let's see the accuracy for the training set and also for the validation set classification.

# In[ ]:


clf.score(train_X, train_Y)
acc = round(clf.score(train_X, train_Y) * 100, 2)
print("RandomForest accuracy (train set):", acc)


# In[ ]:


clf.score(valid_X, valid_Y)
acc = round(clf.score(valid_X, valid_Y) * 100, 2)
print("RandomForest accuracy (validation set):", acc)


# Let's also plot the classification report for the validation set.

# In[ ]:


print(metrics.classification_report(valid_Y, preds, target_names=['Not Survived', 'Survived']))


# In[ ]:


plot_confusion_matrix()


# The accuracy for the train data classification improved further but the accuracy and precision for validation did not improved.    
# 
# Let's add few more engineering features.
# 
# 
# ### Model with {Sex, Age, Pclass, Fare, Parch, SibSp, FamilySize, Title} features

# In[ ]:


predictors = ['Sex', 'Age', 'Pclass', 'Fare', 'Parch', 'SibSp', 'FamilySize', 'Title']
target = 'Survived'


# In[ ]:


train_X = train[predictors]
train_Y = train[target].values
valid_X = valid[predictors]
valid_Y = valid[target].values


# Let's fit the model with the new data.

# In[ ]:


clf.fit(train_X, train_Y)


# We predict the validation set.

# In[ ]:


preds = clf.predict(valid_X)


# Let's plot also the feature importance.

# In[ ]:


plot_feature_importance()


# Let's see the train set and validation set classification accuracy.

# In[ ]:


clf.score(train_X, train_Y)
acc = round(clf.score(train_X, train_Y) * 100, 2)
print("RandomForest accuracy (train set):", acc)


# In[ ]:


clf.score(valid_X, valid_Y)
acc = round(clf.score(valid_X, valid_Y) * 100, 2)
print("RandomForest accuracy (validation set):", acc)


# The classification report for the validation set.

# In[ ]:


print(metrics.classification_report(valid_Y, preds, target_names=['Not Survived', 'Survived']))


# The confusion matrix.

# In[ ]:


plot_confusion_matrix()


# The training set accuracy improved even more. In the same time, validation score is not very much improved. This indicates that most probably the model where we added more features is overfitting on the training set. Actually, the best model obtained until now was the one with {Sex, Age, Pclass, Fare} features, where accuracies for training set and validation set were 82% and 86% (with quite good precision for all categories and small recall for `Survived`).
# 
# Let's try with a simpler model.
# 
# 
# ### Model with {Title, FamilySize, Pclass} features
# 
# The simple model we will try now, with only three features, is actually using only engineered features.

# In[ ]:


predictors = ['FamilySize', 'Title', 'Class*Age']
target = 'Survived'


# In[ ]:


train_X = train[predictors]
train_Y = train[target].values
valid_X = valid[predictors]
valid_Y = valid[target].values


# We fit the model.

# In[ ]:


rf_clf = clf.fit(train_X, train_Y)


# Let's plot the parameters for the classifier.

# In[ ]:


rf_clf


# We predict the validation set.

# In[ ]:


preds = clf.predict(valid_X)


# We check the features importance.

# In[ ]:


plot_feature_importance()


# In[ ]:


clf.score(train_X, train_Y)
acc = round(clf.score(train_X, train_Y) * 100, 2)
print("RandomForest accuracy (train set):", acc)


# In[ ]:


clf.score(valid_X, valid_Y)
acc = round(clf.score(valid_X, valid_Y) * 100, 2)
print("RandomForest accuracy (validation set):", acc)


# The training set classification accuracy is still good, although lower than that of the previous, more complex model.
# The validation set accuracy is the best obtained until now.
# 
# **Title** is the most important feature.  
# 
# Let's check now the classification report for the validation set.

# In[ ]:


print(metrics.classification_report(valid_Y, preds, target_names=['Not Survived', 'Survived']))


# We see that we obtained an substantial improvement of the classification precision for the validation set.   Also the recall is better than in the case of the model with best accuracy and precision until now, besides this one.
# 
# Let's also plot the confusion matrix.

# In[ ]:


plot_confusion_matrix()


# We will pick this last model for submission.
# It doesn't have the best accuracy for train set classification but have one of the best precision and accuracy for the validation set and also the smallest recall. 
# 
# Let's prepare the submission.

# <a href="#0"><font size="1">Go to top</font></a>  
# 
# ## <a id='54'>Submission</a>
# 
# 
# First, we predict for test data using the trained model.

# In[ ]:


test_X = test_df[predictors]
pred_Y = clf.predict(test_X)


# Then, we prepare the submission dataset and export it in the submission file.

# In[ ]:


submission = pd.DataFrame({"PassengerId": test_df["PassengerId"],"Survived": pred_Y})
submission.to_csv('submission.csv', index=False)


# The precision obtained for the test set is approx. **0.79**.

# <a href="#0"><font size="1">Go to top</font></a>  
# 
# ## <a id='55'>Hyperparameters optimization</a>
# 
# 
# Let's continue with tunning the model hyperparameters.   
# We define a set of parameters with several values and will run an algorithm called Gradient Search to detect the best combination of parameters for our model.  
# First, let's fit the model and assign the output of it to **rf_clf**.

# In[ ]:


rf_clf = clf.fit(train_X, train_Y)


# Let's initialize the GradientSearchCV parameters. We will set only few parameters, as following:
# 
# * **n_estimators**: number of trees in the foreset;  
# * **max_features**: max number of features considered for splitting a node;  
# * **max_depth**: max number of levels in each decision tree;  
# * **min_samples_split**: min number of data points placed in a node before the node is split;  
# * **min_samples_leaf**: min number of data points allowed in a leaf node.
# 

# In[ ]:


parameters = {
    'n_estimators': (50, 75,100),
    'max_features': ('auto', 'sqrt'),
    'max_depth': (3,4,5),
    'min_samples_split': (2,5,10),
    'min_samples_leaf': (1,2,3)
}


# We initialize GridSearchCV with the classifier, the set of parameters, number of folds and also the level of verbose for printing out progress.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'gs_clf = GridSearchCV(rf_clf, parameters, n_jobs=-1, cv = 5, verbose = 5)\ngs_clf = gs_clf.fit(train_X, train_Y)')


# Let's see the best parameters.

# In[ ]:


print('Best scores:',gs_clf.best_score_)
print('Best params:',gs_clf.best_params_)


# Let's predict with the validation data.

# In[ ]:


preds = gs_clf.predict(valid_X)


# Let's check the accuracy for the validation set.

# In[ ]:


clf.score(valid_X, valid_Y)
acc = round(clf.score(valid_X, valid_Y) * 100, 2)
print("RandomForest accuracy (validation set):", acc)


# Let's check the validation classification report.

# In[ ]:


print(metrics.classification_report(valid_Y, preds, target_names=['Not Survived', 'Survived']))


# ## <a id='56'>Submission (model with hyperparameters optimization)</a>
# 
# 
# First, we predict for test data using the trained model.
# 

# In[ ]:


test_X = test_df[predictors]
pred_Y = gs_clf.predict(test_X)


# Then we prepare the submission dataset and save it to the submission file.

# In[ ]:


submission = pd.DataFrame({"PassengerId": test_df["PassengerId"],"Survived": pred_Y})
submission.to_csv('submission_hyperparam_optimization.csv', index=False)


# <a href="#0"><font size="1">Go to top</font></a>  

# 
# # <a id='6'>Model ensambling</a>
# 
# 
# Let's continue with creation of second level models. We will train several models and will then use these first level models to train a second level model. This method is powerfull and can enhance the performance of first level models, especially when there is little correlation between the results of first level models.
# 

# ## <a id='61'>Create the ensamble framework</a>
# 
# 
# We start by creating a generic classifier, that extends the functionality of a simple classifier. This generic classifier will be instanciated with few different first level classifiers and then used in the ensamble. We are also using cross-validation (with KFolds).
# 
# First step will be to create the folds used in cross validation.

# In[ ]:


NUMBER_KFOLDS = 5
kf = KFold(n_splits = NUMBER_KFOLDS, random_state = RANDOM_STATE, shuffle = True)


# In[ ]:


# Class to extend the Sklearn classifier
class SklearnBasicClassifier(object):
    def __init__(self, clf, seed=2018, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)

    def get_feature_importances(self,x,y):
        return (self.clf.fit(x,y).feature_importances_)


# ## <a id='62'>Create the Out-of-Fold Predictions</a>
# 
# Let's now define out-of-folds predictions. If we would train the base models on the full training data and generate predictions on the full test set and then output these for the second-level training we might go into trouble. The risk here is that the base model predictions would have seen the test set and thus overfitting when feeding those predictions.

# In[ ]:


ntrain = train_df.shape[0]
ntest = test_df.shape[0]
def get_oof_predictions(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NUMBER_KFOLDS, ntest))
    
    for i, (train_idx, valid_idx) in enumerate(kf.split(train_df)):
        clf.train(x_train[train_idx], y_train[train_idx])
        oof_train[valid_idx] = clf.predict(x_train[valid_idx])
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# ## <a id='63'>Train the first level models</a>  
# 
# 
# Let's define and train few first level models.
# 
# We will use the following first level models:
# 
# 
# * AdaBoost classifer
# * CatBoost Classifier
# * Extra Trees classifier  
# * Gradient Boosting classifer
# * Random Forest classifier  
# * Support Vector Machine
# 
# 
# Let's define the parameters used for training each classifier:

# In[ ]:


# AdaBoost parameters
ada_params = {
    'n_estimators': 200,
    'learning_rate' : 0.75
}
# CatBoost parameters
cat_params = {
    'iterations': 150,
    'learning_rate': 0.02,
    'depth': 12,
    'bagging_temperature':0.2,
    'od_type':'Iter',
    'metric_period':400,
}  
# Extra Trees Parameters
ext_params = {
    'n_jobs': -1,
    'n_estimators':100,
    'max_depth': 8,
    'min_samples_leaf': 3,
    'verbose': 0
}
# Gradient Boosting parameters
gbm_params = {
    'n_estimators': 200,
    'max_depth': 5,
    'min_samples_leaf': 3,
    'verbose': 0
}
# Random Forest parameters
rfo_params = {
    'n_jobs': -1,
    'n_estimators': 50,
    'max_depth': 5,
    'min_samples_leaf': 5,
    'max_features' : 'auto',
    'verbose': 0
}
# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.02
}


# We create 6 objects of type `SklearnBasicClassifier` that represent our 6 models(AdaBoost, CatBoost, ExtraTrees, GradientBoosting, RandomForest and Support Vector Machines).

# In[ ]:


# Create 6 objects that represent our 6 models
ada = SklearnBasicClassifier(clf=AdaBoostClassifier, seed=RANDOM_STATE, params=ada_params)
cat = SklearnBasicClassifier(clf=CatBoostClassifier, seed=RANDOM_STATE, params=cat_params)
ext = SklearnBasicClassifier(clf=ExtraTreesClassifier, seed=RANDOM_STATE, params=ext_params)
gbm = SklearnBasicClassifier(clf=GradientBoostingClassifier, seed=RANDOM_STATE, params=gbm_params)
rfo = SklearnBasicClassifier(clf=RandomForestClassifier, seed=RANDOM_STATE, params=rfo_params)
svc = SklearnBasicClassifier(clf=SVC, seed=RANDOM_STATE, params=svc_params)


# In[ ]:


predictors = ['FamilySize', 'Title', 'Class*Age']
target = 'Survived'


# In[ ]:


y_train = train_df['Survived'].values
train = train_df[predictors]
test = test_df[predictors]
x_train = train.values
x_test = test.values


# In[ ]:


print("Start training")
ada_oof_train, ada_oof_test = get_oof_predictions(ada, x_train, y_train, x_test) # AdaBoost Classifier
print("End AdaBoost")
cat_oof_train, cat_oof_test = get_oof_predictions(cat, x_train, y_train, x_test) # CatBoost Classifier
print("End CatBoost")
ext_oof_train, ext_oof_test = get_oof_predictions(ext, x_train, y_train, x_test) # Extra Trees 
print("End ExtraTrees")
rfo_oof_train, rfo_oof_test = get_oof_predictions(rfo,x_train, y_train, x_test) # Random Forest Classifier
print("End RandomForest")
gbm_oof_train, gbm_oof_test = get_oof_predictions(gbm,x_train, y_train, x_test) # Gradient Boost Classifier
print("End GradientBoost")
svc_oof_train, svc_oof_test = get_oof_predictions(svc,x_train, y_train, x_test) # Support Vector Classifier
print("End training")


# Let's check the features importance for the 5 out of 6 models. We will not include Support Vector Classifier, this one does not have feature importance available.

# In[ ]:


ada_feature_importance = ada.get_feature_importances(x_train,y_train)
cat_feature_importance = cat.get_feature_importances(x_train,y_train)
ext_feature_importance = ext.get_feature_importances(x_train,y_train)
gbm_feature_importance = gbm.get_feature_importances(x_train,y_train)
rfo_feature_importance = rfo.get_feature_importances(x_train,y_train)


# In[ ]:


def plot_feature_importance(feature_importance, classifier):
    tmp = pd.DataFrame({'Feature': predictors, 'Feature importance': feature_importance[0:len(predictors)]})
    tmp = tmp.sort_values(by='Feature importance',ascending=False)
    plt.figure(figsize = (7,4))
    plt.title('Features importance {}'.format(classifier),fontsize=14)
    s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
    s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.show()   


# In[ ]:


plot_feature_importance(ada_feature_importance, '- AdaBoost')
plot_feature_importance(cat_feature_importance, '- CatBoost')
plot_feature_importance(ext_feature_importance, '- ExtraTrees')
plot_feature_importance(gbm_feature_importance, '- GradientBoosting')
plot_feature_importance(rfo_feature_importance, '- RandomForest')


# ## <a id='64'>Correlation of the results</a>
# 
# Let's see the first few results of the predictions using first level models.

# In[ ]:


base_predictions_train = pd.DataFrame( {
     'AdaBoost': ada_oof_train.ravel(),
     'CatBoost': cat_oof_train.ravel(),
     'ExtraTrees': ext_oof_train.ravel(),
     'GradientBoost': gbm_oof_train.ravel(),
     'RandomForest': rfo_oof_train.ravel(),
     'SVM': svc_oof_train.ravel()
    })
base_predictions_train.head(10)


# Let's show now the correlation of the predictions using the first level models. The ensamble prediction is best when we have models with good accuracy and less correlated.

# In[ ]:


trace = go.Heatmap(
        z= base_predictions_train.astype(float).corr().values ,
        x=base_predictions_train.columns.values,
        y= base_predictions_train.columns.values,
          colorscale='Rainbow',
            showscale=True,
            reversescale = False
    )
data = [trace]
layout = dict(width = 600, height=600)
fig = dict(data=data, layout=layout)
iplot(fig, filename='heatmap')


# ## <a id='65'>Build the second level (ensamble) model</a>
# 
# 
# We prepare now, using the Out-Of-Folds values, the training and the test set for the second level model. We concatenate the OOFs from the 6 first level models.

# In[ ]:


x_train = np.concatenate(( ada_oof_train, cat_oof_train, ext_oof_train, gbm_oof_train, rfo_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate(( ada_oof_test, cat_oof_test, ext_oof_test, gbm_oof_test, rfo_oof_test, svc_oof_test), axis=1)


# We prepare as well the second level classifier. 
# 
# We will use in this case a eXtreme Boost Classifier.

# In[ ]:


clf = xgb.XGBClassifier(
 learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1)


# We fit the model.

# In[ ]:


xgbm = clf.fit(x_train, y_train)


# With the fitted second level model we do the prediction for the test data.

# In[ ]:


predictions = xgbm.predict(x_test)


# ## <a id='66'>Submission (ensamble)</a>
# 
# We form the submission dataset and save it to the submission file.

# In[ ]:


submissionStacking = pd.DataFrame({ 'PassengerId': test_df["PassengerId"],'Survived': predictions })
submissionStacking.to_csv("submission_ensamble.csv", index=False)


# <a href="#0"><font size="1">Go to top</font></a>  
# 
# 

# # <a id='7'>References</a>
# 
# [1] https://www.kaggle.com/startupsci/titanic-data-science-solutions  
# [2] https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python  
# [3] https://www.kaggle.com/gpreda/credit-card-fraud-detection-predictive-models  
# [4] https://www.kaggle.com/gpreda/honey-bee-subspecies-classification  
