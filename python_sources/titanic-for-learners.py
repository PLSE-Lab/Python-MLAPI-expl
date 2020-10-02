#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# To learn about notebook findings visit https://timewithai.com/2019/06/29/how-to-think-like-a-data-scientist/

# <h1>Define the Problem</h1>
# <p>The sinking of the RMS Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# <p>One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# <p>In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. 

# <h1>Module Import</h1>

# In[ ]:


# Load in our libraries
import pandas as pd #dataframe
import numpy as np #numerical process
import re #random play
import sklearn #ML god
import seaborn as sns #plot
import matplotlib.pyplot as plt #plot
import matplotlib as mpl
import matplotlib.pylab as pylab

# it's a library that we work with plotly
import plotly.offline as py 
py.init_notebook_mode(connected=True) # this code, allow us to work with offline plotly version
import plotly.graph_objs as go # it's like "plt" of matplot
import plotly.tools as tls # It's useful to we get some tools of plotly
import warnings # This library will be used to ignore some warnings
from collections import Counter # To do counter of some features

import warnings #ignore warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8


# In[ ]:


from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics


# <h1>Understand The data</h1>
# 
# 1. The Survived variable is our outcome or dependent variable. It is a binary nominal datatype of 1 for survived and 0 for did not survive. All other variables are potential predictor or independent variables. It's important to note, more predictor variables do not make a better model, but the right variables.
# 2. The PassengerID and Ticket variables are assumed to be random unique identifiers, that have no impact on the outcome variable. Thus, they will be excluded from analysis.
# 3. The Pclass variable is an ordinal datatype for the ticket class, a proxy for socio-economic status (SES), representing 1 = upper class, 2 = middle class, and 3 = lower class.
# 4. The Name variable is a nominal datatype. It could be used in feature engineering to derive the gender from title, family size from surname, and SES from titles like doctor or master. Since these variables already exist, we'll make use of it to see if title, like master, makes a difference.
# 5. The Sex and Embarked variables are a nominal datatype. They will be converted to dummy variables for mathematical calculations.
# 6. The Age and Fare variable are continuous quantitative datatypes.
# 7. The SibSp represents number of related siblings/spouse aboard and Parch represents number of related parents/children aboard. Both are discrete quantitative datatypes. This can be used for feature engineering to create a family size and is alone variable.
# 8. The Cabin variable is a nominal datatype that can be used in feature engineering for approximate position on ship when the incident occurred and SES from deck levels. However, since there are many null values, it does not add value and thus is excluded from analysis.

# In[ ]:


pd.set_option("display.max_columns",15)
#import training data
train=pd.read_csv("../input/train.csv")
#import testing data
test=pd.read_csv("../input/test.csv")

print(train.head())
print(test.head())


# <h1>Basic data utils</h1>
# 
# 

# In[ ]:


train.info()


# In[ ]:


test.info()


# In both the cases we have Age column ,cabin,Embarked and Fare as missing values

# In[ ]:


train.describe()


# In[ ]:


train.describe(include=["O"])


# In[ ]:


test.describe()


# In[ ]:


test.describe(include=["O"])


# By looking at the data we can understand that passengerID, Name And Ticket is basic informations and can't contribute to our target so get rid of them like your old school gf.

# In[ ]:


train_df=train.drop(["Name","Ticket","PassengerId"],axis=1)
test_df=test.drop(["Name","Ticket","PassengerId"],axis=1)


# In[ ]:


train_df.head(),test_df.head()


# Well Now we have the basic 8 variables to predict the survival rate

# <h1>Handleing Missing Values</h1>
# 
# <p>we are using Age missing values as its median ,Embarkment as mode and fare as median for the reason stated in visualization 

# <h1>Target Analysis</h1>

# In[ ]:


trace0 = go.Bar(
            x = train_df[train_df["Survived"]== 1]["Survived"].value_counts().index.values,
            y = train_df[train_df["Survived"]==1]["Survived"].value_counts().values,
            name='Survived'
    )

trace1 = go.Bar(
            x = train_df[train_df["Survived"]== 0]["Survived"].value_counts().index.values,
            y = train_df[train_df["Survived"]== 0]["Survived"].value_counts().values,
            name='Not Survived'
    )

data = [trace0, trace1]

layout = go.Layout(
    
)

layout = go.Layout(
    yaxis=dict(
        title='Count'
    ),
    xaxis=dict(
        title='Survival Variable'
    ),
    title='Target variable distribution'
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='grouped-bar')


# In[ ]:


labels = ['Not Survived','Survived']
size = train_df['Survived'].value_counts()
colors = ['red', 'green']
explode = [0, 0.1]

plt.rcParams['figure.figsize'] = (7, 7)
plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')
plt.title('A pie chart Representing the Survived population')
plt.axis('off')
plt.legend()
plt.show()


# So our train data contains data where around 38.5% persons are surviving out of 100%

# <h1>Visualization</h1>

# In[ ]:


df_survived = train_df.loc[train_df["Survived"] == 1]['Age'].values.tolist()
df_notsurvived = train_df.loc[train_df["Survived"] == 0]['Age'].values.tolist()
df_age = train_df['Age'].values.tolist()

#First plot
trace0 = go.Histogram(
    x=df_survived,
    histnorm='probability',
    name="survived Persons"
)
#Second plot
trace1 = go.Histogram(
    x=df_notsurvived,
    histnorm='probability',
    name="Died Persons"
)
#Third plot
trace2 = go.Histogram(
    x=df_age,
    histnorm='probability',
    name="Overall Age"
)

#Creating the grid
fig = tls.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],
                          subplot_titles=('Survived','Died', 'General Distribuition'))

#setting the figs
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig.append_trace(trace2, 2, 1)

fig['layout'].update(showlegend=True, title='Age Distribuition', bargap=0.05)
py.iplot(fig, filename='custom-sized-subplot-with-subplot-titles')


# In[ ]:


# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(train_df, col='Survived',palette="hue",row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


# In[ ]:


# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(train_df, col='Sex',palette="hue",row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist,'Survived', alpha=.5, bins=20)
grid.add_legend();


# In[ ]:


df_survived = train_df[train_df["Survived"] == 1]
df_notsurvived = train_df[train_df["Survived"] == 0]

fig, ax = plt.subplots(nrows=2, figsize=(12,8))
plt.subplots_adjust(hspace = 0.4, top = 0.8)

g1 = sns.distplot(df_survived["Age"].dropna(), ax=ax[0], 
             color="g")
g1 = sns.distplot(df_notsurvived["Age"].dropna(), ax=ax[0], 
             color='r')
g1.set_title("Age Distribuition", fontsize=15)
g1.set_xlabel("Age")
g1.set_xlabel("Frequency")

g2 = sns.countplot(x="Age",data=train_df, 
              palette="hls", ax=ax[1], 
              hue = "Survived")
g2.set_title("Age Counting by survival rate", fontsize=15)
g2.set_xlabel("Age")
g2.set_xlabel("Count")
plt.show()


# In[ ]:


#Let's look the Credit Amount column
interval = (0, 20, 40, 60, 80)

cats = ['Child', 'Young', 'Mature', 'Senior']
train_df["Age_cat"] = pd.cut(train_df.Age, interval, labels=cats)


df_survived = train_df[train_df["Survived"] == 1]
df_notsurvived = train_df[train_df["Survived"] == 0]


# In[ ]:


trace0 = go.Box(
    y=df_survived["Fare"],
    x=df_survived["Age_cat"],
    name='Survived',
    marker=dict(
        color='#3D9970'
    )
)

trace1 = go.Box(
    y=df_notsurvived['Fare'],
    x=df_notsurvived['Age_cat'],
    name='NotSurvived',
    marker=dict(
        color='#FF4136'
    )
)
    
data = [trace0, trace1]

layout = go.Layout(
    yaxis=dict(
        title='Fare Amount',
        zeroline=False
    ),
    xaxis=dict(
        title='Age Categorical'
    ),
    boxmode='group'
)
fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='box-age-cat')


# In[ ]:


df_male = train_df[train_df["Sex"] == "male"]
df_female = train_df[train_df["Sex"] == "female"]


trace0 = go.Box(
    y=df_male["Fare"],
    x=df_male["Age_cat"],
    name='Male',
    marker=dict(
        color='#3D9970'
    )
)

trace1 = go.Box(
    y=df_female['Fare'],
    x=df_female['Age_cat'],
    name='Female',
    marker=dict(
        color='#FF4136'
    )
)
    
data = [trace0, trace1]

layout = go.Layout(
    yaxis=dict(
        title='Fare Amount',
        zeroline=False
    ),
    xaxis=dict(
        title='Age Categorical'
    ),
    boxmode='group'
)
fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='box-age-cat-sex')


# In[ ]:


df_1 = train_df[train_df["Pclass"] == 1]
df_2 = train_df[train_df["Pclass"] == 2]
df_3=train_df[train_df["Pclass"] == 3]


trace0 = go.Box(
    y=df_1["Fare"],
    x=df_1["Age_cat"],
    name='Pclass1',
    marker=dict(
        color='#3D9970'
    )
)

trace1 = go.Box(
    y=df_2['Fare'],
    x=df_2['Age_cat'],
    name='Pclass2',
    marker=dict(
        color='#FF4136'
    )
    
)

trace2 = go.Box(
    y=df_3['Fare'],
    x=df_3['Age_cat'],
    name='Pclass3',
    marker=dict(
        color='black'
    )
    
)
    
data = [trace0, trace1,trace2]

layout = go.Layout(
    yaxis=dict(
        title='Fare Amount',
        zeroline=False
    ),
    xaxis=dict(
        title='Age Categorical'
    ),
    boxmode='group'
)
fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='box-age-cat-pclass')


# In[ ]:


# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(train_df,palette="hue",row='Pclass',col="Sex", size=2.2, aspect=1.6)
grid.map(plt.hist, 'Fare', alpha=.5, bins=20)
grid.add_legend();


# In[ ]:


#First plot
trace0 = go.Bar(
    x =train_df[train_df["Survived"]== 1]["Pclass"].value_counts().index.values,
    y = train_df[train_df["Survived"]== 1]["Pclass"].value_counts().values,
    name='Survived',marker=dict(color="black")
)

#Second plot
trace1 = go.Bar(
    x = train_df[train_df["Survived"]== 0]["Pclass"].value_counts().index.values,
    y = train_df[train_df["Survived"]== 0]["Pclass"].value_counts().values,
    name="Not Survived",marker=dict(color="darkred")
)

data = [trace0, trace1]

layout = go.Layout(
    title='class Distribuition'
)


fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='class-Grouped')


# In[ ]:


fig = {
    "data": [
        {
            "type": 'violin',
            "x": df_survived['Pclass'],
            "y": df_survived['Fare'],
            "legendgroup": 'Survived',
            "scalegroup": 'No',
            "name": 'Survived',
            "side": 'negative',
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            },
            "line": {
                "color": 'blue'
            }
        },
        {
            "type": 'violin',
            "x": df_notsurvived['Pclass'],
            "y": df_notsurvived['Fare'],
            "legendgroup": 'Not Survived',
            "scalegroup": 'No',
            "name": 'Not Survived',
            "side": 'positive',
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            },
            "line": {
                "color": 'green'
            }
        }
    ],
    "layout" : {
        "yaxis": {
            "zeroline": False,
        },
        "violingap": 0,
        "violinmode": "overlay"
    }
}


py.iplot(fig, filename = 'violin/split', validate = False)


# In[ ]:


fig = {
    "data": [
        {
            "type": 'box',
            "x": df_survived['Pclass'],
            "y": df_survived['Fare'],
        "legendgroup": 'Survived',
            "scalegroup": 'No',
            "name": 'Survived',
            "side": 'negative',
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            },
            "line": {
                "color": 'red'
            }} ,
        {
            "type": 'box',
            "x": df_notsurvived['Pclass'],
            "y": df_notsurvived['Fare'],"legendgroup": 'Not Survived',
            "scalegroup": 'No',
            "name": 'Not Survived',
            "side": 'positive',
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            },
            "line": {
                "color": 'yellow'
            }
        }],
    "layout" : {
        "yaxis": {
            "zeroline": False,
        }
    }
}


py.iplot(fig, filename = 'violin/split', validate = False)


# In[ ]:


#First plot
trace0 = go.Bar(
    x = train_df[train_df["Survived"]== 1]["Sex"].value_counts().index.values,
    y = train_df[train_df["Survived"]== 1]["Sex"].value_counts().values,
    name='Survived'
)

#First plot 2
trace1 = go.Bar(
    x = train_df[train_df["Survived"]== 0]["Sex"].value_counts().index.values,
    y = train_df[train_df["Survived"]== 0]["Sex"].value_counts().values,
    name="Not Survived"
)

#Second plot
trace2 = go.Box(
    x = train_df[train_df["Survived"]== 1]["Sex"],
    y = train_df[train_df["Survived"]== 1]["Fare"],
    name=trace0.name
)

#Second plot 2
trace3 = go.Box(
    x = train_df[train_df["Survived"]== 0]["Sex"],
    y = train_df[train_df["Survived"]== 0]["Fare"],
    name=trace1.name
)

data = [trace0, trace1, trace2,trace3]


fig = tls.make_subplots(rows=1, cols=2, 
                        subplot_titles=('Sex Count', 'Fare by Sex'))

fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 1, 2)

fig['layout'].update(height=400, width=800, title='Sex Distribuition', boxmode='group')
py.iplot(fig, filename='sex-subplot')


# In[ ]:



#First plot
trace0 = go.Bar(
    x = train_df[train_df["Survived"]== 1]["Embarked"].value_counts().index.values,
    y = train_df[train_df["Survived"]== 1]["Embarked"].value_counts().values,
    name='Survived'
)

#Second plot
trace1 = go.Bar(
    x = train_df[train_df["Survived"]== 0]["Embarked"].value_counts().index.values,
    y = train_df[train_df["Survived"]== 0]["Embarked"].value_counts().values,
    name="Not Survived"
)

data = [trace0, trace1]

layout = go.Layout(
    title='Embarked Distribuition'
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='grouped-bar')


# In[ ]:



trace0 = go.Box(
    x=df_survived["Embarked"],
    y=df_survived["Fare"],
    name='Survived'
)

trace1 = go.Box(
    x=df_notsurvived['Embarked'],
    y=df_notsurvived['Fare'],
    name='Not Survived'
)
    
data = [trace0, trace1]

layout = go.Layout(
    yaxis=dict(
        title='Fare Amount distribuition by Embarkment'
    ),
    boxmode='group'
)
fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='box-age-cat')


# In[ ]:


sns.distplot(train_df[train_df["Survived"]== 1]["Fare"].value_counts())
sns.distplot(train_df[train_df["Survived"]== 0]["Fare"].value_counts())


# In[ ]:


grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()


# In[ ]:


grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()


# In[ ]:


train_df = train_df.drop(['Cabin'], axis=1)
test_df = test_df.drop(['Cabin'], axis=1)


# In[ ]:


train_df['Sex'] = train_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


# In[ ]:





# In[ ]:


train_df.head()


# In[ ]:


train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1


# In[ ]:


X=train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


sns.countplot(train_df["FamilySize"])


# In[ ]:


labels = ['1','2','3','4','5','6','7','8','11']
size = train_df['FamilySize'].value_counts()
#colors = ['red', 'green']
explode = [0, 0.1,0.1,0.1,0.1,0.1,0.2,0.1,0]

plt.rcParams['figure.figsize'] = (7, 7)
plt.pie(size, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')
plt.title('A pie chart Representing the Survived population')
plt.axis('off')
plt.legend()
plt.show()


# In[ ]:


train_df['IsAlone'] = 0
train_df.loc[train_df['FamilySize'] == 1, 'IsAlone'] = 1


# In[ ]:


train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


# In[ ]:


sns.countplot(train_df["IsAlone"])


# In[ ]:


labels = ["Yes","No"]
size = train_df['IsAlone'].value_counts()
#colors = ['red', 'green']
explode = [0, 0.1]

plt.rcParams['figure.figsize'] = (7, 7)
plt.pie(size, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')
plt.title('A pie chart Representing the Alone population')
plt.axis('off')
plt.legend()
plt.show()


# In[ ]:


train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)


# In[ ]:


train_df.head()


# In[ ]:


train_df['Embarked'] = train_df['Embarked'].dropna().map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# In[ ]:


train_df.head()


# In[ ]:


train_df=train_df.drop("Age",axis=1)


# In[ ]:


train_df.head()


# This is our refine dataframe now we can build model with it.

# In[ ]:




