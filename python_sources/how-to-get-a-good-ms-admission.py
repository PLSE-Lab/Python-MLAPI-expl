#!/usr/bin/env python
# coding: utf-8

# # How good is my profile to get a good college in MS?
# 
# ### This is probably a question that every aspiring MS aspirant wants to know. Is my profile good enough to get a good college? Being an aspirant myself even I also have so many doubts, whether my CGPA is good enough, how should I write a solid SOP, etc.
# 
# ### So this kernel will mainly have the following two sections
# 
# * EDA
#      *  Understanding the important factors
#      *  Visualisation        
# * Predictve Modelling

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
color = sns.color_palette()
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
import plotly.tools as tls

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df=pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
df.head()


# In[ ]:


print(df.info())
print(df.isnull().sum())
df.describe()


# Lets see the data distribution of the no. of universities given based on their ratings

# In[ ]:


cnt_srs = df['University Rating'].value_counts().head()
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=cnt_srs.values[::-1],
    orientation = 'h',
    marker=dict(
        color=['#04A46D','#029AF9','#6D0601','#6D016D','#878D02']
    ),
)

layout = dict(
    title='Rating distribution of Universities',
    xaxis=dict(
        title='No. of Universities',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#2F3102'
        )
    ),
    yaxis=dict(
        title='Ratings of the Universities',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#2F3102'
        )
    )
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="univs")


# So the data that we have for admission probability is maximum for 3 rated universities.

# In[ ]:


plt.figure(figsize=(10,8))
sns.heatmap(df.drop('Serial No.',axis=1).corr(),annot=True)


# In[ ]:


trace0 = go.Scatter(
    x = df.ix[df['Chance of Admit ']>=0.7]['GRE Score'],
    y = df.ix[df['Chance of Admit ']>=0.7]['CGPA'],
    name = 'Chance of Admit > 0.7',
    mode = 'markers',
    marker = dict(
        size = 10,
        color = 'rgba(152, 0, 0, .8)',
        line = dict(
            width = 2,
            color = 'rgb(0, 0, 0)'
        )
    )
)

trace1 = go.Scatter(
    x = df.ix[df['Chance of Admit ']<0.7]['GRE Score'],
    y = df.ix[df['Chance of Admit ']<0.7]['CGPA'],
    name = 'Chance of Admit < 0.7',
    mode = 'markers',
    marker = dict(
        size = 10,
        color = 'rgba(255, 182, 193, .9)',
        line = dict(
            width = 2,
        )
    )
)

data = [trace0, trace1]

layout = dict(title = 'GRE and CGPA Score Distribution',
              yaxis = dict(zeroline = False),
              xaxis = dict(zeroline = False)
             )

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='styled-scatter')


# #### Here we see the distribution of the scores of people who have > 0.7 and < 0.7 chances of getting an admission for an MS.
# 
# Please note, no bar for University Rating has been put here, the probability is for any university admission for MS

# ### Next we'll find out the distribution of the scores of the people who had admission chances > 0.7 and that too in 4 & 5 rated universities

# In[ ]:


df1=df.ix[df['Chance of Admit ']>=0.7]
df1=df1[df1['University Rating']>3]
fig = {
    "data": [
        {
            "type": 'violin',
            "y": df1['GRE Score'],
            "legendgroup": 'M',
            "scalegroup": 'M',
            "name": 'GRE Score Distribution of People having >0.7 Chances of Admission in 4 rated University',
            "box": {
                "visible": True
            },
             "fillcolor": '#E9B2AE',
            "meanline": {
                "visible": True
            },
            "line": {
                "color": 'black'
            }
        }],
        "layout" : {
        "yaxis": {
            "zeroline": False,
        },
        "violinmode": "group"
    }
}
py.iplot(fig, filename = 'gre')

fig = {
    "data": [
        {
            "type": 'violin',
            "y": df1['TOEFL Score'],
            "legendgroup": 'M',
            "scalegroup": 'M',
            "name": 'TOEFL Score Distribution of People having >0.7 Chances of Admission in 4 rated University',
            "box": {
                "visible": True
            },
             "fillcolor": '#AEE9B4',
            "meanline": {
                "visible": True
            },
            "line": {
                "color": 'black'
            }
        }],
        "layout" : {
        "yaxis": {
            "zeroline": False,
        },
        "violinmode": "group"
    }
}
py.iplot(fig, filename = 'toefl')

fig = {
    "data": [
        {
            "type": 'violin',
            "y": df1['CGPA'],
            "legendgroup": 'M',
            "scalegroup": 'M',
            "name": 'CGPA Score Distribution of People having >0.7 Chances of Admission in 4 rated University',
            "box": {
                "visible": True
            },
             "fillcolor": '#BA8C24',
            "meanline": {
                "visible": True
            },
            "line": {
                "color": 'black'
            }
        }],
        "layout" : {
        "yaxis": {
            "zeroline": False,
        },
        "violinmode": "group"
    }
}
py.iplot(fig, filename = 'cgpa')

fig = {
    "data": [
        {
            "type": 'violin',
            "y": df1['SOP'],
            "legendgroup": 'M',
            "scalegroup": 'M',
            "name": 'SOP Score Distribution of People having >0.7 Chances of Admission in 4 rated University',
            "box": {
                "visible": True
            },
             "fillcolor": '#E0E5E0',
            "meanline": {
                "visible": True
            },
            "line": {
                "color": 'black'
            }
        }],
        "layout" : {
        "yaxis": {
            "zeroline": False,
        },
        "violinmode": "group"
    }
}
py.iplot(fig, filename = 'sop')

fig = {
    "data": [
        {
            "type": 'violin',
            "y": df1['LOR '],
            "legendgroup": 'M',
            "scalegroup": 'M',
            "name": 'LOR Score Distribution of People having >0.7 Chances of Admission in 4 rated University',
            "box": {
                "visible": True
            },
             "fillcolor": '#AEE3E9',
            "meanline": {
                "visible": True
            },
            "line": {
                "color": 'black'
            }
        }],
        "layout" : {
        "yaxis": {
            "zeroline": False,
        },
        "violinmode": "group"
    }
}
py.iplot(fig, filename = 'lor')


# Well let's have a look at the no. of students who didn't have research experience at all but still got into the top rated universities.

# In[ ]:


print(df1[df1['Research']==0].shape[0],"students have good chances to go into top rated universities without research experience.")
df1[df1['Research']==0]


# ### Low CGPA or getting a GRE score below 320 has always been a worry for students, thinking that if they will get any good colleges or not. Here we'll try to find out if possible, how many students have been able to make it to 3, 4 or 5 rated university with an avg. CGPA or an avg. GRE Score
# 
# #### I'm keeping good CGPA threshold of 8 and GRE Score 320 and admission probability threshold of 0.7 as good.
# 
# We'll see if any of the students has score below these.

# In[ ]:


df_top=df.ix[df['University Rating']>2]
df_top_admits=df_top.ix[df_top['Chance of Admit ']>=0.7]
df_top_admits[df_top_admits['CGPA']<8]


# Hmm, two students has a really good probability to get in, one maybe because SOP is good and the other one because of Research Experience.

# In[ ]:


print(df_top_admits[df_top_admits['GRE Score']<320].shape[0],"students have good chances to get into 3, 4 or 5 rated university with a GRE Score < 320")
df_top_admits[df_top_admits['GRE Score']<320]


# Now, we'll look at the distribution of the CGPA score and the GRE Score for the students having really good probabilty to get into 4 or 5 rated university.

# In[ ]:


df_best=df_top_admits[df_top_admits['University Rating']>3]

trace1 = go.Scatter(
    x = df_best['GRE Score'],
    y = df_best['CGPA'],
    name = 'Chance of Admit >= 0.7 in 4 or 5 rated university',
    mode = 'markers',
    marker = dict(
        size = 10,
        color = '#B4DEB4',
        line = dict(
            width = 2,
        )
    )
)

data = [trace1]

layout = dict(title = 'GRE and CGPA Score Distribution for students having good chances to get into 4 or 5 rated universities',
              yaxis = dict(zeroline = False),
              xaxis = dict(zeroline = False)
             )

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='gre-scatter')


# ### Next part is the predictive modelling part and finding out the most important features for getting a good MS admission.

# ### Predictive Modelling
# 
# I'll try with Linear Regression only to make the predictive model for chances of admission. And then try to visualize the important features.

# In[ ]:


df.head(1)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train,X_test,y_train,y_test=train_test_split(df.drop(['Serial No.','Chance of Admit '],axis=1),df[['Chance of Admit ']],
                                              test_size=0.25, random_state=42) #train_set=75%,test_set=25%

lr=LinearRegression()

lr.fit(X_train,y_train)

preds=lr.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_squared_error

print("The mean squared error for predicting the probability of chances of an admission comes out to be",      str(round(mean_squared_error(preds,y_test),5)))


# In[ ]:


coefficients = pd.concat([pd.DataFrame(X_train.columns).rename(columns={0:'Feature Name'})                          ,pd.DataFrame(np.transpose(lr.coef_)).rename(columns={0:'Importance Value'})], axis = 1)
print('The important features for getting an MS admission in descending order are:')
coefficients=coefficients.sort_values(by=['Importance Value'],ascending=False).reset_index(drop=True)
coefficients


# In[ ]:


data = [go.Bar(
            x=coefficients['Feature Name'],
            y=coefficients['Importance Value']
    )]

py.iplot(data, filename='bar')


# Does these seem to biased? Well I will leave the thoughts upon you!

# In[ ]:




