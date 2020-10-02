#!/usr/bin/env python
# coding: utf-8

# # KERNEL OBJECTIVES
# 
# * Perform Basic EDA
# 
# * Make a Decision Tree Classifier to predict the final amount of a tender (in a value cader)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from plotly.offline import init_notebook_mode, iplot
from plotly.graph_objs import *
import plotly.graph_objs as go
init_notebook_mode()
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
sns.set_style('darkgrid')
# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/trade-permits-current.csv')
df.columns


# In[ ]:


print("Action Information : ",df['Action Type'].unique(), "\n")
print("Worker Information : ",df['Work Type'].unique(), "\n ")
print("Contractor Information : ",df['Contractor'].unique(), len(df['Contractor'].unique()), "\n")
print("Categorical Information :", df['Category'].unique())


# In[ ]:


df.dropna(inplace = True)


# In[ ]:


df.head()


# # Comparison EDA
# 
# Let's have a look at the top grossing contractors at current time and have a look at the best project efficient to earning ratio maintainers in the current tender market. We will be focussing mainly on top 10 players of both categories and see the differences between them.

# In[ ]:


mySummingGroup = df.drop(columns=['Longitude', 'Latitude', 'Application/Permit Number']).groupby(by = 'Contractor').agg({'Value':sum})


# In[ ]:


x = mySummingGroup['Value'].nlargest(10)
x


# # Top Grossing Contractors
# <div id = "topc"></div>

# In[ ]:



data1 = [Bar(
            y=x,
            x=x.keys(),
            marker = dict(
            color = 'rgba(25, 82, 1, .9)'
            ),
            name = "Contractor's amount earned per project"
    )]

layout1 = go.Layout(
    title="Top Grossing Contractors",
    xaxis=dict(
        title='Contractor',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Total Amount Earned',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)
myFigure2 = go.Figure(data = data1 , layout = layout1)
iplot(myFigure2)


# In[ ]:


myMeanGroup = df.drop(columns=['Longitude', 'Latitude', 'Application/Permit Number']).groupby(by = 'Contractor').mean()


# In[ ]:


efficientContractors = myMeanGroup['Value'].nlargest(10)


# In[ ]:


data = [Bar(
            y=efficientContractors,
            x=efficientContractors.keys(),
            marker = dict(
            color = 'rgba(255, 182, 1, .9)'
            ),
            name = "Contractor's amount earned per project"
    )]

layout = go.Layout(
    title="Contractor's amount earned per project",
    xaxis=dict(
        title='Contractor',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Amount per project',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)
myFigure = go.Figure(data = data , layout = layout)
iplot(myFigure)


# # Categorical Pie Charts
# 
# Let's have a look at the % distribution of categories of the tenders. Currently, we have exactly 5 categories. Let's have a look at there distribution.

# In[ ]:


catCount = df.groupby('Category')['Permit Type'].count()


# In[ ]:


fig = { 
    "data":[{
        "values":catCount,
        "labels":catCount.keys(),
        "domain": {"x": [0, 1]},
        "name": "Categories",
        "hoverinfo":"label+percent+name",
        "hole": .4,
        "type": "pie",
        "textinfo": "value"
    }],
    "layout":{
        "title":"Categorical Distribution of Tenders",
        "annotations": [
            {
                "font": {
                    "size": 15
                },
                "showarrow": False,
                "text": "DISTRIBUTION",
                "x": 0.5,
                "y": 0.5
            }]
    }
}

trace = go.Pie(labels = catCount.keys(), values=catCount,textinfo='value', hoverinfo='label+percent', textfont=dict(size = 15))
iplot(fig)


# ### Clearly, we can infer that tenders having "SINGLE FAMILY / DUPLEX" are quite prominent here

# # Let's try estimating the value cadre of a tender.
# 
# Note : Still being developed.
# 
# For that, we will first divide the value system into 5 value cadres 
# 

# In[ ]:



# My Value Encoder
def valueEncoder(value):
    if value > 10000000:
        return 4
    elif value > 100000:
        return 3
    elif value > 10000:
        return 2
    elif value > 100:
        return 1
    else:
        return 0


# In[ ]:


df['ValueLabel'] = df['Value'].apply(valueEncoder)


# ## Now lets encode our categories (In one hot enoding pattern)
# 
# We can afford 5 binary features for our models, I believe

# # THE LONG PROCESS

# In[ ]:


from sklearn.preprocessing import OneHotEncoder, LabelEncoder

genLabel_cat = LabelEncoder()
cat_labels = genLabel_cat.fit_transform(df['Category'])
df['CategoryLabel'] = cat_labels


# In[ ]:


df[['Category','CategoryLabel']].iloc[::2]


# In[ ]:


cat_ohe = OneHotEncoder()
cat_feature_arr = cat_ohe.fit_transform(df[['CategoryLabel']]).toarray()
cat_feature_labels = list(genLabel_cat.classes_)
cat_features = pd.DataFrame(cat_feature_arr, columns=cat_feature_labels)


# # A simple one hot matrix for our labels

# In[ ]:


cat_features.head(10)


# # PS : A shortcut for the same. Just in case you are in a hurry

# In[ ]:


final_one_hot = pd.get_dummies(df['Category'])


# In[ ]:


df2 = pd.concat([df, final_one_hot], axis = 1)
df2 = df2.drop(['Application/Permit Number', 'Address', 'Description', 'Applicant Name','Application Date','Issue Date','Final Date','Expiration Date','Contractor', 'Permit and Complaint Status URL', 'Location'], axis = 1)


# In[ ]:


# also add 'Value', 'Category' when running for first time
df2 = df2.drop(['CategoryLabel'],axis = 1)


# In[ ]:


df2.head()


# # Next steps : Checking and cleaning Permit Type, Action Type, Work Type and Status Parameters

# In[ ]:


df2 = pd.concat([df2, pd.get_dummies(df['Work Type'])], axis = 1)


# In[ ]:


df2 = df2.drop(['Work Type'], axis = 1)


# In[ ]:


df2.head()


# In[ ]:


print(df2['Action Type'].unique() , "\n Total types are ", len(df2['Action Type'].unique()))


# # So, there are a total of 21 types of Actions which have been performed.
# 
# Making dummies for 21 categories may lead to overfitting of data. Let's try Feature Hashing for this scheme

# In[ ]:


from sklearn.feature_extraction import FeatureHasher

fh = FeatureHasher(n_features = 5, input_type = 'string')
hashed_features = fh.fit_transform(df2['Action Type'])
hashed_features = hashed_features.toarray()
df2 = pd.concat([df2, pd.DataFrame(hashed_features)], 
          axis=1).dropna()


# # Our dataframe after feature hashing 

# In[ ]:


df2.iloc[10:20]


# # Now let's have a look at status parameters

# In[ ]:


df2['Status'].unique()


# In[ ]:


# Again, a binary parameter, let's use binary encodings.

df2 = pd.concat([df2, pd.get_dummies(df2['Status'])], axis = 1)
df2 = df2.drop(['Status'], axis = 1)


# In[ ]:


df2.drop(['Value', 'Category'], axis = 1, inplace = True)


# In[ ]:


df2.drop(['Action Type'], axis = 1, inplace = True)


# In[ ]:


df2.drop(['Permit Type'], axis = 1, inplace = True)


# # Our Final Data Base : 

# In[ ]:


df2.head()


# ### Performing the famous train test split

# In[ ]:





# In[ ]:


from sklearn.model_selection import train_test_split
y = df2['ValueLabel']
X = df2.drop(['ValueLabel'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[ ]:


from sklearn.naive_bayes import GaussianNB
myClassifier = GaussianNB()
myClassifier.fit(X_train, y_train)


# In[ ]:


predictions = myClassifier.predict(X_test)


# # Results : Naive Bayes Classifier 

# In[ ]:


from sklearn.metrics import accuracy_score,confusion_matrix

cnf = confusion_matrix(y_test, predictions)
score = accuracy_score(y_test, predictions)

print ("Confusion Matrix for our Naive Bayes classifier is :\n ", cnf)

print("While the accuracy score for the same is %.2f percent" % (score * 100))


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
myClassifier2 = DecisionTreeClassifier(max_depth = 5, min_samples_leaf = 2)


# # Results : Decision Tree Classifier

# In[ ]:


myClassifier2.fit(X_train, y_train)
predictions2 = myClassifier2.predict(X_test)

cnf2 = confusion_matrix(y_test, predictions2)
score2 = accuracy_score(y_test, predictions2)

print ("Confusion Matrix for our Decision Tree classifier is :\n ", cnf2)

print("While the accuracy score for the same is %.2f percent" % (score2 * 100))


# # Conclusions:
# 
# We have finally studied the data and made a classifier to guess the range of the value a particular recent tender may amount to. Fun part is to  notice that Decision Tree Classifier works like a charm (**99.24%**) accuracy while Gaussian Naive Bayes is... well, not really upto the mark for the task.

# In[ ]:




