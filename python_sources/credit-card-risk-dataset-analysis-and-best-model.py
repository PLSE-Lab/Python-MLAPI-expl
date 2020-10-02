#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
data = pd.read_csv('../input/german-credit-data-with-risk/german_credit_data.csv', index_col = 0)


# Data Analysis

# In[ ]:


data.head()


# In[ ]:


data.isna().sum()


# In[ ]:


data.info()


# In[ ]:


data.nunique()


# EDA 

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
print(data['Risk'].value_counts())
sns.catplot(x="Risk", kind="count", palette="ch:.25", data=data);


# In[ ]:


import plotly.graph_objs as go 
import plotly.tools as tls 
import plotly.offline as py 
py.init_notebook_mode(connected=True) 


data_good = data.loc[data["Risk"] == 'good']['Age'].values.tolist()
data_bad = data.loc[data["Risk"] == 'bad']['Age'].values.tolist()
data_age = data['Age'].values.tolist()

#First plot
first = go.Histogram(
    x=data_good,
    histnorm='probability',
    name="Good"
)
#Second plot
second = go.Histogram(
    x=data_bad,
    histnorm='probability',
    name="Bad"
)
#Third plot
third = go.Histogram(
    x=data_age,
    histnorm='probability',
    name="Overall Age"
)

fig = tls.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],
                          subplot_titles=('Good','Bad', 'General Distribuition'))

#setting the figs
fig.append_trace(first, 1, 1)
fig.append_trace(second, 1, 2)
fig.append_trace(third, 2, 1)

fig['layout'].update(showlegend=True, title='Age', bargap=0.05)
py.iplot(fig, filename='plt Age vs Risk')


# In[ ]:


data_good = data.loc[data["Risk"] == 'good']['Job'].values.tolist()
data_bad = data.loc[data["Risk"] == 'bad']['Job'].values.tolist()
data_age = data['Job'].values.tolist()

#first plot
first = go.Histogram(
    x=data_good,
    histnorm='probability',
    name="Good"
)
#Second plot
second = go.Histogram(
    x=data_bad,
    histnorm='probability',
    name="Bad"
)
#Third plot
third = go.Histogram(
    x=data_age,
    histnorm='probability',
    name="Overall Job"
)

fig = tls.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],
                          subplot_titles=('Good','Bad', 'General Distribuition'))

#setting the figs
fig.append_trace(first, 1, 1)
fig.append_trace(second, 1, 2)
fig.append_trace(third, 2, 1)

fig['layout'].update(showlegend=True, title='Job', bargap=0.05)
py.iplot(fig, filename='plt Job vs Risk')


# In[ ]:



first = go.Bar(
    x = data[data["Risk"]== 'good']["Housing"].value_counts().index.values,
    y = data[data["Risk"]== 'good']["Housing"].value_counts().values,
    name='Good credit'
)


second = go.Bar(
    x = data[data["Risk"]== 'bad']["Housing"].value_counts().index.values,
    y = data[data["Risk"]== 'bad']["Housing"].value_counts().values,
    name="Bad Credit"
)

plot = [first, second]

layout = go.Layout(
    title='Housing vs Risk Analysis'
)


fig = go.Figure(data=plot, layout=layout)

py.iplot(fig, filename='plt housing vs Risk')


# In[ ]:



first = go.Bar(
    x = data[data["Risk"]== 'good']["Sex"].value_counts().index.values,
    y = data[data["Risk"]== 'good']["Sex"].value_counts().values,
    name='Good credit'
)


second = go.Bar(
    x = data[data["Risk"]== 'bad']["Sex"].value_counts().index.values,
    y = data[data["Risk"]== 'bad']["Sex"].value_counts().values,
    name="Bad Credit"
)

plot = [first, second]

layout = go.Layout(
    title='Sex vs Risk Analysis'
)


fig = go.Figure(data=plot, layout=layout)

py.iplot(fig, filename='plt Sex vs Risk')


# In[ ]:



first = go.Bar(
    x = data[data["Risk"]== 'good']["Saving accounts"].value_counts().index.values,
    y = data[data["Risk"]== 'good']["Saving accounts"].value_counts().values,
    name='Good credit'
)


second = go.Bar(
    x = data[data["Risk"]== 'bad']["Saving accounts"].value_counts().index.values,
    y = data[data["Risk"]== 'bad']["Saving accounts"].value_counts().values,
    name="Bad Credit"
)

plot = [first, second]

layout = go.Layout(
    title='Saving accounts vs Risk Analysis'
)


fig = go.Figure(data=plot, layout=layout)

py.iplot(fig, filename='plt Saving accounts vs Risk')


# In[ ]:


first = go.Bar(
    x = data[data["Risk"]== 'good']["Purpose"].value_counts().index.values,
    y = data[data["Risk"]== 'good']["Purpose"].value_counts().values,
    name='Good credit'
)


second = go.Bar(
    x = data[data["Risk"]== 'bad']["Purpose"].value_counts().index.values,
    y = data[data["Risk"]== 'bad']["Purpose"].value_counts().values,
    name="Bad Credit"
)

plot = [first, second]

layout = go.Layout(
    title='Purpose vs Risk Analysis'
)


fig = go.Figure(data=plot, layout=layout)

py.iplot(fig, filename='plt Purpose vs Risk')


# In[ ]:



group = (10, 22, 30, 50, 120)

cats = ['student', 'young', 'adult', 'old']
data["Age_Group"] = pd.cut(data.Age, group, labels=cats)
data['Age_Group'].head(10)


# In[ ]:


first = go.Bar(
    x = data[data["Risk"]== 'good']["Age_Group"].value_counts().index.values,
    y = data[data["Risk"]== 'good']["Age_Group"].value_counts().values,
    name='Good credit'
)


second = go.Bar(
    x = data[data["Risk"]== 'bad']["Age_Group"].value_counts().index.values,
    y = data[data["Risk"]== 'bad']["Age_Group"].value_counts().values,
    name="Bad Credit"
)

plot = [first, second]

layout = go.Layout(
    title='Age_Group vs Risk Analysis'
)


fig = go.Figure(data=plot, layout=layout)

py.iplot(fig, filename='plt Age_Group vs Risk')


# Feature Engineering
# 

# In[ ]:


data.columns


# In[ ]:


data = data.drop(['Age'], axis = 1)


# In[ ]:


data['Saving accounts'] = data['Saving accounts'].fillna('no_account')
data['Checking account'] = data['Checking account'].fillna('no_account')
data.head()


# In[ ]:


data.info()


# In[ ]:


data = pd.get_dummies(data=data, columns=['Sex', 'Saving accounts','Checking account','Purpose', 'Age_Group', 'Housing' ])


# In[ ]:


y = data['Risk']


# In[ ]:


X = data.drop(['Risk'], axis = 1)


# In[ ]:


X.head()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)
data = scaler.transform(X)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y)
y = pd.Series(le.transform(y))
y.head()


# In[ ]:


plt.figure(figsize=(20,16))
sns.heatmap(X.astype(float).corr(), annot=True)
plt.show()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, fbeta_score

from sklearn.model_selection import train_test_split, KFold, cross_val_score


# In[ ]:



x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42)


# In[ ]:


models = [RandomForestClassifier(), LogisticRegression(), DecisionTreeClassifier(), KNeighborsClassifier(), LinearDiscriminantAnalysis(),GaussianNB(), SVC(),XGBClassifier()]
for model in models:
    model.fit(x_train,y_train)
    print(model,'Accuracy = ',accuracy_score(y_test, model.predict(x_test)))
    print('classification_report = ',classification_report(y_test, model.predict(x_test)))
    print('\n')


# From above models we can see, We are getting highest accuracy and better values of precision and recall for XGBClassifier

# In[ ]:


xgb_model = XGBClassifier()
xgb_model.fit(x_train, y_train)
print('Accuracy = ',accuracy_score(y_test, xgb_model.predict(x_test)))
print('classification_report = ',classification_report(y_test, xgb_model.predict(x_test)))


# In[ ]:


param_grid = {"max_depth": [3,5, 7, 10],
              "n_estimators":[10,50,250, 500,1000],
              "max_features": [4,7,15,20],
              "learning_rate": [0.1,0.05,0.001]}
model = XGBClassifier()
grid_search = GridSearchCV(model, param_grid=param_grid, cv=3, scoring='recall', verbose=1)
grid_search.fit(x_train, y_train)


# In[ ]:


print(grid_search.best_params_)


# In[ ]:


print('Accuracy_Score',accuracy_score(y_test,grid_search.predict(x_test)))
print("\n")
print('Classification_Report',classification_report(y_test, grid_search.predict(x_test)))


# In[ ]:




