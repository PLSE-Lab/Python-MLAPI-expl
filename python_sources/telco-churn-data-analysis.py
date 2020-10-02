#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


customers = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')


# In[ ]:


customers.head(10)


# In[ ]:


customers.any().isnull()
#no null values
customers.info()
#total charges is an object


# In[ ]:


#customers['TotalCharges'] = pd.to_numeric(customers['TotalCharges'], errors='raise')   
#converting total charges to float, found non integer elements ' '
nonintegers = customers[customers['TotalCharges'] == ' '] 
to_drop = nonintegers.index.tolist()
to_drop


# In[ ]:


customers = customers.drop(to_drop, axis='index')
customers['TotalCharges'] = pd.to_numeric(customers['TotalCharges'], errors='raise') 


# In[ ]:


customers.any().isnull()
#no NaN's in TotalCharges
#i want to convert all the yes/no's to 1's and 0's in churn
customers['Churn'] = customers['Churn'].map(dict({'Yes':1,'No':0}))


# In[ ]:


import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


# Let's check the demographics of the people who discontinued their service.

# In[ ]:


demographics = ['gender', 'Partner', 'SeniorCitizen', 'Dependents']


for i in demographics:
    trace1 = go.Bar(
        x=customers.groupby(i)['Churn'].sum().reset_index()[i],
        y=customers.groupby(i)['Churn'].sum().reset_index()['Churn'],
        name= i
    )

    data = [trace1]
    layout = go.Layout(
        title= i,
        yaxis=dict(
        title='Churn'
        ),
        barmode='group',
        autosize=True,
    width=600,
    height=600,
    margin=go.Margin(
        l=70,
        r=70,
        b=100,
        t=100,
        pad=8
    )
    )
    fig = go.Figure(data=data, layout=layout)
    py.offline.iplot(fig)


# 1869 people cancelled their service. 
# 
# * Regarding the gender of the presumed account holder, it looks like this dataset has an even amount of churners between men and women. 
# * Almost 2x the rate of churn amongst people without a partner.
# * A large marjority were not senior citizens (74%), and had no dependents (82%).
# 
# Also, checking the amount of senior citizens (1142) shows that nearly 42% of them discontinue their service. Compare that to 20% of everyone else, and we can see that seniors are almost 2x as likely to stop using the service. 
# 
# 

# In[ ]:


seniors = customers.loc[customers['SeniorCitizen'] == 1]

nonseniors = customers.loc[customers['SeniorCitizen'] == 0]


# In[ ]:


hist_data = [seniors.groupby('tenure')['Churn'].sum(),nonseniors.groupby('tenure')['Churn'].sum()]
group_labels = ['Seniors', 'Non-Seniors']

import plotly.figure_factory as ff
fig = ff.create_distplot(hist_data, group_labels,bin_size=[1,1], curve_type='normal', show_rug=False)
py.offline.iplot(fig)


# We can see how different the distributions are. Seniors are mostly leaving within 4 months. There is an opportunity there to figure out why such short tenures, but I'm also aware that there may be other age groups that have higher early churn.

# In[ ]:


customers['InternetService'].value_counts()


# In[ ]:


customers.groupby('InternetService')['Churn'].sum()
#Nearly half of the customers who took fiber optic left


# In[ ]:


seniors.groupby('InternetService')['Churn'].sum()


# In[ ]:


mask = ['DSL','Fiber optic']
internet = customers[customers['InternetService'].isin(mask)]


# In[ ]:


categories = ['InternetService', 'OnlineSecurity', 'OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract',
             'PaperlessBilling','PaymentMethod']
internet.info()


# In[ ]:


dumm = pd.get_dummies(internet,columns=categories,drop_first=True)


# In[ ]:


corr = dumm.corr(method='spearman')
corr


# In[ ]:


mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr,cmap=cmap, mask=mask,vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[ ]:


y = dumm.iloc[:,10].values
dumm = dumm.drop(['customerID','Churn'],axis=1)


# In[ ]:


X = dumm


# In[ ]:


y


# In[ ]:


mask = ['gender','Partner','Dependents','PhoneService','MultipleLines']
X = pd.get_dummies(dumm,columns=mask,drop_first=True)


# In[ ]:


X.head()


# In[ ]:


#Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 4)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred1 = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred1)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred1,y_test)
accuracy

print (cm)
print (accuracy)


# In[ ]:


import lightgbm as lgb

d_train = lgb.Dataset(X_train, label=y_train)

params = {}
params['learning_rate'] = 0.2
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['num_leaves'] = 5
params['max_depth'] = 15


clf = lgb.train(params, d_train, 100)


# In[ ]:


#Prediction
y_pred2=clf.predict(X_test)

#convert into binary values
for i in range(0,552):
    if y_pred2[i]>=.52:       
       y_pred2[i]=1
    else:  
       y_pred2[i]=0


# In[ ]:


#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred2)
#Accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred2,y_test)


# In[ ]:


cm


# In[ ]:


accuracy


# In[ ]:


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)



# Predicting the Test set results
y_pred3 = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred3)


# In[ ]:


#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred3)
#Accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred3,y_test)


# In[ ]:


cm


# In[ ]:


accuracy


# The LightGBM model was the best performing in this case at 77%. Further improvements to the model could be made. 
