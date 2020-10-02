#!/usr/bin/env python
# coding: utf-8

# 1. Importing required packegaes

# In[ ]:


import pandas as pd
import numpy as np
import csv


# 2. Importing the dataset

# In[ ]:


file_path = 'Noshows.csv'
Noshows = pd.read_csv(file_path)
Noshows = pd.DataFrame(Noshows)
print(Noshows.head())


# 3. Changing the dates froma to make sure we can do analysis on them

# In[ ]:


Noshows['ScheduledDay'] = Noshows['ScheduledDay'].apply(np.datetime64)
Noshows['AppointmentDay'] = Noshows['AppointmentDay'].apply(np.datetime64)


# 4. Creating a waiting time calculater 

# In[ ]:


Noshows['WaitingTime'] = Noshows['ScheduledDay'] - Noshows['AppointmentDay']
Noshows['WaitingTime'] = Noshows['WaitingTime'].apply(lambda x: x.total_seconds() / (3600 * 24))
Noshows['WaitingTime']


# In[ ]:


Noshows.head()


# 5. Checking data for possible inconsistencies

# In[ ]:


print('Age',sorted(Noshows['Age'].unique()))
print('Gender',Noshows['Gender'].unique())
#This was a little bit long so I just commented it
#print('WaitingTime',sorted(Noshows['WaitingTime'].unique()))
print('Diabetes',Noshows['Diabetes'].unique())
print('Alcoholism',Noshows['Alcoholism'].unique())
print('Hipertension',Noshows['Hipertension'].unique())
print('Handcap',Noshows['Handcap'].unique())
print('Scholarship', Noshows['Scholarship'].unique())
print('SMS_received',Noshows['SMS_received'].unique())
print('No-show',Noshows['No-show'].unique())


# 6. Cleaning

# In[ ]:


#Deleting some outlier from age
Noshows = Noshows[(Noshows['Age'] > 0) & (Noshows['Age'] < 100)]
#Encoding gender
Noshows['IsFemale'] = np.where(Noshows['Gender'] == "F",1, 0) 
#Encoding no-show
Noshows['No-show'] = np.where(Noshows['No-show'] == "Yes",1, 0)


# 7. Creating some new features

# In[ ]:


#checking if an appointment is on Monday
import datetime
Noshows['IsMonday'] = np.where(Noshows['AppointmentDay'].dt.dayofweek == 0, 1, 0)
#Also realized in excel that neighborhood JARDIM CAMBURI has a huge rate of not showing up, thus:
Noshows['IsJARDIMCAMBURI'] = np.where(Noshows['Neighbourhood'] == 'JARDIM CAMBURI', 1, 0)


# 8. Finalizing the dataset

# In[ ]:


Noshows = Noshows[['Age', 'IsFemale', 'Diabetes','Alcoholism','Hipertension','Handcap',
                  'Scholarship', 'SMS_received', 'IsMonday', 'IsJARDIMCAMBURI','WaitingTime','No-show']]


# 9. Dividing the dataset into features (X) and lables (y)

# In[ ]:


features = Noshows[['Age', 'IsFemale', 'Diabetes','Alcoholism','Hipertension','Handcap',
                  'Scholarship', 'SMS_received', 'IsMonday', 'IsJARDIMCAMBURI', 'WaitingTime']]
labels  = Noshows['No-show']


# 10. creating train and test sets

# In[ ]:


from sklearn.model_selection import train_test_split 
tr_features, test_features, tr_labels, test_labels = train_test_split(features, labels,
random_state=0)


# 11. Cross validation (first model tried: decision trees)

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(random_state=0)
Parameters = {'max_depth':[8]}
CV = GridSearchCV(DT, Parameters, cv=5)


# 12. fitting the CV

# In[ ]:


CV.fit(tr_features, tr_labels.values.ravel())
#ignore the warnings here


# 13. Prediction

# In[ ]:


predict = CV.predict(test_features)


# In[ ]:


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score


# In[ ]:


accuracy_score(test_labels, predict)


# In[ ]:


precision_score(test_labels, predict)


# In[ ]:


predict.sum()


# 14. Exporting the results

# In[ ]:


predictions = pd.DataFrame(predict, columns=['DT'])
labels = pd.DataFrame(test_labels, columns=['Labels'])
predictions['Labels'] = labels['Labels']
predictions.to_csv('predictions.csv')


# 15. Trying the process for some other models:

# a. Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(random_state=0)
Parameters = {'C':[0.001, 0.01, 0.1, 1, 10, 100]}
CV = GridSearchCV(LR, Parameters, cv=5)
CV.fit(tr_features, tr_labels.values.ravel())


# In[ ]:


predict = CV.predict(test_features)
accuracy_score(test_labels, predict)


# In[ ]:


precision_score(test_labels, predict)


# b. Multi-layer Preceptron

# In[ ]:


from sklearn.neural_network import MLPClassifier
NN = MLPClassifier(random_state=0)
Parameters = {'hidden_layer_sizes':[(5,)]}
CV = GridSearchCV(NN, Parameters, cv=5)
CV.fit(tr_features, tr_labels.values.ravel())


# In[ ]:


predict = CV.predict(test_features)
accuracy_score(test_labels, predict)


# In[ ]:


precision_score(test_labels, predict)


# c. Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
RF = RandomForestClassifier()
Parameters = {'n_estimators':[1000],
             'max_depth':[100]}
CV = GridSearchCV(RF, Parameters, cv=5)
CV.fit(tr_features, tr_labels.values.ravel())
CV.best_params_


# In[ ]:


predict = CV.predict(test_features)
accuracy_score(test_labels, predict)


# In[ ]:


precision_score(test_labels, predict)


# In[ ]:


recall_score(test_labels, predict)


# In[ ]:


f1_score(test_labels, predict)


# In[ ]:


predict.sum()
CV.best_score_


# The best results I could get was made by using a random forest of 25 estimators and maximu depth of 8. Thees are the scores: <br>
# Accuracy:0.7520565360454682 <br>
# Precision:0.4522641981486115 <br>
# Recall:0.258348623853211 <br>
# F1: 0.2953091684434968 <br>
# Total number of no shows predicted:3997 <br>
# 

# In[ ]:




