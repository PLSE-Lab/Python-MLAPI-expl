#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle


# # **Import Ipl Dataset**

# In[ ]:


ipl_data=pd.read_csv('../input/ipl-dataset/ipl.csv')
ipl_data.shape


# In[ ]:


ipl_data.head(5)


# # Data Cleaning Process

# In[ ]:


#removing unwanted columns
column_delete=['mid','venue','batsman','bowler','striker','non-striker']
ipl_data.drop(labels=column_delete,axis=1,inplace=True)


# In[ ]:


ipl_data.head(5)


# In[ ]:


[ipl_data['bat_team'].unique()]


# In[ ]:


current_team=['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
        'Mumbai Indians', 'Kings XI Punjab',
        'Royal Challengers Bangalore', 'Delhi Daredevils','Sunrisers Hyderabad']


# In[ ]:


ipl_data=ipl_data[(ipl_data['bat_team'].isin(current_team))&(ipl_data['bowl_team'].isin(current_team))]


# In[ ]:


ipl_data.shape


# In[ ]:


#Removing First 6 over i.e Powerplay to that we have atleast 6 over data to predict
ipl_data=ipl_data[ipl_data['overs']>=6.0]


# In[ ]:


ipl_data.head(5)


# In[ ]:


#Convert Data string to datetime object
from datetime import datetime
ipl_data['date']=ipl_data['date'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))


# In[ ]:


#Data Preprocessing : One Hot Encoding
encoded_ipl=pd.get_dummies(data=ipl_data,columns=['bat_team','bowl_team'])


# In[ ]:


encoded_ipl.head(5)


# In[ ]:


#Rearranging Dataset
encoded_ipl.columns


# In[ ]:


encoded_ipl=encoded_ipl[['date', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5',
        'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils',
       'bat_team_Kings XI Punjab', 'bat_team_Kolkata Knight Riders',
       'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
       'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
       'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils',
       'bowl_team_Kings XI Punjab', 'bowl_team_Kolkata Knight Riders',
       'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
       'bowl_team_Royal Challengers Bangalore',
       'bowl_team_Sunrisers Hyderabad','total']]


# # Train Test Split

# In[ ]:


x_train=encoded_ipl.drop(labels='total',axis=1)[encoded_ipl['date'].dt.year<=2016]
x_test=encoded_ipl.drop(labels='total',axis=1)[encoded_ipl['date'].dt.year>2016]


# In[ ]:


y_train=encoded_ipl[encoded_ipl['date'].dt.year<=2016]['total'].values
y_test=encoded_ipl[encoded_ipl['date'].dt.year>2016]['total'].values


# In[ ]:


#remove date column
x_train.drop(labels='date',axis=True,inplace=True)
x_test.drop(labels='date',axis=True,inplace=True)


# # Model Building
#  
Linear Regression
# In[ ]:


from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(x_train,y_train)


# In[ ]:


y_pred=lr.predict(x_test)


# In[ ]:


import seaborn as sns
sns.distplot(y_test-y_pred)


# In[ ]:


# Creating a pickle file for the classifier
filename = 'first-innings-score-lr-model.pkl'
pickle.dump(regressor, open(filename, 'wb'))


# In[ ]:


from sklearn import metrics
print('Mean Absolute Error :',(metrics.mean_absolute_error(y_test,y_pred)))


# In[ ]:





# In[ ]:




