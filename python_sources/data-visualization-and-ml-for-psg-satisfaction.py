#!/usr/bin/env python
# coding: utf-8

# 1. The work on the following project was done by Piyush Singla and Nikhil Sharma 4.
# 2. link to Piyush Singla kaggle account -> https://www.kaggle.com/mpiyu20
# 3. link to Nikhil Sharma kaggle account -> https://www.kaggle.com/nikhilsharma4

# Importing the libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
init_notebook_mode(connected=True)
cf.go_offline()


# Importing the datasets

# In[ ]:


directory = "/kaggle/input/airline-passenger-satisfaction/"
feature_tables = ['train.csv', 'test.csv']

df_train = directory + feature_tables[0]
df_test = directory + feature_tables[1]

# Create dataframes
print(f'Reading csv from {df_train}...')
df = pd.read_csv(df_train)
print('...Complete')

print(f'Reading csv from {df_train}...')
df2 = pd.read_csv(df_test)
print('...Complete')


# In[ ]:


df.head()


# In[ ]:


df.info()


# Checking the missing values in the dataset

# In[ ]:


df.isnull().sum()


# percentage of missing data

# In[ ]:


df.isnull().sum()/len(df)


# We have to remove the missing data in column (later)

# In[ ]:


df.describe().transpose()


# # Exploratory Data Analysis

# Cheking the heatmap of correlation with continous variables

# In[ ]:


plt.figure(figsize=(20,15))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
plt.tight_layout


# In[ ]:


df['satisfaction'].value_counts()


# In[ ]:


sns.countplot(x='satisfaction',data=df)


# From countplot we can say the classes are balanced 

# Now converting the satisfaction column to continuous variable. For this, a function named satisfac is created

# In[ ]:


def satisfac(string):
    if string == 'satisfied': return 1
    else: return 0

df['satisfactionN'] =df['satisfaction'].apply(satisfac)    


# In[ ]:


df.head(5)


# So column with numerical value of satisfaction is created  named satisfactionN,
# we can drop the satisfaction column

# In[ ]:


df.drop('satisfaction',inplace=True,axis=1)


# Now that satisfaction is continous 

# In[ ]:


plt.figure(figsize=(20,15))
sns.heatmap(df.corr(),annot = True,cmap='coolwarm')


# In[ ]:


df.corr()['satisfactionN'].sort_values().drop('satisfactionN').plot(kind='bar')


# Important continuous factors which contribute more correlation with customer satisfaction are  'Inflight wifi service','Flight Distance','cleanliness','Leg room service','on board service','seat comfort','inflight entertainment','online boarding'

# online boarding have maximum correlation with satisfaction

# In[ ]:


df['Online boarding'].value_counts()


# In[ ]:


df['Online boarding'].plot(kind='hist',ec='black')


# In[ ]:


sns.boxplot(x='satisfactionN',y = 'Online boarding',data=df)


# The more satisfied the person is with online boarding then there are greater chances that the person will be satisfied. Same is the case for all the other parameters 

# In[ ]:


df.head()


# Converting the catagorical parameters to continous using pandas inbuild function get_dummies

# In[ ]:


GenderN = pd.get_dummies(df['Gender'],drop_first=True)
CustomerN = pd.get_dummies(df['Customer Type'],drop_first=True)
TypeN = pd.get_dummies(df['Type of Travel'],drop_first=True)
ClassN = pd.get_dummies(df['Class'],drop_first=True)
df = pd.concat([df,GenderN,CustomerN,TypeN,ClassN],axis =1)
df.drop(['Gender','Customer Type','Type of Travel','Class'],inplace =True,axis = 1)


# In[ ]:


plt.figure(figsize=(25,20))
sns.heatmap(df.corr(),annot = True,cmap='coolwarm')


# In[ ]:


df.corr()['satisfactionN'].sort_values().drop('satisfactionN').plot(kind='bar')


# Analysing the catagorical variables, we came to know that People travelling in economy class and on a personal travel are unlikely to get satisfied.Now we know that people choosing online boarding are most satisfied. Lets check the correlation of online boarding with others

# In[ ]:


df.corr()['Online boarding'].sort_values().drop(['Online boarding','satisfactionN']).plot(kind='bar')


# It is majorly correlated with inflight wifi service 

# In[ ]:


sns.boxplot(x='Inflight wifi service',y = 'Online boarding',data=df)


# People who gets better service of inflight wifi likely to apply for online boarding and gives better rating.

# Analysing some general trends from the dataset using Plotly -> interactive plots

# In[ ]:


df['Flight Distance'].iplot(kind='hist',bins=50)


# Most of the flights are between 0 to 1000 kms

# In[ ]:


df['Age'].iplot(kind='hist',bins=50)


# Age has a normal distribution with most people between 20 to 60

# In[ ]:


import plotly.express as px
fig = px.box(df, x="satisfactionN", y="Age", color="Eco")
fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
fig.show()


# From the boxplot we can infer that people who are sitting in economy class between age 24 to 50 are likely to be more satisfied

# In[ ]:


sns.lmplot(x='Departure Delay in Minutes',y='Arrival Delay in Minutes',data=df)


# Arrival Delay and Departure delay have a linear relation, so we can drop one column

# In[ ]:


df.drop('Arrival Delay in Minutes',axis=1,inplace=True)


# id and unnamed column doesn't contain any information for model training, so we can drop those column too.(Later)

# # Preprocessing the data

# Removing missing data

# In[ ]:


df.isnull().sum()/len(df)
# Only 0.002% data is missing, so we can drop the rows 
# Data was missing in only "Arrival delay in Minutes" column , so these steps are not necessary


# In[ ]:


df.dropna(axis=0,inplace=True)


# Removing unnecessary data

# In[ ]:


df.drop(['Unnamed: 0','id'],axis=1,inplace=True)


# # Applying Different models 

# # Based on the performance,these three models suits our dataset quite efficiently
# # 1. Random Forest Classifier
# # 2. XgBoost
# # 3. Naive Bayes(for general classification)

# In[ ]:


#importing the libraries
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB


# In[ ]:


X_train = df.drop('satisfactionN',axis=1)
y_train = df['satisfactionN']


# Also applying all the preprocessing on test set and splitting the data into X_test and y_test

# In[ ]:


df2['satisfactionN'] =df2['satisfaction'].apply(satisfac)
GenderN = pd.get_dummies(df2['Gender'],drop_first=True)
CustomerN = pd.get_dummies(df2['Customer Type'],drop_first=True)
TypeN = pd.get_dummies(df2['Type of Travel'],drop_first=True)
ClassN = pd.get_dummies(df2['Class'],drop_first=True)
df2 = pd.concat([df2,GenderN,CustomerN,TypeN,ClassN],axis =1)
df2.drop(['Gender','Customer Type','Type of Travel','Class'],inplace =True,axis = 1)
df2.drop('Arrival Delay in Minutes',axis=1,inplace=True)
df2.drop(['Unnamed: 0','id'],axis=1,inplace=True)


# In[ ]:


df2.drop('satisfaction',axis=1,inplace=True)
X_test = df2.drop('satisfactionN',axis=1)
y_test= df2['satisfactionN']


# In[ ]:


print("X_train {}\nX_test {}\ny_train {}\ny_test {}".format(X_train.shape,X_test.shape,y_train.shape,y_test.shape))


# # Now applying Random forest

# In[ ]:


classifier1 = RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0,n_jobs=-1)
classifier1.fit(X_train,y_train)


# Predicting on test set 

# In[ ]:


y_pred = classifier1.predict(X_test)


# In[ ]:


# importing accuracy parameters
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(y_test,y_pred))
print('\n\n\n')
print('Confusion matrix : \n{}'.format(confusion_matrix(y_test,y_pred)))
print('\n')
print('Accuracy score : {}'.format(accuracy_score(y_test,y_pred)))
acc_random_forest = accuracy_score(y_test,y_pred)


# # Now applying XGBoost

# In[ ]:


classifier2 = XGBClassifier(n_estimators = 500,n_jobs=-1)
classifier2.fit(X_train,y_train)


# In[ ]:


#Predicting on test set results
y_pred = classifier2.predict(X_test)


# In[ ]:


# importing accuracy parameters
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(y_test,y_pred))
print('\n\n\n')
print('Confusion matrix : \n{}'.format(confusion_matrix(y_test,y_pred)))
print('\n')
print('Accuracy score : {}'.format(accuracy_score(y_test,y_pred)))
acc_xgboost = accuracy_score(y_test,y_pred)


# # Now applying naive_bayes

# In[ ]:


classifier3 = GaussianNB()
classifier3.fit(X_train,y_train)


# In[ ]:


#Predicting on test set results
y_pred = classifier3.predict(X_test)


# In[ ]:


# importing accuracy parameters
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(y_test,y_pred))
print('\n\n\n')
print('Confusion matrix : \n{}'.format(confusion_matrix(y_test,y_pred)))
print('\n')
print('Accuracy score : {}'.format(accuracy_score(y_test,y_pred)))
acc_naive_bayes = accuracy_score(y_test,y_pred)


# # Comparing the results

# In[ ]:


print('Accuracy:-\n')
print("Random Forest {}\nXGBoost {}\nNaive Bayes {}\n".format(acc_random_forest,acc_xgboost,acc_naive_bayes))


# # So Random Forest works best for this class
