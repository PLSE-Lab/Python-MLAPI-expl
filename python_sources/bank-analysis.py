#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


# In[2]:


#Load file 
df = pd.read_csv('../input/bank-additional-full.csv',sep = ';')
df.head() #Take a look at the data 


# In[ ]:


df.info() #10 numerical columns, 11 cateogorical 


# In[ ]:


df.describe()


# In[3]:


#Rename target variable column 
df['subscribed'] = df['y']
df = df.drop(['y'], axis = 1)
df.head()


# In[ ]:


pd.pivot_table(df, index = 'default', values = 'subscribed', aggfunc = np.size)


# In[ ]:


df['age'].plot(kind = 'hist', color = 'lightblue') #See gender distribution of age 


# In[ ]:


jobs =  pd.pivot_table(df, index = 'job', values = 'subscribed', aggfunc = np.size)
jobs.plot(kind = 'bar', color = 'lightblue')
plt.xticks(rotation = 90)
plt.legend().remove()
plt.title('Client Job Distribution ')


# In[ ]:


df.head()


# In[ ]:


df['marital'].value_counts(normalize = True).plot(kind = 'bar') #Plot percentage distribution of marital status
plt.xticks(rotation = 45)
plt.title('Marital Status of Clients')
plt.ylabel('% of Clients')
plt.xlabel('Marital Status')

#Almost 60% of clients are married, while nearly 30% are single 


# Machine Learning Portion 

# In[4]:


y = df['subscribed']
X = df.drop(['subscribed','duration'], axis = 1)
X = pd.get_dummies(X) #one hot encode 
print(X.shape)
print(y.shape) #Shape matching in rows! 


# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_Test, y_train, y_test = train_test_split(X,y, test_size = .2, random_state = 40) #20% of data as test


# In[6]:


from sklearn.ensemble import RandomForestClassifier 
model = RandomForestClassifier(n_estimators = 50)
model.fit(X_train, y_train)


# In[7]:


#Let's create predictions 
preds = model.predict(X_Test)


# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

cv_score = cross_val_score(model,X,y, cv = 10, scoring = 'roc_auc')


# In[9]:


from sklearn import metrics 

print('Accuracy:',round(metrics.accuracy_score(y_test,preds),2)) #88% accuracy- not bad for limited data wrangling 


# In[ ]:


feature_values = pd.DataFrame(model.feature_importances_,
                              index = X_train.columns,
                              columns = ['importance']).sort_values('importance',
                                                                    ascending=False)
feature_values.head(8).plot(kind = 'bar', color = 'lightgreen')
plt.xticks(rotation = 85)
plt.title('Feature Importance')
plt.legend().remove()


# **Let's try out SVM **

# In[10]:


from sklearn import svm 

#Create classifier model 
clf = svm.SVC(kernel = 'linear')

#Train Model 
clf.fit(X_train, y_train)
       
#Predict 
y_pred = clf.predict(X_Test)


# In[11]:


from sklearn import metrics 

print('Accuracy:',round(metrics.accuracy_score(y_test,y_pred),2)) #89% Accuracy- we've gone up about a point! 


# In[16]:


from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors = 4) #Create model

clf.fit(X_train,y_train) #Fit model 

y_pred = clf.predict(X_Test) #Predict on test dataset 

print('Accuracy:',round(metrics.accuracy_score(y_test,y_pred),3))  #Check accuracy of model
88.5% Accuracy! 

