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


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


ad= pd.read_csv('../input/advertising.csv')
ad.head()


# The data is based on a specific Company.

# In[ ]:


ad.info()


# # Lets Visualise the Data
# **We want to see the specific consumer group of Ad for the company**
# 

# In[ ]:


sns.set_style('white')
sns.set_context('notebook')


# In[ ]:


#Lets see a summary with respect to clicked on ad
sns.pairplot(ad, hue='Clicked on Ad', palette='bwr')


# We can see that daily less internet usage tends to click on ad more.

# In[ ]:


#Lets see Click on Ad features based on Sex
plt.figure(figsize=(10,6))
sns.countplot(x='Clicked on Ad',data=ad,hue='Male',palette='coolwarm')


# Female tends to click more on Ads!

# In[ ]:


#Distribution of top 12 country's ad clicks based on Sex 
plt.figure(figsize=(15,6))
sns.countplot(x='Country',data=ad[ad['Clicked on Ad']==1],order=ad[ad['Clicked on Ad']==1]['Country'].value_counts().index[:12],hue='Male',
              palette='viridis')
plt.title('Ad clicked country distribution')
plt.tight_layout()


# Most are developing countries and females are the active contributors.

# In[ ]:


#We will change the datetime object
ad['Timestamp']=pd.to_datetime(ad['Timestamp'])


# In[ ]:


#Now we shall introduce new columns Hour,Day of Week, Date, Month from timestamp
ad['Hour']=ad['Timestamp'].apply(lambda time : time.hour)
ad['DayofWeek'] = ad['Timestamp'].apply(lambda time : time.dayofweek)
ad['Month'] = ad['Timestamp'].apply(lambda time : time.month)
ad['Date'] = ad['Timestamp'].apply(lambda t : t.date())


# In[ ]:


#Hourly distribution of ad clicks
plt.figure(figsize=(15,6))
sns.countplot(x='Hour',data=ad[ad['Clicked on Ad']==1],hue='Male',palette='rainbow')
plt.title('Ad clicked hourly distribution')


# As we can see with uneven daytime frequency, females are the main contributor exceeding males several hours.

# In[ ]:


#Daily distribution of ad clicks
plt.figure(figsize=(15,6))
sns.countplot(x='DayofWeek',data=ad[ad['Clicked on Ad']==1],hue='Male',palette='rainbow')
plt.title('Ad clicked daily distribution')


# Most of the Days Ladies click ad more than Males except Wednesdays and Thursdays.

# In[ ]:


#Monthly distribution of ad clicks
plt.figure(figsize=(15,6))
sns.countplot(x='Month',data=ad[ad['Clicked on Ad']==1],hue='Male',palette='rainbow')
plt.title('Ad clicked monthly distribution')


# Throughout the Year Ladies click on Ad the most except month of March.

# In[ ]:


#Now we shall group by date and see the
plt.figure(figsize=(15,6))
ad[ad['Clicked on Ad']==1].groupby('Date').count()['Clicked on Ad'].plot()
plt.title('Date wise distribution of Ad clicks')
plt.tight_layout()


# In[ ]:


#Top Ad clicked on specific date
ad[ad['Clicked on Ad']==1]['Date'].value_counts().head(5)


# On 14th February 2016 we see most (8) clicks on ad. So Valentine Day is the best selling day for the Company's Ad.

# In[ ]:


ad['Ad Topic Line'].nunique()


# All ad topics are different which makes it really difficult to feed for model.

# In[ ]:


#Lets see Age distribution
plt.figure(figsize=(10,6))
sns.distplot(ad['Age'],kde=False,bins=40)


# Most of them are around 30 years. But is this age group clicking most on Ad?

# In[ ]:


#Lets see Age distribution
plt.figure(figsize=(10,6))
sns.swarmplot(x=ad['Clicked on Ad'],y= ad['Age'],data=ad,palette='coolwarm')
plt.title('Age wise distribution of Ad clicks')


# As its clear from above that around 40 years population are the most contributor to ad clickings and not around 30 years.

# In[ ]:


#Lets see Daily internet usage and daily time spent on site based on age
fig, axes = plt.subplots(figsize=(10, 6))
ax = sns.kdeplot(ad['Daily Time Spent on Site'], ad['Age'], cmap="Reds", shade=True, shade_lowest=False)
ax = sns.kdeplot(ad['Daily Internet Usage'],ad['Age'] ,cmap="Blues", shade=True, shade_lowest=False)
ax.set_xlabel('Time')
ax.text(20, 20, "Daily Time Spent on Site", size=16, color='r')
ax.text(200, 60, "Daily Internet Usage", size=16, color='b')


# As we can see people around 30 years population devote lot of their time on internet and on the site, but they don't click on Ads that frequent. Comapred to them, around 40 years population spend a bit less time but click on Ads more.

# In[ ]:


#Lets see the distribution who clicked on Ad based on area income of sex 
plt.figure(figsize=(10,6))
sns.violinplot(x=ad['Male'],y=ad['Area Income'],data=ad,palette='viridis',hue='Clicked on Ad')
plt.title('Clicked on Ad distribution based on area distribution')


# Both Males and Females with Area income less than 50k are main customers of Ad. As almost all whose income more than 60k are not interested on clicking on Ad.

# **Thus in conclusion, mostly around 40 years Female within income group less than 50k in developing countries are the main consumers of Ad, clicking unevenly throughout the day and mostly during Fridays and Sundays**

# # Data Cleaning

# In[ ]:


#Lets take country value as dummies
country= pd.get_dummies(ad['Country'],drop_first=True)


# In[ ]:


#Now lets drop the columns not required for building a model
ad.drop(['Ad Topic Line','City','Country','Timestamp','Date'],axis=1,inplace=True)


# In[ ]:


#Now lets join the dummy values
ad = pd.concat([ad,country],axis=1)


# # Logistic Regression Model

# In[ ]:


from sklearn.model_selection import train_test_split
X= ad.drop('Clicked on Ad',axis=1)
y= ad['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# In[ ]:


from sklearn.linear_model import LogisticRegression
logmodel= LogisticRegression()
logmodel.fit(X_train,y_train)


# **For better parameters we will apply GridSearch**

# In[ ]:


from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1,10,100,1000]}


# In[ ]:


grid_log= GridSearchCV(LogisticRegression(),param_grid,refit=True, verbose=2)


# In[ ]:


grid_log.fit(X_train,y_train)


# In[ ]:


grid_log.best_estimator_


# In[ ]:


pred_log= grid_log.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
print(confusion_matrix(y_test,pred_log))
print(classification_report(y_test,pred_log))


# **Let's compare it with other Classification Models!**

# # Random Forest Model

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# As the selections are random here, we won't apply Grid Search as it gives different values. Thus we shall safely assume n_estimators as 200.

# In[ ]:


rfc= RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)


# In[ ]:


pred_rfc= rfc.predict(X_test)
print(confusion_matrix(y_test,pred_rfc))
print(classification_report(y_test,pred_rfc))


# **It does pretty good job comapred to Logistic Regressor.**

# # Support Vector Model

# In[ ]:


from sklearn.svm import SVC
svc= SVC(gamma='scale')


# In[ ]:


svc.fit(X_train,y_train)


# **For better parameters we need to apply GridSearch**

# In[ ]:


param_grid = {'C': [0.1,1,10,100,1000,5000]}


# In[ ]:


grid_svc= GridSearchCV(SVC(gamma='scale',probability=True),param_grid,refit=True,verbose=2)


# In[ ]:


grid_svc.fit(X_train,y_train)


# In[ ]:


grid_svc.best_estimator_


# In[ ]:


pred_svc= grid_svc.predict(X_test)
print(confusion_matrix(y_test,pred_svc))
print(classification_report(y_test,pred_svc))


# **It does perform better but a slight less than Random Forest Model.**

# # Let's use Soft Voting Classifier for above three models

# In[ ]:


from sklearn.ensemble import VotingClassifier
vote= VotingClassifier(estimators=[('logmodel',grid_log),('rfc',rfc),('svc',grid_svc)],voting='soft')


# In[ ]:


vote.fit(X_train,y_train)


# In[ ]:


pred_vote= vote.predict(X_test)
print(confusion_matrix(y_test,pred_vote))
print(classification_report(y_test,pred_vote))


# **Ok, So we got a better model. Let's build a final classifier model using KNN.**

# # KNN Classifier Model

# In[ ]:


#let's first scale the variables
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()


# In[ ]:


scaler.fit(ad.drop('Clicked on Ad',axis=1))


# In[ ]:


scaled_features= scaler.transform(ad.drop('Clicked on Ad',axis=1))


# In[ ]:


#Changing it from numpy array to pandas dataframe
train_scaled = pd.DataFrame(scaled_features,columns=ad.columns.drop('Clicked on Ad'))
train_scaled.head()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train_scaled,ad['Clicked on Ad'],test_size=0.20,random_state=101)


# **Let's choose a k-value using elbow method**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
error_rate=[]

for i in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(range(1,50),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.title('Error Rate vs K-value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# We get an elbow after 20. We choose best elbow value at k =40

# In[ ]:


knn= KNeighborsClassifier(n_neighbors=40)
knn.fit(X_train,y_train)


# In[ ]:


pred_knn=knn.predict(X_test)
print(confusion_matrix(y_test,pred_knn))
print(classification_report(y_test,pred_knn))


# **As we can see KNN classifier model provides us very good results.**

# **To overcome remaining errors we need to build our model strong. We need to consider 'AdTopic Line' column in our model.**

# In[ ]:




