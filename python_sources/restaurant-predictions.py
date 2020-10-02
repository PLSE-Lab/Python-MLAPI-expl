#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing required packages
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from plotly.offline import init_notebook_mode, iplot
from plotly.graph_objs import *
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import string
import os


# In[ ]:


#reading data
F1=pd.read_csv("../input/chefmozaccepts.csv")
F2=pd.read_csv("../input/chefmozcuisine.csv")
F3=pd.read_csv("../input/chefmozhours4.csv")
F4=pd.read_csv("../input/chefmozparking.csv")
F5=pd.read_csv("../input/usercuisine.csv")
F6=pd.read_csv("../input/userpayment.csv")
F7=pd.read_csv("../input/userprofile.csv")
F8=pd.read_csv("../input/geoplaces2.csv")

T=pd.read_csv("../input/rating_final.csv")


# Now we are going to explore and preprocess each csv files individualy

# # F1-chefmozaccepts.csv

# In[ ]:


F1.head()


# In[ ]:


F1.info()


# In[ ]:


# plot to visualise most accepted payments by Restaurants
F1plt=F1.Rpayment.value_counts().plot.bar(title="Payments Accepted")
F1plt.set_xlabel('payments mode',size=15)
F1plt.set_ylabel('count',size=15)


# In[ ]:


#creating dummy variables for differen payments.
F1dum = pd.get_dummies(F1,columns=['Rpayment'])
F1dum1 = F1dum.groupby('placeID',as_index=False).sum()
len(F1dum1)


# In[ ]:


F1dum1.head()


# # F2 - chefmozcuisine.csv

# In[ ]:


F2.head()


# In[ ]:


F2.info()


# In[ ]:


#plot to visualize top cuisines offered by the restaurants
F2plt=F2.Rcuisine.value_counts()[:10].plot.bar(title="Top 10 cuisine")
F2plt.set_xlabel('cuisine',size=15)
F2plt.set_ylabel('count',size=15)


# In[ ]:


#creating dummy variables for different cuisines.
F2dum = pd.get_dummies(F2,columns=['Rcuisine'])
F2dum1 = F2dum.groupby('placeID',as_index=False).sum()
len(F2dum1)


# In[ ]:


F2dum1.head()


# # F3 - chefmozhours4.csv

# In[ ]:


F3.head()


# In[ ]:


F3.info()


# # F4 - chefmozparking.csv

# In[ ]:


F4.head()


# In[ ]:


F4.info()


# In[ ]:


#plot to visualize available parking place at the Restaurants
F4plt=F4.parking_lot.value_counts().plot.bar(title="parking place")
F4plt.set_xlabel('Available parking',size=15)
F4plt.set_ylabel('count',size=15)


# In[ ]:


#creating dummy variables for different parking lots.
F4dum = pd.get_dummies(F4,columns=['parking_lot'])
F4dum1 = F4dum.groupby('placeID',as_index=False).sum()
len(F4dum1)


# In[ ]:


F4dum1.head()


# # F5 - usercuisine.csv

# In[ ]:


F5.head()


# In[ ]:


F5.info()


# In[ ]:


#Top 10 favorite cuisines for the customers
F5plt=F5.Rcuisine.value_counts()[:10].plot.bar(title="Top 10 user cuisine")
F5plt.set_xlabel('user cuisine',size=15)
F5plt.set_ylabel('count',size=15)


# In[ ]:


#creating dummy variables for differen usercuisines.
F5dum = pd.get_dummies(F5,columns=['Rcuisine'])
F5dum1 = F5dum.groupby('userID',as_index=False).sum()
len(F5dum1)


# In[ ]:


F5dum1.head()


# # F6 - userpayment.csv

# In[ ]:


F6.head()


# In[ ]:


F6.info()


# In[ ]:


#top type of payments done by the users
F6plt=F6.Upayment.value_counts().plot.bar(title="User payments")
F6plt.set_xlabel('User payments',size=15)
F6plt.set_ylabel('count',size=15)


# In[ ]:


#creating dummy variables for different userpayments.
F6dum = pd.get_dummies(F6,columns=['Upayment'])
F6dum1 =F6dum.groupby('userID',as_index=False).sum()
len(F6dum1)


# In[ ]:


F6dum1.head()


# # F7 - userprofile.csv

# In[ ]:


F7.head()


# In[ ]:


F7.info()


# In[ ]:


# as data contains unknown value, we are replacinf with Nan.
F7rep=F7.replace('?', np.nan)


# In[ ]:


#now we are finding missing value cnt n perct for all variables.
mss=F7rep.isnull().sum()
columns = F7rep.columns
percent_missing = F7rep.isnull().sum() * 100 / len(F7rep)
missing_value_F7rep = pd.DataFrame({'missing_cnt':mss,'percent_missing': percent_missing})
missing_value_F7rep


# In[ ]:


#since the missing value pernt is very low in each variables, we are replacing with mode of that individual column.
for column in F7rep.columns:
    F7rep[column].fillna(F7rep[column].mode()[0], inplace=True)


# In[ ]:


#plotting for marital status vs smoker n drinklevel.
F7rep.groupby('marital_status')['smoker','drink_level'].nunique().plot.bar(rot=0)


# In[ ]:


#plot to visualize user's personal info based on birthyear.
F7repplt=F7rep.groupby('birth_year')['interest','personality','religion','activity'].nunique().plot.bar(figsize=(15, 5))


# In[ ]:


#now performing label encoding to convert char to factors.
F7char=F7rep.select_dtypes(include=['object'])

encoder = LabelEncoder()
F7charLE = F7char.apply(encoder.fit_transform, axis=0)
F7charLE=F7charLE.drop(['userID'],axis=1)
F7charLE[['userID','latitude','longitude','birth_year','weight','height']]=F7rep[['userID','latitude','longitude','birth_year','weight','height']]
F7charLE.head()


# # F8 - geoplaces2.csv

# In[ ]:


F8.head()


# In[ ]:


F8.info()


# In[ ]:


#replacing unknown value with Nan.
F8rep=F8.replace('?', np.nan)


# In[ ]:


#now we are finding missing value cnt n perct for all variables.
mss=F8rep.isnull().sum()
columns = F8rep.columns
percent_missing = F8rep.isnull().sum() * 100 / len(F8rep)
missing_value_F8rep = pd.DataFrame({'missing_cnt':mss,
                                 'percent_missing': percent_missing})
missing_value_F8rep


# In[ ]:


#dropping columns with more than 50% missing values
F8new=F8rep.drop(['fax','zip','url'],axis=1)
#and replacing remaining colvalues with mode
for column in F8new.columns:
    F8new[column].fillna(F8new[column].mode()[0], inplace=True)


# since some variables contains dirty values, we are going to perform data cleaning on those variables.

# In[ ]:


#clean n cnt of city
F8new.city=F8new.city.apply(lambda x: x.lower())
F8new.city=F8new.city.apply(lambda x:''.join([i for i in x 
                            if i not in string.punctuation]))

F8new.city.value_counts()


# In[ ]:


#replacing city with unique. 
F8new['city']=F8new['city'].replace(['san luis potos','san luis potosi','slp','san luis potosi '],'san luis potosi' )
F8new['city']=F8new['city'].replace(['victoria','cd victoria','victoria '],'ciudad victoria' )
F8new.city.value_counts()


# In[ ]:


#clean n cnt of state
F8new.state=F8new.state.apply(lambda x: x.lower())
F8new.state=F8new.state.apply(lambda x:''.join([i for i in x 
                            if i not in string.punctuation]))

F8new.state.value_counts()


# In[ ]:


#replacing state with unique.
F8new['state']=F8new['state'].replace(['san luis potos','san luis potosi','slp'],'san luis potosi' )
F8new.state.value_counts()


# In[ ]:


#clean n cnt of country
F8new.country=F8new.country.apply(lambda x: x.lower())
F8new.country=F8new.country.apply(lambda x:''.join([i for i in x 
                            if i not in string.punctuation]))

F8new.country.value_counts()


# In[ ]:


#label encoding
F8char=F8new.select_dtypes(include=['object'])
F8charLE = F8char.apply(encoder.fit_transform, axis=0)
F8charLE[['placeID','latitude','longitude']]=F8new[['placeID','latitude','longitude']]
F8charLE.head()


# In[ ]:


#plot for facilities provided by Restaurants based on city.
F8newplt=F8new.groupby('city')['alcohol','smoking_area','accessibility','price','Rambience','other_services'].nunique().plot.bar(figsize=(15,5))


# Producing Map location for the Restaurants

# In[ ]:


mapbox_access_token='pk.eyJ1IjoibmF2ZWVuOTIiLCJhIjoiY2pqbWlybTc2MTlmdjNwcGJ2NGt1dDFoOSJ9.z5Jt4XxKvu5voCJZBAenjQ'


# In[ ]:


mcd=F8rep[F8rep.country =='Mexico']
mcd_lat = mcd.latitude
mcd_lon = mcd.longitude

data = [
    go.Scattermapbox(
        lat=mcd_lat,
        lon=mcd_lon,
        mode='markers',
        marker=dict(
            size=6,
            color='rgb(255, 0, 0)',
            opacity=0.4
        ))]
layout = go.Layout(
    title='Restaurants Locations',
    autosize=True,
    hovermode='closest',
    showlegend=False,
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=23,
            lon=-102
        ),
        pitch=2,
        zoom=4.5,
        style='dark'
    ),
)

fig = dict(data=data, layout=layout)
iplot(fig, filename='restaurants')


# # Merging multiple files into one

# In[ ]:


#merging ratingfile(T) with userprofile(F7)
A=pd.merge(T,F7charLE)


# In[ ]:


#merging A with userpayments(F6) 
B=pd.merge(A,F6dum1,how='left',on=['userID'])


# In[ ]:


#merging B with usercuisine(F5)
C=pd.merge(B,F5dum1,how='left',on=['userID'])


# In[ ]:


#merging C with geoplaces2(F8)
D=pd.merge(C,F8charLE,how='left',on=['placeID'])


# In[ ]:


#merging D with chefmozparking(F4)
E=pd.merge(D,F4dum1,how='left',on=['placeID'])


# In[ ]:


#merging E with chefmozcuisine(F2)
F=pd.merge(E,F2dum1,how='left',on=['placeID'])


# In[ ]:


#merging F with chefmozaccepts(F1)
G=pd.merge(F,F1dum1,how='left',on=['placeID'])


# In[ ]:


len(G)


# # Final Data

# In[ ]:


G.head()


# In[ ]:


G.info()


# In[ ]:


print('No of columns',G.shape[1])
print('No of rows',G.shape[0])


# In[ ]:


#check for Null values
G.isnull().values.any()


# In[ ]:


#finding percentage of null values across columns
columns = G.columns
percent_missing = G.isnull().sum() * 100 / len(G)
missing_value_G = pd.DataFrame({'percent_missing': percent_missing})
missing_value_G


# In[ ]:


#replacing missing values with zero and check.
G=G.fillna(0)
G.isnull().values.any()


# In[ ]:


#for modelling purpose we are label encoding userID.
G['userID']=encoder.fit_transform(G['userID'])


# # Model Building

# In[ ]:


#packages for modelling
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score


# In[ ]:


# splitting train and test data as 75/25.
X=G.drop(['placeID','rating','food_rating','service_rating'],axis=1)
y=G['rating']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)


# ##Logistic Regression

# In[ ]:


#model building.
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[ ]:


#predicting on test data.
predictions1 =  logmodel.predict(X_test)


# In[ ]:


print("confusion matrix")
print(confusion_matrix(y_test,predictions1))
print("Accuracy_score")
print(accuracy_score(y_test, predictions1))


# In[ ]:


print("classification_report")
print(classification_report(y_test,predictions1))


# In[ ]:


#kappa score.
cohen_kappa_score(y_test, predictions1)


# ##Decision Tree

# In[ ]:


#model building.
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)


# In[ ]:


#predicting on test data.
predictions2 =  clf.predict(X_test)


# In[ ]:


print("confusion matrix")
print(confusion_matrix(y_test,predictions2))
print("Accuracy_score")
print(accuracy_score(y_test, predictions2))


# In[ ]:


print("classification_report")
print(classification_report(y_test,predictions2))


# In[ ]:


#kappa score.
cohen_kappa_score(y_test, predictions2)


# ##Random Forest

# In[ ]:


#model building.
Rndclf = RandomForestClassifier(max_depth=2, random_state=0) 
Rndclf.fit(X_train,y_train)


# In[ ]:


#predicting on test data.
predictions3 = Rndclf.predict(X_test)


# In[ ]:


print("confusion matrix")
print(confusion_matrix(y_test,predictions3))
print("Accuracy_score")
print(accuracy_score(y_test, predictions3))


# In[ ]:


print("classification_report")
print(classification_report(y_test,predictions3))


# In[ ]:


#kappa score.
cohen_kappa_score(y_test, predictions3)


# ##XGboost model

# In[ ]:


#model building.
xgb = XGBClassifier()
xgb.fit(X_train, y_train)


# In[ ]:


#predicting on test data.
predictions4 = xgb.predict(X_test)


# In[ ]:


print("confusion matrix")
print(confusion_matrix(y_test,predictions4))
print("Accuracy_score")
print(accuracy_score(y_test, predictions4))


# In[ ]:


print("classification_report")
print(classification_report(y_test,predictions4))


# In[ ]:


#kappa score.
cohen_kappa_score(y_test, predictions4)


# # By Comparing the above built models, we find that XGboost is giving better predictions based on Accuracy,F1-score and kappa score.
# 
