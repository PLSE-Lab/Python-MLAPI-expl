#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.tree import DecisionTreeRegressor
import sklearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso


# In[ ]:


df=pd.read_csv("../input/top50spotify2019/top50.csv",encoding='latin-1')


# The encoding techinque of 'latin-1' is used because we have some characteristics/ alphabets whihc are not english

# # Letse see the data and some stats around it 

# In[ ]:


df.head(2)


# In[ ]:


df.shape


# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:


#now we know the data and the columns shape and mean and other values , lets undestand the  data by plotting the varaivles or the features


# In[ ]:


#data_plot=pd.DataFrame['Energy','Danceability']
sns.pairplot(df)
plt.figure(figsize=(10,10))


# In[ ]:


sns.distplot(df['Energy'],hist=False,kde=True)


# In[ ]:


sns.distplot(df['Danceability'],hist=False,kde=True)


# In[ ]:


plt.scatter(df.Danceability,df.Energy,data=df,alpha=0.8)
plt.show()


# In[ ]:


plt.scatter(df.Danceability,df.Popularity,data=df,alpha=0.8)
plt.xlabel('dance')
plt.ylabel('Popularity')
plt.show()


# In[ ]:


#from the columns diaplyed below, we have some discrepency in the coulmns


# In[ ]:


df.columns


# In[ ]:


#lets rename the cplumns to make all the columns looks in sync


# some column names have extra values like '.' .

# In[ ]:


df.rename(columns={'Unnamed: 0':'unnamed','Track.Name':'TrackName','Artist.Name':'ArtistName',
                   'Beats.Per.MinuteBeatsPerMinute':'BeatsPerMinute','Loudness..dB..':'Loudnesdf','Valence.':'Valence',
                   'Acousticness..':'Acousticness','Speechiness.':'Speechiness','Beats.Per.Minute':'BeatsPerMinute',
                   'Length.':'Length','Track.Name':'TrackName',
                   'Artist.Name':'ArtistName'},inplace=True)


# In[ ]:


df.columns


# renamed the column names to make consistency

# In[ ]:


plt.figure(figsize=(10,10))
corr=df.corr()
sns.heatmap(corr,annot=True)


# speechiness and beats per minute are highly correlated
# Loundness and energy as well
# 

# In[ ]:


#by  the above correlation heatma , we can get a clear picture of how each varaible is corealted with the other varaible


# In[ ]:


df.Genre.unique()


# In[ ]:


leble_en=preprocessing.LabelEncoder()
df['Genre']=leble_en.fit_transform(df['Genre'])


# In[ ]:


#lable encoder is the technique of converting categorical variable to continious variable


# In[ ]:


df['Genre'].unique()


# In[ ]:




x=df.drop(['Popularity','TrackName','unnamed','ArtistName'],axis=1)
y=df['Popularity']


# In[ ]:


#for our analysis , lets consider popularity is the Target variable which needs to predicted .
#and in source data we dones needs to above columns which are not adding much value to the prediction


# In[ ]:


x_train,x_test,y_train,y_tes=train_test_split(x,y,test_size=0.8,random_state=20)


# # Lets start with Linear Regession model

# In[ ]:


le_reg=LinearRegression()
le_reg.fit(x_train,y_train)
predict_linear_reg=le_reg.predict(x_test)


# In[ ]:


sqrt(mean_squared_error(predict_linear_reg,y_tes))


# In[ ]:


result=pd.DataFrame({'actual':y_tes,'cal':predict_linear_reg,'diff':abs(y_tes-predict_linear_reg)})


# In[ ]:


result.head(2)


# In[ ]:


#The root mean square value with Linear Regression is 7.5 lets see if we can bring it to down using other models


# # Logistic Regression

# In[ ]:


logistic_reg=LogisticRegression()
logistic_reg.fit(x_train,y_train)
logic_pred=logistic_reg.predict(x_test)


# In[ ]:


sqrt(mean_squared_error(logic_pred,y_tes))


# In[ ]:


#The root mean square error optained by using Logistic Regression is 6.48.Good sign lets see if the error can be
#reducted further


# # Decision Tree

# In[ ]:


model=DecisionTreeRegressor()
model.fit(x_train,y_train)
predict_dec=model.predict(x_test)


# In[ ]:


sqrt(mean_squared_error(predict_dec,y_tes))


# In[ ]:


plt.plot(predict_dec,y_tes)


# In[ ]:


#The root mean square error optained by using Decision Tree Regression is 5.6.


# # KNeighbors

# In[ ]:


knn_model=KNeighborsRegressor()
knn_model.fit(x_train,y_train)
knn_model_predict=knn_model.predict(x_test)


# In[ ]:


sqrt(mean_squared_error(knn_model_predict,y_tes))


# In[ ]:


#The root mean square error optained by using KNeighbor Regression is 5.12 [preety god sign]


# In[ ]:


#lets try if we can further reduce with different values of K


# # Lasso Regression

# In[ ]:


model_lasso=Lasso()
model_lasso.fit(x_train,y_train)
lasso_pred=model_lasso.predict(x_test)


# In[ ]:


sqrt(mean_squared_error(lasso_pred,y_tes))


# In[ ]:


#poor , lasso  Regression doesnt help much here


# In[ ]:


#RandomForestRegressor


# In[ ]:


model_random=RandomForestRegressor()
model_random.fit(x_train,y_train)
random_pred=model_random.predict(x_test)


# In[ ]:


sqrt(mean_squared_error(random_pred,y_tes))


# In[ ]:


#The root mean square error obtained with random forest is the least of all with an error as 5.10


# In[ ]:




