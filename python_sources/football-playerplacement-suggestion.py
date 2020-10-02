#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
url="https://datafaculty.s3.us-east-2.amazonaws.com/Indore/song_football-class13.csv"


# In[ ]:


data=pd.read_csv(url,encoding="latin1")


# In[ ]:


data.head(2)


# In[ ]:


###### Performance of football players
### Find a replacement for the player. 
### Name is Alexandre Song, 


# In[ ]:


data.shape[0]


# In[ ]:


#### How many clusters should I make?
## We do have a context most of the times about the number of cluster


# In[ ]:


## 10 to 20 players, 30 cluster


# In[ ]:


480/20


# In[ ]:


## Agglomerative Clustering


# In[ ]:


data_num=data.drop(['Player Id','Last_Name','First_Name'],axis=1)


# In[ ]:


data_num.head(2)


# In[ ]:


data_num.isnull().sum()


# In[ ]:


data_num.describe()


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler=StandardScaler()


# In[ ]:


data_scaled=scaler.fit_transform(data_num)


# In[ ]:


data_scaled


# In[ ]:


from sklearn import cluster


# In[ ]:


mod1=cluster.AgglomerativeClustering(n_clusters=30)


# In[ ]:


mod1=mod1.fit(data_scaled)


# In[ ]:


data['labels_1']=mod1.labels_


# In[ ]:


data.head(2)


# In[ ]:


data[data['Last_Name']=='Song']


# In[ ]:


data[data['labels_1']==5].to_csv("final_candidates.csv",index=False)


# In[ ]:


url1="https://datafaculty.s3.us-east-2.amazonaws.com/Indore/train.csv"
url2="https://datafaculty.s3.us-east-2.amazonaws.com/Indore/store.csv"


# In[ ]:


train=pd.read_csv(url1)


# In[ ]:


store=pd.read_csv(url2)


# In[ ]:


train.head(2)


# In[ ]:


store.head(2)


# In[ ]:


store.shape


# In[ ]:


train.shape


# In[ ]:


train['Store'].unique()


# In[ ]:


### ML based regressors we can build ts models with predictors


# In[ ]:


store.head(2)


# In[ ]:


store.shape


# In[ ]:


train.head(2)


# In[ ]:


train.shape


# In[ ]:


data=pd.merge(train,store,on='Store',how="left")


# In[ ]:


data.shape


# In[ ]:


data.head(2)


# In[ ]:


###### Time Information, Store level information <===>
### Day,month,year
data['Date']=pd.to_datetime(data['Date'])


# In[ ]:


data['Month']=data['Date'].dt.month


# In[ ]:


data['Year']=data['Date'].dt.year


# In[ ]:


data.isnull().sum()


# In[ ]:


data.head(2)


# In[ ]:


len(data['StoreType'].unique())


# In[ ]:


len(data['Assortment'].unique())


# In[ ]:


data['StateHoliday'].unique()


# In[ ]:


data['SchoolHoliday'].unique()


# In[ ]:


data['PromoInterval'].unique()


# In[ ]:


### Which predictors we would want to use.
### Time-information- Day,Month,Year
### Store specific variables -


# In[ ]:


### Replace the missing values with a default number
### Replace the missing values with a string, "missing", 


# In[ ]:


data['CompetitionDistance'].head(2)


# In[ ]:


data['CompetitionOpenSinceMonth'].unique()


# In[ ]:


#data_cat=data[['StoreType','Assortment','StateHoliday']]


# In[ ]:


data['CompetitionOpenSinceYear'].unique()


# In[ ]:


data['CompetitionOpenSinceMonth']=data['CompetitionOpenSinceMonth'].fillna("missing")


# In[ ]:


data['CompetitionOpenSinceMonth']=data['CompetitionOpenSinceMonth'].map(lambda x: str(x))


# In[ ]:


data['CompetitionOpenSinceMonth'].unique()


# In[ ]:


data['CompetitionOpenSinceYear'].fillna('missing')
data['CompetitionOpenSinceYear']=data['CompetitionOpenSinceYear'].map(lambda x: str(x))


# In[ ]:


data['Promo2SinceWeek'].unique()


# In[ ]:


data['Promo2SinceYear'].unique()


# In[ ]:


data['Promo2SinceWeek']=data['Promo2SinceWeek'].fillna("missing")
data['Promo2SinceWeek']=data['Promo2SinceWeek'].map(lambda x : str(x))
data['Promo2SinceYear']=data['Promo2SinceYear'].fillna("missing")
data['Promo2SinceYear']=data['Promo2SinceYear'].map(lambda x: str(x))


# In[ ]:


data.dtypes


# In[ ]:


data.isnull().sum()


# In[ ]:


data['CompetitionDistance']=data['CompetitionDistance'].fillna(data['CompetitionDistance'].mean())


# In[ ]:


data.isnull().sum()


# In[ ]:


### How would I treat categorical columns


# In[ ]:


2642/data.shape[0]


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


enc=LabelEncoder()
data['CompetitionOpenSinceMonth']=enc.fit_transform(data['CompetitionOpenSinceMonth'])


# In[ ]:


data.head(2)


# In[ ]:


enc=LabelEncoder()
data['CompetitionOpenSinceYear']=enc.fit_transform(data['CompetitionOpenSinceYear'])


# In[ ]:


enc=LabelEncoder()
data['Promo2SinceWeek']=enc.fit_transform(data['Promo2SinceWeek'])


# In[ ]:


enc=LabelEncoder()
data['Promo2SinceYear']=enc.fit_transform(data['Promo2SinceYear'])


# In[ ]:


data.dtypes


# In[ ]:


data.head(2)


# In[ ]:


data['StateHoliday'].unique()


# In[ ]:


enc=LabelEncoder()
data['StateHoliday']=data['StateHoliday'].map(lambda x: str(x))
data['StateHoliday']=enc.fit_transform(data['StateHoliday'])


# In[ ]:


enc=LabelEncoder()
data['StoreType']=enc.fit_transform(data['StoreType'])


# In[ ]:


enc=LabelEncoder()
data['Assortment']=enc.fit_transform(data['Assortment'])


# In[ ]:


data.isnull().sum()


# In[ ]:


features=['Store','DayOfWeek','Open','Promo','StateHoliday','SchoolHoliday','StoreType',         'Assortment','CompetitionDistance','CompetitionOpenSinceMonth','CompetitionOpenSinceYear',
         'Promo2','Promo2SinceWeek','Promo2SinceYear','Month','Year']


# In[ ]:


X=data[features]
y=data['Sales']


# In[ ]:


#### Split data on train and test
import sklearn.model_selection as model_selection


# In[ ]:


X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.20,random_state=42)


# In[ ]:


import sklearn.ensemble as ensemble


# In[ ]:


reg=ensemble.RandomForestRegressor(n_estimators=100,min_samples_leaf=100,n_jobs=-1)


# In[ ]:


reg=reg.fit(X_train,y_train)


# In[ ]:


###X_train.isnull().sum()


# In[ ]:


preds=reg.predict(X_test)


# In[ ]:


X_test['preds']=preds
X_test['actuals']=y_test


# In[ ]:


X_test=X_test.sort_values(['Year','Month','DayOfWeek'])


# In[ ]:


X_test=X_test.reset_index()


# In[ ]:


X_test


# In[ ]:


X_test_store1=X_test.query("Store==1").reset_index()


# In[ ]:


X_test_store1[X_test_store1['actuals']==0]


# In[ ]:


import plotly.express as px


# In[ ]:


fig=px.line(X_test_store1,x=X_test_store1.index,y="preds")
fig=fig.add_scatter(x=X_test_store1.index,y=X_test_store1['actuals'],name="actual")
fig


# In[ ]:




