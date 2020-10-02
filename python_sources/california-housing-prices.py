#!/usr/bin/env python
# coding: utf-8

# # California Housing Prices

# <img src='https://www.ocregister.com/wp-content/uploads/2017/04/oo24dg-web0409buslrealtrendsrisk02.jpg?w=620' />

# ## import important library : we will import other libraries when we need them
# better way is to speculate what you will need and import exactly that thing for saving memory usage.

# In[ ]:


## import important library : we will import other libraries when we need them
#better way is to speculate what you will need and import exactly that thing for saving memory usage.import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


os.listdir()


# In[ ]:


data= pd.read_csv('/kaggle/input/housing.csv')


# In[ ]:


data.head()


# ## DATA WRANGLING

# In[ ]:


data.count()


# In[ ]:


data.total_bedrooms.isnull().sum()


# In[ ]:


data.loc[data.total_bedrooms.isnull()]


# In[ ]:


data.ocean_proximity.unique()


# In[ ]:


data.ocean_proximity.value_counts()


# In[ ]:


data.groupby(['ocean_proximity'])['total_bedrooms'].sum()


# In[ ]:


data.groupby(['ocean_proximity'])['total_bedrooms'].mean()


# In[ ]:


x=4937435/9136
x


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[ ]:


corr = data.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corr,cmap="YlGnBu")


# In[ ]:


data.groupby(['ocean_proximity'])['total_bedrooms'].median()


# ## DATA CLEANING

# In[ ]:


data.ocean_proximity.head()


# In[ ]:


## DATA CLEANING
data.loc[(data.total_bedrooms.isnull())&(data.ocean_proximity=='<1H OCEAN'), 'total_bedrooms']=438.0
data.loc[(data.total_bedrooms.isnull())&(data.ocean_proximity=='INLAND'),'total_bedrooms']= 423.0
data.loc[(data.total_bedrooms.isnull())&(data.ocean_proximity=='ISLAND'),'total_bedrooms']= 512.0
data.loc[(data.total_bedrooms.isnull())&(data.ocean_proximity=='NEAR BAY'),'total_bedrooms']= 423.0
data.loc[(data.total_bedrooms.isnull())&(data.ocean_proximity=='NEAR OCEAN'),'total_bedrooms']= 464.0


# In[ ]:


data.total_bedrooms.isnull().sum()


# In[ ]:


# next 3 figures are learned from https://www.kaggle.com/manisood001/california-housing-optimised-modelling kernel 
plt.figure(figsize=(10,5))
sns.distplot(data['median_house_value'],color='green')
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))

plt.scatter(data['population'],data['median_house_value'],c=data['median_house_value'], s=data['median_income']*10)
plt.colorbar
plt.title('population vs house value' )
plt.xlabel('population')
plt.ylabel('house value')
plt.plot()


# In[ ]:


# s=size of circles, c= color of circles
plt.figure(figsize=(15,15))
plt.scatter(data['longitude'],data['latitude'],c=data['median_house_value'],s=data['population']/10,cmap='viridis')
plt.colorbar()
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.title('house price on basis of geo-coordinates')
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))

sns.stripplot(data=data,x='ocean_proximity',y='median_house_value',jitter=0.3)


# ## ONE HOT ENCODING 

# In[ ]:


data = pd.get_dummies(data)


# In[ ]:


data.head()


# In[ ]:


y=data.pop('median_house_value')


# In[ ]:


data.isnull().sum()


# # Train Test Split : for checking RMSE 

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.33, random_state=42)


# In[ ]:


X_train.head()


# ## Predictive Modeling (Using API)

# ### importing all of Algoriths that we want to use
# #### Linear_model: 
# <font color=blue> Lasso, Ridge 
#     
# #### Neighbors:
# <font color=red> KNeighborsRegressor
# <font color=black>
#     
#     
# #### Tree:
# <font color=blue> DecisionTreeRegressor
# <font color=black>
#     
# #### Neural network:
# <font color=red> MLPRegressor 
# <font color=black>
#     
# #### Ensemble:
# <font color=black> GradientBoostingRegressor

# name=['Linear Regression','Linear Regression CV','Ridge Regression','Ridge Regression CV','Lasso Regression',
#      'Lasso Regression CV','Elastic Net Regression','Elastic Net Regression CV','SGD Regression','SGD Regression CV',
#      'SVM','SVM CV','Decision Tree','Decision Tree Regression','Random Forest','Random Forest CV','Ada Boost','Ada Boost CV',
#      'Bagging','Bagging CV','Gradient Boost','Gradient Boost CV']

# In[ ]:


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np


# In[ ]:


from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor   
from sklearn.ensemble import  GradientBoostingRegressor


# ## LOOP FOR FITTING AND PREDICTING USING ALL MODEL (API) 

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


model= [LinearRegression(), DecisionTreeRegressor() ,   Lasso(), Ridge(),  MLPRegressor(), GradientBoostingRegressor()  ]
name = ['LinearRegression','DecisionTreeRegressor','Lasso','Ridge','MLPRegressor','GradientBoostingRegressor']
SCORE= []
TESTING=[]
RSME=[]
for ku in model:
    #ku will be replaced with each model like as first one is LogisticRegression()
    algorithm = ku.fit(X_train,y_train)
    print(ku)
    #now 'algorithm' will be fitted by API with above line and next line will check score with data training and testing
    predict_ku=ku.predict(X_test)
    print('RSME: {:.4f}'.format(np.sqrt(mean_squared_error(y_test,predict_ku))))
    score=cross_val_score(ku,X_train,y_train,cv=10,scoring='neg_mean_squared_error')
    ku_score_cross=np.sqrt(-score)
    
    print('mean: {:.2f} and std:{:.2f}'.format(np.mean(ku_score_cross),np.std(ku_score_cross)))
    print('---'*10)
    print('training set accuracy: {:.2f}'.format(algorithm.score(X_train,y_train)))
    print('test set accuracy: {:.2f}'.format(algorithm.score(X_test,y_test)))
    print('---'*30)
    #Now we are making a dataframe where by each loop the dataframe is added by SCORE,TESTING
    RSME.append(np.sqrt(mean_squared_error(y_test,predict_ku)))
    SCORE.append(algorithm.score(X_train,y_train))
    TESTING.append(algorithm.score(X_test,y_test))
models_dataframe=pd.DataFrame({'training score':SCORE,'testing score':TESTING,'RSME':RSME},index=name)


# #### MODEL COMPARISON
# SEEING IN ASCENDING ORDER FOR EASIER COMPARISON

# In[ ]:


models_dataframe


# In[ ]:


asendingtraining = models_dataframe.sort_values(by='RSME', ascending=False)
asendingtraining 


# In[ ]:


asendingtraining['RSME'].plot.barh(width=0.8)
plt.title('RSME')
fig=plt.gcf()
fig.set_size_inches(8,8)
plt.show()


# #### Any model will not be able to predict random noise in the data, so the predictive capability of the model can be no better than that noise. If it is better, then you are overfitting the noise. This is a bad thing to do

# we can remove outlier for changing rmse and change algorithms parameters
