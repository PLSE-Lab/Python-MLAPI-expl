#!/usr/bin/env python
# coding: utf-8

# # New York City Airbnb Open Data 

# In[ ]:


from IPython.display import Image

Image(url='https://img4.cityrealty.com/neo/i/p/mig/airbnb_guide.jpg')


# ## Abstract

# Since 2008, guests and hosts have used Airbnb to expand on traveling possibilities and present a more unique, personalized way of experiencing the world. Today, Airbnb became one of a kind service that is used and recognized by the whole world. Data analysis on millions of listings provided through Airbnb is a crucial factor for the company. These millions of listings generate a lot of data - data that can be analyzed and used for security, business decisions, understanding of customers' and providers' (hosts) behavior and performance on the platform, guiding marketing initiatives, implementation of innovative additional services and much more.

# ## Data Source

# This dataset has around 49,000 observations in it with 16 columns and it is a mix between categorical and numeric values.

# ### Importing libraries and loading data

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


ny=pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
ny


# We can observe from the dataset,there are 488895 rows and 16 columns.

# In[ ]:


ny.head()


# In[ ]:


ny.info()


# After loading the dataset in and from the head of AB_2019_NYC dataset we can see a number of things. These 16 columns provide a very rich amount of information for deep data exploration we can do on this dataset. We do already see some missing values, which will require cleaning and handling of NaN values.

# In[ ]:


num_cols=ny.select_dtypes(exclude="object").columns
num_cols


# In[ ]:


cat_cols=ny.select_dtypes(include="object").columns
cat_cols


# In[ ]:


ny.dtypes


# There are 10 columns which has numerical values and 6 columns has categorical values.We also check the types of the columns.

# In[ ]:


ny.isnull().sum()


# In our case, missing data that is observed does not have that much of importance regarding our dataset. Looking into the nature of our dataset we can state further things: columns "name" and "host_name" are irrelevant and insignificant to our data analysis, columns "last_review" and "review_per_month" need very simple handling. To elaborate, "last_review" is date; if there were no reviews for the listing - date simply will not exist. In our case, this column is irrelevant and insignificant therefore appending those values is not needed. For "review_per_month" column we can simply append it with 0.0 for missing values; we can see that in "number_of_review" that column will have a 0, therefore following this logic with 0 total reviews there will be 0.0 rate of reviews per month. Therefore, let's proceed with removing columns that are not important and handling of missing data.

# In[ ]:


ny=ny.drop(["id","host_name","last_review"],axis=1)
ny.head()


# In[ ]:


ny=ny.fillna({"reviews_per_month":0})


# In[ ]:


ny.isnull().sum()


# In[ ]:


ny.duplicated().sum()


# ## Exploratory Data Analysis 

# In[ ]:


ny.neighbourhood_group.value_counts().plot(kind="bar")
plt.title("Share of neighbourhood")
plt.xlabel("neighbourhood_group")
plt.ylabel("Count")


# Manhatten and Brooklyn have the highest share of hotels

# In[ ]:


ny['neighbourhood_group'].value_counts().plot.pie(explode=[0,0.1,0,0,0],autopct='%1.1f%%',shadow=True)


# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(x = 'room_type',hue = "neighbourhood_group",data = ny)
plt.title("Room types occupied by the neighbourhood_group")
plt.show()


# We have got the Neighborhood wise share of rooms

# In[ ]:


sns.barplot(data=ny,x="neighbourhood_group",y="price")


# From the above plot we can observe that the prices are very high in Manhattan.

# In[ ]:


ny["price"].describe()


# In[ ]:


sns.heatmap(ny.corr(),annot=True,cmap="coolwarm")


# From above heatmap we can find that there is no strong corelation except number_of_reviews vs reviews_per_month.

# In[ ]:


f,ax = plt.subplots(figsize=(16,8))
ax = sns.scatterplot(y=ny.latitude,x=ny.longitude,hue=ny.neighbourhood_group,palette="coolwarm")
plt.show()


# Longitude vs Latitude (representing different neighbourhood groups)

# In[ ]:


plt.figure(figsize=(10,6))
ny1=ny[ny.price<500]

ny1.plot(kind='scatter', x='longitude',y='latitude',label='availability_365',c='price',cmap=plt.get_cmap('jet'),colorbar=True,alpha=0.4,figsize=(10,10))

plt.show()


# Comapring prices of different airbnb flats and rooms with different latitudes.Red color dots are the apartment or rooms with higher price.I have considered prices upto 300 $ to get a goo representation on the plot.We can see that Manhattan region has more expensive apartments.

# In[ ]:


sns.stripplot(data=ny,x='room_type',y='price',jitter=True)

plt.show()


# We can see that different room type has different price range.

# In[ ]:


plt.figure(figsize=(10,6))
ny['number_of_reviews'].plot(kind='hist')
plt.xlabel("Price")

plt.show()


# We can see that low cost rooms or in range 0-50 $ have more reviews

# In[ ]:


f,ax = plt.subplots(figsize=(25,5))
ax=sns.stripplot(data=ny,x='minimum_nights',y='price',jitter=True)

plt.show()


# We can see that rooms with low minimum nights have high price.As mentioned before Home/Apt have the highest price

# ### Displaying rooms with maximum reviews

# In[ ]:


ny1=ny.sort_values(by=['number_of_reviews'],ascending=False).head(100)
ny1.head()


# ## Comparision between different models
# ##### Assumptions :Data is following linear regression
# 

# In[ ]:



import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn import preprocessing
from sklearn.feature_selection import RFE

import warnings 
warnings.filterwarnings('ignore')


# In[ ]:


ny["name"] = pd.get_dummies(ny['name'])
ny["neighbourhood_group"]= pd.get_dummies(ny['neighbourhood_group'])
ny["neighbourhood"]= pd.get_dummies(ny['neighbourhood'])
ny["room_type"]= pd.get_dummies(ny['room_type'])


# In[ ]:


ny


# ### Model 1

# In[ ]:


X=ny.drop('price',axis=1)
y=ny['price']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[ ]:


lr=LinearRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
r2_score(y_test,y_pred)


# We are using Superwise linear regression model(SLR) and the accuracy we got is 12.47%.Our model is predicting that how price is varying in newyork on different features.

# ### Model 2

# In[ ]:


rfe = RFE(lr, 7)
#Transforming data using RFE
X_rfe = rfe.fit_transform(X,y)  
#Fitting the data to model
lr.fit(X_rfe,y)
print(rfe.support_)
print(rfe.ranking_)


# In[ ]:


#no of features
nof_list=np.arange(1,10)            
high_score=0
#Variable to store the optimum features
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    lr = LinearRegression()
    rfe = RFE(lr,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    lr.fit(X_train_rfe,y_train)
    score = lr.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))


# We are using recurssive feature elimination model and accuracy we got is 10.39% when number of features are 9. 

# ### Model 3

# In[ ]:


xc=sm.add_constant(X)
lm=sm.OLS(y,xc).fit()
lm.summary()


# In[ ]:


X=X.drop("minimum_nights",axis=1)


# In[ ]:


xc=sm.add_constant(X)
lm=sm.OLS(y,xc).fit()
lm.summary()


# We are using OLS model and we have dropped three features name,neighbourhood,minimum nights and the accuracy we got is 9.1%

# **So we are using SLR model because it gives the highest accuracy 12.47%.** 

# In[ ]:


plt.figure(figsize=(16,8))
sns.regplot(y_test,y_pred)
plt.xlabel('Actual')
plt.ylabel('Prediction')
plt.title("SLR Model")
plt.grid(False)
plt.show()


# There is high concentration of prediction value between 0 to 1000.
# >The accuracy of the model is low because the data is not equally spread along the linear regression line.
# 

# In[ ]:




