#!/usr/bin/env python
# coding: utf-8

# <h1 align="center">  AIRBNB </b> </h1><br>
# ![](https://www.insuremyhouse.ie/wp-content/uploads/2019/09/11.png)

# # Breakdown of this notebook:
# 1. **Importing Libraries**
# 2. **Loading the dataset**
# 3. **Data Cleaning:** 
#  - Deleting redundant columns.
#  - Dropping duplicates.
#  - Cleaning individual columns.
#  - Remove the NaN values from the dataset
#  - Some Transformations
# 4. **Data Visualization:** Using plots to find relations between the features.
#     - Get Correlation between different variables
#     - Plot all Neighbourhood Group
#     - Neighbourhood
#     - Room Type
#     - Relation between neighbourgroup and Availability of Room
#     - Map of Neighbourhood group
#     - Map of Neighbourhood
#     - Availabity of rooom
# 5. **Word Cloud**
# 6. **Regression Analysis**
#  - Linear Regression
#  - Decision Tree Regression
#  - Random Forest Regression
# 

# ## Importing Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ## Loading Dataset

# In[ ]:


airbnb=pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')


# #### Print the Shape of the dataset

# In[ ]:


airbnb.shape


# #### Print the Datatypes of the dataset

# In[ ]:


airbnb.dtypes


# In[ ]:


airbnb.info()


# #### Removing the Duplicates if any

# In[ ]:


airbnb.duplicated().sum()
airbnb.drop_duplicates(inplace=True)


# #### Check for the null values in each column

# In[ ]:


airbnb.isnull().sum()


# #### Drop unnecessary columns

# In[ ]:


airbnb.drop(['name','id','host_name','last_review'], axis=1, inplace=True)


# ### Examining Changes

# In[ ]:


airbnb.head(5)


# #### Rreplace the 'reviews per month' by zero

# In[ ]:


airbnb.fillna({'reviews_per_month':0}, inplace=True)
#examing changes
airbnb.reviews_per_month.isnull().sum()


# #### Remove the NaN values from the dataset

# In[ ]:


airbnb.isnull().sum()
airbnb.dropna(how='any',inplace=True)
airbnb.info() #.info() function is used to get a concise summary of the dataframe


# ### Examine Continous Variables

# In[ ]:


airbnb.describe()


# ### Print all the columns names

# In[ ]:


airbnb.columns


# ### Get Correlation between different variables

# In[ ]:


corr = airbnb.corr(method='kendall')
plt.figure(figsize=(15,8))
sns.heatmap(corr, annot=True)
airbnb.columns


# In[ ]:


airbnb.shape


# In[ ]:


airbnb.head(15)


# ## Data Visualization

# In[ ]:


import seaborn as sns


# In[ ]:


airbnb['neighbourhood_group'].unique()


# ### Plot all Neighbourhood Group

# In[ ]:


sns.countplot(airbnb['neighbourhood_group'], palette="plasma")
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.title('Neighbourhood Group')


# ### Neighbourhood

# In[ ]:


sns.countplot(airbnb['neighbourhood'], palette="plasma")
fig = plt.gcf()
fig.set_size_inches(25,6)
plt.title('Neighbourhood')


# ### Room Type

# In[ ]:


#Restaurants delivering Online or not
sns.countplot(airbnb['room_type'], palette="plasma")
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.title('Restaurants delivering online or Not')


# ### Relation between neighbourgroup and Availability of Room

# In[ ]:


plt.figure(figsize=(10,10))
ax = sns.boxplot(data=airbnb, x='neighbourhood_group',y='availability_365',palette='plasma')


# ## Map of Neighbourhood group

# In[ ]:


plt.figure(figsize=(10,6))
sns.scatterplot(airbnb.longitude,airbnb.latitude,hue=airbnb.neighbourhood_group)
plt.ioff()


# ## Map of Neighbourhood

# plt.figure(figsize=(10,6))
# sns.scatterplot(airbnb.longitude,airbnb.latitude,hue=airbnb.neighbourhood)
# plt.ioff()

# In[ ]:


plt.figure(figsize=(10,6))
sns.scatterplot(airbnb.longitude,airbnb.latitude,hue=airbnb.neighbourhood)
plt.ioff()


# In[ ]:


plt.figure(figsize=(10,6))
sns.scatterplot(airbnb.longitude,airbnb.latitude,hue=airbnb.room_type)
plt.ioff()


# ## Availability of Room

# In[ ]:


plt.figure(figsize=(10,6))
sns.scatterplot(airbnb.longitude,airbnb.latitude,hue=airbnb.availability_365)
plt.ioff()


# ## WordCloud

# In[ ]:


from wordcloud import WordCloud


# In[ ]:


plt.subplots(figsize=(25,15))
wordcloud = WordCloud(
                          background_color='white',
                          width=1920,
                          height=1080
                         ).generate(" ".join(airbnb.neighbourhood))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('neighbourhood.png')
plt.show()


# ## Regression Analysis

# ### Drop Columns

# In[ ]:


airbnb.drop(['host_id','latitude','longitude','neighbourhood','number_of_reviews','reviews_per_month'], axis=1, inplace=True)
#examing the changes
airbnb.head(5)


# In[ ]:


#Encode the input Variables
def Encode(airbnb):
    for column in airbnb.columns[airbnb.columns.isin(['neighbourhood_group', 'room_type'])]:
        airbnb[column] = airbnb[column].factorize()[0]
    return airbnb

airbnb_en = Encode(airbnb.copy())


# In[ ]:


airbnb_en.head(15)


# In[ ]:


#Get Correlation between different variables
corr = airbnb_en.corr(method='kendall')
plt.figure(figsize=(18,12))
sns.heatmap(corr, annot=True)
airbnb_en.columns


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score


# In[ ]:


#Defining the independent variables and dependent variables
x = airbnb_en.iloc[:,[0,1,3,4,5]]
y = airbnb_en['price']
#Getting Test and Training Set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=353)
x_train.head()
y_train.head()


# In[ ]:


x_train.shape


# In[ ]:


#Prepare a Linear Regression Model
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[ ]:


#Prepairng a Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=105)
DTree=DecisionTreeRegressor(min_samples_leaf=.0001)
DTree.fit(x_train,y_train)
y_predict=DTree.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_predict)


# In[ ]:


#Prepairng a Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=105)
DTree=DecisionTreeRegressor(min_samples_leaf=.0001)
DTree.fit(x_train,y_train)
y_predict=DTree.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_predict)


# ## For further updates of this Kernel check into this GitHub Link: https://github.com/chiragsamal/airbnb

# #### Refernces
#  - https://www.kaggle.com/dgomonov/data-exploration-on-nyc-airbnb
#  - https://www.kaggle.com/biphili/hospitality-in-era-of-airbnb
#  - https://www.kaggle.com/geowiz34/maps-of-nyc-airbnbs-with-python

# >  # <font color='orange'> Please UPVOTE if you found these helpful :) </font>
