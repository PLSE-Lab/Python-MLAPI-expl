#!/usr/bin/env python
# coding: utf-8

# #### Let's get started!

# # KC_Housesales_Data
# - Predict House sale prices using Multi-Linear Regression
# - Predict the sales of houses in King County with an accuracy of at least 75-80% and 
# - understand which factors are responsible for higher property value  "$650K and above."
# 

# ### Import Libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings
warnings.filterwarnings("ignore")

#basic eda packages
from matplotlib import pyplot as plt
import seaborn as sns
from collections import Counter
from matplotlib import rcParams
rcParams['figure.figsize'] = 7, 5
pd.options.display.max_columns = 25
sns.set(style='darkgrid')

#validation
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

#models 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('/kaggle/input/kc-housesales-data/kc_house_data.csv')


# In[ ]:


df.head()


# > # Basic eda checks

# In[ ]:


print("Shape of the data :",df.shape)


# In[ ]:


df.info()


# In[ ]:


df.describe()


# # Data Cleaning

# In[ ]:


pd.DataFrame(df.isna().sum()).T


# Here their are no null values found!!

# **correlation** w.r.t price 

# In[ ]:


pd.DataFrame(df.corr()['price']).sort_values(by='price',ascending=False)


# # lets plot the data

# In[ ]:


print(Counter(df.bedrooms))
sns.countplot(df.bedrooms,order=df.bedrooms.value_counts().index);
plt.title("No of Bedrooms count");


# In[ ]:


plt.xticks(rotation=90)
sns.countplot(df.bathrooms,order=df.bathrooms.value_counts().index);
plt.title('No of bathroom Counts');


# In[ ]:


print(Counter(df.floors))
sns.countplot(df.floors,order=df.floors.value_counts().index);
plt.title("Number of Floors");


# Lets have handson on pie chart as well!!

# 

# In[ ]:


print(Counter(df.waterfront))
plt.pie(df.waterfront.value_counts(),explode=[0,0.5],        autopct="%01.1f",labels=df.waterfront.unique(),shadow=True,startangle=10,colors=['gold','red'],        textprops={'fontsize': 14});
plt.axis("equal");
plt.title('Waterfront in Percentage');


# In[ ]:


print(Counter(df.view))
sns.countplot(df.view);


# In[ ]:


sns.barplot(list(Counter(df.grade).keys()),list(Counter(df.grade).values()));
plt.xlabel("Grade");
plt.ylabel("Count of Grades");
plt.title("Types of Grades");


# ### Feature engineering 
# - and yea it's a vast topic but i just went with basics here 

# lets drop the non contribution variables here

# In[ ]:


unwanted = ['id','date']
df.drop(unwanted,axis=1,inplace=True)


# In[ ]:


df.head() #id and data variable coll is droped


# Here we'll calculate the number of year's between present year and mentioned year in the data

# In[ ]:


df['built_age'] = 2020 - df.yr_built 
df.drop('yr_built',axis=1,inplace=True)


# In[ ]:


df.head(1)


# ## Split the data

# In[ ]:


X = list(df.iloc[:,1:].values) # independent variables
y = df.price.values # dependent variable


# # Scaling the data

# In[ ]:


sn = StandardScaler()
X = sn.fit_transform(X)
X


# - We'll check the distrubution of the dependent variable 

# In[ ]:


sns.distplot(y);
plt.xticks(rotation=90);
plt.title("Before normalizing the dependent variable");


# - Here we can see that data is "Highly positively skewed"
# - Lets normalize the y variable (dependent variable)

# In[ ]:


y = np.log10(y)
#we just normalized the y variable using log10 which is available in numpy package 
#now lets plot the data

sns.distplot(y);
plt.xticks(rotation=90);
plt.title("After normalizing the dependent variable");
print("This is also called as normal gaussian distribution")


# # Preprocessing
# - We start machine learning by setting the features and target:
# 
# - Features: x
# - Target: y

# Then, we split them to train and test sets

# In[ ]:


X_train ,X_test , y_train ,y_test = train_test_split(X,y,test_size=0.2,random_state=10)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape) #printing the shape of splited data


# # Machine learning
# > here i have taken 2 regression algorithms i.e
# - LinearRegression
# - RandomForestRegressor
# 
# 
# - Here we'er calculating both train and test data so that we can check if the data is overfitted , underfitted of optimal in state

# # 1. LinearRegression

# In[ ]:


model_line = LinearRegression(normalize=True,fit_intercept=True,n_jobs=1)
model_line.fit(X_train,y_train)

y_train_pred = model_line.predict(X_train)
y_pred = model_line.predict(X_test)


print("Train score:",r2_score(y_train,y_train_pred))
print("Test score:",r2_score(y_test,y_pred))


# # 2.RandomForestRegressor

# In[ ]:


model = RandomForestRegressor(n_estimators=190,max_depth=100,random_state=25,max_features='auto',n_jobs=1)
model.fit(X_train,y_train)

y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)


print("Train score:",r2_score(y_train,y_train_pred))
print("Test score:",r2_score(y_test,y_pred))


# ## If you made it this far, thank you for your attention.
# ### In order to improve my score, I will keep updating this kernel by:
# 
# - Creating new features
# - normalizing the data 
# - I will bring more updates, stay tuned!

# #### if you find this kernal helpfull please vote :)
