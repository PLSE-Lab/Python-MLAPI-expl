#!/usr/bin/env python
# coding: utf-8

# I am newbie at machine learning.This is my first kernel on EDA. There are some personal insights that I have provided after working on the data.This is a basic approach (So any first timer who is yet to write his first kernel can take a look and clear his/her doubts) So if anybody finds the points contradicting feel free to comment. 
# 
# 
# The Basic sequence of steps that I followed:
# 1. Finding and handling missing data.
# 2. Handling outliers.
# 3. Visualising the dataset for EDA.
# 4. A baseline prediction model.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # 1. Import the dataset.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
#ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[ ]:



df=pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
#Lets take a look at some of the entries.
df.head()


# Lets get info about the different features present and their properties.

# In[ ]:


df.describe(), df.info()


# # Handling Missing Data

# In[ ]:


df.isnull().sum()


# I have dropped the *last_review* column as it contains dates and 20% of the values are missing.
# I have also filled the NaN values of *reviews_per_month* with 0, which will also not affect the  *number_of_reviews* column.

# In[ ]:


#Removing the columns which don't have affect on price.
df.drop(['id','host_name','last_review'], axis = 1,inplace=True) 

df.reviews_per_month.fillna(value=0,inplace=True)


# In[ ]:


plt.figure(figsize=(16, 6))
sns.barplot(df.neighbourhood_group,df.price,hue=df.room_type,ci=None)


# The above bar plot concludes:
# 1. Manhattan is the most expensive neighbourhood_group
# 2. The price of entire home/apt is more than any other room type.
# 3. Bronx is the cheapest.

# In[ ]:


plt.figure(figsize=(16, 6))
sns.countplot(df.neighbourhood_group,hue=df.room_type)


# The above count plot concludes:
# 1. Staten Island and Bronx have the least number of entries in the listings.
# 2. Shared rooms are less available in the listings.
# 3. Manhattan and Brooklyn neighbourhoods have far more entries in the listings.
# 
# I will infer from this plot that Brooklyn and Manhattan are towns(IT or Industrial areas) where people come for jobs thats why the renting properties are more there. 
# 

# In[ ]:


df.drop('price', axis=1).corrwith(df.price).plot.barh(figsize=(10, 8), 
                                                        title='Correlation with Response Variable',
                                                        fontsize=15)


# The above plot describes the pearson coefficients of the features(numerical) with price variable.Not much to infer as the values for the coefficient is not upto mark(>0.5).

# # Handling Outliers
# 
# Outliers are the absurd values that can occur in the data due to errors during data collection or sometime can also be indicators of interesting trends. Have a look at [this](http://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba) post to know more about outliers and how to detect and remove them.

# In[ ]:


plt.figure(figsize = (15, 10))
plt.style.use('seaborn-white')
ax=plt.subplot(221)
plt.boxplot(df['number_of_reviews'])
ax.set_title('Numer of Reviews')
ax=plt.subplot(222)
plt.boxplot(df['price'])
ax.set_title('Price')
ax=plt.subplot(223)
plt.boxplot(df['availability_365'])
ax.set_title('availability_365')
ax=plt.subplot(224)
plt.boxplot(df['reviews_per_month'])
ax.set_title('reviews_per_month')


# The above plots conclude:
# 1. All of the above except *availability_365* have outliers.

# In[ ]:


Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1    #IQR is interquartile range. 

oultier_remover = (df['price'] >= Q1 - 1.5 * IQR) & (df['price'] <= Q3 + 1.5 *IQR)
df=df.loc[oultier_remover]

Q1 = df['number_of_reviews'].quantile(0.25)
Q3 = df['number_of_reviews'].quantile(0.75)
IQR = Q3 - Q1    #IQR is interquartile range. 

oultier_remover = (df['number_of_reviews'] >= Q1 - 1.5 * IQR) & (df['number_of_reviews'] <= Q3 + 1.5 *IQR)
airbnb2=df.loc[oultier_remover]


Q1 = df['reviews_per_month'].quantile(0.25)
Q3 = df['reviews_per_month'].quantile(0.75)
IQR = Q3 - Q1    #IQR is interquartile range. 

oultier_remover = (df['reviews_per_month'] >= Q1 - 1.5 * IQR) & (df['reviews_per_month'] <= Q3 + 1.5 *IQR)
airbnb_new=df.loc[oultier_remover]


# In[ ]:


#extract the host_ids having high number of entries in the dataset.
host_with_most_listings=df.host_id.value_counts().head(13)

#extract the most popular neighbourhoods.
most_popular_neighbourhoods=df.neighbourhood.value_counts().head(13)


# In[ ]:


most_popular_neighbourhoods,host_with_most_listings


# In[ ]:


plt.figure(figsize=(16, 6))
host_with_most_listings.plot(kind='bar')


# This plot concludes:
# 1. Some hosts have more number of entries w.r.t to others and the host_id with most number of entries has 327 entries. 

# In[ ]:


plt.figure(figsize=(16, 6))
most_popular_neighbourhoods.plot(kind='bar')


# This plot concludes that:
# 1. Williamsburg  and Bedford-Stuyvesant are the most popular neighbourhood of all.

# In[ ]:


most_popular_neighbourhoods_df=df.loc[df.neighbourhood.isin(['Williamsburg','Bedford-Stuyvesant',   
'Harlem',                
'Bushwick',              
'Upper West Side',       
'Hell\'s Kitchen',        
'East Village',      
'Upper East Side',       
'Crown Heights',     
'Midtown',               
'East Harlem',           
'Greenpoint',            
'Chelsea' ])]


# In[ ]:


# host_with_most_listings_df=df.loc[df.host_id.isin([ '219517861',    
#  '107434423',    
#  '30283594',     
#  '137358866',    
#  '12243051',      
#  '16098958',      
#  '61391963',      
#  '22541573',      
#  '200380610',     
#  '7503643',       
# '1475015',       
#  '120762452',     
#  '2856748' ])]


# In[ ]:


plt.figure(figsize=(20, 6))
sns.catplot(x='neighbourhood', hue='neighbourhood_group', col='room_type', data=most_popular_neighbourhoods_df, kind='count').set_xticklabels(rotation=90)


# These plots concludes:
# 1. Confirms the above inference that Manhattan and Brooklyn are most popular.
# 2. It alse tells which neighbourhood falls in which neighbourhood_group(Can be helpful for someone coming first time to the country :) )

# In[ ]:


plt.figure(figsize=(15, 6))
sns.scatterplot(x=df.neighbourhood_group,y=df.number_of_reviews,ci=False)


# The above plot concludes that:
# 1. If number_of_reviews is taken as the number of people who have stayed and provided there reviews then Queens , Manhattan and Brooklyn are the most preferred places.
# 
# 

# In[ ]:


plt.figure(figsize=(15, 6))
sns.barplot(x=df.neighbourhood_group,y=df.calculated_host_listings_count,ci=False)


# The above plot tells that:
# 1. Manhattan has the most entries in data. 

# The below idea is taken from this [kernel](https://www.kaggle.com/dgomonov/data-exploration-on-nyc-airbnb)

# In[ ]:


#Now lets take the name column and get some insights.
keywords=[]
#the basic motto is to get the individual tokens out of the dataframe column
for name in df.name:
    keywords.append(name)
def split_keywords(name):
    spl=str(name).split()
    return spl
keywords_filtered=[]
for x in keywords:
    for word in split_keywords(x):
        word=word.lower()
        keywords_filtered.append(word)
#These are some of the words that I have removed after working on data.
keywords_filtered=[word for word in keywords_filtered if not word in ['in','of','the','to','1','2','3','and','with','&']]


# In[ ]:


from collections import Counter
#Get the list of most frequent words.
freq_keywords=Counter(keywords_filtered).most_common()


# In[ ]:


freq_keywords_df=pd.DataFrame(freq_keywords)
freq_keywords_df.rename(columns={0:'Words', 1:'Count'}, inplace=True)


# In[ ]:


plt.figure(figsize=(15, 6))
#plotting the top ten
sns.barplot(x='Words',y='Count',data=freq_keywords_df[0:10])


# The above plot  infers taht:
# 1. No fancy words are used.
# 2. Words like 'room' , 'bedroom' ,'private' are most frequent.
# Personal Opinion:
# As the description are simple and not contain much of the adjectives so, it shows that the number of tenants are much more than the number of hosts (i.e they dont have to exaggerate)  

# In[ ]:


plt.figure(figsize=(15, 6))
sns.barplot(x=df.neighbourhood_group,y=df.availability_365,hue=df.room_type,ci=False)


# The above plot concludes:
# 1. Staten Island properties remain unoccupied as they are available most of the days.
# 2. Manhattan properties are most occupied.

# In[ ]:


plt.figure(figsize=(15, 6))
sns.scatterplot(x=df.longitude,y=df.latitude,hue=df.neighbourhood_group)


# The above plot just decibes the demogrphic view of the entries in the data and also provides a clear view of the neighbourhood_groups.

# In[ ]:


plt.figure(figsize=(15, 6))
sns.scatterplot(x=df.longitude,y=df.latitude,hue=df.room_type)


# The above plot shows that :
# 1. The entries for private and Entire home are much more than shared rooms.

# # Baseline Model.
# 
# 1. Preprocessing data

# In[ ]:


df.info()


# In the dataset, there are 5 categorical and 9 numerical features

# In[ ]:


#Applying one hot encoding for categorical variables using get_dummies.
dummy_neighbourhood=pd.get_dummies(df['neighbourhood_group'], prefix='dummy')
dummy_roomtype=pd.get_dummies(df['room_type'], prefix='dummy')
df_new = pd.concat([df,dummy_neighbourhood,dummy_roomtype],axis=1)
#Removing the columns which are not helpful in predicting new prices.
df_new.drop(['neighbourhood_group','room_type','neighbourhood','name','longitude','latitude','host_id'],axis=1, inplace=True)
df_new


# In[ ]:


#Seperating the predictor and target variables.
y=df_new['price']
X=df_new.drop(['price'],axis=1)


# In[ ]:


from sklearn import preprocessing
#Only standardize the numerical colums and not the dummy variables.
X_scaled=preprocessing.scale(X.iloc[:,0:5])
X_scaled = pd.DataFrame(X_scaled, index=X.iloc[:,0:5].index, columns=X.iloc[:,0:5].columns)
X.drop(X.iloc[:,0:5],axis=1,inplace=True)
X=pd.concat([X_scaled,X],axis=1)


# # Cross Validation

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 50, random_state = 0) 
# As there is only option to calculate negative MAE.So lets make it positive.  
scores =  -1 * cross_val_score(regressor, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)
print("Average MAE score (across experiments):",scores.mean())


# The score is coming out to be 38.37. I have tried many things but wasn't able to get any lower.Feel free to comment if you find anything new and interesting.

# In[ ]:


#split the dataset.
from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y=train_test_split(X,y,random_state=1)


# In[ ]:


#Result using Random Forest Regressor
from sklearn.metrics import mean_absolute_error
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(train_X, train_y)
preds = model.predict(test_X)
mean_absolute_error(test_y, preds)


# In[ ]:


#Result using XGBoost.
from xgboost import XGBRegressor

model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
model.fit(train_X, train_y, 
             early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], 
             verbose=False)
predictions = model.predict(test_X)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, test_y)))


# So, the score decreased by a bit to 34.69.
# I am still working on it to reduce more and will update the kernel if anything interesting comes up.

# Kernels that I have referred from:
# 1. Belong Anywhere-NY Airbnb price prediction [here](https://www.kaggle.com/benroshan/belong-anywhere-ny-airbnb-price-prediction/notebook)
# 2. Data Exploration on NYC Airbnb [here](https://www.kaggle.com/dgomonov/data-exploration-on-nyc-airbnb)
# 
