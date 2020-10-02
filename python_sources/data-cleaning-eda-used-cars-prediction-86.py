#!/usr/bin/env python
# coding: utf-8

# # Description

# Craigslist is the world's largest collection of used vehicles for sale, yet it's very difficult to collect all of them in the same place. I built a scraper for a school project and expanded upon it later to create this dataset which includes every used vehicle entry within the United States on Craigslist.

# # Importing Libraries

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


# # Reading Data

# In[ ]:


df=pd.read_csv("/kaggle/input/craigslist-carstrucks-data/vehicles.csv")


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set()


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.info()


# # Data cleaning 

# In[ ]:


r=df.columns
for i in r:
    print(df[i].value_counts())


# Based on above observation the features which are too common or of no use like url can be dropped

# In[ ]:


df= df.drop(columns=['id','url', 'region_url', 'vin', 'image_url', 'description', 'lat', 'long','county','region'], axis=1)


# In[ ]:


df.head()


# In[ ]:


df.describe()


# # Handling outliers

# Lets see if any outliers in Target variable as we remove them the model can be more accurate because they cause a bring a lot of difference in values of mean and SD.

# the difference between 75% value and max value is too large so lets leave 10% values at both ends of a distribution

# In[ ]:


rr=sorted(df["price"])


# In[ ]:


quantile1, quantile3= np.percentile(rr,[10,90])


# In[ ]:


print(quantile1,quantile3)


# In[ ]:


df=df[(df.price < 27500) & (df.price >= 500 )]
df.shape


# Lets observe the odometer column

# In[ ]:


r=sorted(df["odometer"])
r


# There are nan values and only one 0 value

# In[ ]:


df["odometer"].isna().sum()


# In[ ]:


ax = sns.scatterplot(x="odometer", y="price", data=df)


# In[ ]:


df["odometer"].max()


# In[ ]:


df.drop(df[df["odometer"]==64809218.0].index,inplace=True)


# In[ ]:


df.drop(df[df["odometer"]==0.0].index,inplace=True)


# In[ ]:


ax = sns.scatterplot(x="odometer", y="price", data=df)
ax.get_xaxis().get_major_formatter().set_scientific(False)
ax.get_yaxis().get_major_formatter().set_scientific(False)


# In[ ]:


df["odometer"].isna().sum()


# here the values above 3000000  can be considered as outliers

# In[ ]:


df=df[(df.odometer < 3000000)]


# In[ ]:


ax = sns.scatterplot(x="odometer", y="price", data=df)
ax.get_xaxis().get_major_formatter().set_scientific(False)


# Now lets see year column

# In[ ]:


df["odometer"].isna().sum()


# In[ ]:


df["year"].isna().sum()


# the null values in a year column cannot be replaced so lets eliminate them

# In[ ]:


df["year"].min()


# we must alo drop this 0 values

# In[ ]:


df.drop(df[df["year"]==0.0].index,inplace=True)


# In[ ]:


df=df.dropna(subset=['year'])


# In[ ]:


bx = sns.scatterplot(x="year", y="price", data=df)


# In[ ]:


df=df[(df.year > 1940)]


# we have removed outliers from all three numerical columns .
# now lets how we can remove with nan values if any.

# # Handling Null Values

# I am taking out % of null values in each column

# In[ ]:


null_values=df.isnull().sum()
null_values=pd.DataFrame(null_values,columns=['null'])
j=1
sum_tot=len(df)
null_values['percent']=null_values['null']/sum_tot
round(null_values*100,3).sort_values('percent',ascending=False)


# In[ ]:


df.condition.value_counts()


# the misssing values in the condition can be found using odometer as mileage affects condition of car.

# what i am trying to do here is finding mean value of odometer readings grouping by there conditions

# In[ ]:


excellent_odo_mean = df[df['condition'] == 'excellent']['odometer'].mean()
good_odo_mean = df[df['condition'] == 'good']['odometer'].mean()
like_new_odo_mean = df[df['condition'] == 'like new']['odometer'].mean()
salvage_odo_mean = df[df['condition'] == 'salvage']['odometer'].mean()
fair_odo_mean = df[df['condition'] == 'fair']['odometer'].mean()


# In[ ]:


print('Like new average odometer:', round( like_new_odo_mean,2))
print('Excellent average odometer:', round( excellent_odo_mean,2))
print('Good average odometer:', round( good_odo_mean,2))
print('Fair average odometer:', round( fair_odo_mean,2))
print('Salvage average odometer:', round( salvage_odo_mean,2))


# these are mean values regarding each condition.
# 
# now these can be used to group the odometer readings which have nan values in condition.

# In[ ]:


df.loc[df.year>=2019, 'condition'] = df.loc[df.year>=2019, 'condition'].fillna('new')


# the values are being filled by the values from above calculated mean value ranges

# In[ ]:


df.loc[df['odometer'] <= like_new_odo_mean, 'condition'] = df.loc[df['odometer'] <= like_new_odo_mean, 'condition'].fillna('like new')

df.loc[df['odometer'] >= fair_odo_mean, 'condition'] = df.loc[df['odometer'] >= fair_odo_mean, 'condition'].fillna('fair')

df.loc[((df['odometer'] > good_odo_mean) & 
       (df['odometer'] <= excellent_odo_mean)), 'condition'] = df.loc[((df['odometer'] > good_odo_mean) & 
       (df['odometer'] <= excellent_odo_mean)), 'condition'].fillna('excellent')

df.loc[((df['odometer'] > like_new_odo_mean) & 
       (df['odometer'] <= good_odo_mean)), 'condition'] = df.loc[((df['odometer'] > like_new_odo_mean) & 
       (df['odometer'] <= good_odo_mean)), 'condition'].fillna('good')

df.loc[((df['odometer'] > good_odo_mean) & 
       (df['odometer'] <= fair_odo_mean)), 'condition'] = df.loc[((df['odometer'] > good_odo_mean) & 
       (df['odometer'] <= fair_odo_mean)), 'condition'].fillna('salvage')


# In[ ]:


null_values=df.isnull().sum()
null_values=pd.DataFrame(null_values,columns=['null'])
j=1
sum_tot=len(df)
null_values['percent']=null_values['null']/sum_tot
round(null_values*100,3).sort_values('percent',ascending=False)


# I am droping the null values with less then 5% nan.

# In[ ]:


df=df.dropna(subset=['title_status','fuel','transmission','model','manufacturer'])


# I am also droping the columns with more then 30% null values.
# but cylinders can be important feature .

# In[ ]:


df=df.drop(["size"],axis=1)


# In[ ]:


null_values=df.isnull().sum()
null_values=pd.DataFrame(null_values,columns=['null'])
j=1
sum_tot=len(df)
null_values['percent']=null_values['null']/sum_tot
round(null_values*100,3).sort_values('percent',ascending=False)


# In[ ]:


df['paint_color'] = df['paint_color'].fillna(method='ffill')
df['drive'] = df['drive'].fillna(method='ffill')


# In[ ]:


df['type'] = df['type'].fillna(method='ffill')
df['cylinders'] = df['cylinders'].fillna(method='ffill')


# In[ ]:


df.isnull().sum()


# In[ ]:


df=df.dropna(subset=['cylinders','drive'])


# In[ ]:


df.isnull().sum()


# # Exploratory Data Analysis

# In[ ]:


from scipy import stats


# In[ ]:


sns.pairplot(df)


# The above were relation between numerical values of the table.

# In[ ]:


df.info()


# In[ ]:


c=df.columns
for i in c:
    print(df[i].value_counts())


# In[ ]:


ax = sns.barplot(x="condition", y="price", data=df)


# Clearly vehicles with condition  new has highest price as one expects.

# In[ ]:


sns.catplot(y="cylinders", x="price",kind = "violin", data=df)


# This violinplot provides clear idea about the distribution of number of cylinders and the price.

# In[ ]:


sns.catplot(x="fuel", y="price", kind="boxen",
            data=df)


# This figure shows the price range between which majority of each type of car based on fuel lies.
# 
# Gas=5k-17k
# 
# diesel=12k-20k
# 
# hybrid=7k-15k
# 
# other=11k-20k
# 
# electric=10k-18k

# In[ ]:


sns.catplot(x="title_status", y="price",kind="violin", data=df)


# The distribution of price of cars based on title_status can be seen here.

# In[ ]:


sns.catplot(x="transmission", y="price",kind="bar", palette="ch:.25", data=df)


# The relation between price and transmission.

# In[ ]:


sns.violinplot(x=df.drive, y=df.price);


# There doesnt seem too be much difference between the first 2 types of drives .
# 
# The third one is a bit different.

# In[ ]:


sns.catplot(y="type", x="price",kind="boxen", data=df);


# Important observation can be obtained from the above figure regarding the price bracket for each type of vehicle.

# In[ ]:


sns.catplot(y="paint_color", x="price",kind="violin", data=df);


# Important observation can be obtained from the above figure regarding the distribution of price bracket for each color of vehicle.

# In[ ]:


sns.catplot(y="manufacturer", x="price",kind="box", data=df);


# It just gives and idea about prices based on the manufacturer.

# #  Label Processing
# 

# In[ ]:


from sklearn import preprocessing
import pandas as pd
le = preprocessing.LabelEncoder()


# In[ ]:


df.columns


# In[ ]:


df[['manufacturer', 'model', 'condition',
       'cylinders', 'fuel', 'title_status', 'transmission',
       'drive', 'type', 'paint_color', 'state']]=df[['manufacturer', 'model', 'condition',
       'cylinders', 'fuel', 'title_status', 'transmission',
       'drive', 'type', 'paint_color', 'state']].apply(le.fit_transform)


# # Split Train and Test data
# 

# In[ ]:


from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


# In[ ]:


y= df.price
X= df.drop('price',axis=1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)


# # Training Model

# # Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


# In[ ]:


from sklearn import metrics
from sklearn.metrics import mean_squared_error as MSE


# In[ ]:


print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test, y_pred),2))
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test, y_pred),2))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
print(regressor.score(X_test,y_test)*100)


# The accuracy is 86.02 .

# Please leave suggestions in the comments if any.
# 
# Do upvote if you find it usefull.

# In[ ]:




