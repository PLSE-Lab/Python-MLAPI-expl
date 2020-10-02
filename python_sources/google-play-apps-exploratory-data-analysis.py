#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns # used for plot interactive graph.
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import warnings
warnings.filterwarnings('ignore')
from pylab import rcParams
# figure size in inches
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-white')

# machine learning
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.cluster import KMeans
from sklearn import metrics #accuracy measure
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ![](https://images-na.ssl-images-amazon.com/images/I/31pQrwJSnwL.jpg) 
# # EDA Google play store data
# **Objective**
# 
# Notes on performing EDA
# 
# In this kernel,
# 
# *     Simple Exploratory Data Analysis
# *     Data preprocessing
# 
# # EDA - Asking the right questions
# 
# The objective is to find answers to the following questions
# 
# **Descriptive questions**
# 
# 1. What are the total number of applications in each categories?
# 2. What is the average rating of the applications in each categories?
# 3. What is the average price of applications in each categories?
# 4. What are the applications with, most number of reviews, highest rating, most revenue through installs, most number of installs?
# 5. How many percent of applications are free and how many are paid?
# 6. Which is the costliest application in the play store?
# 7. What is the distribution of applications in each category, Content Rating?
# 8. What is the average size of the applications?
# 9. What is the frequency of updates of applications?
# 10. How many applications are not updated after released?
# 
# **Exploratory**
# 
# 1. What characteristics impact the rating of an application?
# 2. How cost plays an important part in the reviews ?
# 
# **Predictive**
# 1. How can we predict the success of an application based on number of reviews?
# 
# 
# 
# **Please provide your feedback for improvements.**

# In[ ]:


df = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')
df_reviews = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore_user_reviews.csv')


# # Step 1 - Understanding the data

# In[ ]:


df_reviews.shape


# In[ ]:


df.shape


# In[ ]:


df_reviews.head()


# In[ ]:


df.head()


# In[ ]:


df_reviews.info()


# In[ ]:


df.info()


# # Step 2 - Cleaning the data

# In[ ]:


df.isna().sum()


# In[ ]:


df_reviews.isna().sum()


# In[ ]:


#calculating the RATING based on mean value
df['Rating'].fillna((df['Rating'].mean()), inplace=True)


# In[ ]:


df.isna().sum()


# In[ ]:


#droping the rest of na values
df1 = df.dropna()


# In[ ]:


df1.isna().sum()


# In[ ]:


#final data shape
df1.shape


# In[ ]:


#check for the duplicated data based on the column APP name
df1[df1.duplicated(['App'])]


# In[ ]:


df1[df1['App']=="Quick PDF Scanner + OCR FREE"]


# In[ ]:


df1.shape


# In[ ]:


df1.sort_values(by=['Reviews'], inplace=True)


# In[ ]:


df1.head()


# In[ ]:


#Drop the duplicates based on the app names
df2 = df1.drop_duplicates(keep='last',subset=['App'])


# In[ ]:


df2.shape


# In[ ]:


df2.head()


# In[ ]:


df2['Installs'].unique()


# In[ ]:


#Converting the Installs number into float value and copying in a different column
df2['Installs_num'] = df2['Installs'].apply(lambda x: float(x.split("+")[0].replace(",","")))


# In[ ]:


df2['Installs_num'].unique()


# In[ ]:


df2['Price'].unique()   


# In[ ]:


#converting the price into float values
df2['Price_USD'] = df2['Price'].apply(lambda x: float(x.replace("$","")))


# In[ ]:


df2['Price_USD'].unique()


# In[ ]:


df2['Reviews'].unique()   


# In[ ]:


#Converting reviews count into int
df2['Reviews_count']= df1['Reviews'].apply(lambda x: int(x))


# In[ ]:


df2['Reviews_count'].unique()   


# In[ ]:


df2.head()


# In[ ]:


df2['Size'].unique()
    


# In[ ]:


len(df2[df2.Size == "Varies with device"])


# In[ ]:


df2['Size'].replace('Varies with device',np.nan,inplace=True)


# In[ ]:


df2.head()


# In[ ]:


df2["Size"] = (df2["Size"].replace(r'[kM]+$', '', regex=True).astype(float) * df2["Size"].str.extract(r'[\d\.]+([kM]+)', expand=False).fillna(1).replace(["k","M"], [10**3, 10**6]).astype(int))


# In[ ]:


df2.head()


# In[ ]:


df2["Android Ver"].unique()


# In[ ]:


len(df2[df2["Android Ver"] == "Varies with device"])


# In[ ]:


df2["Android Ver"].replace('Varies with device',np.nan,inplace=True)


# In[ ]:


df2["Android Ver"].unique()


# In[ ]:


#finding out the minimum android version supported for the apps
import re
df2['min_android_version'] = df2["Android Ver"].apply(lambda x: re.sub("[a-zA-Z]","", str(x)))


# In[ ]:


df2.head()


# In[ ]:


df2['Rating'].describe()


# In[ ]:


df2.isna().sum()


# In[ ]:


df2['Size'].fillna((df2['Size'].mean()), inplace=True)


# In[ ]:


#Our final data frame with all the extra values removed
df3 = df2.drop(['Reviews','Installs','Price','Android Ver'],axis='columns')


# In[ ]:


df3.head()


# # Step 3 - Data analysis

# In[ ]:


rcParams['figure.figsize'] = 11.7,8.27
g = sns.kdeplot(df3.Rating, color="Red", shade = True)
g.set_xlabel("Rating")
g.set_ylabel("Frequency")
plt.title('Distribution of Rating',size = 20)


# In[ ]:


#Total Number of applications in each categories.

fig = plt.figure(figsize=(16,8)) 
df3.groupby('Category').count()["App"].sort_values(ascending=False).plot(kind='bar',title='Number of apps/Category', fontsize=10)
plt.ylabel('Count') 

#how to display %age on secondary axis on bar plot


# In[ ]:


#What is the average rating of the applications in each categories?
fig = plt.figure(figsize=(16,8)) 
df3.groupby('Category').mean().sort_values(by='Rating',ascending='False')['Rating'].plot(kind='bar',title='Average app rating', fontsize=10)
plt.ylabel('Star Rating') 
plt.style.use('seaborn-white')


# In[ ]:


#What is the average price of applications in each categories?
fig = plt.figure(figsize=(16,8)) 
df3.groupby('Category').mean().sort_values(by='Price_USD',ascending='False')['Price_USD'].plot(kind='bar',title='Price in USD', fontsize=10)
plt.ylabel('Average price - USD') 


# In[ ]:


fig = plt.figure(figsize=(16,8)) 
labels = df3['Category'].value_counts(sort = True).index
sizes = df3['Category'].value_counts(sort = True)
plt.pie(sizes,labels=labels,autopct='%1.1f%%', shadow=True)
plt.title('Top categories',size = 20)
plt.legend(labels, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.show()


# In[ ]:


df3.head()


# In[ ]:


#What are the applications and categories with, most number of reviews, highest rating, most revenue through installs, most number of installs?
#df3["Reviews_count"]==df3.Reviews_count.max()
print("Maximum number of reviews count is of : "+ df3[df3['Reviews_count']==df3.Reviews_count.max()]["App"].iloc[0] + " with a value of " + str(df3.Reviews_count.max()))
print("Maximum Rating is of : "+ df3[df3['Rating']==df3.Rating.max()]["App"].iloc[0] + " with a value of " + str(df3.Rating.max()))
print("Maximum number of installs is of : "+ df3[df3['Installs_num']==df3.Installs_num.max()]["App"].iloc[0] + " with a value of " + str(df3.Installs_num.max()))
print("Maximum price is of : "+ df3[df3['Price_USD']==df3.Price_USD.max()]["App"].iloc[0] + " with a value of " + str(df3.Price_USD.max()))


# In[ ]:


#What is the average size of the applications?
print("Mean size " + str(df3["Size"].mean()/(1024*1024)) + " Mb") #converting in MB
print("Heaviest app size " + str(df3["Size"].max()/(1024*1024)) + " Mb") #converting in MB
print("Smalles app size " + str(df3["Size"].min()/(1024*1024)) + " Mb") #converting in MB


# In[ ]:


#What are the categories with, most number of reviews, highest rating, most revenue through installs, most number of installs?
fig = plt.figure(figsize=(16,8)) 
df3.groupby('Category').sum().sort_values(by='Installs_num',ascending='False')['Installs_num'].plot(kind='bar',title='Installs', fontsize=10)
plt.ylabel('Number of Install') 


# In[ ]:



fig = plt.figure(figsize=(16,8)) 
labels = df3['Content Rating'].value_counts(sort = True).index
sizes = df3['Content Rating'].value_counts(sort = True)
plt.pie(sizes,labels=labels,autopct='%1.1f%%', shadow=True)
plt.title('Content Rating',size = 20)
plt.show()


# In[ ]:


#added in version 16
df10= df3.copy()


# In[ ]:


#added in version 16
def is_free(price):
    if price > 0:
        return 1
    else:
        return 0


# In[ ]:


df3.isna().sum()


# In[ ]:


df10["is_free"] = df10["Price_USD"].apply(lambda x: int(is_free(x))) 


# In[ ]:


df10.head()


# In[ ]:


#How many percent of applications are free and how many are paid?
#to check how many apps are free
fig = plt.figure(figsize=(16,8)) 
labels = ['0 = Free','1 = Paid']
sizes = df10['is_free'].value_counts(sort = True)
plt.pie(sizes,labels=labels,autopct='%1.1f%%', shadow=True)
plt.title('Free vs Paid apps',size = 20)
plt.legend()
plt.show()


# In[ ]:


fig = plt.figure(figsize=(16,8)) 
plt.hist(df3['Rating'],edgecolor = 'black', log=True)
plt.title("Rating distribution")
plt.xlabel('Rating')
plt.tight_layout()
plt.legend()


# In[ ]:



fig = plt.figure(figsize=(16,8)) 
plt.hist(df3['Price_USD'],edgecolor = 'black', log=True)
plt.title("Price of apps")
plt.xlabel('Price')
plt.tight_layout()
plt.legend()


# In[ ]:


df3.head()


# In[ ]:


fig = plt.figure(figsize=(16,8)) 
plt.scatter(df3['Reviews_count'],df3['Rating'],edgecolor='white',linewidth=1,alpha=0.60)
plt.title('Rating distribution')
plt.xlabel('Reviews')
plt.ylabel('Rating')


# In[ ]:


df3[df3['Reviews_count']>70000000]


# In[ ]:


df_paid_apps = df3[df3['Price_USD']>0]


# In[ ]:


#Feature engineering - Creating a column called revenue to find out the most revenue app based on the installs/sales
df_paid_apps["Revenue"] = df_paid_apps["Installs_num"] * df_paid_apps["Price_USD"]


# In[ ]:


df_paid_apps.head()


# In[ ]:


df_paid_apps['Revenue'].describe()


# In[ ]:


#plotting the histogram again after removing free apps
fig = plt.figure(figsize=(16,8)) 
plt.hist(df_paid_apps['Price_USD'],edgecolor = 'black', log=True)
median_age = df_paid_apps['Price_USD'].median()
plt.title("Price of apps")
plt.xlabel('Price')
plt.ylabel('Count')
plt.tight_layout()
plt.legend()


# In[ ]:


fig = plt.figure(figsize=(16,8)) 
df_paid_apps.groupby('Category').sum().sort_values(by='Revenue',ascending='False')['Revenue'].plot(kind='bar',title='Revenue', fontsize=10)
plt.ylabel('Revenue from Sales/ Installs')


# In[ ]:


df_paid_apps[df_paid_apps['Revenue']==69900000]


# In[ ]:


df3['last_updated_date'] = pd.to_datetime(df3['Last Updated'])


# In[ ]:


df3.head()


# In[ ]:


#group time series in weekly basis
df3.groupby(pd.Grouper(key='last_updated_date', freq='1M'))["Category"].count().plot(kind="line",animated=True,figsize=(16,8))


# In[ ]:


df3.head()


# In[ ]:


df3.select_dtypes('object').columns


# # Prediction for App Rating

# In[ ]:


df4 = df3.drop(["App","Last Updated","Current Ver","last_updated_date","Last Updated"],axis="columns")


# In[ ]:


df4.head()


# In[ ]:


df4.dtypes


# In[ ]:


df4["Size"] = df4["Size"].astype('int')
df4["Installs_num"] = df4["Installs_num"].astype('int')
df4["Price_USD"] = df4["Price_USD"].astype('int')
df4["Rating"] = df4["Rating"].astype('int')


# In[ ]:


df4.head()


# In[ ]:


df4.Type = df4.Type.map({"Free":0,"Paid":1})
category_dummies = pd.get_dummies(df4.Category , prefix = "Category")
content_rating_dummies = pd.get_dummies(df4["Content Rating"] , prefix = "content_rating")
genres_dummies = pd.get_dummies(df4["Genres"] , prefix = "genres")


# In[ ]:


df5 = pd.concat([df4 , category_dummies,content_rating_dummies,
                             genres_dummies],axis = 1)


# In[ ]:


df5.isna().sum()


# In[ ]:


df5.head()


# In[ ]:


df6 = df5.drop(["Category","Content Rating","Genres","min_android_version"],axis="columns")


# In[ ]:


df6.select_dtypes('object').columns


# In[ ]:


df6.shape


# In[ ]:


X = df6.drop('Rating', axis=1) 
y = df6.Rating


# In[ ]:


X.head()


# In[ ]:


label = df6.Rating
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5,random_state=42)


# In[ ]:


df6.dtypes


# In[ ]:


label = df6.Rating
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5,random_state=42)


# In[ ]:


lm_model = LogisticRegression()
lm_model.fit(X_train,y_train)


# In[ ]:


test_predictions = lm_model.predict(X_test)


# In[ ]:


metrics.accuracy_score(test_predictions,y_test)


# In[ ]:


print(classification_report(y_test,test_predictions))


# In[ ]:


def predict_rating()


# # Observations
# 
# * **Finance apps** are the costliest one
# * **Games** are the **most installed** category of the apps.
# * Average rating is irrelevant to the category
# * Around **19.7%** of the apps are related to family
# * **81.8%** of the apps belong to everyone category
# 
# 
# # Top Grossers
# 
# ***What are the applications with, most number of reviews, highest rating, most revenue through installs, most number of installs?***
# 
# * Maximum number of reviews count is of : Facebook with a value of 78158306
# * Maximum Rating is of : EC SPORTS with a value of 5.0
# * Maximum number of installs is of : Skype - free IM & video calls with a value of 1000000000.0
# * Maximum price is of : I'm Rich - Trump Edition with a value of 400.0
# 
# 
# What is the average size of the applications?  19.4 Mb
# 
