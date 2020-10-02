#!/usr/bin/env python
# coding: utf-8

# ![](https://image.flaticon.com/icons/png/128/202/202821.png)
# 
# Welcome to my kernel.
# 
# In this kernel, We will try to discovering Android Market with EDA.
# 
# I'm new in data science. Your feedback is very important to me. If you like this kernel, please vote :)
# 
# Let's start.

# In[ ]:


#import libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))


# In[ ]:


#load data
data = pd.read_csv('../input/googleplaystore.csv')


# In[ ]:


#show data first 5 record
data.head()


# In[ ]:


#show data last 5 record
data.tail()


# In[ ]:


#get information about data
data.info()


# If We try to predict Rating, I think App, Current ver, Android Ver are unnecessery cols. So I will delete them.

# In[ ]:


data = data.drop(["App","Current Ver","Android Ver"],1)


# There are NaN values in Rating, Type, Content Rating. We must handle them. 
# 

# <h2>Missing Values</h2>

# In[ ]:


#detect null cols and null rate
nulls = [i for i in data.isna().any().index if data.isna().any()[i]==True]
rates = []
counts = []
for i in nulls:    
    rates.append((data[i].isna().sum()/data.shape[0])*100)
    counts.append(data[i].isna().sum())
null_df = pd.DataFrame.from_dict({"Col":nulls,"Count":counts,"Null_Rates":rates})
null_df


# Missing rates are very low. I think, We can delete specially Type,Content Rating easly. 
# 
# But, we need check Rating.

# In[ ]:


#delete Type,Content Rating, Current Ver, Android Ver null values row
df_train = data.copy()
for i in ['Type','Content Rating']:
    df_train = df_train.drop(df_train.loc[df_train[i].isnull()].index,0)
df_train.info()


# In[ ]:


df_train.Rating.describe()


# 75 percet of rating between 4 and 5.

# In[ ]:


#fill rating null values with mean quartiles
x = sum(df_train.Rating.describe()[4:8])/4
df_train.Rating = df_train.Rating.fillna(x)
print("Dataset contains ",df_train.isna().any().sum()," Nan values.")


# <h2>Data Transformation</h2>

# Rating should **not** to be up to 5 

# In[ ]:


df_train = df_train[df_train["Rating"]<=5]


# For machine learning, we need to convert objects to numbers.

# <h3> 1. Category </h3> 

# In[ ]:


#get unique values in Catagory feature 
df_train.Category.unique()


# In[ ]:


# convert to categorical Categority by using one hot tecnique 
df_dummy = df_train.copy()
df_dummy.Category = pd.Categorical(df_dummy.Category)

x = df_dummy[['Category']]
del df_dummy['Category']

dummies = pd.get_dummies(x, prefix = 'Category')
df_dummy = pd.concat([df_dummy,dummies], axis=1)
df_dummy.head()


# <h3> 2. Genres </h3> 

# In[ ]:


#Genres unique val
df_dummy["Genres"].unique()


# In[ ]:


plt.figure(figsize=(25,6))
sns.barplot(x=df_dummy.Genres.value_counts().index,y=df_dummy.Genres.value_counts())
plt.xticks(rotation=80)
plt.title("Genres and their counts")
plt.show()


# In[ ]:


np.sort(df_dummy.Genres.value_counts())


# Some subcategories have very few examples. Therefore, I will classify those who do not have a significant number of examples as others.

# In[ ]:


lists = []
for i in df_dummy.Genres.value_counts().index:
    if df_dummy.Genres.value_counts()[i]<20:
        lists.append(i)

print(len(lists)," genres contains too few (<20) sample")
df_dummy.Genres = ['Other' if i in lists else i for i in df_dummy.Genres] 


# In[ ]:


df_dummy.Genres = pd.Categorical(df_dummy['Genres'])
x = df_dummy[["Genres"]]
del df_dummy['Genres']
dummies = pd.get_dummies(x, prefix = 'Genres')
df_dummy = pd.concat([df_dummy,dummies], axis=1)


# In[ ]:


df_dummy.shape


# <h3> 3. Contant Rating </h3> 

# This variable is ordinal.

# In[ ]:


#get unique values in Contant Rating feature 
df_dummy['Content Rating'].value_counts(dropna=False)


# In[ ]:


#object(string) values transform to ordinal in Content Rating Feature without nan
df = df_dummy.copy()
df['Content Rating'] = df['Content Rating'].map({'Unrated':0.0,
                                                 'Everyone':1.0,
                                                 'Everyone 10+':2.0,
                                                 'Teen':3.0,
                                                 'Adults only 18+':4.0,
                                                 'Mature 17+':5.0})
df['Content Rating'] = df['Content Rating'].astype(float)
df.head()


# <h3>4. Reviews</h3>

# In[ ]:


#change type to float
df2 = df.copy()
df2['Reviews'] = df2['Reviews'].astype(float)


# <h3> 5. Size </h3>

# In[ ]:


df2["Size"].value_counts()


# In[ ]:


#clean 'M','k', fill 'Varies with device' with median and transform to float 
lists = []
for i in df2["Size"]:
    if 'M' in i:
        i = float(i.replace('M',''))
        i = i*1000000
        lists.append(i)
    elif 'k' in i:
        i = float(i.replace('k',''))
        i = i*1000
        lists.append(i)
    else:
        lists.append("Unknown")
    
k = pd.Series(lists)
median = k[k!="Unknown"].median()
k = [median if i=="Unknown" else i for i in k]
df2["Size"] = k

del k,median,lists


# In[ ]:


#clean 'M'and transform to float 
print("old: ",df['Size'][10]," new: ",df2['Size'][10])


# <h3> 6. Price </h3>

# In[ ]:


#clean '$' and transform to float 
df2['Price'] = [ float(i.split('$')[1]) if '$' in i else float(0) for i in df2['Price'] ] 


# In[ ]:


print("old: ",df['Price'][9054]," new: ",df2['Price'][9054])


# <h3> 7. Installs </h3>

# Clean '+' and ',' char. And transform object(string) to float.

# In[ ]:


df2.Installs.unique()


# In[ ]:


df2["Installs"] = [ float(i.replace('+','').replace(',', '')) if '+' in i or ',' in i else float(0) for i in df2["Installs"] ]


# In[ ]:


print("old: ",df['Installs'][0]," new: ",df2['Installs'][0])


# <h3>8. Type</h3>

# In[ ]:


df2["Type"].unique()


# In[ ]:


df2.Type = df2.Type.map({'Free':0,"Paid":1})


# <h3> 9. Last Updated </h3>

# In[ ]:


df2["Last Updated"][:3]


# In[ ]:


from datetime import datetime
df3 = df2.copy()
df3["Last Updated"] = [datetime.strptime(i, '%B %d, %Y') for i in df3["Last Updated"]]


# In[ ]:


df3 = df3.set_index("Last Updated")
df4 = df3.sort_index()
df4.head()


# <h3> Finish Tranformation </h3>

# finally, lets check Nan

# In[ ]:


df4.isna().any().sum()


# <H3>GREAT !!!</H3>

# Let's see what happened to the data. :)

# In[ ]:


data = df4.copy()
data.shape


# In[ ]:


data.info()


# <h1> Let's do real EDA on our data :) </h1>

# In[ ]:


#additional libraries
from scipy.stats import norm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

style = sns.color_palette("ch:2.5,-.2,dark=.3")


# In[ ]:


#histogram
plt.figure(figsize=(10,5))
sns.distplot(data['Rating'],color='g');
plt.title("Rating Distrubition")
plt.show()


# Apps generally have good rates.

# In[ ]:


#skewness and kurtosis
print("Skewness: %f" % data['Rating'].skew())
print("Kurtosis: %f" % data['Rating'].kurt())


# In[ ]:


#histogram
plt.figure(figsize=(10,5))
sns.countplot(data['Type'],color='red',palette=style);
plt.title("Type Distrubition")
plt.show()


# * **The majority of apps in the market are free.**

# In[ ]:


#histogram
plt.figure(figsize=(8,6))
sns.barplot(x=data['Installs'],y=data.Reviews,color='b',palette=sns.color_palette("ch:2.5,-.2,dark=.3"));
plt.title("Installs Distrubition")
plt.xticks(rotation=80)
plt.show()


# * **We see that there are too many comments in applications that are downloaded too much.**
# 
# We've already been waiting for this.

# In[ ]:


#boxplot plot installs/rates
ax = plt.figure(figsize=(10,5))
sns.set()
sns.boxplot(x="Installs", y="Rating", data=data)
plt.title("Installs/Rating")
plt.xticks(rotation=80)
plt.show()


# 
# As Installs increases, we can see that Rating gets higher values.
# 
# * **This means that many downloaded applications have higher ratings.**

# In[ ]:


chart_data = data.loc[:,"Category_ART_AND_DESIGN":"Category_WEATHER"]
chart_data["Rating"] = data["Rating"]
for i in range(0, len(chart_data.columns), 5):
    sns.pairplot(data=chart_data,
                x_vars=chart_data.columns[i:i+5],
                y_vars=['Rating'])


# * **On the one hand, beauty, education, events, weather caught my attention. Because no application in these categories has a rating below 3.**
# 
# * **On the one hand, Finance, Tools, Family caught my attention. Because the applications in these categories are both very bad and very good rating. So users of applications in these categories can be more relevant and more selective. Perhaps people may expect an application from these categories.**

# In[ ]:


import math
#del chart_data["Rating"]
l = len(chart_data.columns.values)
r = math.ceil(l/5)

chart_data["Type"] = data["Type"]
j=1
plt.subplots(figsize=(15,10),tight_layout=True)
for i in chart_data.columns.values:
    if i=="Type":
        continue
    d = chart_data[chart_data[i]==1]
    plt.subplot(r, 5, j)
    plt.hist(d["Type"])
    plt.title(i)
    j +=1
    
plt.show()


# * **Medical, Personalization, Game are the categories with the most paid applications.**

# In[ ]:


chart_data = data[data["Price"]>0]
chart_data = chart_data.sort_values(by=['Price'],ascending=False)
chart_data = chart_data.head(100)
#chart_data
dic = {}
cols = chart_data.loc[:,"Category_ART_AND_DESIGN":"Category_WEATHER"].columns.values
for i in cols:
    dic[i]=0
    
for i in range(100):
    x = chart_data.iloc[[i]]
    x = x.loc[:,"Category_ART_AND_DESIGN":"Category_WEATHER"]
    for j in x.columns.values:
        if (x[j][0] == 1):
            dic[j]= dic[j] + 1

plt.figure(figsize=(12,5))
plt.bar(dic.keys(), dic.values(), color='g')
plt.xticks(rotation=85)
plt.title("Categories of the 100 most expensive applications")
plt.show()
    


# * **Medical, Finance, Family and Lifestyle applications are the most valuable applications. They must be giving information worth spending money for people.**
# * **I think the places where people spend the most money can be the places they care about the most. Apparently Finance, Family and Madicine are very important to people.**

# Let's look corelation.

# In[ ]:


fig,ax = plt.subplots(figsize=(8,7))
ax = sns.heatmap(data[["Reviews","Price","Rating","Installs","Size"]].corr(), annot=True,linewidths=.5,fmt='.1f')
plt.show()


# * **There is a clearly relationship between Installs and Reviews. It like that in real life.**

# **Thank you for reading. if you like, Don't forget to vote.**
# 
# **See you ..**
