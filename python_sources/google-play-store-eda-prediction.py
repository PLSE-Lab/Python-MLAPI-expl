#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


data = pd.read_csv("../input/googleplaystore.csv")


# In[ ]:


data.info()


# **i can see rating column has lesser rows as compare to other, this will impact on my analysis.By this information i want to check other column**

# # Preparing data for data science/analytics
# The process of preparing data like resizing,remarking,changing datatype for analytics or others is known as **DATA MINING**. Working with massive data is also known as **DATA MINING** 
# ### CONTENT
# - Finding missing values 
# - Filling up missing values 
# - Changing Dtype if neended

# In[ ]:


data.isnull().sum()


# # Percentage of missing values or Nan values in each column

# In[ ]:


data.isnull().sum()*100/len(data)


# 

# # Missing values in each row

# In[ ]:


data.isnull().sum(axis = 1)*100/len(data)


# In[ ]:


data.info()


# # Default command to drop Na values from whole dataset

# In[ ]:


data.dropna(inplace=True)


# **data of Reviews has string M in it M stands for million we have to remove "," from values .if the values has 'M' and finally convert to int this method can clean the Review column**

# In[ ]:


data["Reviews"]=data["Reviews"].astype(str).str.replace(",","")
data["Reviews"]=data["Reviews"].astype(str).str.replace("M", "")
data["Reviews"] = data["Reviews"].astype(int)

data["Reviews"]


# this method is used to clean "size" column size column contains the strings like 'M' stands for megabyte "Varies with device","k" stands for kilobyte we convert have to every app size to megabytes and return as float type

# In[ ]:


data["Size"] =data["Size"].astype(str).str.replace('Varies with device', "0")
data["Size"] = data["Size"].astype(str).str.replace("M","")
data["Size"] = data["Size"].str.replace(",","")
data["Size"] = data["Size"].str.replace("+","")
data["Size"] = data["Size"].astype(str).str.replace("k","").astype(float)*1024


# In[ ]:


data["Size"]


# **Third method is used to clean installs column it removes the string "+", "," and returns as integer**

# In[ ]:


data["Installs"]=data["Installs"].astype(str).str.replace("+", "")
data["Installs"]=data["Installs"].astype(str).str.replace(",","")
data["Installs"]=data["Installs"].astype(str).str.replace('Free',"")
data["Installs"] = data["Installs"].astype(int)
data["Installs"]


# Fourth method is used to clean "$" and "Everyone string from the column

# In[ ]:


data["Price"] = data["Price"].astype(str).str.replace('Everyone',"0")
data["Price"] = data["Price"].astype(str).str.replace("$","")
data["Price"] = data["Price"].astype(float)
data["Price"] = data["Price"]*70
data["Price"].unique()


# In[ ]:


#now i am using such of lib for visualisation
get_ipython().run_line_magic('matplotlib', 'inline')
sns.heatmap(data.corr(),cmap='coolwarm')


# In[ ]:


import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


# In[ ]:


number_of_apps_in_category = data['Category'].value_counts().sort_values(ascending=True)

df = [go.Pie(labels = number_of_apps_in_category.index,values = number_of_apps_in_category.values,hoverinfo = 'label+value')]

plotly.offline.iplot(df, filename='active_category')


# In[ ]:


df = [go.Histogram(
        x = data.Rating,
        xbins = {'start': 1, 'size': 0.1, 'end' :5}
)]

print('Average app rating = ', np.mean(data['Rating']))
plotly.offline.iplot(df, filename='overall_rating_distribution')


# In[ ]:


#most reviewed app rating
plt.figure(figsize=(12,6))
sns.distplot(data["Rating"],bins=10,color="red")


# In[ ]:


plt.figure(figsize=(12,6))
sns.barplot(x = data.groupby('Category')['Rating'].mean().index, y = data.groupby('Category')['Rating'].mean().values)
plt.xlabel('Category', fontsize=13)
plt.ylabel('Rating', fontsize=13)
plt.xticks(rotation=90)
plt.title("avg rating table based on category")


# # Most popular app

# In[ ]:


most_popular_apps = data[(data["Reviews"]>10000000) ][ (data["Rating"]>=4.5)]
sns.countplot(most_popular_apps["Category"])
plt.xticks(rotation=90)


# # Categorywise free and paid app in google play store

# In[ ]:


sns.set_context('talk',font_scale=1)
plt.figure(figsize=(17,13))
sns.countplot(data=data,y="Category",hue="Type")


# In[ ]:


# Box plot 


# In[ ]:


plt.figure(figsize=(16,12))
sns.boxplot(data=data,x="Size",y="Category",palette='rainbow')


# 3 # Free and Paid apps in google play store

# In[ ]:


sns.countplot(x=data["Type"])


# # Most popular app

# In[ ]:


plt.figure(figsize=(17,13))
sns.countplot(data=data[data["Reviews"]>1000000],y="Category",hue="Type")
plt.title("most popular apps with 1000000+ reviews")
plt.xlabel("no of apps")


# # Most reviewed app rating

# In[ ]:



plt.figure(figsize=(12,6))
sns.distplot(data[data["Reviews"]>10000]["Rating"],bins=10,color="red")


# # Apps with upper then 1,00,000 reviews

# In[ ]:


plt.figure(figsize=(16,6))
sns.scatterplot(data=data[data["Reviews"]>100000],x="Size",y="Rating",hue="Type")
plt.title("apps with reviews graterthan 100000")


# # The most popular paid apps with decent reviews and ratings

# In[ ]:


x=np.log(data["Installs"])
y=np.log(data["Reviews"])
popular_apps = data[(data["Installs"]>10000000) & (data["Rating"]>=4.7)]

pd.DataFrame(popular_apps[popular_apps["Type"]=="Free"][["App"]])


# # Now its time to predict some rating using machine learning

# - Import required libararies
# - Preprocess data
# - Use regex for pattern matching
# - Split data for cross-validation
# - Train machine on robustfit algo
# - Predict new values using test data
# - Check accuracy score
# - Check Mean Squared Error

# # Let's Start!

# # Import libs

# In[ ]:


from sklearn import metrics
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# # Load dataset

# In[ ]:


mldata = data[["Reviews","Size","Installs","Price","Rating"]]
mldata.dropna(inplace=True)

X=mldata.iloc[:,0:-1].values
y = mldata.iloc[:,-1].values


# # Split data for cross validation

# In[ ]:


xtrain,xtest,ytrain,ytest = train_test_split(X,y)


# In[ ]:


#Fit regressor or model on data
rfr = RandomForestRegressor(n_estimators=300)


# # Fit your train data i**n model/Algo

# In[ ]:


rfr.fit(xtrain,ytrain)
ypre = rfr.predict(xtest)

df=pd.DataFrame()

df["ytest"]=pd.Series(ytest)

df["ypre"] =pd.Series(ypre)
df.sample(10)


# In[ ]:


import collections
count = 1
for i in data['Category'].unique():
    print(count,': ',i)
    count = count + 1

sns.set_style('whitegrid')
plt.figure(figsize=(16,8))
plt.title('Number of apps on the basis of category')
sns.countplot(x='Category',data = data)
plt.xticks(rotation=90)
plt.show()


# # Number of reviews on the basis of category

# In[ ]:


sns.set_style('whitegrid')
plt.figure(figsize=(15,8))
sns.scatterplot(y='Category',x='Reviews',data = data,hue='Category',legend=False)
plt.xticks(rotation=90)
plt.title('Number of reviews on the basis of Category')
plt.show()


# # Top 10 categories and their reviews

# In[ ]:


plt.figure(figsize=(20,8))
data.groupby('Category')['Reviews'].sum().sort_values(ascending=False).head(10).plot(kind='bar');
plt.ylabel('Count', fontsize=16)
plt.xlabel('Ratings', fontsize=16)
plt.title("Total Reviews Number for Top 10 Category", fontsize=16)
plt.xticks(rotation=45)
plt.show()


# # Genreswise rating 

# In[ ]:


plt.figure(figsize=(25,10))
plt.scatter(x=data["Genres"],y=data["Rating"],color="green",marker="o")
plt.xticks(rotation=90)
plt.grid()
plt.show()


# In[ ]:






# In[ ]:




