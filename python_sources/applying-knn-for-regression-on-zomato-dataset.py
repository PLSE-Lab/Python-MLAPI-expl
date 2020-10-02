#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df=pd.read_csv("../input/zomato.csv",encoding = "ISO-8859-1")
country = pd.read_excel('../input/Country-Code.xlsx')
df = pd.merge(df, country, on='Country Code')
df.head()


# In[ ]:


df.shape


# ## 9551 rows and 21 columns are there in the dataset

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame


# # Checking for the data types

# In[ ]:


df.dtypes


# 

# In[ ]:


df.head()


# In[ ]:


df.describe()


# # Above is the Quick view of the dataset

# In[ ]:


df1=df.groupby(["Cuisines"])


# # Grouping data according to cuisines

# In[ ]:


df1.mean()


# # Grouping according to City

# In[ ]:


df2=df.groupby(["City"])
df2.mean()


# In[ ]:


df3=df["City"].value_counts()
df3




# In[ ]:


data_country = df.groupby(['Country'], as_index=False).count()[['Country', 'Restaurant ID']]
data_country.head()
data_country.columns = ['Country', 'No of Restaurant']
plt.figure(figsize=(20,30))
plt.bar(data_country['Country'], data_country['No of Restaurant'],color="brown")
plt.xlabel('Country')
plt.ylabel('No of Restaurant')
plt.title('No of Restaurant')
plt.xticks(rotation = 60)


# > # Inference:
# # 1. Indian city has maximum number of Zomato restaurants
# # 2. Zomato has its presence in 23 countries but the most important country is India

# # So, we should focus on India because of the above reasons

# In[ ]:


data_City = df[df['Country'] =='India']
Total_city =data_City['City'].value_counts()
Total_city.plot.bar(figsize=(20,10))
plt.title('Restaurants by City')                                             
plt.xlabel('City')
plt.ylabel('No of Restaurants')
plt.show()


# # Inference:
# # 1. New Delhi has the highest number of restaraunts associated with Zomato with a count of more than 5000.
# # 2. Gurgaon and Noida are behind New Delhi with count of more than 1000 restaurants associated with Zomato

# In[ ]:


Cuisine_data =df.groupby(['Cuisines'], as_index=False)['Restaurant ID'].count()
Cuisine_data.columns = ['Cuisines', 'Number of Resturants']
Top10= (Cuisine_data.sort_values(['Number of Resturants'],ascending=False)).head(10)
plt.figure(figsize=(20,30))
sns.barplot(Top10['Cuisines'], Top10['Number of Resturants'])
plt.xlabel('Cuisines', fontsize=20)
plt.ylabel('Number of Resturants', fontsize=20)
plt.title('Top 10 Cuisines on Zomato', fontsize=30)
plt.show()


# # Inference:
# # 1. Restaurants providing only North-Indian cuisines are the highest in number with count of approximate 850
# # 2. Restaurants providing both Chinese and North Indian and restaurants providing only Chinese are behind the resturants providing both North-Indianwith a count of approx 450 and 380 respectively.

# In[ ]:


dummy_cuisines=pd.get_dummies(df["Has Online delivery"])
df4=dummy_cuisines.sum()


# In[ ]:


DataFrame(df4)
x=["Yes","No"]
plt.bar(x,df4,color="red")
plt.xlabel("Wether the restaurant has an Online delivery")
plt.ylabel("Count of restaurants")


# # Results:
# # 1. A bar graph presentation which shows how many restaurants    provide Online Delivery.
#  

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# # Table which shows the count of restaurants on the basis of Rating: Average,Excellent,Good,Not rated,Poor,Very Good in different cities.

# In[ ]:


pd.crosstab(df['Rating text'], df['City'])


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


plt.figure(figsize=(20,10))
plt.ylabel("Number of restaurants")
plt.xlabel("Aggregate rating")
sns.barplot(df["Aggregate rating"],range(1,50))

plt.show()


# # Result:
# # 1. Restaurants with ratings 3.3 are highest in number.
# # 2. Near about 30 restaurants are unrated.. May be they are new.
# # 3. There are no restaurants having rating less than 3,which may show that Zomato doesn't collaborate with restaurants having ratings less than 3

# In[ ]:


from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)
wordcloud = (WordCloud(width=1440, height=1080, relative_scaling=1, stopwords=stopwords).generate_from_frequencies(df['Restaurant Name'].value_counts()))


fig = plt.figure(1,figsize=(30,20))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# # Inference:
# # 1. Cafe Coffee Day has maximum number of restaurants associated with Zomato in INDIA followed by Domnio's Pizza and Green Chick Chop.

# In[ ]:


from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import r2_score


# In[ ]:


plt.figure(figsize=(10,8))
plt.scatter(df["Votes"],df["Average Cost for two"],marker="*",color="green")
plt.xlabel("Number of Votes")
plt.ylabel("Average Cost for two")


# # From above scatter plot it is clear that there is almost no relationship between Votes(to restaraunt by its customer) and Average Cost for two.

# ## Correlation between various elements of the dataset

# In[ ]:


df.corr()


# In[ ]:



corrmat = df.corr() 
 
f,ax = plt.subplots(figsize =(9, 8)) 
sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1) 


# ## KNN Regression

# ## Predicting 'Average Cost for Two' using 'Currency'

# ### Importing train_test_split method to split the dataset into training and testing for training the model and then testing it.

# In[ ]:


from sklearn.model_selection import train_test_split


# ## Using KNN for regression

# In[ ]:


x=df[['Currency']]
y=df['Average Cost for two']
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=42)
dummies=pd.get_dummies(x_train)
dummies
dummies2=pd.get_dummies(x_test)
dummies2.head()
k=[]
accu=[]
for i in range(1,50):
    model = neighbors.KNeighborsRegressor(n_neighbors = i)
    model.fit(dummies, y_train)  #fit the model
    pred=model.predict(dummies2) #make prediction on test set
    a=dummies2.shape
    accuracy = r2_score(y_test, pred)
    print("For k=",i)
    print("Accuracy is -",accuracy*100,'%') 
    k.append(i)
    accu.append(accuracy)
    


# ## We can see that the best accuracy score is for K=13

# In[ ]:


plt.plot(k,accu)
plt.xlabel("Value of K")
plt.ylabel("R2_score")


# In[ ]:


model = neighbors.KNeighborsRegressor(n_neighbors = 13)
model.fit(dummies, y_train)  #fit the model
pred=model.predict(dummies2) #make prediction on test set
a=dummies2.shape
accuracy = r2_score(y_test, pred)
for i in range(a[0]):
    print("For ",x_test.iloc[i,:])
    print("average cost for two=")
    print(pred[i])


# In[ ]:



   


# In[ ]:


accuracy = r2_score(y_test, pred)
print("Accuracy is -",accuracy*100,'%') 


# 1. ## Accuracy score is 56.43%.

# ## Predicting 'Average Cost for Two' using 'Currency' and 'Rating text'

# In[ ]:


x=df[['Currency','Rating text']]
y=df['Average Cost for two']
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=42)
dummies=pd.get_dummies(x_train)
dummies
dummies2=pd.get_dummies(x_test)
dummies2.head()
accur=[]
K1=[]
rmse=[]
y_test2=y_test.values##Converting y_test to numpy array

for i in range(1,50):
    model = neighbors.KNeighborsRegressor(n_neighbors = i)
    model.fit(dummies, y_train)  #fit the model
    pred=model.predict(dummies2) #make prediction on test set
    accuracy = r2_score(y_test, pred)
    error=sqrt(mean_squared_error(y_test2,pred))
    print("For K=",i)
    print("Root Mean Squared Error is-",error)
    print("Accuracy is -",accuracy*100,'%') 
    K1.append(i)
    rmse.append(error)
    accur.append(accuracy)
    
 


# ## Root Mean Squared Error vs K values

# In[ ]:


plt.plot(K1,rmse)

plt.xlabel("Value of K")
plt.ylabel("RMSE")


# In[ ]:





# In[ ]:


plt.plot(rmse,accur)
plt.xlabel("RMSE")
plt.ylabel("R2_score")


# ## From this plot it is clear that the highest R2_score is corressponding to the lowest value of RMSE.

# ## R2_Score Error vs K values

# In[ ]:


plt.plot(K1,accur)

plt.xlabel("Value of K")
plt.ylabel("R2_score")


# In[ ]:





# ## We can observe that the maximum accuracy and minimum RMSE value is corressponding to the K-value=2

# In[ ]:


a=dummies2.shape
model = neighbors.KNeighborsRegressor(n_neighbors = 2)
model.fit(dummies, y_train)  #fit the model
pred=model.predict(dummies2) #make prediction on test set
for i in range(a[0]):
    print("For ",x_test.iloc[i,:])
    print("average cost for two=")
    print(pred[i])


# In[ ]:


accuracy = r2_score(y_test, pred)
print("For K=",2)
print("Accuracy is -",accuracy*100,'%')


# ## Accuracy score is about 68.92% which is significant.

# ## **From above it is clear that for various currency based on rating, average cost for two varies.**
#  ## For Indian rupees the average cost for two based on rating are approximated as follows-
#  ## Poor-Rs.550
#  ## Average-Rs.575
#  ## Good-Rs.300
#  ## Very good-Rs.1400
#  ## Not rated-Rs.475
#  ## So,with the above prediction any customer can be sure of how they have to spend for their required quality of foods on Indian Rupees

# ## Using Linear Regression model

# ## 1.Predicting 'Average Cost for Two' using 'Currency' and 'Rating text'

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


x=df[['Currency','Rating text']]
y=df['Average Cost for two']
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=42)
dummies=pd.get_dummies(x_train)
dummies
dummies2=pd.get_dummies(x_test)
dummies2.head()
linear_model=LinearRegression()
linear_model.fit(dummies,y_train)


# In[ ]:


linear_model.coef_


# In[ ]:


linear_model.intercept_


# In[ ]:


prediction=linear_model.predict(dummies2)
r2_score(prediction,y_test)


# In[ ]:


error=sqrt(mean_squared_error(y_test,prediction))
error 


# ## The Regression model did not work well for this as r2 score is very low

# ## 2.Predicting 'Average Cost for Two' using 'Price range' and 'Aggregrate Rating'

# In[ ]:


x=df[['Aggregate rating','Price range']]
y=df['Average Cost for two']
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.6,random_state=42)
linear_model=LinearRegression()
linear_model.fit(x_train,y_train)


# In[ ]:


linear_model.coef_


# In[ ]:


linear_model.intercept_


# In[ ]:


prediction=linear_model.predict(x_test)
r2_score(y_test,prediction)


# In[ ]:


error=sqrt(mean_squared_error(y_test,prediction))
error


# ## Here too, the Linear Regression model didn't work well as R2_score is very low.
# ## Hence, also from correlation values it is understood that we cannot use Linear Regression model for this dataset.
