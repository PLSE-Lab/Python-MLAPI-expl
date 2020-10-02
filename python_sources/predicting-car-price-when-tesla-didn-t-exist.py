#!/usr/bin/env python
# coding: utf-8

# I have been working in automobile industry for last 12 years.So any data on automotive industry is interesting for me.Automotive industry has been going through disruptions like electric cars,car sharing and autonomous vehicles.In this project I will be exploring the data set and predicting the price of the car.This project is work in process and I will be updating the project in coming days.If you like my work please do vote for me.Please go through below blog link to read my views on future of Automotive industry. 
# Electric cars http://btplife.blogspot.com/2017/05/electric-car-disruption.html
# 

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


import matplotlib.pyplot as plt
from PIL import Image
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
img=np.array(Image.open('../input/tesla-mode3/Tesla.jpg'))
fig=plt.figure(figsize=(10,10))
plt.imshow(img,interpolation='bilinear')
plt.axis('off')
plt.show()


# Picture shows the Tesla model 3 which is the highest selling  Electric car in the world

# ### Importing Python Modules

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
#plt.style.use('seaborn')
import seaborn as sns
plt.style.use('fivethirtyeight')


# ### Importing data 

# In[ ]:


car=pd.read_csv('../input/automobile-dataset/Automobile_data.csv')
car.head()


# **Summary of DataSet**

# In[ ]:


print('Rows     :',car.shape[0])
print('Columns  :',car.shape[1])
print('\nFeatures :\n     :',car.columns.tolist())
print('\nMissing values    :',car.isnull().values.sum())
print('\nUnique values :  \n',car.nunique())


# In[ ]:


car.shape


# There are 25 features in the data.We have to figure out which parameter will have highest impact on the price of the car.

# In[ ]:


#car.isnull().sum()


# There are no missing values in the data

# In[ ]:


#car.info


# Using the info command we can see that in many place data is represent by symbol **?**.We have to replace **?** with the mean value of the columns.We have to first prepare the data by doing cleaning on the data set.

# ### Handling missing data

# In[ ]:


a=car[car['normalized-losses']!='?']
b=(a['normalized-losses'].astype(int)).mean()
car['normalized-losses']=car['normalized-losses'].replace('?',b).astype(int)


# We can see than in some places the value for num-of-doors data is marked as **?**.We fill this with appropriate value of num of doors.

# ### Cleaning num-of-doors

# In[ ]:


a=car[car['body-style']=='sedan']
a['num-of-doors'].value_counts()


# In case of sedan cars the value of num-of-doors is missing in two places.We can see in most sedan cars the number of doors is 4.We can replace the missing values of num-of-doors with value 4

# In[ ]:


a=car['num-of-doors'].map({'two':2,'four':4,'?':4})
car['num-of-doors']=a


# ### Cleaning price

# In[ ]:


a=car[car['price']!='?']
b=(a['price'].astype(int)).mean()
car['price']=car['price'].replace('?',b).astype(int)


# ### Cleaning horse power 

# In[ ]:


a=car[car['horsepower']!='?']
b=(a['horsepower'].astype(int)).mean()
car['horsepower']=car['horsepower'].replace('?',b).astype(int)


# ### Cleaning bore

# In[ ]:


a=car[car['bore']!='?']
b=(a['bore'].astype(float)).mean()
car['bore']=car['bore'].replace('?',b).astype(float)


# ### Cleaning the stroke column

# In[ ]:


a=car[car['stroke']!='?']
b=(a['stroke'].astype(float)).mean()
car['stroke']=car['stroke'].replace('?',b).astype(float)


# ### Cleaning peak rpm

# In[ ]:


a=car[car['peak-rpm']!='?']
b=(a['peak-rpm'].astype(float)).mean()
car['peak-rpm']=car['peak-rpm'].replace('?',b).astype(float)


# ### Cleaning the number of cylinders

# In[ ]:


a=car['num-of-cylinders'].map({'four':4,'five':5,'six':6,'?':4})
car['num-of-doors']=a


# ### Getting the basic stats of the data

# In[ ]:


car.describe().T


# Some important Observations Mean Horse Power 104
# 
# Highway mileage is 30.75
# 
# Price is 13207.12 Dollars 

# **Make**

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
car['make'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Make of Car')
#ax[0].set_ylabel('Count')
sns.countplot('make',data=car,ax=ax[1],order=car['make'].value_counts().index)
ax[1].set_title('Make of Car')
#ax[1].set_xticklabels(rotation=30)
plt.show()


# We have more Japanese Cars in the Dataset followed by European

# **Fuel Type**

# In[ ]:


car.columns


# In[ ]:


pd.crosstab(car.make,car['fuel-type'],margins=True).T.style.background_gradient(cmap='summer_r')


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
car['fuel-type'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Fuel Type')
ax[0].set_ylabel('Count')
sns.countplot('fuel-type',data=car,ax=ax[1],order=car['fuel-type'].value_counts().index)
ax[1].set_title('Fuel Type')
plt.show()


# We can see that 90% of the cars in the Data Set are Petrol.US is a Gasoline market unlike Europe and India which have more Share of Diesel Cars

# ### Aspiration type

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
car['aspiration'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Aspiration Type')
ax[0].set_ylabel('Count')
sns.countplot('aspiration',data=car,ax=ax[1],order=car['aspiration'].value_counts().index)
ax[1].set_title('Aspiration Type')
plt.show()


# Most vehicle have standard Aspiration.The reason is this data set is very old.Now a days most vehicles have turbo system which improves efficiency and power output from an engine.

# ### Cars of different make in the data set

# In[ ]:


print('Car makers in the data set are',car['make'].unique())

List contains American,Japanese and European car brands.
# ### Understanding the range of important features

# In[ ]:


car[['engine-size','peak-rpm','curb-weight','horsepower','price','highway-mpg']].hist(figsize=(10,8),bins=50,color='b',linewidth='3',edgecolor='k')
plt.tight_layout()
plt.show()


# **1.Curb weight:** Is the total weight of the vehicle without the weight of the passenger.It includes weight of coolants,oil and fuel.Defination of curb weight may vary based on the standard adopted by a country.In this data set the curb weight of most cars is in the range 2000-3100 lbs.
# 
# **2.Engine Size:** It is the amount of air that can be sucked in by the engine.Generally it is measured in litres.For example an average car in India would have an engine capacity in the rane of 1-1.5 liter.
# 
# **3.Highway-mpg:** It is the kilometer or miles that a  car can travel with one liter of fuel on the highway.In this data set it seems to me that it is the amount of miles the car travels with one gallon of fuel.
# 
# **4.Horse Power:** It is the measure of the power of the engine.One horse power is equivalent of the power of one horse.So 100 hp engine woulf be equivalent to the power of 100 horses.Now a days engine power is measure in Kilowatt which is the unit of power in SI System.
# 
# **5.Peak rpm:** RPM (Reolutions per minute) is the measure of the speed of roation of Engine per minute.The peak rpm of the vehicles are generally in the range 5000-6500 rpm.
# 
# **6.Price:** In US today the median price of the vehicle is around 35000$.This is a old data so it shows very low median car price.

# ### Which Make of Car is More?

# In[ ]:


plt.subplots(figsize=(10,6))
ax=car['make'].value_counts().plot.bar(width=0.9,color=sns.color_palette('RdYlGn',20))
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
plt.xticks(rotation='vertical')
plt.xlabel('Car Maker',fontsize=20)
plt.ylabel('Number of Cars',fontsize=20)
plt.title('Cars Count By Manufacturer',fontsize=30)
ax.tick_params(labelsize=15)
#plt.yticks(rotation='vertical')
plt.show()
plt.show()


# In[ ]:


fig = plt.figure(figsize=(15, 10))
ax=sns.countplot(car['make'],palette='dark',edgecolor='k',linewidth=2,order = car['make'].value_counts().index)
plt.xticks(rotation='vertical')
plt.xlabel('Car Maker',fontsize=20)
plt.ylabel('Number of Cars',fontsize=20)
plt.title('Cars Count By Manufacturer',fontsize=30)
ax.tick_params(labelsize=15)
#plt.yticks(rotation='vertical')
plt.show()


# It seems more Japanese cars are sold in the US.As expected Toyota sold more cars in US.In the 70's due to oil price rise Americans switched to small cars.Japanese car makers were good at making high quality small cars.This is the reason we have more cars been sold from Toyota,Mazda and Nissan.

# In[ ]:


print('Different types of cars',car['body-style'].unique())


# In[ ]:


fig = plt.figure(figsize=(15, 10))
cars_type=car.groupby(['body-style']).count()['make']
ax=cars_type.sort_values(ascending=False).plot.bar(edgecolor='k',linewidth=2)
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1),fontsize=11)
plt.xticks(rotation='vertical')
plt.xlabel('Body Type',fontsize=20)
plt.ylabel('Number of Cars',fontsize=20)
plt.title('Count of Cars by Body Type',fontsize=30)
ax.tick_params(labelsize=15)
#plt.yticks(rotation='vertical')
plt.show()


# In[ ]:


from matplotlib.pyplot import plot
fig = plt.figure(figsize=(25, 25))
a=car.groupby(['body-style','make']).count().reset_index();
a=a[['make','body-style','symboling']]
a.columns=['make','style','count']
a=a.pivot('make','style','count')
a.dropna(thresh=3).plot.bar(width=0.85);
#plot.bar()
plt.ioff()
plt.show()


# ### Getting details of Engine type,Number of doors,type of fuel and body style

# In[ ]:


plt.figure(1)
plt.subplot(221)
ax1=car['engine-type'].value_counts(normalize=True).plot(figsize=(10,8),kind='bar',color='orange',edgecolor='k',linewidth=2)
plt.title("Number of Engine Type frequency diagram")
plt.ylabel('Number of Engine Type',fontsize=15)
ax1.tick_params(labelsize=15)
plt.xlabel('engine-type',fontsize=15);


plt.subplot(222)
ax2=car['num-of-doors'].value_counts(normalize=True).plot(figsize=(10,8),kind='bar',color='purple',edgecolor='k',linewidth=2)
plt.title("Number of Door frequetncy diagram")
plt.ylabel('Number of Doors',fontsize=15)
ax2.tick_params(labelsize=15)
plt.xlabel('num-of-doors',fontsize=15);

plt.subplot(223)
ax3=car['fuel-type'].value_counts(normalize= True).plot(figsize=(10,8),kind='bar',color='green',edgecolor='k',linewidth=2)
plt.title("Number of Fuel Type frequency diagram")
plt.ylabel('Number of vehicles',fontsize=15)
plt.xlabel('fuel-type',fontsize=15)
ax2.tick_params(labelsize=15)

plt.subplot(224)
ax4=car['body-style'].value_counts(normalize=True).plot(figsize=(10,8),kind='bar',color='red',edgecolor='k',linewidth=2)
plt.title("Number of Body Style frequency diagram")
plt.ylabel('Number of vehicles',fontsize=15)
plt.xlabel('body-style',fontsize=15);
plt.tight_layout()
plt.show()


# 1.DHC (Direct overhead cam) tyoe of engines are more in the data.
# 
# 2.Most cars sold have 4 doors 
# 
# 3.Petrol(Gas) cars are more popular in America
# 
# 4.Sedan Cars are most popular in America

# ### Plotting the fuel economy of different car makers

# In[ ]:


fig = plt.figure(figsize=(15, 10))
mileage=car.groupby(['make']).mean()
mileage['avg-mpg']=((mileage['city-mpg']+mileage['highway-mpg'])/2)
ax=mileage['avg-mpg'].sort_values(ascending=False).plot.bar(edgecolor='k',linewidth=2)
plt.xticks(rotation='vertical')
plt.xlabel('Car Maker',fontsize=20)
plt.ylabel('Number of Cars',fontsize=20)
plt.title('Fuel Economy of Car Makers',fontsize=30)
ax.tick_params(labelsize=20)
#plt.yticks(rotation='vertical')
plt.show()
plt.show()


# Chevy which is a brand of General motors had the highest milage followed by the Japanese car makers.European car makers except Volkswagen sell Luxary cars.So the Mileage of European car makers are lower.Cars with lower engine capacity generally have higher fuel economy.

# ### Find out the relation between the horse power and the number of cylinders 

# In[ ]:


plt.rcParams['figure.figsize']=(23,10)
ax=sns.factorplot(data=car, x="num-of-cylinders", y="horsepower");
#ax.set_xlabel('Number of Cyliner',fontsize=30)
#ax.set_ylabel('Horse Power',fontsize=30)
#plt.title('Horse Power Vs Num of Cylinder',fontsize=40)
#ax.tick_params(axis='x',labelsize=20,rotation=90)
plt.ioff()


# We can see that the range of power output from eight cylinder engine is very high.It seems more the engines with higher number of cylinders can give a bigger range of power output

# ### Getting the Price of car based on make

# In[ ]:


plt.rcParams['figure.figsize']=(23,10)
ax = sns.boxplot(x="make", y="price", data=car,width=0.8,linewidth=5)
ax.set_xlabel('Make of Car',fontsize=30)
ax.set_ylabel('Price in $',fontsize=30)
plt.title('Price of Car Based on Make',fontsize=40)
ax.tick_params(axis='x',labelsize=20,rotation=90)


# We can see that Mercedes Benz , BMW and Prosche sell the expensive cars in America.

# From the heat map we can make out that Width,Curb weight,Engine weight,Horse power,Highway-mpg have very high correlation to the price of the car.

# In[ ]:


sns.factorplot(data=car, y="price", x="body-style" , hue="fuel-type" ,kind="point")
plt.xlabel('Type of Engine',fontsize=20)
plt.ylabel('Price in $',fontsize=20)
plt.title('Price Vs Engine Type',fontsize=20)
plt.tick_params(axis='x',labelsize=10,rotation=90)


# In[ ]:


plt.rcParams['figure.figsize']=(23,10)
ax=sns.boxplot(x='drive-wheels',y='price',data=car,width=0.8,linewidth=5)
ax.set_xlabel('Make of Car',fontsize=30)
ax.set_ylabel('Price in $',fontsize=30)
plt.title('Price of Car Based on Make',fontsize=40)
ax.tick_params(axis='x',labelsize=20,rotation=90)


# Rear wheel drive are more expensive with a median price of $17000.Generally four wheel drive are more expensive.This is bit surprising.

# ### Plotting heat map to understand correlations between diffeent features

# In[ ]:


import seaborn as sns
plt.figure(figsize=(20,10))
sns.heatmap(car.corr(),annot=True,cmap='summer');


# ### From a Pair plot we can try to Vizualise the correlation of parameters

# In[ ]:


ax = sns.pairplot(car[["width", "curb-weight","engine-size","horsepower","highway-mpg","fuel-type","price",]], hue="fuel-type",palette='dark') #diag_kind="hist"


# 1.Vehicle with high price have low mileage.This because high priced vehicles go into luxary segment which are meant for high performance and running cost is not very important in this segment.
# 
# 2.As the engine power(horse power) increases the price of the vehicle increases.More horse power also means bigger engine size
# 
# 3.As the engine size increases the price of the vehicle increases.Weight of the engine increases with the increase in engine size.
# 
# 4.High curb weight increases price of the vehicle and decreases the mileage of the vehicle.
# 
# 5.Width has very good positive correlation to the price of the vehicle.
# 
# 

# ### Predicting the price of Cars

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split # for spliting the data into training and test set
from sklearn import metrics # for validating the accuracy of the model


# In[ ]:


train,test=train_test_split(car,test_size=0.05)
train.head()


# In[ ]:


X_train=train[['curb-weight','engine-size','horsepower','width']]
#X_train = train.drop('price',axis=1)
y_train=train.price


# In[ ]:


X_test=test[['curb-weight','engine-size','horsepower','width']]
y_test=test.price


# In[ ]:


model=LinearRegression()
model.fit(X_train,y_train)
prediction=model.predict(X_test)


# In[ ]:


y_test


# In[ ]:


Line=prediction.astype(int)
Line


# ### Logistic regression

# In[ ]:


model=LogisticRegression()
lm=model.fit(X_train,y_train)
prediction=model.predict(X_test)
#print('Accuracy of the Logistic Regression is:',metrics.accuracy_score(prediction,y_test))


# In[ ]:


logi=prediction.astype(int)
logi


# ### Decision Tree

# In[ ]:


model=DecisionTreeClassifier()
model.fit(X_train,y_train)
prediction=model.predict(X_test)


# In[ ]:


DTree=prediction.astype(int)
DTree


# In this data set we have information of customer like age,country,gender,annual salary,credit card score,net worth.We will be using Artificial Neural Network and predict the Networth of an individual.Networth of an individual is an important parameter to access whether someone will buy a product or not.This can help us to do a targeted advertising too.This kernel is a work in process.If you like my work please do vote.

# In[ ]:




