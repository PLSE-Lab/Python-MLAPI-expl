#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# 
#      The dataset "Car Features and MSRP" describes almost 12 000 car models from more than 20 different brands.The aim of this project is   to  compare between the highest and the lowest manufacturer's suggested retail price (MSRP) cars and analyze their differences based on their features:
# 
# 
#               Make   = Cars brands
#               Model  = Cars Models
#               Year   = Year of cars production 
#               Engine HP  = Engine Horsepower
#               highway MPG  = how far the car is able to travel for every gallon of fuel it uses in the highway.
#               city mpg  = how far the car is able to travel for every gallon of fuel it uses around the city. 
#               MSRP = manufacturer's suggested retail price
#               Engine Fuel Type 
#               Engine Cylinders
#               Transmission Type
#               Driven Wheels
#               Number of Doors
#               Market Category
#               Vehicle Size         
#               Vehicle Style               
#               Popularity           
#          
#          
#         * MPG stands for miles per gallon
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
import missingno as msno
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.simplefilter(action='ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


car=pd.read_csv('/kaggle/input/cardataset/data.csv')


# In[ ]:


car.head()


# In[ ]:


plt.style.use('seaborn')
fig= plt.figure(figsize=(16,10))

Market=car['Market Category'].value_counts().head(8).to_frame()
market= Market.style.background_gradient(cmap='Reds')
market


#  Most of the cars  belong to the Market Category " Crossover" .
# 
# * Before starting any data visualization, we need to check if the dataset has any missing values and clean them all of it has any with **Data Cleaning**

# # 1.  Data Cleaning

# In[ ]:


plt.style.use('seaborn')
sns.set_style('whitegrid')
plt.figure(figsize=(15, 3))
car.isnull().mean()


# We have 5 features with missing values of various proportions. We can closely see how these missing values are variated in the plot below.

# In[ ]:


allna = (car.isnull().sum() / len(car))*100
allna = allna.drop(allna[allna == 0].index).sort_values()
plt.figure(figsize=(8, 4))
allna.plot.barh(color=('red', 'black'), edgecolor='black')
plt.title('Missing values percentage per column', fontsize=15, weight='bold' )
plt.xlabel('Percentage', weight='bold', size=15)
plt.ylabel('Features with missing values', weight='bold')
plt.yticks(weight='bold')
plt.show()


# 
# The highest feature with missing values has around 32% of NAs which is not that high, this is a good thing since we can go further with data cleaning without dropping any columns, specially the ones that could be useful and helpful in understanding the dataset in the second part: **Data Visualization**
# 
# * We start with isolating the missing values to have a better idea on how to treat them.

# In[ ]:


NA=car[['Engine Fuel Type','Engine HP', 'Engine Cylinders', 'Number of Doors', 'Market Category']]


# 
# 
# * We split them to:
# 
#     Categorical features
#     
#     Numerical features
# 
# 

# In[ ]:


NAcat=NA.select_dtypes(include='object')
NAnum=NA.select_dtypes(exclude='object')
print('We have :',NAcat.shape[1],'categorical features with missing values')
print('We have :',NAnum.shape[1],'numerical features with missing values')


# * We have 2 categorical features and 3 numerical features to clean.

# 1. Numerical features

# In[ ]:


NAnum.head(3)


#     We can easily clean our numerical features by using the filling forward method since they barely have any missing values.

# In[ ]:


car['Engine HP']=car['Engine HP'].fillna(method='ffill')
car['Engine Cylinders']=car['Engine Cylinders'].fillna(method='ffill')
car['Number of Doors']=car['Number of Doors'].fillna(method='ffill')


# 2. Categorical features

# In[ ]:


NAcat.head(3)


#     Given that Engine Fuel has a very small percentage of missing values, we will clean it using the Fill forward method as well.
# 
# 
#      Market Category has a pretty much high amount of missing values, we will clean it by filling it with 'Luxury' assuming that most of the cars belong to the Market category " Crossover".

# In[ ]:


car['Engine Fuel Type']=car['Engine Fuel Type'].fillna(method='ffill')
car['Market Category']=car['Market Category'].fillna('Crossover')


# In[ ]:


car.isnull().sum().sort_values(ascending=False).head()


#   * Our dataset is clean, we can now move to ***Data* *visualization***.

# # 2. Data Exploration
# 
# 
#         2.1  Car brand distribution

# In[ ]:


carr=car['Make'].value_counts().head(5).to_frame()
m= carr.style.background_gradient(cmap='Blues')
colors=['blue','red','yellow','green','brown']
labels= ['Chevrolet','Ford','Volkswagen','Toyota','Dodge']
sizes= ['1123','881','809','746','626']
explode=[0.1,0.1,0.1,0.1,0.1]
values=car['Make'].value_counts().head(5).to_frame()

#visualization
plt.figure(figsize=(7,7))
plt.pie(values,explode=None,labels=labels,colors=colors,autopct='%1.1f%%')
plt.title('TOP 5 Car brands in the dataset',color='black',fontsize=10)
plt.show()


#    Chevrolet, Ford, Volkswagen, Toyota and Dodge are the brands most present in our dataset. 
# 
#    * Now let's move on to the Brand that has the highest MSRP(manufacturer's suggested retail price)

# In[ ]:


modelp = car.groupby(['Make']).sum()[['MSRP','Popularity','Engine HP','Engine Cylinders','Number of Doors']].nlargest(10, 'MSRP')
modelp.groupby(['Make']).sum()['MSRP'].nlargest(10).iplot(kind='bar', xTitle='Make', yTitle='MSRP',
                                                                     title='Top 10 expensive Car Brands')


#  Chevrolet has the highest manufacturer's suggested retail price (MSRP) value of roughly 33M. What are the feautures that make Cheverolet worth the first ranking?
# 
#   In order to answer this question, we will study the Top 5 most 'expensive' model cars of Chevrolet.

# In[ ]:


Chevrolet=car[car['Make'].str.contains('Chevrolet')]
chev=Chevrolet.sort_values(by=['MSRP'],ascending=False).nlargest(6, 'MSRP')
chevmodel= chev.style.background_gradient(cmap='Greens')
chevmodel


#   The Corvette model ranks the highest in the manufacturer's suggested retail price (MSRP). If we take a look at the features, The models of all 3 years have the same features except for the Vehicle Style: Convertible Corvette are worth more than Coupe Corvette.
#      
#   * That leads us to the question: What are the Vehicle Styles out there and which ones are considered to contribute more to the manufacturer's suggested retail price (MSRP)? 

# In[ ]:


sns.set({'figure.figsize':(20,10)})
VS=sns.barplot(x=car['MSRP'], y=car['Vehicle Style'])
plt.title('Vehicle Style and MSRP', weight='bold', fontsize=18)
plt.xlabel('MSRP', weight='bold',fontsize=14)
plt.ylabel('Vehicle Style', weight='bold', fontsize=14)
plt.xticks(weight='bold')
plt.yticks(weight='bold')


#   Coupe, Convertible, 4dr SUV and Sedan styles : Make the model worth more in terms of the manufacturer's suggested retail price (MSRP)
#   
#   2 dr Suv, Cargo Van, Convertible SUV and 2 dr Hatchback : Make the model worth less in terms of the manufacturer's suggested retail price (MSRP)

# * Now that we've seen the most 'expensive' brand ( Chevrolet), Let's move on to the Least 'expensive' brands.

# In[ ]:


modelp1 = car.groupby(['Make']).sum()[['MSRP','Popularity','Engine HP','Engine Cylinders','Number of Doors']].nsmallest(10, 'MSRP')
modelp1.groupby(['Make']).sum()['MSRP'].nlargest(10).iplot(kind='bar', xTitle='Make', yTitle='MSRP',
                                                                     title='Top 10 least expensive Car Brands')


# * We will choose the brand FIAT in order to study the features of the Least expensive brand. 

# In[ ]:


F=car[car['Make'].str.contains('FIAT')]
FI=F.sort_values(by=['MSRP'],ascending=False).nsmallest(5, 'MSRP')
Fiat= FI.style.background_gradient(cmap='Oranges')
Fiat


#   Looking at the table above, We can come up with 3 remarks:
#      
#    1.  The Vehicle Style of the 500 model of FIAT brand is 2dr Hatchback; and as we've seen in the Vehicle Style plot earlier, 2dr Hatchback  is among the styles that make the brand worth less in terms of the manufacturer's suggested retail price ( MSRP).
# 
# 
#  
#  
#   
#   2. Some features such as Vehicle Size, Number of doors, and Transmission Type could be interchanged between Chevrolet and Fiat, Which could be interpreted as they have **low correlation with MSPR. Therefore, Which features have a high correlation to manufacturer's suggested retail price (MSRP)?**
# 
#   
#   
#   3. Comparing the Chevrolet and Fiat dataframes, We can notice that the higher the Engine Horsepower is, the shorter the distance the car is able to travel for every gallon of fuel it uses in the highway the city and vice versa. 
#          
#      Shouldn't cars with a high engine power produce more power and thus have a high highway and city mpg?
#  
#  
# 
# 
# * In order to answer these questions and understand this better, We will see the correlation between the features using a heatmap. 

#                                   2. Correlation between Features 

# In[ ]:


car_corr=car.corr()
f,ax=plt.subplots(figsize=(12,7))
sns.heatmap(car_corr, cmap='viridis')
plt.title("Correlation between features", 
          weight='bold', 
          fontsize=18)
plt.show()


# There is a positive correlation between: 
# *  Engine HP and Engine Cylinders
# *  Engine HP and the manufacturer's suggested retail price (MSRP)
# * Engine Cylinders and the manufacturer's suggested retail price (MSRP)
# 
# 
#                                             
#                                             

# * Let's plot the graphs to have a better understading of these correlations.

# In[ ]:


plt.figure(figsize=(15,5))
#first row, first col
ax1 = plt.subplot2grid((1,2),(0,0))
plt.scatter(x=car['Engine Cylinders'], y=car['Engine HP'], color='maroon', alpha=0.7)
plt.title('Engine Cylinders on Engine HP', weight='bold', fontsize=18)
plt.xlabel('Engine Cylinders', weight='bold',fontsize=14)
plt.ylabel('Engine HP', weight='bold', fontsize=14)
plt.xticks(weight='bold')
plt.yticks(weight='bold')


#first row sec col
ax1 = plt.subplot2grid((1,2), (0, 1))
sns.regplot(x=car['Engine HP'], y=car['MSRP'], color='maroon')
plt.title('Engine HP on Engine MSRP', weight='bold', fontsize=18)
plt.xlabel('MSRP', weight='bold',fontsize=14)
plt.ylabel('Engine HP', weight='bold', fontsize=14)
plt.xticks(weight='bold')
plt.yticks(weight='bold')

plt.show()


#   An engine with more cylinders produces more power, and more power means a high MSRP.

#                   3.- Shouldn't cars with a high engine power produce more power and thus have a high highway and city mpg?
#                   
#          

# In[ ]:


plt.figure(figsize=(15,5))
#first row, first col
ax1 = plt.subplot2grid((1,2),(0,0))
sns.regplot(x=car["Engine HP"], y=car["highway MPG"], line_kws={"color":"red","alpha":1,"lw":5})
plt.title('Highway MPG and Engine HP', weight='bold', fontsize=18)
plt.xlabel('Engine HP', weight='bold',fontsize=14)
plt.ylabel('Highway MPG', weight='bold', fontsize=14)
plt.xticks(weight='bold')
plt.yticks(weight='bold')

#first row sec col
ax1 = plt.subplot2grid((1,2), (0, 1))
sns.regplot(x=car["MSRP"], y=car["highway MPG"], line_kws={"color":"red","alpha":1,"lw":5})
plt.title('Highway MPG and MSRP', weight='bold', fontsize=18)
plt.xlabel('MSRP', weight='bold',fontsize=14)
plt.ylabel('Highway MPG', weight='bold', fontsize=14)
plt.xticks(weight='bold')
plt.yticks(weight='bold')


#   - Shouldn't cars with a high engine power produce more power and thus have a high highway and city mpg?
#   
# From both the heatmap and the plots above, we can see a negative correlation between highway and city mpg ( how far the car is able to travel for every gallon of fuel it uses in the highway and around the city) and the Engine HP. 
# 
# 
# Horsepower is a measure of **work** that can be performed over a given **time** by an engine. The more power you have the higher the speed you can do it at during **a period of time**. Generally, **higher horsepower** comes from burning more fuel, so get **lower mpg**, more horsepower means less fuel economy.
# 
# 
# Basically, race cars and sports cars such as Corvette Have a High Engine Horsepower and a low mpg since they can speed and thus burn more fuel.
# 
# Cars like FIAT have a lower Engine Horsepower but a higher mpg since the speed is limited and considered to be more fuel economy.
#  

# * How are other features correlated to the manufacturer's suggested retail price (MSRP)?

# In[ ]:



fig= plt.figure(figsize=(16,10))
#2 rows 2 cols
#first row, first col
ax1 = plt.subplot2grid((2,2),(0,0))
sns.boxplot(x=car['Number of Doors'], y=car['MSRP'],color='Red')
plt.title('Number of Doors', weight='bold', fontsize=14)

#first row sec col
ax1 = plt.subplot2grid((2,2), (0, 1))
plt.scatter(x=car['Driven_Wheels'], y=car['MSRP'], color='Orange')
plt.title('Driven Wheels', weight='bold', fontsize=14)



#Second row first column

ax1 = plt.subplot2grid((2,2), (1, 0))
sns.barplot(x=car['Transmission Type'], y=car['MSRP'])
plt.xticks(rotation=35)
plt.title('Transmission Type', weight='bold', fontsize=14)


#second row second column
ax1 = plt.subplot2grid((2,2), (1, 1))
sns.barplot(x=car['Vehicle Size'], y=car['MSRP'])

plt.yticks(weight='bold')
plt.title('Vehicle Size', weight='bold', fontsize=14)


plt.show()


# What contribute to a High Manufacturer's suggested retail price (MSRP) :
#       
#       
#       - Number of doors : Cars with 2 doors 
#       
#       
#       - Driven Wheels : All wheel drive 
#       
#       
#       - Transmission Type: Automated-Manual (It consists of a conventional manual transmission with an electronically-controlled hydraulic clutch and computerized gear shift control, and the driver can usually override the computer control with a clutchless "manual" mode.)
#       
#       
#       - Vehicle Size: Large
#       

# * Now that we've see the Highest and lowest brands of cars in terms of the manufacturer's suggested retail price (MSRP). It would be intresting to look for the 5 TOP popular cars and their features.

# In[ ]:


carmodel = car.groupby(['Make']).sum()[['MSRP','Popularity','Engine HP','Engine Cylinders','Number of Doors']].nlargest(6, 'Popularity')
carmodel.groupby(['Make']).sum()['Popularity'].nlargest(10).iplot(kind='bar', xTitle='Make', yTitle='Popularity',
                                                                     title='Top 5 popular Car Brands')


# Ford ranks the 1st popular brand. 
# 
#   * In order to study its features, we will select the 3 most popular models of this brand

# In[ ]:


Ford=car[car['Make'].str.contains('Ford')]
Fordm=Ford.sort_values(by=['Popularity'],ascending=False).nlargest(3, 'MSRP')
Ford2= Fordm.style.background_gradient(cmap='Reds')
Ford2


# Ford Brand which comparing its manufacturer's suggested retail price (MSRP) to Chevrolet could be very affordable has the least favored features to a high manufacturer's suggested retail price (MSRP) in all : Vehicle Size,Transmission Type, Driven Wheels and Vehicle size ( looking at the plots above). 
# 
# 
# 
# 
# 
# * It would be intresting to go over each Engine Fuel Type and compare their miles per gallon (MPG) values to see how fuel economy they are.

# In[ ]:


EFT=car[['Engine Fuel Type',
      'Engine HP',
      'Engine Cylinders',
      'highway MPG',
         'MSRP',
        'city mpg']].groupby(['Engine Fuel Type']).agg('median').sort_values(by=['Engine HP'],ascending=False)
EFT1= EFT.style.background_gradient(cmap='Purples')
EFT1


# Cars with the Engine Fuel Type 'ELECTRIC' have 0 Engine Cylinders since they have an electric motor and don't rely on engine cylinders to  generate power. 
# 
# Moreover, the electric motor is more efficient than internal combustion engines and save a high proportion of the fuel which explains why electric cars have a very high miles per gallon (mpg) value.
# 

# Recap: 
# 
# * High numbers of Engine Cylinders produce Higher Engine Horsepower.
# * High Engine Horsepower means a High MSRP and a low MPG value ( example: Chevrolet brand/ Corvette model)
# * Cars with an Electric Engine fuel type ( electric motors instead of engine cylinders tend to be very fuel economy and have a high mpg values 
# * Cars with : 2 doors - all wheel drive - large size - Automated manual transmission size and Convertible Style mark a high MSRP Values.
# 
# 
