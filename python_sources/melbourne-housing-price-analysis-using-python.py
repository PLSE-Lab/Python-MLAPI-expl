#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Load necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Reading the dataset
Housing_MB = pd.read_csv('../input/melbourne-housing-market/Melbourne_housing_FULL.csv')


# In[ ]:


Housing_MB.head(5)


# In[ ]:


# exploring and understanding the dataset
print(Housing_MB.shape)
print(Housing_MB.describe())


# # Inferences
# 1. There are minimum of 1 room houses and maximum of 16 room houses sold in the market.Most of the houses sold have 4 rooms.
# 2.Most of the houses are sold are situated at an average distance of 11 kms from CBD.
# 3.Most of the houses sold have an average of 3 bedroom scrapped and which goes up to maximum 30 bedrooms scrapped.
# 4.Most of the houses sold have 2 bathrooms,2 car spots and have an average of Land Size 593 meters
# 5. Most of the houses sold have an average building area of 160 meters and Land size of 593 meters which can go upto maximum of 433k meters.
# 6. Suburbs where most of the houses sold has a average of 7k-10k properties

# In[ ]:


# Identifying the missing values
Housing_MB.info()


# In[ ]:


# Understanding variables in Suburb column
print(Housing_MB['Suburb'].value_counts())


# In[ ]:


# Understanding variables in Type column
print(Housing_MB['Type'].value_counts())
#Most of the houses sold are house,cottage villa or semi terrace type


# In[ ]:


# Understanding variables in Method column
print(Housing_MB['Method'].value_counts())


# In[ ]:


print(Housing_MB['CouncilArea'].value_counts())


# In[ ]:


print(Housing_MB['Regionname'].value_counts())
#Most of the houses sold are from Southern Metropolitan region


# In[ ]:


print(Housing_MB['Suburb'].value_counts())


# In[ ]:


# Total Missing value for each feature
print(Housing_MB.isnull().sum())


# In[ ]:


# Replacing Missing values in columns where we have less than 30% missing values
Housing_MB['Bedroom2'].fillna(Housing_MB['Bedroom2'].median(),axis=0,inplace=True)
Housing_MB['Bathroom'].fillna(Housing_MB['Bathroom'].median(),axis=0,inplace=True)
Housing_MB['Car'].fillna(Housing_MB['Car'].median(),axis=0,inplace=True)
Housing_MB['Landsize'].fillna(Housing_MB['Landsize'].median(),axis=0,inplace=True)
Housing_MB['Lattitude'].fillna(Housing_MB['Lattitude'].median(),axis=0,inplace=True)
Housing_MB['Longtitude'].fillna(Housing_MB['Longtitude'].median(),axis=0,inplace=True)
Housing_MB['Regionname'].fillna(Housing_MB['Regionname'].mode(),axis=0,inplace=True)
Housing_MB['CouncilArea'].fillna(Housing_MB['CouncilArea'].mode(),axis=0,inplace=True)
Housing_MB['Propertycount'].fillna(Housing_MB['Propertycount'].median(),axis=0,inplace=True)


# In[ ]:


Housing_MB['Regionname'].fillna('Southern Metropolitan',inplace=True)
Housing_MB['CouncilArea'].fillna('Boroondara City Council',inplace=True)


# In[ ]:


# Validaing the Missing value after missing value is treated for few feature columns
print(Housing_MB.isnull().sum())


# In[ ]:


Housing_MB['Date']= pd.to_datetime(Housing_MB['Date'],dayfirst=True)


# In[ ]:


# Grouping the features by Date
var = Housing_MB[Housing_MB['Type']=='h'].sort_values('Date',ascending=False).groupby('Date').std()
count = Housing_MB[Housing_MB['Type']=='h'].sort_values('Date',ascending=False).groupby('Date').count()
mean = Housing_MB[Housing_MB['Type']=='h'].sort_values('Date',ascending=False).groupby('Date').mean()


# In[ ]:


var


# In[ ]:


mean


# In[ ]:


# Average Price marked by varaince by comparing with different date or time when the houses were sold
mean["Price"].plot(yerr=var["Price"],ylim=(400000,1500000))


# In[ ]:


# Plotting average Landsize marked by variance in price
mean["Landsize"].plot(yerr=var["Price"])


# In[ ]:


#Group all the features by Date for the houses of type h and Distance less than 14 kms from CBD.
feature_means = Housing_MB[(Housing_MB['Type']=='h')& (Housing_MB['Distance']<14)].sort_values('Date',ascending=False).groupby('Date').mean()
feature_std = Housing_MB[(Housing_MB['Type']=='h') & (Housing_MB['Distance']<14)].sort_values('Date',ascending=False).groupby('Date').std()


# In[ ]:


#Average no. of Bedroom,Bathroom,Car in Houses sold of h type and which is located within the distance of 14 kms from CBD.
feature_means[['Bedroom2','Bathroom','Car']].plot()


# In[ ]:


#Average no.of Bedroom,Bathroom,Car marked by variance in Houses sold of h type and which is located within the distance of 14 kms from CBD.
feature_means[['Bedroom2','Bathroom','Car']].plot(yerr=feature_std)


# In[ ]:


feature_location=Housing_MB[(Housing_MB['Type']=='h')&(Housing_MB['Distance']<14)].sort_values('Date',ascending=False).dropna().groupby(['Suburb']).mean()


# In[ ]:


#Group all the features by Regionname for the houses of type h and Distance less than 14 kms from CBD.
feature_region_mean=Housing_MB[(Housing_MB['Type']=='h')&(Housing_MB['Distance']<14)].sort_values('Date',ascending=False).dropna().groupby('Regionname').mean()
feature_region_std=Housing_MB[(Housing_MB['Type']=='h')&(Housing_MB['Distance']<14)].sort_values('Date',ascending=False).dropna().groupby('Regionname').std()


# In[ ]:


# Plotting the avrega eprice of house sold by Regionname
feature_region_mean['Price'].plot(kind='bar',figsize =(15,8))


# In[ ]:


# Plotting the average no.of Bathroom,Bedroom and Carspots by Regionname
feature_region_mean[['Bedroom2','Bathroom','Car']].plot(yerr=feature_region_std,figsize=(15,8))


# In[ ]:


# Looking at the average price range in suburb for houses sold in Southern Metropolitan
feature_SouthernM = Housing_MB[(Housing_MB['Regionname']=='Southern Metropolitan')& 
                               (Housing_MB['Type']=='h') & 
                                (Housing_MB['Distance']<=14)].sort_values('Date',ascending=False).groupby('Suburb').mean()


# In[ ]:


feature_SouthernM['Price'].plot(kind='bar',figsize=(20,10))


# In[ ]:


#Analyzing Average no. of rooms and Distance for each of the Suburb in Southern Metropolitan Region
feature_South_Suburb = Housing_MB[(Housing_MB['Regionname']=='Southern Metropolitan')& 
                               (Housing_MB['Type']=='h') & 
                                (Housing_MB['Distance']<=14)].sort_values('Date',ascending=False).groupby('Suburb').agg({'Rooms':'median','Distance':'mean'})


# In[ ]:


feature_South_Suburb


# In[ ]:


#Analyzing Average no. of rooms and Distance for each of the Suburb in Western Metropolitan Region
feature_West_Suburb = Housing_MB[(Housing_MB['Regionname']=='Western Metropolitan')& 
                               (Housing_MB['Type']=='h') & 
                                (Housing_MB['Distance']<=14)].sort_values('Date',ascending=False).groupby('Suburb').agg({'Rooms':'median','Distance':'mean'})


# In[ ]:


feature_West_Suburb


# In[ ]:


# Looking at the average price range in suburb for houses sold in Western Metropolitan
feature_WesternM = Housing_MB[(Housing_MB['Regionname']=='Western Metropolitan')& 
                               (Housing_MB['Type']=='h') & 
                                (Housing_MB['Distance']<=14)].sort_values('Date',ascending=False).groupby('Suburb').mean()


# In[ ]:


feature_WesternM['Price'].plot(kind='bar',figsize=(20,10))


# In[ ]:


# Looking at the average price range in suburb for 2 bedroom houses located in the distance of less than 5 kms from CBD sold in Southern Metropolitan 
# Anlyzing the affordable price in the suburbs.
Southern_affordable = Housing_MB[(Housing_MB['Regionname']=='Southern Metropolitan')&
                                (Housing_MB['Rooms']==2)&
                                (Housing_MB['Type']=='h')&
                                (Housing_MB['Distance']<=5)].sort_values('Date',ascending=False).groupby('Suburb').mean()


# In[ ]:


Southern_affordable['Price'].plot(kind='bar',figsize=(20,10))


# In[ ]:


# Looking at the average price range in suburb for 2 bedroom houses located in the distance of less than 5 kms from CBD sold in Southern Metropolitan 
# Anlyzing the affordable price in the suburbs.
Western_affordable = Housing_MB[(Housing_MB['Regionname']=='Western Metropolitan')&
                                (Housing_MB['Rooms']==2)&
                                (Housing_MB['Type']=='h')&
                                (Housing_MB['Distance']<=6)].sort_values('Date',ascending=False).groupby('Suburb').mean()


# In[ ]:


Western_affordable['Price'].plot(kind='bar',figsize=(20,10))


# In[ ]:


sns.kdeplot(Housing_MB[(Housing_MB['Regionname']=='Southern Metropolitan')
                       &(Housing_MB['Type']=='h')
                       &(Housing_MB['Rooms']==2)]
                       ["Price"])


# In[ ]:


sns.kdeplot(Housing_MB[(Housing_MB['Regionname']=='Western Metropolitan')
                       &(Housing_MB['Type']=='h')
                       &(Housing_MB['Rooms']==2)]["Price"])


# In[ ]:


# Plotting the pairplot to understand the distribution and relationship between features
sns.pairplot(Housing_MB.dropna())


# In[ ]:


# Plotting the heatmap to understand the features correlation
fig,ax = plt.subplots(figsize=(15,15))
sns.heatmap(Housing_MB.corr(),annot=True)


# In[ ]:


# Plotting the heatmap to understand the features correlation for houses sold of type h
fig,ax = plt.subplots(figsize=(15,15))
sns.heatmap(Housing_MB[Housing_MB['Type']=='h'].corr(),annot=True)


# In[ ]:


#Drop Null values from dataframe
dataframe_Housing = Housing_MB.dropna().sort_values('Date')


# In[ ]:


# Convert the date column to number of days from the date when the house is sold
from datetime import date
days_since_start = [(x-dataframe_Housing['Date'].min()).days for x in dataframe_Housing['Date']]
dataframe_Housing['Days']= days_since_start


# In[ ]:


# Dropping columns which has less correlation to target variable(Price)
df_Housing=dataframe_Housing.drop(['Date','Address','SellerG','Postcode','Landsize','Propertycount'],axis=1)


# In[ ]:


# understanding the dattyoes from the Housing data frame
df_Housing.dtypes


# In[ ]:


df_Housing['CouncilArea'].value_counts()


# In[ ]:


# Convertig Object columns to dummies
df_dummies = pd.get_dummies(df_Housing[['Type','Method','CouncilArea','Regionname']])


# In[ ]:


df_Housing.columns


# In[ ]:


#Dropping the old columns which have been converted to dummies and creating a new dataframe
df_Housing.drop(['Suburb','Type','Method','CouncilArea','Regionname'],axis=1,inplace=True)
df_Housing=df_Housing.join(df_dummies)


# In[ ]:


df_Housing.head(5)


# In[ ]:


# Splitting indepnedent and dependent features into X and y
from sklearn.model_selection import train_test_split
X= df_Housing.drop(['Price'],axis=1)
y= df_Housing['Price']


# In[ ]:


# Train test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=10)


# In[ ]:


# Train the model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)


# In[ ]:


print(lm.intercept_)


# In[ ]:


lm.score(X_test, y_test)


# In[ ]:


# Arriving at the coeffecient for the features
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
ranked_suburbs = coeff_df.sort_values("Coefficient", ascending = False)
ranked_suburbs


# In[ ]:


predictions =lm.predict(X_test)


# In[ ]:


# Plotting a scatter plot with Predicted and Actual Values based on the trained model
plt.scatter(y_test,predictions)
plt.ylim([200000,1000000])
plt.xlim([200000,1000000])

