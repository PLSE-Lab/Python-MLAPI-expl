#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
df=pd.read_csv('/kaggle/input/old-car-price-data/imports-85.data.txt',names=headers)


# In[ ]:


df.head()


# In[ ]:


df.replace('?',np.nan,inplace=True )


# In[ ]:


df.head()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.dtypes


# In[ ]:


avg_normalized_losses=df['normalized-losses'].astype('float').mean()
avg_stroke=df['stroke'].astype('float').mean()
avg_bore=df['bore'].astype('float').mean()
avg_horsepower=df['horsepower'].astype('float').mean()
avg_peak_rpm=df['peak-rpm'].astype('float').mean()


# In[ ]:


df['normalized-losses'].replace(np.nan,avg_normalized_losses,inplace=True)
df['stroke'].replace(np.nan,avg_stroke,inplace=True)
df['bore'].replace(np.nan,avg_bore,inplace=True)
df['peak-rpm'].replace(np.nan,avg_peak_rpm,inplace=True)
df['horsepower'].replace(np.nan,avg_horsepower,inplace=True)


# In[ ]:


df.dropna(subset=['price'],inplace=True)


# In[ ]:


df.isnull().sum()


# In[ ]:


df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df["normalized-losses"] = df["normalized-losses"].astype("int")
df["price"] = df["price"].astype("float")
df["peak-rpm"] = df["peak-rpm"].astype("float")
df['horsepower']=df['horsepower'].astype('float')


# In[ ]:


df.dtypes


# In[ ]:


fuel_variable=pd.get_dummies(df['fuel-type'])


# In[ ]:


df=pd.concat([df,fuel_variable],axis=1)


# In[ ]:


df.head()


# In[ ]:


df=df.drop('fuel-type',axis=1)


# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(data=df.corr(),cmap='coolwarm',annot=True)


# In[ ]:


sns.regplot('engine-size','price',df)


# In[ ]:


sns.regplot('curb-weight','price',df)


# In[ ]:


sns.regplot('width','price',df)


# In[ ]:


sns.regplot('city-mpg','price',df)


# In[ ]:


sns.regplot('highway-mpg','price',df)


# In[ ]:


sns.regplot('peak-rpm','price',df)


# In[ ]:


sns.boxplot(x='body-style',y='price',data=df)


# In[ ]:


sns.boxplot(x='engine-location',y='price',data=df)


# In[ ]:


sns.boxplot(x="drive-wheels", y="price", data=df)


# In[ ]:


from scipy import stats


# Let's check  Pearson Correlation Coefficient and P-value  for different variables to determine the important variables in predicting price

# Wheel-base VS Price

# In[ ]:


pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of  =", p_value)  


# In[ ]:


#Horsepower VS Price
pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of  = ", p_value)  


# In[ ]:


#Length VS Price
pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of  = ", p_value)  


# In[ ]:


#Width VS Price
pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of  =", p_value ) 


# In[ ]:


#Curb-weight vs Price
pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of  =", p_value ) 


# In[ ]:


#Engine-Size VS Curb
pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of =", p_value) 


# In[ ]:


#Bore VS Price
pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =  ", p_value ) 


# In[ ]:


#City-mpg VS Price
pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of  = ", p_value)  


# In[ ]:


#Highway-mpg VS Price
pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of  = ", p_value ) 


# The important variable that will determine the price of the car are: Length,
# Width,
# Curb-weight,
# Engine-size,
# Horsepower,
# City-mpg,
# Highway-mpg,
# Wheel-base,
# Bore 

# In[ ]:


x=df[['length','width','curb-weight','engine-size','horsepower','city-mpg','highway-mpg','wheel-base','bore']]


# In[ ]:


y=df['price']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lr=LinearRegression()


# In[ ]:


lr.fit(x_train,y_train)


# In[ ]:


pred=lr.predict(x_test)


# In[ ]:


sns.regplot(y_test,pred)


# We can that the model has predicted the values with a good fit 
