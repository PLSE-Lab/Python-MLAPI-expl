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


# Load data in dataframes
# 
# Data: 
# **Country** Name of the country.
# **Region**  Region the country belongs to.
# **Happiness Rank** Rank of the country based on the Happiness Score.
# **Happiness Score**  A metric measured in 2015 by asking the sampled people the question: "How would you rate your happiness on a scale of 0 to 10 where 10 is the happiest."
# **Standard Error**The standard error of the happiness score.
# **Economy (GDP per Capita)** The extent to which GDP contributes to the calculation of the Happiness Score.
# **Family** The extent to which Family contributes to the calculation of the Happiness Score
# **Health (Life Expectancy) **  The extent to which Life expectancy contributed to the calculation of the Happiness Score
# Freedom The extent to which Freedom contributed to the calculation of the Happiness Score.
# **Trust (Government Corruption)** The extent to which Perception of Corruption contributes to Happiness Score.
# **Generosity**   The extent to which Generosity contributed to the calculation of the Happiness Score.
# **Dystopia Residual **  The extent to which Dystopia Residual contributed to the calculation of the Happiness Score.

# In[ ]:


df15=pd.read_csv("../input/2015.csv")
df16=pd.read_csv("../input/2016.csv")
df17=pd.read_csv("../input/2017.csv")


# In[ ]:


#Drop columns which are not required

to_drop= ['Standard Error']
df15.drop(to_drop, axis=1,inplace=True)


# In[ ]:


to_drop= ['Lower Confidence Interval','Upper Confidence Interval']
df16.drop(to_drop, axis=1,inplace=True)


# In[ ]:



to_drop= ['Whisker.high','Whisker.low']
df17.drop(to_drop, axis=1,inplace=True)


# In[ ]:


# Renaming 2017 data to align with 2015,2016 data
new_names={'Happiness.Rank':'Happiness Rank' ,'Happiness.Score':'Happiness Score','Economy..GDP.per.Capita.':'Economy (GDP per Capita)','Health..Life.Expectancy.':'Health (Life Expectancy)','Trust..Government.Corruption.':'Trust (Government Corruption)','Dystopia.Residual':'Dystopia Residual'}
df17.rename(columns=new_names,inplace=True)


# Join all the three dataframes on Country key
# 
# 

# In[ ]:



dfs_3year= df15.merge(df16, suffixes=("2015","2016"), on='Country', how="outer").merge(df17, suffixes=("","2017"), on='Country', how="outer")


# In[ ]:



dfs=dfs_3year[['Country','Happiness Rank2015','Happiness Score2015','Happiness Rank2016','Happiness Score2016','Happiness Rank','Happiness Score']]


# In[ ]:


dfs= dfs.rename(columns={'Happiness Rank': 'Happiness Rank2017','Happiness Score':'Happiness Score2017'})


# In[ ]:


#Checking if any null values in the columns
dfs.loc[dfs.isnull().any(axis=1)]


# In[ ]:


#Dropping the rows with Nan values
dfs.dropna(inplace=True)

# Validating it
dfs.loc[dfs.isnull().any(axis=1)]


# In[ ]:


dfs.head()


# **Plotting Happiness trend  in 3 years**

# In[ ]:


#happiness trend for first 10 countries
import matplotlib.pyplot as plt

factors=dfs[['Happiness Score2015','Happiness Score2016','Happiness Score2017']]
ax=factors.iloc[0:10].plot(kind='bar',fontsize=10,legend=True, figsize=(15,8))
ax.set_ylabel("Happiness Score",fontsize= 10)
ax.set_xlabel("Country", fontsize=10)
ax.set_xticklabels(labels= dfs['Country'].iloc[0:10])
plt.show()


# There's not much change in the Happiness level for first 10 countries

# In[ ]:


#happiness trend for last 10 countries

factors=dfs[['Happiness Score2015','Happiness Score2016','Happiness Score2017']]
ax=factors.iloc[136:147].plot(kind='bar',fontsize=10,legend=True, figsize=(15,8))
ax.set_ylabel("Happiness Score",fontsize= 10)
ax.set_xlabel("Country", fontsize=10)
ax.set_xticklabels(labels= dfs['Country'].iloc[136:147])
plt.show()


# Here Togo shows a significant increase in happiness score 

# Let's calculate the percentage change in happiness from 2015 to 2017

# In[ ]:



def changeinhappiness(y,x):
    return ((y-x)/x)*100


dfs['ChangeinHappiness(2015-2017)%']= dfs.apply(lambda row: changeinhappiness(row['Happiness Score2017'],row['Happiness Score2015']),axis=1)


# In[ ]:


dfs.head()


# In[ ]:


# Sorting Happiness percentage change in decending order
dfs= dfs.sort_values(by='ChangeinHappiness(2015-2017)%', ascending=False)


# In[ ]:


dfs.head()


# In[ ]:


#Plotting changeinHappiness wrt Country
import matplotlib.pyplot as plt

factors= dfs[['ChangeinHappiness(2015-2017)%']]
ax=(factors).plot(kind='barh',stacked= True,figsize=(20,45), fontsize=15,)
ax.set_ylabel("Country",fontsize= 15)
ax.set_xlabel("Change in Happiness Score%", fontsize=15)
ax.set_yticklabels(labels= dfs['Country'])
ax.xaxis.set_ticks_position('both')
ax.set_title("Change in Happiness 2015-2017 %", fontsize=15)
ax.legend(fontsize= 15)
plt.gca().invert_yaxis()
plt.show()


# Togo -- Shows Max increase in happiness score   
# Venezuela -- shows Max drop in Happiness Score

# Finding the factors which affecting the increase/decrease in happiness score of the Togo and Venezuela

# In[ ]:


dfs_3year.head()


# In[ ]:


dfs_togo=dfs_3year.loc[dfs_3year['Country'] == 'Togo']
dfs_venezuela=dfs_3year.loc[dfs_3year['Country'] == 'Venezuela']


# In[ ]:



#Happiness Independent factors comparison for year 2015, 2017 for Togo and Venezuela

factors2015= ['Economy (GDP per Capita)2015','Family2015','Health (Life Expectancy)2015','Freedom2015','Generosity2015','Trust (Government Corruption)2015','Dystopia Residual2015']
factors2017= ['Economy (GDP per Capita)','Family','Health (Life Expectancy)','Freedom','Generosity','Trust (Government Corruption)','Dystopia Residual']

def plot_compare(country):
    title=''.join(country['Country'].values)
   
    y=country[factors2015].values.ravel()
    x= country[factors2015].columns
    y2=country[factors2017].values.ravel()
    labels=['Economy','Family','Health','Freedom','Generosity','Trust','Dystopia Residual']

    plt.figure(figsize=(15,8))
    plt.scatter(x,y, label='2015')
    plt.scatter(x,y2, label='2017')
    plt.vlines(x,y,y2)
    plt.title("Comparison of the 2015 and 2017 independent factors")
    plt.xticks(x,labels, rotation= 'vertical')
    plt.xlabel(title, fontsize='10')
    plt.legend(fontsize= '10')
    plt.show()
    
    

plot_compare(dfs_togo)
plot_compare(dfs_venezuela)


# We can see the Economy, Family are the biggest factors for country Togo
# and Health, Freedom for Venezuela. Dystopia Residual is considered as a benchmark against which the country's happiness is evaluated.

# **Happiness 2017 Analysis**
# 
# 

# In[ ]:


#Plotting 2017 Happiness Score wrt independent factors
import matplotlib.pyplot as plt

factors= df17[['Economy (GDP per Capita)','Family','Health (Life Expectancy)','Freedom','Generosity','Trust (Government Corruption)','Dystopia Residual']]
ax=factors.plot(kind='barh',stacked= True,figsize=(20,45), fontsize=15,)
ax.set_ylabel("Country",fontsize= 15)
ax.set_xlabel("Happiness Score", fontsize=15)
ax.set_yticklabels(df17['Country'])
ax.xaxis.set_ticks_position('both')
ax.set_title("Country Happiness Score 2017", fontsize=15)
ax.legend(fontsize= 15)
plt.gca().invert_yaxis()
plt.show()


# How much each factor affects happiness

# In[ ]:


# function to plot the graphs between happiness score and each factor for year 2017

import matplotlib.pyplot as plt

def happiness_factor_chart(x):
    if isinstance(x,str):
        
        X=df17[x].values
        y=df17['Happiness Score'].values

        fig, ax= plt.subplots()

        ax.scatter(X,y)
        ax.set_xlabel(x)
        ax.set_ylabel('Happiness Score')
        
        plt.show()
    else:
        return "x not str"


# In[ ]:


factors= ["Economy (GDP per Capita)",'Family','Health (Life Expectancy)','Freedom','Trust (Government Corruption)']
for i in factors:
    happiness_factor_chart(i)
    


# Plots show a linear relationship between Happiness Score and Independent factors.

# In[ ]:


# Checking for missing values in data
df17.isnull().values.any()


# In[ ]:


#Checking the correlation between factors
df17_corr=df17.corr()
df17_corr.style.background_gradient(cmap='Blues').set_precision('3')


# Happiness Score is highly correlated to Economy, Family and Health. Freedom, Trust and Dystopia are the medium correlated factors and Generosity being the least.

# **Happiness 2017 prediction**  

#  **Multiple Linear Regression**

# In[ ]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split

feature_col_names=["Economy (GDP per Capita)",'Family','Health (Life Expectancy)','Freedom','Trust (Government Corruption)','Generosity','Dystopia Residual']
score=['Happiness Score']

X = df17[feature_col_names].values
y =df17[score].values
split_size=0.3

X_train, X_test,y_train,y_test= train_test_split(X,y ,test_size=split_size)

#training
mlregr=linear_model.LinearRegression()
mlregr.fit(X_train,y_train)


# In[ ]:


#predicting
y_pred= mlregr.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
print("Coefficients: ", mlregr.coef_)
print("Intercepts:  " , mlregr.intercept_)
print("Mean squared error: %.2f"
      % mean_squared_error(y_test,y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test,y_pred))
#for i in range(len(y_test)):
   
  #  print ("Actual:" ,y_test[i] ," Predicted with linear regression:" ,y_pred[i])


# In[ ]:


#Plotting Predicted vs Actual Happiness Score
plt.scatter(y_test,y_test)
plt.plot(y_test,y_pred,linewidth=1)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()


# This model is a perfect fit for the data

# **SVR**

# In[ ]:


from sklearn import svm

svr= svm.SVR(gamma='scale')
svr.fit(X,y.ravel())


# In[ ]:


y_pred_svr=svr.predict(X_test)


# In[ ]:


print("Mean squared error: %.2f"
      % mean_squared_error(y_test,y_pred_svr))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test,y_pred_svr))


# In[ ]:


#Plotting Predicted vs Actual Happiness Score

plt.scatter(y_test,y_test)
plt.plot(y_test,y_pred_svr)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()


# Accuracy is less for the SVR model

#     Multiple Linear Regression model gives us accurate predictions.

# In[ ]:




