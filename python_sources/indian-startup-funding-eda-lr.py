#!/usr/bin/env python
# coding: utf-8

# #  Indian Startup Funding
# 
# This dataset has funding information of the Indian startups from January 2015 to August 2017.
# It includes columns with the date funded, the city the startup is based out of, the names of the funders, and the amount invested (in USD).
# 
# Perform EDA and apply Linear Regression

# # Import Libraries & load dataset

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
import statsmodels.api as sms
import scipy.stats as stats
import pylab
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.api import het_goldfeldquandt
from statsmodels.stats.diagnostic import linear_rainbow
from statsmodels.stats.stattools import durbin_watson


# In[ ]:


startup = pd.read_csv('../input/indian-startup-funding/startup_funding.csv')
startup.tail()


# # Check Description & Null values

# In[ ]:


startup.info()


# In[ ]:


startup.describe(include=np.object)


# In[ ]:


startup.duplicated().sum()


# # Data Cleaning

# In[ ]:


startup.drop(['Remarks','SNo'],axis=1,inplace=True)


# In[ ]:


startup.head(2)


# In[ ]:


startup['AmountInUSD'] = startup['AmountInUSD'].str.replace(',','')


# In[ ]:


avgfund = startup[pd.notnull(startup['AmountInUSD'])]['AmountInUSD'].astype(int).mean()
round(avgfund)


# In[ ]:


startup['AmountInUSD'].fillna(round(avgfund),inplace=True)


# In[ ]:


startup['AmountInUSD'] = startup['AmountInUSD'].astype(int)


# In[ ]:


startup.info()


# In[ ]:


startup['InvestmentType'].fillna(startup['InvestmentType'].mode()[0],inplace=True)


# In[ ]:


startup.drop('SubVertical',axis=1,inplace=True)


# In[ ]:


startup['IndustryVertical'] = startup['IndustryVertical'].astype(str)


# In[ ]:


#startup['IndustryVertical'].value_counts()


# In[ ]:


def industryMap(category,key):
    startup['IndustryVertical'] =     startup['IndustryVertical'].apply(lambda x: category if (np.str.lower(x).find(key) != -1) else x)  


# In[ ]:


industryMap('Ecommerce','commerce')


# In[ ]:


industryMap('Logistics','logistic')


# In[ ]:


industryMap('Health','health')


# In[ ]:


industryMap('Education','education')


# In[ ]:


industryMap('Food','food')


# In[ ]:


industryMap('Grocery','grocer')


# In[ ]:


industryMap('Technology','analytics')


# In[ ]:


industryMap('Education','ed-tech')


# In[ ]:


industryMap('Technology','data')


# In[ ]:


industryMap('HR','hiring')


# In[ ]:


industryMap('HR','job')


# In[ ]:


industryMap('Food','tea')


# In[ ]:


industryMap('Fashion','fashion')


# In[ ]:


industryMap('Fashion','apparel')


# In[ ]:


industryMap('Entertainment','games')


# In[ ]:


industryMap('Media','news')


# In[ ]:


industryMap('Finance','payment')


# In[ ]:


industryMap('Ecommerce','delivery')


# In[ ]:


industryMap('Wheels','auto')


# In[ ]:


industryMap('Wheels','car')


# In[ ]:


industryMap('Wheels','vehicle')


# In[ ]:


industryMap('Wheels','taxi')


# In[ ]:


industryMap('Wheels','cab')


# In[ ]:


industryMap('Food','tiffin')


# In[ ]:


industryMap('Hospitality','hotel')


# In[ ]:


industryMap('Finance','finance')


# In[ ]:


industryMap('Finance','loan')


# In[ ]:


industryMap('Ecommerce','hyperlocal')


# In[ ]:


industryMap('Health','homeopathy')


# In[ ]:


industryMap('Wheels','commute')


# In[ ]:


industryMap('Hospitality','accomodation')


# In[ ]:


industryMap('Wheels','bike')


# In[ ]:


industryMap('Wheels','wheeler')


# In[ ]:


industryMap('Finance','financ')


# In[ ]:


industryMap('Finance','wallet')


# In[ ]:


industryMap('Health','fitness')


# In[ ]:


industryMap('Hospitality','room')


# In[ ]:


industryMap('Education','learning')


# In[ ]:


industryMap('Health','medical')


# In[ ]:


industryMap('Reality','real estate')


# In[ ]:


industryMap('Reality','residential')


# In[ ]:


industryMap('HR','recruitment')


# In[ ]:


industryMap('Wheels','scooter')


# In[ ]:


industryMap('Travel','travel')


# In[ ]:


industryMap('Internet','internet')


# In[ ]:


industryMap('Internet','web')


# In[ ]:


industryMap('Food','beverage')


# In[ ]:


industryMap('Reality','office')


# In[ ]:


industryMap('Finance','fund')


# In[ ]:


industryMap('Finance','bill')


# In[ ]:


industryMap('Ecommerce','shopping')


# In[ ]:


industryMap('Entertainment','stream')


# In[ ]:


industryMap('Health','pharmacy')


# In[ ]:


industryMap('Online','online')


# In[ ]:


industryMap('Mobile','mobile')


# In[ ]:


industryMap('Mobile','app')


# In[ ]:


industryMap('Technology','platform')


# In[ ]:


industryMap('Marketplace','marketplace')


# In[ ]:


industryMap('Service','service')


# In[ ]:


a = (startup['IndustryVertical'].value_counts() == 1)


# In[ ]:


startup['IndustryVertical'] = startup['IndustryVertical'].apply(lambda x: 'Others' if x in a[a == True].index else x)


# In[ ]:


startup['IndustryVertical'].replace({'nan':np.nan},inplace=True)


# In[ ]:


startup['IndustryVertical'].fillna(method='ffill',inplace=True)


# In[ ]:


startup['IndustryVertical'].value_counts()


# In[ ]:


startup.info()


# In[ ]:


startup['CityLocation'].fillna(startup['CityLocation'].mode()[0],inplace=True)


# In[ ]:


startup['InvestorsName'] = startup['InvestorsName'].str.replace(' ','')


# In[ ]:


startup['InvestorsName'] = startup['InvestorsName'].apply(lambda x:         x.replace('TigerGlobalManagement','TigerGlobal') if (np.str.lower(str(x)).find('tigerglobalmanagement') != -1) else x)  


# In[ ]:


startup['InvestorsName'] = startup['InvestorsName'].apply(lambda x:         x.replace('SequoiaIndia','SequoiaCapital') if (np.str.lower(str(x)).find('sequoiaindia') != -1) else x)  


# In[ ]:


startup['InvestorsName'] = startup['InvestorsName'].apply(lambda x:         x.replace('Undisclosedinvestors','UndisclosedInvestors') if (np.str.lower(str(x)).find('undisclosedinvestors') != -1) else x)  


# ### One hot encoding

# In[ ]:


dd = startup['InvestorsName'].str.get_dummies(sep=',')


# In[ ]:


startup = pd.concat([startup,dd],axis=1)
startup.drop('InvestorsName',axis=1,inplace=True)


# In[ ]:


startup.head()


# In[ ]:


startup['Date'] = startup['Date'].str.replace('.','/')


# In[ ]:


startup['Date'] = startup['Date'].str.replace('//','/')


# In[ ]:


startup['Date'] = pd.to_datetime(startup['Date'])


# In[ ]:


startup['Year'] = startup['Date'].dt.year


# In[ ]:


startup['Month'] = startup['Date'].dt.month


# In[ ]:


startup.drop('Date',axis=1,inplace=True)


# In[ ]:


startup.head()


# # Visualization

# ## How does the funding ecosystem change with time?

# In[ ]:


plt.figure(figsize=(15,10))
sns.set_style('darkgrid')
startup.groupby(['Year','Month'])['Month'].count().plot(color='grey')
plt.show()

we can see between 2015-16 ,funding has been high, but after 2016 trend is decreasing.
# ## Do cities play a major role in funding?

# In[ ]:


plt.figure(figsize=(15,10))
startup['CityLocation'].value_counts().head(10).plot(kind='pie',autopct='%1.1f%%')
plt.show()

Tier-1 has highest share of investment compared to Tier-2.
# ## Which industries are favored by investors for funding?

# In[ ]:


plt.figure(figsize=(15,10))
startup['IndustryVertical'].value_counts().plot(kind='bar',color='purple')
plt.show()

Internet,Technology & Ecommerce are the most favoured industries for funding startup.
# ## Who are the important investors in the Indian Ecosystem?

# In[ ]:


plt.figure(figsize=(12,10))
dd[dd.columns].apply(lambda x : sum(x.values)).sort_values(ascending=False).head(15).plot.barh(color='r')
plt.show()


# ## How much funds does startups generally get in India?

# In[ ]:


startup['StartupName'] = startup['StartupName'].apply(lambda x:         'Flipkart' if (np.str.lower(str(x)).find('flipkart') != -1) else x)  


# In[ ]:


startup['StartupName'] = startup['StartupName'].apply(lambda x:         'Ola' if (np.str.lower(str(x)).find('ola') != -1) else x)  


# In[ ]:


startup['StartupName'] = startup['StartupName'].apply(lambda x:         'Oyo' if (np.str.lower(str(x)).find('oyo') != -1) else x)  


# In[ ]:


startup['StartupName'] = startup['StartupName'].apply(lambda x:         'Paytm' if (np.str.lower(str(x)).find('paytm') != -1) else x)  


# In[ ]:


plt.figure(figsize=(12,10))
sns.barplot(x='StartupName',y='AmountInUSD',data=startup.sort_values('AmountInUSD',ascending=False).head(20))
plt.xticks(rotation=90)
plt.show()

Paytm,Flipkart & Ola are the highest funded startups.
# ## Nature of Investment?

# In[ ]:


startup['InvestmentType'] = startup['InvestmentType'].map({'Private Equity':'PrivateEquity','Seed Funding':'SeedFunding','Crowd Funding':'CrowdFunding'})


# In[ ]:


plt.figure(figsize=(12,10))
sns.countplot(startup['InvestmentType'])
plt.xticks(rotation=90)
plt.show()


# ### Label encoding

# In[ ]:


startup['InvestmentType'] = startup['InvestmentType'].astype(str)


# In[ ]:


startup['InvestmentType'] = LabelEncoder().fit_transform(startup['InvestmentType'])


# In[ ]:


startup['CityLocation'] = LabelEncoder().fit_transform(startup['CityLocation'])


# In[ ]:


startup['IndustryVertical'] = LabelEncoder().fit_transform(startup['IndustryVertical'])


# In[ ]:


startup.head()


# # Applying Linear Regression model

# In[ ]:


#x = startup.drop(['StartupName','AmountInUSD',''],axis=1)
x = startup[['CityLocation','InvestmentType']]
y = startup['AmountInUSD']


# In[ ]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.30,random_state=123)


# In[ ]:


lr = LinearRegression()


# In[ ]:


lr.fit(xtrain,ytrain)
ypred = lr.predict(xtest)


# In[ ]:


r2_score(ytest,ypred)


# In[ ]:


mean_squared_error(ytest,ypred)


# In[ ]:


model = sms.OLS(y,x).fit()
model.summary()


# # Linear Regression Assumptions

# In[ ]:


residual = ytest - ypred


# ### 1. No pattern in residual

# In[ ]:


sns.residplot(ypred,residual)

no pattern in residual.
# ### 2. Normal Distribution

# In[ ]:


stats.probplot(residual,plot=pylab)
plt.show()

Shapio Wilk test of normality

h0(null hypothesis) : residual is normal distribution
h1(alternate hypothesis) : residual is not normal distribution
# In[ ]:


test,pvalue = stats.shapiro(residual)
pvalue

pvalue < 0.05 ,so we reject null hypothesis.
# ### 3. Multicollinearity

# In[ ]:


vif = [variance_inflation_factor(startup[['CityLocation','InvestmentType','AmountInUSD']].values,i) for i in range(startup[['CityLocation','InvestmentType','AmountInUSD']].shape[1])]


# In[ ]:


pd.DataFrame({'vif':vif},index=['CityLocation','InvestmentType','AmountInUSD']).T


# ### 4. Heteroscadastic
# 
# if heteroscadastic, linear regression cannot be used. 
# 
# h0: residual is not heteroscadastic
# 
# h1: residual is heteroscadastic

# In[ ]:


test,pvalue,result = het_goldfeldquandt(residual,xtest)
pvalue

pvalue < 0.05 ,so we reject null hypothesis. ie. Heteroscedastic curve
# ### 5. Auto-correlation
# 
# The errors should not be auto correlated in nature as it will violate the assumptions of the linear regression model.
# 
# - Durbin Watson Test
# 
# 0 to 4
# 
# [0-2) - (+)ve coorelation
# 
# =2 - no correlation
# 
# (2-4] - (-)ve correlaion

# In[ ]:


durbin_watson(residual)

approx 2, so no correlation between residuals.
# ### 6. Linearity
# 
# - Rainbow Test
# 
# h0: linear in nature
# 
# h1: not linear in nature

# In[ ]:


test,pvalue = linear_rainbow(model)
pvalue

pvalue<0.05 ,hence we reject null hypothesis.ie model is not linear.
# # Conclusion

# - Between 2015-16 ,funding has been high, but after 2016 trend is decreasing.
# - Tier-1 has highest share of investment compared to Tier-2.
# - Internet,Technology & Ecommerce are the most favoured industries for funding startup.
# - Paytm,Flipkart & Ola are the highest funded startups.
# - On applying Linear Regression ,accuracy came very small.
# - The model failed linear regression assumptions.
# 
# So we conclude that linear regression is not good fit here.

# In[ ]:




