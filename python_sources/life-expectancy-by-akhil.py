#!/usr/bin/env python
# coding: utf-8

# I have done some changes to the original data files beacuse it has some anamolies.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


# In[ ]:


warnings.filterwarnings('ignore')


# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv('../input/led.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


sns.countplot(df['Status'])


# In[ ]:


df = df.dropna()


# In[ ]:


plt.hist(df['Lifeexpectancy'])


# most of the population die at the age of 70-80
# the minimum value is at below 40 whereas max is at 90

# <h4>Does various predicting factors which has been chosen initially really affect the Life expectancy? What are the predicting variables actually affecting the life expectancy?</h4>

# In[ ]:


sns.barplot(x = df['Status'],y=df['Lifeexpectancy'])


# life expectancy of developed countries are greater than that of developing countries

# In[ ]:


sns.regplot(x=df['AdultMortality'],y=df['Lifeexpectancy'])


# the trend is increase in adult mortality decreases life expectancy

# In[ ]:


sns.regplot(x=df['infantdeaths'],y=df['Lifeexpectancy'])


# the trend is increase in infant deaths may decrease life expectancy

# In[ ]:


sns.regplot(x=df['Alcohol'],y=df['Lifeexpectancy'])


# increase in consumption of alcohol increases life expectancy

# In[ ]:


sns.regplot(x=df['percentageexpenditure'],y=df['Lifeexpectancy'])


# increase in percentage expenditure increases life expectancy

# In[ ]:


sns.regplot(x=df['HepatitisB'],y=df['Lifeexpectancy'])


# they are equally spread, LE not exactly depending on HepatitisB disease

# In[ ]:


sns.regplot(x=df['Measles'],y=df['Lifeexpectancy'])


# they are equally spread, LE not exactly depending on Measles disease

# In[ ]:


sns.regplot(x=df['BMI'],y=df['Lifeexpectancy'])


# the people with better life expectancy have better bmi

# In[ ]:


sns.regplot(x=df['under-fivedeaths'],y=df['Lifeexpectancy'])


# The deaths (under 5) major contribution in descreasing life expectancy

# In[ ]:


sns.regplot(x=df['Polio'],y=df['Lifeexpectancy'])


# the graph is wide spread. Polio immunization have some impact

# In[ ]:


sns.regplot(x=df['Totalexpenditure'],y=df['Lifeexpectancy'])


# the graph is wide spread. hepatitisB immunization have some impact

# In[ ]:


sns.regplot(x=df['Diphtheria'],y=df['Lifeexpectancy'])


# the graph is wide spread. Measles immunization have some impact

# In[ ]:


sns.regplot(x=df['HIV/AIDS'],y=df['Lifeexpectancy'])


# Hiv have large role in decreasing life expectancy

# In[ ]:


sns.regplot(x=df['GDP'],y=df['Lifeexpectancy'])


# high GDP can increase life expecatancy of a country

# In[ ]:


sns.regplot(x=df['Population'],y=df['Lifeexpectancy'])


# In[ ]:


sns.regplot(data = df[df['Population']<1000000000],x='Population',y='Lifeexpectancy')


# populatation not actually effecting life expectancy

# In[ ]:


sns.regplot(x=df['thinness1-19years'],y=df['Lifeexpectancy'])


#  as thinness1-19years increase life expectancy decrease. that means nutritious food is required for better living

# In[ ]:


sns.regplot(x=df['thinness5-9years'],y=df['Lifeexpectancy'])


#  as thinness2-5years increase life expectancy decrease. that means nutritious food is required for better livingm,mm

# In[ ]:


sns.regplot(x=df['Incomecompositionofresources'],y=df['Lifeexpectancy'])


# Incomecompositionofresources increases life expectancy increases

# In[ ]:


sns.regplot(x=df['Schooling'],y=df['Lifeexpectancy'])


# Incomecompositionofresources increases life expectancy increases, May be due to midday meal scheme in several schools across the globe. A person who studies can get a job too for surviving

# <h4>so we can say that schooling, Incomecompositionofresources, thinness5-9years, thinness1-19years, GDP, HIV/AIDS, under-fivedeaths, BMI, Status, Adultmortality, infant deaths have mjor contribution in changing life expecatncy</h4> 

# <h4>Should a country having a lower life expectancy value(less than 65) increase its healthcare expenditure in order to improve its average lifespan? </h4>

# In[ ]:


plt.figure(figsize=(15,5))
plt.subplot(121)
sns.regplot(x=df['Totalexpenditure'],y=df['Lifeexpectancy'])
plt.subplot(122)
sns.regplot(x=df['percentageexpenditure'],y=df['Lifeexpectancy'])


# yes it will be good option for a country to increase expenditure on health care to incrrease life expectancy.

# <h4>How does Infant and Adult mortality rates affect life expectancy? </h4>
# infant and adult mortality rates decreasing life expectancy. because comes down some when one person die at small age

# <h4>Does Life Expectancy has positive or negative correlation with eating habits, lifestyle, exercise, smoking, drinking alcohol etc?.</h4>
# to be suprised life expectancy have a positive correlation with alcohol

# <h4>  What is the impact of schooling on the lifespan of humans?</h4>
# the increase in no of schooling years increase life expecatncy

# <h4>Does Life Expectancy have positive or negative relationship with drinking alcohol? </h4>
# It have positive correlation with the drinking alchohol

# <h4>Do densely populated countries tend to have lower life expectancy?</h4>
# Population is not contributing to life expectancy directly

# <h4>What is the impact of Immunization coverage on life Expectancy?</h4>
# They are also contributing for increase in life expectancy. But not a major impact

# In[ ]:


df.replace(['Developed','Developing'],[1,0],inplace=True)


# In[ ]:


sns.pairplot(df,hue='Status')


# ###### from the above pair plot we can clearly understand that the factors that are contributing to decrease in  life expectancy are very less in developed countries.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['Country'] = encoder.fit_transform(df['Country'])


# In[ ]:


plt.figure(figsize=(15,15))
sns.heatmap(df.corr(),annot=True)


# In[ ]:


# schooling,Status,Incomecompositionofresources,thinness5-9years,thinness1-19years,GDP,HIV/AIDS,Diptheria,Totalexpenditure,Polio,under-fivedaths,BMI,percentageexpenditure,Alchohol,AdultMortality,Country,infantdeaths


# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(df[['Lifeexpectancy','Status','Schooling','thinness1-19years','GDP','HIV/AIDS','Diphtheria','Totalexpenditure','AdultMortality','Country','infantdeaths']].corr(),annot=True)


# In[ ]:


features = ['Schooling','thinness1-19years','GDP','HIV/AIDS','Diphtheria','Totalexpenditure','AdultMortality','Country','infantdeaths']


# In[ ]:


df['Year'].unique()


# Lets divide the data into two, train and test, the years 2014 and 2015 are test data and remaining are train data

# In[ ]:


trdf = df[df['Year']<2014]
tedf = df[df['Year']>=2014]


# In[ ]:


X_train = trdf[features]
X_test = tedf[features]
y_train = trdf['Lifeexpectancy']
y_test = tedf['Lifeexpectancy']


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[ ]:


lin = LinearRegression()
lin.fit(X_train,y_train)
y_pred = lin.predict(X_test)
np.sqrt(metrics.mean_squared_error(y_pred,y_test))


# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


rf = RandomForestRegressor(n_estimators=100,max_depth=4)
rf.fit(X_train,y_train)
y_predr = rf.predict(X_test)
np.sqrt(metrics.mean_squared_error(y_predr,y_test))


# In[ ]:




