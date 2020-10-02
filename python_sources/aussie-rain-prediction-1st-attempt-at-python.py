#!/usr/bin/env python
# coding: utf-8

# **Predicting rain: First Python kernel**
# 
# Hi there. This will be my first Python kernel.
# I will start by checking dataset, then do descriptive on variables with charts. I attempt to run logistic regression to analyze the predictors of raining the next day. The predictors are data within a given day to predict if it will rain the next day. I hope that's not confusing.
# 
# 
# I am doing this to learn how to use Python in data science. I am no meteorologist so do not expect expert subject matter-knowledge on predicting weather. Worse, I am not Australian so my I'll avoid talking much about Aussie geography. 
# 
# 
# P.S. English is just my second language so please excuse grammatical and typographical errors, if any.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split


import os
print(os.listdir("../input"))


# Read the data

# In[ ]:


df=pd.read_csv('../input/weatherAUS.csv')


# Data structure and data types of variables
# 

# In[ ]:


df.shape


# There are 142,193 days (unit of analysis) and 24 columns (variables).
# 
# I think it's a panel data considering that some days are duplicated because
# predictor variable data can change by specific location in Australia. 
# I mean it's the same day but data on predictor variables is recorded multiple times
# because Australia has multiple regions, which causes the values of the predictors to vary.
# 
# To illustrate, on the same day, the temperature (one predictor) in Adelaide is different from Sydney.

# Checking for duplicated rows/days

# In[ ]:


df.duplicated(subset='Date', keep=False).sum()


# The 142,101, out of 142,193, duplicated rows (days) confirms the panel structure of dataset.

# In[ ]:


df.dtypes


# Most variables are float (numeric) while some are classified as object (categorical/nominal/date)

# In[ ]:


#First 10 rows
df.head(10)


# In[ ]:


start=df['Date'].min()
start


# In[ ]:


end=df['Date'].max()
end


# So the time frame of the 'rainy/non-rainy days' data spans from November 1, 2007 to June 25, 2017.
# 
# I got almost 10 years worth of daily data to analyze the predictors of a rainy tomorrow in Aussieland.

# Descriptive statistics per variable

# In[ ]:


df.describe()


# Based on the count, there may be some concerns on missing values.
# 
# 
# Missing values for each variable

# In[ ]:


df.isnull().sum()


# The presence of missing/blank data on predictor variables is confirmed.

# * **Dropping missing values**
# 
# Apparently, I have to remove the missing values because logistic regression requires.
# 
# 
# Random thought: This was much simpler in R Studio because running model will just exclude the rows with missing values. I'm not a fan of imputation/imputing of missing values yet so missing values are to be removed.

# In[ ]:


df=df.dropna(axis=0,how='any')


# In[ ]:


df.shape


# Unfortunately, the dataset's sample size went down to 56,420 days or rows. 
# 
# Update: I had to redo the narrative of descriptives because Pandas does mess up computation of averages
# in the presence of missing values. I really miss R Studio.
# 
# Lesson learned: remove missing values before descriptives in Pandas because it's not as 
# good as R studio in dealing with missing values in a regression. 
# (I am kidding, for Python/Pandas die-hard fans out there.)

# In[ ]:


#Number of missing values by predictor
df.isnull().sum()


# I have removed the missing values.

# **Association between target variable (rainy next day/tomorrow) and predictor variables**

# 1. Crosstabulations

# 1.1 Rain tomorrow and location

# In[ ]:


ct1=pd.crosstab(df.Location, df.RainTomorrow,normalize='index')
ct1


# In[ ]:


#sorting the percentage of rainy next days by location
ct1.iloc[:,1].sort_values(ascending=False)


# Alice Springs and  Woomera experience the fewest rainy 'tomorrows.' 
# Except Portland, none of the locations experience rainy 'tomorrows' of greater than or equal to 40% of all the days within the 10 years the data covers. 
# 
# This confirms that Australia is typically sunny. Congrats to our Aussie friends for having such a great weather in general.

# 1.2 Rain tomorrow and Wind Gust Direction

# ![](http://)

# In[ ]:


ct2=pd.crosstab(df.WindGustDir,df.RainTomorrow,rownames=['Wind Gust Direction'],colnames=['Rain Tomorrow'], normalize='index')
ct2.plot.bar(stacked=True)


# In[ ]:


#sorting the percentage of rainy next days by Wind Gust
ct2.iloc[:,1].sort_values(ascending=False)


# In the 10 years the data covers, the most common instances when it rains the next day are
# when the current day's wind gust direction is towards west northwest, northwest, west, north northwest and north.

# 1.3 Rain Tomorrow and Rain Today

# In[ ]:


ct3=pd.crosstab(df.RainToday,df.RainTomorrow,rownames=['Rain Today'],colnames=['Rain Tomorrow'], normalize='index')
ct3.plot.bar(stacked=True)


# In[ ]:


ct3


# When it rains within the day, 46% of the time it rains the next day.
# When it does not rain, only 15% of the time it rains the next day.

# 2.Averages of numerical predictors whether it rains the next day or not

# 2.1. Rain tomorrow and minimum temperature

# In[ ]:


avg1 = df.groupby(['RainTomorrow'])['MinTemp'].mean()
avg1.plot.bar()


# In[ ]:


avg1


# A higher minimum temperature during the day is typically associated with rain the next day 
# than a non-rainy day.

# 2.2  Rain tomorrow and maximum temperature

# In[ ]:


avg2 = df.groupby(['RainTomorrow'])['MaxTemp'].mean()
avg2.plot.bar()


# In[ ]:


avg2


# Conversely, a lower maximum temperature during the day is typically associated with rain the next day than a non-rainy day.

# 2.3 Rain tomorrow and Rainfall

# In[ ]:


avg3 = df.groupby(['RainTomorrow'])['Rainfall'].mean()
avg3.plot.bar()


# In[ ]:


avg3


# A higher amount of rainfall within the day is associated with rain the next day. 
# So when it rains,it not only pours, but keeps pouring onto the next day.

# 2.4. Rain tomorrow and Evaporation

# In[ ]:


avg4 = df.groupby(['RainTomorrow'])['Evaporation'].mean()
avg4.plot.bar()


# In[ ]:


avg4


# A more "intense" case of evaporation, on average, within a day is associated with no rain tomorrow.
# This defies my amateur opinion that more intense water evaporation induces rain. Apparently, this data is the Class A Evaporation pan value. Evaporation is measured daily as the depth of water (in inches) evaporates from the pan. So I expected that higher average value/'evaporation' should be associated with rain tomorrow and not the opposite, which what the data provides evidence.

# 2.5. Rain Tomorrow and Sunshine 

# In[ ]:


avg5 = df.groupby(['RainTomorrow'])['Sunshine'].mean()
avg5.plot.bar()


# In[ ]:


avg5


# Typically, when the number of hours of sunshine is shorter during the day, 
# then it's associated more with a rainy tomorrow than a non-rainy one. 
# When it rains the next day, the average hours of sushine is around 4 and half hour during the day.
# When it does not rain next day, the average hours of sunshine is around 8 hours and a half during the day.
# 
# Duh!
# 
# Classic case of data being Captain Obvious. 

# 2.6. Rain Tomorrow and Wind Gust Speed 

# In[ ]:


avg6 = df.groupby(['RainTomorrow'])['WindGustSpeed'].mean()
avg6.plot.bar()


# In[ ]:


avg6


# When it rains the next day, it's associated with a higher average speed (km/h) of the strongest wind gust in the 24 hours to midnight during the day. When it rains the next day, wind gust speed is typically at 47 km/h during the day. When it does not rain, the wind gust speed is at 39 on average.  

# 2.7. Rain Tomorrow and Wind Speed at 9am

# In[ ]:


avg7 = df.groupby(['RainTomorrow'])['WindSpeed9am'].mean()
avg7.plot.bar()


# In[ ]:


avg7


# When wind speed at 9am is typically higher (averaging 17 km/hr) during the day, it rains the next day. When it does not rain the next day, the wind speed averages at 15 km/hr.

# 2.8. Rain Tomorrow and Wind Speed at 3pm

# In[ ]:


avg8 = df.groupby(['RainTomorrow'])['WindSpeed3pm'].mean()
avg8.plot.bar()


# In[ ]:


avg8


# When wind speed at 3pm is typically higher (averaging 21 km/hr) during the day, it rains the next day. When it does not rain the next day, the wind speed averages at 19 km/hr.

# 2.9. Rain Tomorrow and Humidity at 9am

# In[ ]:


avg9 = df.groupby(['RainTomorrow'])['Humidity9am'].mean()
avg9.plot.bar()


# In[ ]:


avg9


# When humidity at 9am is typically higher (averaging 75%) during the day, it rains the next day. When it does not rain the next day, the humidity is typically lower at 63%.

# 2.10. Rain Tomorrow and Humidity at 3pm

# In[ ]:


avg10 = df.groupby(['RainTomorrow'])['Humidity3pm'].mean()
avg10.plot.bar()


# In[ ]:


avg10


# When humidity at 3pm is typically higher (averaging 66%) during the day, it rains the next day. When it does not rain the next day, the humidity is typically lower at 44%.

# 2.11. Rain Tomorrow and Pressure at 9am

# In[ ]:


avg11 = df.groupby(['RainTomorrow'])['Pressure9am'].mean()
avg11.plot.bar()


# In[ ]:


avg11


# There's little difference between the average Atmospheric pressure(hpa), in a given day, at 9am whether it rains or does not the next day. When it rains the next day, the pressure is just a little bit lower based on average.

# 2.12. Rain Tomorrow and Pressure at 3pm

# In[ ]:


avg12 = df.groupby(['RainTomorrow'])['Pressure3pm'].mean()
avg12.plot.bar()


# In[ ]:


avg12


# There's little difference between the average Atmospheric pressure(hpa), in a given day, at 3pm whether it rains or does not the next day. When it rains the next day, the pressure is just a little bit lower on average.

# 2.13. Rain Tomorrow and fraction of sky obscured by cloud at 9am

# In[ ]:


avg13 = df.groupby(['RainTomorrow'])['Cloud9am'].mean()
avg13.plot.bar()


# In[ ]:


avg13


# When it rains the next day, a greater fraction of the sky is obscured by clouds at 9am of a given day.

# 2.14. Rain Tomorrow and fraction of sky obscured by cloud at 3pm

# In[ ]:


avg14 = df.groupby(['RainTomorrow'])['Cloud3pm'].mean()
avg14.plot.bar()


# In[ ]:


avg14


# When it rains the next day, a greater fraction of the sky is obscured by clouds at 3pm of a given day.

# 2.15. Rain Tomorrow and Temperature at 9am

# In[ ]:


avg15 = df.groupby(['RainTomorrow'])['Temp9am'].mean()
avg15.plot.bar()


# In[ ]:


avg15


# 2.16. Rain Tomorrow and Temperature at 3pm

# In[ ]:


avg16 = df.groupby(['RainTomorrow'])['Temp3pm'].mean()
avg16.plot.bar()


# In[ ]:


avg16


# I will not do descriptives on Risk-MM since it should be excluded in the model.
# 
# Luckily, I managed to read the description before reaching this point. 

# **LOGISTIC REGRESSION**
# Logistic regression is used to describe data and to explain the relationship between one target binary variable (whether it rains tomorrow or not) and one or more predictor variables (weather observation factors).

# *** ****Splitting data into training/test data sets**
# * 
# Train/test split and cross validation help to avoid overfitting more than underfitting
# 
# source: https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6
# 
# As someone who took graduate courses in applied statistics, I can say that splitting the dataset is not an established practice in inferential statistics and modelling. It was not discussed to me when I took foundational regression courses. I took a traditional applied stats degree not a data science one.
# 
# This is not a criticism of splitting the dataset. In fact, I also agree that it can help on improving accuracy of out-of-sample predictions. I always have the impression that traditional statistical theory, modelling and regression focus on inference. The approach is making sure estimators are unbiased (findings are realistic) and examining causal relationships, or at least associations, between variables. 
# 
# In short, the focus **is not** on maximizing the accuracy of predictions (minimizing residuals). That's why, I think, splitting dataset gained ground in industry, outside academe, because it's practical and straightforward. People doing analytics need not concern themselves much with distributions and their mathematical proof, but instead, just add predictors or find a more suitable model to minimize prediction errors.

# In[ ]:


#Define target and cols (preidctor variables without Risk_MM)
    #I try to do one-hot encoding for the categorical predictors
    #Damn, I miss R. It was so straightforward to deal with categorical predictors there.
cols=['Location','MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 
      'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 
      'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am',
      'Cloud3pm','Temp9am','Temp3pm','RainToday'] 
X=df[cols]
y=df['RainTomorrow']


# I have seen examples where logistic regression with all predictors is run first then the non-significant predictors (p-values less than or equal to 5%) are removed from the model. Then, the logistic regression is trained and tested.
# 
# I am not removing non-significant variables. I am assuming that all of these predictors are important in analyzing the chance of raining.
# 
# **Clarification on p-values**
# 
# "American Statistical Association states that a p-value, or statistical significance, **does not measure **the size of an effect **or the importance of a result**. The threshold of statistical significance that is commonly used is a p-value of 0.05. This is conventional and arbitrary. It does not convey any meaningful evidence of the size of the effect."
# 
# Source:https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5187603/
# 
# I think it's a pitfall to rely on p-values alone on removing predictors in the analysis. Such practice ignores domain knowledge. I am no meteorologist. I just treat all predictors in the data as important to predicting rain the next day.

# Categorical data is subjected to one-hot encoding
#     This is done to fit all categorical predictors in the regression model.

# In[ ]:


#small x for hot encoded predictors.
x=pd.get_dummies(X)


# **Logistic regression model fitting**
# 
#     This observes the common 70-30 split for training and test data, respectively.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# In[ ]:


logreg = LogisticRegression()
a=logreg.fit(X_train, y_train)


# Predicting the test set results and calculating the accuracy

# In[ ]:


y_pred=a.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# 86% out of 100% (perfect accuracy) is not bad. 
# 
# It's even better than the sample I am following here:
# https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8

# Coefficient signs
# 
# I have to figure out a working code on this.

# In[ ]:


logisticRegr = LogisticRegression()


# In[ ]:


m1=logisticRegr.fit(X_test, y_test)


# In[ ]:


#get list of column names/ names of predictor variables
col=list(x)


# I am trying to make a table of coefficient values and signs for each predictor variable.

# In[ ]:


#table of coefficient signs
coef2= pd.DataFrame(m1.coef_, columns=col)
coef2


# The above table shows the coefficient signs for each predictor.
# 
# To illustrate, Rainfall's value is 0.012 (**positive sign/positive association**) means that greater amount of rainfall within a given day is associated with higher probability of rain the next day. 
# 
# On one hand, Sunshine's value is -0.135 (**negative sign/negative association**) means that more hours of bright sunshine within a given day is associated with lower probability of rain the next day.
# 
# The above illustrations are consistent with the descriptive stats.
# 

# Why is statsmodel not working on Kaggle? I have seen older kernels where it worked just fine.
# 
# I wanted to show the coefficient signs and p-values in a better table.
# 
# I have seen better looking and more complete regression tables with statsmodels.

# In[ ]:


import statsmodels.api as sm


# ROC Curve
# 
# I think I will just update the code when I have time to figure it out.
# 
# I attempted to run a sample code I searched on internet. It did not work.
# It also used statsmodel as well.
# 
# Life is so much simpler with R Studio and caret. Hahaha!

# Thanks for reading!
# 
# I will update this kernel, with codes for ROC and other metrics, when I have time.
# 
