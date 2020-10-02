#!/usr/bin/env python
# coding: utf-8

# ## Data Set Description - World Happiness and Corruption ##
# 
# The first data set contains data on the total amount of happiness a country has across 10 regions with 157 countries in total. This set of information also has various other factors which may or may not correlate with the total happiness score of each country. The second data set contains data on the perceived level of corruption for every country in the world. With this the following questions can be asked:
# 
#  1. Is there a correlation between the Happiness Score and the Corruption level of a country?
#  2. Is there a correlation between the Happiness Score and the Global Country Risk Rating of a country?
#  3. Can a Regression Line or Prediction be made from the two correlations above?

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import math
import matplotlib.pyplot as plt
import numpy as np # linear algebra
from sklearn import datasets, linear_model
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data_set16 = pd.read_csv("../input/world-happiness/2016.csv")
data_setCI = pd.read_csv("../input/corruption-index/history.csv")
data_setRR = pd.read_csv("../input/corruption-index/index.csv")


# In[ ]:


country_setHP = data_set16.loc[data_set16['Country'].isin(data_setCI['Country'])]
country_setCP = data_setCI.loc[data_setCI['Country'].isin(country_setHP['Country'])]
country_setRR = data_setRR.loc[data_setRR['Country'].isin(country_setHP['Country'])]


# In[ ]:


country_setHP


# In[ ]:


country_setCP


# In[ ]:


country_setCP


# In[ ]:


country_setHP = country_setHP.sort_values('Country')
country_setCP = country_setCP.sort_values('Country')
country_setRR = country_setRR.sort_values('Country')


# ## Happiness and Corruption Index ##

# In[ ]:


x = country_setHP['Happiness Score'].values.reshape(146, 1)
y = country_setCP['CPI 2016 Score'].values.reshape(146, 1)

regr = linear_model.LinearRegression()
regr.fit(x, y)

plt.scatter(country_setHP['Happiness Score'], country_setCP['CPI 2016 Score'])
plt.plot(x, regr.predict(x), color = 'red')
plt.xlabel('Happiness Score')
plt.ylabel('Corruption Perception Index')
plt.title('Correlation between Happiness Score and CPI in 2016')

print('Pearson\'s R:')
country_setHP['Happiness Score'].corr(country_setCP['CPI 2016 Score'])


# ## Findings ##
# From the graph showed above, we can see that the values for Happiness Score and Corruption Index for each country is very scattered but they do somewhat retain a diagonal line upwards that looks like the predicted regression line. What could be concluded from this is that Happiness Score and Corruption Index are strongly correlated to each other especially with a Pearson's R value at around 0.9, aside from that the regression line created seems to be in line with the shape of the plotted values meaning that being able to predict the next set of values is highly possible.

# ## Happiness Score and Global Country Risk Rating ##

# In[ ]:


x = country_setHP['Happiness Score'].values.reshape(146, 1)
y = country_setRR['Global Insight Country Risk Ratings'].values.reshape(146, 1)

regr = linear_model.LinearRegression()
regr.fit(x, y)

plt.scatter(country_setHP['Happiness Score'], country_setRR['Global Insight Country Risk Ratings'])
plt.plot(x, regr.predict(x), color = 'red')
plt.xlabel('Happiness Score')
plt.ylabel('Global Risk Ratings')
plt.title('Correlation between Happiness Score and Global Risk Ratings in 2016')

print('Pearson\'s R:')
#print('No Relationship')
country_setHP['Happiness Score'].corr(country_setRR['Global Insight Country Risk Ratings'])


# ## Findings ##
# From the graph showed above, we can see that the values for Happiness Score and Global Risk Rating for each country is very scattered but athe same time they are strangely ordered while retaining a somehat diagonal line upwards that looks like the predicted regression line. What could be concluded from this is that Happiness Score and Global Risk Rating are strongly correlated to each other especially with a Pearson's R value at around 0.9, aside from that the regression line created seems to be in line with the shape of the plotted values meaning that being able to predict the next set of values is highly possible though there is noticeably a larger margin of error when compared to the shape of this graph compared to the previous graph of Happiness Score and Corruption Index.

# ## Conclusions##
# 
# From the above, the following conclusions can be made:
# 
#  1. Happiness and Corruption Index appears to be strongly correlated to each other in that as the happiness score of a country goes up, the higher the corruption index is.
#  2. Happiness and Global Risk Rating appers to be somewhat correlated to each other in that as the happiness score of a country goes up, the higher the risk rating is.
#  3. Following this trend, any future values can be expeted to have a higher value than previous ones.
# 
#  
