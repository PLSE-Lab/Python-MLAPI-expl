#!/usr/bin/env python
# coding: utf-8

# Hi all, I just started learning data science and this is my first personal project.
# 
# I'd like to ask if there is a way to apply linear regression for each country on the FacetGrid 'Number of suicides over the years for each country'.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
regr = LinearRegression()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
'''Importing csv files into data frame'''
df = pd.read_csv('../input/who-suicide-statistics/who_suicide_statistics.csv')
df.fillna(0, inplace=True)

hpdf = pd.read_csv('../input/world-happiness/2015.csv')

print(df.head())
print(hpdf.head)

# Any results you write to the current directory are saved as output.


# In[ ]:


'''Total Suicides per year'''

df.groupby('year').suicides_no.sum().plot(kind = 'barh', figsize = (8,8), color = 'salmon')
plt.title('Total Suicides per Year')


# Questions:
# 1. Why is there a significant dip in 2016?

# In[ ]:


df.groupby('country').suicides_no.sum().sort_values(ascending = True).plot(kind = 'barh', figsize = (25,25), color = 'salmon')
plt.title('Total Suicides per Country')


# In[ ]:


'''Number of suicides per gender per age range'''
sns.catplot(x = 'sex', y = 'suicides_no', col = 'age', data = df, kind = 'bar')
plt.show() 


# Questions: 
# 1. What are women doing that results in having significantly lower suicide rates than men.
# 2. Why are men aged 35-54 killing themselves?

# In[ ]:


'''Total number of suicdes per year per age group'''

total_df = df.groupby(['year', 'sex', 'age']).suicides_no.sum().reset_index()
sns.set(style = 'darkgrid')
g2 = sns.FacetGrid(total_df, row = 'sex', col = 'age')
g2.map(plt.bar, "year","suicides_no", color = 'salmon')


# In[ ]:


'''Linear regression for deaths over the years'''
df2 = df.groupby('year').suicides_no.sum().reset_index()

'''change into an numpy array'''
X2 = np.array(df2.year)
y2 = np.array(df2.suicides_no)

'''reshape into an 2d array'''
X2=X2.reshape(-1,1)
y2=y2.reshape(-1,1)

plt.scatter(X2,y2, color = 'salmon', edgecolor = 'white')

regr.fit(X2,y2)
print(regr.coef_, regr.intercept_)

y2_predict = regr.predict(X2)
plt.plot(X2, y2_predict, color='salmon')
plt.show()

'''An unfortunate climb of deaths worldwide over the years. 
I believe the numbers for 2016 is unfinished, creating a low outlier'''


# In[ ]:


'''Linear Regression for top 3 countries with highest suicide numbers'''

plt.figure(figsize=(12,8))

'''RUSSIA
Linear regression for Russia'''
rdf = df[df.country == 'Russian Federation']
rdf = rdf.groupby('year').suicides_no.sum().reset_index()
'''change into an numpy array'''
rx = np.array(rdf.year)
ry = np.array(rdf.suicides_no)
'''reshape into an 2d array'''
rx=rx.reshape(-1,1)
ry=ry.reshape(-1,1)

plt.scatter(rx,ry, color = 'salmon', edgecolor = 'white')

regr.fit(rx,ry)
print(regr.coef_, regr.intercept_)

ry_predict = regr.predict(rx)
plt.plot(rx, ry_predict, color='salmon')


'''AMERICA
Linear regression for America'''
udf = df[df.country == 'United States of America']
udf = udf.groupby('year').suicides_no.sum().reset_index()
'''change into an numpy array'''
ux = np.array(udf.year)
uy = np.array(udf.suicides_no)
'''reshape into an 2d array'''
ux=ux.reshape(-1,1)
uy=uy.reshape(-1,1)

plt.scatter(ux,uy, color = 'teal', edgecolor = 'white')

regr.fit(ux,uy)
print(regr.coef_, regr.intercept_)

uy_predict = regr.predict(ux)
plt.plot(ux, uy_predict, color='teal')

'''JAPAN
Linear regression for JAPAN'''
jdf = df[df.country == 'Japan']
jdf = jdf.groupby('year').suicides_no.sum().reset_index()
'''change into an numpy array'''
jx = np.array(jdf.year)
jy = np.array(jdf.suicides_no)
'''reshape into an 2d array'''
jx=jx.reshape(-1,1)
jy=jy.reshape(-1,1)

plt.scatter(jx,jy, color = 'orchid', edgecolor = 'white')

regr.fit(jx,jy)
print(regr.coef_, regr.intercept_)

jy_predict = regr.predict(jx)
plt.plot(jx, jy_predict, color='orchid')
plt.legend(['Russia Slope', 'America Slope', 'Japan Slope', 'Russia', 'America', 'Japan'], loc = 1)
plt.show()

'''Based on the linear regression:
A Slight decline in deaths for Russia
An ascending slope for America
A steeper rise for Japan'''


# Questions:
# 1. How was the living conditions, political state in Russia during the 1990s
# 2. What caused the sudden increase in suicides in America after 2005

# In[ ]:


'''Number of suicides over the years for each country'''
df3 = df.groupby(['country', 'year']).suicides_no.sum().reset_index()
sns.set(style = 'darkgrid')
g2 = sns.FacetGrid(df3, col = 'country', col_wrap = 4)
g2.map(plt.plot, "year","suicides_no", marker = '.', color = 'salmon')


# In[ ]:


#Create a data frame by joining happiness and suicide.
#delete countries that dont exist in both tables
#create a graph with suicide as y and happiness as x

sns.catplot(x = 'Happiness Score', y = 'Country', data= hpdf, kind = 'bar', height=30, aspect=1)
plt.show()

'''Happiness by region'''
#region = hpdf.groupby('Region').mean().sort_values('Happiness Score', ascending = False)
#print(region)
sns.catplot(x = 'Region', y = 'Happiness Score', data= hpdf, kind = 'bar', height=6, aspect=4)
plt.yticks(range(0,9))
plt.show()

#join hapiness and suicide together in a df
#create a catplot that groups suicide rates together

'''Get the mean of suicides for past 10 years'''
df4 = df[(df.year > 2005)]
df4 = df4.groupby('country').suicides_no.mean().reset_index()

'''Change names of df4 to match hpdf2'''
df4.country = df4.country.replace({"Hong Kong SAR": "Hong Kong", 
                                  "Iran (Islamic Rep of)": "Iran", 
                                  "Occupied Palestinian": "Palestinian Territories",
                                  "Republic of Korea": "South Korea",
                                 "Russian Federation": "Russia",
                                 "Syrian Arab Republic":"Syria",
                                 "United States of America": "United States",
                                 "Venezuela (Bolivarian Republic of)":"Venezuela"})
#print(df4)
hpdf2 = hpdf.sort_values('Country', ascending = True)
#print(hpdf2)

merge = pd.merge(df4, hpdf2, left_on = 'country', right_on='Country', how = 'outer')
#print(merge)
'''drop all rows with nulls'''
merge = merge.dropna()
#print(merge.head())

sns.catplot(x = 'Region', y = 'suicides_no', data = merge, kind = 'bar', height=6, aspect=4)
plt.show()

'''Histogram of happiness score'''
merge = merge.sort_values('Happiness Score', ascending = True)
sns.distplot(merge['Happiness Score'], bins = 20)
plt.show()


# In[ ]:


'''Checking to see if there is any patterns'''

print(merge.head())
'''HAPPINESS Linear Regression'''
plt.figure(figsize=(8,8))
sns.scatterplot(x = merge['Happiness Score'], y = merge.suicides_no)
plt.xticks(rotation=-90)
plt.title('Happiness vs Suicide')

hx = np.array(merge['Happiness Score'])
hy = np.array(merge.suicides_no)
'''reshape into an 2d array'''
hx=hx.reshape(-1,1)
hy=hy.reshape(-1,1)
regr.fit(hx,hy)

hy_predict = regr.predict(hx)
plt.plot(hx, hy_predict, color='salmon')

'''FREEDOM Linear Regression'''
plt.figure(figsize=(8,8))
sns.scatterplot(x = merge['Freedom'], y = merge.suicides_no)
plt.xticks(rotation=-90)
plt.title('Freedom vs Suicide')

fx = np.array(merge.Freedom)
fy = np.array(merge.suicides_no)
'''reshape into an 2d array'''
fx=fx.reshape(-1,1)
fy=fy.reshape(-1,1)
regr.fit(fx,fy)

fy_predict = regr.predict(fx)
plt.plot(fx, fy_predict, color='salmon')

'''TRUST Linear Regression'''
plt.figure(figsize=(8,8))
sns.scatterplot(x = merge['Trust (Government Corruption)'], y = merge.suicides_no)
plt.xticks(rotation=-90)
plt.title('Government Corruption vs Suicide')

tx = np.array(merge['Trust (Government Corruption)'])
ty = np.array(merge.suicides_no)
'''reshape into an 2d array'''
tx=tx.reshape(-1,1)
ty=ty.reshape(-1,1)
regr.fit(tx,ty)

ty_predict = regr.predict(tx)
plt.plot(tx, ty_predict, color='salmon')

'''GENEROSITY Linear Regression'''
plt.figure(figsize=(8,8))
sns.scatterplot(x = merge['Generosity'], y = merge.suicides_no)
plt.xticks(rotation=-90)
plt.title('Generosity vs Suicide')

gx = np.array(merge.Generosity)
gy = np.array(merge.suicides_no)
'''reshape into an 2d array'''
gx=gx.reshape(-1,1)
gy=gy.reshape(-1,1)
regr.fit(gx,gy)

gy_predict = regr.predict(gx)
plt.plot(gx, gy_predict, color='salmon')
plt.show()


'''Controversial, but it seems that using the mean of the past 10 years, the happier with more freedom the country has,the higher the suicide numbers. However, the more trust and generous there is in the country, the lesser the suicide.'''

