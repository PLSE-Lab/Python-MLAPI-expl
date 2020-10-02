#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
pd.set_option('display.precision',10)
from matplotlib import rcParams
import scipy.stats as stats
from matplotlib.pyplot import figure
rcParams['figure.figsize'] = (15,5)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


dataset = pd.read_csv('../input/insurance/insurance.csv')
dataset.head()


# In[ ]:


dataset.isnull().sum()


# No null values are there, we're good to go

# In[ ]:


dataset['sexx'] = dataset['sex'].replace({'male':1,'female':0})
dataset['smokerx'] = dataset['smoker'].replace({'yes':1 , 'no':0})
dataset.head()


# In[ ]:


dataset.describe()


# In[ ]:


correlation_matrix = dataset.corr()
sns.heatmap(correlation_matrix , annot=correlation_matrix )
plt.show()


# In[ ]:


# Stratified Age group
dataset['agegrp'] = pd.cut(dataset.age , [17,30,40,50,60,70])


# In[ ]:


sns.boxplot(dataset.bmi)


# In[ ]:


dataset.groupby(['agegrp','smoker'])['charges'].sum().unstack()


# In[ ]:


ax = plt.subplot(121)
sns.distplot(dataset[dataset.smokerx==1]['charges'] , ax = ax , color = 'c').set_title('Distribution of charges for Smokers')

ax = plt.subplot(122)
sns.distplot(dataset[dataset.smokerx == 0]['charges'] , ax = ax , color = 'r').set_title('Distribution of charges for non smokers')


# In[ ]:


plt.figure(figsize = (15,7))
sns.boxplot(y = 'charges' , x = 'smoker' , hue = 'agegrp' , data = dataset)


# In[ ]:


plt.figure(figsize = (12,5))
sns.countplot(x = 'smoker' , hue = 'sex'  , data = dataset)


# In[ ]:


sns.countplot(x = 'agegrp' , hue = 'smoker' , data = dataset )


# In[ ]:


dx = dataset.groupby(['agegrp','smoker'])['charges'].sum().unstack().reset_index()
dx


# In[ ]:


dx.columns = ['agegrp','Non Smoker' , 'Smoker']
dx


# In[ ]:


sns.barplot(x = 'agegrp' , y = 'Smoker'  , data = dx).set(title = 'Charges according to age group for smoker')
plt.show()


# In[ ]:


sns.barplot(x = 'agegrp' , y = 'Non Smoker'  , data = dx).set(title = 'Charges according to age group for non smoker')
plt.show()


# In[ ]:


plt.figure()
sns.boxplot(x = 'agegrp' , y = 'charges'  , hue = 'smoker' , data = dataset).set(title = 'Charges according to age groups stratified by Smoker')


# In[ ]:


_ = sns.jointplot(x = 'age' , y = 'charges'  , data = dataset[dataset.smokerx == 1] , kind = 'kde').annotate(stats.pearsonr)


# In[ ]:


# For non smokers, we can clearly see the correlation of charges is higher in this case
g = sns.jointplot(x = 'age' , y = 'charges'  , data = dataset[dataset.smokerx == 0] , kind = 'kde').annotate(stats.pearsonr)


# In[ ]:


plt.figure(figsize = (15,5))
sns.lmplot(x = 'age'  , y = 'charges' , hue = 'smoker' , data = dataset , palette='gnuplot2_r' , markers = ['x','o'] , size= 8).set(title = 'Plot for smokers and non smokers')
plt.show()


# ### Now Let's look at region

# In[ ]:


dataset.region.value_counts()


# In[ ]:


dx = dataset.groupby(['region','smoker'])['charges'].mean().unstack().reset_index()
dx.columns = ['region','non smoker' , 'smoker']
dx


# These below results shows the charges for smoker and non smoker in different regions

# In[ ]:


sns.barplot('region', 'non smoker' , data = dx)


# In[ ]:


sns.barplot('region', 'smoker' , data = dx).set(title = 'Charges for smoker in different regions')


# In[ ]:


sns.catplot(x = 'region' , y = 'charges' , hue = 'smoker' , data = dataset , kind  = 'bar')


# In[ ]:


sns.distplot(dataset.bmi).set(title = 'Distribution of Body Mass Index')


# In[ ]:


sns.boxplot(dataset.bmi).set(title = 'BoxPlot for BMI' , xlabel = 'BMI')


# In[ ]:


sns.lmplot(x = 'bmi' , y = 'charges' , hue = 'smoker' , data = dataset)


# In[ ]:


sns.jointplot(x = 'bmi' , y = 'charges' , data = dataset , kind = 'scatter').annotate(stats.pearsonr)


# As we can see for smokers, bmi has very good correlation with charges

# In[ ]:


dx = dataset[dataset.smokerx == 1]
sns.jointplot(x = 'bmi' , y = 'charges' , data = dx , kind = 'scatter').annotate(stats.pearsonr)


# In[ ]:


dx = dataset[dataset.smokerx == 0]
sns.jointplot(x = 'bmi' , y = 'charges' , data = dx , kind = 'scatter').annotate(stats.pearsonr)


# In[ ]:


sns.boxplot(x = 'agegrp' , y = 'bmi' , hue = 'smoker' , data = dataset)


# In[ ]:


dx = dataset.groupby(['sex'])['charges'].sum().reset_index()
dx.columns = ['sex','charges']
sns.catplot(x = 'sex' , y = 'charges' , data = dx , kind = 'bar')


# In[ ]:


sns.catplot(x = 'children' , kind = 'count' , data = dataset)


# In[ ]:


sns.catplot(x = 'children' , y = 'charges' , data = dataset , kind = 'bar').set(title = 'Barplot for charges for number of children')


# In[ ]:


dx = dataset.groupby(['children','smoker'])['charges'].mean().unstack().reset_index()
dx.columns = ['children' , 'no smoker' , 'smoker']
dx


# In[ ]:


sns.catplot(x = 'children' , y = 'charges' , hue = 'smoker' , data = dataset , kind = 'bar')


# In[ ]:


dx = dataset.groupby(['children','smoker'])['bmi'].mean().unstack().reset_index()
dx.columns = ['children' , 'no smoker' , 'smoker']
dx


# In[ ]:


sns.catplot(x = 'children' , y = 'bmi' , hue = 'smoker' , data = dataset , kind = 'bar')


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error , r2_score , accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures ,LabelBinarizer


# In[ ]:


encoder = LabelBinarizer()
c = encoder.fit_transform(dataset.region)
c = pd.DataFrame(c)
c.columns = ['NorthEast' , 'NorthWest' , 'SouthEast' , 'SouthWest']
c.head()

dataset = dataset.join(c)
dataset.head()


# In[ ]:


X = dataset.loc[:,['age','sexx','bmi','children','smokerx','NorthEast','NorthWest','SouthEast','SouthWest']].values
y = dataset.charges.values


# In[ ]:


X_train , X_test , y_train , y_test = train_test_split(X , y , train_size = 0.8 , random_state = 42 , shuffle = True)


# In[ ]:


lin_reg = LinearRegression()
lin_reg.fit(X_train , y_train)
y_pred = lin_reg.predict(X_test)

lin_reg.score(X_test , y_test)


# In[ ]:


pol_reg = PolynomialFeatures(degree = 2)
X_pol = pol_reg.fit_transform(X)

X_train , X_test , y_train , y_test = train_test_split(X_pol , y , train_size = 0.8 , random_state = 42 , shuffle = True)

lin_reg.fit(X_train,y_train)

lin_reg.score(X_test , y_test)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

random_forest = RandomForestRegressor(n_estimators = 100, criterion = 'mse')

random_forest.fit(X_train , y_train)

forest_train_pred = random_forest.predict(X_train)
forest_test_pred = random_forest.predict(X_test)
(random_forest.score(X_train , y_train) ,random_forest.score(X_test , y_test))

print('MSE train data: %.3f , MSE test data: %.3f' %(mean_squared_error(y_train , forest_train_pred) , 
                                                     mean_squared_error(y_test , forest_test_pred)))


print('R2 train data: %.3f, R2 test data: %.3f' % (
r2_score(y_train,forest_train_pred),
r2_score(y_test,forest_test_pred)))


# In[ ]:


plt.scatter(forest_train_pred , forest_train_pred - y_train , c = 'black' , label = 'Train Data')

plt.scatter(forest_test_pred , forest_test_pred - y_test , c = 'c' , label = 'Test Data')

plt.xlabel('Predicted values')
plt.ylabel('Tailings')
plt.legend(loc = 'upper left')
plt.hlines(y = 0, xmin = 0, xmax = 60000, lw = 2, color = 'red')
plt.show()


# In[ ]:




