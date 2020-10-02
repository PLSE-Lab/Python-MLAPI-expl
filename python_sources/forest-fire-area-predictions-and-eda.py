#!/usr/bin/env python
# coding: utf-8

# References : P. Cortez and A. Morais. A Data Mining Approach to Predict Forest Fires using Meteorological Data. In J. Neves, M. F. Santos and J. Machado Eds., New Trends in Artificial Intelligence, Proceedings of the 13th EPIA 2007 - Portuguese Conference on Artificial Intelligence, December, Guimaraes, Portugal, pp. 512-523, 2007. APPIA, ISBN-13 978-989-95618-0-9.(http://www3.dsi.uminho.pt/pcortez/fires.pdf)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Loading the Dataset

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
df_forest = pd.read_csv("/kaggle/input/forest-fire-area/forestfires.csv")
df_forest.head()


# ## Exploratory Data Analysis

# In[ ]:


df_forest_ = df_forest.copy()
df_forest_['area'] = np.log(df_forest_['area'] + 1)


# In[ ]:


plt.rcParams['figure.figsize'] = [10, 10]
sns.heatmap(df_forest.corr(), annot = True)


# In[ ]:


plt.rcParams['figure.figsize'] = [35,12]
month_temp = sns.barplot(x = 'month', y = 'temp', data = df_forest);
month_temp.axes.set_title("Month Vs Temp Barplot", fontsize = 30)
month_temp.set_xlabel("Months", fontsize = 30)
month_temp.set_ylabel("Temp", fontsize = 30)
month_temp.tick_params(labelsize = 22)
month_temp.legend(fontsize = 20)


# In[ ]:


plt.rcParams['figure.figsize'] = [8, 8]
scat = sns.scatterplot(df_forest['temp'], df_forest['area'])
scat.axes.set_title("Scatter Plot of Area and Temperature", fontsize = 20)
scat.set_xlabel("Temp", fontsize = 18)
scat.set_ylabel("Area", fontsize = 18)
scat.tick_params(labelsize = 12)


# In[ ]:


# After Removing the Skewness
plt.rcParams['figure.figsize'] = [8, 8]
scat = sns.scatterplot(df_forest_['temp'], df_forest_['area'])
scat.axes.set_title("Scatter Plot of Area and Temperature", fontsize = 20)
scat.set_xlabel("Temp", fontsize = 18)
scat.set_ylabel("Area", fontsize = 18)
scat.tick_params(labelsize = 12)


# In[ ]:


plt.rcParams['figure.figsize'] = [9, 10]
area_dist = sns.distplot(df_forest['area']);
area_dist.axes.set_title("Area Distribution", fontsize = 30)
area_dist.set_xlabel("Area", fontsize = 20)
area_dist.tick_params(labelsize = 12)


# In[ ]:


# Reduced skewness
sns.distplot(df_forest_['area'])


# In[ ]:


from scipy.stats import norm

# Generate some data for this demonstration.
data = norm.rvs(df_forest['area'])

# Fit a normal distribution to the data:
mu, std = norm.fit(data)

# Plot the histogram.
plt.hist(data, bins=25, density=True, alpha=0.6, color='g')

# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
plt.title(title)

plt.show()


# In[ ]:


plt.rcParams['figure.figsize'] = [10, 10]
day = sns.countplot(df_forest['day'])
day.tick_params(labelsize = 12)
day.set_xlabel("Day", fontsize = 30)
day.set_ylabel("Count", fontsize = 30)


# ## Preprocessing

# In[ ]:


df_forest['area'] = np.log(df_forest['area'] + 1)


# In[ ]:


df_forest.shape


# In[ ]:


df_forest.max()


# In[ ]:


from sklearn.preprocessing import LabelEncoder

categorical = list(df_forest.select_dtypes(include = ["object"]).columns)
for i, column in enumerate(categorical) :
    label = LabelEncoder()
    df_forest[column] = label.fit_transform(df_forest[column])


# In[ ]:


df_forest['day'].value_counts()


# In[ ]:


df_forest.head()


# In[ ]:


outcome = df_forest['area']
features = df_forest.drop(columns = 'area')


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, outcome, test_size = 0.15, random_state = 196)


# In[ ]:


from sklearn.metrics import r2_score


# In[ ]:


X_train.head()


# ## MODELLING
Linear Regression
# In[ ]:


model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, predictions)


# In[ ]:


r2_score(y_test, predictions)


# Polynomial Regression

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(4)
poly_X_train = poly.fit_transform(X_train)
poly_X_test = poly.fit_transform(X_test)


# In[ ]:


poly_X_test


# In[ ]:


model_2 = LinearRegression()
model_2.fit(poly_X_train, y_train)
predictions_poly = model_2.predict(poly_X_test)
mean_squared_error(y_test, predictions_poly)


# In[ ]:


r2_score(y_test, predictions_poly)


# Lasso

# In[ ]:


from sklearn.linear_model import Lasso
model_3 = Lasso(alpha = 100, max_iter = 10000) 
model_3.fit(X_train, y_train)

prediction = model_3.predict(X_test)
mean_squared_error(y_test, prediction)


# In[ ]:


r2_score(y_test, prediction)

Ridge
# In[ ]:


from sklearn.linear_model import Ridge
model_5 = Ridge(alpha = 500)
model_5.fit(X_train, y_train)

pred = model_5.predict(X_test)
mean_squared_error(y_test, pred)


# In[ ]:


r2_score(y_test, pred)


# ElasticNet

# In[ ]:


from sklearn.linear_model import ElasticNet
model_6 = ElasticNet(alpha = 100, max_iter = 10000)
model_6.fit(X_train, y_train)

pred1 = model_6.predict(X_test)
mean_squared_error(y_test, pred1)


# In[ ]:


r2_score(y_test, pred1)


# In[ ]:


df_forest.head()

SVR
# In[ ]:


from sklearn.svm import SVR
model_4 = SVR(C = 100, kernel = 'linear')
model_4.fit(X_train, y_train)


# In[ ]:


prediction = model_4.predict(X_test)
mean_squared_error(y_test, prediction)


# In[ ]:


r2_score(y_test, prediction)


# In[ ]:


prediction = np.exp(prediction - 1)


# In[ ]:


prediction


# In[ ]:


X_test.iloc[0:2,:]


# In[ ]:


from matplotlib.patches import Circle
import cv2

img = plt.imread("/kaggle/input/forest-fire-area/Forest Fire Area.JPG")

x = np.array([620])
Y = np.array([275])

# Create a figure. Equal aspect so circles look circular
fig,ax = plt.subplots(1)
ax.set_aspect('equal')

# Show the image
ax.imshow(img)

# Now, loop through coord arrays, and create a circle at each x,y pair
for xx,YY in zip(x,Y):
    circ = Circle((xx,YY),145, fill=False)
    ax.add_patch(circ)

# Show the image

plt.show()


# In[ ]:


img = plt.imread("/kaggle/input/forest-fire-area/Forest Fire Area.JPG")

x = np.array([550])
Y = np.array([185])

# Create a figure. Equal aspect so circles look circular
fig,ax = plt.subplots(1)
ax.set_aspect('equal')

# Show the image
ax.imshow(img)

# Now, loop through coord arrays, and create a circle at each x,y pair
for xx,YY in zip(x,Y):
    circ = Circle((xx,YY),50, fill=False)
    ax.add_patch(circ)

# Show the image

plt.show()

