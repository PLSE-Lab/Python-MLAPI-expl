#!/usr/bin/env python
# coding: utf-8

# ![](https://media.giphy.com/media/qKltgF7Aw515K/giphy.gif)

# **The goal of this data exploration is to find trends, if any, between:**
# * Spending and Performance
# * Enrollment and Performance
# * Spending and Enrollment
# 
# *Specifically for the State of California
# To be expanded to other nearby states.*
# ![](https://media.giphy.com/media/ijxKTF6iE4K4M/giphy.gif)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import stats
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels.api as sm
import statsmodels.formula.api as sm
from statsmodels.formula.api import ols
from statsmodels.sandbox.regression.predstd import wls_prediction_std
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import plotly.plotly as py
import plotly.graph_objs as go

# MatPlotlib
from matplotlib import pylab

# Scientific libraries
from scipy.optimize import curve_fit

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Let's first explore the smaller Education data set NAEP States Base and look at the tail end of the set

# In[ ]:


df = pd.read_csv('../input/aggregates/Aggregates/naep_states_base.csv')


# In[ ]:


df.tail()


# We can see above the set contains the following features:
# * Year
# * State
# * Average Score
# * Test Subject
# * Test Year

# Let's find out what the unique Test Subjects and Test Years are from their respective columns.

# Let's select only the state of California

# In[ ]:


df = df[df['STATE'].str.contains("CALIFORNIA")] 
df.TEST_SUBJECT.unique()


# The test subjects are:
# * Mathematics
# * Reading

# In[ ]:


df.TEST_YEAR.unique()


# The test years are:
# * 4
# * 8

# ** From this we now know, this set contains the Test Scores of 4th graders and 8th graders in each state with regards to the average score from test subjects Reading and Mathematics for each year.**

# Below the Test Subjects are stored in their own dataframes.

# In[ ]:


dfread = df[df['TEST_SUBJECT'].str.contains("Reading")] 
dfmath = df[df['TEST_SUBJECT'].str.contains("Mathematics")] 


# Here the Test Subjects are reorganized with the test years 4 and 8.

# In[ ]:


dfread4 = dfread.loc[dfread['TEST_YEAR'] == 4]
dfmath4 = dfmath.loc[dfmath['TEST_YEAR'] == 4] 
dfread8 = dfread.loc[dfread['TEST_YEAR'] == 8]
dfmath8 = dfmath.loc[dfmath['TEST_YEAR'] == 8] 


# Let's just quickly preview Average 4th grade Reading Scores for California

# In[ ]:


ax1 = dfread4.plot.scatter(x='YEAR', y='AVG_SCORE', c='DarkBlue')
ax1.grid()
plt.title('Scatterplot of Reading Scores of 4th graders vs Time for CA Education NAEP Scores')


# **Let's see how the reading scores for both grades in CA performed over the years**

# In[ ]:


fig,ax= plt.subplots()
for n, group in dfread.groupby('TEST_YEAR'):
    group.plot(x='YEAR',y='AVG_SCORE', ax=ax,label=n)
    ax.grid()
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=4, mode="expand", title="Test Year/Grade",borderaxespad=2.)
plt.title('Plot of Reading Scores vs Time for CA Education NAEP Scores')


# Here we model the linearity of the CA 4th Grade Reading scores and it appears a good fit.
# Note that the data points below year 2000 has as good spread of points above and below the linear fit.

# In[ ]:


# ========================
# Model for Original Data
# ========================

# Get the linear models
lm_original = np.polyfit(dfread4.YEAR, dfread4.AVG_SCORE, 1)

# calculate the y values based on the co-efficients from the model
r_x, r_y = zip(*((i, i*lm_original[0] + lm_original[1]) for i in dfread4.YEAR))

# Put in to a data frame, to keep is all nice
lm_original_plot = pd.DataFrame({
'YEAR' : r_x,
'AVG_SCORE' : r_y
})
fig, ax = plt.subplots()
 
# Plot the original data and model
dfread4.plot(kind='scatter', color='Blue', x='YEAR', y='AVG_SCORE', ax=ax, title='Linear Regression of CA 4th Graders Reading Scores')
lm_original_plot.plot(kind='line', color='Red', x='YEAR', y='AVG_SCORE', ax=ax)
ax.grid()
 
plt.show()
	
# Needed to show the plots inline
get_ipython().run_line_magic('matplotlib', 'inline')


# **Projected 4th grader Reading Scores**

# In[ ]:



from sklearn.linear_model import LinearRegression
X = dfread4.YEAR.values.reshape(-1, 1)
y = dfread4.AVG_SCORE.values.reshape(-1, 1) 

model = LinearRegression()
model.fit(X, y)
xs = np.array([2018, 2019, 2020, 2021]).reshape((-1, 1))

y_predict = model.predict(xs)

print('Projected Scores:', y_predict, sep='\n')

fig,ax= plt.subplots()
plt.scatter(xs, y_predict)
ax.grid()
plt.title('Scatterplot of Projected Reading Scores of 4th graders vs Time')


# For the 8th grade Reading Scores, there is definitely not a linearity with respect to the years. 
# Let's try a polynomial fit.

# In[ ]:


# ========================
# Model for Original Data
# ========================

# Get the linear models
lm_original = np.polyfit(dfread8.YEAR, dfread8.AVG_SCORE, 1)

# calculate the y values based on the co-efficients from the model
r_x, r_y = zip(*((i, i*lm_original[0] + lm_original[1]) for i in dfread8.YEAR))

# Put in to a data frame, to keep is all nice
lm_original_plot = pd.DataFrame({
'YEAR' : r_x,
'AVG_SCORE' : r_y
})

fig, ax = plt.subplots()
 
# Plot the original data and model
dfread8.plot(kind='scatter', color='Blue', x='YEAR', y='AVG_SCORE', ax=ax, title='Linear Regression of CA 8th Graders Reading Scores')
lm_original_plot.plot(kind='line', color='Red', x='YEAR', y='AVG_SCORE', ax=ax)
ax.grid()
 
plt.show()
	
# Needed to show the plots inline
get_ipython().run_line_magic('matplotlib', 'inline')


# This Polynomial regression of degree 3 provides an R^2 of 91%.
# Fairly good and much better than a linear regression.

# In[ ]:



x = dfread8.YEAR 
y= dfread8.AVG_SCORE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import operator

# transforming the data to include another axis
x = x[:, np.newaxis]
y = y[:, np.newaxis]

polynomial_features= PolynomialFeatures(degree=3)
x_poly = polynomial_features.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
r2 = r2_score(y,y_poly_pred)
print("RMSE=", rmse)
print("R^2=", r2)

plt.scatter(x, y, s=10)
# sort the values of x before line plot

sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)
x, y_poly_pred = zip(*sorted_zip)
plt.plot(x, y_poly_pred, color='m')
plt.xlabel('Year')
plt.ylabel('Average Score')
plt.title('Polynomial Regression of degree 3 of CA 8th Reading Scores')
plt.grid(True)
plt.show()


xs = np.array([2018, 2019, 2020, 2021]).reshape((-1, 1))
x_ = polynomial_features.fit_transform(xs)
y_predict = model.predict(x_)

print('Projected Scores:', y_predict, sep='\n')

fig,ax= plt.subplots()
plt.scatter(xs, y_predict)
ax.grid()
plt.title('Scatterplot of Projected Reading Scores of 8th graders vs Time')


# **Good Questions lead to New Knowledge**
# Looking at the plot above, one should be asking themselves:
# * Why are there scores missing for the 8th Grade Reading scores where there are scores for 4th grade?
# * Why is there a dip in the reading scores around 2002-2003?

# **Let's see how the Math scores for both grades in CA performed over the years**

# In[ ]:


fig,ax= plt.subplots()
for n, group in dfmath.groupby('TEST_YEAR'):
    group.plot(x='YEAR',y='AVG_SCORE', ax=ax,label=n)
    ax.grid()
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=4, mode="expand", title="Test Year/Grade", borderaxespad=2.)
plt.title('Plot of Math Scores vs Time for CA Education NAEP Scores')


# There is a clear linearity to the 8th graders Math scores as seen below.

# In[ ]:


lm_original = np.polyfit(dfmath8.YEAR, dfmath8.AVG_SCORE, 1)

# calculate the y values based on the co-efficients from the model
r_x, r_y = zip(*((i, i*lm_original[0] + lm_original[1]) for i in dfmath8.YEAR))

# Put in to a data frame, to keep is all nice
lm_original_plot = pd.DataFrame({
'YEAR' : r_x,
'AVG_SCORE' : r_y
})

fig, ax = plt.subplots()
 
# Plot the original data and model
dfmath8.plot(kind='scatter', color='Blue', x='YEAR', y='AVG_SCORE', ax=ax, title='Linear Regression of CA 8th Graders Math Scores')
lm_original_plot.plot(kind='line', color='Red', x='YEAR', y='AVG_SCORE', ax=ax)
ax.grid()
 
plt.show()
	
# Needed to show the plots inline
get_ipython().run_line_magic('matplotlib', 'inline')
X = dfmath8.YEAR.values.reshape(-1, 1)
y = dfmath8.AVG_SCORE.values.reshape(-1, 1) 

model = LinearRegression()
model.fit(X, y)
xs = np.array([2018, 2019, 2020, 2021]).reshape((-1, 1))

y_predict = model.predict(xs)

print('Projected Scores:', y_predict, sep='\n')

fig,ax= plt.subplots()
plt.scatter(xs, y_predict)
ax.grid()
plt.title('Scatterplot of Projected Math Scores of 8th graders vs Time')


# For the 4th grade Math Scores, there is definitely not a linearity with respect to the years. 
# Let's try a polynomial fit.

# In[ ]:


lm_original = np.polyfit(dfmath4.YEAR, dfmath4.AVG_SCORE, 1)

# calculate the y values based on the co-efficients from the model
r_x, r_y = zip(*((i, i*lm_original[0] + lm_original[1]) for i in dfmath4.YEAR))

# Put in to a data frame, to keep is all nice
lm_original_plot = pd.DataFrame({
'YEAR' : r_x,
'AVG_SCORE' : r_y
})

fig, ax = plt.subplots()
 
# Plot the original data and model
dfmath4.plot(kind='scatter', color='Blue', x='YEAR', y='AVG_SCORE', ax=ax, title='Linear Regression of CA 4th Graders Math Scores')
lm_original_plot.plot(kind='line', color='Red', x='YEAR', y='AVG_SCORE', ax=ax)
ax.grid()
 
plt.show()
	
# Needed to show the plots inline
get_ipython().run_line_magic('matplotlib', 'inline')


# This Polynomial regression of degree 3 provides an R^2 of 94%.
# That's pretty good and much better than a linear regression.

# In[ ]:



x = dfmath4.YEAR 
y= dfmath4.AVG_SCORE


# transforming the data to include another axis
x = x[:, np.newaxis]
y = y[:, np.newaxis]

polynomial_features= PolynomialFeatures(degree=3)
x_poly = polynomial_features.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
r2 = r2_score(y,y_poly_pred)
print("RMSE=", rmse)
print("R^2=", r2)

plt.scatter(x, y, s=10)
# sort the values of x before line plot

sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)
x, y_poly_pred = zip(*sorted_zip)
plt.plot(x, y_poly_pred, color='m')
plt.xlabel('Year')
plt.ylabel('Average Score')
plt.title('Polynomial Regression of degree 3 of CA 4th Graders Math Scores')
plt.grid(True)

plt.show()
xs = np.array([2018, 2019, 2020, 2021]).reshape((-1, 1))
x_ = polynomial_features.fit_transform(xs)
y_predict = model.predict(x_)

print('Projected Scores:', y_predict, sep='\n')

fig,ax= plt.subplots()
plt.scatter(xs, y_predict)
ax.grid()
plt.title('Scatterplot of Projected Math Scores of 4th graders vs Time')


# **Let's engineering new features: **
# * **Growth of 4th grade to 8th grade reading scores**
# * **Growth of 4th grade to 8th grade math scores**
# 
# This will show us how well the 4th graders did on scores 4 years later, when they were 8th graders.

# **Let us find out what the unique year values are.**

# In[ ]:


print("Years contained in 8th grade math scores")
dfmath8.YEAR.unique()


# Above are the years contained in the 8th grade math scores dataset.
# Below are the years contained in the 4th grade math scores dataset.
# 
# **NOTE: There are no scores for the 4th graders in 2001, 1999, 1988, or 1986 to compare to for the following years of 8th grade scores: 2005, 2003, 1992, and 1990.**
# 
# This effect will be seen in the new dataframe created to understand how scores grew for those 4th graders 4 years later in 8th grade.

# In[ ]:


print("Years contained in 4th grade math scores")
dfmath4.YEAR.unique()


# In[ ]:



growthCAmath=[]
yr = [2017, 2015, 2013, 2011, 2009, 2007, 2005, 2003, 2000, 1996, 1992,1990]
for y in yr:
    b= y-4
    dfm8 = dfmath8['AVG_SCORE'][(dfmath8['YEAR'] == y)].values
    dfm4 = dfmath4['AVG_SCORE'][(dfmath4['YEAR'] == b)].values
    diff = np.subtract(dfm8,dfm4)
    growthCAmath.append(diff)
    cols=['Growth']
    g = pd.DataFrame(growthCAmath, columns=cols)

g.fillna(0, inplace=True)
year=['Year']
dfy = pd.DataFrame(yr, columns=year) 
CAmath8g = pd.concat([dfy, g], axis=1)
print(CAmath8g)


# **Let's plot this growth rate.**

# In[ ]:


ax1 = CAmath8g.plot.scatter(x='Year', y='Growth', c='DarkBlue')
ax1.grid()
plt.title('Scatterplot of Growth of Math Scores vs Time for CA Education NAEP Scores')


# **Interesting....**
# Growth in math scores for students 4 years later in 8th grade is decreasing slightly over time.
# 
# * Did the state receive less money or more money during the past decade?
# * Did the state enrollment change during the past decade?

# Let's check on growth of Reading scores.

# In[ ]:


print("Years contained in 8th grade reading scores")
dfread8.YEAR.unique()


# Above are the years contained in the 8th grade reading scores dataset.
# Below are the years contained in the 4th grade reading scores dataset.
# 
# **NOTE: There are no scores for the 4th graders in 2001 and 1999 to compare to for the following years of 8th grade scores: 2005 and 2003.**
# 
# This effect will be seen in the new dataframe created to understand how scores grew for those 4th graders 4 years later in 8th grade.

# In[ ]:


print("Years contained in 4th grade reading scores")
dfread4.YEAR.unique()


# In[ ]:


growthCAread=[]
yr = [2017, 2015, 2013, 2011, 2009, 2007, 2005, 2003, 2002, 1998]
for y in yr:
    b= y-4
    dfr8 = dfread8['AVG_SCORE'][(dfread8['YEAR'] == y)].values
    dfr4 = dfread4['AVG_SCORE'][(dfread4['YEAR'] == b)].values
    diff = np.subtract(dfr8,dfr4)
    growthCAread.append(diff)
    cols=['Growth']
    gr = pd.DataFrame(growthCAread, columns=cols)

gr.fillna(0, inplace=True)
year=['Year']
dfyr = pd.DataFrame(yr, columns=year) 
CAread8g = pd.concat([dfyr, gr], axis=1)
print(CAread8g)


# **Let's plot the growth of CA Reading scores**

# In[ ]:


ax1 = CAread8g.plot.scatter(x='Year', y='Growth', c='DarkBlue')
ax1.grid()
plt.title('Scatterplot of Growth of Reading Scores vs Time for CA Education NAEP Scores')


# **Let's take note of the growth**
# The growth of the reading scores are slightly steady but they have some mild oscillations. Let's explore how spending and enrollment may have effected these growth rates.

# **Above was a nice simple exploration of the data for CA test scores from the naep_states_base.csv data set.
# However, let's try to gain insight in spending and enrollment on the state level in regards to education.
# The questions that are being explored are:
# Does increased spending lead to increased scores**

# In[ ]:


dffs = pd.read_csv('../input/states_all.csv')
dffs.head()


# In[ ]:


dffs = dffs[dffs['STATE'].str.contains("CALIFORNIA")] 
dffs.head()


# In[ ]:




x = dffs['YEAR']
y1 = dffs['ENROLL']
y2 = dffs['TOTAL_EXPENDITURE']

plt.show()
fig, axs = plt.subplots(2, 1, constrained_layout=True)
axs[0].plot(x, y1, 'x')
axs[0].set_title('Enrollment vs Year')
axs[0].set_xlabel('Year')
axs[0].set_ylabel('Enrollment')
axs[0].grid()
fig.suptitle('Comparison of Enrollment and Total Revenue over the Years', fontsize=16)

axs[1].plot(x, y2, '--')
axs[1].set_xlabel('Year')
axs[1].set_title('Total Expenditure vs Year')
axs[1].set_ylabel('Total Expenditure')
axs[1].grid()
plt.show()

