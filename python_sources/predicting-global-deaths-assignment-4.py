#!/usr/bin/env python
# coding: utf-8

# # ECE 657A - Data and Knowledge Modeling and Analysis 
# # Assignment 4 - Submitted by: Shannon Anthea Dsouza (20771338) and Xuhui Xie (20782765)
# ## Task Submission: Predict Daily Global Deaths
# ## Model for predicting number of deaths by using Support Vector Machine, Linear Regression and Ridge Regression.

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


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import os # accessing directory structure

import numpy as np 
import matplotlib.colors as mcolors
import pandas as pd 
import random
import math
import time
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import operator
plt.style.use('seaborn')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
#     df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            plt.bar(columnDf.index,columnDf.values)
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()


# In[ ]:


# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()


# In[ ]:


# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()


# ## Read in the data and use the plotting functions to visualize the data

# ## Confirmed Global Cases (File 2)

# In[ ]:


nRowsRead = 1000 # specify 'None' if want to read whole file
# time_series_covid19_confirmed_global.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df2 = pd.read_csv('/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_confirmed_global.csv', delimiter=',', nrows = nRowsRead)
df2.dataframeName = 'time_series_covid19_confirmed_global.csv'
nRow, nCol = df2.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[ ]:


df2.head(5)


# In[ ]:


## Distribution graphs (histogram/bar graph) of sampled columns:
plotPerColumnDistribution(df2.iloc[:,-4:], 4, 2)


# In[ ]:


## Correlation matrix:
plotCorrelationMatrix(df2.iloc[:,4:], 15)


# In[ ]:


## Scatter and density plots:
plotScatterMatrix(df2.iloc[:,4:], 20, 10)


# ## Recovered Global Cases (File 3)

# In[ ]:


nRowsRead = 1000 # specify 'None' if want to read whole file
# time_series_covid19_deaths_global.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df3 = pd.read_csv('/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_recovered_global.csv', delimiter=',', nrows = nRowsRead)
df3.dataframeName = 'time_series_covid19_recovered_global.csv'
nRow, nCol = df3.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[ ]:


df3.head(5)


# In[ ]:


## Distribution graphs (histogram/bar graph) of sampled columns:
plotPerColumnDistribution(df3.iloc[:,-4:], 4, 2)


# In[ ]:


## Correlation matrix:
plotCorrelationMatrix(df3.iloc[:,4:], 15)


# In[ ]:


## Scatter and density plots:
plotScatterMatrix(df3.iloc[:,4:], 20, 10)


# ## Global Deaths (File 4)

# In[ ]:


nRowsRead = 1000 # specify 'None' if want to read whole file
# time_series_covid19_deaths_global.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df4 = pd.read_csv('/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_deaths_global.csv', delimiter=',', nrows = nRowsRead)
df4.dataframeName = 'time_series_covid19_deaths_global.csv'
nRow, nCol = df4.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[ ]:


df4.head(5)


# In[ ]:


## Distribution graphs (histogram/bar graph) of sampled columns:
plotPerColumnDistribution(df4.iloc[:,-4:], 4, 2)


# In[ ]:


## Correlation matrix:
plotCorrelationMatrix(df4.iloc[:,4:], 15)


# In[ ]:


## Scatter and density plots:
plotScatterMatrix(df4.iloc[:,4:], 20, 10)


# ## Basic Plotting Code 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot
import seaborn as sns; sns.set()
# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import missingno as msno



# Configure visualisations
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use( 'ggplot' )
plt.style.use('fivethirtyeight')
sns.set(context="notebook", palette="dark", style = 'whitegrid' , color_codes=True)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Any results you write to the current directory are saved as output.


# In[ ]:


filename = "/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_deaths_global.csv"
df = pd.read_csv(filename)
df.head()


# In[ ]:


df.columns


# In[ ]:



df.iloc[:,4:]


# In[ ]:


countries=['Brazil', 'Canada', 'Germany']
y=df.loc[df['Country/Region']=='Italy'].iloc[0,4:]
s = pd.DataFrame({'Italy':y})
for c in countries:    
    #pyplot.plot(range(y.shape[0]),y,'r--')
    s[c] = df.loc[df['Country/Region']==c].iloc[0,4:]
#pyplot.plot(range(y.shape[0]),y,'g-')
pyplot.plot(range(y.shape[0]), s)


# In[ ]:


s.tail(10)


# In[ ]:


for r in df['Country/Region']:
    if r != 'China':
        pyplot.plot(range(len(df.columns)-4), df.loc[df['Country/Region']==r].iloc[0,4:])
#         pyplot.legend()


# In[ ]:


columns = df2.keys()


# In[ ]:


confirmed = df2.loc[:, columns[4]:columns[-1]]
deaths = df4.loc[:, columns[4]:columns[-1]]
recoveries = df3.loc[:, columns[4]:columns[-1]]


# In[ ]:


dates = confirmed.keys()
world_cases = []
total_deaths = [] 
mortality_rate = []
total_recovered = [] 

for i in dates:
    confirmed_sum = confirmed[i].sum()
    death_sum = deaths[i].sum()
    recovered_sum = recoveries[i].sum()
    world_cases.append(confirmed_sum)
    total_deaths.append(death_sum)
    mortality_rate.append(death_sum/confirmed_sum)
    total_recovered.append(recovered_sum)


# In[ ]:


days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
world_cases = np.array(world_cases).reshape(-1, 1)
total_deaths = np.array(total_deaths).reshape(-1, 1)
total_recovered = np.array(total_recovered).reshape(-1, 1)


# ## Predicting the future

# In[ ]:


days_in_future = 15
future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)
adjusted_dates = future_forcast[:-15]


# In[ ]:


##Convert integer into datetime
start = '1/22/2020'
start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
future_forcast_dates = []
for i in range(len(future_forcast)):
    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))


# ## Number of deaths prediction: SVM Model

# In[ ]:


# Split data for model
X_train_deaths, X_test_deaths, y_train_deaths, y_test_deaths = train_test_split(days_since_1_22, total_deaths, test_size=0.15, shuffle=False) 


# In[ ]:


kernel = ['poly', 'sigmoid', 'rbf']
c = [0.01, 0.1, 1, 10]
gamma = [0.01, 0.1, 1]
epsilon = [0.01, 0.1, 1]
shrinking = [True, False]
svm_grid = {'kernel': kernel, 'C': c, 'gamma' : gamma, 'epsilon': epsilon, 'shrinking' : shrinking}

svm = SVR()
svm_search = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=3, n_jobs=-1, return_train_score=True, n_iter=40, verbose=1)
svm_search.fit(X_train_deaths, y_train_deaths)


# In[ ]:


print('Best Params are: ')
svm_search.best_params_


# In[ ]:


svm_deaths = svm_search.best_estimator_
svm_pred_death = svm_deaths.predict(future_forcast)


# In[ ]:


# check against testing data
svm_test_pred = svm_deaths.predict(X_test_deaths)
plt.plot(svm_test_pred)
plt.plot(y_test_deaths)
plt.legend(['Death Cases', 'SVM predictions'])
print('MAE:', mean_absolute_error(svm_test_pred, y_test_deaths))
print('MSE:',mean_squared_error(svm_test_pred, y_test_deaths))


# In[ ]:


##Linear regression model
linear_model = LinearRegression(normalize=True, fit_intercept=True)
linear_model.fit(X_train_deaths, y_train_deaths)
test_linear_pred = linear_model.predict(X_test_deaths)
linear_pred = linear_model.predict(future_forcast)
print('MAE:', mean_absolute_error(test_linear_pred, y_test_deaths))
print('MSE:',mean_squared_error(test_linear_pred, y_test_deaths))


# In[ ]:


print(linear_model.coef_)
print(linear_model.intercept_)


# In[ ]:


plt.plot(y_test_deaths)
plt.plot(test_linear_pred)
plt.legend(['Death Cases', 'Linear Regression predictions'])


# In[ ]:


##Bayesian ridge regression model
tol = [1e-4, 1e-3, 1e-2]
alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4]
alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4]
lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4]
lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4]

bayesian_grid = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2}

bayesian = BayesianRidge()
bayesian_search = RandomizedSearchCV(bayesian, bayesian_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
bayesian_search.fit(X_train_deaths, y_train_deaths)


# In[ ]:


bayesian_search.best_params_


# In[ ]:


bayesian_deaths = bayesian_search.best_estimator_
test_bayesian_pred_deaths = bayesian_deaths.predict(X_test_deaths)
bayesian_pred_deaths = bayesian_deaths.predict(future_forcast)
print('MAE:', mean_absolute_error(test_bayesian_pred_deaths, y_test_deaths))
print('MSE:',mean_squared_error(test_bayesian_pred_deaths, y_test_deaths))


# In[ ]:


plt.plot(y_test_deaths)
plt.plot(test_bayesian_pred_deaths)
plt.legend(['Confirmed Cases', 'Bayesian predictions'])


# In[ ]:


plt.figure(figsize=(10, 7))
plt.plot(adjusted_dates, total_deaths, color='red')
plt.bar(pd.DataFrame(total_deaths).iloc[:,-1].index,pd.DataFrame(total_deaths).iloc[:,-1].values)
plt.title('# of Coronavirus Death Cases Over Time', size=20)
plt.xlabel('Time', size=20)
plt.ylabel('# of Deaths', size=20)
plt.xticks(size=15)
plt.show()


# In[ ]:


plt.figure(figsize=(10, 7))
plt.plot(adjusted_dates, total_deaths, color='red')
plt.plot(future_forcast, svm_pred_death, linestyle='dashed', color='purple')
plt.title('# of Coronavirus Death Cases Over Time', size=20)
plt.xlabel('Days Since 1/22/2020', size=20)
plt.ylabel('# of Cases', size=20)
plt.legend(['Death Cases', 'SVM predictions'])
plt.xticks(size=15)
plt.show()


# In[ ]:


plt.figure(figsize=(10, 7))
plt.plot(adjusted_dates, total_deaths, color='red')
plt.plot(future_forcast, linear_pred, linestyle='dashed', color='orange')
plt.title('# of Coronavirus Death Cases Over Time', size=20)
plt.xlabel('Days Since 1/22/2020', size=20)
plt.ylabel('# of Cases', size=20)
plt.legend(['Death Cases', 'Linear Regression Predictions'])
plt.xticks(size=15)
plt.show()


# In[ ]:


plt.figure(figsize=(10, 7))
plt.plot(adjusted_dates, total_deaths, color='red')
plt.plot(future_forcast, bayesian_pred_deaths, linestyle='dashed', color='green')
plt.title('# of Coronavirus Death Cases Over Time', size=20)
plt.xlabel('Days Since 1/22/2020', size=20)
plt.ylabel('# of Cases', size=20)
plt.legend(['Death Cases', 'Bayesian Ridge Regression Predictions'])
plt.xticks(size=15)
plt.show()


# In[ ]:


# Future predictions using SVM 
print('SVM future predictions:')
set(zip(future_forcast_dates[-14:], svm_pred_death[-14:]))


# In[ ]:


# Future predictions using Bayesian regression
print('Bayesian regression future predictions:')
set(zip(future_forcast_dates[-14:], bayesian_pred_deaths[-14:]))


# In[ ]:


# Future predictions using Linear Regression 
print('Linear regression future predictions:')
print(linear_pred[-14:])


# ## Death and recoveries over time

# In[ ]:


plt.figure(figsize=(10, 7))
plt.plot(adjusted_dates, total_deaths, color='r')
plt.plot(adjusted_dates, total_recovered, color='green')
plt.legend(['Deaths', 'Recoveries'], loc='best', fontsize=20)
plt.title('# of Coronavirus Cases', size=20)
plt.xlabel('Time', size=20)
plt.ylabel('# of Cases', size=20)
plt.xticks(size=15)
plt.show()


# ## References:
# * https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f
# * https://www.kaggle.com/therealcyberlord/coronavirus-covid-19-visualization-prediction
# * https://scikit-learn.org/stable/modules/svm.html
# * https://monkeylearn.com/blog/introduction-to-support-vector-machines-svm/
# * https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html
# * https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Ridge_Regression.pdf
# * https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
# * https://en.wikipedia.org/wiki/Linear_regression
