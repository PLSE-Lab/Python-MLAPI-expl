#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing


# In[ ]:


# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 500]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 50, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.autoscale
    plt.show()


# In[ ]:


# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
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
        columnNames = columnNames[:27]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()


# In[ ]:


def con_plotScatterMatrix(df, plotSize, textSize):
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


# In[ ]:


nRowsRead = None # specify 'None' if want to read whole file
df1 = pd.read_csv('../input/mlb-team-statistics-20182003/mlbteamstats_20182003.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'mlbteamstats_20182003.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[ ]:


df1.head(20)


# In[ ]:


df1.isnull().sum()


# In[ ]:


plotCorrelationMatrix(df1, 10)


# In[ ]:


plotScatterMatrix(df1, 50, 5)


# In[ ]:


con_plotScatterMatrix(df1, 15, 5)


# In[ ]:


dfDist = df1.drop(columns=["YearTeam"])
dfDist.dataframeName = 'mlbteamstats_20182003.csv'
plotPerColumnDistribution(dfDist, 27, 10)


# In[ ]:


y = df1["W"]
X = df1.drop(columns=["YearTeam", "L", "W-L%",'W',])
X_scaled = preprocessing.scale(X)


# In[ ]:


from sklearn.model_selection import train_test_split
import math
from statistics import mean
from sklearn.linear_model import LinearRegression 
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


x_train1, x_test1, y_train1, y_test1 = train_test_split(X_scaled, y, 
                                                   random_state=12)
lr = LinearRegression().fit(x_train1, y_train1)
y_pred1 = lr.predict(x_test1)


# In[ ]:


print('Average error: %.2f ' % mean(y_test1 - y_pred1))
print('Mean absolute error: %.2f' %mean_absolute_error(y_test1, y_pred1))
print("Root mean squared error: %.2f"
      % math.sqrt(mean_squared_error(y_test1, y_pred1)))
print('percentage absolute error: %.2f' %mean(abs((y_test1 - y_pred1)/y_test1)))
print("R Squared: %f" % r2_score(y_test1,y_pred1))


# In[ ]:


import numpy as np
coefficients = np.column_stack((X.columns, lr.coef_))
print("Coefficients: \n", coefficients)


# In[ ]:


import matplotlib.pyplot as pl
pl.scatter(lr.predict(x_train1), y_train1 - lr.predict(x_train1), c='b') #blue for residuals of training data
pl.scatter(y_test1, y_test1 - y_pred1, c='g') #green for residuals of test data
pl.ylabel('residuals')
pl.xlabel('actual value')
pl.show()


# In[ ]:


print(x_train1.shape)
print(x_test1.shape)
print(y_train1.shape)
print(y_test1.shape)


# In[ ]:


y_def = df1["W"]
X_def = df1.drop(columns=["YearTeam", "L", "W-L%",'W', 'OBP', 'R/G', 'R','H1', 'RBI', 'SB', 'SO 1', 'BA','OBP','SLG','GDP','LOB'
])
x_def_scaled = preprocessing.scale(X_def)
x_train_def, x_test_def, y_train_def, y_test_def = train_test_split(x_def_scaled, y_def, 
                                                   random_state=12)
lr = LinearRegression().fit(x_train_def, y_train_def)

y_train_pred_def = lr.predict(x_train_def)
y_test_pred_def = lr.predict(x_test_def)
print("R Squared: %f" % lr.score(x_test_def,y_test_def))


# In[ ]:


import numpy as np
coefficients = np.column_stack((X_def.columns, lr.coef_))
print("Coefficients: \n", coefficients)


# In[ ]:


import matplotlib.pyplot as pl
pl.scatter(lr.predict(x_train_def), y_train_def - lr.predict(x_train_def), c='b') #blue for residuals of training data
pl.scatter(y_test_def, y_test_def - y_test_pred_def, c='g') #green for residuals of test data
pl.ylabel('residuals')
pl.xlabel('actual value')
pl.show()


# In[ ]:


y_off = df1["W"]
X_off = df1.drop(columns=["YearTeam", "L", "W-L%",'W', 'RA/G','DefEff','E','DP','ERA','tSho','H','ER','HR','BB','SO'])
x_off_scaled = preprocessing.scale(X_off)


x_train_off, x_test_off, y_train_off, y_test_off = train_test_split(x_off_scaled, y_off, 
                                                   random_state=12)
lr = LinearRegression().fit(x_train_off, y_train_off)

y_train_pred_off = lr.predict(x_train_off)
y_test_pred_off = lr.predict(x_test_off)
print("R Squared: %f" % lr.score(x_test_off,y_test_off))


# In[ ]:


import numpy as np
coefficients = np.column_stack((X_off.columns, lr.coef_))
print("Coefficients: \n", coefficients)


# In[ ]:


from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV

svc = SVC(kernel="linear")
rfecv = RFECV(estimator=svc, step=1, cv=KFold(3), scoring = "r2")
rfecv.fit(x_train1, y_train1)
print("Optimal Number of features : %d" % rfecv.n_features_)


# In[ ]:


plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


# In[ ]:


print(rfecv.support_)
print(rfecv.ranking_)
print(rfecv.estimator_.coef_)


# In[ ]:


X_rfecv = df1.drop(columns=["YearTeam","RA/G","DefEff","E","DP","W","L","W-L%","tSho","H","ER","HR","BB","SO","R/G","H1","RBI","SB","SO 1","BA","OBP","SLG","GDP","LOB","salary"
])
X_rfecv_scaled = preprocessing.scale(X_rfecv)
y = df1['W']

x_train, x_test, y_train, y_test = train_test_split(X_rfecv_scaled, y, 
                                                   random_state=12)
lr = LinearRegression().fit(x_train, y_train)
y_pred = lr.predict(x_test)


# In[ ]:


print('Average error: %.2f ' % mean(y_test - y_pred))
print('Mean absolute error: %.2f' %mean_absolute_error(y_test, y_pred))
print("Root mean squared error: %.2f"
      % math.sqrt(mean_squared_error(y_test, y_pred)))
print('percentage absolute error: %.2f' %mean(abs((y_test - y_pred)/y_test)))
print("R Squared: %f" % r2_score(y_test,y_pred))


# In[ ]:


import numpy as np
coefficients = np.column_stack((X_rfecv.columns, lr.coef_))
print("Coefficients: \n", coefficients)


# In[ ]:


x_train1 = sm.add_constant(x_test1)
est = sm.GLS(y_test1, x_train1)
est2 = est.fit()
print(est2.summary())


# In[ ]:


y_2 = df1["W"]
X_2 = df1.drop(columns=["YearTeam","RA/G","DefEff","E","DP",
                                          "W","L","W-L%","tSho","H","ER","HR","BB","SO","R/G","H1",
                                          "RBI","SB","SO 1","BA","OBP","SLG","GDP","LOB","salary"])


# In[ ]:


x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, 
                                                   random_state=12)
X2_remove = sm.add_constant(x_test_2)
est_remove = sm.OLS(y_test_2, X2_remove)
est2_remove = est_remove.fit()
print(est2_remove.summary())


# In[ ]:


fig = plt.figure()
fig.set_figheight(100)
fig.set_figwidth(33)
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for n in range(1,4):
    ax = fig.add_subplot(8,3,n)
    fig = sm.graphics.plot_fit(est2_remove, n , ax=ax)
plt.show()


# In[ ]:


forest = RandomForestRegressor(n_estimators = 100,
                              criterion = 'mse',
                              random_state = 1,
                              n_jobs = -1)
forest.fit(x_train,y_train)
forest_train_pred = forest.predict(x_train)
forest_test_pred = forest.predict(x_test)

print('MSE train data: %.3f, MSE test data: %.3f' % (
mean_squared_error(y_train,forest_train_pred),
mean_squared_error(y_test,forest_test_pred)))
print('R2 train data: %.3f, R2 test data: %.3f' % (
r2_score(y_train,forest_train_pred),
r2_score(y_test,forest_test_pred)))


# In[ ]:


pl.figure(figsize=(10,6))

pl.scatter(forest_train_pred,forest_train_pred - y_train,
          c = 'black', marker = 'o', s = 35, alpha = 0.5,
          label = 'Train data')
pl.scatter(forest_test_pred,forest_test_pred - y_test,
          c = 'c', marker = 'o', s = 35, alpha = 0.7,
          label = 'Test data')
pl.xlabel('Predicted values')
pl.ylabel('Residuals')
pl.legend(loc = 'upper left')
pl.hlines(y = 0, xmin = 40, xmax = 100, lw = 2, color = 'red')
pl.show()

