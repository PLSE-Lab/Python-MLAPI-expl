# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# import libraries from repositories (install if needed to proceed)


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Input data files are available in the "../input/" directory.

# Load Airbnb data file from location in your library (change next line to the right directory)
df = pd.read_csv("../input/seattle/seattle.csv")
#name columns inside the file for reference within the module
df.columns = ['RMid', 'HOSTid', 'RMtype', 'REVIEW', 'SAT',
              'PAX', 'RM', 'BATH', 'PRICE', 'LAT', 'LONG', 'LOC',
             'NAME', 'CURR', 'RATE']

#define columns of importance for analysis
cols = ['PAX', 'RM', 'PRICE']
print('AIRBNB SEATTLE - comparison of variables')
print('_'*40)

#show descriptive statistics from important columns
print(df[cols].describe(exclude='O',percentiles=[.05, .25, .5, .75, .95]).round(2))
print('-'*40)

# define basic graphical funcional form
def lin_regplot(X, y, model):
    plt.scatter(X, y, c='lightblue')
    plt.plot(X, model.predict(X), color='red', linewidth=2)
    return

# plot details analytics of defined important columns
sns.set(style='whitegrid', context='notebook')
sns.pairplot(df[cols], height=2.5)
plt.tight_layout()
plt.show()
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
            cbar=True,
            annot=True,
            square=True,
            fmt='.2f',
            annot_kws={'size': 20},
            yticklabels=cols,
            xticklabels=cols,
            linewidths=.5)
b, t = plt.ylim()
b += 0.5
t -= 0.5
plt.ylim(b, t)
plt.tight_layout()
plt.show()
sns.reset_defaults()

# *define variables to test the effect of “number of rooms” (RM) variable,
# on the mean short-term "rental price" (PRICE) derived from the asset. *
X = df[['RM']].values
y = df['PRICE'].values

# *define our linear regresssion and plot*
slr = LinearRegression()
slr.fit(X, y)
y_pred = slr.predict(X)
print('RAW Data Results')
print('ML Slope: %.3f' % slr.coef_[0])
print('ML Intercept: %.3f' % slr.intercept_)
print('R^2: %.3f' % (r2_score(y,y_pred)))
plt.figure(1)
lin_regplot(X, y, slr)
plt.xlabel('ML Average number of rooms [RM]')
plt.ylabel('ML Price in $1000\'s [PRICE]')
plt.tight_layout()
print('-'*40)
plt.show()

# measure how good our model is in terms of predictions as learned form test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)

slr.fit(X_train, y_train)
y_train_predLR = slr.predict(X_train)
y_test_predLR = slr.predict(X_test)
#plot the linear regression results
plt.figure(2)
plt.scatter(y_train_predLR,  y_train_predLR - y_train, c='blue', marker='o', label='Training data')
plt.scatter(y_test_predLR,  y_test_predLR - y_test, c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.tight_layout()
plt.show()

#show the R2 for the LINEAR regression model
print('LINEAR REG Results')
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_predLR),
        mean_squared_error(y_test, y_test_predLR)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_predLR),
        r2_score(y_test, y_test_predLR)))
print('-'*40)
plt.show()

#*analyse the data using RANDOM FOREST method*
forest = RandomForestRegressor(n_estimators=100,
                               criterion='mse',
                               random_state=1,
                               n_jobs=-1)
forest.fit(X_train, y_train)
y_train_predRF = forest.predict(X_train)
y_test_predRF = forest.predict(X_test)

#show the R2 for the RANDOM Forest model and plot
print('RANDOM FOREST Results')
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_predRF),
        mean_squared_error(y_test, y_test_predRF)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_predRF),
        r2_score(y_test, y_test_predRF)))
plt.figure(3)
plt.scatter(y_train_predRF,
            y_train_predRF - y_train,
            c='black',
            marker='o',
            s=35,
            alpha=0.5,
            label='Training data')
plt.scatter(y_test_predRF,
            y_test_predRF - y_test,
            c='lightgreen',
            marker='s',
            s=35,
            alpha=0.7,
            label='Test data')
plt.xlabel('RF Predicted values')
plt.ylabel('RF Residuals')
plt.hlines(y=0, xmin=-10, xmax=1200, lw=2, color='red')
plt.tight_layout()

plt.show()
