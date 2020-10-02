#!/usr/bin/env python
# coding: utf-8

# # Cryptocurrency Regression Analysis
# ### With a total market capitalization of over 140 Billion USD, what can we figure out with some of the top coins/tokens?
# <img style="max-width: 20%; height: auto; float: right;" src="https://www.creativefabrica.com/wp-content/uploads/2018/01/Crypto-Currency-Symbols-by-Benjamin-Melville.jpg"></img>

# In[ ]:


# You can pip from within your code!
get_ipython().system('pip install statsmodels')


# In[ ]:


import os # system portability
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # statistical data visualization
from seaborn import pairplot, heatmap
import matplotlib.pyplot as plt # plotting library
from sklearn import linear_model # linear model
from sklearn.linear_model import LinearRegression # linear regression
from sklearn import svm # support vector machine
from statsmodels.api import OLS # ordinary least squares

#housekeeping
sns.set(color_codes=True)
plt.figure(figsize=(16, 6))


# import data   ```pd.read_csv()```<br>shift by one day ``` df.shift(-24) ```
# 

# In[ ]:


crypto_df = pd.read_csv("../input/crypto_prices.csv", index_col=0, parse_dates = True)
crypto_df["target_value"] = crypto_df[["Bitcoin"]].shift(-24)
crypto_df = crypto_df[:-24]


# In[ ]:


crypto_df.head()


# #### Splitting data into test & train

# In[ ]:


crypto_test = crypto_df[crypto_df.index >= pd.to_datetime('2019-03-01', format='%Y-%m-%d')]
crypto_train = crypto_df[crypto_df.index < pd.to_datetime('2019-03-01', format='%Y-%m-%d')]

print("testing data size:\t" + str(len(crypto_test)/len(crypto_df) * 100)[:5] + " %")
print("training data size:\t" + str(len(crypto_train)/len(crypto_df) * 100)[:5] + " %")


# In[ ]:


crypto_train.describe()


# ## Selecting top 10 correlated columns
# Correlation coefficient between target_value column and the other columns

# In[ ]:


# Correlation coefficient between target_value column and the other df columns
correlation = crypto_train.corr()[['target_value']].sort_values(by='target_value', ascending=False)

# Only referencing top 10 correlated columns for regression
corr_columns = correlation.index[2:12]
correlation[:12]


# ## Ordinary least squares linear regression
# Goal: minimizing the sum of square differences between the actual and predicted values.

# In[ ]:


#  Ordinary least squares linear regression

reg = linear_model.LinearRegression()
reg.fit(crypto_train[corr_columns],crypto_train['target_value'])


# In[ ]:


score_df = pd.DataFrame(columns=['train_1','train_2','test_1','test_2','all_1','all_2'])
reg.coef_


# ### Train/Test Distribution

# In[ ]:


# Train/Test Data Distribution
plt.figure(figsize=(16, 6))
s = sns.scatterplot(crypto_train.index,crypto_train['target_value'], color='purple', label="Training Data", alpha=.2)
sns.scatterplot(crypto_test.index,crypto_test['target_value'], color='green', label="Training Data")
s.set(xlim=([crypto_train.index.min(), crypto_test.index.max()]))
s.set_title('Test/Train Split')
s.set_xlabel('Date')
s.set_ylabel('Price')
s.legend()


# ### Training data
# 

# In[ ]:


# Train plot
plt.figure(figsize=(16, 6))
s = sns.scatterplot(x=crypto_train.index, y=crypto_train['target_value'], label='Actual')
s.plot(crypto_train.index, reg.predict(crypto_train[corr_columns]),'red', label='Predicted')
s.set(xlim=([crypto_train.index.min(), crypto_train.index.max()]))
s.set_title('Training Data')
s.set_xlabel('Date')
s.set_ylabel('Price')
s.legend()
# coefficient of determination (R^2)
score_df['train_1'] = [reg.score(crypto_train[corr_columns],crypto_train['target_value'])]
print("Train regression score: " + str(reg.score(crypto_train[corr_columns],crypto_train['target_value'])))


# ### Test data

# In[ ]:


# Test plot
plt.figure(figsize=(16, 6))
s = sns.scatterplot(x=crypto_test.index, y=crypto_test['target_value'], label='Actual')
s.plot(crypto_test.index, reg.predict(crypto_test[corr_columns]),'red', label='Predicted')
s.set(xlim=([crypto_test.index.min(), crypto_test.index.max()]))
s.set_title('Testing Data')
s.set_xlabel('Date')
s.set_ylabel('Price')
s.legend()
score_df['test_1'] = [reg.score(crypto_test[corr_columns],crypto_test['target_value'])]


print("Test regression score: " + str(reg.score(crypto_test[corr_columns],crypto_test['target_value'])))


# In[ ]:


#reg.summary()


# ### All data

# In[ ]:


# Test & Train data
plt.figure(figsize=(16, 6))
s = sns.scatterplot(x=crypto_df.index, y=crypto_df['target_value'], label='Actual')
s.plot(crypto_df.index, reg.predict(crypto_df[corr_columns]),'red', label='Predicted')
s.set(xlim=([crypto_df.index.min(), crypto_df.index.max()]))
s.set_title('All Data')
s.set_xlabel('Date')
s.set_ylabel('Price')
s.legend()
score_df['all_1'] = [reg.score(crypto_df[corr_columns],crypto_df['target_value'])]


print("All data regression score: " + str(reg.score(crypto_df[corr_columns],crypto_df['target_value'])))


# ## Further restricting correlated columns
# Selecting only columns with correlation coef greater than a set threshold and removing overfits
# 

# In[ ]:


# Selecting only columns with correlation coef greater than .3
correlation
corr_columns = correlation[correlation.target_value > .3]
corr_columns = corr_columns[~corr_columns.index.str.contains('Bitcoin')]
corr_columns = corr_columns[~corr_columns.index.str.contains('target_value')]
corr_columns = corr_columns.index
corr_columns


# In[ ]:


reg = linear_model.LinearRegression()
reg.fit(crypto_train[corr_columns],crypto_train['target_value'])
reg.coef_


# ### Training data with high correlation coefficients

# In[ ]:


# Train data with high correlation coef
plt.figure(figsize=(16, 6))
s = sns.scatterplot(x=crypto_train.index, y=crypto_train['target_value'], label='Actual')
s.plot(crypto_train.index, reg.predict(crypto_train[corr_columns]),'red', label='Predicted')
s.set(xlim=([crypto_train.index.min(), crypto_train.index.max()]))
s.set_title('Training Data w/ High Corr Coef')
s.set_xlabel('Date')
s.set_ylabel('Price')
s.legend()

score_df['train_2'] = [reg.score(crypto_train[corr_columns],crypto_train['target_value'])]
print("Training regression score: " + str(reg.score(crypto_train[corr_columns],crypto_train['target_value'])))


# ### Testing data with high correlation coefficients

# In[ ]:


# Test data with high correlation coef
plt.figure(figsize=(16, 6))
s = sns.scatterplot(x=crypto_test.index, y=crypto_test['target_value'], label='Actual')
s.plot(crypto_test.index, reg.predict(crypto_test[corr_columns]),'red', label='Predicted')
s.set(xlim=([crypto_test.index.min(), crypto_test.index.max()]))
s.set_title('Testing Data w/ High Corr Coef')
s.set_xlabel('Date')
s.set_ylabel('Price')
s.legend()
score_df['test_2'] = [reg.score(crypto_test[corr_columns],crypto_test['target_value'])]

print("Test regression score: " + str(reg.score(crypto_test[corr_columns],crypto_test['target_value'])))


# ### All data with high correlation coefficients

# In[ ]:


# Test & Train data
plt.figure(figsize=(16, 6))
s = sns.scatterplot(x=crypto_df.index, y=crypto_df['target_value'], label='Actual')
s.plot(crypto_df.index, reg.predict(crypto_df[corr_columns]),'red', label='Predicted')
s.set(xlim=([crypto_df.index.min(), crypto_df.index.max()]))
s.set_title('All Data')
s.set_xlabel('Date')
s.set_ylabel('Price')
s.legend()
score_df['all_2'] = [reg.score(crypto_df[corr_columns],crypto_df['target_value'])]
print("All data regression score: " + str(reg.score(crypto_df[corr_columns],crypto_df['target_value'])))


# In[ ]:


score_df


# ## Bulk linear model classifiers
# ### Support vector regression, BayesianRidge, Lasso fit with Least Angle Regression, etc...
# 

# In[ ]:


# collection of linear regression models - regressor algorithms

classifiers = [
    svm.SVR(), # support vector regression
    linear_model.SGDRegressor(),
    linear_model.BayesianRidge(), 
    linear_model.LassoLars(),
    linear_model.ARDRegression(),
    linear_model.PassiveAggressiveRegressor(),
    linear_model.TheilSenRegressor(),
    linear_model.LinearRegression()]


# In[ ]:


trainingData = np.array(crypto_train)
trainingScores = np.array(crypto_train['target_value'])
predictionData = np.array(crypto_df)

df = pd.DataFrame()

for item in classifiers:
    print(item)
    clf = item
    clf.fit(trainingData, trainingScores)
    print(clf.predict(predictionData), '\n')
    df[item] = clf.predict(predictionData)


# In[ ]:





# In[ ]:


df.reset_index(drop=True, inplace=True)
crypto_df.reset_index(drop=True, inplace=True)
results = pd.concat([df,crypto_df['target_value']], axis=1)
results


# ### Correlation between target_value column & other columns in the dataframe

# In[ ]:


# Correlation coefficient between target_value column and the other df columns
comparison = results.corr()[['target_value']].sort_values(by='target_value', ascending=False)


# In[ ]:


print(comparison.iloc[5][0])


# In[ ]:


# final 
a = comparison[comparison['target_value'] < 1]
a[a['target_value'] > 0]


# In[ ]:





# ### Chose PassiveAggressiveRegressor linear model due to strength of fit 

# In[ ]:


reg = linear_model.PassiveAggressiveRegressor()
passiveAgressive = reg.fit(trainingData, trainingScores)
crypto_df['target_prediction'] = passiveAgressive.predict(predictionData)


# ## Final Results
# 

# In[ ]:


# Test & Train data
plt.figure(figsize=(20, 10))
target = np.array(crypto_df['target_value'])
prediction = np.array(crypto_df['target_prediction'])
diff = prediction - target
a = np.mean(diff)
b = '{0:.2f}'.format((a * 100)) + '% Total Error'
sns.set(font_scale=1.5)
s = sns.lineplot(x=crypto_df.index, y=crypto_df['target_prediction'], label='Predicted')
s.plot(crypto_df.index, crypto_df['target_value'], 'red', label='Actual')
s.set(xlim=([crypto_df.index.min(), crypto_df.index.max()]))
s.set_title('Acutal vs Predicted Price with PassiveAggressiveRegressor Modeling (24 Hour)')
s.set_xlabel('Sample Size')
s.set_ylabel('')
s.text(y=15, s=b, x=1100)
s.fill_between(crypto_df.index,crypto_df['target_value'],crypto_df['target_prediction'],color='yellow')
s.legend()


# In[ ]:





# <img src="https://scontent-dfw5-2.cdninstagram.com/vp/225d3257c0e0af2a8ba56cd2453b1f70/5D2BC4C0/t51.2885-15/e35/39981068_1964197956978995_8174854714697973760_n.jpg?_nc_ht=scontent-dfw5-2.cdninstagram.com">

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Additional Regression

# In[ ]:


X = crypto_df.drop('target_value', axis=1)
lm = LinearRegression()

lm.fit(X,crypto_df.target_value)
print("intercept coefficient " + str(lm.intercept_))
print("number of coefficients " + str(lm.coef_))


# In[ ]:


pd.DataFrame(list(zip(X.columns, lm.coef_)), columns = ['variable','estimatedCoefficients'])


# In[ ]:


plt.scatter(crypto_df.target_value,lm.predict(X))
plt.xlabel('actual price $Y_i$')
plt.ylabel('predicted price $\hat{Y}_i$')


# ### Mean squared error

# In[ ]:


# mean squared error
mse = np.mean((crypto_df.target_value - lm.predict(X)) ** 2)
mse


# In[ ]:


# mean squared error
mse_k = np.mean((crypto_df.target_value - lm.predict(X))) ** 2
mse_k


# In[ ]:


z = LinearRegression()
z.fit(X[['price_usd_Litecoin']], crypto_df.target_value)
mse_z = np.mean((crypto_df.target_value - z.predict(X[['price_usd_Litecoin']]))) ** 2
mse_z


# results above confuse me. expected mean squared error of less correlated coefficient to be greater than the mean squared error 

# In[ ]:




