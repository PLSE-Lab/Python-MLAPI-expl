# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

from sklearn import model_selection, preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn import feature_selection

import statsmodels.stats.api as sms
import statsmodels.stats.stattools as sd

from yellowbrick.features import FeatureImportances
from yellowbrick.regressor import ResidualsPlot
from yellowbrick.regressor import PredictionError

from scipy import stats

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

url = os.path.join(dirname, filename)
df = pd.read_csv(url)
orig_df_train = df

df.columns = df.columns.str.replace(' ','_')
df.columns = df.columns.str.replace('/','_')
df.columns = df.columns.str.replace('-','_')

# Shape and type of data
df.shape
df.dtypes

# describes only the numerical columns
df.describe()

# checks if there is any missing data
df.isnull().sum()

# some vizualisation

average_suicide = df.groupby('country').mean().suicides_100k_pop.sort_values(ascending=False).head(15)
x_type = average_suicide.index

plt.figure(figsize=(10,6))
plt.title("Average suicides by country")
sns.barplot(x=x_type, y=average_suicide, order=x_type)
plt.xticks(rotation=45)
plt.ylabel("Average suicides")

average_suicide = df.groupby('sex').mean().suicides_100k_pop.sort_values(ascending=False)
x_type = average_suicide.index

plt.figure(figsize=(10,6))
plt.title("Average suicides by gender")
sns.barplot(x=x_type, y=average_suicide, order=x_type)
plt.xticks(rotation=45)
plt.ylabel("Average suicides")

average_suicide = df.groupby('year').mean().suicides_100k_pop
x_type = average_suicide.index

plt.figure(figsize=(10,6))
plt.title("Average suicides by year")
sns.barplot(x=x_type, y=average_suicide, order=x_type)
plt.xticks(rotation=45)
plt.ylabel("Average suicides")

average_suicide = df.groupby('age').mean().suicides_100k_pop.sort_values(ascending=False)
x_type = average_suicide.index

plt.figure(figsize=(10,6))
plt.title("Average suicides by age")
sns.barplot(x=x_type, y=average_suicide, order=x_type)
plt.xticks(rotation=45)
plt.ylabel("Average suicides")

# removing the columns which are a linear combination of others, also the years as there is
# no relation between the suicide per 100 k and the year. Also the generation which 
# already been represented by the age. We also will get rid of HDI Index as
# 70% of data are missing

df = df.drop(columns=["year","suicides_no","population","country_year","HDI_for_year","_gdp_for_year_($)_","generation"])

# transform qualitative exog variables to dummy variables
df = pd.get_dummies(df,drop_first=True)

X = df.drop(columns=['suicides_100k_pop'])
y = df.suicides_100k_pop

# split the data to train and test

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size = 0.30, random_state=42)

# we will use the random Random Forest

rfr = RandomForestRegressor(random_state=42, n_estimators=100)
rfr.fit(X_train,y_train)
rfr.score(X_test,y_test)
# R² score : 88%
metrics.r2_score(y_test,rfr.predict(X_test))

# Breusch-Pagan test
resid = y_test - rfr.predict(X_test)
hb_rfr = sms.het_breuschpagan(resid, X_test)

labels = ["Lagrange multiplier statistic","p-value","f-value","f p-value"]
for labels, num in zip(labels,hb_rfr):print(f"{labels}:{num:.2}")
# p-value = 0 so the residuals are not homoscedastic

# Durbin-Watson test
dw_rfr = sd.durbin_watson(resid)
print("durbin-Watson test :", dw_rfr)
# the residuals are not auto-correlated

# Normality test : Kolmogorov-Smirnov
kst_rfr = stats.kstest(resid, cdf='norm')
print("Kolmogorov-Smirnov test :", kst_rfr)
# the residuals are not normally distributed

# The 10 most important features
i_feat =[]
importance = []
for col, val in sorted(zip(X.columns,rfr.feature_importances_),key=lambda x: x[1],reverse=True)[:10]:
            print(f"{col:10}{val:10.3f}")
            i_feat.append(col)
            importance.append(val)

plt.figure(figsize=(10,6))
plt.title("Most 10 important features")
sns.barplot(x=i_feat, y=importance, order=i_feat)
plt.xticks(rotation=45)
plt.ylabel("importance in %")

# Diagram of residuals

fig, ax = plt.subplots(figsize=(6,4))
rpv = ResidualsPlot(rfr)
rpv.fit(X_train,y_train)
rpv.score(X_test,y_test)
rpv.show()

# Histogram of Residuals

fig, ax = plt.subplots(figsize=(6,4))
pd.Series(resid, name="residuals").plot.hist(bins=150,ax=ax,title="Residual Histogram")

# Normality of residuals

fig, ax = plt.subplots(figsize=(6,4))
norm = stats.probplot(resid, plot=ax)

# Prediction Error chart

fig, ax = plt.subplots(figsize=(6,6))
pev = PredictionError(rfr)
pev.fit(X_train,y_train)
pev.score(X_test,y_test)
pev.show()
