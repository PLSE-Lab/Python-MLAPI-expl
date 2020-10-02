# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import warnings
warnings.filterwarnings("ignore")
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
import statsmodels.stats.stattools as sd
import statsmodels.api as sm
from yellowbrick.regressor import PredictionError
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# download the data in pandas data

url = os.path.join(dirname, filename)
df = pd.read_csv(url)
# replace the spaces in the names of columns with '_'
df.columns = df.columns.str.replace(' ','_')
orig_df = df
# indicates the shape of the dataframe
df.shape 

# describes only the numerical columns
df.describe()

# checks if there is any missing data
df.isnull().sum() # all the columns&lines are filled: there is no missing data

# transforms any categorial variable to a dummy variable and gets rid of the first variable
# for example : we don't need 2 columns gender, one of them will be enough as leaving the 2 will
# generate biases in the model

df = pd.get_dummies(df,drop_first=True)

# transform the scores of math, reading and righting to %
df.math_score = df.math_score/100
df.reading_score = df.reading_score/100
df.writing_score = df.writing_score/100

# Creating our endo and exog variables : X and y
X = df.drop(columns=["math_score","reading_score","writing_score"])
y_math = df.math_score
y_reading = df.reading_score
y_writing = df.writing_score

# we tried different models and selected the one with the highest R² value:
#Linear Regression R2 : 0.21337015055406705
#SVM R2 : 0.08180476653094604
#KNR R2 : 0.027211950043576127
#D.Tree R2 : -0.07985582444140338
#R.Forest R2 : -0.02005068468331772
#XGBoost R2 : 0.16843858493898944
#LightGBM R2 : 0.10071148438757327

# The Ordinary Least Square Regression offers the higher R²

# Starting the model on the Math results:

# Spliting the data between Train set and Test set

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y_math, test_size = 0.30, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train,y_train)
# residue (errors) analysis
resid = np.array((y_train- lr.predict(X_train)))
mu = np.mean(resid) # mean of the residuals
sigma = np.std(resid)

plt.title("residual histogram")
plt.hist(resid,bins=50)

# and then we start to run tests on the residuals (errors):

# Test 1 : Do the errors have zero mean
z = mu/sigma # z is no different from zero 

# Test 2 : X values are independent from the errors
X_1 = np.array(X_train)
for i in range (len(X_1.transpose())):
    print(round(np.cov(X_1[:,i],resid)[0,1],3))
    # The covariances values are no different from zero

# Test 3 : Homoscedasticity of errors (test Breusch-Pagan)
hb = sms.het_breuschpagan(resid, X_train)
labels = ["Lagrange multiplier statistic","p-value","f-value","f p-value"]
for labels, num in zip(labels,hb):print(f"{labels}:{num:.2}")
    # the p-value = 0, so the model suffers from heteroscedasticity

# Test 4 : Autocorrelation of errors
dw = sd.durbin_watson(resid)
# dw = 2.09 (no autocorrelation problem in residuals)

# a summary using statsmodel.api
mod = sm.OLS(y_train, sm.add_constant(X_train))
res = mod.fit()
print(res.summary())

# Conclusion : the model has a low prediction score (R² of 26%) but some the loadings are highly significant
# based on the t-value in the summary table. Example: being a male, with parental_level_bachelor's degree influences
# positively the Math score. The model suffers from Heteroscedasticity, so the loadings aren't stable within time.

# Diagram of prediction errors
pev = PredictionError(lr)
pev.fit(X_train,y_train)
pev.score(X_test,y_test)
pev.show()

# in order to keep the code short, I won't put the models for the reading and writing scores.

