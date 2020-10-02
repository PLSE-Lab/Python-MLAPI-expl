# %% [code]
#Linear regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

kaggle = pd.read_csv("../input/train.csv")
kaggle_test = pd.read_csv("../input/test.csv")
print("Before dropping of na values: ",kaggle.size, "\n")
kg_new = kaggle.dropna()
print("After dropping of na values: ", kg_new.size, "\n")

kg_new.describe() #Summary of statistics including IQR, mean std
kg_new.head(10)  #to get a view of the data, 10 denotes no of rows

#to calculate mean of indivisual train data
np.mean(kg_new.x)  #50.01430615164521
np.median(kg_new.x)  #49.0
#Mean and median are nearly equal, means distribution is symmetric

np.median(kg_new.y) #48.97302037
np.mean(kg_new.y)  #49.93986917045776
#Mean and median are nearly equal, means distribution is symmetric

#Lets draw some plot for better visualisation of symmetry
plt.hist(kg_new.y, color = "magenta")
plt.hist(kg_new.x, color= "blue")

plt.boxplot(kg_new.x,0,"rs",0)#no outlier else that would be represented by red dot("rs")
plt.boxplot(kg_new.y)

kg_new.skew() #x    0.066766
#y    0.054034, both values are positive i.e positive skewness

#lets plot scatter plot
import seaborn as sns
sns.scatterplot(kg_new.x, kg_new.y)

#lets build linear regression model
model = smf.ols('y~x',data=kg_new).fit()

# For getting coefficients of the varibles used in equation
model.params  #gives the intercept and other parametrs

model.summary()  #R value = 0.991, F-statistic is also too high which is good for model
#"""Out[72]: 
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.991
Model:                            OLS   Adj. R-squared:                  0.991
Method:                 Least Squares   F-statistic:                 7.426e+04
Date:                Thu, 09 Apr 2020   Prob (F-statistic):               0.00
Time:                        22:05:39   Log-Likelihood:                -1712.8
No. Observations:                 699   AIC:                             3430.
Df Residuals:                     697   BIC:                             3439.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -0.1073      0.212     -0.506      0.613      -0.524       0.309
x              1.0007      0.004    272.510      0.000       0.993       1.008
==============================================================================
Omnibus:                        0.170   Durbin-Watson:                   1.966
Prob(Omnibus):                  0.919   Jarque-Bera (JB):                0.216
Skew:                           0.036   Prob(JB):                        0.898
Kurtosis:                       2.952   Cond. No.                         115.
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified."""

model.conf_int(0.95)  #95% conf interval

pred = model.predict(kaggle_test)
print("Predicted values of y using the model ",pred)

# Visualization of regresion line over the scatter plot of Waist and AT
# For visualization we need to import matplotlib.pyplot
import matplotlib.pylab as plt
plt.scatter(x=kg_new.x,y=kg_new.y,color='grey');plt.plot(kg_new.x,pred,color='blue');plt.xlabel('x');plt.ylabel('y')

pred.corr(kg_new.y) 
""" Now that's a good R sq and correlation!!! 
The positive correlation shows that as Y increases so does X."""