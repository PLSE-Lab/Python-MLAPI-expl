#!/usr/bin/env python
# coding: utf-8

# # **Linear regression **

# # **Import libraries**

# In[ ]:


import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings("ignore")


# # **Data set**

# In[ ]:


df = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')
df.head()


# In[ ]:


df.shape


# # **Data cleaning**

# In[ ]:


df = df.drop(['Serial No.'], axis=1)


# Serial No can be removed. It is not useful for model building.

# In[ ]:


df.isnull().sum()


# No empty cell at all.

# In[ ]:


df.columns = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA','Research', 'Chance of Admit']


# Rename the columns, there are some spacing problems

# # **Data Analysis**

# In[ ]:


DV = 'Chance of Admit'
IVs = list(df.columns)
IVs.remove('Chance of Admit')


# In[ ]:


def scatter_plot_with_admit(X,Y,df):
    fig = sns.regplot(x=X, y=Y, data=df)
    plt.title(str(X) + ' vs ' + str(Y))
    plt.show()

def plot_all(IVs, DV, df):
    for IV in IVs:
        scatter_plot_with_admit(IV,DV,df)


# Create plot for all indepenent variables against dependent varibale

# In[ ]:


plot_all(IVs,DV,df)


# Seem all Independent variables are positively correlated to Chance of admit.
# Correlation coefficient will be computed.

# In[ ]:


corr = df.corr()
fig,ax = plt.subplots(figsize= (6, 6))
sns.heatmap(corr, ax= ax, annot= True,linecolor = 'white')
corr.style.background_gradient(cmap='coolwarm').set_precision(2)
plt.show()


# CGPA are highly correlated with GRE Score, TOEFL Score, University rating, SOP. 
# 
# To avoid multicollinearity, only CGPA will be kept as independent variable. 
# 
# Chance of adamit and CGPA have the highest correlation coefficient among all pairs of change of admit and other variables.

# # **Model building**

# In[ ]:


x = df.drop(['Chance of Admit','GRE Score','TOEFL Score','University Rating','SOP'], axis=1)
y = df['Chance of Admit']
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.20, shuffle=False)


# In[ ]:


mod = sm.OLS(y_train,X_train)
fii = mod.fit()
p_values = fii.summary2().tables[1]['P>|t|']
p_values <0.05


# Considering 95% significant, all explanatory variables has a p-value smaller than 0.05. 
# 

# In[ ]:


lm = linear_model.LinearRegression()
lm.fit(X_train, y_train)


# In[ ]:


predictions = lm.predict(X_test)


# In[ ]:


for IV,coef in zip(X_train.columns,lm.coef_):
    print('coefficient of',IV,'is' ,coef)


# In[ ]:


lm.intercept_


# **The Regression formula shows below.**

# Estimated chance of admit = -0.838 + LOR * (0.023) + CGPA * (0.170) + Research * (0.036)

# # **R square and adjusted R square**

# There are two ways to compute R square adn adjusted R square.

# In[ ]:


yhat = lm.predict(X_train)
SS_Residual = sum((y_train-yhat)**2)       
SS_Total = sum((y_train-np.mean(y_train))**2)     
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)
print(r_squared, adjusted_r_squared)


# In[ ]:


print(lm.score(X_train, y_train), 1 - (1-lm.score(X_train, y_train))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))


# About 78.5% of the variation of chance of admit can be explained by this model.
# 
# Notice that this is a multiple linear regression, so we consider adjusted R square.

# # **Assumption Checking**

# In[ ]:


sns.residplot(predictions.reshape(-1),y_test, data=df,lowess=True,
                                  line_kws={'color': 'red', 'lw': 1, 'alpha': 1})
plt.xlabel("Fitted values")
plt.title('Residual plot')


# No obvious trend of residual variance.
# Residual seems independent.

# In[ ]:


residuals = y_test - predictions.reshape(-1)
plt.figure(figsize=(7,7))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Normal Q-Q Plot")


# Residual seems normally distributed.

# # ****MSE and MAE****

# In[ ]:


print('Mean Squared error:',np.sqrt(mean_squared_error(y_test, predictions)))
print('Mean Absolute error:',mean_absolute_error(y_test, predictions))


# Both values are small, the difference between prediction and actual value is small.

# # **Conclusion**

# This model explain 78.5% of the variation of chance of admit by the response variables. No assumputions of regression are violated.
