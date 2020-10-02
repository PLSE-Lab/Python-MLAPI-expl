#!/usr/bin/env python
# coding: utf-8

# Ridge regression uses the same least-squares criterion, but with one difference. During the training phase, it adds a penalty for feature weights. Let's see the comparison. 

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import linear_model



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")
print(data.head())
print(data.shape)
print(data.dtypes)


# Now i will get rid of the variables not required in the modeling: id and date.

# In[ ]:


data = data.drop(['id', 'date'], axis = 1)


# Lets do some Exploratory Analysis

# In[ ]:


import seaborn as sns
sns.distplot( a = data["price"], hist = True, kde = True, 
             kde_kws={"color": "g", "alpha":0.3, "linewidth": 5, "shade":True}
            )


# In[ ]:


sns.lmplot(x = "price", y = "sqft_living", data = data, fit_reg = False)


# We can observe the relation of price with different variables in one go too!

# In[ ]:


p = sns.pairplot(data[['sqft_lot','sqft_above','price','sqft_living','bedrooms']], palette='afmhot',height=1.4)
p.set(xticklabels=[])


# Price seems to have directrelation with most of the variables

# In[ ]:


data.head()


# In[ ]:


y = data.price.values
data = data.drop(['price'], axis = 1)
X = data.to_numpy()
colnames = data.columns


# In[ ]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
linridge = Ridge(alpha = 20.0).fit(X_train, y_train)


# In[ ]:


print("Training R Squared : {}".format(linridge.score(X_train, y_train)))
print("Testing R Squared : {}".format(linridge.score(X_test, y_test)))


# Lets perform Regularization and find the best R-squared for different alpha values! 

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
clf = Ridge().fit(X_train_scaled, y_train)
R_squared = clf.score(X_test_scaled, y_test)


# In[ ]:


R_squared


# As we see, the R-Squared has gone down. The normalization is not effective if we have broad range of variables being proved by this example. 

# In[ ]:


print('Ridge Regression: Effect of alpha regularization paramater')
for this_alpha in [0, 1,  10, 20, 30, 50, 100, 250, 500]:
    linridge = Ridge(alpha = this_alpha).fit(X_train, y_train)
    r2_train = linridge.score(X_train, y_train)
    r2_test = linridge.score(X_test, y_test)
    num_coeff_bigger = np.sum(abs(linridge.coef_) > 1.0)
    print('Alpha = {}\n    num abs(coeff) > 1.0: {},     r-squared training: {}, rsquared test: {}\n '.format(this_alpha, num_coeff_bigger, r2_train, r2_test))


# at Alpha = 20, the R-Squared Test is the highest. 

# The LASSO Regression uses L1 regularization type to reduce error. LASSO is said to be used with lesser variables but with medium/large effects. Ridge instead is used with many variables but small/medium sized effects. Default alpha is 1.0 in Lasso

# In[ ]:


from sklearn.linear_model import Lasso
linlasso = Lasso(alpha= 1.0, max_iter = 10000).fit(X_train_scaled, y_train)
print("Training R Squared : {}".format(linlasso.score(X_train_scaled, y_train)))
print("Testing R Squared : {}".format(linlasso.score(X_test_scaled, y_test)))


# Finding the best parameters

# In[ ]:


print('Lasso Regression: Effect of alpha regularization paramater')
for alpha in [0.5, 1, 2,3,4,5,10,20,50]:
    linlasso = Lasso(alpha, max_iter = 10000).fit(X_train_scaled, y_train)
    r2_train = linlasso.score(X_train_scaled, y_train)
    r2_test = linlasso.score(X_test_scaled, y_test)
    
    print('Alpha = {}\n    Features kept: {}, r-squared training: {},    r-squared testing: {}\n'.format(alpha, np.sum(linlasso.coef_ !=0), r2_train, r2_test))


# Alpha = 5 gives the highest R-Squared on test data. 

# DEPLOYING POLYNOMIAL REGRESSION
# Adding extra polynomial features allows us a much richer set of complex functions that we can use to fit to the data. So you can think of this intuitively as allowing polynomials to be fit to the training data instead of simply a straight line, but using the same least-squares criterion that minimizes mean squared error.

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2)
X_F1_poly = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_F1_poly, y, random_state = 0)


# In[ ]:


from sklearn.linear_model import LinearRegression
linreg = LinearRegression().fit(X_train, y_train)
print('poly degree 2 Linear model R squared training:{}'.format(linreg.score(X_train, y_train)))
print('poly degree 2 Linear model R squared test:{}'.format(linreg.score(X_test, y_test)))


# We can see much improved R-Squared with polynomial model as it allows a better fit
