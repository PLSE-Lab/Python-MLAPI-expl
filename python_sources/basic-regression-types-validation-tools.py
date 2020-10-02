#!/usr/bin/env python
# coding: utf-8

# **ADVERTISING**
# 
# Advertising data set consists of the sales of that product in 200 different markets, along with advertising budgets for the product in each of those markets for three different media:TV, radio, and newspapers.
# In this notebook, we basically examine some regression types and validaiton Tools.
# 
# 1.Plotting sales against TV
# 
# 2.Plotting sales against Radio
# 
# 3.Plotting sales against Newspapers
# 
# 4. Multiple Regression
# 
# 5. Cross Validation
# 
# 6. K-Fold Cross Validation
# 
# 7. Cross_val_score
# 
# 8. Ridge Regression
# 
# 9. Lasso Regression
# 
# 10. Elastic Net
# 

# In[4]:


#importing library
import pandas as pd
import matplotlib.pyplot as plt


# In[5]:


#read the data
advertising = pd.read_csv('../input/Advertising.csv')


# In[6]:


#check the data
advertising.head()


# In[7]:


#Drop unnecessary columns
drop_elements = ['Unnamed: 0']
advertising = advertising.drop(drop_elements, axis=1)


# In[8]:


#check the tail of data
advertising.tail()


# In[9]:


#check the missing columns
advertising.isnull().values.any()


# In[10]:


#Plotting sales against TV
fig,axes = plt.subplots(1,1,figsize=(8,6),sharey=True)
axes.scatter(advertising.TV.values,advertising.Sales.values)
axes.set_xlabel(" TV")
axes.set_ylabel(" Sales")


# In[17]:


#Plotting sales against Radio
fig,axes = plt.subplots(1,1,figsize=(8,6),sharey=True)
axes.scatter(advertising.Radio.values,advertising.Sales.values)
axes.set_xlabel(" Radio")
axes.set_ylabel(" Sales")


# In[16]:


#Plotting sales against Newspapers
fig,axes = plt.subplots(1,1,figsize=(8,6),sharey=True)
axes.scatter(advertising.Newspaper.values,advertising.Sales.values)
axes.set_xlabel(" Newspapers")
axes.set_ylabel(" Sales")


# **Multiple Regression**
# 
# sales = Bo+B1*TV + B2 *radio+B3*newspapers+E 
# 
# You can find more information about regression via [Credit](https://www.slideshare.net/oxygen024/regression-analysis-8672920)

# In[18]:


#Multiple Linear Regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
features = ["TV","Radio","Newspaper"]
x=advertising[features]
y=advertising.Sales
model=lm.fit(x,y)
print (" Model coefficients are: ")
print ( model.coef_)
xpredicted = model.predict(x)
print("R-squared of the model")
print(model.score(x,y))


# Coefficient score refers to the relations between independent and dependent variable. In other words, the coefficient value indicated the cause and effect relationship between two types of variables.  [Credit](https://tampub.uta.fi/bitstream/handle/10024/99598/GRADU-1471263166.pdf?sequence=1)

# **Cross-Validation**
# 
# Memorizing the training set is called overfitting.
# 
# During development,and particularly when training data is scarce,a practice called cross-validation can be used to train and validate an algorithm on the same data.
# 
# This approach involves randomly dividing the available set of observations into two parts, a training set and a validation set.

# In[19]:


#cross validation
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
features = ["TV","Radio","Newspaper"]
x=advertising[features]
y=advertising.Sales

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0)
print("The length of tranning size is %d" % len(X_train))
print("The length of test size is %d " % len(X_test))
model = lm.fit(X_train,y_train)
print("The R-squared value of the model is %.2f" % model.score(X_test,y_test))


# In[ ]:


#Testing 10-times the linear regression model for the Advertising data set.
for i in range(0,15):
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.25)
    model = lm.fit(X_train,y_train)
    print(model.score(X_test,y_test))


# **K-Fold Cross Validation**
# 
# In the sklearn.model_selection there is a method called Kfold which can split the data set into the desired number of folds.

# In[24]:


#K- Fold Cross Validation
from sklearn.model_selection import KFold
import numpy as np

lm = LinearRegression()
features = ["TV","Radio","Newspaper"]
x=advertising[features]
y=advertising.Sales

kf = KFold(n_splits=10)
scores=[]
for train,test in kf.split(x,y):
    model = lm.fit(x.values[train],y.values[train])
    score = model.score(x.values[test],y.values[test])
    print(score)
    scores.append(score)
    
print("The mean score for %d-fold cross validation is %.2f" % (kf.get_n_splits(),np.mean(np.array(scores))))


# **cross_val_score method**
# 
# The simplest way to use cross-validation is to call the cross_val_score helper function on the estimator and the dataset.
# 
# 

# In[ ]:


#cross_val_score method
lm = LinearRegression()
features = ["TV","Radio","Newspaper"]
x=advertising[features]
y=advertising.Sales

from sklearn.model_selection import RepeatedKFold
rkf = RepeatedKFold(n_splits=4,n_repeats=2,random_state=True)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(lm,x,y,cv=rkf)
print(scores)
print("Average score %.2f" % scores.mean())


# **Ridge Regression**
# 
# Ridge regression,penalizes model parameters that become too large.
# 
# Ridge regression modifies the residual sum of the squares cost function by adding the L2 norm of the coefficients,as follows:
# 
# ![](http://businessforecastblog.com/wp-content/uploads/2014/05/ridgeregressionOF.png)

# In[ ]:


#Ridge Regression
from sklearn.linear_model import Ridge

model_ridge = Ridge(alpha=0.5)
features = ["TV","Radio","Newspaper"]
x=advertising[features]
y=advertising.Sales

from sklearn.model_selection import cross_val_score
scores=cross_val_score(model_ridge,x,y,cv=5)
print(scores)
print("Average score %.2f" % scores.mean())


# **Lasso Regression**
# 
# 

# In[ ]:


#Lasso Regression
from sklearn.linear_model import Lasso

model_ridge = Lasso(alpha=0.1)
features = ["TV","Radio","Newspaper"]
x=advertising[features]
y=advertising.Sales

from sklearn.model_selection import cross_val_score
scores=cross_val_score(model_ridge,x,y,cv=5)
print(scores)
print("Average score %.2f" % scores.mean())


# **Elastic Net**

# In[3]:


#Elastic Net
from sklearn.linear_model import ElasticNet

model_elasticnet = ElasticNet(alpha=0.1,l1_ratio=0.5)
features = ["TV","Radio","Newspaper"]
x=advertising[features]
y=advertising.Sales

from sklearn.model_selection import cross_val_score
scores=cross_val_score(model_elasticnet,x,y,cv=5)
print(scores)
print("Average score %.2f" % scores.mean())


# 
