#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pygam import LinearGAM
import pandas as pd
import torch as t
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# Generalized regression model is one of the most powerful statistical learning model. The following notebook demonstrates use of GAM to predict the total daily use of bike in Washinton DC area

# In[ ]:


day_data = pd.read_csv("../input/bike_sharing_daily.csv")
feature =['cnt','season','windspeed','atemp','hum','yr','mnth']
Daily_bike_share= pd.DataFrame()
Daily_bike_share = day_data[feature]


# In[ ]:


Daily_bike_share.head(10)


# The correlation between the variables to be used for the Generative Additive Model can been in the figure below. 

# In[ ]:


correlation = Daily_bike_share.corr()
fig, axes  = plt.subplots(figsize=(10,8))
sns.heatmap(correlation,annot=True,vmin=-1,vmax=1,center=0,ax=axes)


# In[ ]:


Train_data, test_data = train_test_split(Daily_bike_share, test_size = 0.2)

X_train = Train_data.drop(columns = ["cnt"])
y_train = Train_data[['cnt']]
########################################
X_test = test_data.drop(columns = ["cnt"])
y_test = test_data[['cnt']]



# The general model for regression using generalized additive model (GAM) is as follows >>
# $g(E(Y|X)) = \beta_0 + f_0(X_0) + f_i1X_1) + \ldots + f_k(X_k)$ 
# where$ X.T = [X_1, X_2, ..., X_k]$ are independent variables, Y is the dependent variable, and g() is the link function that relates our predictor variables to the expected value of the dependent variable.
# The feature functions $f_i()$ are built using penalized B splines, which allow us to automatically model non-linear relationships without having to manually try out many different transformations on each variable.

# In[ ]:


gam = LinearGAM(n_splines=10).gridsearch(X_train, y_train)


# In[ ]:


XX = gam.generate_X_grid()
fig, axs = plt.subplots(1,6, figsize=(20,4))
titles = feature[1:]

for i, ax in enumerate(axs):
    pdep, confi = gam.partial_dependence(XX, feature=i, width=.95)
    ax.plot(XX[:, i], pdep)
    ax.plot(XX[:, i], *confi, c='r', ls='--')
    ax.set_title(titles[i])
    


# In[ ]:


gam.summary()


# In[ ]:


plt.plot(gam.predict(XX), 'r--')
plt.plot(gam.prediction_intervals(XX, width=.95), color='b', ls='--')

#plt.plot(y_train, facecolor='gray', edgecolors='none')
plt.title('95% prediction interval')


# In[ ]:


plt.plot(y_test, gam.predict(X_test),"*")
plt.xlabel("Predicted Value")
plt.ylabel("Actual value")


# In[ ]:


from sklearn.metrics import mean_absolute_error,r2_score
print("Mean absolute error >> " + str(mean_absolute_error(y_test, gam.predict(X_test))))
print("Coefficient of determination >> " + str(r2_score(y_test, gam.predict(X_test))))


# In[ ]:




