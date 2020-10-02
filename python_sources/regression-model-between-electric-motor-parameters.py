#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from sklearn.preprocessing import Imputer
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.metrics import make_scorer

from sklearn import svm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from sklearn import neighbors
from math import sqrt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Any results you write to the current directory are saved as output.


# [**BASIC LINEAR REGRESSION**](#1)
# 
# [**MULTIPLE LINEAR REGRESSION**](#2)
# 
# [**PRINCIPAL COMPONENT REGRESSION**](#3)
# 
# [**K-NEAREST NEIGHBORHOOD REGRESSION**](#4)
# 
# [**POLYNOMIAL REGRESSION**](#5)

# In[ ]:


# read data
df = pd.read_csv('../input/pmsm_temperature_data.csv', 
                 usecols=[0,1,2,3,4,5,6,7,8,9,10,11])
df.head(10)


# In[ ]:


df.info()


# In[ ]:


df.describe().T


# In[ ]:


df.isnull().values.any()


# In[ ]:


# Count the number of NaNs each column has.
nans=pd.isnull(df).sum()
nans[nans>0]


# In[ ]:


# Count the column types
df.dtypes.value_counts()


# In[ ]:


df.corr()


# In[ ]:


import seaborn as sns
sns.jointplot(x='i_d', y='motor_speed', data=df, kind='reg')


# In[ ]:


#correlation map
f,ax=plt.subplots(figsize=(12,12))
corr=df.corr()

sns.heatmap(corr, annot=True, linewidths=.5, fmt='.2f', 
            mask= np.zeros_like(corr,dtype=np.bool), 
            cmap=sns.diverging_palette(100,200,as_cmap=True), 
            square=True, ax=ax)

plt.show()


# There are high correlation between values.

# <a id= "1" ></a><br>**BASIC LINEAR REGRESSION**

# In[ ]:


import statsmodels.api as sm
#Defining dependet and independent variable
X = df['i_d']
X=sm.add_constant(X)

y = df['motor_speed']

lm=sm.OLS(y,X)
model=lm.fit()

model.summary()


# y=-0.002-0.7245*x this is formula of basic regression model. p-value is less than 0.05 so it is meaningful model.

# In[ ]:


model.params


# In[ ]:


print("f_pvalue:", "%.4f" % model.f_pvalue)


# In[ ]:


model.mse_model #mean squared error is too much. It is not good.


# In[ ]:


model.rsquared #Not bad


# In[ ]:


model.rsquared_adj #Not bad


# In[ ]:


model.fittedvalues[0:5] #Predicted values


# In[ ]:


y[0:5] #Real values


# In[ ]:


#Model equation
print("Motor speed = " + 
      str("%.3f" % model.params[0]) + ' + i_d' + "*" + 
      str("%.3f" % model.params[1]))


# In[ ]:


#Model Visualization 
g=sns.regplot(df['i_d'] , df['motor_speed'], 
              ci=None, scatter_kws={'color': 'r', 's':9})
g.set_title('Model equation: motor_speed = -0.002 + i_d * -0.725')
g.set_ylabel('Motor_speed')
g.set_xlabel('i_d');


# In[ ]:


from sklearn.metrics import r2_score,mean_squared_error

mse=mean_squared_error(y, model.fittedvalues)
rmse=np.sqrt(mse)
rmse


# In[ ]:


k_t=pd.DataFrame({'Real_values':y[0:50], 
                  'Predicted_values' :model.fittedvalues[0:50]})
k_t['error']=k_t['Real_values']-k_t['Predicted_values']
k_t.head()


# In[ ]:


model.resid[0:10] #It is easy way to learn residuals.


# In[ ]:


plt.plot(model.resid);


# <a id= "2" ></a><br>**MULTIPLE LINEAR REGRESSION**

# In[ ]:


X=df.drop("motor_speed", axis=1)
y=df["motor_speed"]


# In[ ]:


from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)

training=df.copy()


# In[ ]:


lm=sm.OLS(y_train, X_train)

model=lm.fit()
model.summary() #All coefficients are significant for the model by looking at the p-value. ( P>|t| )


# In[ ]:


#Root Mean Squared Error for Train
rmse1=np.sqrt(mean_squared_error(y_train,model.predict(X_train)))
rmse1


# In[ ]:


#Root Mean Squared Error for Test
rmse2=np.sqrt(mean_squared_error(y_test,model.predict(X_test)))
rmse2


# In[ ]:


#Model Tuning for Multiple Linear Regression
model = LinearRegression().fit(X_train,y_train)
cross_val_score1=cross_val_score(model, X_train, y_train, cv=10, scoring='r2').mean() #verified score value for train model
print('Verified R2 value for Training model: ' + str(cross_val_score1))

cross_val_score2=cross_val_score(model, X_test, y_test, cv=10, scoring='r2').mean() #verified score value for test model
print('Verified R2 value for Testing Model: ' + str(cross_val_score2))


# In[ ]:


RMSE1=np.sqrt(-cross_val_score(model, X_train, y_train, cv=10, 
                               scoring='neg_mean_squared_error')).mean() #verified RMSE score value for train model
print('Verified RMSE value for Training model: ' + str(RMSE1))

RMSE2=np.sqrt(-cross_val_score(model, X_test, y_test, cv=10, 
                               scoring='neg_mean_squared_error')).mean() #verified RMSE score value for test model
print('Verified RMSE value for Testing Model: ' + str(RMSE2))


# In[ ]:


#Visualizing for Multiple Linear Regression y values

import seaborn as sns
ax1 = sns.distplot(y_train, hist=False, color="r", label="Actual Value")
sns.distplot(y_test, hist=False, color="b", label="Fitted Values" , ax=ax1);


# <a id= "3" ></a><br>**PRINCIPAL COMPONENT REGRESSION**

# In[ ]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

pca=PCA()
X_reduced_train=pca.fit_transform(scale(X_train))


# In[ ]:


explained_variance_ratio=np.cumsum(np.round(pca.explained_variance_ratio_ , decimals=4)* 100)[0:20]


# In[ ]:


plt.bar(x=range(1, len(explained_variance_ratio)+1), height=explained_variance_ratio)
plt.ylabel('percentange of explained variance')
plt.xlabel('principal component')
plt.title('bar plot')
plt.show()
# 7 component is enough for model.


# In[ ]:


lm=LinearRegression()
pcr_model=lm.fit(X_reduced_train,y_train)
print('Intercept: ' + str(pcr_model.intercept_))
print('Coefficients: ' + str(pcr_model.coef_))


# In[ ]:


#Prediction
y_pred=pcr_model.predict(X_reduced_train)
np.sqrt(mean_squared_error(y_train,y_pred))


# In[ ]:


df['motor_speed'].mean()


# In[ ]:


#R squared
r2_score(y_train,y_pred)


# In[ ]:


# Prediction For testing error 
pca2=PCA()

X_reduced_test=pca2.fit_transform(scale(X_test))
pcr_model2=lm.fit(X_test,y_test)

y_pred=pcr_model2.predict(X_reduced_test)

print('RMSE for test model : ' +str(np.sqrt(mean_squared_error(y_test,y_pred))))


# In[ ]:


#Model Tuning for PCR

lm=LinearRegression()
pcr_model=lm.fit(X_reduced_train[:,0:10],y_train)
y_pred=pcr_model.predict(X_reduced_test[:,0:10])

from sklearn import model_selection

cv_10=model_selection.KFold(n_splits=10,
                           shuffle=True,
                           random_state=1)


# In[ ]:


lm=LinearRegression()
RMSE=[]

for i in np.arange(1,X_reduced_train.shape[1] + 1):
    score=np.sqrt(-1*model_selection.cross_val_score(lm,
                                                    X_reduced_train[:,:i],
                                                    y_train.ravel(),
                                                    cv=cv_10,
                                                    scoring='neg_mean_squared_error').mean())
    RMSE.append(score)


# In[ ]:


plt.plot(RMSE)
plt.xlabel('# of Components')
plt.ylabel('RMSE')
plt.title('PCR Model Tuning for Motor_Speed Prediction'); 


# 10 component is good for the model because RMSE value is the smallest for this component number. That's why there is no need to tune the model.

# <a id= "4" ></a><br>**KNN REGRESSION**

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor

from warnings import filterwarnings
filterwarnings('ignore')


# In[ ]:


knn_model=KNeighborsRegressor().fit(X_train, y_train)
y_pred=knn_model.predict(X_test)


# In[ ]:


y_pred.shape


# In[ ]:


#Model Tuning (learning best n_neighbors hyperparameter)
knn_params={'n_neighbors' : np.arange(1,5,1)}

knn=KNeighborsRegressor()
knn_cv_model=GridSearchCV(knn, knn_params, cv=5)

knn_cv_model.fit(X_train,y_train)
knn_cv_model.best_params_["n_neighbors"]


# In[ ]:


# Train error values from n=1 up n=2
RMSE=[]
RMSE_CV=[]
for k in range(2):
    k=k+1
    knn_model=KNeighborsRegressor(n_neighbors=k).fit(X_train, y_train)
    y_pred=knn_model.predict(X_train)
    rmse=np.sqrt(mean_squared_error(y_train,y_pred))
    rmse_cv=np.sqrt(-1*cross_val_score(knn_model,X_train,y_train,cv=2,
                                       scoring='neg_mean_squared_error').mean())

    RMSE.append(rmse)
    RMSE_CV.append(rmse_cv)

    print("RMSE value: ", rmse, 'for k= ',k,
          "RMSE values with applying Cross Validation: ", rmse_cv)


# In[ ]:


#Model Tuning according to best parametre for KNN Regression
knn_tuned=KNeighborsRegressor(n_neighbors=knn_cv_model.best_params_["n_neighbors"])
knn_tuned.fit(X_train,y_train)
np.sqrt(mean_squared_error(y_test,knn_tuned.predict(X_test)))


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error


# <a id= "5" ></a><br> POLYNOMIAL REGRESSION

# In[ ]:


quad = PolynomialFeatures (degree = 2)
x_quad = quad.fit_transform(X_train)

X_train,X_test,y_train,y_test = train_test_split(x_quad,y_train, random_state = 0)

plr = LinearRegression().fit(X_train,y_train)

Y_train_pred = plr.predict(X_train)
Y_test_pred = plr.predict(X_test)

print('Polynomial Linear Regression:' ,plr.score(X_test,y_test))


# In[ ]:


#Plotting Residual in Linear Regression 

from sklearn import linear_model,metrics
#Create linear regression object
reg=linear_model.LinearRegression()

#train the model using the train data sets
reg.fit(X_train,y_train)

#regression coefficients
print("Coefficients: \n", reg.coef_)

#Variance score
print("Variance score: {}".format(reg.score(X_test,y_test)))

plt.style.use('fivethirtyeight')

#plotting residual errors in training data
plt.scatter(reg.predict(X_train),reg.predict(X_train)-y_train, 
            color="green", s=10, label="train data")

#plotting residual errors in test data
plt.scatter(reg.predict(X_test),reg.predict(X_test)-y_test, 
            color="blue", s=10, label="test data")

#plot line for zero residual error
plt.hlines(y=0,xmin=-2, xmax=2, linewidth=2)

#plot legend
plt.legend(loc='upper right')

#plot title
plt.title("residual error")

plt.show()

