#!/usr/bin/env python
# coding: utf-8

# **Predict company profit using Ridge and support vector regressor with grid search CV.**
# * Quick data visiualization 
# * Ridge polynomial regression 
# * Test and Train error analysis
# * Support Vector regression using polynomial kernel
# * Tuning hyper parameters with grid search cross validation

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# This data has 4 attributes (features: R&D spend, Administration , Marketing, and State) from 50 start-up companies that relate to their profit (target).

# In[ ]:


data = pd.read_csv('/kaggle/input/50_Startups.csv')
data.info()
data.head(5)


# By looking at cross correlation chart, the profit is highly correlated with R&D spend and Marketing with R&D spend being the highest.

# In[ ]:


corr_matrix = data.corr()
sns.heatmap(corr_matrix,annot=True,cmap="coolwarm",fmt=".2f",annot_kws={'size':16})
corr_matrix['Profit']


# Same observation: R&D spend is the highest correlation feature that is why I used it do simple regression. Later on we can use all the features and to see if it performs better.

# In[ ]:


sns.set(style="ticks")
sns.pairplot(data)
plt.show()


# In[ ]:


Y = data.iloc[:,-1].values.reshape((len(data),1))
X = data.iloc[:,0].values.reshape((len(data),1))


# Here, I used polynomial ridge and linear regressor to generate my models. As you can see, it seems ridge regressor fits model preety well. But, is that case?

# In[ ]:


plt.figure(figsize=(10,5))
plt.scatter(X,Y,label='data')
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

poly_feature = PolynomialFeatures(degree=10)
X_poly = poly_feature.fit_transform(X)
#creat input data for pltting 
plot_x = np.arange(0,180000,1000).reshape((180,1))
plot_X_poly = poly_feature.fit_transform(plot_x)

linreg = LinearRegression()
linreg.fit(X_poly,Y)
plt.plot(plot_x,linreg.predict(plot_X_poly),'r',label='linear regressor')

ridgreg = Ridge(alpha=1,solver='cholesky')
ridgreg.fit(X_poly,Y)
#plt.scatter(X,ridgreg.predict(X_poly))
plt.plot(plot_x,ridgreg.predict(plot_X_poly),'k',label='Ridge regressor')
plt.xlim([min(X)-5000,max(X)+5000])
plt.ylim([min(Y)-5000,max(Y)+5000])
plt.xlabel('R&D Spend');plt.ylabel('Profit');plt.legend()
plt.show()


# Here, i am looking at test and train RMS error. Overall, my test error is low, but train error is high. This is classic overfitting problem!

# In[ ]:


from sklearn.model_selection import cross_validate
cv_results = cross_validate(ridgreg,X_poly,Y,cv=5,scoring="neg_mean_squared_error")
for a,b in zip(np.sqrt(-cv_results['train_score']),np.sqrt(-cv_results['test_score'])):
    print("training error:",a,"testing error:",b)
print("average training error:",np.mean(np.sqrt(-cv_results['train_score'])),
      "average testing error:",np.mean(np.sqrt(-cv_results['test_score'])))

plt.scatter([1,2,3,4,5],np.log(np.sqrt(-cv_results['train_score'])),label='train RMSE')
plt.scatter([1,2,3,4,5],np.log(np.sqrt(-cv_results['test_score'])),label = 'test RMSE')
plt.xlabel('fold number');plt.ylabel('log RMSE');plt.legend();plt.show()


# Then, I am trying a new regressor to train a new model using support vector regressor.
# **Warning: If your kernels run forever for polynomial kernel or model compeletly does not make sense, try to apply feature scaling!**

# In[ ]:


plt.scatter(X,Y,label='data')
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

def test_regressor(svm_reg):
    # preprocessing data , otherwise wont coverge with SVR
    scx = StandardScaler();X_sc = scx.fit_transform(X)
    scy = StandardScaler();Y_sc = scy.fit_transform(Y)

    svm_reg.fit(X_sc,Y_sc.ravel())
    #creat input data for pltting 
    plot_x = np.arange(0,180000,1000).reshape((180,1))
    y_sc_plot_pre = svm_reg.predict(scx.fit_transform(plot_x))
    y_plot_pre = scy.inverse_transform(y_sc_plot_pre)
    plt.plot(plot_x,y_plot_pre,c='k',label='SVR regresoor')
    plt.xlabel('R&D Spend');plt.ylabel('Profit');plt.legend()
    plt.show()
    return X_sc,Y_sc
    #y_sc_predict = svm_poly_reg.predict(X_sc)
    #y_predict = scy.inverse_transform(y_sc_predict)
    #plt.plot(X,y_predict)

X_sc,Y_sc = test_regressor(SVR(kernel='poly',degree=3,C=100,gamma=0.1,epsilon=0.001))


# Unlike, other regressors used above , SVR needs to tune quite a few paramters. Sklearn has a function, GridSearchCV(), which test models for all possible combinations of input hyper parameters. 

# In[ ]:


from sklearn.model_selection import GridSearchCV
param_grid = {
    'C':[10,100,1000],
    'degree':[1,3,5],
    'epsilon':[0.01,0.1]
}
svm_poly_reg = SVR(kernel='poly',gamma='auto')
grid_search = GridSearchCV(svm_poly_reg,param_grid,cv=5,scoring='neg_mean_squared_error')
grid_search.fit(X_sc,Y_sc.ravel())


# Guess, which one yields the best result? It is actually the lower order polynomial not the higher ones, as we have seen the higher ones was overfitting the data badly in our previous analysis. Do not get fooled by visualization! Always, tune your parameters before move on.

# In[ ]:


gcv_res =grid_search.cv_results_
for mean_score,params in zip(gcv_res["mean_test_score"],gcv_res["params"]):
    print(np.sqrt(-mean_score),params)


# In[ ]:


plt.scatter(X,Y,label='data')
test_regressor(grid_search.best_estimator_)

