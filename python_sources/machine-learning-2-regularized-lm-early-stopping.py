#!/usr/bin/env python
# coding: utf-8

# **Table of Content:** <br/>
# This notebook is an illustration of how regularized linear models work, using the house prices **training** set. <br/>
# I did submit my prediction just to see how it performs on the test set, but the purpose is **not** to enter competition, but to teach the various regularization methods. 
# 
# * Single-variable models (for illustration purposes)
#     * Ridge regression
#     * Lasso regression
#     * Elastic net
# * Multi-variable models
#     * Elastic net
#     * SGD and early stopping (with k-fold splitting)

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
import os
print(os.listdir("../input"))


# In[ ]:


plt.rc('axes', lw = 1.5)
plt.rc('xtick', labelsize = 14)
plt.rc('ytick', labelsize = 14)
plt.rc('xtick.major', size = 5, width = 3)
plt.rc('ytick.major', size = 5, width = 3)


# These dataset are from the House Prices competition, yet the purpose of this Notebook is to learn and experiment with regularized linear models. <br/>
# We will first choose just one independent variable for visualization purpose, and try out different regularized linear models. Later we will expand the models to multiple independent variables. <br/>
# Data description is available here: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

# In[ ]:


# open the training dataset
dataTrain = pd.read_csv('../input/train.csv')
dataTrain.head()


# Since we have not talked about catgorical data, we will focus only on numeric features here. <br/>

# In[ ]:


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
dataTrain = dataTrain.select_dtypes(include=numerics)
dataTrain.head()


# # Single Variable Models (for illustration purposes)
# **We simplify the data for the purpose of illustration and introduction of the regularized models. **<br/>
# Later we will include multiple variables in the models.

# In[ ]:


dataTrain = dataTrain[['GarageArea','SalePrice']]
dataTrain.head()


# Check to see if there are any missing values?

# In[ ]:


dataTrain.isnull().values.any()


# No missing values, so we don't have to worry much about data cleaning.

# In[ ]:


# Take a look at the data. 
plt.plot('GarageArea','SalePrice',data=dataTrain, marker = 'o', linestyle = '')
plt.ylabel('Sale Price (dollars)', fontsize = 18)
plt.xlabel('Garage Area (square feet)', fontsize = 18)
plt.show()


# In[ ]:


# format training data
xTrain = dataTrain['GarageArea'].values.reshape(-1,1) # as_matrix is deprecated since version 0.23.0
yTrain = dataTrain['SalePrice'].values.reshape(-1,1)
xTrain


# Let's first do a degree 10 linear regression model **without** regularization.

# In[ ]:


# Transform the input features
Poly = PolynomialFeatures(degree = 10, include_bias = False)
xTrainPoly = Poly.fit_transform(xTrain)


# Previously (see my Machine Learning 1 kernel), we standardized input features ourselves, but this can also be easily done through **Scikit-learn**:

# In[ ]:


from sklearn.preprocessing import StandardScaler
# standardization
scaler = StandardScaler()
xTrainPolyStan = scaler.fit_transform(xTrainPoly)
scaler.scale_, scaler.mean_


# In[ ]:


# linear regression
reg = LinearRegression()
reg.fit(xTrainPolyStan, yTrain)

# predict
xFit = np.linspace(0,1500,num=200).reshape(-1,1)
xFitPoly = Poly.transform(xFit)
xFitPolyStan = scaler.transform(xFitPoly)
yFit = reg.predict(xFitPolyStan)

# plot
plt.plot(xFit,yFit, lw=3, color='r', zorder = 2)
plt.plot('GarageArea','SalePrice',data=dataTrain, marker = 'o', color = 'b', linestyle = '', zorder = 1)
plt.ylabel('Sale Price (dollars)', fontsize = 18)
plt.xlabel('Garage Area (square feet)', fontsize = 18)
plt.show()


# ## Ridge Regression
# Regular linear regression has the form of: J(theta) = MSE(theta) <br/>
# 
# Ridge regression appy a regularization term proportional to the **square of l2-norm** of feature weights (not including the intercept). A common expression is: <br/>
# J(theta) = MSE(theta) + alpha 1/2 (theta_1^2 + theta_2^2 + ... + theta_n^2)
# 
# The corrsponding expression for gradient of theta and the optimal solution for theta will change, due to the additonal term. We can also use the **Scikit-Learn** package to do ridge regression.

# In[ ]:


from sklearn.linear_model import Ridge


# Ridge regression is sensitive to the input features, therefore **standardization is usually recommended** before Ridge regression. <br/>
# Some useful info here: https://stats.stackexchange.com/questions/111017/question-about-standardizing-in-ridge-regression <br/>
# Here, standardization has already been carried out (see above), so we will go straight to training.

# In[ ]:


i=0
ls = ['-','--',':']
color = ['r','g','orange']

for a in [0,2,2000]:
    ridgeReg = Ridge(alpha=a)
    ridgeReg.fit(xTrainPolyStan, yTrain)

    # predict
    xFit = np.linspace(0,1500,num=200).reshape(-1,1)
    xFitPoly = Poly.transform(xFit)
    xFitPolyStan = scaler.transform(xFitPoly)
    yFit = ridgeReg.predict(xFitPolyStan)
    
    # plot
    plt.plot(xFit,yFit, lw=3, color=color[i], zorder = 2, label= "alpha = " + str(a),linestyle=ls[i])
    i = i + 1
    
plt.plot('GarageArea','SalePrice',data=dataTrain, marker = 'o', color = 'b', linestyle = '', zorder = 1)
plt.ylabel('Sale Price (dollars)', fontsize = 18)
plt.xlabel('Garage Area (square feet)', fontsize = 18)
plt.legend(fontsize = 14)
plt.show()


# In general, the larger alpha is, the "flatter" the fit will be. Eventually, as alpha approaches infinity, the prediction y_hat will just be a constant, since all thetas (except the intercept) will be regularized to zero. <br/>
# 
# In theory, ridge regression with alpha = 0 should give the same result as regular linear regression, but sometimes that is not the case. See one post here: https://stackoverflow.com/questions/40570370/difference-between-linearregression-and-ridgealpha-0. <br/>
# The post describes a polynomial model where ridge regression overflowed, but linear regression did not. (we don't have this problem here.)

# ## Lasso Regression
# Least Absolute Shrinkage and Selection Operator Regression - LASSO <br/>
# Cost function: J(theta) = MSE(theta) + alpha (|theta_1| + |theta_2| + ... + |theta_n|). The penalty is proportional to the **l1-norm** of theta. <br/>
# 
# The advantage of Lasso over ridge regression lies in the diamond shape of contour of the l1-norm penalty, which leads to some of the thetas being eliminated (set to 0) quickly. This means the Lasso regression can perform automatic feature selection, when ridge regression cannot. If you have the book *Hands-on Machine Learning with Scikit-Learn & Tensorflow*, Figure 4-19 gives a more detailed explanation of the feature selection capability of Lass0. You can also understand the difference of ridge and Lasso regression by understanding that, ridge's l2-penalty heavily penalizes large thetas, but has nearly no penalization for small thetas (due to the square), whereas Lasso's l1-penalty gives appropriate penalization to even small thetas. <br/>

# In[ ]:


from sklearn.linear_model import Lasso


# In[ ]:


i=0
ls = ['-','--',':']
color = ['r','g','orange']

for a in [0.1,1,10]:
    lassoReg = Lasso(alpha=a)
    lassoReg.fit(xTrainPolyStan, yTrain)

    # predict
    xFit = np.linspace(0,1500,num=200).reshape(-1,1)
    xFitPoly = Poly.transform(xFit)
    xFitPolyStan = scaler.transform(xFitPoly)
    yFit = lassoReg.predict(xFitPolyStan)
    
    # plot
    plt.plot(xFit,yFit, lw=3, color=color[i], zorder = 2, label= "alpha = " + str(a),linestyle=ls[i])
    i = i + 1
    
plt.plot('GarageArea','SalePrice',data=dataTrain, marker = 'o', color = 'b', linestyle = '', zorder = 1)
plt.ylabel('Sale Price (dollars)', fontsize = 18)
plt.xlabel('Garage Area (square feet)', fontsize = 18)
plt.legend(fontsize = 14)
plt.show()


# As you can see, the three alpha tested gives very similar fits. As mentioned earlier, Lasso regression tends to return sparse theta vector, with many least important features eliminated (set to 0). Even when alpha is small, such elimination can happen, leading to similar fits for certain range of alphas.
# 
# As mentioned in Scikit-Learn's [documentaion](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html), Lasso function is not advised to use with alpha = 0. In such cases, LinearRegression should be used instead. <br/>
# 
# Stochastic gradient descent can be used for any type of optimization problem. Here we show the example of lasso regression using SGDRegressor from Scikit-Learn package.

# In[ ]:


from sklearn.linear_model import SGDRegressor


# In[ ]:


sgd = SGDRegressor(loss='squared_loss', penalty='l1', alpha=0.1)
yTrain = yTrain.ravel() # format required by sgd
sgd.fit(xTrainPolyStan, yTrain)

# predict
xFit = np.linspace(0,1500,num=200).reshape(-1,1)
xFitPoly = Poly.transform(xFit)
xFitPolyStan = scaler.transform(xFitPoly)
yFit = sgd.predict(xFitPolyStan)

plt.plot(xFit,yFit, lw=3, color='r', zorder = 2, label= "alpha = 0.1",linestyle='-')
plt.plot('GarageArea','SalePrice',data=dataTrain, marker = 'o', color = 'b', linestyle = '', zorder = 1)
plt.ylabel('Sale Price (dollars)', fontsize = 18)
plt.xlabel('Garage Area (square feet)', fontsize = 18)
plt.legend(fontsize = 14)
plt.show()


# ## Elastic Net
# Elastic net is somewhere between ridge regression and lasso regression. The cost function is: <br/>
# J(theta) = MSE(theta) + r lasso_penalty + (1-r) ridge_penalty. 

# In[ ]:


from sklearn.linear_model import ElasticNet


# In[ ]:


yTrain = yTrain.reshape(-1,1)
elasticReg = ElasticNet(alpha = 0.1, l1_ratio = 0.5)
elasticReg.fit(xTrainPolyStan, yTrain)

# predict
xFit = np.linspace(0,1500,num=200).reshape(-1,1)
xFitPoly = Poly.transform(xFit)
xFitPolyStan = scaler.transform(xFitPoly)
yFit = elasticReg.predict(xFitPolyStan)

plt.plot(xFit,yFit, lw=3, color='r', zorder = 2, label= "alpha = 0.1",linestyle='-')
plt.plot('GarageArea','SalePrice',data=dataTrain, marker = 'o', color = 'b', linestyle = '', zorder = 1)
plt.ylabel('Sale Price (dollars)', fontsize = 18)
plt.xlabel('Garage Area (square feet)', fontsize = 18)
plt.legend(fontsize = 14)
plt.show()


# **So, which model should we choose in practice?** <br/>
# Ridge regression: a good default. However, if sparse features are expected, ridge should be replaced by lasso or elastic net. <br/>
# Lasso regression: good for sparse feature selection. However, if the number of features is greater than the number of training samples, or when there are strongly correlated features, ridge or elastic net should be used. <br/>
# Elastic net: versatile since the ratio parameter r is tunable. A 50% ratio of l1 and l2-penalty can be a good default, too.

# # Multi-variable models (for actual prediction)
# Here, we try to predict house prices with an elastic net model with fourth order features. <br/>
# Let's look at the training set again:
# 
# 

# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


dataTrain = pd.read_csv('../input/train.csv')
dataTrain.head()


# Let's say we have reasons to believe the OverallQual, LotArea, TotalBsmtSF, 1stFlrSF, 2ndFlrSF,  GarageArea, and OpenPorchSF are some of the most relevant features for sale price prediction, and we want to build a four-degree (including interactions) linear model with elastic net regularization to predict sale price (**this is, of course, a huge simiplification of the actual problem, but here we just want to show how the regularized linear models work, and how well they can perform with limited information** ). Here is how we do it:

# In[ ]:


# Obtain training data
xTrain = dataTrain[['OverallQual','LotArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GarageArea', 'OpenPorchSF']].values
yTrain = dataTrain['SalePrice'].values.reshape(-1,1)


# ## Elastic Net

# In[ ]:


# Transform the data
poly2 = PolynomialFeatures(degree = 4, include_bias = False)
xTrainPoly = poly2.fit_transform(xTrain)
scaler = StandardScaler()
xTrainPolyStan = scaler.fit_transform(xTrainPoly)

# Fit the data
elasticReg = ElasticNet(alpha = 0.1, l1_ratio = 0.85)
elasticReg.fit(xTrainPolyStan, yTrain)

# evaluate performance on training set
yTrainHat = elasticReg.predict(xTrainPolyStan)

# calculate rmse based on log(price)
mse = mean_squared_error(np.log(yTrain), np.log(yTrainHat))
rmse = np.sqrt(mse)
print(rmse)


#  Let's plot predicted sale price and actual sale price:

# In[ ]:


x = np.linspace(0,800000,num=1000)
plt.plot(yTrainHat, yTrain,marker='o', linestyle = '', zorder = 1, color='b')
plt.plot(x, x, linestyle = '-',color='red',zorder=2,lw=3)
plt.xlabel('predicted sale price (dollars)', fontsize = 18)
plt.ylabel('actual sale price (dollars)', fontsize = 18)
plt.show()


# Not bad. Let's try the test set.

# In[ ]:


dataTest = pd.read_csv('../input/test.csv')
dataTest.head()


# In[ ]:


dataTest = dataTest[['Id','OverallQual','LotArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GarageArea','OpenPorchSF']]
dataTest.isnull().any()


# In[ ]:


# fill the nans with respective means.
dictMs = {'TotalBsmtSF':dataTest['TotalBsmtSF'].mean(skipna=True),
          'GarageArea':dataTest['GarageArea'].mean(skipna=True)}
dataTest = dataTest.fillna(value=dictMs)
dataTest.isnull().any()


# In[ ]:


xTest = dataTest[['OverallQual','LotArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GarageArea', 'OpenPorchSF']].values
xTestPoly = poly2.transform(xTest)
xTestPolyStan = scaler.transform(xTestPoly)
yTestHat = elasticReg.predict(xTestPolyStan)


# In[ ]:


sub = pd.DataFrame()
sub['Id'] = dataTest['Id']
sub['SalePrice'] = yTestHat
sub.to_csv('submission.csv',index=False)


# I did submit this result and my model's RMSE on the test set is 0.17055. For a simple regularized linear model with arbitrarily chosen indenpendent variables, this RMSE is good enough. <br/>
# Below I will talk about early stopping method, which ends up producing a similar but slightly worse RMSE on the test set.

# ## SGD and Early Stopping
# We have introduced stochastic gradient descent (SGD) in the previous machine learning tutorial [Machine Learning 1 - Regression, Gradient Descent](https://www.kaggle.com/fengdanye/machine-learning-1-regression-gradient-descent). For iterative learning algorithm like SGD, a good regularization technique is **early stopping**. To use this technique, we split the available data for training into a **training** set and a **validation** set: <br/>
# <img src="https://imgur.com/anuQyV9.jpg" width="500px"/>
# Then, for each epoch, we train the model on the training set and calculate RMSE for both the training set and the validation set. At first, both RMSE_training (blue curve below) and RMSE_validation (red curve below) should be overall decreasing. But starting from a certain epoch, the RMSE_validation will start increasing, indicating the model is now overfitting to the training set:
# <img src="https://upload.wikimedia.org/wikipedia/commons/f/fc/Overfitting.png" width="300px"/>
# The early stopping technique simply stops the training when RMSE_validation reaches its minimum. For SGD, randomness is present and early stopping will stop the training when RMSE_validation is above its minimum for a certain time and roll back to the parameters that give the minimum RMSE_validation value. <br/>
# Let's try SGD with early stopping on our dataset.

# In[ ]:


# Transform and standardize the data
poly2 = PolynomialFeatures(degree = 4, include_bias = False)
xTrainPoly = poly2.fit_transform(xTrain)
scaler = StandardScaler()
xTrainPolyStan = scaler.fit_transform(xTrainPoly)
yTrainLog = np.log(yTrain)


# Here I used log(yTrain) as the target values in training, as it seems to perform better than yTrain.

# In[ ]:


# shuffle data and split into training set and validation set
from sklearn.utils import shuffle
xShuffled, yShuffled = shuffle(xTrainPolyStan, yTrainLog)

train_ratio = 0.8
mTrain = np.int(len(xShuffled[:,0])*train_ratio) # 1168
print("Training sample size is: ", mTrain)

X_train_stan = xShuffled[0:mTrain]
Y_train = yShuffled[0:mTrain].ravel()
X_val_stan = xShuffled[mTrain:]
Y_val = yShuffled[mTrain:].ravel()


# With Scikit-Learn, you can actually do SGD with early stopping in one line, but we will write a more detailed version here for better understanding of the algorithm:

# In[ ]:


from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from copy import deepcopy


# In[ ]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

sgdReg = SGDRegressor(n_iter=1, warm_start = True, penalty=None, learning_rate = 'constant', eta0=0.00001)

mse_val_min = float("inf")
best_epoch = None
best_model = None
rmse_train = []
rmse_val = []

n_no_change = 0

for epoch in range(1,100000):
    sgdReg.fit(X_train_stan, Y_train)
    Y_train_predict = sgdReg.predict(X_train_stan)
    train_error = mean_squared_error(Y_train_predict,Y_train)
    rmse_train.append(np.sqrt(train_error))
    Y_val_predict = sgdReg.predict(X_val_stan)
    val_error = mean_squared_error(Y_val_predict, Y_val)
    rmse_val.append(np.sqrt(val_error))
    
    if val_error < mse_val_min:
        n_no_change = 0
        mse_val_min = val_error
        best_epoch = epoch
        best_model = deepcopy(sgdReg)
    else:
        n_no_change = n_no_change + 1
    
    if n_no_change >= 1000:
        print('Time to stop!')
        print('num epoch =', epoch)
        print('best epoch = ', best_epoch)
        break


# Notice that I used n_no_change >= 1000 as the stopping criterion to be on the safe side. A smaller number is doable.

# In[ ]:


# plot rmse
fig,ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12,4))
plt.subplots_adjust(wspace=0.5)

ax[0].plot(rmse_train, label = 'training')
ax[0].plot(rmse_val, label = 'validation')
ax[0].set_xlabel('epoch', fontsize = 18)
ax[0].set_ylabel('RMSE', fontsize = 18)
ax[0].legend(fontsize=14)

ax[1].plot(rmse_train[best_epoch-3:best_epoch+2], label = 'training')
ax[1].plot(rmse_val[best_epoch-3:best_epoch+2], label = 'validation')
ax[1].set_xlabel('epoch', fontsize = 18)
ax[1].set_ylabel('RMSE', fontsize = 18)
ax[1].set_xticks([0,1,2,3,4])
xticklabels = [str(e) for e in range(best_epoch-2,best_epoch+3)]
ax[1].set_xticklabels(xticklabels)
ax[1].plot(2,rmse_val[best_epoch-1],marker='o',color='r')
ax[1].text(2,rmse_val[best_epoch-1]-0.001,'minimum',color='r',fontsize=14)
ax[1].legend(fontsize=14)
plt.show()


# In[ ]:


# total rmse on the train + validation sets
yTrainHatLog = best_model.predict(xTrainPolyStan)
print(np.sqrt(mean_squared_error(yTrainHatLog,yTrainLog)))


# In[ ]:


# plot the train + validation sets' predicted sale price vs actual sale price
plt.plot(np.exp(yTrainHatLog),np.exp(yTrainLog), marker = 'o', linestyle='', color = 'b')
x = np.linspace(0,800000,num=1000)
plt.plot(x, x, linestyle = '-',color='red',zorder=2,lw=3)
plt.xlabel('predicted sale price (dollars)', fontsize = 18)
plt.ylabel('actual sale price (dollars)', fontsize = 18)
plt.show()


# For sklearn 0.20, early stopping is added to SGDRegressor. You can run the following code to achieve what we did above:
# > sgdReg = SGDRegressor(penalty=None, learning_rate = 'constant', eta0=0.00001, early_stopping = True, n_iter_no_change = 500, max_iter=100000) <br/>
# sgdReg.fit(xShuffled, yShuffled)

# If you run the above code multiple times, you will notice that the algorithm's performance fluctuate a bit with different shuffling results. Some selected training sets may lead to much worse RMSE than others. Let's try to improve the model with k-fold splitting.

# In[ ]:


from sklearn.model_selection import KFold


# In[ ]:


xShuffled, yShuffled = shuffle(xTrainPolyStan, yTrainLog)

sgdReg = SGDRegressor(n_iter=1, warm_start = True, penalty=None, learning_rate = 'constant', eta0=0.00001)

round_num = 0
best_epoch = None
best_model = None
rmse_train = []
rmse_val = []

kf = KFold(n_splits=5)

for train_index, val_index in kf.split(xShuffled):
    round_num = round_num + 1
    print("Round #", round_num)
    X_train_stan, X_val_stan = xShuffled[train_index], xShuffled[val_index]
    Y_train, Y_val = yShuffled[train_index].ravel(), yShuffled[val_index].ravel()
    
    print("Running...")
    mse_val_min = float("inf")
    n_no_change = 0

    for epoch in range(1,100000):
        sgdReg.fit(X_train_stan, Y_train)
        Y_train_predict = sgdReg.predict(X_train_stan)
        train_error = mean_squared_error(Y_train_predict,Y_train)
        rmse_train.append(np.sqrt(train_error))
        Y_val_predict = sgdReg.predict(X_val_stan)
        val_error = mean_squared_error(Y_val_predict, Y_val)
        rmse_val.append(np.sqrt(val_error))
    
        if val_error < mse_val_min:
            n_no_change = 0
            mse_val_min = val_error
            best_epoch = epoch
            best_model = deepcopy(sgdReg)
        else:
            n_no_change = n_no_change + 1
    
        if n_no_change >= 1000:
            print('Time to stop!')
            print('num epoch =', epoch)
            print('best epoch = ', best_epoch,', from round #', round_num)
            break


# In[ ]:


fig,ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6,4))
plt.subplots_adjust(wspace=0.5)

ax.plot(rmse_train, label = 'training')
ax.plot(rmse_val, label = 'validation')
ax.set_xlabel('epoch', fontsize = 18)
ax.set_ylabel('RMSE', fontsize = 18)
ax.legend(fontsize=14)
plt.show()


# Due to the random nature of SGD algorithm, you will have different results every time you run the code. Here is a plot of RMSE for one of my runs: <br/>
# <img src="https://imgur.com/6XyZLSa.png" width="300px"/>
# This is to illustrate how using k-fold splitting can help with model performance. As you can see, a big drop of RMSE takes place when we switch to the second fold (epoch=100000 in this case).

# In[ ]:


# total rmse on the train + validation sets
yTrainHatLog = best_model.predict(xTrainPolyStan)
print(np.sqrt(mean_squared_error(yTrainHatLog,yTrainLog)))


# In[ ]:


# plot the train + validation sets' predicted sale price vs actual sale price
plt.plot(np.exp(yTrainHatLog),np.exp(yTrainLog), marker = 'o', linestyle='', color = 'b')
x = np.linspace(0,800000,num=1000)
plt.plot(x, x, linestyle = '-',color='red',zorder=2,lw=3)
plt.xlabel('predicted sale price (dollars)', fontsize = 18)
plt.ylabel('actual sale price (dollars)', fontsize = 18)
plt.show()


# In[ ]:


# test set (data preprocessing - see Elastic Net part)
yTestHatLog = best_model.predict(xTestPolyStan)
sub = pd.DataFrame()
sub['Id'] = dataTest['Id']
sub['SalePrice'] = np.exp(yTestHatLog)
sub.to_csv('submission2.csv',index=False)


# Thought with lower training RMSE, the SGD + early stopping model actually does not perform as well as the elastic net model. I got RMSE=0.18 for one of my submissions on this SGD model. My speculation:
# * SGD model gives different model each time, due to its random nature. Ensemble learning should help with this probelm.
# * SGD model is overfitting to the training + validation sets, therefore perform not as well on the test set. **Early stopping really only prevents overfitting to the training set (the one with validation set taken out)**. Including a regularization term should help with this problem.

# In[ ]:




