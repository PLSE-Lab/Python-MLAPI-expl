#!/usr/bin/env python
# coding: utf-8

# ## Trying out a linear model: 
# 
# Author: Alexandru Papiu ([@apapiu](https://twitter.com/apapiu), [GitHub](https://github.com/apapiu))
#  
# Modified by Bill Holst for this project.
# 
# If you use parts of this notebook in your own scripts, please give some sort of credit (for example link back to this). Thanks!
# 
# There have been a few [great](https://www.kaggle.com/comartel/house-prices-advanced-regression-techniques/house-price-xgboost-starter/run/348739)  [scripts](https://www.kaggle.com/zoupet/house-prices-advanced-regression-techniques/xgboost-10-kfolds-with-scikit-learn/run/357561) on [xgboost](https://www.kaggle.com/tadepalli/house-prices-advanced-regression-techniques/xgboost-with-n-trees-autostop-0-12638/run/353049) already so I'd figured I'd try something simpler: a regularized linear regression model. Surprisingly it does really well with very little feature engineering. The key point is to to log_transform the numeric variables since most of them are skewed.

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import os
print(os.listdir("../input"))
local = 0
if (local):
    train = pd.read_csv("input/train.csv")
    test = pd.read_csv("input/test.csv")
    # os.chdir('C:\\Users\\wjhol\\Documents\\GitHub\\DataScienceCurriculum\\DataScienceCurriculum\\DiamondRegression')
    # print(os.getcwd())
else:
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


# remove the ID and price
train_clean = train.drop(['ID','price'],1)
test_clean = test.drop('ID',1)
test_clean.head()
train_clean.shape


# In[ ]:





# In[ ]:


all_data = pd.concat((train_clean[:],test_clean[:]))
all_data["caret_sqroot"] = np.sqrt(all_data["carat"])
all_data["caret_cubtroot"] = all_data.carat ** (1/3)
all_data.shape


# ### Data preprocessing: 
# We're not going to do anything fancy here: 
#  
# - First I'll transform the skewed numeric features by taking log(feature + 1) - this will make the features more normal    
# - Create Dummy variables for the categorical features    
# - Replace the numeric missing values (NaN's) with the mean of their respective columns

# In[ ]:


matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"price":train["price"], "log(price + 1)":np.log1p(train["price"])})
prices.hist(bins=40, grid=True)


# In[ ]:


#log transform the target:
train["logprice"] = np.log1p(train["price"])

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
print(numeric_feats)


# In[ ]:





# In[ ]:



#skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
#skewed_feats = skewed_feats[skewed_feats > 0.75]
#skewed_feats = skewed_feats.index
#print(skewed_feats)

#all_data[skewed_feats] = np.log1p(all_data[skewed_feats])


# Now set the catagorical variables to one-hot encoded values. Note: this function is smart enough to only work on the catagorical values like color and clarity.

# In[ ]:


all_data = pd.get_dummies(all_data)
all_data.head()


# In[ ]:


#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())


# In[ ]:


#creating matrices for sklearn:

print (train.shape[0])
print (all_data.shape[0])
# select the all_data values with the number of rows in the train dataset; test is everything else
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
train_price = train["logprice"]
print(train_price.head())
y = train_price


# ### Models
# 
# Now we are going to use regularized linear regression models from the scikit learn module. I'm going to try both l_1(Lasso) and l_2(Ridge) regularization. I'll also define a function that returns the cross-validation rmse error so we can evaluate our models and pick the best tuning par

# In[ ]:


from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV, LinearRegression
from sklearn.model_selection import cross_val_score
# http://localhost:8888/notebooks/notebook-Copy3.ipynb#
def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

def rmse (ypred, yval):
    rmseval = np.sqrt((yval - ypred)^2)
    return (rmseval)


# The first model we investigate in a simple linear regression.

# In[ ]:


lm = LinearRegression()
lm_fit = lm.fit(X_train,y)
lm_pred = np.array(lm_fit.predict (X_train))
#val = rmse(lm_pred,y)
#print (val)
#print(lm_pred.head())
#print (y.head())

print (lm_fit)
print(round(rmse_cv(lm_fit).mean(),4))

# 


# In[ ]:


predp = pd.DataFrame(lm_pred)


# In[ ]:


predp.columns = ['x']
predp.set_index ('x')


# In[ ]:


from scipy.interpolate import interp1d
cdf = predp.sort_values('x').reset_index()
cdf['p'] = cdf.index / float(len(cdf) - 1)
# setup the interpolator using the value as the index
interp = interp1d(cdf['x'], cdf['p'])

# a is the value, b is the percentile
print (cdf.head())
#Now we can see that the two functions are inverses of each other.

print (predp['x'].quantile(0.57))
print (interp(8.0403611))
#interp(0.61167933268395969)
#array(0.57)
print (interp(predp['x'].quantile(0.43)))
#array(0.43)


# In[ ]:


cdfx


# In[ ]:


len(x_values)


# In[ ]:


sz = len(lm_pred)
sz


# This is decent for a simple linear model with no tuning. Let's plot the residuals.

# In[ ]:


from sklearn.metrics import r2_score
lm_predict = lm.predict (X_train)
#print (lm_predict)
#print (y)
print(r2_score(lm_predict,y))


# In[ ]:


plt.scatter(lm_predict,lm_predict - y,c = 'b',s=40,alpha= 1.0)
plt.hlines (y = 0, xmin=5, xmax = 11)
plt.title ('Residuals using training data')
plt.ylabel ('Residuals')


# In[ ]:


model_ridge = Ridge()


# The main tuning parameter for the Ridge model is alpha - a regularization parameter that measures how flexible our model is. The higher the regularization the less prone our model will be to overfit. However it will also lose flexibility and might not capture all of the signal in the data.

# In[ ]:


alphas = [0.00001,0.0001, 0.001, 0.005, 0.01, 0.03, 0.05, 0.08, 0.1, .25, .5, 1]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]


# In[ ]:


cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - RMSE vs Alpha")
plt.xlabel("alpha")
plt.ylabel("rmse")


# Note the U-ish shaped curve above. When alpha is too large the regularization is too strong and the model cannot capture all the complexities in the data. If however we let the model be too flexible (alpha small) the model begins to overfit. A value of alpha = 10 is about right based on the plot above.

# In[ ]:


cv_ridge.min()


# So for the Ridge regression we get a rmsle of about 0..1271, similar to the linear regression.
# 
# Let' try out the Lasso model. We will do a slightly different approach here and use the built in Lasso CV to figure out the best alpha for us. For some reason the alphas in Lasso CV are really the inverse or the alphas in Ridge.

# In[ ]:


model_lasso = LassoCV(alphas = [10, 5,1, 0.1, 0.001, 0.0005],tol = 0.001).fit(X_train, y)


# In[ ]:


rmse_cv(model_lasso).mean()


# The lasso model does not perform quite as well as the other two. However there is another feature about the Lasso is that it does feature selection for you - setting coefficients of features it deems unimportant to zero. Let's take a look at the coefficients:

# In[ ]:


coef = pd.Series(model_lasso.coef_, index = X_train.columns)
print (coef.sort_values(ascending=False))


# In[ ]:


print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")


# Good job Lasso.  One thing to note here however is that the features selected are not necessarily the "correct" ones - especially since there are a lot of collinear features in this dataset. One idea to try here is run Lasso a few times on boostrapped samples and see how stable the feature selection is.

# We can also take a look directly at what the most important coefficients are:

# In[ ]:


imp_coef = pd.concat([coef.sort_values().head(12),
                     coef.sort_values().tail(11)])


# In[ ]:


matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")


# The most important positive features are the square and cube root of carat. This definitely makes sense because carat is a measure of weight, which is a 3 dimensional attribute. Then a few other  location and quality features contributed positively. Some of the negative features make less sense and would be worth looking into more - it seems like they might come from unbalanced categorical variables.
# 
#  Also note that unlike the feature importance you'd get from a random forest these are _actual_ coefficients in your model - so you can say precisely why the predicted price is what it is. The only issue here is that we log_transformed both the target and the numeric features so the actual magnitudes are a bit hard to interpret. 

# In[ ]:


#let's look at the residuals as well:
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

preds = pd.DataFrame({"preds":model_lasso.predict(X_train), "true":y})
preds["residuals"] = preds["true"] - preds["preds"]
preds.plot(x = "preds", y = "residuals",kind = "scatter")


# The residual plot looks pretty good.To wrap it up let's predict on the test set and submit on the leaderboard:

# ### Adding an xgboost model:

# Let's add an xgboost model to our linear model to see if we can improve our score:

# In[ ]:


import xgboost as xgb


# In[ ]:



dtrain = xgb.DMatrix(X_train, label = y)
dtest = xgb.DMatrix(X_test)

params = {"max_depth":2, "eta":0.1}
model = xgb.cv(params, dtrain,  num_boost_round=1000, early_stopping_rounds=100)


# In[ ]:


model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()


# In[ ]:


model_xgb = xgb.XGBRegressor(n_estimators=1000, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
model_xgb.fit(X_train, y)


# In[ ]:


xgb_preds = np.expm1(model_xgb.predict(X_test))
lasso_preds = np.expm1(model_lasso.predict(X_test))


# In[ ]:


predictions = pd.DataFrame({"xgb":xgb_preds, "lasso":lasso_preds})
predictions.plot(x = "xgb", y = "lasso", kind = "scatter")


# Many times it makes sense to take a weighted average of uncorrelated results - this usually imporoves the score although in this case it doesn't help that much.

# In[ ]:


preds = 0.7*lasso_preds + 0.3*xgb_preds


# In[ ]:


solution = pd.DataFrame({"ID":test.ID, "price":preds})
solution.to_csv("ridge_sol.csv", index = False)


# ### Trying out keras?
# 
# Feedforward Neural Nets doesn't seem to work well at all...I wonder why.

# In[ ]:


from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l1
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[ ]:


X_train = StandardScaler().fit_transform(X_train)


# In[ ]:


X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, random_state = 3)


# In[ ]:


X_tr.shape


# In[ ]:


X_tr


# In[ ]:


model = Sequential()
#model.add(Dense(256, activation="relu", input_dim = X_train.shape[1]))
model.add(Dense(1, input_dim = X_train.shape[1], W_regularizer=l1(0.001)))

model.compile(loss = "mse", optimizer = "adam")


# In[ ]:


model.summary()


# In[ ]:


hist = model.fit(X_tr, y_tr, validation_data = (X_val, y_val))


# In[ ]:


pd.Series(model.predict(X_val)[:,0]).hist()

