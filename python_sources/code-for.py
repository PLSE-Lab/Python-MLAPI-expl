#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Preprocessing from: https://www.kaggle.com/apapiu/regularized-linear-models
#Kaggle contest: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
import pandas as pd
import numpy as np
import seaborn as sb

#My set up has issues with
#plotting, ignore the matpltlib.use() line
#if your's does not
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from scipy.stats import skew
from scipy.stats.stats import pearsonr

#Comment this out to run code outside of notebook
#set 'png' here when working on notebook
#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'png'")
#get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, Lars, BayesianRidge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error


# Set up complete, all libraries for models obtained
# and other configurations set up.

# In[2]:


#Path of file to read
train_path = '../input/house-prices-advanced-regression-techniques/train.csv'
test_path = '../input/house-prices-advanced-regression-techniques/test.csv'

train_data_raw = pd.read_csv(train_path)
test_data_raw = pd.read_csv(test_path)
all_feature_data_raw = pd.concat((train_data_raw.loc[:, 'MSSubClass':'SaleCondition'], test_data_raw.loc[:, 'MSSubClass':'SaleCondition']))

train_data = train_data_raw


# In[3]:


matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"price":train_data_raw["SalePrice"], "log(price + 1)":np.log1p(train_data_raw["SalePrice"])})
prices.hist()

#normalize SalePrice for prettier curve
train_data["SalePrice"] = np.log1p(train_data_raw["SalePrice"])


# In[4]:


#Figure out which features need preprocessing
# MSSubClass actually an enum - needs 1 hot encoding
all_data = pd.get_dummies(all_feature_data_raw, columns=["MSSubClass"])
train_data = pd.get_dummies(train_data, columns=["MSSubClass"])


# In[5]:


#now need to normalize (skew) the features
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
numeric_feats = all_data[numeric_feats].dtypes[all_data[numeric_feats].dtypes != "uint8"].index

#compute skewness
skewed_feats = train_data[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

for feat in skewed_feats:
    all_data[feat] = np.log1p(all_data[feat])
    


# In[6]:


#need to make non-numeric or non-sequential features into something we can do math on
all_data = pd.get_dummies(all_data)


# In[7]:


#fill in missing data
all_data = all_data.fillna(all_data.mean())


# In[8]:


#creating our basic Xs and y now that data is cleaned
X_train = all_data[:train_data.shape[0]]
X_test = all_data[train_data.shape[0]:]
y = train_data.SalePrice


# RMSE for ridge model on Kaggle is: 0.56779

# In[9]:


def fit_predict(model, X_train, y):
    model.fit(X_train, y)
    model_predict = model.predict(X_train)
    model_rmse = np.sqrt(mean_squared_error(y, model_predict))
    return model_rmse


# In[10]:


def print_price_to_file(name, model, X):
    model_predict = model.predict(X)
    file = open(name+".csv", "w+")
    id = test_data_raw["Id"]
    file.write("Id,SalePrice\n")
    for i in range(len(id)):
        string = str(id[i])
        string += ","
        string += str(np.expm1(model_predict[i]))
        string += "\n"
        file.write(string)
    file.close()  


# In[11]:


#Lets try a basic Linear Regression model

basic_linear_model = LinearRegression()
print("RMSE for basic linear regression model is: ", fit_predict(basic_linear_model, X_train, y))
print_price_to_file("basic_linear_model_predictions", basic_linear_model, X_test)


# RMSE for basic linear model on Kaggle is: 0.55334

# In[12]:


#test file 123
#file = open("basic_linear_model_predictions.csv", "r")
#print(file.read())
#file.close()


# In[13]:


#Lets try a basic Ridge model

basic_ridge_model = Ridge()
print("RMSE for basic ridge regression model is: ", fit_predict(basic_ridge_model, X_train, y))
print_price_to_file("basic_ridge_model_predictions", basic_ridge_model, X_test)


# RMSE for ridge model on Kaggle is: 0.12575

# In[14]:


#Ridge model with alpha = 0.1

alpha_ridge_model = Ridge(alpha=0.1)
print("RMSE for ridge model of alpha = 0.1 is: ", fit_predict(alpha_ridge_model, X_train, y))
print_price_to_file("alpha_ridge_model_predictions", alpha_ridge_model, X_test)


# RMSE for alpha = 0.1 ridge model on Kaggle is: 0.13032

# In[15]:


#Lets try a basic Lasso model

basic_lasso_model = Lasso()
print("RMSE for basic lasso is: ", fit_predict(basic_lasso_model, X_train, y))
print_price_to_file("basic_lasso_model_predictions", basic_lasso_model, X_test)


# RMSE for lasso model on Kaggle is: 0.27552

# In[16]:


# Lets try out Ridge and Lasso with Cross Validation (CV)
# Lets set some alphas to play with

alphas_r = [2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20]
alphas_l = [0.00045, 0.0005, 0.00052, 0.00053, 0.00054, 0.00056, 0.00058, 0.0006]


# In[17]:


cv_ridge_model_rmse_a = []

for a in alphas_r:
    cv_ridge_model = Ridge(alpha=a)
    
    cv_ridge_rmse = np.sqrt(-cross_val_score(cv_ridge_model, X_train, y, scoring="neg_mean_squared_error", cv = 5))   

    cv_ridge_model_rmse_a.append(cv_ridge_rmse.mean())

cv_ridge_plot = pd.Series(cv_ridge_model_rmse_a, index = alphas_r)

cv_ridge_plot.plot(title = "RMSE Score for RidgeCV")
plt.xlabel("alpha")
plt.ylabel("cv_rmse")


# In[18]:


cv_lasso_model_rmse_a = []

for a in alphas_l:
    cv_lasso_model = Lasso(alpha=a)
    
    cv_lasso_rmse = np.sqrt(-cross_val_score(cv_lasso_model, X_train, y, scoring="neg_mean_squared_error", cv = 5))

    cv_lasso_model_rmse_a.append(cv_lasso_rmse.mean())

cv_lasso_plot = pd.Series(cv_lasso_model_rmse_a, index = alphas_l)

cv_lasso_plot.plot(title = "RMSE Score for LassoCV")
plt.xlabel("alpha")
plt.ylabel("cv_rmse")


# In[19]:


cv_lasso_model_coef_a = []

for a in alphas_l:
    cv_lasso_model = Lasso(alpha=a).fit(X_train, y)
        
    coef = pd.Series(cv_lasso_model.coef_, index = X_train.columns)
    cv_lasso_model_coef = sum(coef != 0)

    cv_lasso_model_coef_a.append(cv_lasso_model_coef)

cv_lasso_plot_coef = pd.Series(cv_lasso_model_coef_a, index = alphas_l)

cv_lasso_plot_coef.plot(title = "Coefficients for Lasso")
plt.xlabel("alpha")
plt.ylabel("coefs")


# In[20]:


print("min of cv_ridge: ",cv_ridge_plot.min())
print("alpha min value of ridge is: ", cv_ridge_plot.idxmin())
print("min of cv_lasso",cv_lasso_plot.min())
print("alpha min value of lasso is: ", cv_lasso_plot.idxmin())


# In[21]:


#Lets try a Ridge model with the above alpha

min_alpha_ridge_model = Ridge(alpha=11)
print("RMSE for ridge with an alpha = 11 is: ", fit_predict(min_alpha_ridge_model, X_train, y))
print_price_to_file("min_alpha_ridge_model_predictions", min_alpha_ridge_model, X_test)


# RMSE for ridge model with alpha = 11 on Kaggle is: 0.12114

# In[22]:


#Lets try a Lasso model with the above alpha

min_alpha_lasso_model = Lasso(alpha=0.00053)
print("RMSE for lasso with an alpha = 0.00053 is: ", fit_predict(min_alpha_lasso_model, X_train, y))
print_price_to_file("min_alpha_lasso_model_predictions", min_alpha_lasso_model, X_test)


# RMSE for lasso model with alpha = 0.00053 on Kaggle is: 0.12068

# In[23]:


#Taking my Lasso output from above, throw that into a Ridge model

lasso_train_data = {'lasso_results': min_alpha_lasso_model.predict(X_train)}
lasso_test_data = {'lasso_results': min_alpha_lasso_model.predict(X_test)}

lasso_train_data_df = pd.DataFrame(data = lasso_train_data)
lasso_test_data_df = pd.DataFrame(data = lasso_test_data)

Xl_train = pd.concat([X_train, lasso_train_data_df], axis=1)
Xl_test = pd.concat([X_test, lasso_test_data_df], axis=1)

lassod_alpha_ridge_model = Ridge(alpha=11)
print("RMSE for ridge with an alpha = 11 with a lasso result column is: ", fit_predict(lassod_alpha_ridge_model, Xl_train, y))
print_price_to_file("lassod_alpha_ridge_model_predictions", lassod_alpha_ridge_model, Xl_test)


# RMSE for ridge model with alpha = 0.00053 and lasso input on Kaggle is: 0.12065

# In[24]:


#Taking my Lasso and Ridge output from above, throw that into a Ridge model (stacked!)

ridge_train_data = {'ridge_results': min_alpha_ridge_model.predict(X_train)}
ridge_test_data = {'ridge_results': min_alpha_ridge_model.predict(X_test)}

ridge_train_data_df = pd.DataFrame(data = ridge_train_data)
ridge_test_data_df = pd.DataFrame(data = ridge_test_data)

Xlr_train = pd.concat([Xl_train, ridge_train_data_df], axis=1)
Xlr_test = pd.concat([Xl_test, ridge_test_data_df], axis=1)

stacked_alpha_ridge_model = Ridge(alpha=11)
print("RMSE for ridge with an alpha = 11 and stacking is: ", fit_predict(stacked_alpha_ridge_model, Xlr_train, y))
print_price_to_file("stacked_alpha_ridge_model_predictions", stacked_alpha_ridge_model, Xlr_test)


# RMSE for ridge model with alpha = 0.00053 and stacking on Kaggle is: 0.12088

# In[25]:


#Trying out the built in CV versions of ridge and lasso

CV_ridge_model = RidgeCV(alphas=alphas_r, cv=5)
CV_lasso_model = LassoCV(alphas=alphas_l, cv=5)
print("RMSE for ridge with CV is: ", fit_predict(CV_ridge_model, X_train, y))
print("RMSE for lasso with CV is: ", fit_predict(CV_lasso_model, X_train, y))
print_price_to_file("CV_ridge_model_predictions", CV_ridge_model, X_test)
print_price_to_file("CV_lasso_model_predictions", CV_lasso_model, X_test)


# RMSE for ridge model with CV on Kaggle is: 0.12114
# RMSE for ridge model with CV on Kaggle is: 0.12072

# In[26]:


#Trying out running lasso ontop of lasso output 

lassod_alpha_lasso_model = Lasso(alpha=0.00053)
print("RMSE for lasso with an alpha = 0.00053 with a lasso result column is: ", fit_predict(lassod_alpha_lasso_model, Xl_train, y))
print_price_to_file("lassod_alpha_lasso_model_predictions", lassod_alpha_lasso_model, Xl_test)


# RMSE for lasso model with lasso input on Kaggle is: 0.12074

# In[27]:


#Trying out running lasso ontop of stacked output 

stacked_alpha_lasso_model = Lasso(alpha=0.00053, max_iter=1500)
print("RMSE for lasso with an alpha = 0.00053 with stack data input is: ", fit_predict(stacked_alpha_lasso_model, Xlr_train, y))
print_price_to_file("stacked_alpha_lasso_model_predictions", stacked_alpha_lasso_model, Xlr_test)


# RMSE for lasso model with lasso input on Kaggle is: 0.12037

# In[28]:


#Trying out basic Lars model

n_nonzero_coefs= [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101]

lars_model_rmse_a = []

for n in n_nonzero_coefs:
    lars_model = Lars(n_nonzero_coefs=n).fit(X_train, y)
        
    lars_model_rmse = np.sqrt(mean_squared_error(y, lars_model.predict(X_train)))

    lars_model_rmse_a.append(lars_model_rmse.mean())

lars_model_plot_rmse = pd.Series(lars_model_rmse_a, index = n_nonzero_coefs)

lars_model_plot_rmse.plot(title = "RMSE for # of Coefficients for Lar")
plt.xlabel("coefs")
plt.ylabel("rmse")

basic_lars_model = Lars(n_nonzero_coefs=101)
print("RMSE for basic lar model is: ", fit_predict(basic_lars_model, X_train, y))
print_price_to_file("basic_lars_model_predictions", basic_lars_model, X_test)


# RMSE for a basic lars model on Kaggle is: 0.12465

# In[29]:


#Trying out basic Baysian Ridge with lasso results model

basic_br_model = BayesianRidge()
print("RMSE for basic baysian ridge model with lasso input is: ", fit_predict(basic_br_model, X_train, y))
print_price_to_file("basic_br_model_predictions", basic_br_model, X_test)


# RMSE for a basic  baysian ridge model on Kaggle is: 0.12172

# In[30]:


#Trying out basic Baysian Ridge with lasso results model

lassoed_br_model = BayesianRidge()
print("RMSE for basic baysian ridge model with lasso input is: ", fit_predict(lassoed_br_model, Xl_train, y))
print_price_to_file("lassoed_br_model_predictions", lassoed_br_model, Xl_test)


# RMSE for a lassoed baysian ridge model on Kaggle is: 0.12141

# In[31]:


#Taking my Lars output from above, throw that into a Lasso with the other outputs model (stacked!)

lars_train_data = {'lars_results': basic_lars_model.predict(X_train)}
lars_test_data = {'lars_results': basic_lars_model.predict(X_test)}

lars_train_data_df = pd.DataFrame(data = lars_train_data)
lars_test_data_df = pd.DataFrame(data = lars_test_data)

Xlrl_train = pd.concat([Xlr_train, lars_train_data_df], axis=1)
Xlrl_test = pd.concat([Xlr_test, lars_test_data_df], axis=1)

stacked2_alpha_lasso_model = Lasso(alpha=0.00053, max_iter=1500)
print("RMSE for lasso with an alpha = 0.00053 with stack ridge, lasso and lars data input is: ", fit_predict(stacked2_alpha_lasso_model, Xlrl_train, y))
print_price_to_file("stacked2_alpha_lasso_model_predictions", stacked2_alpha_lasso_model, Xlrl_test)


# RMSE for a very stacked(lasso, ridge, lars) lasso model on Kaggle is: 0.12057

# 
# Best score was Ridge and Lasso data from best model added as features and run through another Lasso model.
# 
# RMSE score on training data: 0.1027218828756981
# 
# Kaggle RMSE score: 0.12037
# 
# Position on leaderboard: 1138 (see ozm59)
