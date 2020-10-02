#!/usr/bin/env python
# coding: utf-8

# # Predicting Ames House Prices

# ## 1. Introduction
# We look at the Ames Housing Dataset in order to predict the house prices given 80 variables or features. A lot of trial and error towards went into coming up with this notebook this includes work in the exploratory data analysis stage, testing various models 
# 
# ![Ames,Iowa 2](https://ap.rdcpix.com/1728889637/7ee96ed20d1b1e15104ec519e3983f84l-m1xd-w1020_h770_q80.jpg)
# source:https://www.realtor.com/realestateandhomes-detail/3835-Brookdale-Ave_Ames_IA_50010_M86459-29535#photo1

# In[ ]:


# Main Libraries 
import numpy as np 
import pandas as pd 


# Visualization Libraries
import matplotlib.pyplot as plt  
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
#color = sns.color_palette()
#sns.set_style('darkgrid')
#import warnings
#def ignore_warn(*args, **kwargs):
#    pass
#warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


# Scipy Libraries
from scipy import stats
from scipy.stats import norm, skew #for some statistics
from scipy.special import boxcox1p


# Sklearn Libraries
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

import os


# ## 2. Read Data

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.head()


# In[ ]:


# Define Global Variables
dependent_var = 'SalePrice'
Id = 'Id'

#Save the 'Id' column
train_ID = train_df[Id]
test_ID = test_df[Id]

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train_df.drop(Id, axis = 1, inplace = True)
test_df.drop(Id, axis = 1, inplace = True)


# ## 3. Feature Correlation
# Before perfoming EDA we first observe and the correlation between the features.This will enable us to infer which particular features have an impact on the sale price of the house and a way to consider outliers and skewness of the data.

# In[ ]:


# Get correlation matrix
corrmat = train_df.corr()

cols = corrmat.nlargest(10, dependent_var)[dependent_var].index # Get 10 most higly correlated columns with the dependent_var
cm = np.corrcoef(train_df[cols].values.T) # Correlation coefficients of cols

# Plotting the correlation heatmap
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(12, 9))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
               xticklabels=cols.values)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()
figure = hm.get_figure()    
figure.savefig('hm.png', dpi=400)


# ### Outliers
# From the correlation heatmap we determine the outliers that we remove from the higlyy correlated column with the dependent variable. Below we plot the correlation of the dependent variable (SalePrice) vs the GrLiveArea before the outliers are removed and after they are removed.  

# In[ ]:


plt.subplots(figsize=(18, 5))

plt.subplot(1, 2, 1)
before_plot = sns.regplot(x=train_df['GrLivArea'], y=train_df[dependent_var], fit_reg=False).set_title("Before")

# Delete outliers
train_df1 = train_df[~((train_df['GrLivArea'] > 4500) & (train_df[dependent_var] < 300000))]

plt.subplot(1, 2, 2)                                                                                
after_plot = sns.regplot(x=train_df1['GrLivArea'], y=train_df1[dependent_var], fit_reg=False).set_title("After")
plt.tight_layout()
plt.savefig('Outliers.png')


# ### Observation of the dependent variable
# Looking at the distribution of the target variable

# In[ ]:


#plotting our dependent variable it can be seen that it is skewed to the left 
f, ax = plt.subplots(figsize=(11, 6))
sns.distplot(train_df[dependent_var], hist=True, kde=True, 
             bins= 40, color = 'green',
             hist_kws={'edgecolor':'black'})
sns.set(font_scale=0.8)
plt.savefig('dist1.png')


# We then normalize the distribution of the by applying the log1p function to transfrom the dependent variable 

# In[ ]:


#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train_df[dependent_var] = np.log1p(train_df[dependent_var])

f, ax = plt.subplots(figsize=(11, 6))
sns.distplot(train_df[dependent_var], hist=True, kde=True, 
             bins= 40, color = 'green',
             hist_kws={'edgecolor':'black'})
sns.set(font_scale=2.2)
plt.savefig('dist2.png')


# ### Combine the train and test dataset

# In[ ]:


#ntrain = train_df.shape[0]
#ntest = test_df.shape[0]
y_train = train_df[dependent_var].values
#y_train = train_df.SalePrice.values
all_data = pd.concat((train_df, test_df)).reset_index(drop=True)
all_data.drop([dependent_var], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))


# ## 4. Exploratory Data Analysis

# ### Missing Values
# Check all missing values and drop columns with no missing values so that we remain with the ones with missing values and print out the columns

# In[ ]:


#Check remaining missing values if any 
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()


# #### Check the skewness of the Data

# In[ ]:


# Checking which columns are the highly skewed features
numeric_vars = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_vars = all_data[numeric_vars].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_vars})
skewness.head(15)


# In[ ]:


skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))


skewed_vars = skewness.index
lam = 0.15
for var in skewed_vars:
    #all_data[feat] += 1
    all_data[var] = boxcox1p(all_data[var], lam)


# #### Change categorical variables to numerical ones

# In[ ]:


all_data = pd.get_dummies(all_data)
all_data.shape


# In[ ]:


all_data = all_data.fillna(all_data.mean())


# In[ ]:


#Check remaining missing values if any 
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()


# In[ ]:


train_df = all_data[:train_df.shape[0]]
test_df = all_data[train_df.shape[0]:]
train_df.shape


# # 5. Modelling
# After getting all of our data cleaned we then proceed to creating model for prediction. We apply four models to fit the data and test their accuracy using the cross-validation function that is defined below. We then compute the average between the four models and predict the final price

# ### Cross Validation
# As with any predictive model we want to best avoid overfitting and underfitting of the model. The Validation technique helps to evaluate the quality of the model
# ![kfold](https://cdn-images-1.medium.com/max/800/1*4G__SV580CxFj78o9yUXuQ.png)  
#               source: https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6

# In[ ]:


X_train,X_test,Y_train,y_test = train_test_split(train_df,y_train,test_size = 0.2,random_state= 0)


# In[ ]:


n_splits = 5  # Split the data into 5 subsets

def cross_val(model):
    kf = KFold(n_splits=5, shuffle=True, random_state=42).get_n_splits(train_df.values)
    rmse = np.sqrt(-cross_val_score(model, train_df.values, y_train,scoring="neg_mean_squared_error", cv = kf))
    return rmse


# #### (ii). Kernel Ridge Regression

# In[ ]:


krr = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
score = cross_val(krr)
print("Kernel Ridge score: {:.2f}\n".format(score.mean()))


# #### (i). Lasso Regression

# In[ ]:


lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
score = cross_val(lasso)
print("Lasso score: {:.2f}\n".format(score.mean()))


# #### (iii). Elastic Net Regression

# In[ ]:


elas_net = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
score = cross_val(elas_net)
print("Elastic Net score: {:.2f} \n".format(score.mean()))


# #### (iv) Gradient Boosting

# In[ ]:


g_boost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
score = cross_val(g_boost)
print("Gradient Boosting score: {:.2f}\n".format(score.mean()))


# ### Fitting the models

# In[ ]:


lasso_model = lasso.fit(train_df.values,y_train)
elas_net_model = elas_net.fit(train_df.values,y_train)
krr_model = krr.fit(train_df.values,y_train)
g_boost_model = g_boost.fit(train_df.values,y_train)


# ### Visualisation of predict and test data

# In[ ]:


y_lasso_pred = lasso_model.predict(X_test)
y_elasn_pred = elas_net_model.predict(X_test)
y_krr_pred = krr_model.predict(X_test)
y_gboost_pred = g_boost_model.predict(X_test)
#plt.subplots(2,2,sharey =True, sharex=True)
fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(15,5))
axs[0].plot(y_test, label='Test', color='r')
axs[0].plot(y_krr_pred, label='Predict',color ='b')
axs[1].plot(y_test, label='Test', color='r')
axs[1].plot(y_lasso_pred, label='Predict',color ='g')
axs[0].title.set_text('Kernel Ridge Regression')
axs[1].title.set_text('Lasso')
axs[0].legend()
axs[1].legend()
axs[0].set_ylabel('SalePrice')
plt.savefig('pred1.png')


# In[ ]:


y_krr_pred = krr_model.predict(X_test)
y_gboost_pred = g_boost_model.predict(X_test)
#plt.subplots(2,2,sharey =True, sharex=True)
fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(18,7))
axs[0].plot(y_test, label='Test', color='orange')
axs[0].plot(y_elasn_pred, label='Predict',color = 'b')
axs[1].plot(y_test, label='Test', color='orange')
axs[1].plot(y_gboost_pred, label='Predict',color = 'g')
axs[0].title.set_text('Elastic')
axs[1].title.set_text('GBoost')
axs[0].legend()
axs[1].legend()
axs[0].set_ylabel('SalePrice')
plt.savefig('pred12.png')


# ### Final Model
# The final model is the average of the four models that inversely transformed back

# In[ ]:


final_model = (np.expm1(lasso_model .predict(test_df.values)) 
           + np.expm1(elas_net_model.predict(test_df.values)) 
           + np.expm1(krr_model.predict(test_df.values)) 
           + np.expm1(g_boost_model.predict(test_df.values))) / 4


# ### Printing out the first 10 values of each model

# In[ ]:


print('Lasso', '\t\t', 'Kernel Ridge', '\t', 'Elastic Net', '\t', 'Gradient Boost')
for i in range(10):
    print(round(np.expm1(lasso_model .predict(test_df.values))[i]), '\t',
          round(np.expm1(elas_net_model.predict(test_df.values))[i]), '\t',
          round(np.expm1(krr_model.predict(test_df.values))[i]), '\t',
          round(np.expm1(g_boost_model.predict(test_df.values))[i]))


# ### Create a submission file that will be submmited to the competition
# The csv file contains two columns: the test Id column from the test dataframe and the final model predictions of the SalePrice model.

# In[ ]:


sub = pd.DataFrame()
sub[Id] = test_ID
sub[dependent_var] = final_model
sub.to_csv('submission.csv',index=False)


# ## 6. Conclusion 
# Looking at the four models we have used the accuracy score for each we find that all the models perform better hence taking the average predictions of the four combined models. In theory Gradient Boosting should perform better than other models but this is not the case in our analysis since the accuracy score of the Gradient boosting is higher than the three other models. From the leaaderboard score on Kaggle our model predictions were good.

# ## 7. References
# 
# Box Cox Transformation:  
# 1.https://www.statisticshowto.datasciencecentral.com/box-cox-transformation/  
# 2.https://www.isixsigma.com/tools-templates/normality/making-data-normal-using-box-cox-power-transformation/
# 
# Lasso Regression:  
# 1.https://www.statisticshowto.datasciencecentral.com/lasso-regression/  
# 2.https://www.coursera.org.lecture/machine-learning-data-analysis/what-is-lasso-regression-0Kly7
# 
# Ridge Regression:  
# 1.https://towardsdatascience.com/ridge-and-lasso-regression-a-complete-guide-with-python-scikit-learn-e20e34bcbf0b
