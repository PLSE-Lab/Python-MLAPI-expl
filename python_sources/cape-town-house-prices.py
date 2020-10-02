#!/usr/bin/env python
# coding: utf-8

# # The Cape Town Property Prices
#  
#  * On This Notebook I will Be Pridicting The House Prices Of Cape Town Using Machine Learning 

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy import stats
from scipy.stats import norm, skew  
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc


# In[ ]:


df_train=pd.read_csv('../input/test.csv')
df_test=pd.read_csv('../input/train.csv')


# In[ ]:


df_train.head()


# In[ ]:


df_train.columns


# #### Checking Columns On The Dataset 

# In[ ]:


df_test.columns


# #### To Make Things Easy I Concatinate the Test And Train For Easy Data Cleaning Not Including The Depended Variable 

# In[ ]:


df=pd.concat([df_train, df_test])


# In[ ]:


df.columns


# In[ ]:


df.info()


# #### Checking The Missing Values In The Dataset 

# In[ ]:


df.isnull().sum()


# #### This Are The Columns That Contain The Missing Values 
#  * bathroom 
#  * bedroom 
#  * buildingSize 
#  * erfSize  
#  * garage   

# #### Let Us Check The Corroletion to Our Target Variable 

# In[ ]:


matrix = df_train.corr()
mask = np.zeros_like(matrix, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(25, 15))
sns.heatmap(matrix, mask=mask, vmax=0.8, vmin=0.05, annot=True);


# #### Only Concenatrating On The Scale Above 0.5
# From The Above Correlation The Columns That have High Correlations Are
# 
# * buildingsize and bathroom
# * buildingsize and bedroom
# * bedroom and bathroom 
# * garadge and buildingsize 
#  
#  
#  House Id, Data_date, data-url And Data-Isonshow have less Relationship with The depended Variable We May Have to drop This Two Columns 

# #### Lets Take A Look At Vissuals Of This Columns With The Target Variable 

# In[ ]:


# fig, axes = plt.subplots(nrows= 1,ncols = 3, figsize=(20,12))
# axes[0,0].scatter(df_train['buildingSize'], df_train['data-price'], color='orange')
# axes[0,1].scatter(df_train['bedroom'], df_train['data-price'],  color='green')
# axes[0,2].scatter(df_train['bathroom'], df_train['data-price'], color='blue')

# axes[0,0].set_title('YearBuilt')
# axes[0,1].set_title('GarageYrBlt')
# axes[0,2].set_title('GrLivArea')


# #### I have Decided To not Drop Any Column as I can See Most Columns Are Correlated To each Other

# #### The Null Values in The Dataset

# In[ ]:


is_null=df.isnull().sum().sort_values(ascending=False)
NaN_train=(is_null[is_null>0])
dict(NaN_train)
NaN_train


# #### Visualissing the Missing Values In The Dataset 

# In[ ]:


plt.figure(figsize=(15, 8))
sns.barplot(NaN_train,NaN_train.index)
plt.title('Missing  Data In The Dataset')


# In[ ]:


df[['garage','buildingSize','erfSize','bedroom','bathroom']].head()


# * All The columns with Missing Values Above Are All Numeriacal Values
# * We can Impute The MIssing Variables Using The Mean Or meadin Or Zero 
# * For The garage I Will Have To Impute By Zero Since I believe That Not All Houses or Apartments Have The Garage  
# * I believe For Some Columns I can Use The Mean Of Meadian Since Every house Has to have 
#   * bathroom
#   * buildingSize
#   * bedroom
#   * erfSize
#   

# #### Filling In The Missing Values In The Dataset  

# In[ ]:


df["garage"] = df["garage"].fillna(0)
df["buildingSize"] = df["buildingSize"].fillna(df['buildingSize'].mean())
df["bathroom"] = df["bathroom"].fillna(df['bathroom'].mean())
df["bedroom"] = df["bedroom"].fillna(df['bedroom'].mean())
df["erfSize"] = df["erfSize"].fillna(df['erfSize'].mean())


# In[ ]:


df.isnull().sum()


# * The Data now has No Missing Values 

# #### We Have To Drop Columns That Have No Correlation And That We Dont Need 

# In[ ]:


# This Code Will Remove Columns That Are Mentioned In The Code 
df=df.drop(['data-isonshow','data-url','data-date','house-id'], axis=1)


# In[ ]:


df.head()


# #### Now The Data Is Clean And We Have Columns That We Need We Convert Catergorial Values To Numerical 

# In[ ]:


df= pd.get_dummies(df)
X_test=df.iloc[1351:,:]
X_train=df.iloc[:1351,:]
df.head()


# In[ ]:


# X = X_train=df.iloc[:1351,:]
X = X_train


# Our Data Now Has Numerical Values Only Is Ready To Be Used For Machine Learning 

# #### Lets Check If The Dependednt Variable Is Normalised 

# In[ ]:





# In[ ]:





# The quantile_plot Tels Us That The Data Is not Normalised So We Will Have To Normalise IT to Get Better Predictions 

# In[ ]:


# df_train["data-price"] = np.log1p(df_train["data-price"])
# y = df_train["data-price"]
# y.head()


# In[ ]:


(mu, sigma) = norm.fit(df_train['data-price'])
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.distplot(df_train['data-price'],fit=norm)
plt.ylabel('Frequency')
plt.title('data-price distribution')
plt.subplot(1, 2, 2)
quantile_plot=stats.probplot(df_train['data-price'], plot=plt)


# #### Cross Validation Function 
# 

# In[ ]:


#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train.values)
    rmse= np.sqrt(-cross_val_score(model, X_train.values, y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# Lasso Regression

# In[ ]:


lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0001, random_state=1))


# ENet

# In[ ]:


ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0001, l1_ratio=.9, random_state=1))


# KRR 

# In[ ]:


KRR = KernelRidge(alpha=0.999, kernel='polynomial', degree=2, coef0=2.888)


# GBoost

# In[ ]:


GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.09999,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# #### The Scores Of The Models 

# In[ ]:


score = rmsle_cv(lasso)
print("\nLASSO: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


# LASSO: 0.2893 (0.0120)


# In[ ]:


score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


ElasticNet score: 0.2893 (0.0120)


# In[ ]:


score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


Kernel Ridge score: 0.2998 (0.0247)


# In[ ]:


score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


Gradient Boosting score: 0.2847 (0.0140)


# In[ ]:





# In[ ]:


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   


# In[ ]:


averaged_models = AveragingModels(models = (ENet,lasso,GBoost))

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


Averaged base models score: 0.2560 (0.0123)
Averaged base models score: 0.2569 (0.0128) 
Averaged base models score: 0.2586 (0.0123)
Averaged base models score: 0.2587 (0.0122)
   


# In[ ]:





# In[ ]:


averaged_models.fit(X,y)
y_average=np.expm1(averaged_models.predict(X_test))


# In[ ]:





# In[ ]:


p= pd.DataFrame(y_average)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


Hp= pd.DataFrame(i)
Hp['price'] = p


# In[ ]:


Hp.head()


# In[ ]:


Hp.to_csv('last50000.csv',index=False)


# In[ ]:


Hp


# In[ ]:




