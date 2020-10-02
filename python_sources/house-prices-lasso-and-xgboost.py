#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[26]:


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from scipy.stats import skew, skewtest
from subprocess import check_output

def exportResults(classifier, idnumber,validationdf):
    validationdf=pd.DataFrame({"Id" : list(validationdf.index), "SalePrice" : list(classifier.predict(validationdf))}) 
    validationdf.to_csv("test{}.csv".format(idnumber), index=False)


# In[27]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv("../input/train.csv", index_col=0)
validation = pd.read_csv("../input/test.csv", index_col=0)
target = train['SalePrice']  #target variable

df_train = train.drop('SalePrice', axis=1)

df_train['training_set'] = True
validation['training_set'] = False

df_full = pd.concat([df_train, validation])
# Any results you write to the current directory are saved as output.


# In[28]:


train_numeric = train.select_dtypes(include=[np.number])
train_object = train.select_dtypes(include=['object'])
validation_numeric = validation.select_dtypes(include=[np.number])


# In[29]:


labels=list(train_object.apply(pd.Series.nunique).index)
values=train_object.applymap(lambda x: pd.isnull(x)).sum()*100/train_object.shape[0]
plt.figure(figsize=(15,5))
plt.bar(range(len(values)), values, align='center', color='r')
plt.xticks(range(len(labels)), labels, rotation=90)
plt.ylabel("# NAs (%)")
plt.ylim([0,100])
plt.xlim([-1, train_object.shape[1]])
plt.title("Number of NAs for each feature (%)")
plt.show()


# Several variables contain more than 80 % of NAs. I should explore what this means. Are the NAs related to missing data or can be mapped to something else? let's get unique values for Alley, FirePlaceQu, PoolQC, Fence and MiscFeature...

# In[30]:


for col in ["Alley", "FireplaceQu", "PoolQC", "Fence", "MiscFeature"]:
    print("Unique values for {}:".format(col))
    print(train_object.loc[:,col].unique())
    print(" ")


# In case of alley unique values correspond to graval and pave. Thus, it seems logical, that **NAs do not correspond to missing values but to asphalt alleys instead**.
# 
# On the other hand, I have no ideas what fireplace quality values are meant. Therefore I do not feel confident to use the variable at all.** I am removing the variable.**
# 
# In case of pool quality the values probably mean "excellent" and "fair". Yet there is so many missing values. Missing values probably mean no pool. **Therefore I will encode it as 'NA'.**
# 
# The same goes for fence. Too many missing values and while missing values could mean no fence.**I will encode missing values as 'NA'**. 
# 
# As for miscellenaous feature, **I see no reason to keep this feature in the dataset**.
# 
# **It would be best to include all value transoformation into the pipeline. This way, I can maintain consistency between train and test set processing.**
# 
# The category pipeline must include the following:
# * selecting categorical features (ie object)
# * removing undesirable features
# * recoding Alley feature
# * fill missing values using the most common one
# * recoding features as type category
# * creating dummy variables
# * get all possible values for dummy variables (from both train and test)
# 
# To do that, I will have to define several custom transformers:

# removing pool quality, fence, fireplace, miscellaneous
# 

# In[31]:


df_full.drop(['FireplaceQu','MiscFeature'], inplace=True, axis=1)
df_full['Alley']=df_full.loc[:,'Alley'].fillna('Asphalt')
df_full['PoolQC']=df_full.loc[:,'PoolQC'].fillna('NA')
df_full['Fence']=df_full.loc[:,'Fence'].fillna('NA')

print(df_full.info())

# perform this for categorical variables only
for feature in df_full.select_dtypes(include=['object']).columns: # do it for categorical columns only
    replaceValue=df_full.loc[:,feature].value_counts().idxmax()
    df_full[feature]=df_full.loc[:,feature].fillna(replaceValue)
df_full = pd.get_dummies(df_full)
print(df_full.shape)
print(df_full.info())

df_full=df_full.apply(lambda x: x.fillna(x.mean()),axis=1)

target_skewed = np.log1p(target)
skewed_feats = df_full.apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
df_full_skewed = df_full.copy()
df_full_skewed[skewed_feats] = np.log1p(df_full_skewed[skewed_feats])
print(df_full.info())


# end of category data preprocessing

# Since one hot encoding yeilds dramatically expanded feature space, I would like to eliminate as many new features as possible. To do that, I will look at semilology graphs first.

# In[32]:


pca_cat_train = PCA().fit((df_full)) 
print(type(pca_cat_train))
#plt.semilogy(pca_xtrain.explained_variance_ratio_, '--o');
plt.semilogy(pca_cat_train.explained_variance_ratio_.cumsum(), '--o')
plt.xlabel("ncomponents")
#plt.ylim([0.5, 1])
plt.xlim([0,50])
plt.title("Cumulative Variance Explained")


# It looks like I should be on the safe side if I choose 250 components. Therefore, below is the final categorical pipeline.

# # Train-test split
# For internal validation only, I split the data into train and test set. I also fix the random_state so that I obtain consistent results.

# In[33]:


df_train = df_full[df_full['training_set']==True]
df_train = df_train.drop('training_set', axis=1)
df_test = df_full[df_full['training_set']==False]
df_test = df_test.drop('training_set', axis=1)

df_train_skewed = df_full_skewed[df_full_skewed['training_set']==True]
df_train_skewed = df_train_skewed.drop('training_set', axis=1)
df_test_skewed = df_full_skewed[df_full_skewed['training_set']==False]
df_test_skewed = df_test_skewed.drop('training_set', axis=1)


X_train, X_test, y_train, y_test = train_test_split(df_train, target, test_size=0.2, random_state=42)
print(X_train.columns)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# random forest modeling (as oren)

# In[34]:


rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)

rf.fit(df_train, target)

rf_skewed = RandomForestRegressor(n_estimators=100, n_jobs=-1)
rf_skewed.fit(df_train_skewed, target_skewed)

#print(np.sqrt(mean_squared_log_error(y_test, rf.predict(X_test))))    
cv_num=4
scores = cross_val_score(rf, df_train, target, cv=cv_num, scoring='neg_mean_squared_log_error')
print("cross val scores:" , np.sqrt(-scores))
print('average root mean squared log error=', np.mean(np.sqrt(-scores)))
preds = rf.predict(df_test)

scores = cross_val_score(rf_skewed, df_train_skewed, target_skewed, cv=cv_num, scoring='neg_mean_squared_error')
print("cross val scores (skewed dataset):" , np.sqrt((-scores)))
print('average root mean squared log error (skewed dataset) =', np.mean(np.sqrt(-scores)))
preds_skewed = np.expm1(rf_skewed.predict(df_test_skewed))


# In[35]:


plt.hist(preds)
plt.hist(target, alpha = 0.3)
plt.show()
fig, ax = plt.subplots(figsize=(15,15))  
sns.residplot(y_test, y_test-rf.predict(X_test), lowess=True, color="b")
plt.show()
#my_submission_nopipe = pd.DataFrame({'Id': df_test.index, 'SalePrice': preds})
#my_submission_nopipe.to_csv('submission_nopipe.csv', index=False)
my_submission_nopipe_skewed = pd.DataFrame({'Id': df_test.index, 'SalePrice': preds_skewed})
my_submission_nopipe_skewed.to_csv('submission_nopipe_skewed.csv', index=False)


# In[36]:


pipeline = Pipeline(memory=None, 
                          steps =[
                              ('rescaling', StandardScaler()),
                            #  ('log_transformation',LogTransformer() ),
                              ('reduce_dim', PCA(n_components=50)), 
                              ('regression', RandomForestRegressor(n_estimators=100, n_jobs=-1))
                               ]
                              )
pipeline.fit(df_train, target)
cv_num=4
scores = cross_val_score(pipeline, df_train, target, cv=cv_num, scoring='neg_mean_squared_log_error')
print("cross val scores:" , np.sqrt(-scores))
print('average root mean squared log error=', np.mean(np.sqrt(-scores)))


# optimizing random forest regressor

# n_estimators_grid = np.linspace(10,300, 30, dtype='int')
# min_samples_leaf_grid = np.linspace(1,10, 5, dtype='int')
# rfgrid={'n_estimators' : n_estimators_grid, 
#         'min_samples_leaf' : min_samples_leaf_grid}
# rfgrid_cv=GridSearchCV(RandomForestRegressor(), rfgrid, cv=cv_num, refit=True, n_jobs=10, scoring='neg_mean_squared_log_error')
# 
# rfgrid_cv.fit(df_train_skewed, target_skewed)
# print(n_estimators_grid)
# print(min_samples_leaf_grid)
# print(rfgrid_cv.best_params_)

# ideal parameters={'min_samples_leaf': 1, 'n_estimators': 170}

# In[37]:


rf = RandomForestRegressor(n_estimators=170, min_samples_leaf=1, n_jobs=10)
rf.fit(df_train_skewed, target_skewed)
#print(np.sqrt(mean_squared_log_error(y_test, rf.predict(X_test))))    
cv_num=4
scores = cross_val_score(rf, df_train_skewed, target_skewed, cv=cv_num, scoring='neg_mean_squared_error')
print("cross val scores:" , np.sqrt(-scores))
print('average root mean squared log error=', np.mean(np.sqrt(-scores)))
preds = rf.predict(df_test)


# # Linear Regression modeling
# 

# # Lasso Regression

# In[38]:


from  sklearn.linear_model import LassoCV
model_lasso = LassoCV(alphas = np.logspace(-4, -2, 20), cv = 4, n_jobs=4, max_iter=500)
model_lasso.fit(df_train_skewed, target_skewed)
scores = cross_val_score(model_lasso, df_train_skewed, target_skewed, cv=cv_num, scoring='neg_mean_squared_error')
print("cross val scores (lasso):" , np.sqrt(-scores))
print('average root mean squared log error (lasso)=', np.mean(np.sqrt(-scores)))
preds_lasso_skewed = np.expm1(model_lasso.predict(df_test_skewed))
my_submission_nopipe_lasso_skewed = pd.DataFrame({'Id': df_test.index, 'SalePrice': preds_lasso_skewed})
my_submission_nopipe_lasso_skewed.to_csv('submission_nopipe_lasso_skewed.csv', index=False)


# In[54]:


from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
pipeline_lasso = Pipeline(memory=None, 
                          steps =[
                              #('rescaling', StandardScaler()),
                            #  ('log_transformation',LogTransformer() ),
                              #('reduce_dim', PCA(n_components=200)), 
                              ('regression', LassoCV(alphas = np.logspace(-4, -2, 20), cv = 4, n_jobs=4, max_iter=500, normalize=False))
                               ]
                              )
pipeline_lasso.fit(df_train_skewed, target_skewed)
scores = cross_val_score(pipeline_lasso, df_train_skewed, target_skewed, cv=cv_num, scoring='neg_mean_squared_error')
print("cross val scores (lasso pipeline):" , np.sqrt(-scores))
print('average root mean squared log error (lasso pipeline)=', np.mean(np.sqrt(-scores)))


# # Trying xgboost
# This custom xgboost regressor was created based on https://www.kaggle.com/fiorenza2/journey-to-the-top-10 kernel
# and https://www.kaggle.com/opanichev/ensemble-of-4-models-with-cv-lb-0-11489

# In[59]:


from sklearn.base import BaseEstimator, RegressorMixin
from xgboost.sklearn import XGBRegressor
class CustomEnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, regressors=None):
        self.regressors = regressors

    def fit(self, X, y):
        for regressor in self.regressors:
            regressor.fit(X, y)

    def predict(self, X):
        self.predictions_ = list()
        for regressor in self.regressors:
            self.predictions_.append((regressor.predict(X).ravel()))
        return (np.mean(self.predictions_, axis=0))
    
xgb1 = XGBRegressor(colsample_bytree=0.2,
                 learning_rate=0.05,
                 max_depth=3,
                 n_estimators=1200
                )

xgb2 = XGBRegressor(colsample_bytree=0.2,
                 learning_rate=0.05,
                 max_depth=3,
                 n_estimators=1200,
                seed = 1234
                )

xgb3 = XGBRegressor(colsample_bytree=0.2,
                 learning_rate=0.05,
                 max_depth=3,
                 n_estimators=1200,
                seed = 1337
                )

xgb_ens = CustomEnsembleRegressor([xgb1,xgb2,xgb3])
scores = cross_val_score(cv=cv_num,estimator=xgb_ens,X = df_train_skewed,y = target_skewed, n_jobs = -1, scoring='neg_mean_squared_error')
print("cross val scores (xgboost ensemble):" , np.sqrt(-scores))
print('average root mean squared log error (xgboost ensemble)=', np.mean(np.sqrt(-scores)))
preds_xgb_skewed = np.expm1(xgb_ens.predict(df_test_skewed))
my_submission_nopipe_xgb_skewed = pd.DataFrame({'Id': df_test.index, 'SalePrice': preds_xgb_skewed})
my_submission_nopipe_xgb_skewed.to_csv('submission_nopipe_xgb_skewed.csv', index=False)

