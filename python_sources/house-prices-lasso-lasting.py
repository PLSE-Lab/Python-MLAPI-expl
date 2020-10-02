#!/usr/bin/env python
# coding: utf-8

# # Problem Statement
# 
# - This project aims at predicting house prices (residential) in Ames, Iowa, USA using advanced regression techniques.
# - Train data consist of 1460 rows and 80 cols including target variable.
# - Test data consist of 1459 rows and 79 cols.
# - This dataset have lot of features to deal with. So, Feature Engineering comes to play.

# In[ ]:


# import data from kaggle
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # 1. Importing Libraries.

# In[ ]:


import numpy as np # Linear Algebra, Matrix operations .....etc
import pandas as pd # Data manipulation.
import seaborn as sns # Data Visualization.
import matplotlib.pyplot as plt # plots
get_ipython().run_line_magic('matplotlib', 'inline')
#plot in jupyter notebook don't open new window for plotting.

# Future warnings ignore
import warnings
warnings.filterwarnings("ignore")

# Machine Learning API's
from sklearn.model_selection import cross_val_score # cross validation.
from sklearn.linear_model import LinearRegression,Lasso 
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor


# # 2. Load data into pandas dataframe.

# In[ ]:


train_df=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv") # train data
test_df=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv") # test data.
train_df.head() # view top rows default 5 rows get displayed.


# In[ ]:


test_df.head() # view test data.


# # 3. Copy target variable to new variable.

# In[ ]:


target=train_df["SalePrice"].copy()
train_df.drop("SalePrice",axis=1,inplace=True) # Drop target from train dataset.


# # 4. Combine train and test data.

# In[ ]:


df=pd.concat([train_df,test_df])


# In[ ]:


# check shape of combined data.
df.shape


# - combined data has 2919 rows with 80 features

# In[ ]:


# Droping Id column
del df["Id"]


# # 5. Check for missing values.
# - Missing values treatment.

# In[ ]:


from IPython.display import Image
Image("/kaggle/input/missing-values-mechanism/Missingtheory.png")


# In[ ]:


null_cols=df.columns[df.isna().any()] # Get cols with missing values


# In[ ]:


null_per=round(df[null_cols].isna().agg("mean").sort_values(ascending=False),5) # Get percentage of missing values of each column.


# In[ ]:


null=pd.DataFrame({"Features":null_per.index,"percentage":null_per.values}) # create data frame  for barplot.


# In[ ]:


plt.figure(figsize=(15,5)) # set width and height for plot
sns.barplot(x=null.columns[0],y=null.columns[1],data=null)
plt.xticks(rotation = 90) # Prevent labels from overlapping.
plt.show()


# - PoolQC Followed by MiscFeature and Alley, Fence features having highest percentage of missing values above 50%. 

# In[ ]:


# Drop features with 50% values missing in it.
df.dropna(thresh=int(0.5*len(df)),inplace=True,axis=1)


# In[ ]:


df.shape # shape after dropping cols.


# In[ ]:


rem_null=null[4:]# Get remaining missing values which are less than 50%.
plt.figure(figsize=(15,5)) # set width and height for plot
sns.barplot(x=rem_null.columns[0],y=rem_null.columns[1],data=rem_null)
plt.xticks(rotation = 90) # Prevent labels from overlapping.
plt.show()


# # we need to impute this missing values 
# - You need to have good understanding of each variable.
# - Three types of missing data mechanisms - MCAR, MAR, MNAR.
#    1. Missing completely at random (MCAR) - Missingness is not related to any variables. Missingness on variable is completely unsystematic.
#    2. Missing at random (MAR) - Missingness is related to other variables not itself. it can be imputated with other variables which relates to missing                                 data.
#    3. Missing not at random (MNAR) - Also know as "non-ignorable" because should not be ignored. this shows up when neither MCAR or MAR and it's related                                      to both observed(Missing) and unobserved(other features) data.
# - MCAR and MAR is found by statistical testing.
# - MNAR is found by understanding data.

# # Imputing numeric features.

# In[ ]:


# Get numeric features and store into new variable
numeric_features=df.select_dtypes(include=np.number)


# In[ ]:


numeric_features.dtypes.value_counts() # check dtypes.


# In[ ]:


# Imputation using KNN it finds similar values to impute missing ones.
from fancyimpute import KNN
impute=KNN(k=3).fit_transform(numeric_features) # imputing


# > - **NOTE**: It returns series of numpy arrays and convert whole data dtypes into float you need to convert back original dtypes.

# In[ ]:


# Creating dataframe because knn imputation returns series of numpy arrays.
imputed=pd.DataFrame(impute,columns=numeric_features.columns)
imputed.head()


# In[ ]:


cast_int=imputed.dtypes[numeric_features.dtypes != imputed.dtypes] # Get dtypes to change into int format.
imputed[cast_int.index]=imputed[cast_int.index].applymap(int) # casting into original dtypes.


# In[ ]:


imputed.head()


# In[ ]:


imputed.dtypes.value_counts() # count of dtypes


# In[ ]:


# check you imputated to null values
imputed.isna().any().sum()


# In[ ]:


# Statistical information of numeric data.
imputed.describe()


# # Imputing categorical columns.

# In[ ]:


# Check for missingness count
cate_features=df.select_dtypes(exclude=np.number)
cate_cols=cate_features.columns[cate_features.isna().any()]
cate_per=round(df[cate_cols].isna().agg("mean").sort_values(ascending=False),5) # Get percentage of missing values of each column.
cate_per


# In[ ]:


cate_features[cate_cols].isna().sum()


# In[ ]:


cate_features["FireplaceQu"]=cate_features["FireplaceQu"].fillna("Missing")
cols=cate_features[cate_cols].isna().sum() <100
impute_cols=cols.index
for i in impute_cols:
    cate_features[i].fillna(cate_features[i].mode()[0],inplace=True)


# In[ ]:


cate_features[cate_cols].isna().sum()


# In[ ]:


cate_features.columns.value_counts().sum()


# # 6. Data exploration

# In[ ]:


# check distribution of target variable.
plt.figure(figsize=(8,5))
sns.distplot(target)


# - It is right skewed or positive skewed. 
# - Check skewness of target feature.

# In[ ]:


target.skew() # skewness


# In[ ]:


# converting right skewed to normal distribution (or) close to normal.
plt.figure(figsize=(8,5))
sns.distplot(np.log(target))
np.log(target).skew()


# > Converting target from right skewed to normal will helps model to map relationship between independent variables to target variable. 

# In[ ]:


recombined=pd.concat([numeric_features,cate_features],axis=1)


# In[ ]:


recombined.head()


# # 7. Corelation between numeric variables

# In[ ]:


train=recombined[:train_df.shape[0]]
test=recombined[train_df.shape[0]:]
train.shape,test.shape


# In[ ]:


trainn=pd.concat([train,target],axis=1)


# In[ ]:


plt.figure(figsize=(25,15))
corr=trainn.corr(method="spearman")
sns.heatmap(corr,annot=True)


# In[ ]:


print (corr['SalePrice'].sort_values(ascending=False)[:20]) #top 15 values
print ('----------------------')
print (corr['SalePrice'].sort_values(ascending=False)[-5:]) #last 5 values`


# # Visualize the relation between correlated variables with target.

# In[ ]:


# 1st correlated variable.
trainn['OverallQual'].unique()


# In[ ]:


#let's check the mean price per quality and plot it.
pivot = trainn.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.median)
sns.barplot(x="OverallQual",y="SalePrice",data=trainn)


# - Overall quality increases saleprice also increases

# In[ ]:


pivot = trainn.pivot_table(index='GrLivArea', values='SalePrice', aggfunc=np.median)
sns.jointplot(x="GrLivArea",y="SalePrice",data=trainn)


# - It consist of outliers, outliers spoil the model .we need to analyse it and then decide remove or leave.
# - we are using advanced regression models randomforest,xgboost they handle outliers.
# - Likewise, analyse other variables which are correlated with target.

# # Analysing categorical variables.

# In[ ]:


cate_data=recombined.select_dtypes(include="object")
cate_data.describe()


# In[ ]:


# we need to check the relation between categorical values using ANOVA .
from scipy import stats
cat = [f for f in trainn.columns if trainn.dtypes[f] == 'object']
def anova(frame):
    anv = pd.DataFrame()
    anv['features'] = cat
    pvals = []
    for c in cat:
           samples = []
           for cls in frame[c].unique():
                  s = frame[frame[c] == cls]['SalePrice'].values
                  samples.append(s)
           pval = stats.f_oneway(*samples)[1]
           pvals.append(pval)
    anv['pval'] = pvals
    return anv.sort_values('pval')

cate_data['SalePrice'] = trainn.SalePrice
k = anova(cate_data) 
k['disparity'] = np.log(1./k['pval'].values) 
plt.figure(figsize=(15,8))
sns.barplot(data=k, x = 'features', y='disparity') 
plt.xticks(rotation=90) 
plt.show()


# - Here we see that among all categorical variables Neighborhood turned out to be the most important feature followed by ExterQual, KitchenQual, etc.  

# In[ ]:


# View all numeric features distribution.
trainn.hist(figsize=(20,18))
plt.show()


# - Check if target variable distribution is matching with independent variables.

# In[ ]:


# Also visualize the categorical variables using boxplot
def boxplot(x,y,**kwargs):
            sns.boxplot(x=x,y=y)
            x = plt.xticks(rotation=90)

cat = [f for f in train.columns if trainn.dtypes[f] == 'object']

p = pd.melt(trainn, id_vars='SalePrice', value_vars=cat)
g = sns.FacetGrid (p, col='variable', col_wrap=4, sharex=False, sharey=False, height=4)
g = g.map(boxplot, 'value','SalePrice')
g


# # 8. Feature engineering

# In[ ]:


# we need to convert categorical variables in to numeric. unless it gives you any error all models works with numerical mathematical formuales.
# combine train and test data ,split target into new variable
target=trainn["SalePrice"].copy()
del trainn["SalePrice"]
fina_df=pd.concat([trainn,test])


# In[ ]:


# Look out features behavior and perform feature engineering.
# Select features which contribute to target variable and also create new .

fina_df.select_dtypes(include="object").nunique() # unique values in each categorical variable.


# - we need to convert this values into numeric by label encoding and one-hot encoding.
# - Also you should know the difference between label encoding and OHE
# - label assumes assumes they are ordinal values. it should be appiled to only ordinal data.
# - OHE extends the dimensionality of data should be used wisely.

# In[ ]:


# perfrom label encoding on ordinal data or mapping function.
# features which are have quality are ordinal.
features_ord=["GarageQual","KitchenQual","FireplaceQu","BsmtQual","ExterQual"]
fina_df[features_ord].nunique()


# In[ ]:


# multiple column label encoder class
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


# In[ ]:


fina_df=MultiColumnLabelEncoder(columns = features_ord).fit_transform(fina_df)


# In[ ]:


fina_df.head()


# In[ ]:


# converting other categorical values to numeric using OHE , pandas get_dummies.
fina_df=pd.get_dummies(fina_df)


# In[ ]:


# splitting train and test data.
train_split=fina_df[:train_df.shape[0]]
test_split=fina_df[train_df.shape[0]:]


# In[ ]:


# log transformation target before training
y=np.log(target)


# In[ ]:


train_split.head()


# In[ ]:


# some cols still have null values that are hidden by imputation methods.
remaining=train_split.columns[train_split.isna().any()]
remaining


# In[ ]:


for i in remaining:
    train_split[i].fillna(0,inplace=True) # Filling with zeros


# In[ ]:


remaining=test_split.columns[test_split.isna().any()]
for i in remaining:
    test_split[i].fillna(0,inplace=True) # filling with zeros


# In[ ]:


# final check for null values in both train and test data.
print(train_split.columns[train_split.isna().any()])
print(test_split.columns[test_split.isna().any()])


# In[ ]:


from sklearn.preprocessing import StandardScaler
stand=StandardScaler()
values=stand.fit_transform(train_split)
X_train=pd.DataFrame(values,columns=train_split.columns)


# In[ ]:


from sklearn.preprocessing import StandardScaler
stand=StandardScaler()
values=stand.fit_transform(test_split)
X_test=pd.DataFrame(values,columns=test_split.columns)


# # 9. Model building.
# - Using 3 models Xgboost, Lasso, Gradient Boosting.

# In[ ]:


import xgboost as xgb
# below parameters are valued by cross-validation.
regr = xgb.XGBRegressor(colsample_bytree=0.2,
                       gamma=0.0,
                       learning_rate=0.05,
                       max_depth=6,
                       min_child_weight=1.5,
                       n_estimators=7200,
                       reg_alpha=0.9,
                       reg_lambda=0.6,
                       subsample=0.2,
                       seed=42,
                       silent=1)

regr.fit(X_train, y) # train data, log transformed target.


# In[ ]:


from sklearn.metrics import mean_squared_error
# root mean square error function
def rmse(y_test,y_pred):
      return np.sqrt(mean_squared_error(y_test,y_pred)) 

# Run prediction on training set to get an idea of how well it does
y_pred = regr.predict(X_train)
y_test = y
print("XGBoost score on training set: ", rmse(y_test, y_pred))


# In[ ]:


# make prediction on the test set
y_pred_xg = regr.predict(X_test)
xg_ex = np.exp(y_pred_xg)
pred1 = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': xg_ex})
pred1.to_csv('xgboost.csv', header=True, index=False)


# ### Lasso Regression.

# In[ ]:


# Lasso
from sklearn.linear_model import Lasso

# found this best alpha through cross-validation
best_alpha = 0.00099

lasso = Lasso(alpha=best_alpha, max_iter=50000)
lasso.fit(X_train, y)

# Metrics score on train data
y_pred = lasso.predict(X_train)
y_test = y
print("Lasso score on training set: ", rmse(y_test, y_pred))


# In[ ]:


#make prediction on the test set
lasso_pred = lasso.predict(X_test)
lasso_ex = np.exp(lasso_pred)
pred1 = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': lasso_ex})
pred1.to_csv('lasso.csv', header=True, index=False)


# ### Gradient Boosting Regressor

# In[ ]:


# Gradient boosting regressor model training
from sklearn.ensemble import GradientBoostingRegressor
est = GradientBoostingRegressor(n_estimators= 1000, max_depth= 2, learning_rate= .01)
est.fit(X_train, y)


# In[ ]:


# metrics score on train data.
y_pred = est.predict(X_train)
y_test = y
print("Gradient score on training set: ", rmse(y_test, y_pred))


# In[ ]:


# submission to kaggle
GBC_pred = est.predict(X_test) # prediction
GBC_ex = np.exp(GBC_pred) # converting back to original values
pred1 = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': GBC_ex})
pred1.to_csv('GBC.csv', header=True, index=False)

