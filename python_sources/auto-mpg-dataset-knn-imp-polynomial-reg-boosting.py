#!/usr/bin/env python
# coding: utf-8

# Kernel is about the prediction of auto milage using 'Auto MPG Data Set' from UCI repository. 
# 
# What's **inside**?
# 
# 1. Basic **EDA** and Implemetation of **KNN** for missing value imputation.
# 
# 2. Converting catergorical to numerical variable.
# 
# 3. Feature Selection using **Backward elemination**.
# 
# 4. **Principle Componenet Analysis**
# 
# 5. Algorithms implemented are **Linear Regression,Polynomial Regression,AdaBoost,Gradient Boost**.
# 
# Metric Used: **RMSE**
# 
# **UpVote** if you like it.

# Attribute Information 
# 
# 1. mpg: continuous
# 2. cylinders: multi-valued discrete
# 3. displacement: continuous
# 4. horsepower: continuous
# 5. weight: continuous
# 6. acceleration: continuous
# 7. model year: multi-valued discrete
# 8. origin: multi-valued discrete  (1 - American, 2 - European, 3 - Japenese)
# 9. car name: string (unique for each instance)

# # **Importing Libraries**

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score,KFold,train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.decomposition import PCA


# In[ ]:


#Load the Dataset
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/autompg-dataset/auto-mpg.csv")
df.head(3)


# In[ ]:


df.shape #Check the number of rows and columns


# In[ ]:


df.isnull().sum() #Check for null values


# In[ ]:


df.info() #Check the datatype of all variables


# * Numerical Variables: 
# 'mpg', 'displacement', 'horsepower', 'weight' and 'acceleration'. 
# Note: "horsepower" is continous variable which is wrongly taken as discrete variable
# 
# * Catergorical Variables: 
# 'cylinders', 'model year', 'origin', 'car name'.

# In[ ]:


df['horsepower']=pd.to_numeric(df['horsepower'],errors='coerce') #Changed datatype to continuous.


# In[ ]:


#Let's convert the numerical variables to categorical
df["origin"] = df["origin"].astype('object') 
df["cylinders"] = df["cylinders"].astype('object')
df["model year"] = df["model year"].astype('object')


# In[ ]:


#'car name' does not make any difference to a car's milage. Therefore we drop the column
df.drop("car name", axis= 1, inplace= True)


# In[ ]:


df.describe().T #Basic description of the numerical variables.


# In[ ]:


df.describe(include = "O").T #Basic description of the categorical variables.


# # Exploratory Data Analysis

# In[ ]:


print("Skewness:{} \nKurtosis: {}".format(df["mpg"].skew(),df["mpg"].kurt()))
sns.distplot(df["mpg"]) #Slight positive skewness.


# In[ ]:


plt.figure(figsize=(15,5))
plt.subplot(121)
ax = sns.countplot(df["cylinders"], order= df['cylinders'].value_counts().index)
plt.subplot(122)
sns.boxplot(df["cylinders"],df["mpg"])
for i in ax.patches:
    ax.annotate('{}'.format(i.get_height()),(i.get_x()+0.3,i.get_height()))


# More number of car are manufactured with four cylinders and milage for it is the highest of any cylinder.
# Also,we can obseve that as the number of cylinders increase, there is decrease in the milage.

# In[ ]:


plt.figure(figsize=(15,10))
plt.subplot(221)
ax = sns.countplot(df["model year"])
plt.subplot(222)
sns.lineplot(data= df, x="model year",y="mpg")
plt.subplot(223)
sns.lineplot(data= df, x="model year",y="acceleration")
plt.subplot(224)
sns.lineplot(data= df, x="model year",y="weight")
for i in ax.patches:
    ax.annotate('{}'.format(i.get_height()),(i.get_x()+0.2,i.get_height()))


# * More number of car models were available during 1973 and 1978.
# 
# * From the lineplots, there is a sudden spike in the average milage, acceleration and decrease in the body weight in the year 1980, indicating shift in the auto-technology.
# 
# * There is a steady increase in the milage every year

# In[ ]:


plt.figure(figsize=(15,5))
plt.subplot(121)
ax= sns.countplot(df["origin"])
plt.subplot(122)
sns.boxplot(data= df, x="origin",y="mpg")
for i in ax.patches:
    ax.annotate('{}'.format(i.get_height()),(i.get_x()+0.35,i.get_height()))


# USA had more car model's manufactured, but their milage was the least compared to the rest. Japan had the highest milage.

# In[ ]:


plt.figure(figsize=(10,10))
plt.subplot(221)
sns.scatterplot(data= df, x="displacement",y="mpg")
plt.subplot(222)
sns.scatterplot(data= df, x="acceleration",y="mpg")
plt.subplot(223)
sns.scatterplot(data= df, x="weight",y="mpg")
plt.subplot(224)
sns.scatterplot(data= df, x="horsepower",y="mpg")
plt.tight_layout(pad= 5)


# * Milage has a strong negative correlation with displacement, weight and horsepower.
# 
# * Acceleration has slight positive correlation.

# In[ ]:


sns.heatmap(df.corr(),annot=True)


# * Numerical valriables are well correlated with the target variable "mpg". But, there is multi-collinearity between the features.

# In[ ]:


plt.figure(figsize=(10,10))
plt.subplot(221)
sns.distplot(df["displacement"], fit= stats.norm, hist=False)
plt.subplot(222)
sns.distplot(df["acceleration"],fit= stats.norm, hist=False)
plt.subplot(223)
sns.distplot(df["weight"], fit= stats.norm, hist=False)
plt.subplot(224)
sns.distplot(df["horsepower"], fit= stats.norm, hist=False)
plt.tight_layout(pad= 5)


# Horsepower, displacement, weight are right skewed.Whereas, acceleration is normally distributed.

# # Missing Value Imputation using KNN

# In[ ]:


df[df["horsepower"].isna() == True]


# In[ ]:


imp = KNNImputer(missing_values=np.nan, n_neighbors=5, weights='uniform', metric='nan_euclidean')
df_imp = imp.fit_transform(df)


# In[ ]:


df_imp = pd.DataFrame(df_imp, columns= df.columns)


# In[ ]:


df = df_imp.astype(df.dtypes.to_dict())


# In[ ]:


df.iloc[[32,126,336,336,354,374],:]


# # Convert catrgorical features to numeric

# In[ ]:


cat = df.describe(include =['O']).keys()
for feat in cat:
    k=list(df[feat].value_counts().index.sort_values())[:-1]
    for i in k:
        name = str(i) + '_' + feat
        df[name] = (df[feat]==i).astype(int)
    del df[feat]


# In[ ]:


df.head(2)


# In[ ]:


df.info()


# # Model Building

# In[ ]:


X = df.drop("mpg", axis= 1)
y = df[["mpg"]]


# # Standardization

# In[ ]:


std = StandardScaler().fit_transform(X)
X_std = pd.DataFrame(std, columns= X.columns)


# # Feature Selection

# In[ ]:


X_constant = sm.add_constant(X)


# In[ ]:


lin_rig_b = sm.OLS(y,X_constant).fit()


# In[ ]:


lin_rig_b.summary()


# In[ ]:


col = list(X_constant.columns)

while len(col)>1:
    X_1 = X_constant[col]
    model = sm.OLS(y,X_1,random_state = 0).fit()
    p = model.pvalues
    max_p = max(p)
    feature_maxp = p.idxmax()
    if max_p>0.05:
        print('Column Removed',feature_maxp,'\npob:',max_p)
        col.remove(feature_maxp)
    else:
        break


# Above removed feature does not pass the statistical test because their probability null hypothesis being true was more than 0.05.

# In[ ]:


lin_rig_b = sm.OLS(y,X_constant[col]).fit()


# In[ ]:


lin_rig_b.summary()


# # Train Test Split (Without PCA)

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_std[col[1:]], y, test_size=0.3, random_state=0)


# # Linear Regressor

# In[ ]:


LR = LinearRegression()
cv_rmse = cross_val_score(LR, X_train, y_train,cv=3, scoring='neg_mean_squared_error')
print("Mean of cv rmse's : ",np.mean(np.sqrt(np.abs(cv_rmse))))
print("Variance of cv rmse's: ",np.std(np.sqrt(np.abs(cv_rmse)),ddof=1))


# # Adaboost Regressor (Non-Linear)

# In[ ]:


AB_bias=[]
AB_var= []
for n in np.arange(1,100):
  AB=AdaBoostRegressor(n_estimators=n,random_state=0)
  cv_rmse=cross_val_score(AB,X_train, y_train,cv=3,scoring='neg_mean_squared_error')
  rmse=np.sqrt(np.abs(cv_rmse))
  AB_bias.append(np.mean(rmse))
  AB_var.append(np.var(rmse))


# In[ ]:


np.argmin(AB_bias), AB_bias[18], AB_var[18] #Min bias error for AdaBoost model with n_estimator = 19


# In[ ]:


x_axis=np.arange(len(AB_bias))


# In[ ]:


fig, ax = plt.subplots()
ax.plot(x_axis,AB_bias)
ax.set_ylabel('Bias Error')
ax1 = ax.twinx()
ax1.plot(x_axis,AB_var,"g")
ax1.set_ylabel('Variance Error')
fig.legend(labels = ('Bias Error','Variance Error'),loc=0)
plt.show()


# # AdaBoost with Linear Regression as Base Estimator

# In[ ]:


ABLR_bias=[]
ABLR_var= []
LR = LinearRegression()
for n in np.arange(1,150):
  ABLR=AdaBoostRegressor(base_estimator=LR, n_estimators=n,random_state=0)
  cv_rmse=cross_val_score(ABLR,X_train, y_train,cv=5,scoring='neg_mean_squared_error')
  rmse=np.sqrt(np.abs(cv_rmse))
  ABLR_bias.append(np.mean(rmse))
  ABLR_var.append(np.var(rmse))


# In[ ]:


np.argmin(ABLR_bias), ABLR_bias[2], ABLR_var[2] #Min bias error for AdaBoost with LR model with n_estimator = 3


# In[ ]:


x_axis=np.arange(len(ABLR_bias))
fig, ax = plt.subplots()
ax.plot(x_axis,ABLR_bias)
ax.set_ylabel('Bias Error')
ax1 = ax.twinx()
ax1.plot(x_axis,ABLR_var,"g")
ax1.set_ylabel('Variance Error')
fig.legend(labels = ('Bias Error','Variance Error'),loc=0)
plt.show()


# # GradientBoost Regressor (Non-Linear)

# In[ ]:


GB_bias=[]
GB_var= []
for n in np.arange(1,100):
  GB=GradientBoostingRegressor(n_estimators=n,random_state=0)
  scores=cross_val_score(GB,X_train,y_train,cv=3,scoring='neg_mean_squared_error')
  rmse=np.sqrt(np.abs(scores))
  GB_bias.append(np.mean(rmse))
  GB_var.append(np.var(rmse))


# In[ ]:


np.argmin(GB_bias), GB_bias[94], GB_var[94] #Min bias error for GradientBoost model with n_estimator = 95


# In[ ]:


x_axis=np.arange(len(GB_bias))
fig, ax = plt.subplots()
ax.plot(x_axis,GB_bias)
ax.set_ylabel('Bias Error')
ax1 = ax.twinx()
ax1.plot(x_axis,GB_var,"g")
ax1.set_ylabel('Variance Error')
fig.legend(labels = ('Bias Error','Variance Error'),loc=0)
plt.show()


# # Linear Regression with PCA

# Let us now remove the multicollinearity using PCA and see how diffrent algorithms learn and predict.

# To find the number n_components.

# In[ ]:


X_std.shape[0]


# In[ ]:


X_std.head(2)


# In[ ]:


cov_matrix = np.dot(X_std[col[1:]].T,X_std[col[1:]])/(X_std[col[1:]].shape[0]-1) #Covariance matrix


# In[ ]:


eig_val, eig_vec = np.linalg.eig(cov_matrix)


# In[ ]:


eigen_pairs = [(np.abs(eig_val[i]), eig_vec[ :, i]) for i in range(len(eig_val))]
eigen_pairs


# In[ ]:


tot = sum(eig_val)
var_exp = [( i /tot ) * 100 for i in sorted(eig_val, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print("Cumulative Variance Explained", cum_var_exp)


# In[ ]:


X_pca = PCA(n_components=17).fit_transform(X_std)


# In[ ]:


X_pca = pd.DataFrame(X_pca)
X_pca.head()


# In[ ]:


X_constant_p = sm.add_constant(X_pca)


# In[ ]:


model_ols = sm.OLS(y,X_constant_p).fit()


# In[ ]:


model_ols.summary()


# In[ ]:


X_constant_p.columns


# Above removed feature does not pass the statistical test because their probability null hypothesis being true was more than 0.05.

# In[ ]:


X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=0)


# In[ ]:


LR = LinearRegression()
cv_rmse = cross_val_score(LR, X_train_pca, y_train,cv=3, scoring='neg_mean_squared_error')
print("Mean of cv rmse's : ",np.mean(np.sqrt(np.abs(cv_rmse))))
print("Variance of cv rmse's: ",np.std(np.sqrt(np.abs(cv_rmse)),ddof=1))


# PCA is not giving us appreciable increase in the R^2.

# # Polynomial Regression

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
qr = PolynomialFeatures(degree=2)
X_P = qr.fit_transform(X[["displacement", "weight", "horsepower"]])


# In[ ]:


df_pn = pd.DataFrame(X_P)


# In[ ]:


X_l = X[col[1:]].columns.to_list()
for i in X_l:
    df_pn[i] = X[i]


# In[ ]:


df_pn.head(2)


# In[ ]:


df_pn.drop([0,1,2,3,4], axis=1, inplace=True)


# In[ ]:


df_pn_mat = StandardScaler().fit_transform(df_pn)


# In[ ]:


df_pn_std = pd.DataFrame(df_pn_mat, columns=df_pn.columns)
df_pn_std.head(2)


# In[ ]:


X_constant_pn = sm.add_constant(df_pn_std)


# In[ ]:


model_pn = sm.OLS(y,X_constant_pn).fit()


# In[ ]:


model_pn.summary()


# In[ ]:


col_pn = list(X_constant_pn.columns)

while len(col_pn)>1:
    X_1 = X_constant_pn[col_pn]
    model = sm.OLS(y,X_1,random_state = 0).fit()
    p = model.pvalues
    max_p = max(p)
    feature_maxp = p.idxmax()
    if max_p>0.05:
        print('Column Removed',feature_maxp,'\npob:',max_p)
        col_pn.remove(feature_maxp)
    else:
        break


# In[ ]:


X_constant_pn[col_pn]


# In[ ]:


model_pn_col = sm.OLS(y,X_constant_pn[col_pn]).fit()


# In[ ]:


model_pn_col.summary()


# # Train Test Split

# In[ ]:


X_train_pn, X_test_pn, y_train_pn, y_test_pn = train_test_split(X_constant_pn[col_pn[1:]], y, test_size=0.3, random_state=0)


# # Polynomial Regression

# In[ ]:


LR_pn = LinearRegression()
cv_rmse = cross_val_score(LR, X_train_pn, y_train_pn,cv=3, scoring='neg_mean_squared_error')
print("Mean of cv rmse's : ",np.mean(np.sqrt(np.abs(cv_rmse))))
print("Variance of cv rmse's: ",np.std(np.sqrt(np.abs(cv_rmse)),ddof=1))


# # AdaBoost with Polynomial Regressor as base estimator

# In[ ]:


ABLR_bias_pn=[]
ABLR_var_pn= []
LR = LinearRegression()
for n in np.arange(1,150):
  ABLR_pn=AdaBoostRegressor(base_estimator=LR, n_estimators=n,random_state=0)
  cv_rmse=cross_val_score(ABLR_pn,X_train_pn, y_train_pn,cv=3,scoring='neg_mean_squared_error')
  rmse=np.sqrt(np.abs(cv_rmse))
  ABLR_bias_pn.append(np.mean(rmse))
  ABLR_var_pn.append(np.var(rmse))


# In[ ]:


np.argmin(ABLR_bias_pn),ABLR_bias_pn[4],ABLR_var_pn[4]


# In[ ]:


x_axis=np.arange(len(ABLR_bias_pn))
fig, ax = plt.subplots()
ax.plot(x_axis,ABLR_bias_pn)
ax.set_ylabel('Bias Error')
ax.legend(loc=0)
ax1 = ax.twinx()
ax1.plot(x_axis,ABLR_var_pn,"g")
ax1.set_ylabel('Variance Error')
fig.legend(labels = ('Bias Error','Variance Error'),loc=0)
plt.show()


# Comparing the performance of models:
#     
#     Linear Regressor:
#     Mean of cv rmse's :  3.0102316679354444
#     Variance of cv rmse's:  0.39081554701283516
#     
#     Adaboost Regressor (Non-Linear):
#     Mean of cv rmse's :  3.8944193121996804
#     Variance of cv rmse's: 0.28608205454520774
#     
#     
#     AdaBoost with Linear Regressor as Base Estimator: 
#     Mean of cv rmse's :  3.0189631108440036
#     Variance of cv rmse's: 0.12382538317856155
#     
#     GradientBoost Regressor (Non-Linear):
#     Mean of cv rmse's :  3.482224364942708
#     Variance of cv rmse's: 0.22426231450451306
#     
#     Linear Regeression with PCA:
#     Mean of cv rmse's :  3.438499393433773
#     Variance of cv rmse's:  0.4864115738460171
#     
#     Polynomial Regressor(Linear regressor on polynomial of degree 2):
#     Mean of cv rmse's :  2.6991073515825064
#     Variance of cv rmse's:  0.3579384722624285
#     
#     AdaBoost with Polynomial Regressor as Base Estimator:
#     2.7563229334181187, 0.08866989184222245
#     
#     We see that Polynomial regression provides us with the least RMSE.

# In[ ]:


((2.7563229334181187-2.6991073515825064)/2.7563229334181187)*100


# In[ ]:


((0.3579384722624285-0.08866989184222245)/0.3579384722624285)*100


#  Polynomial Regressor and AdaBoost with Polynomial Regressor as Base Estimator give us the least RMSE. 
#  We can trade off 2% of bias error to get a boost of 75% on the variance error.
#  
#  Therefore, best model to be considered is AdaBoost with Polynomial Regressor as Base Estimator.

# **Polynimial Regression with AdaBoost with best scores**

# In[ ]:


lin_reg = LinearRegression()
ABLR_pn=AdaBoostRegressor(base_estimator=lin_reg, n_estimators=4,random_state=0)
model = ABLR_pn.fit(X_train_pn, y_train_pn)
print(f'R^2 score for train: {ABLR_pn.score(X_train_pn, y_train_pn)}')
print(f'R^2 score for test: {ABLR_pn.score(X_test_pn, y_test_pn)}')


# In[ ]:


train_rmse,test_rmse=(mean_squared_error(y_train_pn,ABLR_pn.predict(X_train_pn))),np.sqrt(mean_squared_error(y_test_pn,ABLR_pn.predict(X_test_pn)))
print("Test RMSE: {} , Train RMSE: {} ".format(train_rmse,test_rmse))


# **UpVote if you like it.**
# 
# **Any feedback or suggestions are welcome.**

# In[ ]:




