#!/usr/bin/env python
# coding: utf-8

# ## **Housing Prices Advanced Regression : EDA and Regression Models**
# 
# In this notebook I have performed thorough EDA on the housing dataset and tried to identify some keen underlying trends. Dtaa pre-processing is done after which different models have been applied. This kernel is an introduction to predictive modelling and demonstrates the various techniques. And it is must see if you are a beginner in regression and predictive modelling like me ;)
# 
# Let's dive in !!!.

# ## [**Do upvote the kernel ;) **]

# In[ ]:





# ## CONTENTS::

# [ **1 ) Importing the Modules and Loading the Dataset**](#content1)

# [ **2 ) Exploratory Data Analysis (EDA)**](#content2)

# [ **3 ) Missing Values Treatment**](#content3)

# [ **4 ) Handling Skewness of Features**](#content4)

# [ **5 ) Prepare the Data**](#content5)

# [ **6 ) Regression Models**](#content6)

# [ **7 ) Saving and Making Submission to Kaggle**](#content7)

# In[ ]:





# <a id="content1"></a>
# ## 1) Importing the Modules and Loading the Dataset

# In[ ]:


# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.legend_handler import HandlerBase
import seaborn as sns
import missingno as msno
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#import the necessary modelling algos.

#classifiaction.
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

#regression
from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV,ElasticNet
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

#model selection
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

#evaluation metrics
from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error # for regression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score  # for classification

from scipy import stats
from scipy.stats import norm, skew   # specifically for staistics


# In[ ]:


train=pd.read_csv(r'../input/train.csv')
test=pd.read_csv(r'../input/test.csv')


# In[ ]:


train.head(10)
#test.head(10)


# In[ ]:





# <a id="content2"></a>
# ## 2) Exploratory Data Analysis (EDA)

# ## 2.1 ) The Features and the 'Target' variable

# In[ ]:


df=train.copy()
#df.head(10)
df.shape


# In[ ]:


df.drop(['Id'],axis=1,inplace=True)
test.drop(['Id'],axis=1,inplace=True)


# We can drop the 'Id' column as the frames are already indexed.

# In[ ]:


df.index # the indices of the rows.


# In[ ]:


df.columns 


# ## 2.2 ) Check for Missing Values

# In[ ]:


df.isnull().any()


# In[ ]:


msno.matrix(df) # just to visulaize. 


# * #### Many columns have missing values and that will be treated later in the notebook.

# In[ ]:





# ## 2.3 ) Separate Dataframes (depending on data type)

# Might be useful when we consider features of different data types.

# #### CATEGORICAL FEATURES

# In[ ]:


cat_df=df.select_dtypes(include='object')


# In[ ]:


cat_df.head(10)
cat_df.shape


# In[ ]:


cat_df.columns   # list of the categorical columns.


# #### NUMERIC FEATURES

# In[ ]:


num_df=df.select_dtypes(include='number')
num_df.shape


# In[ ]:


num_df.columns # list of numeric columns.


# #### FEATURES WITH MISSING VALUES

# In[ ]:


nan_df=df.loc[:, df.isna().any()]
nan_df.shape
nan_df.columns   # list of columns with missing values.


# #### MERGING THE TRAIN & TEST SETS

# In[ ]:


all_data=pd.concat([train,test])


# In[ ]:


print(all_data.shape)
all_data = all_data.reset_index(drop=True)


# In[ ]:


all_data.head(10)


# In[ ]:


print(all_data.loc[1461:,'SalePrice'])  
# note that it is Nan for the values in test set as expected. so we drop it here for now.
all_data.drop(['SalePrice'],axis=1,inplace=True)


# ## 2.4 ) Analyzing the Target i.e. 'SalePrice'

# In[ ]:


# analyzing the target variable ie 'Saleprice'
sns.distplot(a=df['SalePrice'],color='#ff4125',axlabel=False).set_title('Sale Price')


# #### **The distribution of target is a bit right skewed. Hence taking the 'log transform' is a reasonable option.**

# #### ALSO LINEAR REGRESSION IS BASED ON THE ASSUMPTION OF THE 'HOMOSCADESITY' AND HENCE TAKING LOG WILL  BE A GOOD IDEA TO ENSURE 'HOMOSCADESITY' (that the varince of errors is constant.). A bit scary but simple ;) 
# 
# **You can read more about this on wikipedia.**

# In[ ]:


#Get also the qq-plot (the quantile-quantile plot)
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()


# ####  TAKING 'Log Transform' OF THE TARGET

# In[ ]:


df['SalePrice']=np.log1p(df['SalePrice']) 


# In[ ]:


# now again see the distribution.
sns.distplot(a=df['SalePrice'],color='#ff4125',axlabel=False).set_title('log(1+SalePrice)')  # better.


# In[ ]:





# ## 2.5 ) Most Related Features to the Target

# In[ ]:


cor_mat= df[:].corr()
cor_with_tar=cor_mat.sort_values(['SalePrice'],ascending=False)


# In[ ]:


print("The most relevant features (numeric) for the target are :")
cor_with_tar.SalePrice


# #### INFERENCES--
# 
# 1. Note that some of the features have quite high corelation with the target. These features are really significant.
# 
# 2. Of these the features with corelation value >0.5 are really important. Some features like GrLivArea etc.. are even more important.
# 
# 3. We will consider these features (i.e. GrLivArea,OverallQual) etc.. in more detail in subsequent sections during univariate and bivariate analysis.

# In[ ]:


# using a corelation map to visualize features with high corelation.
cor_mat= df[['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath',
             'YearBuilt','YearRemodAdd','GarageYrBlt','TotRmsAbvGrd','SalePrice']].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig=plt.gcf()
fig.set_size_inches(30,12)
sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True)

# some inference section.


# ## 2.6 ) Univariate Analysis

# In this section the univariate analysis is performed; More importantly I have considered the features that are more importanht with the 'Target' that  have high corelation with the Target.
# 
# For the numeric features I have used a 'distplot' and 'boxplot' to analyze their distribution.
# 
# Similarly for categorical features the most reasonable way to visualize the distribution is to use a 'countplot' which shows the relative counts for each category or class. Can use a pie-plot also to be a bit more fancy.

# #### NUMERIC FEATURES

# In[ ]:


def plot_num(feature):
    fig,axes=plt.subplots(1,2)
    sns.boxplot(data=df,x=feature,ax=axes[0])
    sns.distplot(a=df[feature],ax=axes[1],color='#ff4125')
    fig.set_size_inches(15,5)


# In[ ]:


plot_num('GrLivArea')


# In[ ]:


plot_num('GarageArea')


# In[ ]:


plot_num('TotalBsmtSF') 


# #### Note the features are a bit right skewed. We can therefore take 'log transform' of the features or a BoXCox transformation. Both shall work well. 

# #### CATEGORICAL FEATURES

# In[ ]:


def plot_cat(feature):
  sns.countplot(data=df,x=feature)
  ax=sns.countplot(data=df,x=feature)
   


# In[ ]:


plot_cat('OverallQual')


# Most of them are in 'average','above average' or 'good' classes.

# In[ ]:


plot_cat('FullBath')


# In[ ]:


plot_cat('YearBuilt')


# In[ ]:


plot_cat('TotRmsAbvGrd') # most of the houses have 5-7 rooms above the grd floor.


# #### Lastly we plot the countplot for some important features that are numerical here but are actually categorica. It seems if they have been label encoded.

# In[ ]:


plot_cat('GarageCars')


# In[ ]:


sns.factorplot(data=df,x='Neighborhood',kind='count',size=10,aspect=1.5)


# ## 2.7 ) Bivariate Analysis

# In this section the Bivariate Analysis have been done. I have plotted various numeric as well as categorical features against the target ie 'SalePrice'.

# #### NUMERIC FEATURES

# In[ ]:


fig, ax = plt.subplots()
ax.scatter(x = df['GrLivArea'], y = df['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea')
plt.show()


# #### Note that there are two outliers on the lower right hand side and can remove them.

# In[ ]:


df = df.drop(df[(df['GrLivArea']>4000) & (df['SalePrice']<13)].index) # removing some outliers on lower right side.


# In[ ]:


# again checking
fig, ax = plt.subplots()
ax.scatter(x = df['GrLivArea'], y = df['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea')
plt.show()


# In[ ]:


# garage area
fig, ax = plt.subplots()
ax.scatter(x =(df['GarageArea']), y = df['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('GarageArea')
plt.show()
# can try to fremove the points with gargae rea > than 1200.


# In[ ]:


# basment area
fig, ax = plt.subplots()
ax.scatter(x =(df['TotalBsmtSF']), y = df['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('TotalBsmtSF')
plt.show()   # check >3000 can leave here.


# #### CATEGORICAL FEATURES

# In[ ]:


#overall qual
sns.factorplot(data=df,x='OverallQual',y='SalePrice',kind='box',size=5,aspect=1.5)


# The SalePrice increases with the overall quality as expected.

# **Similar inferences can be drawn from other plots and graphs.**

# In[ ]:


#garage cars
sns.factorplot(data=df,x='GarageCars',y='SalePrice',kind='box',size=5,aspect=1.5)


# In[ ]:


#no of rooms
sns.factorplot(data=df,x='TotRmsAbvGrd',y='SalePrice',kind='bar',size=5,aspect=1.5) # increasing rooms imply increasing SalePrice as expected.


# In[ ]:


#neighborhood
sns.factorplot(data=df,x='Neighborhood',y='SalePrice',kind='box',size=10,aspect=1.5)


# Price varies with neighborhood.More posh areas of the city will have more price as expected.

# In[ ]:


#sale conditioin
sns.factorplot(data=df,x='SaleCondition',y='SalePrice',kind='box',size=10,aspect=1.5)


# In[ ]:





# <a id="content3"></a>
# ## 3 ) Missing Values Treatment

# In this section of the notebook I  have handled the missing values in the columns.
# 
# Firstly I have droped a couple of columns that have a really high % of missing values.
# 
# For other features I have analyzed if it that feaure is important or not and accordingly either have drooped it or imputed the values in it.
# 
# For imputation I have considered the meaning of the corressponding feature from the description. Like for a categorical feature if values are missing I have imputed "None" just to mark a separate category meaning absence of that thing. Similarly for a numeric feature I have imputed with 0 in case the missing value implies the 'absence' of that feature.
# 
# In all other cases I have imputed the categorical features with 'mode' i.e the most frequent class and with 'mean' for the numeric features.

# In[ ]:


nan_all_data = (all_data.isnull().sum())
nan_all_data= nan_all_data.drop(nan_all_data[nan_all_data== 0].index).sort_values(ascending=False)
nan_all_data
miss_df = pd.DataFrame({'Missing Ratio' :nan_all_data})
miss_df


# In[ ]:


#delet some features withvery high number of missing values.  
all_data.drop(['PoolQC','Alley','Fence','Id','MiscFeature'],axis=1,inplace=True)


# In[ ]:


test.drop(['PoolQC','Alley','Fence','MiscFeature'],axis=1,inplace=True)
df.drop(['PoolQC','Alley','Fence','MiscFeature'],axis=1,inplace=True)


# In[ ]:


# FireplaceQu
# it is useful but many of the values nearly half are missing makes no sense to fill half of the values. so deleting this
all_data.drop(['FireplaceQu'],axis=1,inplace=True)
test.drop(['FireplaceQu'],axis=1,inplace=True)
df.drop(['FireplaceQu'],axis=1,inplace=True)


# In[ ]:


#Lot Frontage
print(df['LotFrontage'].dtype)
plt.scatter(x=np.log1p(df['LotFrontage']),y=df['SalePrice'])
cr=df.corr()
print(df['LotFrontage'].describe())
print("The corelation of the LotFrontage with the Target : " , cr.loc['LotFrontage','SalePrice'])


# #### Above analysis shows that there is some relation of LotArea with the SalePrice both by scatter plot and also by the corelation value. Therefore instead of deleting I will impute the values with the mean for now.

# In[ ]:


all_data['LotFrontage'].fillna(np.mean(all_data['LotFrontage']),inplace=True)
all_data['LotFrontage'].isna().sum()


# In[ ]:


#Garage  related features.
# these features eg like garage qual,cond,finish,type seems to be important and relevant for buying car. 
# hence I will not drop these features insted i will fill them with the 'none' for categorical and 0 for numeric as nan here implies that there is no garage.

all_data['GarageYrBlt'].fillna(0,inplace=True)
print(all_data['GarageYrBlt'].isnull().sum())

all_data['GarageArea'].fillna(0,inplace=True)
print(all_data['GarageArea'].isnull().sum())

all_data['GarageCars'].fillna(0,inplace=True)
print(all_data['GarageCars'].isnull().sum())

all_data['GarageQual'].fillna('None',inplace=True)   # creating a separate category 'none' which means no garage.
print(all_data['GarageQual'].isnull().sum())

all_data['GarageFinish'].fillna('None',inplace=True)   # creating a separate category 'none' which means no garage.
print(all_data['GarageFinish'].isnull().sum())

all_data['GarageCond'].fillna('None',inplace=True)   # creating a separate category 'none' which means no garage.
print(all_data['GarageCond'].isnull().sum())

all_data['GarageType'].fillna('None',inplace=True)   # creating a separate category 'none' which means no garage.
print(all_data['GarageType'].isnull().sum())


# In[ ]:


# basement related features.
#missing values are likely zero for having no basement

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col].fillna(0,inplace=True)
    
# for categorical features we will create a separate class 'none' as before.

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col].fillna('None',inplace=True)
    
print(all_data['TotalBsmtSF'].isnull().sum())


# In[ ]:


# MasVnrArea 0 and MasVnrType 'None'.
all_data['MasVnrArea'].fillna(0,inplace=True)
print(all_data['MasVnrArea'].isnull().sum())

all_data['MasVnrType'].fillna('None',inplace=True)
print(all_data['MasVnrType'].isnull().sum())


# In[ ]:


#MSZoning.
# Here nan does not mean no so I will with the most common one ie the mode.
all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0],inplace=True)
print(all_data['MSZoning'].isnull().sum())


# In[ ]:


# utilities
sns.factorplot(data=df,kind='box',x='Utilities',y='SalePrice',size=5,aspect=1.5)


# #### Note that training set has only 2 of the possible 4 categories (ALLPub and NoSeWa) while test set has other categories. Hence it is of no use to us.

# In[ ]:


all_data.drop(['Utilities'],axis=1,inplace=True)


# In[ ]:


#functional
# fill with mode
all_data['Functional'].fillna(all_data['Functional'].mode()[0],inplace=True)
print(all_data['Functional'].isnull().sum())


# In[ ]:


# other rem columns rae all cat like kitchen qual etc.. and so filled with mode.
for col in ['SaleType','KitchenQual','Exterior2nd','Exterior1st','Electrical']:
  all_data[col].fillna(all_data[col].mode()[0],inplace=True)
  print(all_data[col].isnull().sum())


# #### Lastly checking if any null value still remains.

# In[ ]:


nan_all_data = (all_data.isnull().sum())
nan_all_data= nan_all_data.drop(nan_all_data[nan_all_data== 0].index).sort_values(ascending=False)
nan_all_data
miss_df = pd.DataFrame({'Missing Ratio' :nan_all_data})
miss_df


# #### Finally no null value remain now;)

# In[ ]:


all_data.shape


# In[ ]:





# <a id="content4"></a>
# ## 4 ) Handling Skewness

# For handling skewnesss I will take the log transform of the features with skewness > 0.5.
# 
# You can also try the BoxCox transformation as mentioned before.

# In[ ]:


#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.50]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])


# In[ ]:





# <a id="content5"></a>
# ## 5 ) Prepare the Data

# ## 5.1 ) LabelEncode the Categorical Features

# In[ ]:


for col in all_data.columns:
    if(all_data[col].dtype == 'object'):
        le=LabelEncoder()
        all_data[col]=le.fit_transform(all_data[col])


# ## 5.2 ) Splitting into Training and Validation Sets

# In[ ]:


train=all_data.loc[:(df.shape)[0]+2,:]
test=all_data.loc[(df.shape)[0]+2:,:]


# In[ ]:


train['SalePrice']=df['SalePrice']
train['SalePrice'].fillna(np.mean(train['SalePrice']),inplace=True)
train.shape
print(train['SalePrice'].isnull().sum())


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(train.drop(['SalePrice'],axis=1),train['SalePrice'],test_size=0.20,random_state=42)


# In[ ]:





# <a id="content6"></a>
# ## 6 ) Regression Models

# Lastly it is the time to apply various regression models and check how are we doing. I have used various regression models from the scikit.
# 
# Parameter tuning using GridSearchCV is also done to improve performance of some algos.

# #### The evalauton metric that I have used is the Root Mean Squared Error between the 'Log of the actual price' and 'Log of the predicted value' which is also the evaluation metric used by the kaggle.
# 
# #### To get abetter idea one may also use the K-fold cross validation insteadof the normal holdout set approach to cross validation.

# #### LINEAR REGRESSION

# In[ ]:


reg_lin=LinearRegression()
reg_lin.fit(x_train,y_train)
pred=reg_lin.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,pred)))


# #### LASSO (and tuning with GridSearchCV)

# In[ ]:


reg_lasso=Lasso()
reg_lasso.fit(x_train,y_train)
pred=reg_lasso.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,pred)))


# In[ ]:


params_dict={'alpha':[0.001, 0.005, 0.01,0.05,0.1,0.5,1]}
reg_lasso_CV=GridSearchCV(estimator=Lasso(),param_grid=params_dict,scoring='neg_mean_squared_error',cv=10)
reg_lasso_CV.fit(x_train,y_train)
pred=reg_lasso_CV.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,pred)))


# **Note the significant decrease in the RMSE on tuning the Lasso Regression.**

# In[ ]:


reg_lasso_CV.best_params_


# #### RIDGE (and tuning with GridSearchCV)

# In[ ]:


reg_ridge=Ridge()
reg_ridge.fit(x_train,y_train)
pred=reg_ridge.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,pred)))


# In[ ]:


params_dict={'alpha':[0.1, 0.15, 0.20,0.25,0.30,0.35,0.4,0.45,0.50,0.55,0.60]}
reg_ridge_CV=GridSearchCV(estimator=Ridge(),param_grid=params_dict,scoring='neg_mean_squared_error',cv=10)
reg_ridge_CV.fit(x_train,y_train)
pred=reg_ridge_CV.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,pred)))


# In[ ]:


reg_ridge_CV.best_params_


# #### GRADIENT BOOSTING

# In[ ]:


#the params are tuned with grid searchCV.

reg_gb=GradientBoostingRegressor(n_estimators=2000,learning_rate=0.05,max_depth=3,min_samples_split=10,max_features='sqrt',subsample=0.75 ,loss='huber')
reg_gb.fit(x_train,y_train)
pred=reg_gb.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,pred)))


# #### XGBoost

# In[ ]:


import xgboost as xgb
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_xgb.fit(x_train,y_train)
pred=model_xgb.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,pred)))


# Note that the parameters aren't optimized. This can get a lot better tahn this for sure.

# <a id="content7"></a>
# ## 7 ) Saving and Making Submission to Kaggle

# **The Gradient Boosting gives the best performance on the validation set and so I am using it to make predictions to Kaggle (on the test set).**

# In[ ]:


# predictions on the test set.
 
pred=reg_gb.predict(test)
pred_act=np.exp(pred)
pred_act=pred_act-1
len(pred_act)


# In[ ]:


test_id=[]
for i in range (1461,2920):
    test_id.append(i)
d={'Id':test_id,'SalePrice':pred_act}
ans_df=pd.DataFrame(d)
ans_df.head(10)


# In[ ]:


ans_df.to_csv('answer.csv',index=False)


# ## THE END!!!

# ## [Please star/ upvote if u liked it.]

# In[ ]:




