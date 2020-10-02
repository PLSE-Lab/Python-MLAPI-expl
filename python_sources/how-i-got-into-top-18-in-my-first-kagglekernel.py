#!/usr/bin/env python
# coding: utf-8

# # A FRIENDLY MESSAGE

# This is my first ever kernel in Kaggle. As nervous as I was, I landed up at top 18% in my first submission. Learned a lot in the process and looking forward to keep going on with other submissions.
# 
# I know it is a very common problem set for anyone to submit and upload a kernel but I thought of sharing it anyway to share my approach hoping to get feedbacks from the community on how I can improve. 
# 
# I have shared my mistakes (commented out) along with my code to showcase my different approaches and the results I got from them. If you are a beginner it makes you see the thought process of other beginners while approaching a problem and become more comfortable with your doubts.
# 
# I have taken a very basic approach and I hope you find it useful. If you do, please upvote, it'll just motivate me more to keep trying more and more problems. 
# 
# HOPING TO HEAR FROM YOU ALL. THANK YOU IN ADVANCE :)

# # IMPORTS & GETTING READY

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing
import seaborn as sns #data visualization
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import preprocessing as prep
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

print('train data shape: ', train.shape, '\ntest data shape: ', test.shape)
print('Reading done!')


# In[ ]:


train.drop('Id', axis = 1, inplace = True)
test.drop('Id', axis = 1, inplace = True)
print("Dropped redundant column 'Id' from both train and test data")

target = 'SalePrice'
print('Target variable saved in a variable for further use!')


# In[ ]:


def getnumcatfeat(df):
    
    """Returns two lists of numeric and categorical features"""
    
    numfeat, catfeat = list(df.select_dtypes(include=np.number)), list(df.select_dtypes(exclude=np.number))
    return numfeat, catfeat

numfeat, catfeat = getnumcatfeat(train)
numfeat.remove(target)

print('Categorical & Numeric features seperated in two lists!')


# Completing seperation of features wrt their type, the first thing I want to see is how my data looks.
# ### Starting off with the target variable.

# In[ ]:


fig, a = plt.subplots(nrows=1, ncols=2, figsize = (20,7))


sns.distplot(train[target], fit = norm, ax=a[0])
(mu, sig) = norm.fit(train[target]) 
a[0].legend(['$\mu=$ {:.2f}, $\sigma=$ {:.2f}, Skew = {:.4f}, Kurtosis = {:.4f}'.format(mu,sig,train[target].skew(),train[target].kurt())])
a[0].set_ylabel('Frequency')
a[0].axvline(train[target].mean(), color = 'Red') 
a[0].axvline(train[target].median(), color = 'Green') 
a[0].set_title(target + ' distribution')

temp=np.log1p(train[target])

sns.distplot(temp, fit = norm, ax=a[1])
(mu, sig) = norm.fit(temp) 
a[1].legend(['$\mu=$ {:.2f}, $\sigma=$ {:.2f}, Skew = {:.4f}, Kurtosis = {:.4f}'.format(mu,sig,temp.skew(),temp.kurt())])
a[1].set_ylabel('Frequency')
a[1].axvline(temp.mean(), color = 'Red')
a[1].axvline(temp.median(), color = 'Green') 
a[1].set_title('Transformed '+ target + ' distribution')

train[target] = np.log1p(train[target])

plt.show()


# Log transform gives the best result as data has almost exponential tendency.

# The target variable is very close to a Normal Distribution now. 
# ### Moving ahead to another variables.

# In[ ]:


temp = 'OverallQual'
f,a = plt.subplots(figsize=(8,6))
sns.boxplot(x= temp, y = target, data = train)
plt.show()


# Data has linear form with a few outliers. 

# In[ ]:


train.drop(train[((train[temp]==3) | (train[temp]==4)) & (train[target]<10.75)].index, inplace=True)


# 3 outliers removed.

# In[ ]:


temp = 'GrLivArea'
f,a = plt.subplots(figsize=(8,6))
sns.scatterplot(x=temp, y=target, data=train)
corr, _ = stats.pearsonr(train[temp], train[target])
plt.title('Pearsons correlation: %.3f' % corr)
plt.show()


# Looks linear but two outliers very evident.

# In[ ]:


train.drop(train[(train[temp]>4000) & (train[target]<12.5)].index, inplace=True)


# Removed 2 outliers.

# In[ ]:


temp = 'GarageCars'
f,a = plt.subplots(figsize=(8,6))
sns.boxplot(x=temp,y=target,data=train)
plt.show()


# I feel nothing requires removal for garagecars

# In[ ]:


temp = 'TotalBsmtSF'
f,a = plt.subplots(figsize=(8,6))
sns.scatterplot(x=temp,y=target,data=train)
corr, _ = stats.pearsonr(train[temp], train[target])
plt.title('Pearsons correlation: %.3f' % corr)
plt.show()


# Shows a very good correlation shown. So nothing requires removal.

# I feel like dropping a few columns which show multicolinearity.

# In[ ]:


cor = train.corr()
f,a = plt.subplots(figsize=(15,10))
sns.heatmap(cor)
plt.show()


# Numerical data will help better decide which rows to drop.

# In[ ]:


topn = 20
print('Top ', topn, ' correlated features to target features')
cor[target].sort_values(ascending=False)[1:(topn+1)]


# In[ ]:


s = cor.unstack().sort_values(ascending = False)[len(cor):]

topn = 20
print('Top', int(topn/2), 'correlated features\n')
s[:topn:2]


# In[ ]:


train_labels = train[target].reset_index(drop=True)
train_features = train.drop(target, axis=1)
test_features = test

df = pd.concat([train_features, test_features]).reset_index(drop=True)

## dropping the columns with multicollinearity
df.drop(['GarageArea', 'GarageYrBlt', 'TotRmsAbvGrd', '1stFlrSF'], axis=1, inplace=True)

df.shape


# I tried to remove a few least correlated features from the training set. The model turned out to give bad results so I stuck with not dropping them

# In[ ]:


# topn = 20
# # print(cor[target].sort_values()[:topn])
# temp = list(cor[target].sort_values()[:topn].index)
# df.drop(temp, axis=1, inplace=True)

# print('Removed least ', topn, ' correlated features')


# In[ ]:


numfeat, catfeat = getnumcatfeat(df)


# ### To check if any feature type is misclassified

# In[ ]:


temp = 0


# In[ ]:


assert(temp<len(catfeat))
print('Feature name: ', catfeat[temp], '\nUnique values: ', df[catfeat[temp]].unique(), 
      '\nData type: ', df[catfeat[temp]].dtype, '\nValue ', temp, ' out of ', len(catfeat))
temp +=1


# None of the categorical features seem to be misclassified. Checking for numeric features.

# In[ ]:


temp = 0


# In[ ]:


assert(temp<len(numfeat))
print('Feature name: ', numfeat[temp], '\nUnique values: ', df[numfeat[temp]].unique(), 
      '\nData type: ', df[numfeat[temp]].dtype, '\nIndex ', temp, ' out of ', len(numfeat)-1)
temp +=1


# In[ ]:


df['YearBuilt'] = df['YearBuilt'].astype('category')
df['YearRemodAdd'] = df['YearRemodAdd'].astype('category')
df['MoSold'] = df['MoSold'].astype('category')
df['YrSold'] = df['YrSold'].astype('category')


# In[ ]:


numfeat, catfeat = getnumcatfeat(df)


# Changed variable type of category as misclassified in data. Chose category instead of object/string as models work faster on them.

# Feeling good about the data. Moving towards the combined data preprocessing.

# # DATA PREPROCESSING

# ## Data Cleaning

# Dealing with missing values

# In[ ]:


temp = df.isnull().sum().sort_values(ascending=False)/df.shape[0]*100
temp = temp[temp>0]
temp = pd.DataFrame(temp, columns = ['MissPercent'])

f,a = plt.subplots(figsize=(12,10))

sub = sns.barplot(x='MissPercent', y = temp.index, data=temp, orient='h')
plt.title('Percent of missing values, size = '+ str(temp.shape[0]))
## Annotating the bar chart
for p,t in zip(sub.patches, temp['MissPercent']):
    plt.text(2.3+p.get_width(), p.get_y()+p.get_height()/2, '{:.2f}'.format(t), ha='center', va = 'center')

sns.despine(top=True, right=True)

plt.show()


# In[ ]:


def findit(df, strin):
    """
    CONVENIENCE FUNCTION FOR ANOTHER FUNCTION
    """

    temp = []
    for col in df.columns:
        if col[:len(strin)]==strin:
            temp.append(col)
    if len(temp)==0:
        return 0
    return temp

def fillit(df, strin):
    """
    
    CONVENIENCE FUNCTION
    
    Finds features beginning with 'strin' in its beginning.
    Then fills null values of categorical and numeric features
    with str('None') and int(0) values respectively.
    
    """
    temp = findit(df,strin)
    for col in temp:
        if df[col].dtype == object:
            df[col].fillna('None', inplace=True)
        else:
            df[col].fillna(0, inplace=True)
    return None


# In[ ]:


df['PoolQC'].fillna('None', inplace=True)
df['MiscFeature'].fillna('None', inplace=True)
df['Alley'].fillna(df['Alley'].mode()[0], inplace=True)
df['Fence'].fillna('None', inplace=True)
df['FireplaceQu'].fillna("None", inplace=True)
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
fillit(df,'Garage')
fillit(df,'Bsmt')
fillit(df,'Mas')
df['MSZoning'] = df.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
# df['MSZoning'].fillna(df['MSZoning'].mode()[0], inplace=True)
df['Functional'].fillna('Typ', inplace=True)
# Replace the missing values in each of the columns below with their mode
df['Electrical'].fillna(df['Electrical'].mode()[0],inplace=True)
df['KitchenQual'].fillna(df['KitchenQual'].mode()[0],inplace=True)
df['Exterior1st'].fillna(df['Exterior1st'].mode()[0],inplace=True)
df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0],inplace=True)
df['SaleType'].fillna(df['SaleType'].mode()[0],inplace=True)
df['Utilities'].fillna(df['Utilities'].mode()[0], inplace=True)
df['TotalBsmtSF'].fillna(df['TotalBsmtSF'].mode()[0],inplace=True)


# In[ ]:


df.isnull().sum().sort_values(ascending=False)


# In[ ]:


temp = list(df[df['PoolQC']=='None']['PoolArea'].unique())
df['PoolArea'] = df['PoolArea'].replace(temp,0)


# A few wrong values exist for pool area where PoolQC is None. 

# No more null values. Looking forward to outlier removal and data transformation.

# In[ ]:


def minfun(lamb):
    return round(pd.Series(stats.boxcox(1+df[temp],lmbda=lamb)).skew(), 2)

def retlamb(df, numfeat, tol, n_iter=100):
    """
    
    CONVENIENCE FUNCTION
    
    Returns optimized values of lambda to be used in
    boxcox transformation for each feature so that 
    skewness is minimized
    
    """
    valLambda = {}
    lim1, lim2 = 0, 4
    idx=0
    for temp in numfeat:
        lim1, lim2 = 0, 2
        for i in range(n_iter):
            lamb1=0.5*(lim1+lim2)-tol
            cal1 = round(pd.Series(stats.boxcox(1+df[temp],lmbda=lamb1)).skew(), 4)
            lamb2=0.5*(lim1+lim2)+tol
            cal2 = round(pd.Series(stats.boxcox(1+df[temp],lmbda=lamb2)).skew(), 4)
            if abs(cal1)<abs(cal2):
                lim2=lamb2
            elif abs(cal1)>=abs(cal2) :
                lim1=lamb1
        valLambda[idx] = 0.5*(lim1+lim2)
        idx+=1
    return valLambda


# In[ ]:


valLambda = retlamb(df,numfeat, tol=0.0001, n_iter=1000)
valLambda


# In[ ]:


temp = 2
lamb = valLambda[temp]
# temp, lamb = 0, 1
f,a = plt.subplots(nrows=1, ncols=2, figsize=(16,6))

sns.distplot(df[numfeat[temp]], ax=a[0], kde=False)
a[0].legend(['$\mu=$ {:.2f}, $\sigma=$ {:.2f}, Skew = {:.4f}, Kurtosis = {:.4f}'.format(mu,sig,df[numfeat[temp]].skew(),df[numfeat[temp]].kurt())])
a[0].set_title('Distribution of '+ numfeat[temp])

tempdf = pd.Series(stats.boxcox(1+df[numfeat[temp]],lmbda=lamb))

sns.distplot(tempdf, ax=a[1], kde=False)
a[1].legend(['$\mu=$ {:.2f}, $\sigma=$ {:.2f}, Skew = {:.4f}, Kurtosis = {:.4f}'.format(mu,sig,tempdf.skew(),tempdf.kurt())])
a[1].set_title('Transformed distribution of '+ numfeat[temp])

valLambda[temp] = lamb

plt.show()


# In[ ]:


for temp, lamb in valLambda.items():
    df[numfeat[temp]] = stats.boxcox(1+df[numfeat[temp]],lmbda=lamb)
    
# ## SIMPLE SIMPLE FIX FOR ALL DRAMA DONE IS ALL THE ABOVE 3 CELLS but my approach gave better results so I happily stuck with it
# for temp in range(len(numfeat)):
#     df[numfeat[temp]] = stats.boxcox(1+df[numfeat[temp]], stats.boxcox_normmax(df[numfeat[temp]] + 1))


# In[ ]:


df[numfeat].skew()


# In[ ]:


df = pd.get_dummies(df).reset_index(drop=True)
df.shape


# In[ ]:


minmaxscalar = prep.MinMaxScaler()
df1 = pd.DataFrame(minmaxscalar.fit_transform(df), columns = df.columns)
df1.head()


# In[ ]:


X = df1.iloc[:len(train_labels),:]
X_test = df1.iloc[len(train_labels):, :]
y = train_labels

X.shape, X_test.shape, y.shape


# # MODEL FITTING

# In[ ]:


from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.linear_model import Lasso


# In[ ]:


# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

kf = KFold(n_splits = 7, random_state=0, shuffle=True)

scores = {}


# In[ ]:


def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)


# ## Light GB model

# In[ ]:


lightgbm = LGBMRegressor(objective='regression', 
                       num_leaves=6,
                       learning_rate=0.009, 
                       n_estimators=7000,
                       max_bin=200, 
                       bagging_fraction=0.8,
                       bagging_freq=4, 
                       bagging_seed=8,
                       feature_fraction=0.2,
                       feature_fraction_seed=8,
                       min_sum_hessian_in_leaf = 11,
                       verbose=-1,
                       random_state=42)


# In[ ]:


score = cv_rmse(lightgbm)
print("lightgbm: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['lgb'] = (score.mean(), score.std())


# ## XG Boost model

# In[ ]:


xgboost = XGBRegressor(learning_rate=0.01,
                       n_estimators=6000,
                       max_depth=4,
                       min_child_weight=0,
                       gamma=0.6,
                       subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:squarederror',
                       nthread=-1,
                       scale_pos_weight=1,
                       seed=27,
                       reg_alpha=0.00006,
                       random_state=42)


# In[ ]:


score = cv_rmse(xgboost)
print("xgboost: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['xgb'] = (score.mean(), score.std())


# ## Random Forest model

# Showing one scenario of how I tuned the parameters of my model.

# In[ ]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 50)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 300, 100)]

# Minimum number of samples required to split a node
min_samples_split = [int(x) for x in np.linspace(2,500,100)]

# Minimum number of samples required at each leaf node
min_samples_leaf = [int(x) for x in np.linspace(2,500,100)]

# Create the random grid
rforestgrid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}


# In[ ]:


rforest = RandomForestRegressor()


# In[ ]:


# clf = RandomizedSearchCV(rforest, rforestgrid, cv=5, n_iter=50, n_jobs=1)
# # search = clf.fit(X,y)


# In[ ]:


# search.best_params_


# In[ ]:


rforest_tuned = RandomForestRegressor(n_estimators = 1000,
                                       min_samples_split = 2,
                                       min_samples_leaf = 1,
                                       max_features = 'auto',
                                       max_depth = 100
                                      )


# In[ ]:


score = cv_rmse(rforest_tuned)
print("rforest: {:4f} ({:.4f})".format(score.mean(), score.std()))
scores['rforest'] = (score.mean(), score.std())


# ## Lasso Regr Model

# In[ ]:


lasso = Lasso(alpha=0.000328)


# This is a parameter I tuned myself and it gave pretty good results. :P

# In[ ]:


score = cv_rmse(lasso)
print("lasso: {:4f} ({:.4f})".format(score.mean(), score.std()))
scores['lasso'] = (score.mean(), score.std())


# # Submission 

# In[ ]:


model = lasso.fit(X, y)


# In[ ]:


submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
submission.shape


# In[ ]:


model.predict(X_test).shape


# In[ ]:


submission.iloc[:,1] = np.floor(np.expm1(model.predict(X_test)))


# In[ ]:


submission.to_csv("submission_try4.csv", index=False)


# In[ ]:


## Downloading the submission file

from IPython.display import FileLink
FileLink('submission_try4.csv')

