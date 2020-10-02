#!/usr/bin/env python
# coding: utf-8

# **Importing essential libraries**

# In[346]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[347]:


train = pd.read_csv('../input/train.csv')
train.head()


# In[348]:


test = pd.read_csv('../input/test.csv')
test.head()


# In[349]:


train.info()


# Let us first deal with the outliers in the dataset..
# According to the dataset GrLivArea seems to be best to plot against sale price to find outliers.

# In[350]:


from IPython.display import IFrame
IFrame('https://public.tableau.com/profile/nitin2520#!/vizhome/FindingOutliersHouseRegression/Sheet1?publish=yes', width=1000, height=925)


# It is very clear that the two points on the bottom right are the outliers of the data. But it is not always good to remove outliers, removing many of them can decrease the quality of the model. But here are only two so it's better to remove them

# In[351]:


train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
train.reset_index(drop=True, inplace=True)
train.info()


# **Checking for Null values**

# In[352]:


sns.heatmap(train.isnull(),yticklabels=False, cmap = 'RdYlGn',cbar=False)


# In[353]:


# for LotFrontage it seems good to fill null values using mean
train['LotFrontage'].fillna(train['LotFrontage'].mean(), inplace = True)
test['LotFrontage'].fillna(test['LotFrontage'].mean(), inplace = True)


# In[354]:


# those not having MasVnrType we can take them as None Type which is also the mode
# Having None MasVnrType will also have 0 area which is also the mode so filling both of them by mode
train['MasVnrType'].fillna(train['MasVnrType'].mode()[0], inplace = True)
train['MasVnrArea'].fillna(train['MasVnrArea'].mode()[0], inplace = True)
test['MasVnrType'].fillna(test['MasVnrType'].mode()[0], inplace = True)
test['MasVnrArea'].fillna(test['MasVnrArea'].mode()[0], inplace = True)


# In[355]:


print('Pearson: ')
train[['BsmtFinType1', 'BsmtFinSF1']].corr(method='pearson')


# In[356]:


# Tableau visualization sheet 2  and sheet 3 needed


# as we can see clearly in the graph that all the points whose BsmtFinType is NA have BsmtFinSF1 as 0 so we will fill all those by Unf the same is the case with BsmtFinType2

# In[357]:


train['BsmtFinType1'].fillna('Unf', inplace = True)
train['BsmtFinType2'].fillna('Unf', inplace = True)
test['BsmtFinType1'].fillna('Unf', inplace = True)
test['BsmtFinType2'].fillna('Unf', inplace = True)


# In[358]:


# sheet 4


# In[359]:


# all those points having NA in BsmtQual lies on BsmtFullBath as 0 and in that the maximum nuber of time TA occurs so we will fill this 
# with the mode which is TA in this the same is with the test set
train['BsmtQual'].fillna('TA', inplace = True)
test['BsmtQual'].fillna('TA', inplace = True)


# In[360]:


# Similarly the BsmtCond also depends according to BsmtFullBath on 0 so we will also fill null values with TA
train['BsmtCond'].fillna('TA', inplace = True)
test['BsmtCond'].fillna('TA', inplace = True)


# In[361]:


# Similarly we will fill BsmtExposure with No
train['BsmtExposure'].fillna('No', inplace = True)
test['BsmtExposure'].fillna('No', inplace = True)


# In[362]:


# all those data having fireplace 0 have FireplaceQu as null so we will fill those with new value as No
train['FireplaceQu'].fillna('No', inplace = True)
test['FireplaceQu'].fillna('No', inplace = True)


# In[363]:


# We will drop these features as thse have maximum as null values
train.drop(['MiscFeature', 'Fence', 'PoolQC', 'Alley'], axis = 1, inplace = True)
test.drop(['MiscFeature', 'Fence', 'PoolQC', 'Alley'], axis = 1, inplace = True)


# Now most of the training data is cleaned but still the test data have many null values which need to be cleared.
# So now we will also visualize the test data.

# In[364]:


train.info()
test.info()


# In[365]:


# As can be easily seen in graph there is only one data point which has bot garagecars and garagearea value as null
# So as there are also 76 other records in which both are 0 so will fill this record with also 0
test['GarageCars'].fillna( 0 , inplace = True)
test['GarageArea'].fillna( 0 , inplace = True)


# In[366]:


# sheet 7
# In Train Set all records having GarageQual as NA has GarageCars as 0 so will replace all NA with new value as NO
# -----------/
# while in Test Set there are all record having GarageQual as NA has GarageCars also 0 but there is 1 record which have GarageCars as 1
# So of that special record we will fill with 'TA' while of others with new value as 'NO'
train['GarageQual'].fillna('No', inplace = True)
test['GarageQual'].fillna('No', inplace = True)


# In[367]:


def compute_garageQual(cols):
    cars = cols[0]
    qual = cols[1]
    
    if str(qual) == 'No' :
        if cars == 1 :
            return 'TA'
        else:
            return 'No'
    else:
        return qual


# In[368]:


# replacing No with TA for particular datapoint
test['GarageQual'] = test[['GarageCars', 'GarageQual']].apply(compute_garageQual, axis = 1)
test.info()


# In[369]:


#  Sheet 8
# In GargageType where values are null all have GarageCars as 0 so will create a new value as No
train['GarageType'].fillna('No', inplace = True)
test['GarageType'].fillna('No', inplace = True)
# train.info()
# test.info()


# In[370]:


# Sheet 9 and Sheet 2
# The case co GarageYrBlt is same to GarageQual 
# Similarly to that there is 1 point which has GarageCars as 1 and GarageYrBlt as null 
#  We will fill taht special value with the median as expressed in the graph
train['GarageYrBlt'].fillna( 0, inplace = True)
test['GarageYrBlt'].fillna( 0, inplace = True)
# train.info()


# In[371]:


def compute_yrblt(cols):
    cars = cols[0]
    yrblt = cols[1]
    
    if yrblt == 0 :
        if cars == 1 :
            return 1956.5
        else:
            return 0
    else:
        return yrblt


# In[372]:


# Replacing in test set with median for special case
test['GarageYrBlt'] = test[['GarageCars', 'GarageYrBlt']].apply(compute_yrblt, axis = 1)
# test.info()


# In[373]:


#  Sheet 3
# Similarly is the case with GaregeCond
train['GarageCond'].fillna( 'No', inplace = True)
test['GarageCond'].fillna( 'No', inplace = True)


# In[374]:


def compute_garagecond(cols):
    cars = cols[0]
    cond = cols[1]
    
    if str(cond) == 'No' :
        if cars == 1 :
            return 'TA'
        else:
            return 'No'
    else:
        return cond


# In[375]:


# Replacing the special Value with 'TA'
test['GarageCond'] = test[['GarageCars', 'GarageCond']].apply(compute_garagecond, axis = 1)
# test.info()


# In[376]:


# The Same is the case with GarageFinish foe special Value
train['GarageFinish'].fillna( 'No', inplace = True)
test['GarageFinish'].fillna( 'No', inplace = True)


# In[377]:


def compute_garagefinish(cols):
    cars = cols[0]
    finish = cols[1]
    
    if str(finish) == 'No' :
        if cars == 1 :
            return 'RFn'
        else:
            return 'No'
    else:
        return finish


# In[378]:


# Replacing with special value
test['GarageFinish'] = test[['GarageCars', 'GarageFinish']].apply(compute_garagefinish, axis = 1)
# train.info()
# test.info()


# In[379]:


# There is 1 null value left in the training set of Electrical Column so dropping the row
# As dropping single row will not affect our model
train.dropna(inplace = True)
# train.info()
# test.info()


# **Now All the training set is cleaned buut there are few test set columns still left now dealing with them**

# In[380]:


# Comparing on Different MS Sub Classes MSZoning as RL is always Maximum so we will fill its null values with Rl
test['MSZoning'].fillna( 'RL', inplace = True)
# test.info()


# In[381]:


# Replacing the utilities with mode as its optimum for filling those
test['Utilities'].fillna( test['Utilities'].mode()[0], inplace = True)
# test.info()


# In[382]:


# Replacing null values of Exterior1st and 2nd with mode
test['Exterior1st'].fillna( test['Exterior1st'].mode()[0], inplace = True)
test['Exterior2nd'].fillna( test['Exterior2nd'].mode()[0], inplace = True)
# test.info()


# In[383]:


# Replacing null values of SF1, SF2, SF with zero
test['BsmtFinSF1'].fillna( 0, inplace = True)
test['BsmtFinSF2'].fillna( 0, inplace = True)
test['BsmtUnfSF'].fillna( 0, inplace = True)
test['TotalBsmtSF'] = test['BsmtFinSF1'] + test['BsmtFinSF2'] + test['BsmtUnfSF']
# test.info()


# In[384]:


# Those whose BsmtFullBath and HalfBath are not given it would be best to fill them with 0
test['BsmtFullBath'].fillna( 0, inplace = True)
test['BsmtHalfBath'].fillna( 0, inplace = True)
# test.info()


# In[385]:


# replacing functional with mode
test['Functional'].fillna( test['Functional'].mode()[0], inplace = True)
# test.info()


# In[386]:


# sheet 8 of train
# As when we compare kitchenqual with kitchenabvGr always 'TA' are the most so replacing them with 'TA'
test['KitchenQual'].fillna( 'TA', inplace = True)
# test.info()


# In[387]:


# Thwrw is 1 null in saletype and the sale condition of that is normal 
# so will replace it with 'WD'
test['SaleType'].fillna( 'WD', inplace = True)
# train.info()
# test.info()


# In[388]:


train.drop(['Id'], axis = 1, inplace = True)
test.drop(['Id'], axis = 1, inplace = True)


# In[389]:


#  just copyiing to a new dataframe
train2 = train.copy()
test2 = test.copy()


# Target Variable Sale Price Checking Skewness and Improving it with transformation

# In[390]:


Y = train2['SalePrice'].values
train2.drop(['SalePrice'], axis = 1, inplace = True)
train2.info()


# In[391]:


#Acquiring mu and sigma for normal distribution plot of 'SalePrice'
from scipy import stats
from scipy.stats import norm,skew
(mu, sigma) = norm.fit(Y)

#Plotting distribution plot of 'SalePrice' and trying to fit the normal distribution corve on that
plt.figure(figsize=(8,8))
ax = sns.distplot(train['SalePrice'] , fit=norm);
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.show()


# It can be clearly seen that the above distribution is right skewed

# In[392]:


fig, ax = plt.subplots(figsize=(8,8))

sns.distplot(Y, kde=False,color = 'green', fit=stats.lognorm)

ax.set_title("Log Normal",fontsize=24)
plt.show()


# In[393]:


Y


# In[394]:


y


# In[395]:


# Applying the log transformation now
y = np.log(Y)


# In[396]:


plt.figure(figsize=(8,8))
ax = sns.distplot(y , fit=stats.norm);
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
plt.legend(['Normal Distribution'],
            loc='best')
plt.show()


# In[397]:


import category_encoders as ce
ohe = ce.OneHotEncoder(handle_unknown='ignore', use_cat_names=True)
train2 = ohe.fit_transform(train2)
test2 = ohe.transform(test2)
train2.info()


# In[398]:


train_final = train2.values
test_final = test2.values


# In[399]:


from sklearn.preprocessing import MinMaxScaler
SC = MinMaxScaler(feature_range = (0,1))
train_final = SC.fit_transform(train_final)
test_final = SC.transform(test_final)


# In[400]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_final, 
                                                    y, test_size=0.2, 
                                                    random_state=42)


# In[401]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'poly', degree = 4, epsilon = 0.01, gamma = 0.5 )
regressor.fit(X_train, y_train)


# In[402]:


y_pred = (regressor.predict(X_test))
from sklearn.metrics import mean_squared_error, r2_score
print("R2 score : %.2f" % r2_score(y_test,y_pred))
print("Mean squared error: %.2f" % mean_squared_error(y_test,y_pred))


# In[403]:


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X_train)
X_test = poly_reg.transform(X_test)


# In[404]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression(n_jobs = -1)
lin_reg.fit(X_poly,y_train)


# In[405]:


y_pred = (lin_reg.predict(X_test))


from sklearn.metrics import mean_squared_error, r2_score
print("R2 score : %.2f" % r2_score(y_test,y_pred))
print("Mean squared error: %.2f" % mean_squared_error(y_test,y_pred))


# In[406]:


from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error


# In[407]:


params = {'n_estimators': 2000, 'max_depth': 6, 'min_samples_split': 30, 'min_samples_leaf': 1, 'max_features': 50,
          'learning_rate': 0.01, 'loss': 'huber', 'subsample': 0.8 , 'validation_fraction': 0.01}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)
#mse = mean_squared_error(y_test, clf.predict(X_test))
#print("MSE: %.4f" % mse)


# In[408]:


# y_pred = clf.predict(X_test)
# from sklearn.metrics import mean_squared_error, r2_score
# print("R2 score : %.2f" % r2_score(y_test,y_pred))
# print("Mean squared error: %.2f" % mean_squared_error(y_test,y_pred))


# In[410]:


y_pred = regressor.predict(test_final)
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)
#cm
y_pred = np.exp(y_pred)
y_pred = list(y_pred)
print(len(y_pred))


test4 = pd.read_csv('../input/test.csv')
#test4.head()
passengerid = list(test4['Id'])
dictionary = {'Id':passengerid, 'SalePrice':y_pred}
df = pd.DataFrame(dictionary)
df.head()
df.to_csv('gradient.csv',index = False)

