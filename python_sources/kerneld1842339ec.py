#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import some necessary librairies

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns

from scipy import stats
from scipy.stats import norm, skew #for some statistics


# In[ ]:


# import the train and test datasets 

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[ ]:


test.shape


# In[ ]:


#display the first five rows of the train dataset.
train.head()


# In[ ]:


##display the first five rows of the test dataset.
test.head()


# In[ ]:


#check the numbers of samples and features
train.shape


# In[ ]:


test.shape


# In[ ]:


#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']


# In[ ]:


#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)


# In[ ]:


#check again the numbers of samples and features
train.shape


# In[ ]:


test.shape


# In[ ]:


#descriptive statistics summary
train['SalePrice'].describe()


# In[ ]:


#scatter plot grlivarea/saleprice
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


#scatter plot totalbsmtsf/saleprice
fig, ax = plt.subplots()
ax.scatter(x = train['TotalBsmtSF'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('TotalBsmtSF', fontsize=13)
plt.show()


# In[ ]:


#Deleting outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# In[ ]:


sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()


# In[ ]:


#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train["SalePrice"] = np.log1p(train["SalePrice"])

#Check the new distribution 
sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()


# In[ ]:


ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))


# In[ ]:


#percentage missing Data 
total = all_data.isnull().sum().sort_values(ascending=False)
percent = (all_data.isnull().sum()/all_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[ ]:


#Correlation map to see how features are correlated with SalePrice
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)


# In[ ]:


#saleprice correlation matrix
#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[ ]:


#dealing with missing data
all_data = all_data.drop((missing_data[missing_data['Total'] > 2346]).index,1)
all_data.isnull().sum().max() #just checking that there's no missing data missing...


# In[ ]:


# new percentage missing Data 
total = all_data.isnull().sum().sort_values(ascending=False)
percent = (all_data.isnull().sum()/all_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(30)


# In[ ]:


#imputing the remaining missing values with most applicable values

for col in ('Fence','FireplaceQu','GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','MSSubClass'):
    all_data[col] = all_data[col].fillna('None')


# In[ ]:


#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))


# In[ ]:


all_data.drop('GarageArea',axis =1, inplace = True)
for col in ('GarageYrBlt', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)


# In[ ]:


for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)


# In[ ]:


all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)


# In[ ]:


all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])


# In[ ]:


all_data = all_data.drop(['Utilities'], axis=1)


# In[ ]:


all_data["Functional"] = all_data["Functional"].fillna("Typ")


# In[ ]:


all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])


# In[ ]:


all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])


# In[ ]:


all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])


# In[ ]:


all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])


# In[ ]:


#Check remaining missing values if any 
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing percentage' :all_data_na})
missing_data.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street','CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print('Shape all_data: {}'.format(all_data.shape))


# In[ ]:


# Adding total sqfootage feature 
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']


# In[ ]:


numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(15)


# In[ ]:


skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)
    
#all_data[skewed_features] = np.log1p(all_data[skewed_features])


# In[ ]:


all_data = pd.get_dummies(all_data)
print(all_data.shape)


# In[ ]:


train = all_data[:ntrain]
test = all_data[ntrain:]


# In[ ]:


test.shape , train.shape


# In[ ]:


X = train
y = y_train


# In[ ]:


X.shape, test.shape


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler = StandardScaler()


# In[ ]:


X_scaled = scaler.fit_transform(X)
test_scaled = scaler.fit_transform(test)
X_scaled.shape, test_scaled.shape


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, 
                                                    y, 
                                                    test_size=0.20, 
                                                    shuffle=False)


# In[ ]:


X_train.shape


# In[ ]:


from sklearn.linear_model import Lasso


# In[ ]:


lasso = Lasso(alpha=0.01)


# In[ ]:


lasso.fit(X_train, y_train)


# In[ ]:


X_train.shape , test_scaled.shape


# In[ ]:


intercept = float(lasso.intercept_)


# In[ ]:


coeff = pd.DataFrame(lasso.coef_, X.columns, columns=['Coefficient'])


# In[ ]:


print("Intercept:", float(intercept))


# In[ ]:


coeff


# In[ ]:


from sklearn import metrics


# In[ ]:


# check training accuracy
train_lasso = lasso.predict(X_train)

print('Training MSE')
print('Lasso :', metrics.mean_squared_error(y_train, train_lasso))


# In[ ]:


test_lasso = lasso.predict(test_scaled)

#print(test_lasso)


# In[ ]:


predict = np.exp(test_lasso)


# In[ ]:


predict


# In[ ]:


test_ID.shape


# In[ ]:


#Prepare Submission File
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = predict
sub.to_csv('submit.csv',index=False)


# In[ ]:


sub


# In[ ]:





# In[ ]:




