#!/usr/bin/env python
# coding: utf-8

# # Quantile Regression vs. Gradient Boosting Regression

# Advanced machine learning algorithms are taking over traditional methods a lot lately.  
# But how close traditional model prediction can get to advance ML prediction?  
# 
# This notebook gives a chance to compare predictions of Gradient Boosting regression vs. traditional regression.
# To keep this comparison adequate we will build:
# 1. Quantile Regression (median)
# 2. Gradient boosted regression with least absolute deviation loss function (quantile=0.5)
# 
# So in both cases we are predicting median house price, hense the loss functions used in both approaches is the same (LAD).
# 
# #### Data and the Challenge
# With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this Kaggle competition challenges you to predict the final price of each home. More details about the competition & data can be found [here](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
# 
# 

# #### Importing the Libraries

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from statsmodels.graphics.gofplots import qqplot
plt.style.use('bmh')
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.decomposition import PCA
import statsmodels.formula.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points


# #### Read in Training Data

# In[ ]:


df_train = pd.read_csv('../input/train.csv')


# In[ ]:


df_train.head()


# In[ ]:


df_test = pd.read_csv('../input/test.csv')
set(df_train.columns).difference(df_test.columns)


# In[ ]:


df_train['IsTrain'] = 1
df_test['IsTrain']=0
df = pd.concat((df_train, df_test), sort=False).reset_index(drop=True)


# In[ ]:


df.groupby('IsTrain')['IsTrain'].count()


# *Will leave the testing sample out of analysis for now (we're not submitting anything here)*

# ## Exploratory Data Analysis

# Let's drop ID into a separate dataframe for later

# In[ ]:


df_IDs = df[['Id','IsTrain']]
del df['Id']


# #### Identifying categorical features by number of unique values

# In[ ]:


VarCatCnt = df[:].nunique()
df_uniq = pd.DataFrame({'VarName': VarCatCnt.index,'cnt': VarCatCnt.values})
df_uniq.head()


# Create a subset of continuous numeric features (i.e. that have more than 30 categories)

# In[ ]:


df_contVars = df_uniq[(df_uniq.cnt > 30) | (df_uniq.VarName == 'SalePrice') | (df_uniq.VarName == 'IsTrain')]
df_num = df[df_contVars.VarName]
df_num.shape


# Create a subset of categorical features (i.e. with less than 30 categories)

# In[ ]:


df_categVars = df_uniq[(df_uniq.cnt <= 30) | (df_uniq.VarName == 'SalePrice') | (df_uniq.VarName == 'IsTrain')]
df_categ = df[df_categVars.VarName]
df_categ.shape


# ### Imputing Missing Values

# Will impute 0 for missings in **numeric** features, since there should be no confusion around the meaning of it

# In[ ]:


for i in range(0, len(df_num.columns)-1):
    if (df_num[df_num.columns[i]].isnull().sum() != 0) & (df_num.columns[i] != 'SalePrice'):
        print(df_num.columns[i] + " Null Count:" + str(df_num[df_num.columns[i]].isnull().sum()))
        df_num[df_num.columns[i]] = df_num[df_num.columns[i]].fillna(0)


# Will impute 'NA' for missing values in **categoric** features (object type)

# In[ ]:


for i in range(0, len(df_categ.columns)-1):
    if (df_categ[df_categ.columns[i]].dtype == object) & (df_categ[df_categ.columns[i]].isnull().sum() != 0):
            #print(df_categ.columns[i] + " Null Count:" + str(df_categ[df_categ.columns[i]].isnull().sum()))
            df_categ[df_categ.columns[i]].replace(np.nan, 'NA', inplace= True)


# Checking if there are missing values in features with numeric-categories

# In[ ]:


df_categ_N = df_categ.select_dtypes(include = ['float64', 'int64'])
df_categ_N.info()


# In[ ]:


df_categ['BsmtFullBath'].replace(np.nan, 0, inplace= True)
df_categ['BsmtHalfBath'].replace(np.nan, 0, inplace= True)


# ### Categoric Feature Analysis

# Boxplot gives a good idea about how SalePrice is distributed across the categories.  
# Distribution charts show how well each category is populated.

# In[ ]:


c = 0
len_c = 3 # (len(df_categ.columns)-2)
fig, axes = plt.subplots(len_c, 2, figsize=(10, 13))     # fig height = 70 -> in figsize(width,height)
for i, ax in enumerate(fig.axes):
    if (c < len_c) & (i % 2 == 0):
        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
        sns.countplot(x=df_categ.columns[c], alpha=0.7, data=df_categ, ax=ax)

    if (c < len_c) & (i % 2 != 0):
        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
        sns.boxplot(data = df_categ, x=df_categ.columns[c], y='SalePrice', ax=ax)
        c = c + 1
fig.tight_layout()


# > Will do part of encoding using 'ranking' based on median Sale price at later step

# Let's move the numeric features that don't need to be encoded to *df_num* dataframe  
# (they fell-in the categoric feature pot,  because of small number of values)

# In[ ]:


df_num.insert(loc=20, column='OverallQual', value=df_categ[['OverallQual']])
df_num.insert(loc=20, column='OverallCond', value=df_categ[['OverallCond']])
df_num.insert(loc=20, column='YrSold', value=df_categ[['YrSold']])
df_num.insert(loc=20, column='TotRmsAbvGrd', value=df_categ[['TotRmsAbvGrd']])
df_num.insert(loc=20, column='Fireplaces', value=df_categ[['Fireplaces']])
df_num.insert(loc=20, column='GarageCars', value=df_categ[['GarageCars']])


# In[ ]:


df_num['GarageCars'].fillna(0, inplace=True)


# In[ ]:


df_categ.drop(['GarageCars'
                ,'Fireplaces'
                ,'TotRmsAbvGrd'
                ,'YrSold'
                ,'OverallQual'
                ,'OverallCond']
            , axis=1, inplace = True)


# ### Categoric Feature Encoding

# Encoding the Quality type categoric features

# In[ ]:


CatVarQual = ['ExterQual','BsmtQual','HeatingQC','KitchenQual','FireplaceQu','GarageQual']
map_dict = {'Ex': 5,
            'Gd': 4,
            'TA': 3,
            'Fa': 2,
            'Po': 1,
            'NA': 3}

df_categQ = pd.DataFrame()
for i in range(0, len(CatVarQual)):
    df_categQ[CatVarQual[i]+'_N'] = df_categ[CatVarQual[i]].map(map_dict)


# List the other categoric features that need to be encoded

# In[ ]:


CatVarList = [column for column in df_categ if (column not in set(CatVarQual))]


# Transforming **categorical features** into numeric, based on the order of median *SalePrice* per category (and actual median value, giving the score to each category)

# In[ ]:


df_categN = df_categ
for i in range(0, len(CatVarList)-2):
    catVar = CatVarList[i]  #catVar = 'MSSubClass'
    cl = df_categ.groupby(catVar)['SalePrice'].median().sort_values()
    df_cl = pd.DataFrame({'Category': cl.index,'SortVal': cl.values})
    df_cl.replace(np.nan, df_categ['SalePrice'].median(), inplace= True)
    df_cl[catVar+'_N']=df_cl['SortVal']/10000
    #df_cl[catVar+'_N']=df_cl['SortVal'].rank()
    #print(df_cl) #if want to see how the categories got ranked
    df_categN = pd.merge(df_categN,
                        df_cl[['Category', catVar+'_N','SortVal']],
                        left_on=catVar,
                        right_on='Category',
                        how = 'left')
    df_categN.drop(['Category','SortVal',catVar], axis=1, inplace = True)
df_categN.drop(CatVarQual, axis=1, inplace = True)
#df_categN.columns


# In[ ]:


sns.pairplot(data=df_categ,
            x_vars='Neighborhood',
            y_vars=['SalePrice'],
            size = 6)
plt.xticks(rotation=45);


# In[ ]:


sns.pairplot(data=df_categN,
            x_vars='Neighborhood_N',
            y_vars=['SalePrice'],
            size = 6);


# Spearman correlation to see the affect of category ranking by median *SalePrice*

# In[ ]:


print("Before category encoding:")
df_n = df_categ[df_categ['IsTrain'] == 1]
print(spearmanr(df_n['SalePrice'],df_n['Neighborhood']))

print("After category encoding:")
df_n = df_categN[df_categN['IsTrain'] == 1]
print(spearmanr(df_n['SalePrice'],df_n['Neighborhood_N']))
print("Pearson corr = "+ str(df_n.corr(method='pearson')['SalePrice']['Neighborhood_N']))


# Let's join the 2 datasets of encoded variables

# In[ ]:


df_categN = pd.merge(df_categN, df_categQ, left_index=True, right_index=True, sort=False)
#df_categN.columns


#     
# #### Will join the numeric and encoded categoric features

# In[ ]:


df = pd.merge(df_categN[df_categN.columns[2:]], df_num, left_index=True, right_index=True, sort=False)


# ### *Numeric Feature Analysis*

# What's the shape of our target?    
# Let's check out some key stats and histogram of *SalePrice*

# In[ ]:


df_num['SalePrice'].describe()


# In[ ]:


#skewness and kurtosis
print("Skewness: %f" % df['SalePrice'].skew())
print("Kurtosis: %f" % df['SalePrice'].kurt())


# In[ ]:


sns.distplot(df[df['SalePrice'].isnull() == False]['SalePrice'],fit=norm);


# **The QQ-plot**
# gives a nice picture of how close the sample quantiles are to the theoretical normal distribution quantiles

# In[ ]:


qqplot(df[df['SalePrice'].isnull() == False]['SalePrice'], line='s');


# > Can see that the dependent variable will need a transformation to fulfill normality assumption, if linear regression is applied. Unlike the quantile regression, which is insensitive to monotone transformations of target variable. 

# #### Histogram of other Numeric Features

# In[ ]:


len_c = 4   #(len(df_num.columns)-2)
fig, axes = plt.subplots(round(len_c / 2), 2, figsize=(12, 10))     # fig height = 70 -> in figsize(width,height)
for i, ax in enumerate(fig.axes):
    if (i < len_c):
        sns.distplot(df_num[df_num['IsTrain']==1][df_num.columns[i]], label="IsTrain = 1", fit=norm, ax=ax)

fig.tight_layout()


# >There are features that have low coverage, keep in mind when evaluating its performance in the model outputs 

# In[ ]:


# Scatterplots SalePrice vs. numeric Vars
sns.pairplot(data=df_num,
            x_vars=df_num.columns[:4],
            y_vars=['SalePrice']);


# ### Correlation
# Let's check which features have strongest linear relationship with *SalePrice*

# In[ ]:


df_corr = df.corr(method='pearson')['SalePrice'][:-2]   
golden_features_list = df_corr[abs(df_corr) > 0.5].sort_values(ascending=False)
print("There is {} correlated values with SalePrice:\n{}".format(len(golden_features_list), golden_features_list))
#df[golden_features_list.index].head()


# Create ***TotArea*** feature to cover all the area inside house, since all of these are highly corelated and sensibly important variables when predicting *SalePrice*

# In[ ]:


df.insert(loc=0, column='TotArea', value=(df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']) )


# In[ ]:


df.drop(['TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea'], axis=1, inplace = True)


# Let's check the strength (and direction) of linear relationship between key features

# In[ ]:


df_corr = df.corr(method='pearson')['SalePrice'][:-2]  
golden_features_list = df_corr[abs(df_corr) > 0.3].sort_values(ascending=False)
Top_features_list = df_corr[abs(df_corr) > 0.5].sort_values(ascending=False)
#print("There is {} correlated values with SalePrice:\n{}".format(len(golden_features_list), golden_features_list))


# In[ ]:


#correlation matrix heatmap
corrmat = df[Top_features_list.index].corr()
f, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(corrmat, cmap="RdBu", vmin=-1, vmax=1, square=True, annot=True, fmt=".1f");


# > There is strong correlation between some features. Will use PCA to gather the correlated features into components.

# ## 1. Quantile Regression

# ### 1.1. Prep: Initial train/test split for Visualised Model Evaluation

# Will split the original training sample into train/test for model evaluation (70% training/ 30% testing)

# In[ ]:


# Separating out the features (and Training sample from Testing)
X = df.loc[:1459, golden_features_list.index].values

# Separating out the target
y = df.iloc[:1460,-2].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# ### 1.2. Standardize Data

# Feature standardization is required for Principal component analysis (PCA)

# In[ ]:


sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_Y = StandardScaler()
y_trainS = sc_Y.fit_transform(y_train.reshape(-1,1))
y_testS = sc_Y.transform(y_test.reshape(-1,1))


# In[ ]:


print("mean = " + str(np.mean(X_train[:,4])))
print("std = " + str(np.std(X_train[:,4])))


# ### 1.3. PCA for feature reduction and independence

# Let's put the potential predictors through PCA to create few independent components that could go into the model instead

# In[ ]:


pca = PCA(n_components = 5)
principalComponents = pca.fit_transform(X_train)
principalComponentsTest = pca.transform(X_test)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['PrincComp_1', 'PrincComp_2','PrincComp_3','PrincComp_4','PrincComp_5'])
principalDftest = pd.DataFrame(data = principalComponentsTest
             , columns = ['PrincComp_1', 'PrincComp_2','PrincComp_3','PrincComp_4','PrincComp_5'])


# Percentage of variace explained by the components together and seperately

# In[ ]:


print('Variance explained by all components: ' + str(pca.explained_variance_ratio_.sum()))
pca.explained_variance_ratio_


# Let's see which variables mostly contribute to which component

# In[ ]:


compareDf = pd.concat((principalDf, pd.DataFrame(X_train, columns = golden_features_list.index)), axis=1)


# In[ ]:


corrmat = compareDf.corr()['PrincComp_1':'PrincComp_5']
f, ax = plt.subplots(figsize=(20, 5))
sns.heatmap(corrmat, cmap="RdBu", vmin=-1, vmax=1, square=False, annot=True, fmt=".1f");


# We assign to create as many components as would explain 95% of the variance

# ### 1.4. Quantile Regression - model fitting

# The Least Absolute Deviation model is a special case of quantile regression where q=0.5 (median)

# In[ ]:


principalDf['SalePrice'] = y_trainS


# In[ ]:


mod = sm.quantreg('SalePrice ~ PrincComp_1 + PrincComp_2 + PrincComp_3 + PrincComp_4 + PrincComp_5', principalDf)
res = mod.fit(q=.5)
print(res.summary())


# ### 1.5. Model Evaluation

# Inverse the transformation (since we applied the transformation prior to analysis)

# In[ ]:


pred = res.predict(principalDftest) # make the predictions by the model
y_pred = sc_Y.inverse_transform(pred)


# See if the errors increase when SalePrice increases (e.g. maybe the model is performing better on lower-value houses rather than the expensive)

# In[ ]:


# Plot the y_test and the prediction (y_pred)
fig = plt.figure(figsize=(15, 5))
plt.plot(np.arange(0,len(y_test),1), y_test, 'b.', markersize=10, label='Actual')
plt.plot(np.arange(0,len(y_test),1), y_pred, 'r-', label='Prediction', alpha =0.5)
plt.xlabel('Obs')
plt.ylabel('SalePrice')
#plt.ylim(-10, 20)
plt.legend(loc='upper right');


# In[ ]:


DFyy = pd.DataFrame({'y_test':y_test,'y_pred': y_pred})
DFyy.sort_values(by=['y_test'],inplace=True)
plt.plot(np.arange(0,len(DFyy),1), DFyy['y_pred'])
plt.plot(np.arange(0,len(DFyy),1), DFyy['y_test'], alpha=0.5)
#plt.ylim(0,500000)
plt.ylabel('Red= y_test,  Blue = y_pred')
plt.xlabel('Index ')
plt.title('Predicted vs. Real');
print('Observations sorted by y_test values, i.e. higher index => higher SalePrice value');


# In[ ]:


plt.plot(np.arange(0,len(DFyy),1), DFyy['y_pred']/DFyy['y_test'])
plt.ylabel('Ratio = pred/real')
plt.xlabel('Index')
plt.title('Ratio of Predicted vs. Real (1=excellent Prediction)');
print('Observations sorted by y_test values, i.e. higher index => higher SalePrice value');


# In[ ]:


plt.scatter(y_test, y_pred)
plt.ylim(-1, 500000)
plt.xlim(-1, 500000)
plt.plot(y_test, y_test, "r")
plt.xlabel('y_actual')
plt.ylabel('y_predicted');


# See how the relative residuals are distributed across testing sample

# In[ ]:


plt.scatter(np.arange(0,len(DFyy),1), (DFyy['y_test'] - DFyy['y_pred'])/DFyy['y_test'] )
plt.ylim(-0.75,0.75)
plt.ylabel('Relative Error = (real - pred)/real')
plt.xlabel('Index')
plt.title('Relative Error in Testing sample');
print('Observations sorted by y_test values, i.e. higher index => higher SalePrice value');


# Define RMSLE metrics for model evaluation and comparison

# In[ ]:


# A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(real, predicted):
    sum=0.0
    for x in range(len(predicted)):
        if predicted[x]<0 or real[x]<0: #check for negative
            print('Warning:1 negative value skipped')
            continue
        p = np.log(predicted[x]+1)
        r = np.log(real[x]+1)
        sum = sum + (p - r)**2
    return (sum/len(predicted))**0.5


# In[ ]:


print('Prediction accuracy on Testing Sample:')
print('RMSLE       = %f' % (rmsle(y_test, y_pred)))
print('r-squared   = %f' % (r2_score(y_test, y_pred)))


# In[ ]:


pred_train = res.predict(principalDf)
y_pred_train = sc_Y.inverse_transform(pred_train)


# In[ ]:


print('Model accuracy on Training Sample:')
print('RMSLE       = %f' % (rmsle(y_train, y_pred_train)))
print('r-squared   = %f' % (r2_score(y_train, y_pred_train)))


# ### 1.6. Model Stability check - using kFolds

# In[ ]:


skf = KFold(n_splits=3, shuffle=True, random_state=123)

i = 1
Intercept_CV   = list()
PrincComp_1_CV = list()
PrincComp_2_CV = list()
PrincComp_3_CV = list()
PrincComp_4_CV = list()
PrincComp_5_CV = list()

for train_index, test_index in skf.split(X,y):
    #------ Standardize -------
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X[train_index])
    X_test = sc_X.transform(X[test_index])
    sc_Y = StandardScaler()
    y_trainS = sc_Y.fit_transform(y[train_index].reshape(-1,1))
    y_test = y[test_index]
    #----- PCA ----------------
    pca = PCA(n_components = 5)
    principalComponents = pca.fit_transform(X_train)
    principalComponentsTest = pca.transform(X_test)
    
    #---- Applying the Standard scaler so the parameters are easier to compare
    ScPCA = StandardScaler()
    principalComponents = ScPCA.fit_transform(principalComponents)
    principalComponentsTest = ScPCA.transform(principalComponentsTest)
    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['PrincComp_1', 'PrincComp_2','PrincComp_3','PrincComp_4','PrincComp_5'])
    principalDftest = pd.DataFrame(data = principalComponentsTest
             , columns = ['PrincComp_1', 'PrincComp_2','PrincComp_3','PrincComp_4','PrincComp_5'])
    principalDf['SalePrice'] = y_trainS
    
    #principalDf = StandardScaler().fit_transform(principalDf)
    
    #----- QUANTILE REGRESSION ----------
    mod = sm.quantreg('SalePrice ~ PrincComp_1 + PrincComp_2 + PrincComp_3 + PrincComp_4 + PrincComp_5', principalDf)
    res = mod.fit(q=.5)
    pred = res.predict(principalDftest) # make the predictions by the model
    y_pred = sc_Y.inverse_transform(pred)
    print('Split No.: ' + str(i))
    print('RMSLE       = %f' % (rmsle(y_test, y_pred)))
    print('r-squared   = %f' % (r2_score(y_test, y_pred)))
    
    Intercept_CV.append(res.params['Intercept'])
    PrincComp_1_CV.append(res.params['PrincComp_1'])
    PrincComp_2_CV.append(res.params['PrincComp_2'])
    PrincComp_3_CV.append(res.params['PrincComp_3'])
    PrincComp_4_CV.append(res.params['PrincComp_4'])
    PrincComp_5_CV.append(res.params['PrincComp_5'])
    i=i+1

#--- Model Stability Check: Mean and standard deviation of each regression parameter
print('Intercept mean = ' + str(np.mean(Intercept_CV)) + ' std = ' + str(np.std(Intercept_CV)))
print('PrincComp_1 mean = ' + str(np.mean(PrincComp_1_CV)) + ' std = ' + str(np.std(PrincComp_1_CV)))
print('PrincComp_2 mean = ' + str(np.mean(PrincComp_2_CV)) + ' std = ' + str(np.std(PrincComp_2_CV)))
print('PrincComp_3 mean = ' + str(np.mean(PrincComp_3_CV)) + ' std = ' + str(np.std(PrincComp_3_CV)))
print('PrincComp_4 mean = ' + str(np.mean(PrincComp_4_CV)) + ' std = ' + str(np.std(PrincComp_4_CV)))
print('PrincComp_5 mean = ' + str(np.mean(PrincComp_5_CV)) + ' std = ' + str(np.std(PrincComp_5_CV)))


# ## 2. Gradient Boosted Regression
# Compare the outputs of quantile regression accuracy to Gradient boosted regression outputs

# In[ ]:


# Separating out the features (and Training sample from Testing)
X = df.iloc[:1460, :-2].values

# Separating out the target
y = df.iloc[:1460,-2].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[ ]:


sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[ ]:


GBRmedian = GradientBoostingRegressor(loss='quantile', alpha=0.5,
                                n_estimators=250, max_depth=5,
                                learning_rate=.1, min_samples_leaf=10,
                                min_samples_split=20)
GBRmedian.fit(X_train, y_train);


# Let's fit the Gradient boosted regression tree based on quantile =0.5 (LAD) loss function (i.e. estimating the median value)

# In[ ]:


# Make the prediction on the Testing sample
y_pred = GBRmedian.predict(X_test)
y_pred_train = GBRmedian.predict(X_train)


# In[ ]:


print('Model Accuracy on Training sample:')
print('RMSLE       = %f' % (rmsle(y_train, y_pred_train)))
print('r-squared   = %f' % (r2_score(y_train, y_pred_train)))


# In[ ]:


print('Accuracy of prediction on Testing sample:')
print('RMSLE       = %f' % (rmsle(y_test, y_pred)))
print('r-squared   = %f' % (r2_score(y_test, y_pred)))


# In[ ]:


# Plot feature importance
feature_importance = GBRmedian.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
# Let's plot to top10 most important variables
sorted_idx = sorted_idx[-10:]
pos = np.arange(sorted_idx.shape[0]) + .5
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, df.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance');


# In[ ]:


# Plot the y_test and the prediction (y_pred)
fig = plt.figure(figsize=(15, 5))
plt.plot(np.arange(0,len(y_test),1), y_test, 'b.', markersize=10, label='Actual')
plt.plot(np.arange(0,len(y_test),1), y_pred, 'r-', label='Prediction', alpha = 0.5)
plt.xlabel('Obs')
plt.ylabel('SalePrice')
#plt.ylim(-10, 20)
plt.legend(loc='upper right');


# In[ ]:


DFyy = pd.DataFrame({'y_test':y_test,'y_pred': y_pred})
DFyy.sort_values(by=['y_test'],inplace=True)
plt.plot(np.arange(0,len(DFyy),1), DFyy['y_pred'])
plt.plot(np.arange(0,len(DFyy),1), DFyy['y_test'], alpha=0.5)
#plt.ylim(0,500000)
plt.ylabel('Red= y_test,  Blue = y_pred')
plt.xlabel('Index ')
plt.title('Predicted vs. Real');
print('Observations sorted by y_test values, i.e. higher index => higher SalePrice value');


# In[ ]:


plt.plot(np.arange(0,len(DFyy),1), DFyy['y_pred']/DFyy['y_test'])
plt.ylabel('Ratio = pred/real')
plt.xlabel('Index')
plt.ylim(0,2)
plt.title('Ratio Predicted vs. Real (1=excellent Prediction)');
print('Observations sorted by y_test values, i.e. higher index => higher SalePrice value');


# In[ ]:


plt.scatter(y_test, y_pred)
plt.ylim(-1, 500000)
plt.xlim(-1, 500000)
plt.plot(y_test, y_test, "r")
plt.xlabel('y_actual')
plt.ylabel('y_predicted');


# See how the residuals are distributed across testing sample

# In[ ]:


plt.scatter(np.arange(0,len(DFyy),1), (DFyy['y_test'] - DFyy['y_pred'])/DFyy['y_test'] )
plt.ylim(-0.75,0.75)
plt.ylabel('Relative Error = (real - pred)/real')
plt.xlabel('Index')
plt.title('Ratio Predicted vs. Real (1=perfectPrediction)');
print('Observations sorted by y_test values, i.e. higher index => higher SalePrice value');


# > When we compare prediction error in the tails, gradient boosting seems to give better accuracy for lowest and highest value houses.

# ### GBR Model Stability

# In[ ]:


skf = KFold(n_splits=3, shuffle=True, random_state=123)

for train_index, test_index in skf.split(X,y):
    #------ Standardize -------
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X[train_index])
    X_test = sc_X.transform(X[test_index])
    y_train = y[train_index]
    y_test = y[test_index]
    
    #----- Gradient Boosted Regression ----------
    GBRmedian = GradientBoostingRegressor(loss='quantile', alpha=0.5,
                                    n_estimators=250, max_depth=5,
                                    learning_rate=.1, min_samples_leaf=10,
                                    min_samples_split=20)
    GBRmedian.fit(X_train, y_train)
    # Make the prediction on the Testing sample
    y_pred = GBRmedian.predict(X_test)
    y_pred_train = GBRmedian.predict(X_train)

    print('Accuracy of prediction on Testing sample:')
    print('RMSLE       = %f' % (rmsle(y_test, y_pred)))
    print('r-squared   = %f' % (r2_score(y_test, y_pred)))
    
    #---------------------------
    # Plot feature importance
    feature_importance = GBRmedian.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    # Let's plot to top10 most important variables
    sorted_idx = sorted_idx[-10:]
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, df.columns[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show();


# ### Conclusion
# Gradient Boosted Regression gives better accuracy compared to quantile regression, though the difference is not that big between RMSLEs (0.13 vs. 0.17). In the evaluation charts can see and compare how well the models predict across the testing sample.
# If we compare model stability outputs - predictor variable importance order is changing in every fold, where in quantile regression equation slope coefficients (of principal components) doesn't change much, i.e. the importance of predictors (and contribution to the model) is stable. This is one of the reasons why some business sectors still choose to use traditional methods over advanced machine learning.

# ### References
# 
# - Great tips on Exploratory Data Analysis by [Pedro Marcelino](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python) and [Tuatini Godard](https://www.kaggle.com/ekami66/detailed-exploratory-data-analysis-with-python), Categorical Data Analysis by [Samarth Agrawal](https://www.kaggle.com/nextbigwhat/eda-for-categorical-variables-a-beginner-s-way) 
# - Principal Component Analysis https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60  
# - Quantile Regression http://www.statsmodels.org/dev/generated/statsmodels.regression.quantile_regression.QuantReg.html  
