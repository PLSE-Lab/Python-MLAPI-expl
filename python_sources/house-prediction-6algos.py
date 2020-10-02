#!/usr/bin/env python
# coding: utf-8

# # 6 Algos used are:-
# 1. RandomForest
# 2. SVM
# 3. ANN
# 4. AdaBoost
# 5. Gradient Boost
# 6. XgBoost 
# 
# Ensemble technique added
# 
# If you like it , please upvote & fork it. Keep learning , Keep Sharing:-)
# (Wait for more intresting insights like ensemble techniques, proper functions.)
# 

# In[ ]:


import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df_train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df_train


# In[ ]:


df_test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
df_test


# In[ ]:


# Copying original dataset.
df=df_train


# In[ ]:


df.info()


# In[ ]:


df.columns[df.isnull().any()] # Missing values. Column with true values are printed.


# In[ ]:


a=df.columns[df.isnull().any()]
a


# In[ ]:


a=list(a)
a #easy to work with lists


# In[ ]:


r,c=df.shape
r,c


# In[ ]:


missing_df=df[a]
missing_df


# In[ ]:


r1,c1=missing_df.shape
r1,c1


# In[ ]:


missing_df.isna()


# In[ ]:


missing_df.isna().sum()*100/r1
# miss_per_df.sum()*100/r1


# In[ ]:


def percent_missing(df):
    data = pd.DataFrame(df)
    df_cols = list(pd.DataFrame(data)) #List of Columns of dataframe named 'data'
    dict_x = {}
    for i in range(0, len(df_cols)):
        dict_x.update({df_cols[i]: round(data[df_cols[i]].isnull().mean()*100,2)}) #Make dictionary of missing columns as key and percentage (upto 2 decimal places rounded up ) as value
    
    return dict_x


# In[ ]:


missing = percent_missing(missing_df)
df_miss = sorted(missing.items(), key=lambda x: x[1], reverse=True)# lambda x:x[1] i.e. sorting acc. to second parameter i.e. percentage of missing values
print('Percent of missing data')
df_miss


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


sns.set_style("white")
f, ax = plt.subplots(figsize=(8, 7))
sns.set_color_codes(palette='deep') #Color of bar graph.
missing = round(missing_df.isnull().mean()*100,2)
# missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar(color="b")

# Tweak the visual presentation
ax.grid(True)
ax.set(ylabel="Percent of missing values")
ax.set(xlabel="Features")
ax.set(title="Percent missing data by feature")
sns.despine(trim=True, left=True)# Outer borders


# ## Dropping columns with missing values >= 20%

# In[ ]:


# Dropping columns with >=20% missing data
# missing_data_20=missing_df.isna().sum()*100/r1>=20
# missing_data_20


# In[ ]:


# missing_data_20.values


# In[ ]:


dropping=[]
# for i in range(c1):
#     if missing_data_20.values[i]==True:
#         dropping.append(missing_data_20.index[i])
dropping=['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']
dropping


# In[ ]:


df_updated=df.drop(columns=dropping,axis=1)
df_updated


# In[ ]:


#test data is copied
c_test=df_test


# In[ ]:


c_test=c_test.drop(columns=dropping,axis=1)
c_test


# ## Filling missing values.

# In[ ]:


y=df_updated.iloc[:,[-1]]
y.head()


# In[ ]:


x=df_updated.iloc[:,:-1]
x.head()


# In[ ]:


# categorical_col=list(x.columns)
# len(categorical_col)


# In[ ]:


# s='LotFrontage LotArea MasVnrArea BsmtFinSF1 BsmtFinSF2 BsmtUnfSF TotalBsmtSF 1stFlrSF 2ndFlrSF LowQualFinSF GrLivArea GarageYrBlt GarageCars GarageArea WoodDeckSF OpenPorchSF EnclosedPorch 3SsnPorch ScreenPorch PoolArea MiscVal MoSold YrSold'
# non_categorical_col=s.split()


# In[ ]:


len(missing_df.columns)


# In[ ]:


missing_rest=list(missing_df.columns)
(missing_rest)


# In[ ]:


dropping


# In[ ]:


missing_rest=list(set(missing_rest).difference(dropping))
missing_rest


# In[ ]:


missing_rest=df.loc[:, missing_rest] 
missing_rest


# In[ ]:


missing_rest.describe()


# In[ ]:


# mean and mode is same for GarageYrBlt and LotFRONTage. SO filling them by mean. THe data seems normally distributed


# mean and mode is same for GarageYrBlt and LotFRONTage. SO filling them by mean. THe data seems normally distributed

# In[ ]:


b=pd.DataFrame(missing_rest['MasVnrArea'])  #External layer of house              
b.plot(kind='hist')
# Hence filling this volumn by zero.Better to drop it.No meaning of 0 sq. feet


# In[ ]:


b['MasVnrArea'].value_counts()


# In[ ]:


missing_rest['BsmtFinType2'].value_counts()
# 1256/1460= 86%


# In[ ]:


missing_rest['BsmtQual'].value_counts()
# 1423


# In[ ]:


missing_rest['BsmtCond'].value_counts()
# 1311/1460 =


# In[ ]:


missing_rest['GarageType'].value_counts()
# 1379


# In[ ]:


missing_rest['BsmtFinType1'].value_counts()
# 1423


# In[ ]:


missing_rest['GarageQual'].value_counts()
# 1311/1460=


# In[ ]:


missing_rest['GarageFinish'].value_counts()
# 1379


# In[ ]:


missing_rest['MasVnrType'].value_counts()
# 1452


# In[ ]:


missing_rest['BsmtExposure'].value_counts()
# 1422


# In[ ]:


missing_rest['GarageCond'].value_counts()
# 1326/1460= 


# ## Imputer

# In[ ]:


from sklearn.impute import SimpleImputer


# ## Fitting for test data also

# In[ ]:


c_test1=c_test
c_test1


# In[ ]:





# In[ ]:


sns.set_style("white")
f, ax = plt.subplots(figsize=(8, 7))
sns.set_color_codes(palette='deep') #Color of bar graph.
missing = round(missing_rest.isnull().mean()*100,2)
# missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar(color="b")

# Tweak the visual presentation
ax.grid(True)
ax.set(ylabel="Percent of missing values")
ax.set(xlabel="Features")
ax.set(title="Percent missing data by feature")
sns.despine(trim=True, left=True)# Outer borders


# In[ ]:


(df[df['GarageArea']==0]).index
(df[df['GarageCars']==0]).index
(df[df['GarageYrBlt'].isnull()]).index
(df[df['GarageQual'].isnull()]).index
(df[df['GarageCond'].isnull()]).index
(df[df['GarageFinish'].isnull()]).index

(df[df['GarageYrBlt'].isnull()]).index
# (df[df['BsmtFinSF1'].isnull()]).index

# df.loc[[17],["BsmtFinSF1"]]


# ### Same indices are missing.So it means that if area=0, those coln are missing.

# In[ ]:


for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    missing_rest[col] = missing_rest[col].fillna('NA')


# In[ ]:


for col in ['GarageYrBlt']:
    missing_rest[col] = missing_rest[col].fillna(0)    


# In[ ]:


(df[df['BsmtQual'].isnull()]).index
(df[df['BsmtCond'].isnull()]).index
(df[df['BsmtExposure'].isnull()]).index
(df[df['BsmtFinType1'].isnull()]).index
(df[df['BsmtFinType2'].isnull()]).index

df.loc[[39],["BsmtFinSF1"]]
# df['B']


# In[ ]:


for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    missing_rest[col] = missing_rest[col].fillna('NA') 


# In[ ]:


df['Neighborhood'].unique()


# In[ ]:


see=df.groupby('Neighborhood')['LotFrontage']
see.get_group('Blmngtn')


# In[ ]:


missing_rest['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))


# In[ ]:


(df[df['MasVnrArea'].isnull()]).index
df['MasVnrType'].value_counts()


# In[ ]:


seeyou=df.groupby('MasVnrType')['MasVnrArea']
seeyou.get_group('None').sort_values(ascending=False)


# In[ ]:


# for col in ('MasVnrType')
missing_rest['MasVnrType'] = missing_rest['MasVnrType'].fillna('None') 
missing_rest['MasVnrArea'] = missing_rest['MasVnrArea'].fillna(0) 


# ## Test data

# In[ ]:


for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    c_test1[col] = c_test1[col].fillna('NA')

for col in ['GarageYrBlt']:
    c_test1[col] = c_test1[col].fillna(0)    

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    c_test1[col] = c_test1[col].fillna('NA')
    
c_test1['MasVnrType'] = c_test1['MasVnrType'].fillna('None') 
c_test1['MasVnrArea'] = c_test1['MasVnrArea'].fillna(0) 

# c_test1['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))


# In[ ]:


c_test=c_test1


# In[ ]:


mode_imputer=SimpleImputer(missing_values=np.nan, strategy='most_frequent')


# In[ ]:


missing_rest['Electrical'].value_counts()


# In[ ]:


a1=missing_rest.loc[:,['Electrical']]
a1.info()


# In[ ]:


mode_imputer.fit(a1)
a11 = mode_imputer.transform(a1)


# ## Test data

# In[ ]:


c_test1['Electrical'].value_counts()


# In[ ]:


c_test1.loc[:,['Electrical']].info()


# In[ ]:


# c_test1.loc[:,['Electrical']]=mode_imputer.transform(c_test1.loc[:,['Electrical']])


# In[ ]:


c_test=c_test1


# In[ ]:


(missing_rest[missing_rest['Electrical'].isnull()]).index
missing_rest['Electrical'][1379]='SBrkr'


# In[ ]:


missing_rest.info()


# In[ ]:


# c_test1.loc[:,['Electrical']].info()


# ## As of now simply encoding. Later try giving more rank to excellent  Then see o/p. Similarly assign less rank to excell.

# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


zz=x
zz


# In[ ]:


remove='GarageType BsmtExposure GarageQual BsmtCond GarageYrBlt MasVnrArea GarageFinish BsmtFinType2 GarageCond Electrical BsmtQual MasVnrType LotFrontage BsmtFinType1'


# In[ ]:


remove=remove.split()
remove


# In[ ]:


x=x.drop(columns=remove)
x = pd.concat([x, missing_rest], axis=1, sort=False)


# In[ ]:


x


# # Some Missing values in test data

# In[ ]:


c_test1_missing_values=c_test1.columns[c_test1.isnull().any()] # Missing values in test data.
c_test1_missing_values=list(c_test1_missing_values)
c_test1_missing_values


# In[ ]:


df_test_c_missing=c_test1.loc[:,c_test1_missing_values]
df_test_c_missing


# In[ ]:


c_test1.loc[:,c_test1_missing_values].info()


# In[ ]:


c_missing=SimpleImputer(missing_values=np.nan, strategy='most_frequent')

c_missing.fit(x.loc[:,c_test1_missing_values])
c_test1.loc[:,c_test1_missing_values] = c_missing.transform(c_test1.loc[:,c_test1_missing_values])


# In[ ]:


c_test1.columns[c_test1.isnull().any()]


# In[ ]:


c_test1.loc[:,c_test1_missing_values].info()


# In[ ]:


#For categorical columns:-


# In[ ]:


chalo=c_test1.loc[:,['Exterior2nd','Utilities','Exterior1st',
 'SaleType',
 'KitchenQual',
 'Functional',
 'MSZoning']]
chalo


# In[ ]:


set(chalo.columns)


# In[ ]:


c_test1['Foundation'].unique()
# 16, 10 ,5, 5, 25, 8, 6, 6, 9


# In[ ]:


# count=0
# for i in range(0,1459):
#     if type(chalo['Utilities'][i])==str:
#         continue
# #         print('hi')
#     else:
#          print(i,(chalo['Utilities'][i]),(chalo['Utilities'][i-1]))
# #          count+=1
# # print(count)
#    


# In[ ]:


# Exterior 2nd --691---13.0
chalo.Exterior2nd.value_counts()
chalo['Exterior2nd'][691]='VinylSd'
chalo['Exterior2nd'][691]


# In[ ]:


# Utilities
# 455 0.0 AllPub
# 485 0.0 AllPub
chalo.Utilities.value_counts()
chalo['Utilities'][455]='AllPub'
chalo['Utilities'][485]='AllPub'

chalo['Utilities'][455], chalo['Utilities'][485]


# In[ ]:


# Exterior1st
# 691 12.0 BrkFace
chalo.Exterior1st.value_counts()
chalo['Exterior1st'][691]='VinylSd'
chalo['Exterior1st'][691]


# In[ ]:


# SaleType 1029 noo (8.0, 'WD')
chalo.SaleType.value_counts()
chalo['SaleType'][1029]='WD'
chalo['SaleType'][1029]


# In[ ]:


# KitchenQual 95 noo (3.0, 'TA')
chalo.KitchenQual.value_counts()
chalo['KitchenQual'][95]='TA'
chalo['KitchenQual'][95]


# In[ ]:


# # Functional
# 756 noo (6.0, 'Typ')
# 1013 noo (6.0, 'Typ')
chalo.Functional.value_counts()
chalo['Functional'][756]='Typ'
chalo['Functional'][1013]='Typ'
chalo['Functional'][756],chalo['Functional'][1013]


# In[ ]:


# 
# 455 noo (3.0, 'RM')
# 756 noo (3.0, 'RM')
# 790 noo (3.0, 'RL')
# 1444 noo (3.0, 'RL')

chalo.MSZoning.value_counts()
chalo['MSZoning'][756]='RL'
chalo['MSZoning'][455]='RL'
chalo['MSZoning'][790]='RL'
chalo['MSZoning'][1444]='RL'

chalo['MSZoning'][756],chalo['MSZoning'][455],chalo['MSZoning'][790],chalo['MSZoning'][1444]


# In[ ]:


chalo_columns=[]


# In[ ]:


for i in chalo.columns:
    chalo_columns.append(i)
chalo_columns    


# # Train data of these columns used for label encoding

# In[ ]:


x.loc[:,chalo_columns[0]]


# In[ ]:


x_copy=x.loc[:,chalo_columns]
x_copy


# In[ ]:


chalo


# In[ ]:


chalo[chalo_columns[0]]


# In[ ]:


def chalo_encoder(i):
    l=LabelEncoder()
    
    l.fit(x.loc[:,chalo_columns[i]])
#     x.loc[:,chalo_columns[i]]=l.transform(x.loc[:,chalo_columns[i]])
    chalo[chalo_columns[i]]=l.transform(chalo[chalo_columns[i]])


# ## Label Encoding these

# In[ ]:


c_test1=c_test1.drop(columns=chalo.columns)


# In[ ]:


c_test1 = pd.concat([c_test1, chalo], axis=1, sort=False)
c_test1


# In[ ]:


c_test=c_test1


# ## Now label encoding

# In[ ]:


num_cols = x._get_numeric_data().columns
num_cols,len(num_cols)


# In[ ]:


categorical=list(set(x.columns) - set(num_cols))
categorical,len(categorical)


# In[ ]:


zz=x
zz


# In[ ]:



num_cols = zz._get_numeric_data().columns
len(num_cols)


# ## For test data

# In[ ]:


num_co = c_test1._get_numeric_data().columns
num_co


# In[ ]:


categorical_test=list(set(c_test1.columns) - set(num_co))
categorical_test,len(categorical_test)


# In[ ]:


num_c = c_test1._get_numeric_data().columns
len(num_c)


# In[ ]:


num_cols = zz._get_numeric_data().columns
len(num_cols)
# BEFORE


# In[ ]:


num_cols = c_test1._get_numeric_data().columns
len(num_cols)
# BEFORE


# # Label Encoder of Train and test:-
# 

# In[ ]:


def lebel_enc(yoo):
    l1=LabelEncoder()
    
    l1.fit_transform(zz[yoo])
    zz[yoo]=l1.transform(zz[yoo])
    c_test1[yoo]=l1.transform(c_test1[yoo])


# In[ ]:


for i in categorical:
    lebel_enc(i)


# In[ ]:


num_cols = zz._get_numeric_data().columns
len(num_cols)
# AFter


# In[ ]:


num_cols = c_test1._get_numeric_data().columns
len(num_cols)
# AFter


# In[ ]:


x


# In[ ]:





# In[ ]:


import seaborn as sns


# <!-- # **<u> 1.Using Univariate Selection</u>** -->

# In[ ]:


# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2


# In[ ]:



# bestfeatures = SelectKBest(score_func=chi2, k=30)
# fit = bestfeatures.fit(x,y)

# dfscores = pd.DataFrame(fit.scores_)
# dfcolumns = pd.DataFrame(x.columns)

# #concat two dataframes for better visualization 
# featureScores = pd.concat([dfcolumns,dfscores],axis=1)
# featureScores.columns = ['Specs','Score']  #naming the dataframe columns
# print(featureScores.nlargest(30,'Score'))  #print 10 best features


# In[ ]:


# hi=featureScores.sort_values(by='Score',ascending=False).iloc[:30,[0]]
# hi


# In[ ]:


# hii_selectBestK=hi['Specs'].to_list()


# In[ ]:


# hii_selectBestK


# In[ ]:


# select_bestK_columns=hii_selectBestK


# In[ ]:


# x_select_bestk=x.loc[:,select_bestK_columns]


# In[ ]:


# x_select_bestk


# In[ ]:


# c_test_select_bestk=c_test.loc[:,select_bestK_columns]


# In[ ]:


# c_test_select_bestk


# In[ ]:





# # **<u> Using Heatmap</u>**

# In[ ]:


temp_df=x.iloc[:,0:40]
temp_df=pd.concat([temp_df,y], axis=1, sort=False)
temp_df


# In[ ]:


# Plotting 1st 40 columns 


# In[ ]:


plt.figure(figsize = (70,70))
corr1 = temp_df.corr()

ax = sns.heatmap(
    corr1, 
    vmin=-1, vmax=1,
    cmap=sns.diverging_palette(20, 220, n=200),linewidth=2
    
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right',size=40
);

ax.set_yticklabels( ax.get_yticklabels(),size=(40), rotation=45,);


# In[ ]:


corr_heatmap1=corr1.loc[:,['SalePrice']].sort_values(by='SalePrice',ascending=False)*100 
corr_heatmap1=corr_heatmap1[corr_heatmap1['SalePrice']>10]
corr_heatmap1


# In[ ]:


corr_heatmap1.shape


# In[ ]:


corr1_columns=(corr_heatmap1.index.tolist()[1:])
len(corr1_columns)


# In[ ]:


# Plotting 41 till last columns
temp_df=x.iloc[:,41:]
temp_df=pd.concat([temp_df,y], axis=1, sort=False)
temp_df


# In[ ]:


plt.figure(figsize = (70,70))
corr2 = temp_df.corr()

ax = sns.heatmap(
    corr2, 
    vmin=-1, vmax=1,
    cmap=sns.diverging_palette(20, 220, n=200),linewidth=2
    
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right',size=40
);

ax.set_yticklabels( ax.get_yticklabels(),size=(40), rotation=45,);


# In[ ]:


corr_heatmap2=corr2.loc[:,['SalePrice']].sort_values(by='SalePrice',ascending=False)*100 
corr_heatmap2=corr_heatmap2[corr_heatmap2['SalePrice']>10]
corr_heatmap2


# In[ ]:


len(corr_heatmap2.index)


# In[ ]:


corr2_columns=(corr_heatmap2.index.tolist()[1:])
len(corr2_columns)


# In[ ]:


columns_from_heatmap=corr1_columns+corr2_columns
columns_from_heatmap,len(columns_from_heatmap)


# In[ ]:


x_heatmap=x.loc[:,columns_from_heatmap]
# x1.insert(0, "Id", x.iloc[:,0:1], True) 
x_heatmap


# In[ ]:



c_test_heatmap=c_test.loc[:,columns_from_heatmap]
# c1_test.insert(0, "Id", c_test.iloc[:,0:1], True) 
c_test_heatmap


# In[ ]:


columns_from_heatmap


# # **<u>3.Feature Importance</u>**

# In[ ]:


# X = x  #independent columns

# from sklearn.ensemble import ExtraTreesClassifier
# import matplotlib.pyplot as plt

# model = ExtraTreesClassifier()
# model.fit(X,y)

# print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
# #plot graph of feature importances for better visualization

# feat_importances = pd.Series(model.feature_importances_, index=X.columns)
# feat_importances.nlargest(10).plot(kind='barh')
# plt.show()


# In[ ]:


# data = {'Specs':X.columns,
#         'Score':model.feature_importances_}


# In[ ]:


# data


# In[ ]:


# feat_importance1 = pd.DataFrame(data)
# feat_importance1


# In[ ]:


# hi=feat_importance1.sort_values(by='Score',ascending=False).iloc[1:31,[0]]
# Taking from 1 to 31 as 1st one is 'Id' column. 
# hi


# In[ ]:


# hii=hi['Specs'].to_list()
# feature_importance_columns=hii

# x_feature_importance=x.loc[:,feature_importance_columns]


# In[ ]:


# x_feature_importance


# In[ ]:


# c_test_feature_importance=c_test.loc[:,feature_importance_columns]
# c_test_feature_importance


# In[ ]:


# feature_importance_columns


# In[ ]:


# s1=set(hii_selectBestK)
# s2=set(columns_from_heatmap)
# s3=set(feature_importance_columns)


# In[ ]:


# set1=s1.intersection(s1)
# len(set1)


# In[ ]:


# set2=s3.intersection(set1)
# len(set2)


# # One error according to me is label encoding too many columns.

# # Train,test,split:-

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


xtrain,xtest,ytrain,ytest=train_test_split(x_heatmap,y,test_size=0.3,random_state=0)
# x1 for heatmap
# x_select_bestk for selectbest k
# x_feature_importance for feature importance


# In[ ]:


ytest


# In[ ]:





# # **<u>1.Random Forests</u>**

# In[ ]:





# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# ### Q)when to use only fit and fit_transform? Only fit used below?

# In[ ]:


reg=RandomForestRegressor(n_estimators=2000,random_state=0,max_depth=15)
reg.fit(xtrain,ytrain)


# In[ ]:


randomForest_predict=reg.predict(xtest)


# In[ ]:


randomForest_predict


# In[ ]:





# <!-- ## K-fold cross validation -->

# <!-- ### Q)can k-fold cross validation be applied to regression -->

# In[ ]:


# from sklearn.model_selection import cross_val_score/


# In[ ]:


# accuracies=cross_val_score(estimator=reg,X=xtrain,y=ytrain,cv=10)#Vector having accuracy computed by combinations of k models.Usually k=10 is taken


# In[ ]:


# accuracies.mean()*100


# In[ ]:


# accuracies.std()*100


# In[ ]:


# print(accuracies.mean()*100-accuracies.std()*100,
#       accuracies.mean()*100+accuracies.std()*100)


# <!-- ## Grid Search -->

# <!-- ### Q)grid search for random forest regression/ types of hyperparameter tuning -->

# <!-- #### Types of scoring in grid? -->

# In[ ]:


# (bootstrap=True, ccp_alpha=0.0, criterion='mse',
#                       max_depth=None, max_features='auto', max_leaf_nodes=None,
#                       max_samples=None, min_impurity_decrease=0.0,
#                       min_impurity_split=None, min_samples_leaf=1,
#                       min_samples_split=2, min_weight_fraction_leaf=0.0,
#                       n_estimators=1000, n_jobs=None, oob_score=False,
#                       random_state=0, verbose=0, warm_start=False)


# In[ ]:


# from sklearn.model_selection import GridSearchCV


# In[ ]:


# parameters = [{'n_estimators': [1, 10, 100, 1000,2000], 'max_depth': [10,15,50,100]}]


# In[ ]:



# grid_search = GridSearchCV(estimator = reg,
#                            param_grid = parameters,
#                            scoring='neg_mean_squared_error',
#                            cv = 10,
#                            n_jobs = -1)


# In[ ]:



# grid_search = grid_search.fit(xtrain, ytrain)


# In[ ]:



# best_accuracy = grid_search.best_score_
# best_parameters = grid_search.best_params_


# In[ ]:


# best_parameters


# In[ ]:





# # **<u>2.SVM</u>**

# ### Before applying SVR, Scaling needs to be done......
# <!-- ## Q) Various types of scalar are there. See which is best? -->

# In[ ]:


from sklearn.preprocessing import StandardScaler as SS


# In[ ]:


sc_x=SS()
sc_y=SS()


# In[ ]:


xtrain.iloc[:,:]


# In[ ]:


ytrain


# In[ ]:


xtrain_SVR=sc_x.fit_transform(xtrain)
xtest_SVR=sc_x.transform(xtest)


# In[ ]:


ytrain_SVR=sc_y.fit_transform(ytrain)
ytest_SVR=sc_y.transform(ytest)


# In[ ]:


from sklearn.svm import SVR
SVR_regressor=SVR(kernel='rbf') #Usually experiment with rbf 1st
SVR_regressor.fit(xtrain_SVR,ytrain_SVR)


# In[ ]:


SVR_regressor.predict(xtest_SVR)


# In[ ]:


SVR_xtest_prediction=sc_y.inverse_transform(SVR_regressor.predict(xtest_SVR))


# In[ ]:


SVR_xtest_prediction


# In[ ]:


SVR_xtrain_prediction=sc_y.inverse_transform(SVR_regressor.predict(xtrain_SVR))


# In[ ]:


SVR_xtrain_prediction


# In[ ]:


ytest


# In[ ]:


# # 1. Accuracy of SVR-----submit only svr pred too
# 2. Which one is best
# 3. Viz both w.r.t. to real answer
# 4. Ensemble of both
# 5. see what weights is good
# 6. eval matrix


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.scatter(xtrain.index[0:1000],ytrain[0:1000])
plt.scatter(xtrain.index[0:1000],SVR_xtrain_prediction[0:1000],color='red')
# plt.scatter(xtrain.index[0:1000],reg.predict(xtrain)[0:1000],color='y')

# ytest.plot(xtest.index,kind='scatter')


# In[ ]:


plt.scatter(xtrain.index[0:1000],ytrain[0:1000])
# plt.scatter(xtrain.index[0:100],SVR_xtrain_prediction[0:100],color='red')
plt.scatter(xtrain.index[0:1000],reg.predict(xtrain)[0:1000],color='yellow')


# In[ ]:


plt.scatter(xtrain.index[0:1000],SVR_xtrain_prediction[0:1000],color='red')
plt.scatter(xtrain.index[0:1000],reg.predict(xtrain)[0:1000],color='yellow')


# ### From viz. it seems svr performs better only for lower values. RF for both.
# If value >350000 then use rf. See error for value less than this

# # **<u>3. ANN</u>**

# In[ ]:


xtrain_ann,xtest_ann,ytrain_ann,ytest_ann=train_test_split(x_heatmap,df.iloc[:,-1],test_size=0.3,random_state=0)


# In[ ]:


# xtrain_ann=xtrain_ann.drop(columns='Id')
# xtrain_ann

# xtest_ann=xtest_ann.drop(columns='Id')
# xtest


# In[ ]:


import tensorflow as tf


# In[ ]:


tf.__version__


# In[ ]:


ann=tf.keras.models.Sequential() #Initialise the layer


# In[ ]:


ann.add(tf.keras.layers.Dense(units=69,activation='relu')) #1st Layer


# In[ ]:


ann.add(tf.keras.layers.Dense(units=69,activation='relu')) #2nd layer


# In[ ]:


ann.add(tf.keras.layers.Dense(units=1)) #O/P layer----No activation function for last layer of regression


# In[ ]:


ann.compile(optimizer='adam',loss='mean_squared_error')


# In[ ]:


ann.fit(xtrain_ann.values,ytrain_ann.values, batch_size=32, epochs=300)


# In[ ]:


ypred_ann=ann.predict(xtest_ann)


# In[ ]:


ypred_ann.reshape(438,)


# # **<u>4. AdaBoost</u>**

# In[ ]:


from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor


# In[ ]:





# In[ ]:


adaboost=AdaBoostRegressor(base_estimator=DecisionTreeRegressor(),n_estimators=1000,learning_rate=0.1,random_state=20)


# In[ ]:


adaboost.fit(xtrain,ytrain)


# In[ ]:


adaboost_xtest_prediction=adaboost.predict(xtest)


# # **<u>5. Gradient Boost</u>**

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor


# In[ ]:


gradient= GradientBoostingRegressor(n_estimators=100)


# In[ ]:


gradient


# In[ ]:



gradient.fit(xtrain, ytrain)


# In[ ]:


gradient_xtest_prediction=gradient.predict(xtest)


# [Hyperparameter tuning of Gradient Boosting](https://medium.com/all-things-ai/in-depth-parameter-tuning-for-gradient-boosting-3363992e9bae)

# # **<u>6. XgBoost</u>**

# In[ ]:


import xgboost as xgb


# In[ ]:


xgboost=xgb.XGBRegressor()


# In[ ]:


xgboost.fit(xtrain,ytrain)


# In[ ]:


xgboost_xtest_prediction=xgboost.predict(xtest)


# # Ensemble- Blending

# In[ ]:


( 0.2* xgboost.predict(xtest)+ 0.5* reg.predict(xtest)+ 0.2* gradient.predict(xtest)+
    0.2* adaboost.predict(xtest)+0.2*ann.predict(xtest).reshape(438,)).shape


# In[ ]:


xtest.shape[0]


# In[ ]:


def all_pred(xtest):
    r1=xtest.shape[0]
    return(
   0.3* xgboost.predict(xtest)+
    0.2* gradient.predict(xtest)+
    0.2* adaboost.predict(xtest)+
    0.2* reg.predict(xtest)+
    0.1*ann.predict(xtest).reshape(r1,))
    


# In[ ]:


blend=all_pred(xtest)


# In[ ]:


blend.shape


# In[ ]:





# # <u>**Teaser:- Hyperparam of XgBoost gave me Best place on leaderboard**

# ### Hyperparam of XgBoost

# # Q)RandmoizedSearchCV vs GridSearchCV

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


params=[{'learning_rate':[0.1,0.5,1],
         'gamma':[0.1,0.5,1],
         'n_estimator':[10,50,100],
         'max_depth':[2,6,8,12],
         'min_child_weight':[1,3,5,7]}]


# In[ ]:


grid_search = GridSearchCV(estimator = xgboost,
                           param_grid = params,
                           scoring='neg_mean_squared_error',
                           cv = 10,
                           n_jobs = -1)


# In[ ]:


grid_search = grid_search.fit(xtrain, ytrain)


# In[ ]:


best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


# In[ ]:


print(best_accuracy,best_parameters)


# In[ ]:


xgboost=xgb.XGBRegressor(gamma= 0.1, learning_rate=0.1, max_depth= 6, min_child_weight= 3, n_estimator= 10)


# In[ ]:


xgboost.fit(xtrain,ytrain)


# In[ ]:


xgboost_xtest_prediction=xgboost.predict(xtest)
xgboost_xtest_prediction.shape


# # Vizualation after performing algos.

# ### Q) Ways to evaluate a model

# In[ ]:


from sklearn.metrics import mean_absolute_error


# In[ ]:


mean_absolute_error(ytest,blend)


# In[ ]:


from sklearn.metrics import mean_squared_error
mean_squared_error(ytest, blend)


# In[ ]:


90,58,03,841
1,46,62,19,706
71,08,80,528
73,49,95,111
2,45,30,67,944
74,86,54,667
71,77,82,987


# # Evaluation:
# 1. All 70 columns after pre-processing:
#     a. n-estimator=1000
#         1. Absolute-ERROR:- 16,989 
#         2. MSE:- 87,41,82,817
#      ## Leaderboard RMSE:-0.47
# 
# # <u> Heatmap:- </u>
# 2. Selecting 47 columns from heatmap using pearson correlation--- (columns with corr>0 taken only)
#     a. n-estimator=1000
#         1. Absolute-ERROR:- 17207 
#         2. MSE:- 87,87,95,851
#       ## Error actually increased on xtest data. But predictions on C_Test Data gave error as 0.1513
# 3. Selecting 47+'Id' columns from heatmap using pearson correlation
#     a. n-estimator=1000
#         1. Absolute-ERROR:- 17350 
#         2. MSE:- 88,88,11,105     
# 4. Selecting 31 columns from heatmap using pearson correlation-- (columns with corr>40 taken only)  
# 
#        a. n-estimator=1000
#             1. Absolute-ERROR:- 17112 
#             2. MSE:- 85,35,55,830 
#         
#         
#          b. n_estimators=10k  
#       
#             1. Absolute-ERROR:-17,102
#             2. MSE:- 84,61,60,610
#    
# 36 columns selected.n_estimators=2k, max_depth=15   
#             1. Absolute-ERROR:-17,115   
#             2. MSE:- 84,25,83,579

# # SVR:
# 1. 36 features are selected
#         1. Absolute-ERROR:- 18,740
#         2. MSE:-1,46,62,19,706
#         Absolute Error decreased , but MSE increased as compared to RandomForest
# 

# # Adaboost + D.T.:
# 1. 36 features are selected
#         1. Absolute-ERROR:- 16,448
#         2. MSE:-71,08,80,528
#         Absolute Error decreased , but MSE increased as compared to RandomFore

# # Testing on test data

# In[ ]:


(xgboost.predict(c_test_heatmap))


# In[ ]:


xgboost_ypred=xgboost.predict(c_test_heatmap).astype(int)


# In[ ]:


all_pred(c_test_heatmap).astype(int)


# In[ ]:


blend_final=all_pred(c_test_heatmap).astype(int)


# ## SVR

# In[ ]:


sc_y.inverse_transform(SVR_regressor.predict(sc_x.transform(c_test_heatmap)))


# In[ ]:


SVR_final_pred=sc_y.inverse_transform(SVR_regressor.predict(sc_x.transform(c_test_heatmap)))


# In[ ]:





# ## ANN

# In[ ]:


ypred_ann_sub=ann.predict(c_test_heatmap)
ypred_ann_sub.shape


# In[ ]:


asd=[]
for i in range(len(ypred_ann_sub)):
    asd.append(ypred_ann_sub[i][0])
# asd=(ypred_ann)


# In[ ]:


asd


# In[ ]:


dfd = pd.DataFrame(asd, columns = ['SalePrice'])
dfd


# In[ ]:


p1=pd.DataFrame({'Id':c_test.Id.values})
p1


# # DF for ANN

# In[ ]:


p1=p1.join(dfd)
p1


# # Submission file

# In[ ]:


p2= pd.DataFrame({'Id':c_test.Id.values,'SalePrice': blend_final})


# In[ ]:


p2


# In[ ]:


p2['SalePrice'].value_counts()


# In[ ]:


p2.to_csv('house_blend_5_per.csv',index=False)


# In[ ]:


# Even after trying 1000 trees instead of 100 trees, only 6 jump up in leaderboard. Hence i think, data cleaning is required more.


# In[ ]:


# After few trys , i saw that taking very less fetures is results in bad score.


# In[ ]:





# ## I now want to drop similar columns by looking at the data.But unable to find intersection between columns.

# In[ ]:


x["Condition1"].value_counts()
# 1260/1460= 86%


# In[ ]:


x["Condition2"].value_counts()
# 1445/1460= 98%


# In[ ]:


z=pd.DataFrame(x[['Condition1','Condition2']])
z


# In[ ]:


# z['C'] = [(set(a) & set(b)) for a, b in zip(df.Condition1, df.Condition1) ]


# In[ ]:


z


# In[ ]:





# In[ ]:


x["Exterior1st"].value_counts()


# In[ ]:


x["Exterior2nd"].value_counts()


# Exterior1st and Exterior2nd columns seems almost same.Majority of them are same. But intersection should be same too.It may happen that different rows are same and not same rows.  

# In[ ]:





# 1. Feature Selection
# 2. Feature Scaling
# 3. Feature Engineering
# 4. Normization vs Standardization
# 5. Bell curves.
# 6. Probab. density function
# 7. z score for outliers.

# In[ ]:





# # Learning from others kernels

# In[ ]:


y.plot(kind='kde')


# In[ ]:


y.plot(kind='hist')


# In[ ]:


# sns.set_style("white") #Background color:- white, dark,etc.
# sns.set_color_codes(palette='deep') #Color user chooses
f, ax = plt.subplots(figsize=(8, 7))
#Check the new distribution 
sns.distplot(y, color="b");
# ax.grid(True)
ax.set(ylabel="Frequency")
ax.set(xlabel="SalePrice")
ax.set(title="SalePrice distribution")
# sns.despine(trim=True, left=True) #Removes lines of boxes in which image is plotted
plt.show()


# In[ ]:


# sns.set_style("white")
# sns.set_color_codes(palette='coolwarm')
f, ax = plt.subplots(figsize=(8, 7))
#Check the new distribution 
sns.distplot(y);
# ax.grid(True)
ax.set(ylabel="Frequency")
ax.set(xlabel="SalePrice")
ax.set(title="SalePrice distribution")
sns.despine(trim=True, left=True)
plt.show()


# The data is right skewed. It may not affect tree-based models but then we can't use other models. Hence to make it normal distribution, we apply Log transformation.
# 

# In[ ]:


y.skew(), y.kurt()


# # Questions:
# 1. deal with positively skewed data
# 2. handling bimodal distribution
# 3. kurtosis?- i think length of tail
# 4. https://towardsdatascience.com/transforming-skewed-data-73da4c2d0d16
# 5. 
# 6. **how does standard deviation affect data analysis?**
# 7. https://towardsdatascience.com/exploring-normal-distribution-with-jupyter-notebook-3645ec2d83f8
# 8. 

# # Handling skewed data by applying log(x+1). +1 as,if zero present thrn error.

# In[ ]:


y=np.log1p(y) #or np.log(y + 1) 


# In[ ]:


from scipy.stats import norm #Ideal normal distribution curve


# In[ ]:


sns.set_style("white")

f, ax = plt.subplots(figsize=(8, 7))

sns.distplot(y,fit=norm);
(mu, sigma) = norm.fit(y)
print(mu,sigma) #mu is mean
ax.set(ylabel="Frequency")
ax.set(xlabel="SalePrice")
ax.set(title="SalePrice distribution")
sns.despine(trim=True, left=True)
plt.show()


# # how does standard deviation/sigma affect data analysis?

# In[ ]:


y.describe()


# In[ ]:


x['OverallQual'].plot(kind='hist')


# In[ ]:


y.describe()


# In[ ]:


df['SalePrice']


# # These are Outliers

# In[ ]:


x[(x['OverallQual']<5) & (df['SalePrice']>200000)].index


# In[ ]:


x[(x['GrLivArea']>4500) & (df['SalePrice']>30000)].index


# ## Don;t know how to find outliers
# https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba
# See when to apply scatter( both continuous, or categorical?)
# Seebox plots

# In[ ]:


plt.scatter(x['OverallQual'],df['SalePrice'])


# In[ ]:


plt.scatter(x['GrLivArea'],df['SalePrice'])
plt.scatter(x=4500, y=300000, color='r')
plt.scatter(x=4676, y=184750, color='y')
plt.scatter(x=5642, y=160000, color='g')


# 2 outliers.   
# ## coordinates of point on scatter plot seaborn?

# In[ ]:


a=df[x['GrLivArea']>4500]
# a['GrLivArea']
a


# In[ ]:





# # Missing value function

# In[ ]:


def percent_missing(df):
    data = pd.DataFrame(df)
    df_cols = list(pd.DataFrame(data)) #List of Columns of dataframe named 'data'
    dict_x = {}
    for i in range(0, len(df_cols)):
        dict_x.update({df_cols[i]: round(data[df_cols[i]].isnull().mean()*100,2)}) #Make dictionary of missing columns as key and percentage (upto 2 decimal places rounded up ) as value
    
    return dict_x


# In[ ]:



missing = percent_missing(missing_df)
df_miss = sorted(missing.items(), key=lambda x: x[1], reverse=True)# lambda x:x[1] i.e. sorting acc. to second parameter i.e. percentage of missing values
print('Percent of missing data')
df_miss


# In[ ]:


sns.set_style("white")
f, ax = plt.subplots(figsize=(8, 7))
sns.set_color_codes(palette='deep')
missing = round(missing_df.isnull().mean()*100,2)
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar(color="b")
# Tweak the visual presentation
ax.grid(True)
ax.set(ylabel="Percent of missing values")
ax.set(xlabel="Features")
ax.set(title="Percent missing data by feature")
sns.despine(trim=True, left=True)


# In[ ]:





# Some of the non-numeric predictors are stored as numbers; convert them into strings 

# In[ ]:


# df['MoSold']
# all_features['YrSold']
# all_features['MoSold']


# In[ ]:


df['MSZoning'].mode()


# In[ ]:


x['MSZoning'].plot(kind='hist')


# In[ ]:


df['PoolQC'].value_counts()


# In[ ]:


df[df['GarageCars']==0]


# In[ ]:


(df[df['GarageYrBlt'].isnull()]).index


# In[ ]:


(df[df['GarageArea']==0]).index


# In[ ]:


df.loc[[39],["GarageYrBlt"]]


# In[ ]:


# df[df['GarageYrBlt'].isnull()==True]


# ## GarageYrBlt column should be replaced by 0 not mean.

# In[ ]:


df['GarageYrBlt'].mean()


# In[ ]:


df.loc[:,['GarageQual', 'GarageCond']]


# In[ ]:


# df['GarageQual'].value_counts()
(df[df['GarageQual'].isnull()]).index


# In[ ]:


# df['GarageCond'].value_counts()
(df[df['GarageCond'].isnull()]).index


# In[ ]:


df['GarageType'].value_counts()
(df[df['GarageType'].isnull()]).index


# In[ ]:


df['GarageFinish'].value_counts()
(df[df['GarageFinish'].isnull()]).index


# In[ ]:


df.loc[:,['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']]


# In[ ]:





# In[ ]:


df['BsmtQual'].value_counts()
(df[df['BsmtQual'].isnull()]).index


# In[ ]:


df['BsmtCond'].value_counts()
(df[df['BsmtCond'].isnull()]).index


# In[ ]:


df['BsmtExposure'].value_counts()
(df[df['BsmtExposure'].isnull()]).index


# In[ ]:


df['BsmtFinType1'].value_counts()
(df[df['BsmtFinType1'].isnull()]).index


# In[ ]:


df['BsmtFinType2'].value_counts()
(df[df['BsmtFinType2'].isnull()]).index


# In[ ]:


df.loc[:,['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']]


# In[ ]:





# In[ ]:


def handle_missing(features):
#     # the data description states that NA refers to typical ('Typ') values
#     features['Functional'] = features['Functional'].fillna('Typ')
#     # Replace the missing values in each of the columns below with their mode
#     features['Electrical'] = features['Electrical'].fillna("SBrkr")
#     features['KitchenQual'] = features['KitchenQual'].fillna("TA")
#     features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])
#     features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
#     features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])
    features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
#     ---------->No logic

    # the data description stats that NA refers to "No Pool"-------------------------------->No logic
    features["PoolQC"] = features["PoolQC"].fillna("None")
    
  *  # Replacing the missing values with 0, since no garage (i.e. 'GarageArea'=0) = no cars in garage
# GarageYrBlt is missing for GarageArea=0.
#     for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
#         features[col] = features[col].fillna(0)
        
    # Replacing the missing values with None------------------------------------------------->No logic
    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        features[col] = features[col].fillna('None')
        
    # NaN values for these categorical basement features, means there's no basement---------->No logic
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        features[col] = features[col].fillna('None')
        
    # Group the by neighborhoods, and fill in missing value by the median LotFrontage of the neighborhood---------->No logic
    features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

    # We have no particular intuition around how to fill in the rest of the categorical features
    # So we replace their missing values with None
    objects = []
    for i in features.columns:
        if features[i].dtype == object:
            objects.append(i)
    features.update(features[objects].fillna('None'))
        
    # And we do the same thing for numerical features, but this time with 0s
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric = []
    for i in features.columns:
        if features[i].dtype in numeric_dtypes:
            numeric.append(i)
    features.update(features[numeric].fillna(0))    
    return features

all_features = handle_missing(all_features)


# In[ ]:





# In[ ]:





# # Finding skewed columns

# In[ ]:


numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric = []
for i in df.columns:
    if df[i].dtype in numeric_dtypes:
        numeric.append(i)


# In[ ]:


from scipy.stats import skew


# In[ ]:


# Find skewed numerical features
skew_features = df[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

print("There are {} numerical features with Skew > 0.5 :".format(high_skew.shape[0]))
skewness = pd.DataFrame({'Skew' :high_skew})
skew_features


# In[ ]:


len(numeric)
# numeric.remove('SalePrice')


# In[ ]:


# df['Utilities'].plot(kind='box')
# Create box plots for all numeric features
sns.set_style("white")
f, ax = plt.subplots(figsize=(8, 7))
ax.set_xscale("log") #For scaling x-axis
ax = sns.boxplot(data=df[numeric] , orient="h", palette="Set1")
# ax.xaxis.grid(False)
# ax.set(ylabel="Feature names")
# ax.set(xlabel="Numeric values")
# ax.set(title="Numeric Distribution of Features")
# sns.despine(trim=True, left=True)


# Boxplot:-   
# https://medium.com/dayem-siddiqui/understanding-and-interpreting-box-plots-d07aab9d1b6c
# https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51

# ### Applying box--cox1 transformation:   
# https://www.youtube.com/watch?v=2gVA3TudAXI   
# https://medium.com/@ODSC/transforming-skewed-data-for-machine-learning-90e6cc364b0   
# https://medium.com/@ronakchhatbar/box-cox-transformation-cba8263c5206

# In[ ]:


from scipy.special import boxcox1p
for i in skew_index:
    missing_df[i] = boxcox1p(missing_df[i], boxcox_normmax(all_features[i] + 1))


# again find skewness after transformation

# In[ ]:


z=pd.DataFrame()


# In[ ]:


z['BsmtFinType1_Unf'] = 1*(df['BsmtFinType1'] == 'Unf')


# In[ ]:


df['BsmtFinType1']


# In[ ]:


df['BsmtFinType1'].value_counts()


# In[ ]:


df['BsmtFinType1']
z


# In[ ]:


df['Street'].value_counts()


# In[ ]:


df.loc[:,['OverallQual','OverallCond']]
# df['YearRemodAdd'].value_counts()


# In[ ]:


df['TotalBsmtSF'].apply(lambda x: np.exp(6) if x <= 0.0 else x)


# In[ ]:


df[df['TotalBsmtSF']<0]


# In[ ]:


df['TotalBsmtSF']


# In[ ]:


all_features['BsmtFinType1_Unf'] = 1*(all_features['BsmtFinType1'] == 'Unf')
all_features['HasWoodDeck'] = (all_features['WoodDeckSF'] == 0) * 1
all_features['HasOpenPorch'] = (all_features['OpenPorchSF'] == 0) * 1

all_features['HasEnclosedPorch'] = (all_features['EnclosedPorch'] == 0) * 1
all_features['Has3SsnPorch'] = (all_features['3SsnPorch'] == 0) * 1
all_features['HasScreenPorch'] = (all_features['ScreenPorch'] == 0) * 1

#******************************************#
all_features['YearsSinceRemodel'] = all_features['YrSold'].astype(int) - all_features['YearRemodAdd'].astype(int)
all_features['Total_Home_Quality'] = all_features['OverallQual'] + all_features['OverallCond']
all_features = all_features.drop(['Utilities', 'Street', 'PoolQC',], axis=1)#Logical to drop as all same values in cols

all_features['TotalSF'] = all_features['TotalBsmtSF'] + all_features['1stFlrSF'] + all_features['2ndFlrSF']
all_features['YrBltAndRemod'] = all_features['YearBuilt'] + all_features['YearRemodAdd']

all_features['Total_sqr_footage'] = (all_features['BsmtFinSF1'] + all_features['BsmtFinSF2'] +
                                 all_features['1stFlrSF'] + all_features['2ndFlrSF']) # See 

all_features['Total_Bathrooms'] = (all_features['FullBath'] + (0.5 * all_features['HalfBath']) +
                               all_features['BsmtFullBath'] + (0.5 * all_features['BsmtHalfBath']))

all_features['Total_porch_sf'] = (all_features['OpenPorchSF'] + all_features['3SsnPorch'] +
                              all_features['EnclosedPorch'] + all_features['ScreenPorch'] +
                              all_features['WoodDeckSF'])
#****************************************#
# WTH IS THIS
all_features['TotalBsmtSF'] = all_features['TotalBsmtSF'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
all_features['2ndFlrSF'] = all_features['2ndFlrSF'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)
all_features['GarageArea'] = all_features['GarageArea'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
all_features['GarageCars'] = all_features['GarageCars'].apply(lambda x: 0 if x <= 0.0 else x)
all_features['LotFrontage'] = all_features['LotFrontage'].apply(lambda x: np.exp(4.2) if x <= 0.0 else x)
all_features['MasVnrArea'] = all_features['MasVnrArea'].apply(lambda x: np.exp(4) if x <= 0.0 else x)
all_features['BsmtFinSF1'] = all_features['BsmtFinSF1'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)

all_features['haspool'] = all_features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
all_features['has2ndfloor'] = all_features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
all_features['hasgarage'] = all_features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
all_features['hasbsmt'] = all_features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
all_features['hasfireplace'] = all_features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)


# ### Again check skewness/ apply log, sq transformations...........see kernel

# In[ ]:


df = pd.get_dummies(df).reset_index(drop=True)
df.shape


# In[ ]:


df

