#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


pd.set_option('display.max_columns',300)


# In[ ]:


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# # Load datasets and check info

# In[ ]:


train = pd.read_csv('../input/amexpert-2019/train.csv')
train.head()


# In[ ]:


train.info()


# In[ ]:


train.coupon_id.nunique()


# In[ ]:


campaign = pd.read_csv('../input/amexpert-2019/campaign_data.csv')
campaign.head()


# In[ ]:


campaign.info()


# In[ ]:


coupon = pd.read_csv('../input/amexpert-2019/coupon_item_mapping.csv')
coupon.head()


# In[ ]:


coupon.info()


# In[ ]:


transaction = pd.read_csv('../input/amexpert-2019/customer_transaction_data.csv')
transaction.head()


# In[ ]:


train.shape


# In[ ]:


transaction.info()


# In[ ]:


item = pd.read_csv('../input/amexpert-2019/item_data.csv')
item.head()


# In[ ]:


item.category.unique()


# In[ ]:


demograph = pd.read_csv('../input/amexpert-2019/customer_demographics.csv')
demograph.head()


# In[ ]:


demograph.isnull().sum()/760


# In[ ]:


demograph.info()


# # Data preprocessing

# In[ ]:


campaign['start_date'] = pd.to_datetime(campaign['start_date'])
campaign['end_date'] = pd.to_datetime(campaign['end_date'])
#pd.to_datetime(campaign['end_date'])


# In[ ]:


campaign['duration'] = abs((campaign['end_date'] -  campaign['start_date']).dt.days)


# In[ ]:


demograph.income_bracket.unique()


# In[ ]:


demograph['marital_status'] = demograph.groupby(['family_size','age_range'])['marital_status'].apply(lambda x: x.fillna(x.mode()[0]))


# In[ ]:


demograph.marital_status.unique()


# In[ ]:


demograph.drop('no_of_children',axis=1,inplace=True)


# ### check if balanced set

# In[ ]:


sns.countplot(train.redemption_status)


# ## Merge dataframes

# In[ ]:


mtc = pd.merge(train,campaign,on='campaign_id',how='left')
mtc.head()


# In[ ]:


mtc.shape


# In[ ]:


mci = pd.merge(coupon,item,on='item_id',how='left')
mci.head()


# In[ ]:


mci.coupon_id.nunique()


# In[ ]:


mci.shape


# In[ ]:


mci.groupby('coupon_id').count().reset_index()[['coupon_id','item_id']].head()


# In[ ]:


mci_group = pd.DataFrame()


# In[ ]:


mci_group[['coupon_id','category_count']] = mci.groupby('coupon_id').count().reset_index()[['coupon_id','item_id']]


# In[ ]:


mci.groupby('coupon_id').max().reset_index().head()


# In[ ]:


mci_group[['brand_type','category']] = mci.groupby('coupon_id').max().reset_index()[['brand_type','category']]


# In[ ]:


mci_group.head()


# In[ ]:


#tgroup = transaction.groupby(['customer_id','item_id','date']).sum().reset_index()
tgroup = transaction.groupby(['customer_id']).sum().reset_index()


# In[ ]:


tgroup.head()


# In[ ]:


tgroup.drop('item_id',axis=1,inplace=True)


# In[ ]:


tgroup.shape


# In[ ]:


mdtg = pd.merge(tgroup,demograph,on='customer_id',how='outer')
mdtg.head()


# In[ ]:


mdtg.shape


# In[ ]:


mergeddata = pd.merge(mtc,mdtg,on=['customer_id'],how='left')
mergeddata.head()


# In[ ]:


mergeddata.shape


# In[ ]:


mergeddata.info()


# In[ ]:


mergeddata.isnull().sum()/78369


# ## fill null values

# In[ ]:


mergeddata['marital_status'].fillna(mergeddata['marital_status'].mode()[0],inplace=True)


# In[ ]:


mergeddata['age_range'].fillna(mergeddata['age_range'].mode()[0],inplace=True)


# In[ ]:


mergeddata['family_size'].fillna(mergeddata['family_size'].mode()[0],inplace=True)


# In[ ]:


mergeddata['rented'].fillna(mergeddata['rented'].mode()[0],inplace=True)


# In[ ]:


mergeddata['income_bracket'].fillna(mergeddata['income_bracket'].median(),inplace=True)


# In[ ]:


mergeddata = pd.merge(mergeddata,mci_group,on=['coupon_id'],how='left')
mergeddata.head()


# In[ ]:


mergeddata.info()


# # Label Encoder

# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


lc = LabelEncoder()


# In[ ]:


mergeddata['age_range'] = lc.fit_transform(mergeddata['age_range'])
mergeddata['family_size'] = lc.fit_transform(mergeddata['family_size'])


# In[ ]:


mergeddata.info()


# In[ ]:


item.category.unique()


# In[ ]:


cat_list = ['Bakery', 'Packaged Meat', 'Seafood', 'Dairy, Juices & Snacks',
            'Prepared Food','Meat','Salads', 'Alcohol','Vegetables (cut)']


# In[ ]:


def mapCategory(x):
    if x in cat_list:
        return 'consumable'
    else:
        return 'non-consumable'


# In[ ]:


mergeddata['category'] = mergeddata['category'].apply(mapCategory)


# In[ ]:


mergeddata['final_price'] = mergeddata['selling_price']+ mergeddata['other_discount'] + mergeddata['coupon_discount']


# In[ ]:


mergeddata.drop(['selling_price','other_discount','coupon_discount'],axis=1,inplace=True)


# In[ ]:


mergeddata.income_bracket.unique()


# In[ ]:


sns.countplot(mergeddata.income_bracket)


# In[ ]:


def mapIncome(x):
    if (x<4):
        return 'low'
    elif (x>=4 and x<=7):
        return 'middle'
    elif (x>7 and x<=10):
        return 'upper-middle'
    elif (x>10):
        return 'high'


# In[ ]:


mergeddata['income_bracket'] = mergeddata['income_bracket'].apply(mapIncome)


# In[ ]:


inc_dict = {'low':1,'middle':2,'upper-middle':3,'high':4}
inc_dict


# In[ ]:


mergeddata['income_bracket'] = mergeddata['income_bracket'].map(inc_dict)


# # One hot encoding

# In[ ]:


dummydata = pd.get_dummies(mergeddata.drop(['redemption_status','coupon_id','customer_id','id','campaign_id','start_date','end_date'],axis=1))
dummydata.head()


# # RobustScaler

# In[ ]:


from sklearn.preprocessing import RobustScaler


# In[ ]:


rc = RobustScaler()


# In[ ]:


scaledData = pd.DataFrame(rc.fit_transform(dummydata),columns=dummydata.columns)
scaledData.head()


# In[ ]:


x = scaledData
y = mergeddata['redemption_status']


# In[ ]:


test = pd.read_csv('../input/amexpert-2019/test_QyjYwdj.csv')
test.head()


# # Test data processing

# In[ ]:


mtc_test = pd.merge(test,campaign,on='campaign_id',how='left')

mci_test = pd.merge(coupon,item,on='item_id')

mdtg_test = pd.merge(tgroup,demograph,on='customer_id',how='outer')
mergeddata_test = pd.merge(mtc_test,mdtg_test,on=['customer_id'],how='left')

mergeddata_test['marital_status'].fillna(mergeddata_test['marital_status'].mode()[0],inplace=True)
mergeddata_test['age_range'].fillna(mergeddata_test['age_range'].mode()[0],inplace=True)
mergeddata_test['family_size'].fillna(mergeddata_test['family_size'].mode()[0],inplace=True)
mergeddata_test['rented'].fillna(mergeddata_test['rented'].mode()[0],inplace=True)
mergeddata_test['income_bracket'].fillna(mergeddata_test['income_bracket'].median(),inplace=True)

mergeddata_test = pd.merge(mergeddata_test,mci_group,on=['coupon_id'],how='left')

mergeddata_test['age_range'] = lc.fit_transform(mergeddata_test['age_range'])
mergeddata_test['family_size'] = lc.fit_transform(mergeddata_test['family_size'])

mergeddata_test['category'] = mergeddata_test['category'].apply(mapCategory)
mergeddata_test['final_price'] = mergeddata_test['selling_price']+ mergeddata_test['other_discount'] + mergeddata_test['coupon_discount']
mergeddata_test.drop(['selling_price','other_discount','coupon_discount'],axis=1,inplace=True)
mergeddata_test['income_bracket'] = mergeddata_test['income_bracket'].apply(mapIncome)
mergeddata_test['income_bracket'] = mergeddata_test['income_bracket'].map(inc_dict)

dummydata_test = pd.get_dummies(mergeddata_test.drop(['coupon_id','customer_id','id','campaign_id','start_date','end_date'],axis=1))
scaledData_test = pd.DataFrame(rc.fit_transform(dummydata_test),columns=dummydata_test.columns)
scaledData_test.head()


# In[ ]:


mergeddata_test.info()


# # Model Building

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y,random_state=0)


# In[ ]:


from sklearn.metrics import accuracy_score,roc_auc_score


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ## LogisticRegression

# In[ ]:


lor = LogisticRegression()


# In[ ]:


lor.fit(xtrain,ytrain)
ypredlor = lor.predict(xtest)


# In[ ]:


accuracy_score(ytest,ypredlor)


# In[ ]:


yproba_yes = lor.predict_proba(xtest)[:,1]


# In[ ]:


roc_auc_score(ytest,yproba_yes)


# ## RandomForestClassifier

# In[ ]:


rf = RandomForestClassifier()
rf.fit(xtrain,ytrain)
ypredrf = rf.predict(xtest)
print(accuracy_score(ytest,ypredrf))
yproba_yes = rf.predict_proba(xtest)[:,1]
roc_auc_score(ytest,yproba_yes)


# In[ ]:


lor.fit(x,y)
ypredlor = lor.predict(scaledData_test)
ypredlor


# In[ ]:


print(test.id.shape)
print(ypredlor.shape)
print(scaledData_test.shape)


# In[ ]:


transaction.shape


# In[ ]:


#mergeddata_test[mergeddata_test.duplicated()]


# In[ ]:


dummydata_test.shape


# In[ ]:


submission = pd.DataFrame({'id':test['id'],'redemption_status':ypredlor})
submission.head()


# In[ ]:


submission.to_csv('rahul_amex_lor.csv',index=False)


# In[ ]:


rf = RandomForestClassifier()
rf.fit(x,y)
ypredrf = rf.predict(scaledData_test)
submission = pd.DataFrame({'id':test['id'],'redemption_status':ypredrf})
submission.head()
submission.to_csv('rahul_amex_rf.csv',index=False)


# In[ ]:


param_rf = {
    'max_depth': [2, 3, 4],
    'bootstrap': [True, False],
    'max_features': ['auto', 'sqrt', 'log2', None],
    'criterion': ['gini', 'entropy']
}


# In[ ]:





# In[ ]:


#from sklearn.model_selection import GridSearchCV
# gridRf = GridSearchCV(rf, cv = 10,param_grid=param_rf,scoring='roc_auc')
# gridRf.fit(x,y)
# gridRf.best_params_


# # XGBClassifier

# In[ ]:


from xgboost import XGBClassifier


# In[ ]:


xg = XGBClassifier()


# In[ ]:


xg.fit(xtrain,ytrain)
ypredxg = xg.predict(xtest)
print(accuracy_score(ytest,ypredxg))
yproba_yes = xg.predict_proba(xtest)[:,1]
roc_auc_score(ytest,yproba_yes)


# In[ ]:


xg = XGBClassifier()
xg.fit(x,y)
ypredxg = xg.predict(scaledData_test)
submission = pd.DataFrame({'id':test['id'],'redemption_status':ypredxg})
submission.head()
submission.to_csv('rahul_amex_xgb.csv',index=False)


# # AdaBoostClassifier

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier


# In[ ]:


ada = AdaBoostClassifier()


# In[ ]:


ada.fit(xtrain,ytrain)
ypredada = ada.predict(xtest)
print(accuracy_score(ytest,ypredada))
yproba_yes = ada.predict_proba(xtest)[:,1]
roc_auc_score(ytest,yproba_yes)


# # GradientBoostingClassifier

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


gba = GradientBoostingClassifier()
gba.fit(xtrain,ytrain)
ypredgba = gba.predict(xtest)
print(accuracy_score(ytest,ypredgba))
yproba_yes = gba.predict_proba(xtest)[:,1]
roc_auc_score(ytest,yproba_yes)


# In[ ]:


gba = GradientBoostingClassifier()
gba.fit(x,y)
ypredgba = gba.predict(scaledData_test)
submission = pd.DataFrame({'id':test['id'],'redemption_status':ypredgba})
submission.head()
submission.to_csv('rahul_amex_gba.csv',index=False)


# In[ ]:


from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# # GaussianNB

# In[ ]:


gba = GaussianNB()
gba.fit(xtrain,ytrain)
ypredgba = gba.predict(xtest)
print(accuracy_score(ytest,ypredgba))
yproba_yes = gba.predict_proba(xtest)[:,1]
roc_auc_score(ytest,yproba_yes)


# # KNeighborsClassifier

# In[ ]:


gba = KNeighborsClassifier()
gba.fit(xtrain,ytrain)
ypredgba = gba.predict(xtest)
print(accuracy_score(ytest,ypredgba))
yproba_yes = gba.predict_proba(xtest)[:,1]
roc_auc_score(ytest,yproba_yes)


# In[ ]:


# gba = SVC()
# gba.fit(xtrain,ytrain)
# ypredgba = gba.predict(xtest)
# print(accuracy_score(ytest,ypredgba))
# yproba_yes = gba.predict_proba(xtest)[:,1]
# roc_auc_score(ytest,yproba_yes)


# # downsampling NearMiss

# In[ ]:


from imblearn.under_sampling import NearMiss


# In[ ]:


nm = NearMiss()


# In[ ]:


downx,downy = nm.fit_sample(x.drop(['age_range','rented','campaign_type_Y','marital_status_Single', 'category_non-consumable','brand_type_Established','income_bracket'],axis=1),y)


# In[ ]:


xtrain2,xtest2,ytrain2,ytest2 = train_test_split(downx,downy,random_state=0)


# In[ ]:


rf = RandomForestClassifier(n_estimators=300)
rf.fit(xtrain2,ytrain2)
ypredrf2 = rf.predict(xtest2)
print(accuracy_score(ytest2,ypredrf2))
yproba_yes2 = rf.predict_proba(xtest2)[:,1]
roc_auc_score(ytest2,yproba_yes2)


# In[ ]:


xg = XGBClassifier(n_estimators=150,learning_rate=0.2)
xg.fit(xtrain2,ytrain2)
ypredxg2 = xg.predict(xtest2)
print(accuracy_score(ytest2,ypredxg2))
yproba_yes2 = xg.predict_proba(xtest2)[:,1]
roc_auc_score(ytest2,yproba_yes2)


# In[ ]:


ada = AdaBoostClassifier(n_estimators=150,learning_rate=0.4)
ada.fit(xtrain2,ytrain2)
ypredada2 = ada.predict(xtest2)
print(accuracy_score(ytest2,ypredada2))
yproba_yes2 = ada.predict_proba(xtest2)[:,1]
roc_auc_score(ytest2,yproba_yes2)


# In[ ]:


gba = GradientBoostingClassifier(n_estimators=150,learning_rate=0.2)
gba.fit(xtrain2,ytrain2)
ypredgba2 = gba.predict(xtest2)
print(accuracy_score(ytest2,ypredgba2))
yproba_yes2 = gba.predict_proba(xtest2)[:,1]
roc_auc_score(ytest2,yproba_yes2)


# # upsampling SMOTE

# In[ ]:


from imblearn.over_sampling import SMOTE


# In[ ]:


sm = SMOTE()


# In[ ]:


upx,upy = sm.fit_sample(x.drop(['age_range','rented','campaign_type_Y','marital_status_Single', 'category_non-consumable','brand_type_Established','income_bracket'],axis=1),y)


# In[ ]:


scaledData_test_drop = scaledData_test.drop(['age_range','rented','campaign_type_Y','marital_status_Single',                                'category_non-consumable','brand_type_Established','income_bracket'],axis=1)


# In[ ]:


xtrain2,xtest2,ytrain2,ytest2 = train_test_split(upx,upy,random_state=0)


# In[ ]:


rf = RandomForestClassifier(n_estimators=150)
rf.fit(xtrain2,ytrain2)
ypredrf2 = rf.predict(xtest2)
print(accuracy_score(ytest2,ypredrf2))
yproba_yes2 = rf.predict_proba(xtest2)[:,1]
roc_auc_score(ytest2,yproba_yes2)


# In[ ]:


xg = XGBClassifier(n_estimators=250,learning_rate=0.8)
xg.fit(xtrain2,ytrain2)
ypredxg2 = xg.predict(xtest2)
print(accuracy_score(ytest2,ypredxg2))
yproba_yes2 = xg.predict_proba(xtest2)[:,1]
roc_auc_score(ytest2,yproba_yes2)


# In[ ]:


gba = GradientBoostingClassifier(n_estimators=250,learning_rate=0.8)
gba.fit(xtrain2,ytrain2)
ypredgba2 = gba.predict(xtest2)
print(accuracy_score(ytest2,ypredgba2))
yproba_yes2 = gba.predict_proba(xtest2)[:,1]
roc_auc_score(ytest2,yproba_yes2)


# In[ ]:


ada = AdaBoostClassifier(n_estimators=150,learning_rate=0.4)
ada.fit(xtrain2,ytrain2)
ypredada2 = ada.predict(xtest2)
print(accuracy_score(ytest2,ypredada2))
yproba_yes2 = ada.predict_proba(xtest2)[:,1]
roc_auc_score(ytest2,yproba_yes2)


# # Feature Selection

# In[ ]:


pd.Series(rf.feature_importances_,scaledData_test_drop.columns).plot.barh()


# In[ ]:


pd.Series(gba.feature_importances_,scaledData_test_drop.columns).plot.barh()


# In[ ]:


pd.Series(xg.feature_importances_,scaledData_test_drop.columns).plot.barh()


# In[ ]:


import statsmodels.api as sms


# In[ ]:


temp = x.copy()
temp['constant'] = 1


# # Logit

# In[ ]:


sms.Logit(y,temp).fit().summary()


# In[ ]:


from sklearn.ensemble import VotingClassifier


# up

# In[ ]:


# vf = VotingClassifier(estimators=estimator,voting='soft')
# vf.fit(xtrain2,ytrain2)
# ypredada2 = vf.predict(xtest2)
# print(accuracy_score(ytest2,ypredada2))
# yproba_yes2 = vf.predict_proba(xtest2)[:,1]
# roc_auc_score(ytest2,yproba_yes2)


# down

# In[ ]:


# vf = VotingClassifier(estimators=estimator,voting='soft')
# vf.fit(xtrain2,ytrain2)
# ypredada2 = vf.predict(xtest2)
# print(accuracy_score(ytest2,ypredada2))
# yproba_yes2 = vf.predict_proba(xtest2)[:,1]
# roc_auc_score(ytest2,yproba_yes2)


# In[ ]:


# vf = VotingClassifier(estimators=estimator,voting='soft')
# vf.fit(upx,upy)
# ypredvf = vf.predict(scaledData_test)
# submission = pd.DataFrame({'id':test['id'],'redemption_status':ypredvf})
# submission.head()
# submission.to_csv('rahul_amex_vf.csv',index=False)


# In[ ]:


rf = RandomForestClassifier(n_estimators=150)
rf.fit(downx,downy)
ypredrf = rf.predict(scaledData_test_drop)
yproba_rf = rf.predict_proba(scaledData_test_drop)[:,1]
#submission = pd.DataFrame({'id':test['id'],'redemption_status':ypredrf})
submission = pd.DataFrame({'id':test['id'],'redemption_status':yproba_rf})
submission.to_csv('rahul_amex_rf.csv',index=False)


# In[ ]:


gba = GradientBoostingClassifier(n_estimators=250,learning_rate=0.8)
gba.fit(upx,upy)
ypredgba = gba.predict(scaledData_test_drop)
yproba_gba = gba.predict_proba(scaledData_test_drop)[:,1]

submission = pd.DataFrame({'id':test['id'],'redemption_status':yproba_gba})
submission.head()
submission.to_csv('rahul_amex_gba.csv',index=False)


# In[ ]:


xg = XGBClassifier(n_estimators=150,learning_rate=0.2)
xg.fit(downx,downy)
ypredxgb = xg.predict(scaledData_test_drop.values)
yproba_xgb = xg.predict_proba(scaledData_test_drop.values)[:,1]

submission = pd.DataFrame({'id':test['id'],'redemption_status':yproba_xgb})
submission.head()
submission.to_csv('rahul_amex_xgb.csv',index=False)


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[ ]:


vif = [variance_inflation_factor(dummydata.drop(['age_range','rented','campaign_type_Y',                                        'marital_status_Single','category_non-consumable','brand_type_Established','income_bracket'],axis=1).values,i) for i in range(dummydata.drop(['age_range','rented','campaign_type_Y',                                        'marital_status_Single','category_non-consumable','brand_type_Established','income_bracket'],axis=1).shape[1])]
pd.Series(vif,index=dummydata.drop(['age_range','rented','campaign_type_Y',                                        'marital_status_Single','category_non-consumable','brand_type_Established','income_bracket'],axis=1).columns)


# In[ ]:


from sklearn.svm import SVC


# In[ ]:


# svm = SVC()
# svm.fit(xtrain2,ytrain2)
# ypredsvm = svm.predict(xtest2)
# print(accuracy_score(ytest2,ypredsvm))
# yproba_yes2 = svm.predict_proba(xtest2)[:,1]
# roc_auc_score(ytest2,yproba_yes2)


# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


from sklearn.preprocessing import PolynomialFeatures


# # PCA

# In[ ]:


pca = PCA(n_components=5)


# In[ ]:


pcax = pca.fit_transform(upx)


# In[ ]:


xtrain2,xtest2,ytrain2,ytest2 = train_test_split(pcax,upy,random_state=0)


# In[ ]:


gba = GradientBoostingClassifier(n_estimators=150,learning_rate=0.4)
gba.fit(xtrain2,ytrain2)
ypredgba2 = gba.predict(xtest2)
print(accuracy_score(ytest2,ypredgba2))
yproba_yes2 = gba.predict_proba(xtest2)[:,1]
roc_auc_score(ytest2,yproba_yes2)


# In[ ]:


xg = XGBClassifier(n_estimators=150,learning_rate=0.4)
xg.fit(xtrain2,ytrain2)
ypredxg2 = xg.predict(xtest2)
print(accuracy_score(ytest2,ypredxg2))
yproba_yes2 = xg.predict_proba(xtest2)[:,1]
roc_auc_score(ytest2,yproba_yes2)


# # Polynomial Features

# In[ ]:


pl = PolynomialFeatures(degree=3)


# In[ ]:


polyx = pl.fit_transform(upx)


# In[ ]:


polyxtest = pl.fit_transform(scaledData_test_drop)


# In[ ]:


xtrain2,xtest2,ytrain2,ytest2 = train_test_split(polyx,upy,random_state=0)


# In[ ]:


gba = GradientBoostingClassifier(n_estimators=150,learning_rate=0.5)
gba.fit(xtrain2,ytrain2)
ypredgba2 = gba.predict(xtest2)
print(accuracy_score(ytest2,ypredgba2))
yproba_yes2 = gba.predict_proba(xtest2)[:,1]
roc_auc_score(ytest2,yproba_yes2)


# In[ ]:


# gba = GradientBoostingClassifier(n_estimators=150,learning_rate=0.4)
# gba.fit(polyx.values,upy)
# ypredgba = gba.predict(polyxtest.values)
# yproba_gba = gba.predict_proba(polyxtest.values)[:,1]

# submission = pd.DataFrame({'id':test['id'],'redemption_status':yproba_gba})
# submission.head()
# submission.to_csv('rahul_amex_gba.csv',index=False)


# In[ ]:




