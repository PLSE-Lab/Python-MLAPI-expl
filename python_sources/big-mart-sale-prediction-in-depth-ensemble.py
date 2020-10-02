#!/usr/bin/env python
# coding: utf-8

# # BIG MART SALES PREDICTION
# 
# ## Feel free to ask doubts, give suggestions and upvote if you like my work :)

# ## Importing files

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df=pd.read_csv('../input/Train.csv')


# ## Let's explore training data

# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


df.hist(figsize=(15,12))


# In[ ]:


df.info()


# ## Column ITEM WEIGHT and OUTLET SIZE contain missing values. 

# ## Lets see correlation b/w target and features

# In[ ]:


corr_matrix=df.corr()
corr_matrix['Item_Outlet_Sales']


# ## ITEM_MRP seems to have a good correlation with targeted ITEM_OUTLET_SALES and other columns are not very useful for prediction of target value

# ## Lets start checking columns relation with Target ITEM_OUTLET_SALES Price
# ## First is ITEM_IDENTIFIER

# In[ ]:


df.Item_Identifier.value_counts()


# ## From above output we can say that ITEM_IDENTIFIER should be categorical columns
# ## Since ITEM_WEIGHT column correlation strength is very low so we can drop it
# ## Next Column is ITEM_FAT_CONTENT

# In[ ]:


df.Item_Fat_Content.value_counts()


# ## LF, low fat belong to same category that is Low Fat and reg belong to Regular category so replacing LF, low fat and reg to thier category by

# In[ ]:


df.Item_Fat_Content=df.Item_Fat_Content.replace('LF','Low Fat')


# In[ ]:


df.Item_Fat_Content=df.Item_Fat_Content.replace('reg','Regular')
df.Item_Fat_Content=df.Item_Fat_Content.replace('low fat','Low Fat')


# In[ ]:


df.Item_Fat_Content.value_counts()


# ## For further data processing we need to convert column type into their correct type

# In[ ]:


df.Item_Identifier=df.Item_Identifier.astype('category')
df.Item_Fat_Content=df.Item_Fat_Content.astype('category')
df.Item_Type=df.Item_Type.astype('category')
df.Outlet_Identifier=df.Outlet_Identifier.astype('category')
df.Outlet_Establishment_Year=df.Outlet_Establishment_Year.astype('int64')

df.Outlet_Type=df.Outlet_Type.astype('category')
df.Outlet_Location_Type=df.Outlet_Location_Type.astype('category')
df.Outlet_Size=df.Outlet_Size.astype('category')


# ## Now ITEM_MRP column. Correlation strength of this column with target column is very high so we need can exploit this column for further infomation about target column

# In[ ]:


fig,axes=plt.subplots(1,1,figsize=(12,8))
sns.scatterplot(x='Item_MRP',y='Item_Outlet_Sales',hue='Item_Fat_Content',size='Item_Weight',data=df)


# ## ITEM_MRP column contain prices which are in clusters so it would be better if we convert this columnn into bins for further processing

# In[ ]:


df.describe()


# In[ ]:


fig,axes=plt.subplots(1,1,figsize=(10,8))
sns.scatterplot(x='Item_MRP',y='Item_Outlet_Sales',hue='Item_Fat_Content',size='Item_Weight',data=df)
plt.plot([69,69],[0,5000])
plt.plot([137,137],[0,5000])
plt.plot([203,203],[0,9000])


# ## We can use these perpendicular lines to divide data into proper bins. So from above graph we got out bin value. Now

# In[ ]:


df.Item_MRP=pd.cut(df.Item_MRP,bins=[25,69,137,203,270],labels=['a','b','c','d'],right=True)


# In[ ]:


df.head()


# ## Now lets explore other columns

# In[ ]:


fig,axes=plt.subplots(3,1,figsize=(15,12))
sns.scatterplot(x='Item_Visibility',y='Item_Outlet_Sales',hue='Item_MRP',ax=axes[0],data=df)
sns.boxplot(x='Item_Type',y='Item_Outlet_Sales',ax=axes[1],data=df)
sns.boxplot(x='Outlet_Identifier',y='Item_Outlet_Sales',ax=axes[2],data=df)


# In[ ]:


fig,axes=plt.subplots(2,2,figsize=(15,12))
sns.boxplot(x='Outlet_Establishment_Year',y='Item_Outlet_Sales',ax=axes[0,0],data=df)
sns.boxplot(x='Outlet_Size',y='Item_Outlet_Sales',ax=axes[0,1],data=df)
sns.boxplot(x='Outlet_Location_Type',y='Item_Outlet_Sales',ax=axes[1,0],data=df)
sns.boxplot(x='Outlet_Type',y='Item_Outlet_Sales',ax=axes[1,1],data=df)


# ## From above plots we can say that we can drop ITEM_VISIBILiTY along with ITEM_WEIGHT . Further more both of these column have very low correlation strength with target column.
# 
# ## Therefore Columns for model training will be

# In[ ]:


attributes=['Item_MRP','Outlet_Type','Outlet_Location_Type','Outlet_Size','Outlet_Establishment_Year','Outlet_Identifier','Item_Type','Item_Outlet_Sales']


# In[ ]:


fig,axes=plt.subplots(2,2,figsize=(15,12))
sns.boxplot(x='Outlet_Establishment_Year',y='Item_Outlet_Sales',hue='Outlet_Size',ax=axes[0,0],data=df)
sns.boxplot(x='Outlet_Size',y='Item_Outlet_Sales',hue='Outlet_Size',ax=axes[0,1],data=df)
sns.boxplot(x='Outlet_Location_Type',y='Item_Outlet_Sales',hue='Outlet_Size',ax=axes[1,0],data=df)
sns.boxplot(x='Outlet_Type',y='Item_Outlet_Sales',hue='Outlet_Size',ax=axes[1,1],data=df)


# In[ ]:


data=df[attributes]


# In[ ]:


data.info()


# In[ ]:


fig,axes=plt.subplots(1,1,figsize=(8,6))
sns.boxplot(y='Item_Outlet_Sales',hue='Outlet_Type',x='Outlet_Location_Type',data=data)


# In[ ]:


data[data.Outlet_Size.isnull()]


# ## One thing to observe is when OUTLET_TYPE = supermarket type 1 and OUTLET_LOCATION_TYPE is Tier 2 then outlet size is null furthermore when OUTLET_TYPE = Grocery store and OUTLET_LOCATION_TYPE is Tier 3 then outlet size is always null 

# In[ ]:


data.groupby('Outlet_Type').get_group('Grocery Store')['Outlet_Location_Type'].value_counts()


# In[ ]:


data.groupby('Outlet_Type').get_group('Grocery Store')


# In[ ]:


data.groupby(['Outlet_Location_Type','Outlet_Type'])['Outlet_Size'].value_counts()


# In[ ]:


(data.Outlet_Identifier=='OUT010').value_counts()


# In[ ]:


data.groupby('Outlet_Size').Outlet_Identifier.value_counts()


# ## Tier 1 have small and medium size shop. Tier 2 have small and (missing 1) type shop. Tier 3 have 2-medium and 1 high and (missing 2) shop
# ## Tier 2 will have medium size shop in missing 1 and Tier 3 will be high or medium size shop

# In[ ]:


def func(x):
    if x.Outlet_Identifier == 'OUT010' :
        x.Outlet_Size == 'High'
    elif x.Outlet_Identifier == 'OUT045' :
        x.Outlet_Size == 'Medium'
    elif x.Outlet_Identifier == 'OUT017' :
        x.Outlet_Size == 'Medium'
    elif x.Outlet_Identifier == 'OUT013' :
        x.Outlet_Size == 'High'
    elif x.Outlet_Identifier == 'OUT046' :
        x.Outlet_Size == 'Small'
    elif x.Outlet_Identifier == 'OUT035' :
        x.Outlet_Size == 'Small'
    elif x.Outlet_Identifier == 'OUT019' :
        x.Outlet_Size == 'Small'
    elif x.Outlet_Identifier == 'OUT027' :
        x.Outlet_Size == 'Medium'
    elif x.Outlet_Identifier == 'OUT049' :
        x.Outlet_Size == 'Medium'
    elif x.Outlet_Identifier == 'OUT018' :
        x.Outlet_Size == 'Medium'
    return(x)


# In[ ]:


data.Outlet_Size=data.apply(func,axis=1)


# ## Now lets checkout OUTLIERS 

# In[ ]:


data.head()


# In[ ]:


sns.boxplot(x='Item_MRP',y='Item_Outlet_Sales',data=data)


# In[ ]:


data[data.Item_MRP=='b'].Item_Outlet_Sales.max()


# In[ ]:


data[data.Item_Outlet_Sales==7158.6816]


# In[ ]:


data=data.drop(index=7796)
data.groupby('Item_MRP').get_group('b')['Item_Outlet_Sales'].max()


# In[ ]:


sns.boxplot(x='Outlet_Type',y='Item_Outlet_Sales',data=data)


# In[ ]:


sns.boxplot(x='Outlet_Location_Type',y='Item_Outlet_Sales',data=data)


# In[ ]:


data[data.Outlet_Location_Type=='Tier 1'].Item_Outlet_Sales.max()


# In[ ]:


data[data['Item_Outlet_Sales']==9779.9362]


# In[ ]:


data=data.drop(index=4289)


# In[ ]:


sns.boxplot(x='Outlet_Size',y='Item_Outlet_Sales',data=data)


# In[ ]:


sns.boxplot(x='Outlet_Establishment_Year',y='Item_Outlet_Sales',data=data)


# In[ ]:


data.Outlet_Establishment_Year=data.Outlet_Establishment_Year.astype('category')
data_label=data.Item_Outlet_Sales
data_dummy=pd.get_dummies(data.iloc[:,0:6])


# In[ ]:


data_dummy['Item_Outlet_Sales']=data_label


# In[ ]:


data_dummy.shape


# # Now we are ready to apply ML algorithms

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


train,test = train_test_split(data_dummy,test_size=0.20,random_state=2019)


# In[ ]:


train.shape , test.shape


# In[ ]:


train_label=train['Item_Outlet_Sales']
test_label=test['Item_Outlet_Sales']
del train['Item_Outlet_Sales']
del test['Item_Outlet_Sales']


# # Applying Linear Regression 
# 

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lr=LinearRegression()


# In[ ]:


lr.fit(train,train_label)


# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


predict_lr=lr.predict(test)


# In[ ]:


mse=mean_squared_error(test_label,predict_lr)


# In[ ]:


lr_score=np.sqrt(mse)


# In[ ]:


lr_score


# # Cross Val for Linear Regression

# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


score=cross_val_score(lr,train,train_label,cv=10,scoring='neg_mean_squared_error')


# In[ ]:


lr_score_cross=np.sqrt(-score)


# In[ ]:


np.mean(lr_score_cross),np.std(lr_score_cross)


# # Ridge Regression

# In[ ]:


from sklearn.linear_model import Ridge
r=Ridge(alpha=0.05,solver='cholesky')
r.fit(train,train_label)
predict_r=r.predict(test)
mse=mean_squared_error(test_label,predict_r)
r_score=np.sqrt(mse)
r_score


# # Cross Val Ridge

# In[ ]:


r=Ridge(alpha=0.05,solver='cholesky')
score=cross_val_score(r,train,train_label,cv=10,scoring='neg_mean_squared_error')
r_score_cross=np.sqrt(-score)
np.mean(r_score_cross),np.std(r_score_cross)


# # LASSO

# In[ ]:


from sklearn.linear_model import Lasso
l=Lasso(alpha=0.01)
l.fit(train,train_label)
predict_l=l.predict(test)
mse=mean_squared_error(test_label,predict_l)
l_score=np.sqrt(mse)
l_score


# # Cross VAl LAsso

# In[ ]:


l=Lasso(alpha=0.01)
score=cross_val_score(l,train,train_label,cv=10,scoring='neg_mean_squared_error')
l_score_cross=np.sqrt(-score)
np.mean(l_score_cross),np.std(l_score_cross)


# # Elastic NEt

# In[ ]:


from sklearn.linear_model import ElasticNet
en=ElasticNet(alpha=0.01,l1_ratio=0.5)
en.fit(train,train_label)
predict_r=en.predict(test)
mse=mean_squared_error(test_label,predict_r)
en_score=np.sqrt(mse)
en_score


# # Cross val Elastic

# In[ ]:


en=ElasticNet(alpha=0.01,l1_ratio=0.5)
score=cross_val_score(en,train,train_label,cv=10,scoring='neg_mean_squared_error')
en_score_cross=np.sqrt(-score)
np.mean(en_score_cross),np.std(en_score_cross)


# # Stochastic gradient

# In[ ]:


from sklearn.linear_model import SGDRegressor
sgd=SGDRegressor(penalty='l2',n_iter=100,alpha=0.05)
sgd.fit(train,train_label)
predict_r=sgd.predict(test)
mse=mean_squared_error(test_label,predict_r)
sgd_score=np.sqrt(mse)
sgd_score


# # Cross Val Stochastic Gradient

# In[ ]:


sgd=SGDRegressor(penalty='l2',n_iter=100,alpha=0.05)
score=cross_val_score(sgd,train,train_label,cv=10,scoring='neg_mean_squared_error')
sgd_score_cross=np.sqrt(-score)
np.mean(sgd_score_cross),np.std(sgd_score_cross)


# # SVR

# In[ ]:


from sklearn.svm import SVR
svm=SVR(epsilon=15,kernel='linear')
svm.fit(train,train_label)
predict_r=svm.predict(test)
mse=mean_squared_error(test_label,predict_r)
svm_score=np.sqrt(mse)
svm_score


# # Cross VAl SVR

# In[ ]:


svm=SVR(epsilon=15,kernel='linear')
score=cross_val_score(svm,train,train_label,cv=10,scoring='neg_mean_squared_error')
svm_score_cross=np.sqrt(-score)
np.mean(svm_score_cross),np.std(svm_score_cross)


# # Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor()
dtr.fit(train,train_label)
predict_r=dtr.predict(test)
mse=mean_squared_error(test_label,predict_r)
dtr_score=np.sqrt(mse)
dtr_score


# # Cross Val Decision Tree

# In[ ]:


dtr=DecisionTreeRegressor()
score=cross_val_score(dtr,train,train_label,cv=10,scoring='neg_mean_squared_error')
dtr_score_cross=np.sqrt(-score)
np.mean(dtr_score_cross),np.std(dtr_score_cross)


# # Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
rf.fit(train,train_label)
predict_r=rf.predict(test)
mse=mean_squared_error(test_label,predict_r)
rf_score=np.sqrt(mse)
rf_score


# # Cross Val RandomForest

# In[ ]:


rf=RandomForestRegressor()
score=cross_val_score(rf,train,train_label,cv=10,scoring='neg_mean_squared_error')
rf_score_cross=np.sqrt(-score)
np.mean(rf_score_cross),np.std(rf_score_cross)


# # Bagging Regressoion

# In[ ]:


from sklearn.ensemble import BaggingRegressor


# In[ ]:


br=BaggingRegressor(max_samples=70)


# In[ ]:


br.fit(train,train_label)


# In[ ]:


score=br.predict(test)


# In[ ]:


br_score=mean_squared_error(test_label,score)


# In[ ]:


br_score=np.sqrt(br_score)
br_score


# # Cross Val Bagging

# In[ ]:


br=BaggingRegressor()
score=cross_val_score(br,train,train_label,cv=10,scoring='neg_mean_squared_error')
br_score_cross=np.sqrt(-score)
np.mean(br_score_cross),np.std(br_score_cross)


# # ADAPTIVE BOOSTING

# In[ ]:


from sklearn.ensemble import AdaBoostRegressor
ada=AdaBoostRegressor()
ada.fit(train,train_label)
g=ada.predict(test)
ada_score=mean_squared_error(test_label,g)
ada_score=np.sqrt(ada_score)
ada_score


# # Cross val for ADA BOOST

# In[ ]:


ada=AdaBoostRegressor()
score=cross_val_score(ada,train,train_label,cv=10,scoring='neg_mean_squared_error')
ada_score_cross=np.sqrt(-score)
np.mean(ada_score_cross),np.std(ada_score_cross)


# # Gradient BOOSTING

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
gbr=GradientBoostingRegressor()
gbr.fit(train,train_label)
p=gbr.predict(test)
gb_score=mean_squared_error(test_label,p)
gb_score=np.sqrt(gb_score)
gb_score


# # Cross Val for Gb

# In[ ]:


gb=GradientBoostingRegressor()
score=cross_val_score(gb,train,train_label,cv=10,scoring='neg_mean_squared_error')
gb_score_cross=np.sqrt(-score)
np.mean(gb_score_cross),np.std(gb_score_cross)


# # Dataframe
# 

# In[ ]:


name=['Linear Regression','Linear Regression CV','Ridge Regression','Ridge Regression CV','Lasso Regression',
     'Lasso Regression CV','Elastic Net Regression','Elastic Net Regression CV','SGD Regression','SGD Regression CV',
     'SVM','SVM CV','Decision Tree','Decision Tree Regression','Random Forest','Random Forest CV','Ada Boost','Ada Boost CV',
     'Bagging','Bagging CV','Gradient Boost','Gradient Boost CV']


# In[ ]:


go=pd.DataFrame({'RMSE':[lr_score,lr_score_cross,r_score,r_score_cross,l_score,l_score_cross,en_score,en_score_cross,
                     sgd_score,sgd_score_cross,svm_score,svm_score_cross,dtr_score,dtr_score_cross,rf_score,rf_score_cross,
                     ada_score,ada_score_cross,br_score,br_score_cross,gb_score,gb_score_cross]},index=name)


# In[ ]:


go['RMSE']=go.applymap(lambda x: x.mean())


# In[ ]:


go.RMSE.sort_values()


# In[ ]:


fig=plt.figure(figsize=(10,6))
plt.scatter(np.arange(1,100,10),predict_r[0:100:10],color='blue')
plt.scatter(np.arange(1,100,10),p[0:100:10],color='yellow')
plt.scatter(np.arange(1,100,10),test_label[0:100:10],color='black')
plt.legend(['Random_Forest','Gradient Boosting','Real Value'])


# # It seems like Gradient Boosting doing better than others
# # So lets Do grid search on to tune hyper parameter

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


gb=GradientBoostingRegressor(max_depth=7,n_estimators=200,learning_rate=0.01)
param=[{'min_samples_split':[5,9,13],'max_leaf_nodes':[3,5,7,9],'max_features':[8,10,15,18]}]
gs=GridSearchCV(gb,param,cv=5,scoring='neg_mean_squared_error')
gs.fit(train,train_label)


# In[ ]:


gs.best_estimator_


# In[ ]:


gb=gs.best_estimator_


# # Now Train our model on Training Data

# In[ ]:


total=pd.concat([train,test],axis=0,ignore_index=True)


# In[ ]:


total_label=pd.concat([train_label,test_label],axis=0,ignore_index=True)


# In[ ]:


total_label.shape,total.shape


# In[ ]:


gb.fit(total,total_label)


# # TEST IMPORTING

# In[ ]:


test=pd.read_csv('../input/Test.csv')


# In[ ]:


test.shape


# # Test Data Preprocessing

# In[ ]:


attributes=['Item_MRP',
 'Outlet_Type',
 'Outlet_Size',
 'Outlet_Location_Type',
 'Outlet_Establishment_Year',
 'Outlet_Identifier',
 'Item_Type']


# In[ ]:


test=test[attributes]


# In[ ]:


test.shape


# In[ ]:


test.info()


# In[ ]:


test.Item_MRP=pd.cut(test.Item_MRP,bins=[25,75,140,205,270],labels=['a','b','c','d'],right=True)
test.Item_Type=test.Item_Type.astype('category')
test.Outlet_Size=test.Outlet_Size.astype('category')
test.Outlet_Identifier=test.Outlet_Identifier.astype('category')
test.Outlet_Establishment_Year=test.Outlet_Establishment_Year.astype('int64')
test.Outlet_Type=test.Outlet_Type.astype('category')
test.Outlet_Location_Type=test.Outlet_Location_Type.astype('category')


# In[ ]:


test.info()


# In[ ]:


test.Outlet_Establishment_Year=test.Outlet_Establishment_Year.astype('category')


# In[ ]:


test.info()


# In[ ]:


def func(x):
    if x.Outlet_Identifier == 'OUT010' :
        x.Outlet_Size == 'High'
    elif x.Outlet_Identifier == 'OUT045' :
        x.Outlet_Size == 'Medium'
    elif x.Outlet_Identifier == 'OUT017' :
        x.Outlet_Size == 'Medium'
    elif x.Outlet_Identifier == 'OUT013' :
        x.Outlet_Size == 'High'
    elif x.Outlet_Identifier == 'OUT046' :
        x.Outlet_Size == 'Small'
    elif x.Outlet_Identifier == 'OUT035' :
        x.Outlet_Size == 'Small'
    elif x.Outlet_Identifier == 'OUT019' :
        x.Outlet_Size == 'Small'
    elif x.Outlet_Identifier == 'OUT027' :
        x.Outlet_Size == 'Medium'
    elif x.Outlet_Identifier == 'OUT049' :
        x.Outlet_Size == 'Medium'
    elif x.Outlet_Identifier == 'OUT018' :
        x.Outlet_Size == 'Medium'
    return(x)


# In[ ]:


test.Outlet_Size=test.apply(func,axis=1)


# In[ ]:


test_dummy=pd.get_dummies(test.iloc[:,0:6])


# In[ ]:


test_dummy.head()


# # Now predict price of test data with our ML Model

# In[ ]:


predict=gb.predict(test_dummy)


# In[ ]:


predict.shape


# In[ ]:


sample=pd.read_csv('../input/Submission.csv')


# In[ ]:


sample.head()


# In[ ]:


del sample['Item_Outlet_Sales']


# In[ ]:


df=pd.DataFrame({'Item_Outlet_Sales':predict})
corr_ans=pd.concat([sample,df],axis=1)
del corr_ans['Unnamed: 0']
corr_ans


# In[ ]:


corr_ans.to_csv('correct.csv',index=None)


# # Feel free to give your suggestions and don't forget to give upvote :)
