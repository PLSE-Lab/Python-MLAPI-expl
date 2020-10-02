#!/usr/bin/env python
# coding: utf-8

# 1. Relationship with Target Data
#     
#     In this Section we will discuss the relation between the target data which is "SalePrice" and *all the numerical features*, and see which one effect the target most.

# In[ ]:


#importing the libaries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


#     After that we will import Just the train file first for testing the regression and leave the test file until the end of the algorithm to apply.

# In[ ]:


data=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
data.head(10)
#Everything is good for now


#     Before we go into this section we should study the target data "SalePrice"
#     

# In[ ]:


data['SalePrice'].describe()


# In[ ]:


sns.boxplot(data['SalePrice'],linewidth=1.5)


# In[ ]:


sns.distplot(data['SalePrice'],color='red')


#     Now we found some important point that our data have a positive skewness.that explain the value of the mean.
# 

# Let's move to see the correlation matrix

# In[ ]:


corrmap = data.corr()
ax = plt.subplots(figsize=(19, 16))
sns.heatmap(corrmap, vmax=.8, square=True,cmap='coolwarm')


# In[ ]:


#here we are going to choose the most correlated features to target data 
corr_cols=corrmap.nlargest(9,'SalePrice')['SalePrice'].index
ax = plt.subplots(figsize=(9, 7))
sns.heatmap(np.corrcoef(data[corr_cols].values.T),cbar=True,cmap='coolwarm',
           annot=True,square=True,fmt='.2f',annot_kws={'size':9},
            xticklabels=corr_cols.values,yticklabels=corr_cols.values)
#here we choose just best8 features the other below 0.5 so it better focus on the important one 


# In[ ]:


LAvsPrice=pd.concat([data['SalePrice'],data['GrLivArea']],axis=1)
sns.regplot(x='GrLivArea',y='SalePrice',data=LAvsPrice)


# We notice there is two point are far from regression line and these two may effect the study and make mislead to the predicted data so the solution here is to remove them.

# In[ ]:


data.sort_values(by='GrLivArea',ascending = False)[:2]
data=data.drop(data[data['Id']==1299].index)
data=data.drop(data[data['Id']==524].index)
print("done!!")


# In[ ]:


LAvsPrice=pd.concat([data['SalePrice'],data['GrLivArea']],axis=1)
sns.regplot(x='GrLivArea',y='SalePrice',data=LAvsPrice)


# Now everything looks good

# 2. Testing to get best possible accuracy 
# 
#     In this section we are going to create the test and train data from the train file and test the data before moving to the test file.

# In[ ]:


X=data.filter(['OverallQual', 'GrLivArea', 'GarageCars',
      'TotalBsmtSF','1stFlrSF', 'FullBath','TotRmsAbvGrd'],axis=1)
#notice here that we didnt include 'GarageArea' 'cause obviously is the same as 'GarageCars'
y=data.filter(['SalePrice'],axis=1)

X.head(10)
#y.head(10)


# In[ ]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)


# Now it's time to split the data we will make 10% for test the the rest for the train 

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=40)

#X_train
#X_test
#y_train
#y_test


# In[ ]:


y_train


# We are going to use The Gradient Boosting Regressor but before we need to know what the best parameter to use also we are going to need GridSearchCV for this job.

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

SelectedModel = GradientBoostingRegressor(learning_rate=0.05, max_depth=2, 
                                        min_samples_leaf=14,
                                        min_samples_split=50, n_estimators=3000,
                                        random_state=40)
SelectedParameters = {'loss':('ls','huber','lad'
                            ,'quantile'),'max_features':('auto','sqrt','log2')}


GridSearchModel = GridSearchCV(SelectedModel,SelectedParameters,return_train_score=True)
GridSearchModel.fit(X_train, y_train)


# In[ ]:


sorted(GridSearchModel.cv_results_.keys())
GridSearchResults = pd.DataFrame(GridSearchModel.cv_results_)[['mean_test_score', 'std_test_score'
                                                               , 'params' , 'rank_test_score' , 'mean_fit_time']]

print('All Results are :\n', GridSearchResults )
print('Best Score is :', GridSearchModel.best_score_)
print('Best Parameters are :', GridSearchModel.best_params_)#this is what we need 
print('Best Estimator is :', GridSearchModel.best_estimator_)


# To the next Step !!

# In[ ]:


#accourding to GridSearchCV the best parametre is {'loss': 'huber', 'max_features': 'sqrt'} 
GBR = GradientBoostingRegressor(learning_rate=0.05, loss='huber', max_depth=2, 
                                       max_features='sqrt', min_samples_leaf=14,
                                       min_samples_split=50, n_estimators=3000,
                                       random_state=42)
GBR.fit(X_train, y_train)
print("done!!!")


# In[ ]:


print("Train Score", GBR.score(X_train, y_train))
print("Test Score", GBR.score(X_test, y_test))


# In[ ]:


#predict the test data 
y_pred = GBR.predict(X_test)
print("done again !!")


# In[ ]:


from sklearn.metrics import mean_absolute_error
#Calculating Mean Absolute Error
MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average')
print('Mean Absolute Error Value is : ', MAEValue)


# Now let move the last step which is importing the test data and apply the GBR algorithm

# In[ ]:


RealData=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
RealData.head(10)


# In[ ]:


X1=RealData.filter(['OverallQual', 'GrLivArea', 'GarageCars',
      'TotalBsmtSF','1stFlrSF', 'FullBath','TotRmsAbvGrd'],axis=1)

X1.head(10)


# In[ ]:


#there is some values are null so we need to get rid of them 
X1=X1.fillna(0)
print("coool!!")


# In[ ]:


#repeat the same step we did with the train file ...
sc0 = StandardScaler()
X1 = sc0.fit_transform(X1)

#now let create the file and save our prediction 

Doc=pd.DataFrame()
Doc['Id']=RealData['Id']
Doc['SalePrice']=np.round(GBR.predict(X1),2)
print(Doc['SalePrice'].head(10))


# The Last Step is to save the file

# In[ ]:


Doc.to_csv('SalePrice_submission.csv',index=False)
print('great !!! ')

