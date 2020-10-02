#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
import statsmodels.api as sm


# In[ ]:


train=pd.read_csv('../input/big-mart-sales-prediction/Train.csv')
train.head()


# ### CHAPTER 2 Explore the data

# In[ ]:


def data_modified(df,pred=None):
    obs=df.shape[0] # return the number of rows
    types=df.dtypes #return the type of data
    counts=df.apply(lambda x:x.count()) #store the number of not null values for eac column
    nulls=df.apply(lambda x:x.isnull().sum())#store the total number of nulls for each column
    distincts=df.apply(lambda x:x.unique().shape[0])#sotre the unique memeber of each column
    missing_ratio=round(df.isnull().sum()/obs*100,2) 
    skewness=round(df.skew(),2)
    kurtosis=round(df.kurt(),2)
    if pred is None:
        cols=['types','counts','nulls','distincts','missing_ratio','skewness','kurtosis']
        result=pd.concat([types,counts,nulls,distincts,missing_ratio,skewness,kurtosis],axis=1)
    else:
        corr=round(df.corr()[pred],2) #computing correlation between each column and SalePrice
        corr_name='corr '+pred
        result=pd.concat([types,counts,nulls,distincts,missing_ratio,skewness,kurtosis,corr],axis=1)
        cols=['types','counts','nulls','distincts','missing_ratio','skewness','kurtosis',corr_name]
    result.columns=cols
    result=result.sort_values(by=corr_name,ascending=False)
    result=result.reset_index()
    return result
    


# In[ ]:


data_modified(train,pred='Item_Outlet_Sales')


# Except for item visibility, none of the features are deemed to be heavily asymmetrical since thier values are below 1 or -1. Item visibility is the one we should be cautious of managing it where we have solid evidence to conclude that its distribution is rightly skewed with outliers appearing.  

# ### Chapter 3 Transforming Categorical Variable

# In[ ]:


object_column_number=[i for i,j in enumerate(train.dtypes) if j=='object']
object_columns=train.iloc[:,object_column_number]
#drop item_idtentifier which does not help in our predicting model
object_columns=object_columns.drop(['Item_Identifier','Outlet_Identifier'],axis=1)
object_columns_labels=object_columns.columns
table=[[object_columns_labels[i],list(object_columns.iloc[:,i].unique())] for i in range(5)]
table.insert(0,['columns','members'])


# In[ ]:


#prepare a table containing columns and their members
result=ff.create_table(table)
result.layout.annotations[5].font.size=10
result.layout.annotations[11].font.size=10
result.layout.update(width=1700)
result.show()


# ### Chapter 3 Filling Null Values 
# 
# We have checked that two features have missing observatiosn whose missing ratio is quite high. 
# Outlet_Size is a categorical data so that we can easily predict the missing values with machine learning algorithms for classification. Since data has a low number of dimension of faetures, I will use the most simple but powerful model,K-nearest neighbors. Even though some may argue that there is too much cost complexity associated with the higher dimensions, it does not assume anything about the data, this
# feature making the model itesel most prominent among others. 
# 

# In[ ]:


data=train.copy()
data.pivot_table('Item_Outlet_Sales',index='Outlet_Size',columns='Outlet_Location_Type',aggfunc='count',margins=True)


# From the table above, we have seen that
# 
# - The medium or big sized outlets are located in tier 3
# - Only small shops are avaialbe in tier 2
# - Either Small or Medium are operating but none of large shops are operating in tier 1
# 

# In[ ]:


data.pivot_table('Item_Outlet_Sales',index='Outlet_Type',columns='Outlet_Location_Type',aggfunc='count',margins=True)


# Compare two tables and there are couple of facts we could find are
# 
# - In the district of tier 1, Only either small or medium sized shops are currently running and alos belong to one of two kinds, grocery store and supermarket Type . 
# 
# 
# - All the stores in the distict 2 belong to the supermarket type 2. We have 1855 rows left with unknown size.
# 
# - It is the only in the tier 3 where consumer can visit all business types.With a simple calculation, we could understand that the      
#   grocery stores are currently not assigned thier size.
# 

# In[ ]:


data.groupby(['Outlet_Location_Type','Outlet_Type'])['Outlet_Size'].value_counts()


# ### 3.1 Outlet_Size
# #### 3.1.1 Grocery store in Tier 3

# In[ ]:


display(data.pivot_table(['Item_Outlet_Sales','Item_MRP'],index=[data.Outlet_Type,data.Outlet_Location_Type],aggfunc='mean'))
#to find out the unique member of business belong to Grocery Store currently operating in tier 1
data[(data.Outlet_Location_Type=='Tier 1')&(data.Outlet_Type=='Grocery Store')].loc[:,'Outlet_Size'].unique()


# It seems to be a safeguard for us to assume that all the grocery stores currently operating in district 3 is a small size one.  This is because we have only one comparison to estimate the size of them.

# In[ ]:


index=data[(data.Outlet_Size.isnull())&(data.Outlet_Type=='Grocery Store')].loc[:,'Outlet_Size'].index
#assign 'small'
data.loc[index,'Outlet_Size']='Small'


# #### 3.1.2 Missing Values in Tier 2

# In[ ]:


index=data[data.Outlet_Size.isnull()].loc[:,'Outlet_Size'].index
data.loc[index,'Outlet_Size']='Small'
data.isnull().any()


# ### 3.2 Outlet_Size
# 
# Outlet_Size is a continuous varialbe and we should take a somehow different approach to takle this problem from the previous case. This is a regression task and the model we will rely on is the forest regrssion, the most widespread model in this century. 
# 
# ### 3.2.1 Establish the base line
# 
# Before actually making and evaluating predictions, it is always a good practice to establish a baseline, a sensible measure we could beat with our predicting model. If our model can not improve upon the baseline, we must turn down our model and try a different model or conclude that the model is not suitable for our problem. The base line I will adopt is simply the avearage value of Item_Weight.
# 
# Steps to set up the base line
# 
# 1. Generate the random indicies of original data without missing values
# 2. Store the real value in the separate object named as _testvalue_
# 3. Assign the average value to the feature in the random incidies and store the values in the objected _predictVAlue_
# 4. Calculate the error rate by subtracting the test_value from predict_value (Make sure you put an absoulte value to each error)
# 5. Calcaute the average of error rates

# In[ ]:


base_data=data.drop(['Item_Identifier','Outlet_Identifier',],axis=1)
base_columns=base_data.columns


# In[ ]:


#Chosee every row with Item_Weight having some value 
base_data=base_data[base_data.Item_Weight.isnull()==False]
predict_value=base_data.Item_Weight.mean()
#Generate random indicies
random_rows=np.random.choice(base_data.index,np.int(base_data.index.shape[0]*0.25))
#Store up the true value of Item_Weight
test_value=base_data.loc[random_rows,'Item_Weight']
error=test_value.map(lambda x:np.abs(x-predict_value))
print('Average Baseline Error:{0} degrees'.format(round(np.mean(error),2)))


# Now we have a guide to compare! If we can not achieve at least below the degress, then we need to rethink our approach

# ###  3.2.2 Random Forest
# 
# In prediction of targe values with the radom forest, the most critical choice we should bear in mind is how many decision trees we need to
# take in our model.From many text books, it is strongly recommended that over 2000 decision trees and the minimum number of samples required to splits an internal node is number of features/3 for regression model in order to meet the strong law of large number. 

# In[ ]:



data.Item_Fat_Content=data.Item_Fat_Content.map(lambda x: 'low fat' if x=='Low Fat'and 'LF'and 'low fat' else 'regular')


# In[ ]:



# mapping Item_Fat_Content to either 1 or 2
ce_ord=ce.OrdinalEncoder()
data.Item_Fat_Content=ce_ord.fit_transform(data.Item_Fat_Content)

#maping Item_Type to ordinal category from 1 to 16
ce_ord=ce.OrdinalEncoder()
data.Item_Type=ce_ord.fit_transform(data.Item_Type)

#Outlet location

data.Outlet_Location_Type=data.Outlet_Location_Type.map(lambda x:x[-1]).astype(int)

#Outlet Type
ce_ord=ce.OrdinalEncoder()
data.Outlet_Type=ce_ord.fit_transform(data.Outlet_Type)

#Outlet Size
data.Outlet_Size=data.Outlet_Size.map(lambda x: 0 if x=='Small' else 1 if x=='Medium' else  2 if x=='High' else x )




# In[ ]:


data_null_free=data[data.Item_Weight.isnull()==False]
data_null=data[data.Item_Weight.isnull()==True]


# In[ ]:


X=data_null_free.drop(['Item_Identifier','Item_Weight','Outlet_Identifier'],axis=1)
y=data_null_free['Item_Weight']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
#Instantiate model with 2000 decision trees 
rf=RandomForestRegressor(n_estimators=2000,random_state=42,min_samples_split=3)
#train the model on training data
rf.fit(X_train,y_train)


# Now,our model has been trained to learn the relations between the features and the targets. The next step is to find out how accruate our model is! 

# In[ ]:


predictions=rf.predict(X_test)
errors=abs(predictions-y_test)
print('Mean Absoulte Error:{0} degrees'.format(round(np.mean(errors),2)))


# Our average estimate is off by about 1 degree. That is more than a 1 degree improvement over the baseline. To put our predictions in perspective, we can calucate the accuracy using the mean average error subtracted from 100 %.

# In[ ]:


#Calculate mean absoulte the percentage error
error_percent=100*errors/y_test
accuracy=100-np.mean(error_percent)
print("Accuracy: {0}%".format(round(accuracy,2)))


# In[ ]:


importance=list(rf.feature_importances_)
features=X.columns
features_importances=[(features,round(importances*100,2)) for features,importances in zip(features,importance)]
features_importances=sorted(features_importances,key=lambda x:x[1],reverse=True)
[print('Variable: {:30} Importance: {}'.format(*pair)) for pair in features_importances];


# In[ ]:


sns.set()
fig,ax=plt.subplots(figsize=(11,5))
plt.bar(features,importance,alpha=0.7,color='coral')
ax.set_xticklabels(features,rotation=45)
ax.set_xlabel('features',fontsize=15)
ax.set_ylabel('importance(percentage)',fontsize=15)
ax.set_title('Features Importances',fontsize=20)


# In[ ]:


data_null=data_null.drop(['Item_Identifier','Item_Weight','Outlet_Identifier'],axis=1)
#store the indicies with missing values
data_null_index=data_null.index
#predict the values with our random forest model
predict=rf.predict(data_null)
#assign them to the null values 
data.iloc[data_null_index,1]=predict


# In[ ]:


# double check wether we have removed all the missing values in every feature
data.isnull().any()


# In[ ]:


# Assigin the finding results to original data(train)
train.Outlet_Size=data.Outlet_Size
train.Outlet_Size=train.Outlet_Size.map(lambda x: 'small' if x==0 else 'medium' if x==1 else 'high')
train.Outlet_Size=train.Outlet_Size.astype('category')
train.Item_Weight=data.Item_Weight


# ### Chapter 4 Detailed Investigation on Features

# In[ ]:


#Finding out the correlation with Item_Outlet_Sales
result=data.corr()['Item_Outlet_Sales']
print('Correlation with Item_Outlet_Sales')
print('-'*100)
#Sort the results in a descedning order
result=result.sort_values(ascending=False)
display(result)
result_columns=result.index
data=data.loc[:,result_columns]


# ### 4.1 MRP 
# 
# We are not given any information of what MRP stands for. However, one thing for sure is that there is an incresing trend of Sales as MRP rises. I wish the description of it were provied for the better analysis. One outstanding feature we could curiously see is that in some intervals observations are absent, thereby being grouped according to the specific range of MRP.  

# In[ ]:


sns.set()
fig,axes=plt.subplots(figsize=(15,10))
sns.scatterplot(x=data.Item_MRP,y=data.iloc[:,0],ax=axes,hue=data.Outlet_Size,palette='Spectral')


# ### 4.2 Item_Weight and Item_Visibility
# 
# Normally, a lower correlation with the target variable is not into our consideration and we should drop the features before constructing a predicting model. However, before permanently discarding them, I always combine two varaibles to see if we can get some benefits from this interaction. 
# 
# I guess a sound guess that the heavier items become, the greater visible they are to consumers. With common sense, the heavier items should have a bigger size proportionally. Let's take some experiments in combing two varialbes by either multiplication or division.

# In[ ]:


fig,ax=plt.subplots(3,1,figsize=(15,20))
weight_vis=data.Item_Weight*data.Item_Visibility
weight_by_vis=data.Item_Weight/data.Item_Visibility
sns.scatterplot(x=data.Item_Visibility,y=data.iloc[:,0],ax=ax[0],hue=data.Outlet_Location_Type,palette='Spectral')
ax[0].set_title('Correaltion with Sales: {0}'.format(data.Item_Outlet_Sales.corr(data.Item_Visibility)),fontsize=14)
sns.scatterplot(x=weight_vis,y=data.Item_Outlet_Sales,hue=data.Outlet_Size,palette='Spectral',ax=ax[1])
ax[1].set_title('Correaltion with Sales: {0}'.format(data.Item_Outlet_Sales.corr(weight_vis)),fontsize=14)
sns.scatterplot(x=weight_by_vis,y=data.iloc[:,0],hue=data.Outlet_Size,palette='Spectral',ax=ax[2])


# The interaction between the two features does not help to improve the correlation. But, the scatter plots give some clues of which points are to be removed. In fact, this has correctly identified outliers a single feature failed to detect.

# ### Removing Outliers

# In[ ]:


#remove the outliers
data=data[data.Item_Weight*data.Item_Visibility<4.7]
data=data[data.Item_Weight/data.Item_Visibility<2500]
data=data[data.Item_Outlet_Sales<12000]


# ### 4.3 Item Weight and Fat content
# 
# Take another interesting experiment before building a predicting model. Why not we think there is a close relation between Item weight and fat content?  I have done the same procedure I did in the previous chapter. 
# The result is still disappointing even though a slight improvement over the correlation exists.  We are not seeing any outliers since we have removed a majority of them in the previous steps. 

# In[ ]:


fig,ax=plt.subplots(3,1,figsize=(15,20))
weight_fat=data.Item_Weight*data.Item_Fat_Content
weight_by_fat=data.Item_Weight/data.Item_Fat_Content
sns.scatterplot(x=data.Item_Weight,y=data.iloc[:,0],ax=ax[0],hue=data.Outlet_Location_Type,palette='Spectral')
ax[0].set_title('Correlation between Item_Weight and Sale revenue',fontsize=20)
ax[0].text(x=4,y=11000,s='Correaltion:{0}'.format(data.Item_Outlet_Sales.corr(data.Item_Weight)),fontsize=14)
sns.scatterplot(x=weight_fat,y=data.iloc[:,0],ax=ax[1],hue=data.Outlet_Location_Type,palette='Spectral')
ax[1].set_title('Correlation between Weight_fat and Sale revenue',fontsize=20)
ax[1].text(x=4,y=11000,s='Correaltion:{0}'.format(data.Item_Outlet_Sales.corr(weight_fat)),fontsize=14)
data.Item_Outlet_Sales.corr(weight_vis)
sns.scatterplot(x=weight_by_fat,y=data.iloc[:,0],ax=ax[2],hue=data.Outlet_Location_Type,palette='Spectral')
ax[2].set_title('Correlation between Weight_by_fat and Sale revenue',fontsize=20)
ax[2].text(x=2.5,y=11000,s='Correaltion:{0}'.format(data.Item_Outlet_Sales.corr(weight_by_fat)),fontsize=14)


# ### 4.4 Other Features

# In[ ]:


fig,axes=plt.subplots(2,2,figsize=(15,12))
sns.boxplot(x='Outlet_Establishment_Year',y='Item_Outlet_Sales',ax=axes[0,0],data=data)
sns.boxplot(x='Outlet_Size',y='Item_Outlet_Sales',ax=axes[0,1],data=data)
sns.boxplot(x='Outlet_Location_Type',y='Item_Outlet_Sales',ax=axes[1,0],data=data)
sns.boxplot(x='Outlet_Type',y='Item_Outlet_Sales',ax=axes[1,1],data=data)


# ### 4.4 Conclusion
# Two continuous columns demonstrate a much lower correlation with the target variable than what is normally accepted. Therefore, I decied to drop these two continuous features. 

# In[ ]:


data=data.drop(['Item_Weight','Item_Visibility'],axis=1)
y=data.Item_Outlet_Sales
X=data.iloc[:,1:]
X.iloc[:,1:6]=X.iloc[:,1:6].astype('category')


# ## 5. Constructing Model

# In[ ]:


# Preparing the data sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# ### 5.1 Simple Linear Regression (Base Line)

# In[ ]:


#Train the model 
sl=LinearRegression()
sl.fit(X_train,y_train)


# In[ ]:


#predict the taraget variable based on X_test
predict_sl=sl.predict(X_test)
#Calculate Mean Squared Error
mse=np.mean((predict_sl-y_test)**2)
#Score
sl_score=np.sqrt(mse)
print('Score of Simple regrssion model : {0}'.format(sl_score))


# ### 5.2 Rigid Regression

# In[ ]:


r=Ridge(alpha=0.5,solver='cholesky')
r.fit(X_train,y_train)
predict_r=r.predict(X_test)
mse=np.mean((predict_r-y_test)**2)
r_score=np.sqrt(mse)
r_score
print('Score of Rigid Regression : {0}'.format(sl_score))


# ### 5.3 Lasso

# In[ ]:


l=Lasso(alpha=0.01)
l.fit(X_train,y_train)
predict_r=r.predict(X_test)
mse=np.mean((predict_r-y_test)**2)
l_score=np.sqrt(mse)
l_score
print('Score of Lasso : {0}'.format(l_score))


# ### 5.4 Elastic Net

# In[ ]:


en=ElasticNet(alpha=0.01)
en.fit(X_train,y_train)
predict_r=en.predict(X_test)
mse=np.mean((predict_r-y_test)**2)
l_score=np.sqrt(mse)
l_score
print('Score of Elastic Net: {0}'.format(l_score))


# ### 5.5 Support Vector machine

# In[ ]:


svm=SVR(epsilon=15,kernel='linear')
svm.fit(X_train,y_train)
predict_r=svm.predict(X_test)
mse=np.mean((predict_r-y_test)**2)
l_score=np.sqrt(mse)
l_score
print('Score of Support Vector machine: {0}'.format(l_score))


# ### 5.6 Decision Tree

# In[ ]:


dtr=DecisionTreeRegressor()
dtr.fit(X_train,y_train)
predict_r=dtr.predict(X_test)
mse=np.mean((predict_r-y_test)**2)
l_score=np.sqrt(mse)
l_score


# ### 5.7 Comments 
# 
# No alternative models could beat the base model(multi regression model). Therefore,we will look at model to see if the further evelation steps are needed. Unfortunately, the adjusted R-squareds is really lower than what I expcet it to be. Thefore, I close my remark saying 
# that the data iteself is not sutiable for predicing the Item_Outlet Sales

# In[ ]:


y=train.Item_Outlet_Sales
X=train.iloc[:,[2,4,5,7,8,9,10]]
columns=['Item_MRP',
         'Item_Fat_Content',
         'Item_Type',
         'Outlet_Size',
         'Outlet_Location_Type',
         'Outlet_Type',
        'Outlet_Establishment_Year']
#rearrange the columns
X=pd.DataFrame(X,columns=columns)
#All the object varibles are converted into categories one
X.iloc[:,1:]=X.iloc[:,1:7].astype('category')
X=pd.get_dummies(X)
#Create a OLS model
model=sm.OLS(y,X)
results=model.fit()


# In[ ]:


results.summary()


# In[ ]:




