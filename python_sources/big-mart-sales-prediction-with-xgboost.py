#!/usr/bin/env python
# coding: utf-8

# ## Big mart sales prediction--> Xgboost and random forest ensemble methods
# 1. EDA
# 2. Model Building
# 3. Prediction

# ### **Please give an upvote if you like my contribution and feel free to give suggestion in comment section**

# ## **Importing necessary libraries**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import cross_val_score
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

plt.rcParams["figure.figsize"]=(8,5)
plt.rcParams["font.size"]=10


# ### **Reading Train and Test data sets**
# 
# ---
# 
# 

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train=pd.read_csv("../input/big-mart-sales-prediction/Train.csv")
test=pd.read_csv("../input/big-mart-sales-prediction/Test.csv")
train.shape,test.shape


# **Concatenate both data sets into one to preprocess at once**

# In[ ]:


sales=pd.concat([train,test],ignore_index=True)
sales.shape


# ## **Data Exploration**

# In[ ]:


sales.head()


# In[ ]:





# In[ ]:


sales.info()


# In[ ]:


# statistics of numerical data
sales.describe().T


# In[ ]:


sales.isna().sum()


# 
#                missing values are present in 'Item_Weight' and 'Outlet_Size'. 
#                Ignore missing values in 'Item_Outlet_Sales'
# 

# In[ ]:


# Distribution of target values
sns.distplot(sales['Item_Outlet_Sales'],bins=20,rug=True,hist=True)
plt.show()


# In[ ]:


sales.Item_Outlet_Sales.mean()


# ## **Categorical data**

# In[ ]:


sales['Outlet_Size'].unique()


# In[ ]:


sales['Outlet_Identifier'].unique()


# In[ ]:


# OUTLET IDENTIFIER CATEGORIES
plt.figure(figsize=(10,5))
sales.groupby(['Outlet_Identifier']).size().plot(kind='bar')
plt.xticks(rotation=45,horizontalalignment='right')
plt.show()


# In[ ]:


sales['Item_Type'].unique()


# In[ ]:


# DIFFERENT ITEM TYPES PRESENT IN THE RETAIL STORE
plt.figure(figsize=(10,5))
sns.countplot('Item_Type',data=sales)
plt.xticks(rotation=45,horizontalalignment='right')
plt.show()


# In[ ]:


sales['Outlet_Type'].unique()


# In[ ]:


np.sort(sales['Outlet_Establishment_Year'].unique())


# In[ ]:


sales['Item_Fat_Content'].unique()


# In[ ]:


#PROPORTION OF DIFFERENT TYPES OF OUTLET SIZES
plt.figure(figsize=(5,5))
plt.pie(x=sales['Outlet_Size'].value_counts(),
        labels=["Medium","Small",'High'],
        autopct='%1.2f%%',
        explode=[0.04,0.01,0.02],shadow=True,)

plt.title("Types of Outlet Size",fontsize=16)
plt.show()


# In[ ]:


sales['Item_MRP'].plot(kind='box')
# no outliers present in Item_MRP


# ### **b. Bivariate Analysis**

# In[ ]:


plot_fig =sales.groupby('Outlet_Type').agg({'Item_Outlet_Sales':'mean'}).sort_values(by='Item_Outlet_Sales',ascending=False).reset_index()
plot_fig


# In[ ]:


plt.figure(figsize=(5,3))
fig=px.pie(plot_fig,names='Item_Outlet_Sales',values='Outlet_Type')
fig.show()


# In[ ]:


plt.pie(x=plot_fig['Item_Outlet_Sales'],
        labels=plot_fig['Outlet_Type'],
        autopct='%1.1f%%',
        explode=[0.04,0.01,0.02,0.02],shadow=True)
plt.title("Item Outlet Sales in Different Outlets Types",fontsize=16)
plt.show()


# In[ ]:


#checking relation between establishment year with outlet type
sales.groupby(['Outlet_Establishment_Year','Outlet_Type']).size()  


# In[ ]:


sales.groupby(['Outlet_Location_Type','Outlet_Size','Outlet_Type']).size().plot(kind='bar')
plt.xticks(rotation=45,horizontalalignment='right',fontsize=8)


# In[ ]:


sales[(sales['Outlet_Type']=='Supermarket Type1') & (sales['Outlet_Location_Type']=='Tier 1')]['Outlet_Size'].value_counts()
#supermarket type 1 and tier 1 have small and high outlet size


# In[ ]:


sales.groupby(['Outlet_Size','Outlet_Location_Type']).size()


# In[ ]:


# visualizing the above groupby command individually
sns.countplot('Outlet_Size',hue='Outlet_Location_Type',data=sales)
plt.legend(loc="upper right",shadow=True,fancybox=True)


# In[ ]:


sns.countplot("Outlet_Type",hue='Outlet_Location_Type',data=sales)
plt.legend(loc="upper center",shadow=True,fancybox=True)
plt.show()


# In[ ]:


# SCATTER PLOT OF TARGET VARIABLE WITH ITEM MRP
plt.scatter(sales['Item_MRP'],sales['Item_Outlet_Sales'])
plt.xlabel("Item_MRP")
plt.ylabel("Item_Outlet_Sales")
plt.show()


# In[ ]:


plt.scatter(sales['Item_Visibility'],sales['Item_Outlet_Sales'])
plt.xlabel("Item_Visibility")
plt.ylabel("Item_Outlet_Sales")
plt.show()


# In[ ]:


plt.scatter(sales['Outlet_Location_Type'],sales['Item_Outlet_Sales'])


# In[ ]:


sales[sales['Outlet_Location_Type']=='Tier 2']
# tier 2 has only small and nan values


# In[ ]:


sales[(sales['Outlet_Type']=='Grocery Store') & (sales['Outlet_Location_Type']=='Tier 1')]['Outlet_Size'].unique()
#small outlet_size is present only where location is Tier 2 and type is grocery


# In[ ]:


# OUTLET TYPE CATEGORIES
sns.countplot('Outlet_Type',data=sales)


# In[ ]:


pd.crosstab(sales["Outlet_Size"],sales["Outlet_Type"])


# In[ ]:


pd.crosstab(sales["Outlet_Size"],sales["Outlet_Location_Type"])


# Inference from above crosstabs--> 'Small' is present only in grocery and Tier 2.
# 
# 
# 1.   Impute small as outlet type where type is Grocery.
# 2.   Impute small as outlet type where location is Tier 2.
# 
# 
# 

# In[ ]:


sales.loc[sales['Outlet_Type']=='Grocery Store','Outlet_Size']='Small'


# In[ ]:


sales.loc[sales['Outlet_Location_Type']=='Tier 2','Outlet_Size']='Small'


# In[ ]:


sales['Outlet_Size'].isna().sum()
# now no missing values are there


# In[ ]:


sales['Outlet_Size'].value_counts().plot(kind='bar',color=['orange','lightgreen','skyblue'])
plt.xticks(rotation=45,horizontalalignment='center',fontsize=14)


# In[ ]:


# CHECHING RELATION BETWEEN OUTLET SIZE WITH OUTLET LOCATION TYPE

plt.figure(figsize=(5,3))
sns.catplot('Outlet_Size',col='Outlet_Location_Type',
            kind='count',col_order=['Tier 1','Tier 2','Tier 3'],
            order=['Small','Medium','High'],data=sales)
plt.show()


# ### Fill missing values in item weight with mean of their corresponding item identifier

# In[ ]:


sales['Item_Weight']=sales['Item_Weight'].fillna(sales.groupby('Item_Identifier')['Item_Weight'].transform('mean'))


# In[ ]:


sales.groupby('Item_Identifier')['Item_Weight'].transform('mean')


# In[ ]:


sales['Item_Weight'].isna().any()


# In[ ]:


#sales.loc[sales['Item_Type']=='Dairy']['Item_Weight'].mean()


# In[ ]:


sales.to_csv("modified_sales.csv")
s1=pd.read_csv("modified_sales.csv")
#s1=s1.reset_index()


# ## **Peprocessing item visibility**
#                  Item visibility contains 0 which in real has no meaning. 
#                  Item should be visible in Outlet.
# 
# 

# 

# In[ ]:


sales['Item_Visibility'].replace(0.0,value=np.nan,inplace=True)  # first replace 0 with nan values


# In[ ]:


# fill nan values with corresponding item identifier mean value
sales['Item_Visibility']=sales['Item_Visibility'].fillna(sales.groupby('Item_Identifier')['Item_Visibility'].transform('mean'))


# In[ ]:


sales['Item_Weight'].isna().sum()


# In[ ]:


sales.describe().T


# ### **Feature Engineering**

# In[ ]:


plt.figure(figsize=(7,5))
sns.countplot('Item_Fat_Content',data=sales)


# LF, low fat, Low Fat are same value and reg, Regular is another same type

# In[ ]:


sales['Item_Fat_Content'].replace({'LF':'Low Fat','reg':'Regular','low fat':'Low Fat'},inplace=True)


# In[ ]:


sales['Item_Fat_Content'].value_counts().plot(kind='bar',figsize=(7,4),color=['darkblue','orange'])
plt.xticks(rotation=0,horizontalalignment='center',fontsize=14)
plt.show()


# In[ ]:


sales['Outlet_Years']=2020-sales['Outlet_Establishment_Year']
sales['Outlet_Years'].value_counts()


# In[ ]:


sales['Item_Identifier']=sales['Item_Identifier'].str[0:2]
sales['Item_Identifier'].value_counts()


# In[ ]:


sales['Item_Identifier']=sales['Item_Identifier'].replace({'FD':'Food','DR':'Drinks','NC':'Non-Consumable'})
sales['Item_Identifier'].unique()


# In[ ]:


# mark non consumable as separate category in low fat
sales.loc[sales['Item_Identifier']=='Non-Consumable','Item_Fat_Content']='Non-Edible'


# In[ ]:


sns.countplot('Item_Identifier',hue="Item_Fat_Content",data=sales)
plt.legend(loc="upper center",shadow=True,fancybox=True)
plt.show()


# In[ ]:


sales['Item_Fat_Content'].value_counts().plot(kind='bar',color=['darkblue','orange','darkgreen'])
plt.xticks(rotation=0,horizontalalignment='center',fontsize=12)
plt.xlabel(" Fat Content in items",fontsize=16)
plt.ylabel("Count",fontsize=16)
plt.show()


# In[ ]:


sales.groupby(["Item_Identifier","Item_Type"])['Item_Identifier'].count().plot(kind='bar',figsize=(10,5))


# ### **Convert categorical columns into numerical values**
# 
# > **a.Label Encoding**
# 
# 

# In[ ]:


# Label encoding for variables which have internal dependency.

var_cat=['Outlet_Size','Outlet_Location_Type','Item_Type','Outlet_Years']
le=LabelEncoder()
for i in var_cat:
  sales[i]=le.fit_transform(sales[i])


# 
# 
# > **One Hot Encoding**
# 
# 

# In[ ]:


#one hot encoding for variables which have no internal dependency
sales=pd.get_dummies(sales,columns=['Item_Identifier','Outlet_Type','Item_Fat_Content','Outlet_Identifier'])


# In[ ]:


sales.head()


# In[ ]:


sales1=sales.copy()
sales.drop(columns=['Outlet_Establishment_Year',],inplace=True)


# **Normalize all columns except Target variable**

# In[ ]:


scale=MinMaxScaler()
col=list(sales.columns.drop('Item_Outlet_Sales'))
col
sales2=scale.fit_transform(sales)


# In[ ]:


sales=pd.DataFrame(sales2,columns=sales.columns)
sales["Item_Outlet_Sales"]=sales1['Item_Outlet_Sales']
sales.head()


# In[ ]:


#Variance
sales[col].var().sort_values(ascending=False)


# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(sales.corr(),annot=True,center=True,robust=True)
plt.show()
#checking correlation


# In[ ]:


sales.columns.drop(['Item_Fat_Content_Non-Edible','Outlet_Identifier_OUT018','Outlet_Identifier_OUT027','Outlet_Identifier_OUT019','Outlet_Identifier_OUT010',])


# ### After all, Splitting the train and test datasets into their original form as they were before

# In[ ]:


sales_train=sales.iloc[:8523,:]
sales_test=sales.iloc[8523:,:]
sales_test.drop(columns=['Item_Outlet_Sales'],inplace=True)
sales_train.shape,sales_test.shape


# the following list are the features

# In[ ]:


#features=['Item_Fat_Content','Outlet_Location_Type','Outlet_Years','Outlet_Size','Item_Identifier','Item_MRP','Item_Type','Item_Visibility','Outlet_Type']
features=['Item_Visibility', 'Item_Type', 'Item_MRP',
       'Outlet_Size', 'Outlet_Location_Type',
       'Outlet_Years', 'Item_Identifier_Drinks', 'Item_Identifier_Food',
       'Item_Identifier_Non-Consumable', 'Outlet_Type_Grocery Store',
       'Outlet_Type_Supermarket Type1', 'Outlet_Type_Supermarket Type2',
       'Outlet_Type_Supermarket Type3', 'Item_Fat_Content_Low Fat',
       'Item_Fat_Content_Regular', 'Outlet_Identifier_OUT013',
       'Outlet_Identifier_OUT017', 'Outlet_Identifier_OUT035',
       'Outlet_Identifier_OUT045', 'Outlet_Identifier_OUT046',
       'Outlet_Identifier_OUT049']


# In[ ]:


X=sales_train[features]
y=sales_train['Item_Outlet_Sales']
X_test_sales=sales_test[features]


# In[ ]:


X_train,X_val,y_train,y_val=train_test_split(X,y,random_state=42,test_size=0.20)
X_train.shape,X_val.shape,y_val.shape


# ## **Model Building**

# In[ ]:


# defining a function which calculates details of each algorithm
def model_details(model,alg):
    y_pred=model.predict(X_val)
    rmse=np.sqrt(mse(y_val,y_pred))
    acc=round(model.score(X_val,y_val)*100,2)
    cvs=cross_val_score(model,X_val,y_val,cv=5)
    mean=round(cvs.mean()*100,2)
    std=round(cvs.std()*2,2)
    print("Model Report")
    print('Accuracy of {}: {}%'.format(alg,acc),)
    print('RMSE Value: ',round(rmse,2))
    print('Cross Validation Score: Mean - {} | Std - {}'.format(mean,std))


# ## **Linear Regression**

# In[ ]:


reg=LinearRegression(normalize=True)


# In[ ]:


reg.fit(X_train,y_train)


# In[ ]:


model_details(reg,'LinearRegression')


# In[ ]:


ypred=reg.predict(X_val)
y_val,ypred[:6]


# ## **Decision Tree Algorithm**

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor(max_depth=15,min_samples_leaf=100,min_samples_split=5)


# In[ ]:


dtr.fit(X_train,y_train)


# In[ ]:


model_details(dtr,'DecisionTreeRegressor')


# In[ ]:


np.argsort(dtr.feature_importances_)


# In[ ]:


y_test_dtr=y_dtr=dtr.predict(sales_test[features])
y_test_dtr[:5]


# ## **Random Forest Regressor**

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


rf=RandomForestRegressor(n_estimators=200,min_samples_split=7,random_state=42,
                         max_depth=15)
# rf --> random forest


# In[ ]:


rf.fit(X_train,y_train)


# In[ ]:


model_details(rf,'RandomForestRegressor')


# ## **XGBoost Regressor**

# In[ ]:


import xgboost as xgb


# In[ ]:


dmat=xgb.DMatrix(data=sales_train[features],label=sales_train['Item_Outlet_Sales'])


# In[ ]:


xg_reg=xgb.XGBRegressor(colsample_bytree=0.3,learning_rate=0.1,max_depth=5,n_estimators=100,reg_alpha=0.75,reg_lambda=0.45,subsample=0.6,seed=42)


# In[ ]:


xg_reg.fit(X_train,y_train)


# In[ ]:


model_details(xg_reg,'XGBoost')


# In[ ]:


y=xg_reg.predict(sales_test[features])
y[:10]


# ## **Submission**

# In[ ]:


sub=pd.read_csv("../input/big-mart-sales-prediction/Submission.csv")

test_sales_pred=rf.predict(sales_test[features])

sub['Item_Outlet_Sales']=y
sub.head(10)


# In[ ]:


sub.to_csv("My_submission.csv")


# In[ ]:


#prediction of item sales on train dataset
y_test_xgb=xg_reg.predict(X_val)


# In[ ]:


pred_rf=pd.DataFrame(y_test_xgb,columns=['predicted_rf'])
true_values=list(y_val.values)
pred_rf['true_value']=true_values


# In[ ]:


# comparision between true and predicted value
comp=pred_rf
comp=comp.iloc[:15]
true_value=comp['true_value']
predicted_value=comp['predicted_rf']


# In[ ]:


plt.plot(true_value)
plt.plot(predicted_value)
plt.ylabel('Item Outlet Sales')
plt.legend(['Actual','Predicted'])
plt.title("Item Outlet Sales--> Actual vs Predicted",fontsize=16)
plt.show()


# ## **XGBoost model has highest accuracy of 61.14**
# 
# > Hence XGBoost will be used for prediction
# 
# 

# In[ ]:




