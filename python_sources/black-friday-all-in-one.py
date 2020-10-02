#!/usr/bin/env python
# coding: utf-8

# # Begineer Pack + Intermediate (ALL IN ONE)

# ### Objective : 
# Main obejctive behind this notebook is to give an idea along with workflow of Machine Learning Processes.
# 
# Starting from **Getting data informaion to Exploratory Data Analysis, Data Manipulation, Building and then Validation of Model.**
# 
# I am trying to keep it as **simple** as i can so that newbie can also understand the workflow.
# 
# If you learn anything useful from this notebook then **Give Upvote :)
# ** All **QUESTION/DOUBTS/SUGGESTIONS** are welcomed here
# 

# ## Contents of the Notebook:
# 
# #### Part1: Exploratory Data Analysis(EDA):
# a) Analysis of the features.
# 
# b) Finding any relations or trends considering multiple features.
# 
# #### Part2: Data Cleaning:
# a) Handling Missing Values
# 
# b) Outliers handling Using Statistics
# 
# c) Correlation
# 
# d) Converting features into suitable form
# 
# e) Dimension Reduction
#  
# 
# #### Part3: Predictive Modeling
# a) Fitting ML models
# 
# b) Cross Validation
# 
# c) Model Comparison 
# 
# d) Ensemble model (Will update in enxt Version)

# ## Part1: Exploratory Data Analysis(EDA)

# ### Attribute Information:
#     1. User ID                          2. Product ID
#     3. Gender                           4. Age
#     5. Occupation                       6. City Category
#     7. Stayed in current city           8. Marital Status
#     9. Product Category 1               10. Product Category 2
#     11.Product Category 3               12. Purchase

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
color=sns.color_palette()


# In[ ]:


data=pd.read_csv('../input/BlackFriday.csv')
data.head()


# #### Let's check Target columns (Purchase) detail ?

# In[ ]:


fig = plt.figure(figsize=(8,5))
sns.distplot(data.Purchase)
plt.title('Purchase Distribution')


# In[ ]:


print("skew",data.Purchase.skew(),"kurt",data.Purchase.kurt())


# From above distribution graph we can conclude that Purchase ranges from around 100+ to 24000+ and distribution seems to be very little positive skewed. We need to dig more to get info related to distibution.

# Let's explore more feature to get more insight from dataset

# In[ ]:


data.describe(include = ['object', 'integer', 'float'])


# With above description we can say : Total entries present in dataset is 537577. Other than that we can conclude Uniquely identified Gender is 2, Uniquely identified Age is 7, City_Category is 3, Stayed_In_Current_City is 5.

# There are very small small things which can be infer from above description

# #### Let's check for Missing Values

# In[ ]:


sns.heatmap(data.isnull(), cmap= 'Blues')


# **Product_Category_2 and Product_Category_3** contain missing value. Most of the values are missing specially from **Product_Category_3**. We have to take care these columns.

# ## 1. Feature analysis

# ### a)->User_ID

# In[ ]:


data.User_ID.plot.hist(bins=70)


# User_ID columns seems to be important feature. We can't use it as raw. But we can extract some information.

# **What if we get number of times user make purchase using Columns User_ID.** 

# **We can use this attribute to train our model**. Lets start 

# In[ ]:


df = pd.DataFrame(data = data.User_ID.value_counts())
df=df.reset_index()
df.columns = ['users', 'Purchase_History']
data = data.merge(df, left_on = 'User_ID', right_on = 'users')


# In[ ]:


data.head()


# Here we have added one more column which show's **How many times order placed using same User_ID**.
# Now we can delete **User_ID** as we extracted information

# In[ ]:


del data['User_ID']
del data['users']


# Since Product_ID seems to be no use therefore deleting it here itself

# In[ ]:


del data['Product_ID']


# Let's explore other feature

# ### b)-> Gender

# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(14,5))
sns.boxplot(x='Gender',y='Purchase',data=data,palette='Set3',ax=ax[0])
ax[0].set_title("F -> Female , M -> Male",size=12)
data.Gender.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',shadow=True, explode=[0.1,0],cmap='Blues')
ax[1].set_title("Total")


# Male are more active in shopping than Female. **This is interesting**. With plot we can expect median of around 9000. 

# Little bit of outliers we can expect that is above 23000. Will dig more let's move to next feature 

# ### c) Age

# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(14,5))
sns.boxplot(x='Age',data=data,y='Purchase',palette='Set2',ax=ax[0])
ax[0].set_title('Purchase v/s Age',size=12)
data.Age.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',shadow=True,cmap='Oranges')


# Here also we can easily see lots of outliers are present. With pie plot we can see mostly youngster are pretty active in online purchase. 

# let's move to next feature

# ### d) Occupation

# In[ ]:


fig,ax=plt.subplots(2,1,figsize=(15,12))
sns.boxplot(x='Occupation',data=data,y='Purchase',palette='Set1',ax=ax[0])
ax[0].set_title('Purchase v/s Occupation')
data.Occupation.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',shadow=True,cmap='Reds')


# Most of the people who are in occupation 17 spend more than others. Most of people belong to occupation 4.  

# Here also we are seeing lots of outliers. Will treat all in **Data Cleaning** section later in notebook

# ### e)  City_Category

# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(14,5))
sns.boxplot(x='City_Category',data=data,y='Purchase',palette='Set3',ax=ax[0])
ax[0].set_title('Purchase v/s City Category', size=12)
data.City_Category.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',shadow=True,cmap='Greens', 
                                           explode=[0.05,0.05,0.05])


# People from City C spend more than other two cities. People from City B are more active than other two cities

# Now next

# ### f) Stayed_In_Current_City_Years

# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(14,5))
sns.boxplot(x='Stay_In_Current_City_Years',data=data,y='Purchase',palette='Set3',ax=ax[0])
ax[0].set_title('Purchase v/s No. of years stayed')
data.Stay_In_Current_City_Years.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',shadow=True,cmap='Greys')


# **Stayed_In_Current_City_Years feature seems to identical for all the years**. May be this feature don't have much impact on purchase. 

# ### g) Marital_status

# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(14,5))
sns.boxplot(x='Marital_Status',data=data,y='Purchase',palette='Set3',ax=ax[0])
data.Marital_Status.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',explode=[0.05,0.05],shadow=True, cmap='Blues')


# **Married people spend more than unmarried ones.** This isn't surprise me :P

# This insight will be very usefull for our model

# ### h) Product_Category_1, Product_Category_2, Product_Category_3

# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(15,6))
sns.boxplot(y='Purchase',data=data,x='Product_Category_1',palette='Set2',ax=ax[0])
sns.countplot(x= 'Product_Category_1',ax=ax[1], data= data)


# Most of the purchase made from Product_category 5. Whereas product Category 17 contain least number.

# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(15,6))
sns.boxplot(y='Purchase',data=data,x='Product_Category_2',palette='Set2',ax=ax[0])
sns.countplot(x= 'Product_Category_2',ax=ax[1], data= data)


# Product Category 7  have very less entries.

# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(15,6))
sns.boxplot(y='Purchase',data=data,x='Product_Category_3',palette='Set2',ax=ax[0])
sns.countplot(x= 'Product_Category_3',ax=ax[1], data= data)


# Product_Category_3 contains most of highest numbers of missing values. Filling this feature will not be easy therefore let's delete it.

# ## 2. Data Cleaning

# ### a) Handling missing values

# Since we know Product_Category_2 and Produuct_Category_3 contain missing values. therefore we will delete Produuct_Category_3 as most of the rows are filled with missing values. For Produuct_Category_2 we will fill all category as 0 so that other feature will be utilize to make prediction

# In[ ]:


del data['Product_Category_3']
data.Product_Category_2.fillna(0, inplace=True)


# ### b) Handling Outliers

# #### As we seen in EDA part that Purchase (target) column contains lots of outliers.

# #### Let's use some statistic to handle Outliers

# ![image.png](attachment:image.png)

# Will set upper and lower limit on Purchase column. And any entries outside bounded region wil be deleted

# In[ ]:


def outliers(df):
    q1= pd.DataFrame(df.quantile(0.25))
    q3= pd.DataFrame(df.quantile(0.75))
    iqr = pd.DataFrame(q3[0.75] - q1[0.25])
    iqr['lower'] = q1[0.25] - 1.5 * iqr [0]
    iqr['upper'] = q3[0.75] + 1.5 * iqr [0]
    return(np.where(df > iqr['upper']) or (df < iqr['lower']))


# In[ ]:


x = outliers(pd.DataFrame(data.Purchase))
data = data.drop(x[0])


# #### Let's check some of the plot after removing outliers

# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(14,5))
sns.boxplot(x='Gender',y='Purchase',data=data,palette='Set3',ax=ax[0])
ax[0].set_title("F -> Female , M -> Male",size=12)
sns.boxplot(x='Age',data=data,y='Purchase',palette='Set2',ax=ax[1])
ax[1].set_title('Purchase v/s Age',size=12)


# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(14,5))
sns.boxplot(x='City_Category',data=data,y='Purchase',palette='Set1',ax=ax[0])
sns.boxplot(x='Stay_In_Current_City_Years',data=data,y='Purchase',palette='Set2',ax=ax[1])


# #### Now we can see the difference

# Initially there was lots of Outliers but now we can see most of the outliers are deleted from dataset. Also conserved **Variance of about ~ 95%**

# ### c) Correlation between different features

# In[ ]:


fig=plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), annot= True, cmap='Blues')


# Since we know that all predictors are Categorical variables. therefore correlation can't determine relationship between predictor and target.

# **Will add Chi-squared test** in next version to improve this part

# In[ ]:


data.info()


# ### d) Feature conversion

# In[ ]:


data.Product_Category_1=data.Product_Category_1.astype('category')
data.Marital_Status=data.Marital_Status.astype('category')
data.Occupation=data.Occupation.astype('category')
data.Product_Category_2=data.Product_Category_2.astype('category')


# In[ ]:


data_label=data['Purchase']
del data['Purchase']
data_label=pd.DataFrame(data_label)


# In[ ]:


data=pd.get_dummies(data,drop_first=True)
data.head()


# #### Normalization (To bring each feature at same scale)

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
data_scaled=MinMaxScaler().fit_transform(data)
data_scaled=pd.DataFrame(data=data_scaled, columns=data.columns)


# In[ ]:


data_scaled.head()


# In[ ]:


data_scaled.shape


# ### e) Dimension Reduction

# Let''s reduce dimension of data so that it will be easy for ML to process. It may result in reduction of accuracy but it will result in fasten training process

# **NOTE: Normalisation is should be done before dimension reduction**. As we already done Normalisation in our last section so we can proceed further

# Main problem is **number of principal components**. 

# Let's fit and see output

# In[ ]:


from sklearn.decomposition import PCA
variance_ratio = []
for i in range(5,65,5):
    pca=PCA(n_components = i)
    pca.fit_transform(data_scaled)
    variance_ratio = np.append(variance_ratio,np.sum(pca.explained_variance_ratio_))


# In[ ]:


df =pd.Series(data = variance_ratio, index = range(5,65,5))
df.plot.bar(figsize=(8,6))


# Here we can see if we set compenents near ~ 30-40 then we will be able to capture around 90~95% of variance.

# It will be easy for our model to quickly do prediction. So lets **take dimension as 40**

# In[ ]:


pca=PCA(n_components = 40, whiten = False, random_state=876)
data_scaled = pd.DataFrame(pca.fit_transform(data_scaled), index= data_scaled.index)


# In[ ]:


data_scaled.head()


# ## Part3: Predictive Modeling

# ### a) Fitting ML models

# #### Splittting data into test and train set

# In[ ]:


from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest = train_test_split(data_scaled, data_label, test_size=0.30,random_state=54368)


# ### Importing ML libraries

# In[ ]:


from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor


# ### Evalutation metrics to check model performance

# In[ ]:


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


# ### Cross validation helper function

# In[ ]:


def CrossVal(dataX,dataY,mode,cv=3):
    score=cross_val_score(mode,dataX , dataY, cv=cv, scoring='neg_mean_squared_error')
    return(np.sqrt(np.mean((-score))))


# ### a)Stochastic Gradient Descent 

# In[ ]:


sgd=SGDRegressor(random_state=324,penalty= "l1", alpha=0.4)
score_sgd=CrossVal(Xtrain,Ytrain,sgd)
print("RMSE is : ",score_sgd)


# ### b) Linear Regression

# In[ ]:


lr=LinearRegression(n_jobs=-1)
score_lr=CrossVal(Xtrain,Ytrain,lr)
print("RMSE is : ",score_sgd)


# ### c) Decision Tree Classifier

# In[ ]:


dtc=DecisionTreeRegressor(random_state=42234)
score_dtc=CrossVal(Xtrain,Ytrain,dtc)
print("RMSE is : ",score_dtc)


# ### d) Random Forest Classifier

# In[ ]:


rf=RandomForestRegressor(n_estimators=10, n_jobs=-1, random_state=487987)
score_rf= CrossVal(Xtrain,Ytrain,rf)
print('RMSE is:',score_rf)


# ### e) Extra Trees Classifier

# In[ ]:


etc=ExtraTreesRegressor(n_estimators=10, n_jobs=-1, random_state=3141)
score_etc= CrossVal(Xtrain,Ytrain,etc)
print('RMSE is:',score_etc)


# ### Model accuracy plot

# In[ ]:


model_accuracy = pd.Series(data=[score_sgd, score_lr, score_dtc, score_rf, score_etc], 
                           index=['Stochastic GD','linear Regression','decision tree', 'Random Forest',
                            'Extra Tree'])
fig= plt.figure(figsize=(8,8))
model_accuracy.sort_values(ascending= False).plot.barh()
plt.title('MODEL RMSE SCORE')


# Without any Hyper parameter tuning, Random forest is doing better than other models.

# **Will add bagging , boosting , stacking , Hyper param tuning in next update. Keep checking**

# ### Stay tuned for more updates. And don't forget to give an upvote if you like it 

# ### Feel free to ask any doubt/question/ or to give any suggestion :)
