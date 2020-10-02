#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Black Friday EDA (with a little Machine Learning towards the end)

# Let's first import all the necessary packages
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


file = '../input/BlackFriday.csv'
df = pd.read_csv(file)


# # Table of Contents
# 
# 1. Brief Overview 
# 2. Age, Gender, Occupation Breakdown
# 3. Purchase Price Distribution
# 4. Marital Status Breakdown
# 5. Product Category Study
# 6. Age vs. Product Category Study
# 7. Correlations Matrix
# 8. Machine Learning Modelling 

# # A Brief Overview 
# 
# Here, we will start off with studying some basics of the data. We will look at some of the data structures of the data and its formats. We will then study some descriptive statistics on the columns. It's also important to check for null values to see if we can drop/correct it before developing applying machine learning models. 

# In[ ]:


df.shape


# In[ ]:


df.head(10)


# In[ ]:


df.describe()


# It looks like the only meaningful column we can study from this data-set is the Purchase column. The mean purchase price was 9333 with a standard deviation of approximately 4981. It looks like there could be a few outliers with the max going up to 23,961. We will need to keep this mind when creating our machine learning algorithms because outliers can affect the accuracy of our model. 

# In[ ]:


df.isnull().sum()


# # Age, Occupation, Gender Breakdowns
# 

# In[ ]:


age = df['Age'].value_counts()
occupation = df['Occupation'].value_counts()
gender = df['Gender'].value_counts()

labels_1 = df['Age'].unique()
labels_2 = df['Occupation'].unique()
labels_3 = df['Gender'].unique()

f,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(12,40))
ax1.pie(age,labels=labels_1,autopct='%1.1f%%')
ax1.set_title('Age Breakdown')
ax2.pie(occupation,labels=labels_2,autopct='%1.1f%%')
ax2.set_title('Occupation Breakdown')
ax3.pie(gender,labels=labels_3,autopct='%1.1f%%')
ax3.set_title('Gender Breakdown')
plt.show()


# We can see from the first graph that the majority of people in the store were between the ages 0 and 17. There was also a significant portion of people 55+ and between 26-35. Taking ages 35 and below to be within the 'Young' Age-group, we can see that they make a total of 60% of the population. It looks like this store definitely has stuff that caters to the young population, so we probably expect to see a correlation between age and spending. In terms of occupation, jobs 16,15 & 10 were the top 3. Studying gender, we can see that the overwhelming gender is Male. 

# # Purchase Price Distribution
# 
# Here, we want to study every the spending distribution of every user. Because each user may be multiple items, we will want to get the total spending price of each user at the store before plotting the distribution. From the graph below, we can see that the distribution is very left skewed with a few users spending largely (tail) as possible outliers. 

# In[ ]:


# Purchase distribution
from scipy.stats import norm
price = df.groupby('User_ID')['Purchase'].agg('sum')
plt.figure(figsize=(10,10))
sns.distplot(price)
plt.title('Mean Purchase Price Distribution')
plt.show()


# # Marital Status
# 
# Let's study the amount of people married vs. not married and study this for each city category. We can see that in each city, the number of unmarried people were higher than married people at this store, with city C having the most number of unmarried and married people overall. This is obviously absolute values and so it doesn't make sense to compare these numbers when the total number of customers from each city is different. 

# In[ ]:


# Marital Status Ratio
df['Marital_Status'].unique()
df['Marital_Status'].value_counts()
#Ratio per city
plt.figure(figsize=(10,8))
sns.countplot(x='City_Category',hue='Marital_Status',data=df)
plt.show()


# # Product Category Study
# 
# It looks like each product falls into 3 main categories. In each category, there are various sub-categories. Let's:
# 
# 1. Fill all NaN with 0, and convert float columns to int
# 2. Study the sub-category ranges for each product category
# 3. Let's study the count plot of the different categories

# In[ ]:


df[['Product_Category_2','Product_Category_3']] = df[['Product_Category_2','Product_Category_3']].fillna(0).astype(int) 
print('Category 1:',sorted(df['Product_Category_1'].unique()))
print('Category 2:',sorted(df['Product_Category_2'].unique()))
print('Category 3:',sorted(df['Product_Category_3'].unique()))


# In[ ]:


f, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(18,10))
sns.countplot(x='Product_Category_1',data=df,ax=ax1,color='r')
sns.countplot(x='Product_Category_2',data=df,ax=ax2,color='b')
sns.countplot(x='Product_Category_3',data=df,ax=ax3,color='g')
plt.show()


# It looks like in product category 1, sub-category 1, 5 and 8 were the most popular in the story. In sub-category 2, 0 was the largest but because we populated NaNs with 0, this just means there were a lot of unknown product 2 sub-categories under product category 2. The only known popular sub-categories were 2, 8 and 14. The same process applies for Product Category 3 where there were a lot of unknowns populated as 0 there as well. It looks like product category 1 will probably be the more useful column in helping predicting purchase price as we have a lot of known information on this. 
# 
# I'm curious to see if Age plays a crucial role in how popular product sub-categories are. I would think that people who are relatively young vs. old will have different preferences in what they buy, and that will accordingly affect prices. 

# # Age vs. Product Category
# 
# Since we previously saw how there was a huge young population in and out of this store, I'm curious to see if there is a trend in the types of products bought vs. their age (after all, there's got to be something these ages like in this store to come shop in it). I'm going to do a little feature engineering first by setting some group labels based on age ranges:
# 
# - [**0-17, 18-25, 26-35] : Young**
# - [**36-45, 46-50, 51-55] : Middle-Aged**
# - [**51+] : Old Age**

# In[ ]:


# Let set a function for this
def age_group(age):
    if (age == '0-17') or (age=='18-25') or (age == '26-35'):
        return 'Young'
    if (age == '36-45') or (age=='46-50') or (age == '51-55'):
        return 'Middle'
    else:
        return 'Old'

df['Age_group'] = df['Age'].apply(age_group)

# Young population product study
f, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(20,8))
young_1 = df[df.Age_group == 'Young']['Product_Category_1'].value_counts()[:3].plot(kind='bar',ax=ax1)
ax1.set_xlabel('Product 1 Sub-Category')
ax1.set_ylabel('Frequency')
ax1.set_title('Young Population')
young_2 = df[df.Age_group == 'Young']['Product_Category_2'].value_counts()[:3].plot(kind='bar',ax=ax2)
ax2.set_xlabel('Product 2 Sub-Category')
ax2.set_ylabel('Frequency')
ax2.set_title('Young Population')
young_3 = df[df.Age_group == 'Young']['Product_Category_3'].value_counts()[:3].plot(kind='bar',ax=ax3)
ax3.set_xlabel('Product 3 Sub-Category')
ax3.set_ylabel('Frequency')
ax3.set_title('Young Population')

f, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(20,8))
middle_1 = df[df.Age_group == 'Middle']['Product_Category_1'].value_counts()[:3].plot(kind='bar',ax=ax1)
ax1.set_xlabel('Product 1 Sub-Category')
ax1.set_ylabel('Frequency')
ax1.set_title('Middle Aged Population')
middle_2 = df[df.Age_group == 'Middle']['Product_Category_2'].value_counts()[:3].plot(kind='bar',ax=ax2)
ax2.set_xlabel('Product 2 Sub-Category')
ax2.set_ylabel('Frequency')
ax2.set_title('Middle Aged Population')
middle_3 = df[df.Age_group == 'Middle']['Product_Category_3'].value_counts()[:3].plot(kind='bar',ax=ax3)
ax3.set_xlabel('Product 3 Sub-Category')
ax3.set_ylabel('Frequency')
ax3.set_title('Middle Aged Population')

f, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(20,8))
old_1 = df[df.Age_group == 'Old']['Product_Category_1'].value_counts()[:3].plot(kind='bar',ax=ax1)
ax1.set_xlabel('Product 1 Sub-Category')
ax1.set_ylabel('Frequency')
ax1.set_title('Old Age Population')
old_2 = df[df.Age_group == 'Old']['Product_Category_2'].value_counts()[:3].plot(kind='bar',ax=ax2)
ax2.set_xlabel('Product 1 Sub-Category')
ax2.set_ylabel('Frequency')
ax2.set_title('Old Age Population')
old_3 = df[df.Age_group == 'Old']['Product_Category_3'].value_counts()[:3].plot(kind='bar',ax=ax3)
ax3.set_xlabel('Product 1 Sub-Category')
ax3.set_ylabel('Frequency')
ax3.set_title('Old Aged Population')

plt.show()


# Wow, it looks like there doesn't seem to be a huge correlation between age group and the categories of products bought, except for a few differences seen. I expected to see a difference in product categories from Young & middle but it looks like the top 3 product sub-categories for product category 1 & 2 were the same. For the old age, looks like product category 8 topped the charts. It looks like age-group probably won't be highly correlated with purchase price since we see that product categories are similar. 

# # Correlation Matrix
# 
# Let's study the correlation matrix between the different features and see if we can find a high correlation between variables. The correlation function will automatically find the correlation between numerical columns which is pretty convenient!

# In[ ]:


# Correlation matrix between features.
corr = df[['Gender','Age','Occupation','City_Category','Stay_In_Current_City_Years','Marital_Status','Product_Category_1','Product_Category_2','Product_Category_3','Purchase']].corr()
sns.heatmap(corr, xticklabels=corr.columns.values,yticklabels=corr.columns.values, annot=True)
plt.show()


# A few things I can take-away from this correlation matrix:
# 
# - There seems to be a relatively strong** negative correlation** between **product category 1 and purchase price**. 
# - There also seems to be a relatively **positive correlation** between** product category 3 and purchase price** 
# - There is a **negative correlation** between product category 1 and product category 3** (we should watch out for this as it could possibly over-fit our machine learning model). 

# # Machine Learning Models
# 
# Machine learning models always work better with numerical values instead of categorical values, so we can help code these categories into unique numerical identifiers using the pandas dummy function (pd.get_dummies()). This will encode all the different labels into numerical vectors.  
# 
# 
# Here, I'm to try and follow this procedure outlined below to pave way to the right model:
# 
# -  Pre-process the data using pd.get_dummies() and LabelEncoder(). I use label encoder on specific columns that have too many unique categories because using dummies will make the data too large. 
# 2. Split the data between input features and the output variable which is the purchase price
# 3. Run linear regression, random forest regressor and XGboost regressor

# In[ ]:


from sklearn.preprocessing import LabelEncoder
label_enc = LabelEncoder()
product_enc = LabelEncoder()
df['User_ID'] = label_enc.fit_transform(df.User_ID)
df['Product_ID'] = product_enc.fit_transform(df.Product_ID)

# One Hot Encoding Age, Stay in Current City Years, City_Category
df_Age = pd.get_dummies(df.Age)
df_city = pd.get_dummies(df.City_Category)
df_staycity = pd.get_dummies(df.Stay_In_Current_City_Years)
df_Gender = pd.Series(np.where(df.Gender == 'M',1,0), name='Gender')
df_Agegroup = pd.get_dummies(df.Age_group)

df_new = pd.concat([df,df_Gender,df.Age_group, df_Age, df_city, df_staycity], axis=1)
df_new.drop(['Age','Gender','City_Category','Age_group','Stay_In_Current_City_Years'], axis=1, inplace=True)
df_new = df_new.rename(columns={'0':'Gender'})


# In[ ]:


# let's sample only half the df_new data
df_sample = df_new.sample(frac=0.05, random_state=100)
X2 = df_sample.drop(['Purchase'], axis=1)
y2 = df_sample.Purchase


# In[ ]:


# Linear Model
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
mod = LinearRegression()

scoring = 'neg_mean_squared_error'
linear_cv = cross_val_score(mod, X2,y2, cv=5, scoring=scoring)
print((-1*linear_cv.mean())**0.5)


# In[ ]:


import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import learning_curve
train_sizes, train_scores, valid_scores = learning_curve(mod, X2, y2, cv=3, scoring='neg_mean_squared_error')

train_scores = (-1*train_scores)**0.5
valid_scores = (-1*valid_scores)**0.5
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)
valid_scores_std = np.std(valid_scores, axis=1)

plt.figure()
plt.plot(train_sizes,valid_scores_mean,label='valid')
plt.plot(train_sizes,train_scores_mean,label='train')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.3,color="g")
plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std,valid_scores_mean + valid_scores_std, alpha=0.3, color="b")
plt.xlabel('Number of samples')
plt.ylabel('RMSE')
plt.legend()
plt.show()


# From this graph, I noticed that as the model fits more and more training data, the rmse goes up (which it naturally should as there will always be a residual error between actual and predicted value based on training data). However, when studying the validation test-set, it doesn't seem like the model is doing a good job at predicting the output. This seems like there is a **high bias**, where the model isn't complex enough to fit the training data and predict test data. We need a more complex model - one that has a low bias and low variance. Variance is the model's ability to see the generalized trend without overfitting on the training-data. Let's try a Random Forest Regressor, which is known to be a complex model but one that doesn't overfit.

# In[ ]:


# Random Forest Regressor 
from sklearn.ensemble import RandomForestRegressor
mod = RandomForestRegressor()
scoring = 'neg_mean_squared_error'
RF_cv = cross_val_score(mod, X2,y2, cv=5, scoring=scoring)
print((-1*RF_cv.mean())**0.5)


# It looks like we have reduced our mean squared errror from around 4000 to around 3000, a positive sign! Let's have a look at the learning curve performance of the Random Forest Regressor, to see if the model has a high variance or bias. 

# In[ ]:


# Random Forest Regressor Learning Curve
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import learning_curve
train_sizes, train_scores, valid_scores = learning_curve(RandomForestRegressor(), X2, y2, cv=3, scoring='neg_mean_squared_error')

train_scores = (-1*train_scores)**0.5
valid_scores = (-1*valid_scores)**0.5
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)
valid_scores_std = np.std(valid_scores, axis=1)

plt.figure()
plt.plot(train_sizes,valid_scores_mean,label='valid')
plt.plot(train_sizes,train_scores_mean,label='train')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.3,color="g")
plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std,valid_scores_mean + valid_scores_std, alpha=0.3, color="b")
plt.xlabel('Number of samples')
plt.ylabel('RMSE')
plt.legend()
plt.show()


# From the looks of it, the model seems to be **over-fitting** where the training score is way lower than the test score (which means there is high variance in the model and we need to generalize it further). We need to generalize the model a little more. Let's study the feature importances of the random forest regressor and see if we can eliminate any features. 

# In[ ]:


rf = RandomForestRegressor(n_estimators=100).fit(X2,y2)
f_im = rf.feature_importances_.round(3)
ser_rank = pd.Series(f_im,index=X2.columns).sort_values(ascending=False)
plt.figure()
sns.barplot(y=ser_rank.index,x=ser_rank.values,palette='deep')
plt.xlabel('relative importance')


# Product Category 1 looks like the most important feature in helping shape how much a product costs! That followed by the actual user and product ID. However, since we know that the user doesn't really purchase based on their own user ID and product ID, these 2 variables may be causing an over-fitting issue. It's most likely that user ID and product ID is random amongst the population of buyers and there probably isn't a clear trend. Let's remove these two variables and see if the model performs better.

# In[ ]:


X2 = df_sample.drop(['User_ID','Product_ID','Purchase'], axis=1)
y2 = df_sample.Purchase

# Random Forest Regressor 
from sklearn.ensemble import RandomForestRegressor
mod = RandomForestRegressor()
scoring = 'neg_mean_squared_error'
RF_cv = cross_val_score(mod, X2,y2, cv=5, scoring=scoring)
print((-1*RF_cv.mean())**0.5)


# Looks like the error went down a little, but it didn't change that much. Maybe we can try another ensemble learning known as XGBOOST. 

# # Gradient Boosting (XGBOOST)
# 
# Now, we'll try another ensemble tree learning method known as Gradient Boosting. XGBoost has two types of learners, linear based learner and a tree based learner. We will try both and see which one performs better. The default learner is the tree-based learner which I will use in conjunction with train_test_split and evaluate the error of our model.

# In[ ]:


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.20, random_state=123)
xg_reg = xgb.XGBRegressor()
xg_reg.fit(X_train,y_train)
preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))


# # XGBOOST (Linear base learner)

# In[ ]:


DM_train = xgb.DMatrix(X_train,y_train)
DM_test =  xgb.DMatrix(X_test,y_test)
params = {"booster":"gblinear", "objective":"reg:linear"}
xg_reg = xgb.train(params = params, dtrain=DM_train, num_boost_round=5)
preds = xg_reg.predict(DM_test)
rmse = np.sqrt(mean_squared_error(y_test,preds))
print("RMSE: %f" % (rmse))


# It looks like we should stick with our tree-based learner and xgboost performs a little better than Random Forest. Now we can try optimizing this model and also analyze whether the algorithm is being over-fitted or not. My next steps in optimizing this XGBoost would be:
# 
# - try different learning rates 
# - try varying the sample of data and number of feature columns used to train XGBoost decision trees
# 

# # Ending Note
# 
# That's the best I could do as far as algorithm tuning to get the error down. It just might be that we need more info on the product sub-categories of products 2 & 3 in order to develop better models. If anyone has suggestions for me on how I can better my EDA and modelling, I would highly appreciate it! But I hope I was able to provide some insight and help a few of you kagglers out there. I'm still pretty new to kaggle but I hope to improve on this analysis in the future with your help! 
# 
# Stay Tuned!
# 
# Cheers. 

# In[ ]:




