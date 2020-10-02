#!/usr/bin/env python
# coding: utf-8

# ### The aim of this kernel is to build a model that predicts the best facebook ad campaign for a given customer. 
# #### There will be feature engineering, missing value imputation, and visualisations

# This data science assignment involved building an add-bidding model to offer the right
# type of ad out of three distinct campaigns to the right people.
# Firstly, the given data had two
# columns of campaign id and fb
# campaign id with about half of the
# data missing. Upon further
# investigation, the missing data
# seemed to be missing not at
# random(MNAR). This was derived
# due to the missing value columns
# following a identical distribution with
# rest of the data.
# 
# I made the assumption that the missing campaign_ids would most likely be 1178, as only
# three distinct campaign ids were given in this data and 1178 would follow a natural pattern
# in the data structure. I applied KNN-imputation with the choice of K-value of 3 that
# compared the three closest data points and it imputed everything to 1178. However, this
# approach may not be correct and the missing columns may also include the other two
# campaign ids.
# 
# Regarding data preprocessing, I used binary categorization for gender and hot-encoded
# age. This was done mainly due to technical correctness in order to remove perceived
# mathematical ordering of the data by the machine learning model. I also took the log of the
# continuous variables for scaling and combating kurtosis.
# 
# I employed simple Logistic Regression due to the small data size and small cost-benefit for
# exploration of more advanced models. I implemented Logistic Regression with recursive
# feature elimination to see which variables were redundant for predictive power. It turned
# out that all of the features had an increase on the predictive power.

# In[ ]:


import numpy as np # linear algebra
import matplotlib.pyplot as plt # For plotting
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.display.max_rows = 1000


# In[ ]:


df = pd.read_csv("../input/data.csv")


# ## Restructuring

# ### The data has some structural problems starting from row number 761. Let's restructure the dataset and apply  basic feature engineering and then visualise wheter the we can use the data from row 761 onwards. 

# In[ ]:



#split the data to two dataframes - df2 with missing values
df1 = df[0:761]
df2 = df[761:]
# restructure the df by shifting the columns to match between df1 and df2
c = list(df2)
for x in range(12):
    c[x+1] = c[x+3]
    
df2.columns = c
# further restructuring
df2 = df2.iloc[:, :-2]
df2.rename(columns={'campaign_id': 'reporting_start','fb_campaign_id': 'reporting_end'}, inplace=True)

df2.insert(3, 'campaign_id',np.NaN)
df2.insert(4,'fb_campaign_id',np.NaN)

df2.head()


# In[ ]:


df = df1.append(df2, ignore_index=True) # final dataframe 
df.head() 


# ### Feature Engineering

# In[ ]:


import datetime
#see how long the campaign durations have been
df['reporting_start'] = pd.to_datetime(df['reporting_start'] )
df['reporting_end'] = pd.to_datetime(df['reporting_end'] )

df['campaign_duration']= df['reporting_start']-df['reporting_end']

df['campaign_duration'].value_counts()


# ##### Since all of the campaigns lasted within one day, I decided to remove the variables as they do not offer any information gain

# In[ ]:


df.drop(['campaign_duration','reporting_start','reporting_end'],inplace=True,axis=1)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df['gender']=lb.fit_transform(df['gender']) # label encode gender


# In[ ]:


df['total_conversion'] = df['total_conversion'].astype(int) # change these variables to the proper format of an integer
df['approved_conversion'] = df['approved_conversion'].astype(int)
df['impressions'] = df['impressions'].astype(int)


# In[ ]:


df = pd.concat([df,pd.get_dummies(df['age'],prefix='age')],axis=1) # get dummies for age 
df.drop('age',inplace=True,axis=1)


# In[ ]:


# The dataframe should be solid now. 
df.head()


# ### Imputation

# ### The two column values of campaign_id and fb_campaign_id are missing. As campaign_id has only three distinct values, it will be a fundamental part of the add bidding model. Consequently, I will focus my attention on it. 

# In[ ]:


sns.countplot(df["campaign_id"])


# In[ ]:


# imputation of missing values
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, IterativeImputer, BiScaler
X = pd.DataFrame(KNN(k=3).fit_transform(df))
X.columns = df.columns
X.index = df.index


# In[ ]:


X['campaign_id'] = X['campaign_id'].astype(int)
X['fb_campaign_id'] = X['fb_campaign_id'].astype(int)
X['campaign_id'] = X['campaign_id'].replace(1177,1178) 


# In[ ]:


sns.countplot(X["campaign_id"])


# In[ ]:


sns.countplot(df["campaign_id"]) # compared to the original


# ### Visualisation
# 
# #### Now that the dataframe is restructured - let's identify some patterns in the data. Specifc focus point is the difference between the two dataframes

# In[ ]:


def distComparison(df1, df2): # A function to see the distribution of each feature
    a = len(df1.columns)
    if a%2 != 0:
        a += 1
    
    n = np.floor(np.sqrt(a)).astype(np.int64)
    
    while a%n != 0:
        n -= 1
    
    m = (a/n).astype(np.int64)
    coords = list(itertools.product(list(range(m)), list(range(n))))
    
    numerics = df1.select_dtypes(include=[np.number]).columns
    cats = df1.select_dtypes(include=['category']).columns
    
    fig = plt.figure(figsize=(15, 15))
    axes = gs.GridSpec(m, n)
    axes.update(wspace=0.25, hspace=0.25)
    
    for i in range(len(numerics)):
        x, y = coords[i]
        ax = plt.subplot(axes[x, y])
        col = numerics[i]
        sns.kdeplot(df1[col].dropna(), ax=ax, label='df').set(xlabel=col)
        sns.kdeplot(df2[col].dropna(), ax=ax, label='df_missing')
        
    for i in range(0, len(cats)):
        x, y = coords[len(numerics)+i]
        ax = plt.subplot(axes[x, y])
        col = cats[i]

        df1_temp = df1[col].value_counts()
        df2_temp = df2[col].value_counts()
        df1_temp = pd.DataFrame({col: df1_temp.index, 'value': df1_temp/len(df1), 'Set': np.repeat('df1', len(df1_temp))})
        df2_temp = pd.DataFrame({col: df2_temp.index, 'value': df2_temp/len(df2), 'Set': np.repeat('df2', len(df2_temp))})

        sns.barplot(x=col, y='value', hue='Set', data=pd.concat([df1_temp, df2_temp]), ax=ax).set(ylabel='Percentage')


# In[ ]:


import itertools
import matplotlib.gridspec as gs

df_missing= X[761:]
df_not_missing= X[0:761]

distComparison(df_not_missing, df_missing)


# ### We can discover some interesting patterns from the data. 
# ### The age distribution is fairly same in both datasets, df_missing having proportionally more younger people. The gender is interestingly split. Df_missing consists proportionally more of females and df proportionally more of males. 
# 
# ### Lastly, it seems to be that df_missing has a higher mean and a far higher standard deviation for features of clicks, spent, and conversions. 

# ## Model Fitting

# In[ ]:


df = X.copy()

df.spent=df.spent.astype(int)
df.interest1=df.interest1.astype(int)
df.interest2=df.interest2.astype(int)
df.interest3=df.interest3.astype(int)
df.campaign_id=df.campaign_id.astype('category')

df.dtypes


# In[ ]:


df.dropna(inplace=True)

df['approved_conversion'] = df['approved_conversion'].replace([range(2,22)], 1)

df.approved_conversion=df.approved_conversion.astype('category')
df['approved_conversion'].value_counts()

## to have class balance and for the purpose of the add-bidding model. 


# In[ ]:


from sklearn.metrics import  classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE


# In[ ]:


df.replace([np.inf, -np.inf], np.nan, inplace=True)

df.isna().sum()


# In[ ]:


df.dropna(inplace=True)


# In[ ]:


# Taking the log of the continious variables to mitigate kurtosis and skewdness as much as possible

col= [['interest1','interest2','interest3']]
for cols in col:
    df[cols] = np.log(df[cols])
    df[cols] = np.log(df[cols])


# In[ ]:


X = df[[ 'campaign_id','interest1','interest2','interest3','gender','age_30-34','age_35-39','age_40-44','age_45-49']]
y = df['approved_conversion']


# In[ ]:


## With oversampling follwed by undersampling to improve the score. Uncomment to see the improvement

from imblearn.combine import SMOTETomek

# smt = SMOTETomek(ratio='auto')
# X, y = smt.fit_sample(X, y)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


logmodel = LogisticRegression()
logmodel= RFE(logmodel, 9)
logmodel.fit(X_train,y_train)


# In[ ]:


predictions = logmodel.predict(X_test)
print(classification_report(y_test,predictions))
print(logmodel.ranking_) ## For RFE 
print(logmodel.support_) ## For RFE

# All features are required, if you set RFE to less than 9 the performance (f1-score) will decrease


# ## Add-bidding Model

# In[ ]:


X_test[:10]  # Here is the testing data that the model hasn't seen before. 


# In[ ]:


# model's prediction for the first 10 rows of test data
logmodel.predict(X_test[:10])


# In[ ]:


#Prediction for the first row
X1 = X_test[:1]
X1
logmodel.predict(X1)


# In[ ]:


# prediction when the campaign_id is changed to 916
X1['campaign_id'] = X1['campaign_id'].replace(936, 916)
logmodel.predict(X1)


# In[ ]:


#prediction to when the campaign id is change to 1178
X1['campaign_id'] = X1['campaign_id'].replace(916, 1178)
logmodel.predict(X1)


# ### Given any input vector, one can change the campaign_id variable to see which campaign would end up in an conversion according to the model. 
