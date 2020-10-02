#!/usr/bin/env python
# coding: utf-8

# This is my first attempt with minimal feature engineering and basic logistic regression model
# on Costa Rican Household Poverty model.

# In[ ]:


#import basic libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['font.size'] = 12


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
print('train data shape :',train_df.shape)
print('test data shape :',test_df.shape)


# #### Here, we can see that number of observations in training data set is very less than test dataset (almost 40%).

# In[ ]:


#snapshot of training data
train_df.head()


# In[ ]:


#check distribution of target classes in training dataset
y=train_df['Target']
y.value_counts()


# In[ ]:


#check nulls in train dataset
train_null = train_df.isnull().sum()
train_null[train_null > 0]


# In[ ]:


#check nulls in test dataset
test_null = test_df.isnull().sum()
test_null[test_null > 0]


# #### We can see from above two cells that columns v2a1, v18q1, rez_esc are missing for most of the observations in both training and test dataset.

# In[ ]:


train_df.info()


# #### From discussion in [thread](https://www.kaggle.com/c/costa-rican-household-poverty-prediction/discussion/61403#358941), poverty level is not consistent throughout the household. As per suggested by organizers, it is a data discrepany. So we will try to handle it here.

# In[ ]:


# Groupby the household and figure out the number of unique values
train_grphh = train_df.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)
err_train = train_grphh[train_grphh !=True]
print('Number of households with incorrect poverty level :',len(err_train))


# #### It is clarified in discussion, that correct poverty level is poverty level of head of the family. We can identity it using parentesco1 column with value 1. Let's use this to correct poverty level in errorneous records.

# In[ ]:


#let's correct the poverty level in incorrect records
for household in err_train.index:
    #find correct poverty level
    target = int(train_df[(train_df['idhogar']==household) & (train_df['parentesco1']==1.0)]['Target'])
    #set correct poverty level
    train_df.loc[train_df['idhogar']==household,'Target'] = target


# ## Exploratory Data Analysis

# In[ ]:


poverty_level = OrderedDict({1: 'extreme', 2: 'moderate', 3: 'vulnerable', 
                               4: 'non vulnerable'})
fig,ax = plt.subplots(figsize=(10,6))
ax.yaxis.label.set_color('black')
ax.xaxis.label.set_color('black')
sns.countplot(x='Target',data=train_df,color='blue')
plt.xticks([x - 1 for x in poverty_level.keys()], 
           list(poverty_level.values()), rotation = 60)
plt.xlabel('Poverty Level')
plt.title('Poverty Level Distribution')


# #### *As seen in above figure, we are dealing with imbalanced class problem!*

# In[ ]:


#let's check distribution of number of males & females in a household
legend_label = ('male < 12','male > 12','female < 12','female > 12')
plt.figure(figsize=(20,14))
for i,col in enumerate(['r4h1','r4h2','r4m1','r4m2']) :
    ax = plt.subplot(2,2,i+1)
    #sns.kdeplot(data = train_df[col])
    sns.barplot(x='Target',y=col,data=train_df)
    plt.title(legend_label[i]+' distribution')
    plt.xticks([x - 1 for x in poverty_level.keys()], 
           list(poverty_level.values()))
    plt.xlabel('Poverty Level')
    ax.legend('') #to remove legend


# #### We can see in above figure that number of childrens(both male and female) have direct impact on poverty level. It has a decreasing trend in poverty level, as number of children decreases. 

# #### Here, we can see that household having more males less than 12(male childs) tend to fall in extreme category.

# In[ ]:


#let's check distribution of basic faicilities like water and electricity in a household
legend_label = ('Water Supply','Electricity Supply')
plt.figure(figsize=(16,10))
for i,col in enumerate(['abastaguano','noelec']) :
    ax = plt.subplot(1,2,i+1)
    #sns.kdeplot(data = train_df[col])
    sns.barplot(x='Target',y=col,data=train_df)
    plt.title(legend_label[i]+' distribution')
    plt.xticks([x - 1 for x in poverty_level.keys()], 
           list(poverty_level.values()))
    plt.xlabel('Poverty Level')
    ax.legend('') #to remove legend


# In[ ]:


#let's check availabilty of electronic appliances like refrigerator and television in households
legend_label = ('Refrigerator','Television')
plt.figure(figsize=(16,10))
for i,col in enumerate(['refrig','television']) :
    ax = plt.subplot(1,2,i+1)
    #sns.kdeplot(data = train_df[col])
    sns.barplot(x='Target',y=col,data=train_df)
    plt.title(legend_label[i]+' distribution')
    plt.xticks([x - 1 for x in poverty_level.keys()], 
           list(poverty_level.values()))
    plt.xlabel('Poverty Level')
    ax.legend('') #to remove legend


# #### Surprisingly, more people at moderate poverty level have no water supply as comparison to extreme. But for electricity, it seems normal as more people at extreme poverty level have no electricity as compared to other levels.

# In[ ]:


#let's check distribution of electronic gadgets in a household
legend_label = ('Mobile','Computer','Tablet')
plt.figure(figsize=(20,14))
for i,col in enumerate(['mobilephone','computer','v18q']) :
    ax = plt.subplot(1,3,i+1)
    #sns.kdeplot(data = train_df[col])
    sns.barplot(x='Target',y=col,data=train_df)
    plt.title(legend_label[i]+' distribution')
    plt.xticks([x - 1 for x in poverty_level.keys()], 
           list(poverty_level.values()))
    plt.xlabel('Poverty Level')
    ax.legend('') #to remove legend


# #### We can see there is not much signigicant difference between people of different poverty level who owns a mobile phone. And least people from extreme poverty level owns a computer. For a tablet, there is not much significant difference between extreme and moderate poverty level.

# In[ ]:


#let's check number of rooms and bedrooms in households with respect to poverty level
legend_label = ('Rooms','Bedrooms')
plt.figure(figsize=(16,10))
for i,col in enumerate(['rooms','bedrooms']) :
    ax = plt.subplot(1,2,i+1)
    #sns.kdeplot(data = train_df[col])
    sns.barplot(x='Target',y=col,data=train_df)
    plt.title(legend_label[i]+' distribution')
    plt.xticks([x - 1 for x in poverty_level.keys()], 
           list(poverty_level.values()))
    plt.xlabel('Poverty Level')
    ax.legend('') #to remove legend


# #### We can see people at extreme poverty level have at least 4 rooms including two bedrooms.

# In[ ]:


#let's check number of rooms and bedrooms in households with respect to poverty level
legend_label = ('Children below 19','Seniors 65+')
plt.figure(figsize=(16,10))
for i,col in enumerate(['hogar_nin','hogar_mayor']) :
    ax = plt.subplot(1,2,i+1)
    #sns.kdeplot(data = train_df[col])
    sns.barplot(x='Target',y=col,data=train_df)
    plt.title(legend_label[i]+' distribution')
    plt.xticks([x - 1 for x in poverty_level.keys()], 
           list(poverty_level.values()))
    plt.xlabel('Poverty Level')
    ax.legend('') #to remove legend


# #### People at extreme poverty level have more children(<19) as compared to other level.

# In[ ]:


#Distribution of total members
plt.figure(figsize=(16,8))
sns.kdeplot(train_df['hogar_total'])
plt.xlabel('Total members')
plt.title('Distribution of total members in household')


# #### Around 40% of households have 4 individuals in family.A few households have more than 8 members in family.

# ## Feature Engineering

# In[ ]:


#Align training and test dataset to find common features
train_df,test_df = train_df.align(test_df,join='inner',axis=1)

print('Training Features shape: ', train_df.shape)
print('Testing Features shape: ', test_df.shape)


# In[ ]:


#let's join train and test data
data = pd.concat([train_df,test_df],axis=0)
data.head()


# In[ ]:


#check distinct values in object(categorical) column
cat_cols = data.nunique()==2
cat_cols = list(cat_cols[cat_cols].keys())


# In[ ]:


#change 'yes' to 1 and 'no' to 0 in below 3 columns to make consistent values in columns 
# as per description given on data page
cols = ['edjefe', 'edjefa','dependency']
data[cols] = data[cols].replace({'no': 0, 'yes':1}).astype(float)

#interaction features
data['hogar_mid'] = data['hogar_adul'] - data['hogar_mayor']
data['bedroom%'] =  data['bedrooms']/data['rooms']
data['person/rooms'] = data['rooms']/data['hhsize']
data['male_ratio'] = data['r4h3']/data['r4t3']
data['female_ratio'] = data['r4m3']/data['r4t3']
data['female_per_room'] = data['r4m3']/data['rooms']
data['female_per_bedroom']  = data['r4m3']/data['bedrooms']
data['hogarmid_per_bedroom']  = data['hogar_mid']/data['bedrooms']


# In[ ]:


#drop columns contains mostly null values and other columns like id and idhogar not useful in model
data.drop(labels=['v2a1','v18q1','rez_esc','Id','idhogar'],axis=1,inplace=True)
print('data shape after dropping null columns :',data.shape)


# In[ ]:


#impute missing values with -1
data.fillna(-1,inplace=True)


# In[ ]:


train_df = data[:len(train_df)]
test_df = data[len(train_df):]
print('Training data shape: ', train_df.shape)
print('Testing data shape: ', test_df.shape)


# ## Modelling

# In[ ]:


#import ML models and metrics
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,make_scorer
from sklearn.model_selection import cross_val_score

# Custom scorer for cross validation
scorer = make_scorer(f1_score, greater_is_better=True, average = 'macro')


# In[ ]:


X = train_df
logR = LogisticRegression(class_weight='balanced',C=0.0005)
cv_score = cross_val_score(logR, X, y, cv = 10, scoring = scorer)
print(f'10 Fold Cross Validation F1 Score = {round(cv_score.mean(), 4)} with std = {round(cv_score.std(), 4)}')


# In[ ]:


logR.fit(X,y)
preds_log = logR.predict(test_df)
sub_log = pd.DataFrame({'Id':test_df.index, 'Target':preds_log})
sub_log.to_csv('sub_log1.csv', index=False)


# In[ ]:




