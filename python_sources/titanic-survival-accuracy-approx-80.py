#!/usr/bin/env python
# coding: utf-8

# # **Titanic**
# 
# ![](https://cdn.britannica.com/s:700x500/79/4679-050-BC127236/Titanic.jpg)
# #### There has never been universal agreement over the number of lives lost in the sinking of the Titanic. Beginning with the first news reports of the disaster, inquirers have found it unwise to trust the original passenger and crew lists, which were rendered inaccurate by such factors as misspellings, omissions, aliases, and failure to count musicians and other contracted employees as either passengers or crew members. Agreement was made more difficult by the international nature of the disaster, essentially involving a British-registered liner under American ownership that carried more than 2,000 people of many nationalities. Immediately after the sinking, official inquiries were conducted by a special committee of the U.S. Senate (which claimed an interest in the matter on the grounds of the American lives lost) and the British Board of Trade (under whose regulations the Titanic operated). The figures established by these hearings are as follows:
# 
# ### U.S. Senate committee: 1,517 lives lost
# 
# ### British Board of Trade: 1,503 lives lost
# 
# ### Confusion over these figures was immediately aggravated by the official reports of these inquiries to the U.S. Senate and the British Parliament; these reports revised the numbers to 1,500 and 1,490, respectively. The figures have been revised, officially and unofficially, so many more times since 1912 that most researchers and historians concede that they will never know how many of the people sailing on the Titanic died.

# Locate the dataset in kaggle directory

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Importing important libraries

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# In[ ]:


# fetching train.csv and test.csv
train_df=pd.read_csv('/kaggle/input/titanic/train.csv')
test_df=pd.read_csv('/kaggle/input/titanic/test.csv')
train_df.head()


# ## lets check for columns

# In[ ]:


# lets print these
print(train_df.columns)
print(test_df.columns)


# As we can see that test data does not contain survived columns as we have to predict that.

# In[ ]:


# lets take deep observation about data
print(train_df.info())
print(test_df.info())


# As we can see that there are 891 entries in train.csv and there are 418 enteries in test.csv

# In[ ]:


#lets check for null values in both datset
print(train_df.isnull().sum())
print('\t\t\t')
print(test_df.isnull().sum())


# As we can see that train data has 177 missing age columns and 687 missing cabin columns.
# for test data age columns have 86 as well as cabin columns have 327 missing values.
# we would handle these values later.

# In[ ]:


# lets describe more about data
train_df.describe(include='all')


# # Exploratory Data Analysis

# In[ ]:


#lets see relation bw different columns
corr_matrix=train_df.corr()
plt.figure(figsize=(15,6))
sns.heatmap(corr_matrix,annot=True)


# In[ ]:


fig,ax=plt.subplots(2,2,figsize=(15,10))
sns.countplot(train_df['Sex'],ax=ax[0][0])
sns.countplot(train_df['Embarked'],ax=ax[0][1])
sns.countplot(train_df['Pclass'],ax=ax[1][0])
sns.countplot(train_df['SibSp'],ax=ax[1][1])


ax[0][0].set_title('Total no of male and female')
ax[0][1].set_title('Embarked distribution')
ax[1][0].set_title('Passenger class distribution')
ax[1][1].set_title('Sibling or spouse Distribution')


# From the above plot we  can observe some points on train data before doing feature engineering.
# 1.There are approx 300 female and 600 male on ship.
# 2.Mostly above 600 belong to embarked (S) and 2nd most are C.
# 3.approx 500 of them belongs to pclass-3 and approx 200 each from class 1 and 2.
# 4.More than 600 people didnt have sibsp.
# These data will be very useful for feature engineering process.

# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(15,5))
sns.distplot(train_df['Age'],hist=True,ax=ax[0])
sns.distplot(train_df['Fare'],hist=True,ax=ax[1])


ax[0].set_title('Age distribution')
ax[1].set_title('Fare distribution')



# In[ ]:


plt.figure(figsize=(15,15))
sns.pairplot(train_df)


# # Feature Engineering

# In[ ]:


# check for null value
train_df.isnull().sum()


# As we can see age has 177 null columns it means we have to fill data very significantly as it is one of the important feature.
# we will do feature engineering on both test as well as train data side by side.

# In[ ]:


#we will replace the age columns by finding the mean of age with respect to Gender as well as passenger class 
train_gp=train_df.groupby(['Sex','Pclass'])['Age'].mean()
print(train_gp)
test_gp=test_df.groupby(['Sex','Pclass'])['Age'].mean()
print(test_gp)



# Now we can say that it is accurate way to fill null data in age columns.

# In[ ]:


# this function will fill null value with the desire mean value
def fillAgeNa(df):
    for i in range(len(df)) : 
        if pd.isnull(df.loc[i, "Age"]):
            if (df.loc[i,'Sex']=='female') and (df.loc[i,'Pclass']==1) :
                df.loc[i,'Age']=37
            elif(df.loc[i,'Sex']=='female') and (df.loc[i,'Pclass']==2) :
                 df.loc[i,'Age']=26
            elif(df.loc[i,'Sex']=='female') and (df.loc[i,'Pclass']==3):
                 df.loc[i,'Age']=22
            elif(df.loc[i,'Sex']=='male') and (df.loc[i,'Pclass']==1):
                 df.loc[i,'Age']=40
            elif(df.loc[i,'Sex']=='male') and (df.loc[i,'Pclass']==2):
                 df.loc[i,'Age']=30
            elif(df.loc[i,'Sex']=='male') and (df.loc[i,'Pclass']==3):
                 df.loc[i,'Age']=25
    return df
            
                
    


# In[ ]:


ndf=train_df.copy()
train_df=fillAgeNa(ndf)
train_df.isnull().sum()
#similarly for test data we will fill like
ndf=test_df.copy()
test_df=fillAgeNa(ndf)
train_df['Embarked'].fillna('S',inplace=True)


# In[ ]:


# now check for null values
train_df.isnull().sum()


#  Now our age column is fully sorted now. and we will have to look for cabin columns

# In[ ]:


# filling null value in fare column with mean
test_df['Fare'].fillna(test_df['Fare'].mean(),inplace=True)
test_df.isnull().sum()
test_df.shape


# cabin is drop as it is not that much important for our prediction and it also contains bunch of null data

# In[ ]:


# dropping cabin columns
train_data=train_df.drop('Cabin',axis=1)
test_data=test_df.drop('Cabin',axis=1)
test_data.shape


# In[ ]:



        


# In[ ]:





# In[ ]:





# In[ ]:


train_data.drop(['PassengerId','Name','Ticket'],inplace=True,axis=1)
test_data.drop(['PassengerId','Name','Ticket'],inplace=True,axis=1)


# ### Handling categorical value such as SEX and EMBARKED

# In[ ]:


#perform encoding for categorical variable
encd1=pd.get_dummies(train_data[['Sex','Embarked']],drop_first=True)
encd2=pd.get_dummies(test_data[['Sex','Embarked']],drop_first=True)


# In[ ]:


# now lets concat the encoded categorical data
train_data=pd.concat([train_data,encd1],axis=1)
test_data=pd.concat([test_data,encd2],axis=1)


# lets see the data now

# In[ ]:


print(train_data.head(2))
print(test_data.head(2))


# In[ ]:


# now we will count the size of family
train_data['FamilySize']=train_data['SibSp']+train_data['Parch']+1
test_data['FamilySize']=test_data['SibSp']+test_data['Parch']+1


# we will make another feature as if person is alone or not

# In[ ]:


# here we are apply lambda function 
train_data['Isalone']=train_data['FamilySize'].apply(lambda x : 1 if x>1 else 0)
test_data['Isalone']=test_data['FamilySize'].apply(lambda x : 1 if x>1 else 0)


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


train_data


# In[ ]:


# now lets drop unnecessry columns like Sex and Embarked
train_data.drop(['Sex','Embarked','FamilySize'],axis=1,inplace=True)
test_data.drop(['Sex','Embarked','FamilySize'],axis=1,inplace=True)


# ## Now that everything is sorted lets scale our data 

# In[ ]:


scaler=MinMaxScaler()
scaler.fit(train_data[['Fare']])
train_data['Fare']=scaler.transform(train_data[['Fare']])
train_data['Age']=scaler.fit_transform(train_data[['Age']])
test_data['Fare']=scaler.fit_transform(test_data[['Fare']])
test_data['Age']=scaler.fit_transform(test_data[['Age']])


# In[ ]:


# here we are applying lamda function to check if sibling or spouse is present or not
train_data['SibSp']=train_data['SibSp'].apply(lambda x: 1 if x>0 else 0)
test_data['SibSp']=test_data['SibSp'].apply(lambda x: 1 if x>0 else 0)


# In[ ]:


# lets create one mopre important feature by multiplying age with pclass
train_data['nw']=train_data['Age']*train_data['Pclass']
test_data['nw']=test_data['Age']*test_data['Pclass']


# ### After doing that much our data is pretty sorted and lets check how our data looks

# In[ ]:


print(train_data.head(2))
print(test_data.head(2))


# In[ ]:


# now everything looks good lets divide our data in x_train and y_train
X_train=train_data.drop('Survived',axis=1)
y_train=train_data[['Survived']]
print('shape of x train and y train')
print(X_train.shape,y_train.shape)
X_test=test_data
print('shape of x test')
print(X_test.shape)


# # Model Training

# ## Model selection
# ## As there are lots of model to apply classification we are using some of the model and the model which perform better will be applied
# ### 1.Random Forest
# ### 2. XGBoost classifier
# ### 3.Support vector Machine classifier
# ### 4.Logistics Regression
# ### 5.KNeighbors classifier
# 

# In[ ]:


model1=RandomForestClassifier()
model2=XGBClassifier()
model3=LogisticRegression()
model4=SVC(kernel='poly',gamma=1,C=0.1)
model5=KNeighborsClassifier(n_neighbors=23,leaf_size=23,p=1)
#model.fit(X_train,y_train)
model=[model1,model2,model3,model4,model5]


# In[ ]:


c=0
for m in model:
    c+=1
    m.fit(X_train,y_train)
    accur=round(m.score(X_train,y_train)*100,2)
    print('Model',c)
    print('accuracy =',accur)


# As we can clearly see the accuracy of Random forest is very good for train data so we perform classification on random forest

# ### Here, we are doing hyper parameter optimization on our Random forest algorithm to increase accuracy and improve model

# we will coment out this as i find best params already so to save time i comment out on parameter tuning

# In[ ]:


''' param_grid={'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}'''
#grid = RandomizedSearchCV(RandomForestClassifier(), param_grid, refit = True, verbose = 3) 
#grid.fit(X_train, y_train)


# In[ ]:


#grid.best_params_


# Now we will train our model on this data

# In[ ]:


model=RandomForestClassifier(n_estimators= 2000,
  min_samples_split= 5,
  min_samples_leaf= 2,
  max_features= 'sqrt',
  max_depth= None)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
y_pred=pd.Series(y_pred)
y_pred=y_pred.apply(lambda x: 1 if x else 0)
accur=round(model.score(X_train,y_train)*100,2)
accur


# # Submission file

# In[ ]:


# sample submission file look like this
dataframe=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
dataframe.head()


# In[ ]:


# here we will create submission file according to the given sample
subm=pd.concat([test_df['PassengerId'],y_pred],axis=1)
subm.rename(columns={'PassengerId':'PassengerId',0:'Survived'},inplace=True)


# In[ ]:


# lets verify our submission file with sample file
subm.head()


# In[ ]:


subm['Survived'].value_counts()


# ### Saving the output file

# In[ ]:


subm.to_csv('submission.csv',index=False)


# ### This is My first Submission for kaggle compete.So score is not that much good if you have any advice or suggestion you can comment.
# # If You like it plese motivate me by upvoting. Thank you.
