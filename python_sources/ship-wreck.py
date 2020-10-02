#!/usr/bin/env python
# coding: utf-8

# ## **Ship Wreck** 
# ### This is a begginer to beginner kernel. It has not advanced methods in it. Just purposed to give a brief idea about Exploratory Data Analysis,Data Preprocessing and Logistic Regression.

# ### Import required libraries.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
get_ipython().run_line_magic('matplotlib', 'inline')
import os


# ### Get the list of data which will be used in this study.

# In[ ]:


print(os.listdir('../input'))


# ### Read the CSV files which are given as competition data.

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sub = pd.read_csv('../input/gender_submission.csv')


# ### Let's look at the Data Dictionary for the *Train* dataset which is given to us, to get information.

# * PassengerId -- The id belongs to the passenger.
# * Survived -- Indicates whether the passenger survived from the Titanic.
# * Sex -- Gender of the passenger.
# * Age -- Age of the passenger.
# * SibSp -- Number of siblings and spouses aboard the Titanic.
# * Parch -- Number of parents and childeren aboard the Titanic.
# * Pclass -- Class of the passenger's ticket. (1st, 2nd and 3rd)
# * Fare -- The amount which passenger has paid for the ticket.
# * Cabin -- The cabin number of the passenger.
# * Embarked -- The port where the passenger aboarded the Titanic. (C,Q,S)

# ## **Exploratory Data Analysis**

# ### Checking the first two rows of the "Train Data Frame" to inspect our columns and data types.

# In[ ]:


train.head(2)


# In[ ]:


train.info()


# ### A visual approach to the columns to determine the density of null values in a column.

# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# ### Drawing a countplot by using *Seaborn* library to show the number of people survived and couldn't survived from the accident.

# In[ ]:


sns.countplot(x='Survived',data=train)


# ### Drawing a countplot by using *Seaborn* library to show the number of people survived and couldn't survived from the accident with the aspect of *Sex*.

# In[ ]:


sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')


# ### Drawing a countplot by using *Seaborn* library to show the number of people survived and couldn't survived from the accident with the aspect of *Person Class*.

# In[ ]:


sns.countplot(x='Survived',hue='Pclass',data=train)


# ### Drawing a distribution plot by using *Seaborn* library to show the variance in the *Age*.

# In[ ]:


sns.distplot(train['Age'].dropna(),kde=False,bins=30)


# ### Drawing a countplot by using *Seaborn* library to show the number of Siblings and Spouses in the ship.

# In[ ]:


sns.countplot(x='SibSp',data=train)


# ### Drawing a histogram to show the variance of the *Fare*

# In[ ]:


train['Fare'].hist(bins=40,figsize=(10,4))


# ### Drawing a boxplot by using *Seaborn* library to show the maximum, minimum and the mean ages with the aspect of *Person Class*

# In[ ]:


plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass',y='Age',data=train)


# ## **Data Preprocessing**

# ### impute_age() and impute_age_test() methods are created to fill the null values in the *Age* column with the help of *Pclass* column.
# #### The idea here is to get the mean age of each person class to fill null values in the age column with a closer estimation.

# In[ ]:


mean_pc1 = train[train['Pclass'] == 1]['Age'].mean()
mean_pc2 = train[train['Pclass'] == 2]['Age'].mean()
mean_pc3 = train[train['Pclass'] == 3]['Age'].mean()

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return mean_pc1
        elif Pclass == 2:
            return mean_pc2
        else:
            return mean_pc3
    else:
        return Age


# In[ ]:


mean_pc1 = test[test['Pclass'] == 1]['Age'].mean()
mean_pc2 = test[test['Pclass'] == 2]['Age'].mean()
mean_pc3 = test[test['Pclass'] == 3]['Age'].mean()

def impute_age_test(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return mean_pc1
        elif Pclass == 2:
            return mean_pc2
        else:
            return mean_pc3
    else:
        return Age


# ### The null values in the *Age* column of train and test data frames are filled.

# In[ ]:


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
test['Age'] = test[['Age','Pclass']].apply(impute_age_test,axis=1)


# ### Checking if is there a null value in the *Age* column.

# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# ### If we look carefully to the heatmap which created a cell above. It shows that there are so many null values in the *Cabin* column in the train Data Frame. Also, there is not a logical relationship between *Cabin* and rest of the columns. So, it will be a good call to drop that column and get rid of unrelated data.

# In[ ]:


train.drop('Cabin',axis=1,inplace=True)
test.drop('Cabin',axis=1,inplace=True)


# ### As shown in the output below. There are 2 null values in the *Embarked* column. 

# In[ ]:


train['Embarked'].isna().value_counts()


# ### So, fill 'em up with the most occured value in the column which is known as the *mode* of that column.

# In[ ]:


mode_emb = train['Embarked'].mode()[0]
train['Embarked'] = train['Embarked'].fillna(mode_emb)
print(mode_emb)


# ### Finally we got rid of our null values in the Data Frame, as shown in the heatmap below. Yaay!

# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# ### In *Name* column, there are titles for every passenger in the Data Frame. So, I decided to get them to create a new column.

# In[ ]:


train['Title'] = train['Name'].apply(lambda x : x.split(',')[1].split('.')[0].strip())
test['Title'] = test['Name'].apply(lambda x : x.split(',')[1].split('.')[0].strip())


# ### After getting titles, *Name* column will no longer be needed. So, it is dropped from the Data Frame

# In[ ]:


train.drop('Name',axis=1,inplace=True)
test.drop('Name',axis=1,inplace=True)


# ### Checking the unique titles in *Title* column.

# In[ ]:


train['Title'].value_counts()


# ### It is not advised that to use so many indicators, it will make everything harder for our upcoming ML model. So, impute_title() method is used to create **three** groups out of them to simplify our process.

# In[ ]:


def impute_title(cols):
    Title = cols
    if Title == 'Mr':
        return 0
    elif Title == 'Miss' or Title == 'Mrs':
        return 1
    else:
        return 2


# In[ ]:


train['Title'] = train['Title'].apply(impute_title)
test['Title'] = test['Title'].apply(impute_title)


# ### According to Data Dictionary *SibSp* shows the number of siblings and spouses, *Parch* shows the number of parents and childeren boarded to the ship.

# In[ ]:


train['FamilySize'] = train['SibSp'] + train['Parch']
test['FamilySize'] = test['SibSp'] + test['Parch']


# ### As we see there are not any **null** columns in the *train* dataset. We can not be sure that there won't be a null *Fare* value in the test dataset. So impute_fare() method is created to fill null Fare values with the mean *Fare* of the passenger's *Person Class*

# In[ ]:


def impute_fare(cols):
   Fare = cols[0]
   Pclass = cols[1]
   if pd.isnull(Fare):            
       return test[test['Pclass'] == Pclass]['Fare'].mean()
   else:
       return Fare


# In[ ]:


test['Fare'] = test[['Fare','Pclass']].apply(impute_fare,axis=1)


# ## **Creating Dummies**
# ### Dummies are used to convert *categorical* variable to *dummy/indicator* variables.

# ### Hence, *Sex* , *Embarked*, *Pclass* and *Embarked* are considered as categorical data, dummies can be applied. This get_dummies() method splits columns into the number of unique values of ones and zeros corresponding to their values. Also, first column of the splitted data is dropped to prevent overfitting of the upcoming machine learning model.

# In[ ]:


sex = pd.get_dummies(train['Sex'],drop_first=True)
pclass = pd.get_dummies(train['Pclass'],drop_first=True)
title = pd.get_dummies(train['Title'],drop_first=True)
emb = pd.get_dummies(train['Embarked'],drop_first=True)


# In[ ]:


sex_t = pd.get_dummies(test['Sex'],drop_first=True)
emb_t = pd.get_dummies(test['Embarked'],drop_first=True)
pclass_t = pd.get_dummies(test['Pclass'],drop_first=True)
title_t = pd.get_dummies(test['Title'],drop_first=True)


# ### These new columns must be inserted to train and test Data Frames. So concat() method is used for this operation.

# In[ ]:


train = pd.concat([train,sex,emb,pclass,title],axis=1)
test = pd.concat([test,sex_t,emb_t,pclass_t,title_t],axis=1)
train.head()


# ### After creating dummy columns, the initial columns can be dropped.

# In[ ]:


train.drop(['PassengerId','Pclass','SibSp','Parch','Title','Sex','Embarked','Ticket'],axis=1,inplace=True)
test.drop(['PassengerId','Pclass','SibSp','Parch','Title','Sex','Embarked','Ticket'],axis=1,inplace=True)


# ## **Machine Learning Model Implementation**
# ### We are trying to predict whether the passenger survived or not. So, the final prediction should consist boolean values which are *True* and *False*.
# ### First of all X and y Data Frames are created to train our model. X has *all* columns except *Survived*, y has *Survived* column which will be predicted by the model.

# In[ ]:


X = train.drop('Survived',axis=1)
y = train['Survived']


# In[ ]:


X.head()


# ### train_test_split is used to create *train* and *test* datasets from the initial train dataset which is given. *test* dataset contains randomly collected 20%, *train* dataset contains randomly collected 80% of the initial dataset.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42)


# ### A logistic regression object is created from LogisticRegression class.

# In[ ]:


reg = LogisticRegression()


# ### *X_train* and *y_train* which are created above is fitted to our *Logistic Regression* model.

# In[ ]:


reg.fit(X_train,y_train)


# ### **Hey, you have reached to the target.**

# In[ ]:


reg.score(X_train,y_train)


# ### *y_pred* dataset contains the predicted data from our initial *test* dataset. We can call it our *target*.

# In[ ]:


y_pred = reg.predict(test)


# In[ ]:


y_pred


# ## **Creating submission file**

# ### Hence, a sample submission file has given. We can use it as a template to ourselves.

# In[ ]:


sub.head()


# ### Set our *y_pred* array to *Survived* column of sample submission file.

# In[ ]:


sub['Survived'] = y_pred


# ### Finally, we have completed our work here and the last step is create the *CSV* file to submit.

# In[ ]:


sub.to_csv('csv_to_submit.csv',index=False)


# ### **Thank you for looking at my kernel.**
# ### **Please share your thoughts with me in the comments section. **

# In[ ]:




