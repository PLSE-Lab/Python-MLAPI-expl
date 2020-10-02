#!/usr/bin/env python
# coding: utf-8

# In this kernel I will be trying logistic regression and will be using only the training data.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import os
print(os.listdir("../input"))


# In[ ]:


train=pd.read_csv("../input/train.csv")
train.head()


# Lets Create a simple heat map to see where we are missing most of our data.

# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='Blues')


# From the above heatmap we can conlude that we are missing some of the age information and a lot of cabin information. 

# Lets further explore the data at a visual level

# In[ ]:


sns.countplot(x='Survived',data=train)


# The above plot shows how around 550 people did not survive and about 350 people suvived.

# In[ ]:


sns.countplot(x='Survived',hue='Sex',data=train)


# Looking at this plot we can observe a trend. Among people who did not survive there were more males and among people that did survive there were more females.

# In[ ]:


sns.countplot(x='Survived',hue='Pclass',data=train)


# Again we we notice a trend and we can clearly see that people who did not survive were of the lowest class(the 3rd class) and among people who did survive there were more people from the first class. But we should also consider the number of people in each class before making any conclusions.

# In[ ]:


sns.distplot(train['Age'].dropna(),kde=False,bins=30)#we dropped na to prevent them from messing up our distplot


# We can now observe the generall age of people on the titanic. There are quiet a few children but we can observe that most of the passangers are between 20-30   

# In[ ]:


sns.countplot(x='SibSp',data=train)


# From the above plot we can clearly see that most of the passengers did not have any children or spouse on board.

# In[ ]:


sns.distplot(train['Fare'],bins=40,kde=False)


# We observe that most of the fare were between 0 and 50 which makes sense as there were more people riding the cheaper third class.

# Now that we have explored the data by going though every column of the test data lets **clean** our data. Using heatmap we obseved that we have missing data for the Age and the Cabin columns. There is a lot of missing data in the Cabin column and so we will be setting it apart for now and focus our attention on the age column.
# We can fill-in the missing data from the age column instead of just dropping it all off.

# In[ ]:


plt.figure(figsize=(10,6))
sns.boxplot(x='Pclass',y='Age',data=train)


# In order to fill in the age we can use the average age values and impute it based on the passenger class(Pclass). We can observe the average age for each class in the boxplot above.

# In[ ]:


a=train.groupby(train['Pclass'])
mean=a['Age'].mean()
print(mean)


# We now know the exact mean age values for every class

# In[ ]:


def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Age):
        
        if Pclass==1:
            return 38
        elif Pclass==2:
            return 30
        else:
            return 25
    else:
        return Age
     


# In[ ]:


train['Age']=train[['Age','Pclass']].apply(impute_age,axis=1)


# Now that we have replaced the null values of age lets look at the heat map again

# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='Blues')


# We can now drop the cabin column as it has too much missing information. We can also get rid of any null values as they wont be useful to us.

# In[ ]:


train.drop('Cabin',axis=1,inplace=True)
train.dropna(inplace=True)
train.head()


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='Blues')


# Since we get a heatmap with a solid color we can now confirm that there are no null values in our dataset

# In order to apply ML algortihms to this data set we need to convert the categorical features into  **Dummy variables** using pandas. We do this as our algorith wont be able to directly take these features as inputs.
# In this dataset we have two categorical features, **Sex** and **Embarked**.

# In[ ]:


sex=pd.get_dummies(train['Sex'],drop_first=True)
sex.head()


# In[ ]:


embark=pd.get_dummies(train['Embarked'],drop_first=True)
embark.head()


# We dropped the First columns from both Sex and Embark to avoid Multicollinearity.
# Lets add these new dataframes to the train dataframe.

# In[ ]:


train=pd.concat([train,sex,embark],axis=1)
train.head(2)


# Now that we have created dummy variables for our categorical features we can now get rid of all the columns that we dont require.
# These mainly include columns that contain text like the name of the passanger and their ticket. We will also be dropping Sex and Embarked as we have created dummy variabled for them. We also be dropping the Passenger Id as it does not determine if someone will survive or die.

# In[ ]:


train.drop(['Name','Sex','Embarked','Ticket','PassengerId'],axis=1,inplace=True)
train.head()


# Lets train and use a  Model to predict if a passanger survived on the titanic. In this kernel we will be assume that the train dataset is all the data we have and perform a train-test split using scikit-learn so we dont have to clean the test.csv again.

# In[ ]:


X=train.drop('Survived',axis=1)
y=train['Survived']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.35, random_state=42)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel=LogisticRegression()


# In[ ]:


logmodel.fit(X_train,y_train)


# In[ ]:


predictions=logmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test,predictions))


# From the above table we can see the percision, f1score and recall values for our model.
# Considreing we only used the training data we can see that we did well as our f1-score quit close to 1.
