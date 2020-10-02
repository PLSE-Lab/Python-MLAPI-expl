#!/usr/bin/env python
# coding: utf-8

# # Importing libraries:

# In[150]:


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
print("imported!")


# # Reading and discovering the Dataset:

# In[151]:


Data=pd.read_csv('../input/train.csv')


# In[152]:


Data.head()


# # Dropping some useless features:

# In[153]:



Data.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)
Data.head()


# # Looking for missing values:

# In[154]:



print("#Age missing entries =",Data.Age.isnull().sum())
print("#survived missing entries =",Data.Survived.isnull().sum())
print("#Pclass missing entries =",Data.Pclass.isnull().sum())
print("#SibSp missing entries =",Data.SibSp.isnull().sum())
print("#Parch missing entries =",Data.Parch.isnull().sum())
print("#Fare missing entries =",Data.Fare.isnull().sum())
print("#Cabin missing entries =",Data.Cabin.isnull().sum())
print("#Embarked missing entries =",Data.Embarked.isnull().sum())




# In[155]:


Data[Data.Embarked.isnull()]


# The two empty records are for people who survived, let's check which Embarking region is more likely to survive, so that we can fill those two missing records.

# In[156]:



g = sns.catplot(x="Embarked", y="Survived",  data=Data,
                   height=6, kind="bar", palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# As we see:
# * 55% embarked from 'C' survived.
# * 35% embarked from 'Q' survived.
# * 32% embarked from 'S' survived.
# 
# So, it seems reasonable to fill the empty records with 'C'.
# 

# In[157]:


Data=Data.fillna({'Embarked':'C'})


# In[158]:


print("#Embarked missing entries =",Data.Embarked.isnull().sum())


# Now, Embarked feature has no missing values!

# # Let's play with the Cabin feature as it has more than 80% empty entries.

# In[159]:


Data=Data.fillna({"Cabin":'X'})
Data.head()


# In[160]:


Data["Cabin"]=Data["Cabin"].str.slice(0,1)


# In[161]:


Data.head(10)


# In[162]:


plot1=sns.catplot(x="Cabin", y="Survived",  data=Data,
                   height=6, kind="bar", palette="muted")


# It seems like passangers with no Cabin are more likely to not survive! Let's convert Cabin categorical feature into numbers so our ML algorithm can deal with it.

# In[163]:


Data['Cabin']=Data['Cabin'].replace(['A','B','C','D','E','F','G','T','X'],[0,1,2,3,4,5,6,7,8])
Data.head(10)


# In[164]:


# Converting other categorical features as well.
Data['Sex']=Data['Sex'].replace(['male','female'],[0,1])
Data['Embarked']=Data['Embarked'].replace(['S','C','Q'],[0,1,2])
Data.head()


# In[165]:


sns.heatmap(Data[["Age","Sex","SibSp","Parch","Pclass","Embarked","Fare","Cabin","Survived"]].corr(),annot=True)


# It is clear that Survival is strongly correleated with some features than others. Among these features is the Cabin feature which you think that it is better to be dropped at the first glance.

# It seems that age feature is most correlated to features "Pclass" and "SibSp" respectively, so we will fill NaN age values with the mean or median of each group of these two features.

# In[166]:


age_means=np.zeros((3,9))
median=Data.Age.mean()
for classNum in range (0,Data.Pclass.max()):  # 0 --> 1st class
    for sibNum in range (0,Data.SibSp.max()+1): # adding one to take the range [0,8] not [0,8[.
        age_means[classNum][sibNum]=Data["Age"][(Data["Pclass"]==(classNum+1)) & (Data["SibSp"]==sibNum)].mean()
        if np.isnan(age_means[classNum][sibNum]):
            age_means[classNum][sibNum]=median


# In[167]:


print(age_means)


# In[168]:


Null_indx=list(Data["Age"][Data["Age"].isnull()].index)


# In[169]:


for i in Null_indx:
    Data["Age"].iloc[i]=age_means[Data.Pclass[i] - 1][Data.SibSp[i]]
    
print("#Age missing entries =",Data.Age.isnull().sum())


# # Now, our data is cleaned and nothing is missing!

# Lets, split our data into train and test sets using Sklearn built in function.

# In[170]:


Y=Data.Survived
X=Data
X.drop(['Survived'],axis=1,inplace=True)


# In[171]:


X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.2, random_state=3)


# # It's time for Calling in Naive Bayes!

# In[172]:


classifier= GaussianNB()
classifier.fit(X_train, Y_train)
classifier.class_prior_


# In[173]:


predicts=classifier.predict(X_test)
accuracy=round(accuracy_score(predicts,Y_test),3)
print(accuracy)


# # Creating the submission file:

# In[174]:


test=pd.read_csv('../input/test.csv')
IDs=test.PassengerId
test.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)
test.head()


# In[175]:


test.Sex=test.Sex.replace(['male','female'],[0,1])
test.Embarked=test.Embarked.replace(['S','C','Q'],[0,1,2])
test=test.fillna({"Cabin":'X'})
test["Cabin"]=test["Cabin"].str.slice(0,1)
test.head()


# In[176]:


Null_test=list(test["Age"][test["Age"].isnull()].index)
for i in Null_test:
    test["Age"].iloc[i]=age_means[test.Pclass[i] - 1][test.SibSp[i]]
print("#Age missing entries =",Data.Age.isnull().sum())
print("#Pclass missing entries =",test.Pclass.isnull().sum())
print("#SibSp missing entries =",test.SibSp.isnull().sum())
print("#Parch missing entries =",test.Parch.isnull().sum())
print("#Fare missing entries =",test.Fare.isnull().sum())
print("#Embarked missing entries =",test.Embarked.isnull().sum())


# In[177]:


test['Cabin']=test['Cabin'].replace(['A','B','C','D','E','F','G','T','X'],[0,1,2,3,4,5,6,7,8])
test=test.fillna({'Fare':34})
subPredictions=classifier.predict(test)
subFile=pd.DataFrame({'PassengerId': [],'Survived':[]})
subFile.PassengerId=IDs
subFile.Survived=subPredictions
subFile.to_csv( 'MySubmissionCabin' ,index=False)
subFile.head()


# In[ ]:




