#!/usr/bin/env python
# coding: utf-8

# Hello and Welcome to Kaggle, the online Data Science Community to learn, share, and compete. I decided to post a kernal for the beginners like me,who are willing to gain knowledge.I am sharing my process of approaching a dataset and solving it.I wrote this kernal in a way that I am explaining to myself the steps and process so that anyone who read this will find it in an interesting way rather than a blog. I am new to machine learning and hoping to learn a lot, so feedback is very welcome and also suggest me some new techniques in Data Cleaning process! Please upvote if you find this kernal is useful. 

# ## 1. DataPreprocessing

# In[ ]:


#First step is importing the required Libraries,these four libaries are useful for our intial process so I make it a habit to do this as a first step.
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


# In[ ]:


df=pd.read_csv("../input/titanic/train.csv")
test=pd.read_csv("../input/titanic/test.csv")
print(df.shape)
df.head() #After importing into dataframe use head fucntion to have a look at dataset and columns.


# In[ ]:


print(test.shape)
test.head()


# ## 2. Handling Missing Values
# After Loading the data into DataFrame,First thing what I learnt was to takecare of missing values,for this step I am going in the following way

# In[ ]:


# First Calculate the percent of missing values of each attribute in Training and Test sets
a=df.isnull().sum().sort_values(ascending=False)
percent=(df.isnull().sum()/df.isnull().count())*100
a=pd.concat([a,percent],axis=1)
a.head()


# In[ ]:


a=test.isnull().sum().sort_values(ascending=False)
percent=(test.isnull().sum()/test.isnull().count())*100
a=pd.concat([a,percent],axis=1)
a.head()


# ### 2.1 Dropping Columns and using fillna

# The next step is dropping those columns which have more than 50% missing values, because most of the values are missing there is no use in filling that much values.So next step is dropping columns.
# Here what I am doing is I am just thinking in a normal way, while dropping some Columns, For example, Now I am going to drop Cabin(percent missing values is more),along with that I decided to drop "Ticket","Fare","PassengerId".Just think in a way how cost of a ticket will effect our target variable we already  have Pclass attribute which will help us in estimating the ticket status.So in that way I decided to drop these features.
# If we want we can drop Name of the Passenger also,But after learning from some Kernals that name also plays a role,just think in a way that title will tells about a person status,as we Know in a Common way, So I am going to keep Name attribute.
# This is second step I am going to do in any Dataset,by studying about independent Features and dropping which will be of no use.(please provide feedback about this way of dropping)

# In[ ]:


main_df=test["PassengerId"]# For Submission file we will need Id 


# In[ ]:


datasets=[df,test]  
for dataset in datasets:
    dataset.drop(["Cabin","PassengerId","Ticket","Fare"],axis=1,inplace=True)
    dataset["Age"].fillna(dataset["Age"].mode()[0],inplace=True)
    dataset["Embarked"].fillna(dataset["Embarked"].mode()[0],inplace=True)


# # Exploratory Data Analysis

# In EDA,we will look at our remaining attributes and their distribution and relation to our target variable

# In[ ]:


df.head()# have a look at our remaining features


# ### P Class
# Pclass is categorical variable. Let's look at the distribution.

# In[ ]:


sns.countplot(x="Pclass",hue="Survived",data=df)
# from below Countplot we can see that Pclass plays a role in prediction of Survived,so we will keep that feature


# In[ ]:


ax=sns.barplot(x="Pclass",y="Survived",data=df)
ax.set_ylabel("Survival Probability")


# ### Sex

# In[ ]:


sns.countplot(x="Sex",hue="Survived",data=df) # sex also plays a role,we will keep that 


# In[ ]:


sns.barplot(x="Sex",y="Survived",data=df)


# ### Age

# In[ ]:


fig = plt.figure(figsize=(10,8),)
a=sns.kdeplot(df.loc[(df["Survived"]==0),"Age"],color="g",shade=True,label="Not Survived")
a=sns.kdeplot(df.loc[(df["Survived"]==1),"Age"],color="b",shade=True,label="Survived")
plt.title('Age Distribution - Surviver V.S. Non Survivors', fontsize = 20)
plt.xlabel("Passenger Age", fontsize = 12)
plt.ylabel('Frequency', fontsize = 12);


# In[ ]:


sns.lmplot('Age','Survived',data=df)

# We can also say that the older the passenger the lesser the chance of survival


# ### Sibsp

# In[ ]:


ax=sns.countplot(x="SibSp",hue="Survived",data=df)


# In[ ]:


sns.factorplot(x="SibSp",y="Survived",data=df,kind="bar")


# ### Parch

# In[ ]:


sns.countplot(x="Parch",hue="Survived",data=df)


# In[ ]:


sns.factorplot(x="Parch",y="Survived",data=df,kind="bar")


# Lets perfrom some feature engineering from what we have observed in our EDA

# We are extracting the title from name attribute and giving a value accordingly.

# In[ ]:


train_test_data = [df, test]
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False) 
for dataset in train_test_data:
    dataset["Title"]=dataset["Title"].map({"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 })
df.drop("Name",axis=1,inplace=True)
test.drop("Name",axis=1,inplace=True)    


# From EDA we observed that SibSp and Parch also have a role,so we will combine them into one feature(Family size) instead of two.

# In[ ]:


df["FamilySize"]=df["SibSp"]+df["Parch"]
test["FamilySize"]=test["SibSp"]+test["Parch"]
labels=['SibSp', 'Parch']
df.drop(labels,axis=1,inplace=True)
test.drop(labels,axis=1,inplace=True)


# 

# In[ ]:


test.head()


# Converting Dummy Variables and applying feature scaling 

# In[ ]:


# for Training set
s=pd.get_dummies(df,drop_first=True)# Dont Forget to add drop_first=True(used to avoid Dummy Variable Trap)
target=s["Survived"]
train_data=s.iloc[:,1:]
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
train_data=sc.fit_transform(train_data)


# In[ ]:


# for Test set
s=pd.get_dummies(test,drop_first=True)
sc2=StandardScaler()
test=sc.fit_transform(s)


# # Model Selection and Evaluation

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# # KNN

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier()
score=cross_val_score(clf,train_data,target,cv=k_fold)
print(score)


# In[ ]:


round(np.mean(score)*100,2)


# # SVC

# In[ ]:


from sklearn.svm import SVC
clf=SVC(kernel="rbf")
score=cross_val_score(clf,train_data,target,cv=k_fold)
print(score)


# In[ ]:


round(np.mean(score)*100,2)


# # Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()
score=cross_val_score(clf,train_data,target,cv=k_fold)
print(score)


# In[ ]:


round(np.mean(score)*100,2)


# # Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(criterion="entropy")
score=cross_val_score(clf,train_data,target,cv=k_fold)
print(score)


# In[ ]:


round(np.mean(score)*100,2)


# # Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=300,criterion="entropy")
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


round(np.mean(score)*100,2)


# # GradientBoostingClassifier

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
clf=GradientBoostingClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
round(np.mean(score)*100,2)


# ## from above models GradientBoostingClassifier is best model

# In[ ]:


model=GradientBoostingClassifier()
model.fit(train_data,target)


# In[ ]:


y_hat=model.predict(test)
main_df=pd.DataFrame(main_df)
main_df["Survived"]=y_hat
main_df.to_csv("Final Submission.csv",index=False)

