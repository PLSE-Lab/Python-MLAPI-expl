#!/usr/bin/env python
# coding: utf-8

# # Titanic: Who survived?
# 
# This notebook is one of my attempts to tackle the Titanic challenge. However, by far not being best-in-class, it demonstrates you how implementing a machine learning can be done. I tried to explain as much as necessary for the different steps, but I am sure you will come across some questions. If so, please do not hesitate to ask me.
# 
# ### Steps:
# __1 - Preprocessing and exploring__
# <br/>    1.1 - Load the data sets
# <br/>    1.2 - Explore the training data
# <br/>    1.3 - How to handle missing values in Age?
# <br/>    1.4 - Derive additional feature FamilySize
# <br/>    1.5 - Conclude feature engineering
# <br/><br/>
# __2 - Train and Test three different models: Random Forest, SVM and k-Nearest Neighbors__
# 
# __3 - Submission__
#     
#    

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings


# ## 1- Preprocessing and exploring

# ## 1.1 - Load the data sets

# Import data from CSV files _train.csv_ and _test.csv_ into separate pandas DataFrames. Combine both into one frame (for exploration).

# In[ ]:


train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
titanic=pd.concat([train, test], sort=False)
len_train=train.shape[0]


# ## 1.2 - Explore the training data

# Let's have a short look, how the data is structured

# In[ ]:


train.head(20)


# Find all missing values in training set

# In[ ]:


train.isnull().sum()[train.isnull().sum()>0]


# Find all missing values in test set

# In[ ]:


test.isnull().sum()[test.isnull().sum()>0]


# ### Fare

# There is only one value missing in 'Fare'. We can replace that with the mean.

# In[ ]:


test.Fare=test.Fare.fillna(titanic.Fare.mean())


# ### Cabin

# Spoiler:  Cabin might have no correlation with survival, thus we simply replace unknown cabins with value 'unknown'.
# Later on we can still decide to drop 'Cabin' or to keep it.

# In[ ]:


train.Cabin=train.Cabin.fillna("unknow")
test.Cabin=test.Cabin.fillna("unknow")


# ### Embarked

# Embarked has category values and also here only two values are missing in the train set. 
# We will replace them with the mode, which is 'S'.

# In[ ]:


train.Embarked.mode()[0]


# In[ ]:


train.Embarked=train.Embarked.fillna(train.Embarked.mode()[0])
test.Embarked=test.Embarked.fillna(train.Embarked.mode()[0])


# Now we plot what we have so far and set it into relation to 'Survived', e.g. What percentage of passengers of which 'Pclass' survived?
# I will use bar chart, but pie chart or any other visualization that helps you to see a distribution is fine as well.

# In[ ]:


warnings.filterwarnings(action="ignore")
plt.figure(figsize=[20,15])
plt.subplot(3,3,1)
sns.barplot('Pclass','Survived',data=train).set_title("Survivors by passenger class")
plt.subplot(3,3,2)
sns.barplot('SibSp','Survived',data=train).set_title("Survivors that had siblings")
plt.subplot(3,3,3)
sns.barplot('Parch','Survived',data=train).set_title("Survivors with Partner")
plt.subplot(3,3,4)
sns.barplot('Sex','Survived',data=train).set_title("Survivors by gender")
plt.subplot(3,3,5)
sns.barplot('Ticket','Survived',data=train).set_title("Correlation of Ticket")
plt.subplot(3,3,6)
sns.barplot('Cabin','Survived',data=train).set_title("Correlation of Cabin")
plt.subplot(3,3,7)
sns.barplot('Embarked','Survived',data=train).set_title("Correlation of Embarkment location")
plt.subplot(3,3,8)
sns.distplot(train[train.Survived==1].Age.dropna(), color='green', kde=False).set_title("Distribution of Survivors by Age")
sns.distplot(train[train.Survived==0].Age.dropna(), color='orange', kde=False)
plt.subplot(3,3,9)
sns.distplot(train[train.Survived==1].Fare, color='green', kde=False).set_title("Distribution of Survivors by Fare")
sns.distplot(train[train.Survived==0].Fare, color='orange', kde=False)


# ### Findings from Data exploration 
# 
# 1. Most passengers in 1st and 2nd class survived, while most in 3rd class died
# 2. Most female passengers (70%) survived, while most men died
# 3. SibSp and Parch don't seem to have a clear relationship with the target, but putting them together can be a good idea (see later on).
# 4. For Ticket and Cabin we don't see any relation, so let's forget about them.
# 5. Most survivors embarked in Cherbourg ('C')
# 6. Age matters! Passengers between 15 and 35 had the best chances to survive. However, there is a lot of Ages missing (250 across both sets). See next chapter how to handle this
# 

# ## 1.3 - How to handle missing values in 'Age' ?
# 
# From looking on the values and possible correlations, Age definitively will have a correlation to 'Survived', so we need
# to find a way to handle the missing values in 'Age'.
# Inspired by: https://www.kaggle.com/sedrak/titanic-survivals-with-age-prediction we predict the missing 'Ages' with a RandomForestClassifier instead of simply impute them with mean or median.

# The 'Title' in the name indicates a certain age group and a certain social level as well as gender, so we introduce this as a derived feature from 'Name'.
# We will just classify the titles a bit before, as there is many titles indicating same age, gender and social group.
# Defined classes are:  Officer (=male, adult), Royalty, Mrs (=female, adult), Miss (=female, junior adult), Mr (=male, adult), Master (child, infant)

# In[ ]:


train['Title']=train.Name.apply(lambda x: x.split('.')[0].split(',')[1].strip())
test['Title']=test.Name.apply(lambda x: x.split('.')[0].split(',')[1].strip())

newtitles={
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"}
train['Title']=train.Title.map(newtitles)
test['Title']=test.Title.map(newtitles)


# This would be the replacements for Age if we would use mean per group of Title and gender.

# In[ ]:


train.groupby(['Title','Sex']).Age.mean()


# When classifying, unlabeled Strings have to be converted to numerical in order to build classified, labeled data, so we map our newly created titles, as well as 'Embarked' and 'Sex' to categorical numeric data

# In[ ]:


title_mapping = {'Master': 1, 'Miss':2,"Mr": 3, "Mrs": 4, "Officer":5, "Royalty":6}
train['Title'] = train.Title.map(title_mapping)
test['Title'] = test.Title.map(title_mapping)
embarked_mapping = {'S': 1, 'C':2, 'Q': 3, '': 0}
train['Embarked'] = train.Embarked.map(embarked_mapping)
test['Embarked'] = test.Embarked.map(embarked_mapping)
gender_mapping = {'male': 0, 'female': 1}
train['Sex'] = train.Sex.map(gender_mapping)
test['Sex'] = test.Sex.map(gender_mapping)
train = train.astype({'Title': 'int32', 'Embarked' : 'int32', 'Sex' : 'int32'})
test = test.astype({'Title': 'int32', 'Embarked' : 'int32', 'Sex' : 'int32'})
train.head()


# As said before, we try a more sophisticated approach by predicting each missing Age value individually with a RandomForestClassifier.
# To train the classifier, we combine train and test dataset, we drop all columns not necessary and clean all NaN.

# In[ ]:


train_age = pd.concat([train, test], sort=False).drop(['PassengerId','Survived','SibSp','Parch','Ticket','Fare','Cabin','Name'],axis=1).dropna()
train_age = train_age.astype({'Age': 'int32'})
train_age.head(10)


# #### Now we train the Random Forest Classifier with this dataset.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=150,criterion='gini',random_state=1)
train_age_X = train_age.drop(['Age'],axis=1)
train_age_y = train_age['Age']
rf.fit(train_age_X, train_age_y)


# And predict the ages that is missing for the train dataset and the test dataset

# In[ ]:


train_predict = train.drop(['PassengerId','Survived','SibSp','Parch','Ticket','Fare','Cabin','Name','Age'],axis=1)
train['Age_predicted'] = rf.predict(train_predict)
test_predict = test.drop(['PassengerId','SibSp','Parch','Ticket','Fare','Cabin','Name','Age'],axis=1)
test['Age_predicted'] = rf.predict(test_predict)


# Check some of the predictions

# In[ ]:


train[np.isnan(train.Age)].head(20)


# In[ ]:


test[np.isnan(test.Age)].head(20)


# ## 1.4 Derive additional Feature FamilySize

# When exploring the data we recognized, that SibSp and Parch do not have a strong correlation to Survived.
# Nevertheless, families on board had a higher chance to survive than a single person, because of the rule "women and children first".
# See the statistical evidence here on the new feature FamilySize:

# In[ ]:


train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
    
pd.crosstab(train['FamilySize'], train['Survived'], normalize='index').plot(kind='bar', stacked=True, title="Survived by family size (%)")


# We see that especially families with 2-4 members have survived and will use FamilySize as an additional features for prediction.

# ## 1.5 - Conclude Feature Engineering

# After exploring the data, we have already seen that the following columns can help us a features for a model:
# - Pclass
# - Sex (gender)
# - Embarked
# - Age
# For Age we solved the issue of missing values by predicting them based on a model that was trained with the available data.
# Furthermore, we derived a new feature 'FamilySize' based on SibSp and Parch.
# 
# Hence, now we drop all other features that will not help us. PassengerID we will also drop for training a model, but will add it again later for submission.
# 

# In[ ]:


train.drop(['PassengerId','Name','Ticket','SibSp','Parch','Fare','Ticket','Cabin','Age'],axis=1,inplace=True)
test.drop(['PassengerId','Name','Ticket','SibSp','Parch','Fare','Ticket','Cabin','Age'],axis=1,inplace=True)
train.head(10)


# ## 2 - Train and Test three different models: Random Forest, SVM and k-Nearest Neighbors

# Within this chapter we will now apply threee classification models to our prepared training set. Why those three? Because I selected them for demonstration purpose. You can of course try out even more approaches and see if you can exceed the results.

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

train_survival_X = train.drop(['Survived'],axis=1)
train_survival_y = train['Survived']


# ### Random Forest

# This one is known from the age prediction. Random Forest produces a 'forest' of decision trees to finally rate them and select best matching decision tree for a prediction. The details are explained very comprehensive in this article: https://towardsdatascience.com/understanding-random-forest-58381e0602d2
# 
# We will actually try out several random forest classifier settings:
# - estimators: between 100 and 150
# - criterion: gini or entropy
# 
# and test different parameter settings by using GridSearchCV for hyper parameter tuning.
# Finally, we print the best score we reached and the parameter setting that was used to produce this score.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=1)

rf_params=[{'n_estimators':[100,150],'criterion':['gini','entropy']}]
gs_rf=GridSearchCV(estimator=rf, param_grid=rf_params, scoring='accuracy')
gs_rf.fit(train_survival_X,train_survival_y)

print (gs_rf.best_score_)
print (gs_rf.best_estimator_)


# ### SVM - Support Vector Machine

# Another classification model approach we will try is the Support Vector Machine (SVM). This is making use of support vectors, defining the hyperplane with the smallest margin. Didn't understand a word? Then read this please: https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47 
# 
# Please be patient, running SVM can really take some time.

# In[ ]:


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

svc=make_pipeline(StandardScaler(),SVC(random_state=1))
r=[0.0001,0.001,0.1,1,10,50,100]
svm_params=[{'svc__C':r, 'svc__kernel':['linear']},
      {'svc__C':r, 'svc__gamma':r, 'svc__kernel':['rbf']}]
gs_svm=GridSearchCV(estimator=svc, param_grid=svm_params, scoring='accuracy', cv=10)
gs_svm.fit(train_survival_X.astype(float), train_survival_y)
print (gs_svm.best_score_)
print (gs_svm.best_estimator_)


# ### k-Nearest Neighbors or KNN Classifier

# KNN is a rather simple algorithm, but perfect for classifying based on numerical values. It calculates the distances of data points and selects the _k_ smallest distances in order to find values that are as similar as possible.Finally it returns the mode of the labels (in our case 'Survived') for the list of _k_ smallest distances. More details here: 
# https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761
# 
# We define also here different parameters you will find in the dict *knn\_params*

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn_params = {'algorithm': ['auto'], 'weights': ['uniform', 'distance'], 'leaf_size': list(range(1,50,5)), 
               'n_neighbors': [6,7,8,9,10,11,12,14,16,18,20,22]}
gs_knn=GridSearchCV(estimator = KNeighborsClassifier(), param_grid = knn_params, scoring = "accuracy", cv=15)
gs_knn.fit(train_survival_X, train_survival_y)
print(gs_knn.best_score_)
print(gs_knn.best_estimator_)


# # 3 - Submission

# The best out of our three models was the SVM. So we will use this to predict our results we want to submit based on the provided test set.
# The results are matched with the passenger IDs and stored in a file we can submit to Kaggle.

# In[ ]:


prediction = gs_svm.best_estimator_.predict(test)

# load test set again to ensure we have the correct PassengerIDs as we dropped them before
test=pd.read_csv("../input/test.csv")
output=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':prediction})

output.to_csv('submission_rf.csv', index=False)

