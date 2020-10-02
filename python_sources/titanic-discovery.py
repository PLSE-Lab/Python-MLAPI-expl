#!/usr/bin/env python
# coding: utf-8

# # Data discovery

# Library and Data loading

# In[ ]:


import os
import numpy as np 
import pandas as pd 
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss
from sklearn.feature_selection import RFECV, RFE

sns.set(style="white") 
sns.set(style="whitegrid", color_codes=True)

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


files = os.listdir("../input")
train_data = pd.read_csv("../input/"+files[1])


# In[ ]:


train_data.head(5)


# In[ ]:


print("Tha Dataset Titanic is described with %d features and %d examples" %(train_data.shape[1] ,train_data.shape[0]))


# PassengerId : type should be integers<br>
# Survived : 1 = Survived, 0 = Not<br>
# Pclass : Class of Travel = {1,2,3}<br>
# Name : Name of Passenger<br>
# Sex : Gender = {male, female}<br>
# Age : Age of Passengers<br>
# SibSp : Number of Sibling/Spouse aboard<br>
# Parch : Number of Parent/Child aboard<br>
# Ticket : Ticket number<br>
# Fare : Ticket price<br>
# Cabin : Cabin number<br>
# Embarked : The port in which a passenger has embarked = {C - Cherbourg, S - Southampton, Q - Queenstown} 

# In[ ]:


print("% of missing values in each columns")
print(round(100*train_data.isna().sum()/train_data.shape[0],2))


# There is some missing sell calling for our help !!! <br>
# For the both columns "Age" and "Embarked" we can use some statistical technics to fill theme, but for you "Cabin" I am sorry I can't do it I will just drop you SORRY!:(<br>

# # Data preprocessing

# ## Feature engineering 

# In[ ]:


import re
def extractTitle(name):
    p = re.compile(', (.*)\.')
    #p = re.compile('(.*, )|(\\..*)')
    return p.findall(name)


# In[ ]:


train_data['title'] = train_data.Name.apply(lambda name:extractTitle(name)[0])


# In[ ]:


rare_title = ['Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']


# In[ ]:


train_data.title.replace(to_replace=['Mlle', 'Ms'], value='Miss', inplace= True)
train_data.title.replace(to_replace=['Mme'], value='Mrs', inplace= True)
train_data.title.replace(to_replace=train_data.title.where(train_data.title.isin(rare_title)), value='rare_tite', inplace= True)
train_data.title.replace(to_replace='Mrs. Martin (Elizabeth L', value='Mrs', inplace= True)


# In[ ]:


pd.crosstab(index=train_data.Sex,columns=train_data.title)


# ## Missing values 

# #### Embarked feature

# Is there any relation between the price, OF COURSE !!<br>
# we will use the boxplot to describe the Embarked stations by the price and per class 

# In[ ]:


missing_embarked = train_data.Fare[train_data.Embarked.isna()]
print("The passengers wirth missing embarked value have paied %.2f $ and %.2f $" 
      %(missing_embarked.iloc[0], missing_embarked.iloc[1]))


# Fortunatly both passengers paied some price => not much wor to do :D <br>

# In[ ]:


plt.figure(figsize=(17,20))
plt.axhline(y=80, color='red')
sns.boxplot(x='Embarked', y='Fare', data=train_data, hue='Pclass')


# The line of 80$ cross the median of the first class of Embarekment C, which means we can replace safely the missing values with "C"

# #### let's Some Fun

# People embarked from "S" are the ones with lot of outliers !!. As a Data scientists we should ask WHY ?<br>
# From some research in wikepidia, the boat started the journey from Southampton, so we can say that people coming last and were excited to live the american dream paid much more to get a ticket.
# 

# In[ ]:


train_df = train_data.copy()


# In[ ]:


train_data.Embarked.fillna(value='C', inplace = True)


# #### Age feature

# To fill the missing values in the "Age" column we can use the imputers technics. For the first version of this notebook, I will just use the median, mean values for this task

# In[ ]:


print("Percentage of missing values in Age column = "+ str(round(100 * train_data[train_data.Age.isna()].shape[0] / train_data.shape[0],2)) +"%")


# In[ ]:


plt.figure(figsize=(20,10))
sns.distplot(train_data.Age.dropna(), color='teal')


# The DF is scewed left, which means using the mean to fill the missing values will change the distribution, that is to say that the best value to use is the median

# In[ ]:


print('The mean of the "Age" is %.2f' %(train_data.Age.mean(skipna=True)))
print('The media of the "Age" is %.2f' %(train_data.Age.median(skipna=True)))


# In[ ]:


train_data.Age.fillna(train_data.Age.median(skipna = True), inplace = True)


# In[ ]:


plt.figure(figsize=(20,10))
sns.distplot(train_data.Age, color='teal', bins=15)
sns.distplot(train_df.Age.dropna(), color = 'red', bins = 15)
sns.distplot(train_df.Age.fillna(train_data.Age.mean(skipna = True)), color = 'blue', bins = 15)
plt.legend(['Adjusted Age', 'Raw Age'])


# In[ ]:


train_data.drop(['Cabin'], axis=1, inplace = True)


# In[ ]:


train_data['TravelAlone']=np.where((train_data["SibSp"]+train_data["Parch"])>0, 0, 1)
train_data.drop('SibSp', axis=1, inplace=True)
train_data.drop('Parch', axis=1, inplace=True)


# In[ ]:


training=pd.get_dummies(train_data, columns=["Pclass","Embarked","Sex","title"])
training.drop('Sex_female', axis=1, inplace=True)
training.drop('PassengerId', axis=1, inplace=True)
training.drop('Name', axis=1, inplace=True)
training.drop('Ticket', axis=1, inplace=True)


# In[ ]:


plt.figure(figsize=(15,7))
sns.kdeplot(training.Age[training.Survived == 1], color = 'blue', shade = "True")
sns.kdeplot(training.Age[training.Survived == 0], color = 'red', shade = "True")
plt.legend(['Survived', 'Died'])


# In[ ]:


plt.figure(figsize=(20,10))
survival_byage = training[["Age","Survived"]].groupby(['Age'], as_index = False).mean()
sns.barplot(x='Age', y='Survived', data=survival_byage, color = 'green')


# In[ ]:


training['IsMinor']=np.where(training['Age']<=16, 1, 0)
training['IsMinor']=np.where(training['Age']<=16, 1, 0)


# In[ ]:


plt.figure(figsize=(15,8))
sns.kdeplot(training["Fare"][training.Survived == 1], color = 'black', shade=True)
sns.kdeplot(training["Fare"][training.Survived == 0], color = 'orange', shade=True)
plt.legend(['Survived', 'Died'])


# In[ ]:


sns.barplot(x='Pclass', y='Survived', data=train_df, color="darkturquoise")


# In[ ]:


sns.barplot('Embarked', 'Survived', data=train_df, color="teal")


# In[ ]:


training.columns


# In[ ]:


sns.barplot('TravelAlone', 'Survived', data=training, color="mediumturquoise")


# In[ ]:


sns.barplot('Sex', 'Survived', data=train_df, color="aquamarine")


# In[ ]:


cols = ['Age', 'Fare', 'TravelAlone', 'Pclass_1', 'Pclass_2',
       'Pclass_3', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Sex_male',
       'title_Master', 'title_Miss', 'title_Mr', 'title_Mrs',
       'title_rare_tite', 'IsMinor']
X = training[cols]
y = training['Survived']
# Build a logreg and compute the feature importances
model = LogisticRegression()
# create the RFE model and select 8 attributes
rfe = RFE(model, 8)
rfe = rfe.fit(X, y)
# summarize the selection of the attributes
print('Selected features: %s' % list(X.columns[rfe.support_]))


# In[ ]:


# Create the RFE object and compute a cross-validated score.
# The "accuracy" scoring is proportional to the number of correct classifications
rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=10, scoring='accuracy')
rfecv.fit(X, y)

print("Optimal number of features: %d" % rfecv.n_features_)
print('Selected features: %s' % list(X.columns[rfecv.support_]))

# Plot number of features VS. cross-validation scores
plt.figure(figsize=(10,6))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


# In[ ]:


Selected_features = ['Age', 'TravelAlone', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked_C', 'Embarked_Q',
                     'Embarked_S', 'Sex_male', 'title_Master', 'title_Miss', 'title_Mr', 'title_Mrs',
                     'title_rare_tite', 'IsMinor']
X = training[Selected_features]
plt.subplots(figsize=(10, 8))
sns.heatmap(X.corr(), annot=True)
plt.show()


# In[ ]:


plt.subplots(figsize=(12, 8))
sns.heatmap(training.corr(), annot=True)
plt.show()


# train test split

# In[ ]:


# create X (features) and y (response)
X = training[Selected_features]
y = training['Survived']

# use train/test split with different random_state values
# we can change the random_state values that changes the accuracy scores
# the scores change a lot, this is why testing scores is a high-variance estimate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# check classification scores of logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)


# In[ ]:


print('Train/Test split results:')
print(logreg.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_test, y_pred))


# Model evaluation based on K-fold cross-validation using cross_val_score() function

# In[ ]:


logreg = LogisticRegression()
scores_accuracy = cross_val_score(logreg, X, y, cv=10, scoring='accuracy')


# In[ ]:


print('K-fold cross-validation results:')
print(logreg.__class__.__name__+" average accuracy is %2.3f" % scores_accuracy.mean())


# In[ ]:


scoring = {'accuracy': 'accuracy'}

modelCV = LogisticRegression()

results = cross_validate(modelCV, X, y, cv=10, scoring=list(scoring.values()), 
                         return_train_score=False)


# In[ ]:


print('K-fold cross-validation results:')
print(logreg.__class__.__name__+" average accuracy is %2.3f" % results['test_accuracy'].mean())

