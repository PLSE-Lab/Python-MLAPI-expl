#!/usr/bin/env python
# coding: utf-8

# ## Titanic Lab Competition Entry 2018

# Lillian Ellis // Completed 12/9/18

# **Abstract:**

# In this lab, I participated in the Kaggle Titanic competition where we worked to train a machine to predict who lived and who died on the Titanic. We were given both a training data set and a test data set to train and test our machines! As I found throughout the lab, there were some reoccuring trends for who survived and who did not. Throughout my lab, I learned a lot about how to take analysis from a data set and acctually get to make predictions on it!

# ## Import Packages

# First, I imported all of my dataset packages, along with machine learning packages to use for the lab. 

# In[ ]:


#import data packages
import math as m
import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns #(sam seaborn!!)
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# ## Input Data Sets

# Next, I imported and defined my data sets (both train and test) that I will use throughout the rest of my lab. Throughout my lab, I made sure that everything I did for test, I also did for train, to ensure both data sets contained the same values. 

# In[ ]:


#Input csv filed into two data fram object: train_df and test_df
#train = 891 entries of passengers, features, with labels if survived or died 
#test = 418 entries of titanic passengers, features, no labels
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]


# Note: Throughout my lab, I greatly referred to Manav Sehgal's notebook on the Kaggle competition. In Manav's notebook, he includes the combine data set defined above (test and train combined). I had some troubles with combine throughout my lab, but it was helpful when converting male and female passengers to 0 and 1. Here is the link to his notebook: https://www.kaggle.com/startupsci/titanic-data-science-solutions 

# ## Check/Analyze Data

# First, I wanted to take a clean look at the first 10 lines of data from the train set (which I have printed below) to get a look at what trends I notice off the bat, what the headers are, and what I may need to clean. 

# In[ ]:


#Examine first 10 rows to take a look at the data 
#note: passengerID randomly assigned by Kaggle (correlates with index)
train_df[:10]


# Most of my data looks pretty clean, but I do notice that there are a few missing values. I next printed the info of train to get an overview of the entire data set. 

# **Printing Train Info**

# In[ ]:


#print info
train_df.info()


# From looking at this, we have lots of data for everything execpt for age (limited knowledge) and cabin (cabin numbers for 1st class). Data is also missing data for 2 passengers on where they embarked. We have fairly solid and consistent data with all of the other columns, which have 891 values each. 

# **Printing Test Info**

# In[ ]:


#lookin at test set
print(test_df.info())


# Still lots of missing ages, cabins (lots of missing data!), one fare missing. There are integer, object, and float values. 
# 
# Overall:
# - both sets have all features, there is nothing off here
# - most features are pretty complete exept for age, cabin
# - some missing features surrounding embarking location and fare

# ## Feature Analysis and Clean Up of Data

# I conducted some analysis and clean up on the features to determine a good set to work with when training my classifier- based on the features that will tell us the labels (these must be numerical)
# 
# Determining:
# - which features to get rid of 
# - which features to clean
# - **which features correlate to labels (and how accurate each one is)**
# - how to handle missing data
# - maybe we will need to combine data (gender and class? gender and embarked? etc.)
# 
# Need to clean up features for both training and test at the same time (so these align and chance at accurate feature is good)

# List of Data Cleaning Tasks:
# - print basic correlations between passenger, class, sex, SipSp, Embarked, etc. (as these are not missing much data, pretty telling)
# - print graphs to see other correlations 
# - change female and male to 1 and 0
# - combine SibSp and Parch into family 
# - Create other values based on finds from correlations

# In[ ]:


train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False) 


# In[ ]:


train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train_df[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False) 


# Citation for correlation tests: https://www.kaggle.com/startupsci/titanic-data-science-solutions lines 9-12 from Manav Sehgal. Throughout my lab, I referred to his notebook, as it was a very helpful way to get started. I only used a few pieces of his code, but referring to his code on data analysis, and changing female and male to integers was really helpful. 

# Here I printed some graphs to make the correlation tests above easier to read. 

# In[ ]:


#sns.barplot(x = "Family", y = "Survived", data = train_df)


# (note: I commented out my first graph, although it worked, because it was using too much memory and it kept erroring out) 

# In[ ]:


# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
#grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
#grid.map(plt.hist, 'Age', alpha=.5, bins=20)
#grid.add_legend();


# In[ ]:


# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()


# **Main Takeaways from Correlations:**

# - sex is the biggest indicator of survival!
#     - 74% of females survive , 19% of males survive
# - class is also a pretty big indicator 
#     - 63% of those in 1st class survived, 47% in 2nd, and 24% in 3rd
# - whether you were alone or not is also pretty big!
#     - the more family members you had on board, the less likely you were to survive 

# So, I decided to especially focus on sex, class, and family members. 

# My first step in cleaning the sex column was to convert females to 1 and males to 0. I got this code from Manav's notebook. 

# In[ ]:


# changing males to 0 and females to 1 , changing the type to an integer 
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()


# Then, I combined columns for parents/children and sibling/spouses to just family members- because there is a consistant linear regression: those who had more family members were less likely to survive. 

# In[ ]:


#combining Sibling and spouses with parents and children to create a new family column 
train_df["Family"] = train_df["SibSp"] + train_df["Parch"]
test_df["Family"] = test_df["SibSp"] + test_df["Parch"] 

train_df.head()


# Next, I deleted  sibsp and parch in both data sets - as I will now instead using new column of family 

# In[ ]:


#drop columns that Family replaces 
train_df.drop('Parch', axis = 1, inplace = True)
train_df.drop('SibSp', axis = 1, inplace = True)
test_df.drop('Parch', axis = 1, inplace = True)
test_df.drop('SibSp', axis = 1, inplace = True)

train_df.head()


# However, I then decided to create an "IsAlone" column- which indicates if you have family members or are all alone (because if you are alone- higher chance of survival). This new column simplified data, and eliminated potential outliers. I also got this idea from Manav, but I modified it as I was having trouble with the "combine" data set. 

# In[ ]:


test_df["IsAlone"] = 0
train_df["IsAlone"] = 0
train_df.loc[train_df['Family'] == 1, 'IsAlone'] = 1
test_df.loc[test_df['Family'] == 1, 'IsAlone'] = 1

#This was Manav's code (which I modified)
#for dataset in combine:
    #dataset['IsAlone'] = 0
    #dataset.loc[dataset['Family'] == 1, 'IsAlone'] = 1

train_df.info()


# As you can see below, being alone was a clear advantage, as 34% of those not alone survived, while 55% of those who were alone survived!

# In[ ]:


train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


# The last step I took before training my machine was deleting all of the columns other than the ones I found were the most relavant, telling, full, and now clean! (IsAlone, Class, and Sex). I am also keeping passenger Id in both, which I will delete later, but I need it for identifying each passenger for my final submission. 

# In[ ]:


for val in train_df.columns:
    if val != 'IsAlone' and val != 'PassengerId' and val != 'Pclass' and val != 'Sex' and val != 'Survived':
        train_df = train_df.drop([val], axis=1)
for val in test_df.columns:
    if val != 'IsAlone' and val != 'PassengerId' and val != 'Pclass' and val != 'Sex':
        test_df = test_df.drop([val], axis=1)


# These are what my two data sets now look like:

# In[ ]:


train_df.head()


# In[ ]:


test_df.head() 


# ## Training Classifiers

# Now that I am done with cleaning, and selected the data I want to keep in both, I am ready to train my classifier. I first created the machine, fed in the feature set to train the machine with my labels, and then started to make predictions with 6 different classifiers and printed the accuracy method! 

# TO DO:
# - create machine
# - feed feature set to train machine with the labels
# - start to make predictions and get accuracy rate up! 

# In[ ]:


#split training data into the feature list w/o passenger id and labels (survived or not)
# features - x
# labels - y
X_train = train_df.drop(["Survived", "PassengerId"], axis=1)
Y_train = train_df["Survived"]

X_test = test_df.drop('PassengerId', axis=1)

X_train.head()
Y_train.head()


# In[ ]:


#from sklearn.cross_validation import train_test_split
#train_test_split splits our data for us into training and testing! (in a cross validation)
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split

#this means a 70-30 split, pulling out 30% of data for test size (what percent of the data goes into the test)
#4 variables are returned from the method call

valid_X_train, valid_X_test, valid_Y_train, valid_Y_test = train_test_split(X_train, Y_train, train_size = .7, test_size = .3)

valid_X_train.shape, valid_X_test.shape, valid_Y_train.shape, valid_Y_test.shape 
 


# In[ ]:


# Logistic Regression
logreg = LogisticRegression()
logreg.fit(valid_X_train, valid_Y_train)
Y_pred = logreg.predict(valid_X_test)

print(accuracy_score(valid_Y_test, Y_pred)* 100)

#a check to see if data overfitted- if these numbers are close together, then the data is not overfit 
#this uses the method of the used classifier
# I got this specific code from Ms. Sconyers!
print((logreg.score(X_train, Y_train)*100), logreg.score(valid_X_train, valid_Y_train)*100)


# In[ ]:


svc = SVC()
svc.fit(valid_X_train, valid_Y_train)
Y_pred = svc.predict(valid_X_test)
#acc_svc = round(svc.score(valid_X_train, valid_Y_train) * 100, 2)
#acc_svc

print(accuracy_score(valid_Y_test, Y_pred)* 100)

#a check to see if data overfitted- if these numbers are close together, then the data is not overfit 
print((svc.score(X_train, Y_train)*100), svc.score(valid_X_train, valid_Y_train)*100)


# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(valid_X_train, valid_Y_train)
Y_pred = knn.predict(valid_X_test)
#acc_knn = round(knn.score(valid_X_train, valid_Y_train) * 100, 2)
#acc_knn

print(accuracy_score(valid_Y_test, Y_pred)* 100)

#a check to see if data overfitted- if these numbers are close together, then the data is not overfit 
print((knn.score(X_train, Y_train)*100), knn.score(valid_X_train, valid_Y_train)*100)


# In[ ]:


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(valid_X_train, valid_Y_train)
Y_pred = gaussian.predict(valid_X_test)
#acc_gaussian = round(gaussian.score(valid_X_train, valid_Y_train) * 100, 2)
#acc_gaussian

print(accuracy_score(valid_Y_test, Y_pred)* 100)

#a check to see if data overfitted- if these numbers are close together, then the data is not overfit 
print((gaussian.score(X_train, Y_train)*100), gaussian.score(valid_X_train, valid_Y_train)*100)


# Both decision tree and random forest consistently print an accuracy score around 80, which is highest out of all of my classifiers. 

# In[ ]:


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(valid_X_train, valid_Y_train)
Y_pred = decision_tree.predict(valid_X_test)
#acc_decision_tree = round(decision_tree.score(valid_X_train, valid_Y_train) * 100, 2)
#acc_decision_tree

print(accuracy_score(valid_Y_test, Y_pred)* 100)

#a check to see if data overfitted- if these numbers are close together, then the data is not overfit 
print((decision_tree.score(X_train, Y_train)*100), decision_tree.score(valid_X_train, valid_Y_train)*100)


# In[ ]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(valid_X_train, valid_Y_train)
Y_pred = random_forest.predict(valid_X_test)
random_forest.score(valid_X_train, valid_Y_train)
#acc_random_forest = round(random_forest.score(valid_X_train, valid_Y_train) * 100, 2)
#acc_random_forest

print(accuracy_score(valid_Y_test, Y_pred)* 100)

#a check to see if data overfitted- if these numbers are close together, then the data is not overfit 
print((random_forest.score(X_train, Y_train)*100), random_forest.score(valid_X_train, valid_Y_train)*100)


# ## Pick the Best Classifier and Create Submission 

# Now that I have finished printing my accuracy, I have decided that I will use logreg as my submission and winning classifier, as it consistently prints an accuracy score around 80. I will now predict with the X_test (not valid_X_test), and create a submission csv with that classifier. 

# In[ ]:


tree_pred = decision_tree.predict(X_test)


# In[ ]:


final_pred = tree_pred
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': final_pred
})
submission.to_csv('submission.csv', index=False)


# In[ ]:


submission.info()


# **Areas of Future Exploration:**

# - I would have really loved to have achieved a higher accuracy score, and I think I could have done this by changing the weight for the various columns I ended up training my classifiers with
#     - for example, because sex is the greatest determinate of survival, it would be great if I could find a way to have that column carry more weight overall. I think this could be done with neural networks, and I hope to explore this in the future! 
#     

# **Additional Notes:**

# - I tried to combine different columns (other than just SibSp and Parch) into a combined data to maybe give more weight to one column over the other, as I thought that overlapping would boost my accuracy score
#     - I tried a few different combinations, and ultimatley decided that it acctually greatly decreased my accuracy score
#     - In the future, however, I hope to find out different ways to overlap data 

# **Questions I still Have:**

# - Why didn't combine work for my data set? (still a mystery)
# - Julia ended up getting a higher accuracy score than mine: using sex, passenger Id, and fare
#     - But when I tried just using the same features she used, my accuracy score greatly decreased
#     - Why did that happen? 

# **Acknowledgements:**

# - Manav Sehgal greatly helped me with my code, and I used a lot of strategy and code from him. His notebook allowed me to analyze effectively and I learned many new classifiers from his notebook. 
# - Ms. Sconyers greatly helped me navigate through errors in my code, and helped me strategize on which data I ended up using for training my classifiers. I couldn't have completed this lab without all of her help!

# Links to Sources:
# 
# https://towardsdatascience.com/predicting-the-survival-of-titanic-passengers-30870ccc7e8 
# 
# https://www.kaggle.com/startupsci/titanic-data-science-solutions
