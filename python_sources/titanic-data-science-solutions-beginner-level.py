#!/usr/bin/env python
# coding: utf-8

# # Titanic: Machine Learning from Disaster
# This notebook is written during Computational Materials Design Lecture as one of the semester training projects.
# The notebook is the work on "Titanic: Machine Learning from Disaster".
# 
# ## Contents:
# 1. Introduction
# 2. Analysing Data
# 3. Feature Engineering
# 4. Modelling
# 5. Cross Validation
# 6. Creating Submission File
# 7. Further Improvements
# 8. Reference

# ## 1. Introduction
# On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. Translated 32% survival rate.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this notebook, we will try to use the knowledge from data science to analysis the relation between features and survival rate, and to predict some passengers' survival rate based on the features.

# ## 2. Analysing Data
# In this section, we will try to explore our data to know what kind of data we are dealing with and to have a brief idea of features.

# ### 2.1 Data importing and brief analysing

# We use Pandas, a Python library, to read the "train.csv" and "test.csv", and to show a first look of the dataset.

# In[ ]:


#data analysis libraries 
import numpy as np
import pandas as pd

#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#import train and test CSV files
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


#take a look at the training data
train.describe(include="all")


# In[ ]:


#show a sample of the training data
train.head()


# In[ ]:


#get a list of the features within the training dataset
print(train.columns)


# In[ ]:


#take a look at the test data
test.describe(include="all")


# In[ ]:


#show a sample of the test data
test.head()


# In[ ]:


#get a list of the features within the test dataset
print(test.columns)


# From the description and samples shown above, we could see that there are 891 rows of data and 11 features in *train.csv*, while 418 rows of data and 10 features in *test.csv*. As the target, the *Survived* feature is only shown in *train.csv* instead of *test.csv*.
# 
# Among these features, there are numerical features, categorical features and alphanumeric features. The data type of each feature is stated below:
# * Passenger ID: int
# * Survived: int
# * Pclass: int
# * Name: string
# * Sex: string
# * Age: float
# * SibSp: int
# * Parch: int
# * Ticket: string
# * Fare: float
# * Cabin: string
# * Embarked: string

# Besides, as shown below, some data in the dataset are missing. 
# In *train.csv*, 177 rows of the *Age* feature are missing (~19.9% missing); 687 rows of the *Cabin* feature are missing (~77.1% missing); 2 rows of the *Embarked* feature are missing (~0.2% missing).
# In *test.csv*, 86 rows of the *Age* feature are missing (~20.6% missing); 327 rows of the *Cabin* feature are missing (~78.2% missing); 1 row of the *Fare* feature is missing (~0.2% missing).

# In[ ]:


#count missing data in train.csv
print(pd.isnull(train).sum())


# In[ ]:


#count missing data in test.csv
print(pd.isnull(test).sum())


# ### 2.2 Data analysis and visualisation
# As we stated above, the *Survived* column would be our target, and other columns are features. We already had an idea about what type of data do we have and how it looks like. Now let's try to visualize the data in *train.csv* to see whether the features really related to suvive rate.

# ### 1) Overall survival states

# In[ ]:


#visualize the survival states
f,ax=plt.subplots(1,2,figsize=(13,5))
train['Survived'].value_counts().plot.pie(explode=[0,0.05],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived',data=train,ax=ax[1])
ax[1].set_title('Survived')
plt.show()


# Overall, in *train.csv*, only about 38.4% people survived.

# ### 2) PassengerID Feature
# PassengerID obviously doesn't relate to survival rate. So we won't analyze it too much, and later on we will drop it in Feature Engineering.

# ### 3) Pclass Feature

# In[ ]:


#draw a bar plot of survival by Pclass
sns.barplot(x="Pclass", y="Survived", data=train)
plt.show()

#print percentage of people by Pclass that survived
#print("Percentage of Pclass = 1 who survived:", train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]*100)
#print("Percentage of Pclass = 2 who survived:", train["Survived"][train["Pclass"] == 2].value_counts(normalize = True)[1]*100)
#print("Percentage of Pclass = 3 who survived:", train["Survived"][train["Pclass"] == 3].value_counts(normalize = True)[1]*100)


# From the boxplot above, we could see that passengers in *Pclass1* have highest survival rate, while passengers in *Pclass3* have lowest survival rate.

# ### 4) Name Feature
# The *Name* feature should not directly related to the survival rate. However, there are some titles included in the *Name* feature, which may give us some information later on.

# ### 5) Sex Feature

# In[ ]:


#draw a bar plot of survival by sex
sns.barplot(x="Sex", y="Survived", data=train)
plt.show()

#print percentages of females vs. males that survive
#print("Percentage of females who survived:", train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]*100)
#print("Percentage of males who survived:", train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)[1]*100)


# It is shown that females have a much higher chance of survival than males. The Sex feature is essential in our predictions.

# ### 6) Age Feature
# So as to simply visualize the *Age* feature, we group people in to 8 groups (*AgeGroup*) based on their age:
# * Unknown: Age data missing
# * Baby: 0-5 years old
# * Child: 5-18 years old
# * Young Adult: 18-35 years old
# * Adult: 35-60 years old
# * Senior: above 60 years old

# In[ ]:


#sort the ages into logical categories
'''
train["Age"] = train["Age"].fillna(-0.5)
test["Age"] = test["Age"].fillna(-0.5)
bins = [-1, 0, 5, 18, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)

#draw a bar plot of Age vs. survival
sns.barplot(x="AgeGroup", y="Survived", data=train)
plt.show()
'''
#sort the ages into logical categories
#train["Age"] = train["Age"].fillna(-0.5)
#test["Age"] = test["Age"].fillna(-0.5)
bins = [0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)

#draw a bar plot of Age vs. survival
sns.barplot(x="AgeGroup", y="Survived", data=train)
plt.show()


# We could see that babies have higher survival rate, while seniors have lower survival rate. Babies are small and easy to be kept in boats. Seniors may be less strong to survive from the cold water. But anyway, it shows us that the *Age* feature is very important.

# In[ ]:


train = train.drop(['AgeGroup'], axis = 1)
test = test.drop(['AgeGroup'], axis = 1)


# ### 7) SibSp Feature
# *SibSp* means how many siblings and spouses the passenger has aboard.

# In[ ]:


#draw a bar plot for SibSp vs. survival
sns.barplot(x="SibSp", y="Survived", data=train)
plt.show()

#I won't be printing individual percent values for all of these.
#print("Percentage of SibSp = 0 who survived:", train["Survived"][train["SibSp"] == 0].value_counts(normalize = True)[1]*100)
#print("Percentage of SibSp = 1 who survived:", train["Survived"][train["SibSp"] == 1].value_counts(normalize = True)[1]*100)
#print("Percentage of SibSp = 2 who survived:", train["Survived"][train["SibSp"] == 2].value_counts(normalize = True)[1]*100)


# In general, people with more (>2) siblings or spouses aboard were less likely to survive. However, comparing to passengers with one or two siblings or spouses, people with no siblings or spouses were also less likely to survive.

# ### 8) Parch Feature
# *Parch* means how many parents or children the passenger has aboard.

# In[ ]:


#draw a bar plot for Parch vs. survival
sns.barplot(x="Parch", y="Survived", data=train)
plt.show()


# It is shown that passengers with less than four parents or children aboard are more likely to survive than those with four or more. Again, people traveling alone are less likely to survive than those with 1-3 parents or children.

# ### 9) Ticket Feature
# It seems *Ticket* feature does not give too much information. We may drop it later.

# ### 10) Fare Feature
# As there are lots of fare, so it is diffcult to visualize the *Fare* feature like before. Here we take another strategy to visualize it ---- we calculate the average fare of the survived passengers and unsurvived passengers.

# In[ ]:


fare_not_survived = train['Fare'][train['Survived'] == 0]
fare_survived = train['Fare'][train['Survived'] == 1]

average_fare = pd.DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare = pd.DataFrame([fare_not_survived.std(), fare_survived.std()])
average_fare.plot(yerr=std_fare, kind='bar', legend=False)

plt.show()


# Survived passengers averagely have higher fare. 

# ### 11) Cabin Feature
# As more than 75% of the *Cabin* feature are missing, it is a bit hard to deal with it. Firstly, let's make two groups: one with the *Cabin* data recorded (marked as "1"), and the other with the *Cabin* data missing (marked as "0").

# In[ ]:


train["has_Cabin"] = (train["Cabin"].notnull().astype('int'))
test["has_Cabin"] = (test["Cabin"].notnull().astype('int'))

#calculate percentages of CabinBool vs. survived
#print("Percentage of CabinBool = 1 who survived:", train["Survived"][train["CabinBool"] == 1].value_counts(normalize = True)[1]*100)

#print("Percentage of CabinBool = 0 who survived:", train["Survived"][train["CabinBool"] == 0].value_counts(normalize = True)[1]*100)
#draw a bar plot of CabinBool vs. survival
sns.barplot(x="has_Cabin", y="Survived", data=train)
plt.show()


# It shows us that people with a recorded Cabin number are more likely to survive. It might is because that collecting survived passengers' Cabin information is easier. 
# But as too many *Cabin* data are missing, we will discuss later in *Feature Engineering* about how to deal with this feature.

# ## 12) Embarked feature
# There are three ports where people get on board: 
# * Southampton (marked as "S")
# * Cherbourg (marked as "C")
# * Queenstown (marked as "Q")

# In[ ]:


sns.barplot(x="Embarked", y="Survived", data=train)
#sns.barplot('Embarked', 'Survived', data=train, size=3, aspect=2)
#sns.factorplot('Embarked', 'Survived', data=train, size=3, aspect=2)
#plt.title('Embarked and Survived rate')
plt.show()


# It tells us that people embarked at Cherbourg have highest chance to survive, while people embarked at Southampton are less likely to survive. This is probably because that people from different ports have some difference on socialeconamic class and may stayed in different part of the ship (not homogeneously).

# ## 3. Feature Engineering
# As we mentioned before, some data are missing, so in this section we'll decide what to do with the missing data: either fill them with some predictions or just drop them. 
# 
# Besides, we will do encoding for the features that are not numerical.

# ### 1) Pclass Feature
# The *Pclass* feature seems important in our previous analysis, and is already numerical without any data missing, so we will keep the *Pclass* feature as it is.

# ### 2) Name Feature
# From the *Name* feature, we will extract the "title" information to be a new feature, because these title like "Lady", "Sir", "Dr" tells us a bit about the social economic situation or even approximate age range about the passengers. After extraction, we could drop the *Name* feature.

# In[ ]:


#create a combined group of both datasets
combine = [train, test]

#extract a title for each Name in the train and test datasets
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])


# In[ ]:


#replace various titles with more common names
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# The table above shows us that the survival rate obviously differs with different titles. So let's keep the title as an feature.
# 
# Then, let's encode the *Title* feature.

# In[ ]:


#map title
for dataset in combine:
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)


# In[ ]:


#to show the new column "Title"
train.head()


# In[ ]:


#drop Name feature
train = train.drop(['Name'], axis = 1)
test = test.drop(['Name'], axis = 1)


# ### 3) Sex Feature
# The *Sex* feature is not "string" type, so we need to encode it.

# In[ ]:


#map each Sex value to a numerical value
sex_mapping = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

train.head()


# ### 4) Age Feature
# Next we'll fill in the missing values in the *Age* feature. Since a higher percentage of values are missing, it would be illogical to fill all of them with the same value. Instead, we will try to predict the missing ages.

# In[ ]:


#calculate the correlation between features
train_corr = train.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
train_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
train_corr[train_corr['Feature 1'] == 'Age']


# The table above tells us , that correlation coefficient between *Age* and *Pclass* is about 0.41, while others are much lower. So we'll take *Pclass* into account for *Age* prediction.
# 
# Besides, we think some title like "Dr", "Miss" or "Mrs" can also tell us some information about age. So we would take *Title* into account, too.

# In[ ]:


#take the median value for Age feature based on 'Pclass' and 'Title'
train['Age'] = train.groupby(['Pclass', 'Title'])['Age'].apply(lambda x: x.fillna(x.median()))
test['Age'] = test.groupby(['Pclass', 'Title'])['Age'].apply(lambda x: x.fillna(x.median()))


# ### 5)SibSp and Parch Feature
# These two features are already numerical and there is no missing, so let's keep them as they are.

# ### 6)Ticket Feature
# We can also drop the Ticket feature since it's unlikely to yield any useful information

# In[ ]:


#we can also drop the Ticket feature since it's unlikely to yield any useful information
train = train.drop(['Ticket'], axis = 1)
test = test.drop(['Ticket'], axis = 1)


# ### 7) Fare Feature
# It's time separate the fare values into some logical groups as well as filling in the single missing value in the test dataset.

# In[ ]:


#fill in missing Fare value in test set based on mean fare for that Pclass 
'''
for x in range(len(test["Fare"])):
    if pd.isnull(test["Fare"][x]):
        pclass = test["Pclass"][x] #Pclass = 3
        test["Fare"][x] = round(train[train["Pclass"] == pclass]["Fare"].mean(), 4)
'''     
#map Fare values into groups of numerical values
#train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])
#test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])
for dataset in combine:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
    
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

#drop Fare values
train = train.drop(['Fare'], axis = 1)
test = test.drop(['Fare'], axis = 1)


# ### 8) Cabin Feature
# As mentioned in the last section, we could see survival rate can be affected by the *Cabin*. However, there are too many data missing in the *Cabin* feature (more than 70%), it need more advanced "domain knowledge" (like knowledge about the ship) and data processing technique to deal with the missing data. Thus, at our beginer level, we will first drop the *Cabin* feature, but keep the *has_Cabin* feature.

# In[ ]:


#drop the Cabin feature
train = train.drop(['Cabin'], axis = 1)
test = test.drop(['Cabin'], axis = 1)


# ### 9) Embarked Feature
# There are only two passenger's embarking information is missing in *train.csv*. By Google, we found that Mrs. George Nelson and her maid Miss. Amelie Icard embarked at Southampton, which can be seen from [this website](https://www.encyclopedia-titanica.org/titanic-survivor/martha-evelyn-stone.html). Thus, we fill these two missing data with "S".
# 
# Then we encode the *Embarked* feature.

# In[ ]:


#replacing the missing values in the Embarked feature with S
train = train.fillna({"Embarked": "S"})


# In[ ]:


#map each Embarked value to a numerical value
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)

train.head()


# ## 4. Modelling

# ### 4.1 Splitting the Training Data
# We will use part of our training data (20% in this case) to test the accuracy of our different models.

# In[ ]:


#count missing data in train.csv
print(pd.isnull(train).sum())


# In[ ]:


target = train["Survived"]
predictors = train.drop(['Survived', 'PassengerId'], axis = 1)


# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.20, random_state = 0)


# ### 4.2 Testing Different Models
# I will be testing the following models with my training data (got the list from [here](http://https://www.kaggle.com/startupsci/titanic-data-science-solutions)):
# * Gaussian Naive Bayes
# * Logistic Regression
# * Support Vector Machines
# * Perceptron
# * Decision Tree Classifier
# * Random Forest Classifier
# * KNN or k-Nearest Neighbors
# * Stochastic Gradient Descent
# * Gradient Boosting Classifier
# * XGBoost Classifier
# 
# For each model, we set the model, fit it with 80% of our training data, predict for 20% of the training data and check the accuracy.

# In[ ]:


predictors.head()


# In[ ]:


# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gaussian)


# In[ ]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)


# In[ ]:


# Support Vector Machines
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)


# In[ ]:


# Linear SVC
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_linear_svc)


# In[ ]:


# Perceptron
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_perceptron)


# In[ ]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)


# In[ ]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)


# In[ ]:


# KNN or k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)


# In[ ]:


# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_sgd)


# In[ ]:


# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbk)


# In[ ]:


# XGBoost
from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb.fit(x_train, y_train)
y_pred = xgb.predict(x_val)
acc_xgb = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_xgb)


# Let's compare the accuracies of each model.

# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 
              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier', 'XGBoost'],
    'Score': [acc_svc, acc_knn, acc_logreg, 
              acc_randomforest, acc_gaussian, acc_perceptron,acc_linear_svc, acc_decisiontree,
              acc_sgd, acc_gbk, acc_xgb]})
models.sort_values(by='Score', ascending=False)


# ## 5. Cross Validation
# In the previous part, the accuracy actually highly depends on the splitting of the training set (we can try different seperation percentage, then results change a lot). To avoid that, we will use cross validation. Here, K-fold cross validation is used.

# In[ ]:


from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
from xgboost import XGBClassifier

X = train.drop(['Survived', 'PassengerId'], axis=1)
Y = train["Survived"]

kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts
xyz=[]
accuracy=[]
std=[]
classifiers=['Support Vector Machines', 'K-Nearst Neighbor', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 
              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier', 'XGBoost']
models=[SVC(), KNeighborsClassifier(), LogisticRegression(), RandomForestClassifier(), GaussianNB(), 
        Perceptron(), LinearSVC(), DecisionTreeClassifier(), SGDClassifier(), 
        GradientBoostingClassifier(), XGBClassifier()]
for i in models:
    model = i
    cv_result = cross_val_score(model,X,Y, cv = kfold,scoring = "accuracy")
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)
new_models_dataframe2=pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)       
new_models_dataframe2


# We could see that, *Gradient Boosting Classifier* and *XGBoost* give the most accurate result. Thus, we would choose *XGBoost Classifier* to predict the results for *test.csv*.

# ## 6. Creating Submission File
# Now let's apply our model on *test.csv* to create a *submission.csv* file so that we can upload to the Kaggle competition.

# In[ ]:


#check missing data
print(pd.isnull(test).sum())


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


#set ids as PassengerId and predict survival 
ids = test['PassengerId']
predictions = xgb.predict(test.drop('PassengerId', axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission.csv', index=False)
print("The submission was successfully saved!")


# ## 7. Further improvement
# As for further improvement, we think there are lots of things can be done. For example,
# * ensemble models, like stacking
# * more carefully deal with features, such as the *Cabin* feature and the *Age* feature

# ## 8. Reference
# When writting this notebook, we refer a lot to the notebooks below:
# > [Nadin Tamer. Titanic Survival Predictions (Beginner)](https://www.kaggle.com/nadintamer/titanic-survival-predictions-beginner)
# 
# > [Anisotropic. Introduction to Ensembling/Stacking in Python](https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python)
# 
# > [Gunes Evitan. Titanic - Advanced Feature Engineering Tutorial](https://www.kaggle.com/gunesevitan/titanic-advanced-feature-engineering-tutorial)
# 
# > [Manav Sehgal. Titanic Data Science Solutions](https://www.kaggle.com/startupsci/titanic-data-science-solutions)
# 
# > [Vikum Sri Wijesinghe. Beginners Basic Workflow Introduction](https://www.kaggle.com/vikumsw/beginners-basic-workflow-introduction)
