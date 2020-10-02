#!/usr/bin/env python
# coding: utf-8

# My first Kaggle notebook for Titanic competition followed by https://www.kaggle.com/lfl1001/titanic-data-science-solutions

# Import required Python libraries
# --------------------------------

# In[ ]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# Acquire Data
# ------------

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]


# Describing data
# ---------------
# Categorical Data:
#  'Survived', 'Sex' , 'Cabin',  'Embarked'
# 
# Ordinal (distance unknown):
# 'Pclass'
#  
# 
# Numerical Data:
# 
# Continuous: 'Age', 'Fare', 
# 
# Discrete: 'SibSp', 'Parch'
# 
# Data Dictionary
# 
# Variable	Definition	Key
# survival	Survival	0 = No, 1 = Yes
# pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
# sex	Sex	
# Age	Age in years	
# sibsp	# of siblings / spouses aboard the Titanic	
# parch	# of parents / children aboard the Titanic	
# ticket	Ticket number	
# fare	Passenger fare	
# cabin	Cabin number	
# embarked	Port of Embarkation

# In[ ]:


print(train_df.columns.values)


# In[ ]:


# preview the data
train_df.head()


# Ticket is mixed type numeric and alphanumeric. Carbin is alphanumeric.

# In[ ]:


# view tail of data
train_df.tail()


# Check which features contains blank, null or empty value.
# Five are int, two are float,  

# In[ ]:


train_df.info()
print('_'*40)
test_df.info()


# Show the descriptive of samples. 

# In[ ]:


train_df.describe()


# Distribution of numerical features
# ----------------------------------
# 
#  - Sample size is 891.  Actual passengers are on board Titanic.  
#  - 342  passengers in sample are survived. survival rate is 38%.  Actual survived rate is 32%.  
#  - Over 70% passenger did not travel with parents or children.  
#  - Around 30% passenger had sibling and/or spouse aboard.
#  - Few elderly passenger (<1%) within age range 65-80. 
#  - Fares varied significantly with few passengers (<1%) paying as high as $512.

# In[ ]:


train_df.describe(include=['O'])


# Distribution of categorical features
# ------------------------------------
# 
#  - names are unique across the data set.
#  - sex as possible value with 65% male.
#  - cabin value has few duplicated value as some passengers shared the cabin.
#  - Embarked takes three possible values. S port used by most of passengers.

# Assumptions based on data analytics
# -----------------------------------
# Completing features to survival:
# 
#  - Age, Embarked
# 
# Correcting features:
# 
#  - Ticket features dropped because of high ratio of duplicates.
#  - Cabin features because of highly null value in training and test dataset.
#  - Passengerid & name drop because it does not contribute to survival.
# 
# Creating:
# 
#  - Family features based on Parch & SibSp to get family members.
#  - Title from Name feature
#  - Age bands from numerical feature into an ordinal categorical feature.
#  - Fare rage.
# 
# Classifying:
# 
#  - Women were more likely to have survived.
#  - The upper-class passenger(PClass=1) were more likely to have survived.
#  - Children  Age lower than x were more likely to have survived.

# In[ ]:


train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# ## Correlating numerical and ordinal features ##
# 
# combine multiple features for identifying correlations using a single plot
# 
#  - Pclass=3 had most passengers, however most did not survived
#  - Infant passengers in Pclass=2 and Pclass=3 mostly survived.
#  - Most passengers in Pclass=1 survived
# 
# **Decisions.**
# 
#  - Consider Pclass for model training

# In[ ]:


# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


# ## Correlating categorical features ##
# 
# Observations:
# 
#  - Female passengers had much better survival rate than males.
#  - Exception in Embarked=C where males had higher survival rate. Could be a correlation between Pclass and Embarked and in turn Pclass and Survived, not necessarily direct correlation between Embarked and Survived. 
#  - Males had better survival rate in Pclass=3 when compared with Pclass=2 for C and Q ports.
#  - Ports of embarkation have varying survival rate for Pclass=3 and among male passengers.
# 
# Decisions.
# 
#  - Add Sex feature to model training
#  - Complete and add Embarked feature to model training

# In[ ]:


# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()


# ## Correlating categorical and numerical features ##
# 
# Correlate categorical features (with non-numeric values) and numeric features. We can consider correlating Embarked (Categorical non-numeric), Sex (Categorical non-numeric), Fare (Numeric continuous), with Survived (Categorical numeric).
# 
# **Observations:**
# 
#  - Higher fare paying passengers had better survival.
#  - Port of embarkation correlates with survival rates.
# 
# **Decisions**
# 
#  - Consider banding Fare feature.

# In[ ]:


# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()


# ## Wrangle data ##
# 
# Dropping Carbin and Ticket feature

# In[ ]:


print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape


# Creating new feature extracting from existing
# ---------------------------------------------
# Extract titles from Name before dropping Name and Passengerid features.
# 
# **Observation**
# 
#  - Most titles band Age groups accurately. For example: Master title has Age mean of 5 years.
#  - Survival among Title Age bands varies slightly.
#  - Certain titles mostly survived (Mme, Lady, Sir) or did not (Don, Rev, Jonkheer).

# In[ ]:


for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])


# Replace many titles with a more common name or classify them as Rare.

# In[ ]:


for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# Convert categorical titles to ordinal

# In[ ]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()


# Drop the Name & Passengerid from training and testing datasets.

# In[ ]:


train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape


# ## Converting a categorical feature ##
# 
# Estimating and completing features with missing or null value.

# In[ ]:


for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()


# ## Completing a numerical continuous feature ##
# Consider three method to complete a numerical continuous feature.
# 
#  1. generate random numbers between mean and SD.
#  2. Guess Age values using median value for Age cross of Pclass and Gender feature combinations. median Age for Pclass=1 and Gender=0, Pclass=1 and Gender=1 and so on...
#  3. Combine methods 1 and 2. Use random numbers between mean and standard deviation, based on sets of Pclass and Gender combinations.

# In[ ]:


# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


# Let us start by preparing an empty array to contain guessed Age values based on Pclass x Gender combinations.

# In[ ]:


guess_ages = np.zeros((2,3))
guess_ages


# Now we iterate over Sex (0 or 1) and Pclass (1, 2, 3) to calculate guessed values of Age for the six combinations.

# In[ ]:


for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) &                                   (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()


# create Age bands and determine correlations with Survived

# In[ ]:


train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# create Age bands and determine correlations with Survived.

# In[ ]:


train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# replace Age with ordinals based on these bands.

# In[ ]:


for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_df.head()


# Remove AgeBand feature

# In[ ]:


train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()


# Create new feature combining existing features
# 
# create a new feature for FamilySize which combines Parch and SibSp. This will enable us to drop Parch and SibSp from our datasets.

# In[ ]:


for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# Create another feature called IsAlone

# In[ ]:


for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


# drop Parch, SibSp, and FamilySize features in favor of IsAlone.

# In[ ]:


train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

train_df.head()


# We can also create an artificial feature combining Pclass and Age.

# In[ ]:


for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)


# Completing a categorical feature
# --------------------------------

# In[ ]:


freq_port = train_df.Embarked.dropna().mode()[0]
freq_port


# In[ ]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# Converting categorical feature to numeric
# -----------------------------------------
# convert EmbarkedFill feature by creating a new numeric Port feature

# In[ ]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()


# Fill N/A Fare feature

# In[ ]:


test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()


# create FareBand feature

# In[ ]:


train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


# Convert the Fare feature to ordinal values based on the FareBand.

# In[ ]:


for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
train_df.head(10)


# And the test dataset

# In[ ]:


test_df.head(10)


# ## Model, predict and solve ##
# 
# Out problem is a classification and regression problem. We need to identify relationship between output (Survived or not) with variables or features(Gender, Age, Port...). We perform the supervised learning and train out model with given dataset. With these two criteria - Supervised Learning plus Classification and Regression, we can narrow down our choice of model to a few algorithms:
# 
#  - Logistic Regression
#  - KNN, k-Nearest Neighbors
#  - Support Vector Machine
#  - Naive Bayes classifier
#  - Decision Tree
#  - Random Forrest
#  - Perceptron
#  - Artificial neural network
#  - RVM or Relevance Vector Machine

# In[ ]:


X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# Logistic Regression
# -------------------

# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# Calculating the coefficient of the features in the decision function.  Positive coefficients increase the log-odds of the response (and thus increase the probability), and negative coefficients decrease the log-odds of the response (and thus decrease the probability).
# 
#  - Sex is highest positivie coefficient, implying as the Sex value increases (male: 0 to female: 1), the probability of Survived=1 increases the most.
#  - Inversely as Pclass increases, probability of Survived=1 decreases the most.
#  - This way Age*Class is a good artificial feature to model as it has second highest negative correlation with Survived.
#  - So is Title as second highest positive correlation.
# 
# 

# In[ ]:


coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
coeff_df.sort_values(by='Correlation', ascending=False)


# ## Support Vector Machines ##

# In[ ]:


svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# k-Nearest Neighbors 
# -------

# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# ## Gaussian Naive Bayes ##

# In[ ]:


gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# ## Perceptron ##

# In[ ]:


perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron


# ## Linear SVC ##

# In[ ]:


linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc


# ## Stochastic Gradient Descent ##

# In[ ]:


sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd


# ## Decision Tree ##

# In[ ]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# Random Forest
# -------------
# 
# 

# In[ ]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# Model Evaluation
# ----------------
# While both Decision Tree and Random Forest score the same, we choose to use Random Forest as they correct for decision trees' habit of overfitting to their training set.

# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)

