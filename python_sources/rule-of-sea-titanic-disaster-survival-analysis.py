#!/usr/bin/env python
# coding: utf-8

# Hello,
# 
# Welcome aboard! The rule of Sea states that  "Women and children first" (or to a lesser extent) is a code of conduct dating from 1852, whereby the lives of women and children were to be saved first in a life-threatening situation, typically abandoning ship, when survival resources such as lifeboats were limited. https://en.wikipedia.org/wiki/Women_and_children_first
# 
# In this notebook, we will predict what kind of of people were likely to survive.

# In[89]:


# import necessary packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# machine learning packages
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[90]:


# Train and Test datasets
input_train = '../input/train.csv'
input_test = '../input/test.csv'


# In[91]:


# Reading the datasets
df_test = pd.read_csv(input_test)
df_train = pd.read_csv(input_train)
combine = [df_train, df_test]
df_train.head()


# In[92]:


df_test.head()


# The first step is to check whether the data is complete or not and then decide if it is worth to complete the missing features in the dataset.

# In[93]:


## Checking for missing data
df_train.isna().sum()


# In[94]:


# some initial analysis 
len(df_train[df_train["Sex"]=='male'])
len(df_train[df_train['Age']>60])
set(df_train['Survived'])
len(df_train[df_train["Survived"]==1])
len(df_train[df_train["Pclass"]==3])
len(df_train[df_train["SibSp"]>0])
len(df_train[df_train["Parch"]>0])
set(df_train["Fare"])


# In[95]:


df_train.describe()


# **Data Characteristics - **
# 
# *  The given data consists of information about 891 passengers. Out of them, 577 (64.7%) are males and rest 314 (35.2%) are females. 
# 
# * Mean Age is 30 years. Mostly young people are travelling. There are only 22 (2.4%) senior citizens.
# 
# * There are three different class, namely 1,2,3 with 1 being upper-class (24.2%), 2 being middle-class (20.6%) and 3 lower-class (55.1%). 
# 
# * Survived columm has two enteries, namely 0 (not survived) and 1 (survived). According to this dataset, 342 (38.4%) persons had suvived the tragedy.
# 
# * SibSp feature tells number of siblings / spouses aboard the Titanic. 283 (31.8%) had siblings/spouses aboard.
# 
# * Parch feature tells of parents / children aboard the Titanic. 213 (24%) had parents/children abroad the Titanic.
# 
# * Passenger Fare varies significantly from 0 to 512. The mean fare is 32, however, the distribution of fare is quite skewed. 

# **Incomplete/Complete features - **
# 
# *  There are three columns with missing data - Cabin, Age, Embarked. The cabin column has maximum of missing data (~77%) followed by Age column (~20%) and Embarked column (~0.002%). There is way too much missing enteries in cabin column. Let's see if we can use it later for a prediction on a subset of data. We will definitley complete the 'Age' and 'Embarked' because they might be related to the survival rate. 
# * We can drop PassengerID from our analysis as it does not contribute to survival rate.
# * We can also drop 'Name' column as it may not contribute to survival rate. However, we will create a new column 'Title' which might be related to survival. 

# **Features Study - **
# 
# Next, we will study all features to decide which are important for this problem. We will consider features one by one and make some decisions whether to include them in Model predictions.
# 
# Here are some basic analysis - 

# In[96]:


df_train[['Pclass', 'Survived']].groupby(['Pclass']).mean().sort_values(by='Survived', ascending=False)


# In[97]:


df_train[['Sex', 'Survived']].groupby(['Sex']).mean().sort_values(by='Survived', ascending=False)


# In[98]:


df_train[['SibSp', 'Survived']].groupby(['SibSp']).mean().sort_values(by='Survived', ascending=False)


# In[99]:


df_train[['Parch', 'Survived']].groupby(['Parch']).mean().sort_values(by='Survived', ascending=False)


# * There is a very high correlation between Pclass and Survived features. The upper-class passengers has 62.9% survival rate.
# * Women have more survival rate (74%) than men.
# * Parch/Sib - It has mix correlation rate for various values of Parch/Sib. Some are not correlated at all and some are highly correlated. 

# **Age Feature - **
# 
# As Age feature is continous one, we should see its distribution for survival and non-survivals.

# In[100]:


grid = sns.FacetGrid(df_train, col='Survived')
grid.map(plt.hist, 'Age', bins=20)


# Observations - 
# * Age varies from 0.42 years to 80 years with most passengers belong to age of 15 to 40 years.
# * Young children (age <=5) had high survival rate.
# * The oldest passengers (age ) had survived.
# * A large number of passengers between age 16 and 30 years did not survive.

# **Pclass Feature - **
# 
# Feature Pclass has three different values. We can see its correlation with Survival in a combine grid. 

# In[101]:


grid = sns.FacetGrid(df_train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


# In[102]:


#len(df_train[df_train['Pclass']==1])
len(df_train[df_train['Pclass']==2])
#len(df_train[df_train['Pclass']==3])


# Observations - 
# 
# * Pclass 3 has maximum passengers (55%), followed by Pclass 1 (24%) and rest is Pclass 2 (20%). However, the Pclass 3 passengers did not survive much. Pclass 1 has maximum survival passengers.
# * Children below age 5 mostly survived.

# **Embarked Feature -** 
# 
# Embarked Feature has three different entires - S, Q and C ports. We can just see its distribution with survival as - 

# In[103]:


grid = sns.catplot(x='Embarked',y='Survived',kind='point',data=df_train)
grid.add_legend();


# The survival rate for Port C is the highest (~0.55) and it is the lowest for S (~0.34).
# 
# We can further split its distribution with respect to Pclass and Sex features. 

# In[104]:


f,ax=plt.subplots(2,2,figsize=(15,15))
sns.countplot('Embarked',data=df_train,ax=ax[0,0])
ax[0,0].set_title('No. Of Passengers Boarded')
sns.countplot('Embarked',hue='Sex',data=df_train,ax=ax[0,1])
ax[0,1].set_title('Male-Female Split for Embarked')
sns.countplot('Embarked',hue='Survived',data=df_train,ax=ax[1,0])
ax[1,0].set_title('Embarked vs Survived')
sns.countplot('Embarked',hue='Pclass',data=df_train,ax=ax[1,1])
ax[1,1].set_title('Embarked vs Pclass')
plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.show()


# Observations:
# 
# * Maximum passengers boarded from S. The majority of them are from Pclass 3. However, this port has maximum passengers of all three classes in comparison to other ports.
# 
# * Inspite of higher number of passengers from Pclass 1, the survival rate is still bit low because many passengers from Pclass 3 around 81% didn't survive.
# 
# * Since small number passengers boarded from port C and they mainly belong to Pclass1, the survival rate is better than other two ports.
# 
# * Port Q had almost 95% of the passengers from Pclass 3 and hence the low survival rate.

# **Sex Feature - **

# In[105]:


sns.factorplot('Pclass','Survived',hue='Sex',col='Embarked',data=df_train)
plt.show()


# Observations:
# 
# * The survival chances are always high for Pclass 1 and Pclass 2 than Pclass 3.
# 
# * Pclass passengers of Port S (irrespective of gender) has very low survival rate in comparison to other ports.
# 
# * The males boarded from Port Q has the lowest survival rate.

# **Fare Feature - **

# In[106]:


grid = sns.FacetGrid(df_train, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Fare', alpha=.5, ci=None)
grid.add_legend()


# Observations - 
# 
# * Higher fare paying passengers had better survival rates. Thus, we can create fare ranges for our model.
# 
# * Port of embarkation correlates with survival rates. 
# 

# **Observations in a Nutshell for all features - **
# 
# * Females have better survival rates.
# * Survival rates for Pclass - 1>2>3
# * Embarkation Port correlates with survival rate. 
# * Port C has better survival rate than ports S and Q.
# * Higher fare paying passengers had better survival rates.
# * Siblings/Parch feature does not correlate much with survival rate. 

# **Feature Enginnering - **
# 
# We have analysed the dataset and made some assumptions and decisions regarding the features. Now, we will apply those decisions. This addition/extraction/removal of features is termed as feature engineering.
# 
# An example would be binning the Fare feature to use for Predictive Modeling.
# 
# Below we apply Feature Engineering to this dataset -

# **Dropping some features - **
# 
# By dropping, we can eliminate useless features. It will increase the speed and eases the analsis because the data has reduced by dropping some features.
# 
# Based on our assumptions and decisions, we would like to drop the Cabin and Ticket features as they don't correlate much with the survival rate.

# In[107]:


print("Before dropping", df_train.shape, df_test.shape, combine[0].shape, combine[1].shape)

df_train = df_train.drop(['Ticket', 'Cabin'], axis=1)
df_test = df_test.drop(['Ticket', 'Cabin'], axis=1)
combine = [df_train, df_test]

"After dropping", df_train.shape, df_test.shape, combine[0].shape, combine[1].shape


# **To Create a new feature from existing - **
# 
# Name feature is itself not a useful one. However, we can extract the Title of the passengers from it and use it to correlate with the Survival rate. 
# 
# Below we extract Title feature using regular expressions. The RegEx pattern (\w+\.) matches the first word which ends with a dot character within Name feature. The expand=False flag returns a DataFrame.

# In[108]:


for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-z]+)\.', expand=False)

pd.crosstab(df_train['Title'], df_train['Sex'])


# Observations.
# 
# Earlier we found that there is a correlation between Age and Survival feature. In order to bring out the correlation, we can band the age into different bands and these are similar to Titles.
# 
# For example: Master title has Age mean of 5 years. There is a slight variation in survival rates among Title age bands.
# Certain titles mostly survived (Mme, Lady, Sir) and certain did not survive (Don, Rev, Jonkheer).
# 
# Thus, we keep the new Title feature for model training.
# 

# We can replace many titles with a more common name or classify them as Rare.

# In[109]:


for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
df_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# We can convert the categorical titles to ordinal.

# In[110]:


dataset['Title']


# In[111]:


all_titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(all_titles)
    dataset['Title'] = dataset['Title'].fillna(0)

df_train.head()


# Now we can safely drop the Name feature from training and testing datasets. We also do not need the PassengerId feature in the training dataset.

# In[112]:


df_train = df_train.drop(['Name', 'PassengerId'], axis=1)
df_test = df_test.drop(['Name'], axis=1)
combine = [df_train, df_test]
df_train.shape, df_test.shape


# There are another two features which can be combined to make one feature - Parch and SibSp. This new feature will tell us the total family members aboarded the Ship. 
# 
# After creating this new feature, we can drop Parch and SibSp.

# In[113]:


for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

df_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# We can create another feature called IsAlone

# ** To Complete a numerical continuous feature - **
# 
# There are three features with missing enteries - Cabin, Age and Embarked. We are dropping Cabin feature. So, we should start with completing other two features - Age and Embarked. 
# 
# There are three methods to complete a numerical continuous feature.
# 
# 1. To generate random numbers between mean and standard deviation. This is simple way without worrying about the correlation between different features. 
# 
# 2. To use other correlated features which is more accurate way of guessing missing values. In our case, we note correlation among Age, Gender, and Pclass. We can replace missing Age values using median values for Age across sets of Pclass and Gender feature combinations. So, median Age for Pclass=1 and Gender=0, Pclass=1 and Gender=1, and so on...
# 
# 3. To combine methods 1 and 2. So instead of guessing age values based on median, use random numbers between mean and standard deviation, based on sets of Pclass and Gender combinations.
# 
# Method 1 and 3 will introduce random noise into our models. The results from multiple executions might vary. We will prefer method 2.

# In[114]:


grid = sns.FacetGrid(df_train, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


# Let us start by preparing an empty array to contain fill up missing Age values based on Pclass x Gender combinations.

# In[115]:


missing_ages = np.zeros((2,3))
missing_ages


# Now we iterate over Sex (0 or 1) and Pclass (1, 2, 3) to calculate guessed values of Age for the six combinations.

# In[116]:


gender=['male','female']
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == gender[i]) &                                   (dataset['Pclass'] == j+1)]['Age'].dropna()
            #print(guess_df)
            age_guess = guess_df.median()
            #print(age_guess)
            # Convert random age float to nearest .5 age
            missing_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == gender[i]) & (dataset.Pclass == j+1),                    'Age'] = missing_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

df_train.head()


# We can further create Age bands and determine correlations with Survived.

# In[117]:


df_train['AgeBand'] = pd.cut(df_train['Age'], 5)
df_train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# We replace Age with numericals based on these bands.

# In[118]:


for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
df_train.head()


# We can remove the AgeBand feature.

# In[119]:


df_train = df_train.drop(['AgeBand'], axis=1)
combine = [df_train, df_test]
df_train.head()


# We can create another feature called IsAlone

# In[120]:


for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

df_train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


# Let us drop Parch, SibSp, and FamilySize features in favor of IsAlone.

# In[121]:


df_train = df_train.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
df_test = df_test.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [df_train, df_test]

df_train.head()


# **To complete a categorical feature - **
# 
# Embarked feature takes S, Q, C values based on port of embarkation. Our training dataset has two missing values. We simply fill these with the most common occurance.

# In[122]:


freq_port = df_train.Embarked.dropna().mode()[0]
freq_port


# In[123]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# **To complete and convert a numeric feature into bands - **
# 
# The test data has one missing entry of Fare feature. We can replace this with the most common entry in this feature.
# 
# We will also round off the fare to two decimals as it represents currency.

# In[124]:


df_test['Fare'].fillna(df_test['Fare'].dropna().median(), inplace=True)
df_test.head()


# We can create FareBand.

# In[125]:


df_train['FareBand'] = pd.qcut(df_train['Fare'], 4)
df_train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


# We now convert the Fare feature to ordinal values based on the FareBand.

# In[126]:


for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

df_train = df_train.drop(['FareBand'], axis=1)
combine = [df_train, df_test]
    
df_train.head(10)


# In[127]:


df_test.head(10)


# **Converting a categorical feature into numerical one - **
# 
# Features like Sex and Embarked have alphabetic values. We can convert these strings values into numerical values. This is required by most model algorithms. 
# 
# Let us start by converting alphabetic values of Sex feature to a new feature  where female=1 and male=0.

# In[128]:


for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

df_train.head()


# We can now convert the Embarked feature by creating a new numeric Port feature. However, before that we should complete this features as there were two missing enteries in this column.

# In[129]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

df_train.head()


# ** Model, predict and solve -  **
# 
# Now its time to model and predict the survived passengers. This problem is a classification and regression type. We have to predict the relationship between the output (survived or not) with the other features (Pclass, Sex, Age etc). Here, we will perform a supervised machine learning techinque as we are training our model with a given dataset. 
# 
# Based on the current machine learning problem, i.e., Classification, Supervised Learning and Regression, we can choose following models - 
# 
# * Logistic Regression
# * KNN or k-Nearest Neighbors
# * Support Vector Machines
# * Naive Bayes classifier
# * Decision Tree
# * Random Forrest
# * Perceptron
# * Artificial neural network
# * RVM or Relevance Vector Machine

# First, we separate the input and output data. We drop the Survived column (output) from the data and make it a separate Y column.
# 

# In[130]:


def split_train_test(data,size=0.3):
        arr = np.arange(len(data))
        np.random.shuffle(arr)
        train = data.iloc[arr[0:int(len(data)*(1-size))]]
        test = data.iloc[arr[int(len(data)*(1-size)):len(data)]]
        return train,test


# In[131]:


dtrain,dtest=split_train_test(df_train,size = 0.3)
X_train=dtrain[dtrain.columns[1:]]
Y_train=dtrain[dtrain.columns[:1]]
X_test=dtest[dtest.columns[1:]]
Y_test=dtest[dtest.columns[:1]]
XX=df_train[df_train.columns[1:]]
YY=df_train['Survived']
XT=df_test[df_test.columns[1:]]


# **Logistic Regression**
# 
# Logistic Regression predicts the relationship between independent features and dependent categorical feature by estimating the probabilities using a logistic function.
# 
# Note the confidence score generated by the model based on our training dataset.

# In[132]:


# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
logreg_acc = round(accuracy_score(Y_test,Y_pred) * 100, 2)
logreg_acc


# In[133]:


coeff_df = pd.DataFrame(df_train.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)


# We can use Logistic Regression to validate our assumptions and decisions for feature creating and completing goals. This can be done by calculating the coefficient of the features in the decision function.
# 
# Positive coefficients increase the log-odds of the response (and thus increase the probability), and negative coefficients decrease the log-odds of the response (and thus decrease the probability).
# 
# * Sex correlates positively with Survival, being the highest positivie coefficient. Implies - As the Sex value increases (male: 0 to female: 1), the probability of Survived=1 increases the most.
# * Inversely as Pclass increases, probability of Survived=1 decreases the most.
# * Also Title as second highest positive correlation.

# **k-Nearest Neighbors algorithm**
# 
# The k-Nearest Neighbors algorithm (k-NN) is basically a classification by finding the most similar data points in the training data, and making an educated guess based on their classifications. A sample is classified by a majority vote of its neighbors, with the sample being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor.

# In[134]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
knn_acc = round(accuracy_score(Y_test,Y_pred) * 100, 2)
knn_acc


# **Support Vector Machines (SVM)**
# 
# A Support Vector Machine (SVM) is a discriminative classifier formally defined by a separating hyperplane. This algorithm is also used in Supervised learning. Given the labeled training data, the algorithm outputs an optimal hyperplane which categorizes new examples. In two dimentional space this hyperplane is a line dividing a plane in two parts where in each class lay in either side.

# In[135]:


# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
svc_acc = round(accuracy_score(Y_test,Y_pred) * 100, 2)
svc_acc


# **Naive Bayes classifier**
# 
# Naive Bayes classifiers is based on applying Bayes' theorem with strong (naive) independence assumptions/hypothesis between the features. In this, the best hypothesis is chosen by comparing the conditional probabilities based on different hypothesis.

# In[136]:


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
gauss_acc = round(accuracy_score(Y_test,Y_pred) * 100, 2)
gauss_acc


# The Perceptron algorithm is the simplest type of artificial neural network. It is a model of a single neuron that can be used for two-class classification problems. 

# In[137]:


# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
perc_acc = round(accuracy_score(Y_test,Y_pred) * 100, 2)
perc_acc


# In[138]:


# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
linsvc_acc = round(accuracy_score(Y_test,Y_pred) * 100, 2)
linsvc_acc


# In[139]:


# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
sgd_acc = round(accuracy_score(Y_test,Y_pred) * 100, 2)
sgd_acc


# In[140]:


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
dtree_acc = round(accuracy_score(Y_test,Y_pred) * 100, 2)
dtree_acc


# In[141]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
ran_forest_acc = round(accuracy_score(Y_test,Y_pred) * 100, 2)
ran_forest_acc


# **Model Comparison**
# 
# We now rank all the above models to choose the best one for our problem. 

# In[142]:


models = pd.DataFrame({
    'Model': ['Logistic Regression','KNN','Support Vector Machines', 
              'Naive Bayes','Perceptron','Linear SVC', 
              'Stochastic Gradient Decent','Decision Tree','Random Forest'],
    'Score': [logreg_acc,knn_acc, svc_acc,gauss_acc, 
              perc_acc,linsvc_acc,sgd_acc,dtree_acc,ran_forest_acc]})
models.sort_values(by='Score', ascending=False)


# The highest score is from Logistic Regression model. Thus submitting Y_pred for that model.

# In[143]:


Y_pred = logreg.predict(XT)
submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)


# In[ ]:




