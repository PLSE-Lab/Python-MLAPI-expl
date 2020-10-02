#!/usr/bin/env python
# coding: utf-8

# This post takes its motivation in part from the https://www.kaggle.com/startupsci/titanic-data-science-solutions tutorial .We are covering the visualization and additional visualization insights for the titanic dataset.

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


# In[ ]:


train_df=pd.read_csv('../input/train.csv')
test_df=pd.read_csv('../input/test.csv')
combine=[train_df,test_df]


# In[ ]:


train_df


# In[ ]:


#classification of features into certain types 
#categorical->survived,sex,embarked,(these can be divided into categories)
#ordinal->Pclass(this is a feature where the logical difference in categories is there ,as we move the class of tickets gets better)
#numeric features->age,Sibsp,Parch,Fare
#mixed data type - ticket,cabin
#error in typo-name
#empty values-age,cabin,Embarked
#7 int or float 
#5 string features 
train_df.info()
print('_'*100)
test_df.info()


# In[ ]:


#now let us try to generate the plots for the numeric features 
#plot for the age,Sibsp,Parch,Fare
((train_df['Age'].value_counts()/len(train_df))*100).sort_index().plot.line()


# As we can see the youngest people in the age of 20-30 are the most in number 

# In[ ]:


((train_df['SibSp'].value_counts()/len(train_df))*100).sort_index().plot.bar()


# As valeu corresponding to 0 is 0% , nearly 70% of the people are not tranveling with siblings (sibling = brother, sister, stepbrother, stepsister)

# In[ ]:


((train_df['Parch'].value_counts()/len(train_df))*100).sort_index().plot.line()


# More than 75% of the people are not travelling with their parents or children.

# In[ ]:


((train_df['Fare'].value_counts()/len(train_df))*100).sort_index().plot.line()


# The fares are concentrated in the regions of 0-100. Some people pay fares as high as 500
# 
# Next let us do the analysis for the categorical features ,
# 
# Recall:
# 1) categorical->survived,sex,embarked,(these can be divided into categories)
# 
# 2) ordinal->Pclass(this is a feature where the logical difference in categories is there ,as we move the class of tickets gets better)

# In[ ]:


#selectign the features which were(include=['O'])
train_df.describe(include=['O'])


# We can see that majority of the people here are male (577) and have embarked from the (S) station (644).
# Logically we might expect that the count of the people who are male and have survived must be greater that the number of females who have survived.Lets check that 

# In[ ]:


train_df.groupby(['Sex','Survived']).size().sort_values


# Females:314
# Males:577
# Females Survived:74.2%
# Males survived: 18.8%
# Surprise!!!!! Even though the number of males are higher , yet the more percentage of females were surviving.This means that preference was given to females during the Titanic saving efforts.

# ### Assumtions based on data analysis
# 
# We arrive at following assumptions based on data analysis done so far. We may validate these assumptions further before taking appropriate actions.
# 
# **Correlating.**
# 
# We want to know how well does each feature correlate with Survival. We want to do this early in our project and match these quick correlations with modelled correlations later in the project.
# 
# **Completing.**
# 
# 1. We may want to complete Age feature as it is definitely correlated to survival.
# 2. We may want to complete the Embarked feature as it may also correlate with survival or another important feature.
# 
# **Correcting.**
# 
# 1. Ticket feature may be dropped from our analysis as it contains high ratio of duplicates (22%) and there may not be a correlation between Ticket and survival.
# 2. Cabin feature may be dropped as it is highly incomplete or contains many null values both in training and test dataset.
# 3. PassengerId may be dropped from training dataset as it does not contribute to survival.
# 4. Name feature is relatively non-standard, may not contribute directly to survival, so maybe dropped.
# 
# **Creating.**
# 
# 1. We may want to create a new feature called Family based on Parch and SibSp to get total count of family members on board.
# 2. We may want to engineer the Name feature to extract Title as a new feature.
# 3. We may want to create new feature for Age bands. This turns a continous numerical feature into an ordinal categorical feature.
# 4. We may also want to create a Fare range feature if it helps our analysis.
# 
# **Classifying.**
# 
# We may also add to our assumptions based on the problem description noted earlier.
# 
# 1. Women (Sex=female) were more likely to have survived.
# 2. Children (Age<?) were more likely to have survived. 
# 3. The upper-class passengers (Pclass=1) were more likely to have survived.
# 
# ## Analyze by pivoting features
# 
# To confirm some of our observations and assumptions, we can quickly analyze our feature correlations by pivoting features against each other. We can only do so at this stage for features which do not have any empty values. It also makes sense doing so only for features which are categorical (Sex), ordinal (Pclass) or discrete (SibSp, Parch) type.
# 
# - **Pclass** We observe significant correlation (>0.5) among Pclass=1 and Survived (classifying #3). We decide to include this feature in our model.
# - **Sex** We confirm the observation during problem definition that Sex=female had very high survival rate at 74% (classifying #1).
# - **SibSp and Parch** These features have zero correlation for certain values. It may be best to derive a feature or a set of features from these individual features (creating #1).

# In[ ]:


train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# ## Analyze by visualizing data
# 
# Now we can continue confirming some of our assumptions using visualizations for analyzing the data.
# 
# ### Correlating numerical features
# 
# Let us start by understanding correlations between numerical features and our solution goal (Survived).
# 
# A histogram chart is useful for analyzing continous numerical variables like Age where banding or ranges will help identify useful patterns. The histogram can indicate distribution of samples using automatically defined bins or equally ranged bands. This helps us answer questions relating to specific bands (Did infants have better survival rate?)
# 
# Note that x-axis in historgram visualizations represents the count of samples or passengers.
# 
# **Observations.**
# 
# - Infants (Age <=4) had high survival rate.
# - Oldest passengers (Age = 80) survived.
# - Large number of 15-25 year olds did not survive.
# - Most passengers are in 15-35 age range.
# 
# **Decisions.**
# 
# This simple analysis confirms our assumptions as decisions for subsequent workflow stages.
# 
# - We should consider Age (our assumption classifying #2) in our model training.
# - Complete the Age feature for null values (completing #1).
# - We should band age groups (creating #3).

# In[ ]:


g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# ### Correlating numerical and ordinal features
# 
# We can combine multiple features for identifying correlations using a single plot. This can be done with numerical and categorical features which have numeric values.
# 
# **Observations.**
# 
# - Pclass=3 had most passengers, however most did not survive. Confirms our classifying assumption #2.
# - Infant passengers in Pclass=2 and Pclass=3 mostly survived. Further qualifies our classifying assumption #2.
# - Most passengers in Pclass=1 survived. Confirms our classifying assumption #3.
# - Pclass varies in terms of Age distribution of passengers.
# 
# **Decisions.**
# 
# - Consider Pclass for model training.

# In[ ]:


grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


# ### Correlating categorical features
# 
# Now we can correlate categorical features with our solution goal.
# 
# **Observations.**
# 
# - Female passengers had much better survival rate than males. Confirms classifying (#1).
# - Exception in Embarked=C where males had higher survival rate. This could be a correlation between Pclass and Embarked and in turn Pclass and Survived, not necessarily direct correlation between Embarked and Survived.
# - Males had better survival rate in Pclass=3 when compared with Pclass=2 for C and Q ports. Completing (#2).
# - Ports of embarkation have varying survival rates for Pclass=3 and among male passengers. Correlating (#1).
# 
# **Decisions.**
# 
# - Add Sex feature to model training.
# - Complete and add Embarked feature to model training.

# In[ ]:


grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()


# A small explanation how the seaborn pointplot works 
# 
# 1.  -row parameter using Embarked tells seaborn that Embarked feature varies in the rows. so the graph will now be plotted for all possible values of Embarked feature (R,C,Q)
# 2.  - pclass assumes the x axis and survivability as the y axis . The sex feature varies as the hue of the pointplot 
# 3.  -at the end , we genrate the plot for different embarked features, and the bars on the points represent the degree of accuracy in the measurement.
# 
# 

# ### Correlating categorical and numerical features
# 
# We may also want to correlate categorical features (with non-numeric values) and numeric features. We can consider correlating Embarked (Categorical non-numeric), Sex (Categorical non-numeric), Fare (Numeric continuous), with Survived (Categorical numeric).
# 
# **Observations.**
# 
# - Higher fare paying passengers had better survival. Confirms our assumption for creating (#4) fare ranges.
# - Port of embarkation correlates with survival rates. Confirms correlating (#1) and completing (#2).
# 
# **Decisions.**
# 
# - Consider handing Fare feature.

# In[ ]:


# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()


# drop features, ticket and cabin , as they have no relation with the survivability 

# In[ ]:


print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)


# now let us try to extract some indsight from the name ,(whether the person is mr. or ms etc)

# In[ ]:


for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


# Let us create a seaborn gridmap which shows how the distribution of ages corresponding to different titles is there 

# In[ ]:


sns.countplot(test_df['Title'])
print(test_df.Title.nunique())


# In[ ]:


pd.crosstab(train_df['Title'], train_df['Sex'])


# We see that a large number of the titles are rare , so let us replace them with rare keyword

# In[ ]:


for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


# In[ ]:


train_df[['Title', 'Survived']].groupby('Title',as_index=False)[['Survived']].mean()


# We have appended the titles to the dataset ,and as we can see
# 1. children boys survive less than children girls 
# 2. males survive less than females 
# 3. rare people survive around 34 percent of the time
# 

# convert categorical features into ordinals 

# In[ ]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()


# Let us drop the Name feature from training and testing datasets. We also do not need the PassengerId feature in the training dataset.

# In[ ]:


train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape


# #Converting the male /female features into 0 and 1 
# 

# In[ ]:


gender_mapping = {"male": 0, "female": 1}
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map(gender_mapping)


# In[ ]:


train_df.head()


# Now let us complete the values which are not filled in the features 
# empty values-age,cabin,Embarked
# to guess age , take median values of age across gender,pclass combination and then see the median out of that combination 
# 

# In[ ]:


grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', )
grid.map(plt.hist, 'Age', alpha=.5, bins=40)
grid.add_legend()


# In[ ]:


guess_ages = np.zeros((2,3))
guess_ages


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


# Now let us segregate the age values into bins 

# In[ ]:


train_df['AgeBand']=pd.cut(train_df['Age'],bins=5)


# In[ ]:


train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# In[ ]:


for dataset in combine:
    dataset.loc[dataset['Age']<=16,'Age']=0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 64) , 'Age'] = 4


# In[ ]:


test_df.Age.nunique()


# Remove the age band feature now 

# In[ ]:


print(train_df.shape)
train_df=train_df.drop('AgeBand',axis=1)
#recreate the combine 
combine=[train_df,test_df]


# In[ ]:


#create a new feature totalfamilysize 
for dataset in combine: 
    dataset['FamilySize']=dataset['SibSp']+dataset['Parch']+1
train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# createanother feature names IsAlone

# In[ ]:


for dataset in combine:
    dataset['IsAlone']=0
    dataset.loc[dataset['FamilySize']==1,'IsAlone']=1
    


# In[ ]:


train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


# As we can see that people who are alone are able to survive better, since they so not have to worry for their families to survive 
# 

# In[ ]:


train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

train_df.head()


# #let us create a feature from age and pclass

# In[ ]:


for dataset in combine:
    dataset['Age*Pclass']=dataset['Age']*dataset['Pclass']


# In[ ]:


train_df


# In[ ]:


freq_port = train_df.Embarked.dropna().mode()[0]
freq_port


# In[ ]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)


# Convert the categorical feature Embarked to the map integer value 

# In[ ]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()


# In[ ]:


test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]


# Now we have to create a model , a standard classification and regression problem (this is a binary classification problem )
# 

# In[ ]:


X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# In[ ]:


#logistic regression 
logreg=LogisticRegression()
logreg.fit(X_train,Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# In[ ]:


coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
coeff_df.sort_values(by='Correlation', ascending=False)


# In[ ]:


# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# In[ ]:


maximum=0
idx=1
for i in range(1,50):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
    print(acc_knn)
    if(acc_knn>maximum):
        maximum=acc_knn
        idx=i
    


# In[ ]:


idx


# In[ ]:


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# In[ ]:


# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron


# In[ ]:


# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc


# In[ ]:


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# In[ ]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
##submission.to_csv('../output/submission.csv', index=False)


# In[ ]:


submission


# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




