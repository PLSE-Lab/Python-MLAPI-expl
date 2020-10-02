#!/usr/bin/env python
# coding: utf-8

# # Titanic passenger survival prediction with feature engineering and support vector machines
# ## Introduction
# In this notebook, I present a simple way to generate (relatively) accurate predictions of titanic passenger survival using feature engineering and a support vector machines (SVM) model. The feature engineering part is based on the [greate notebook written by Sina](https://www.kaggle.com/sinakhorami/titanic-best-working-classifier). We will use Scikit-learn to implement the SVM model, and perform a simple grid search to find close-to-optimal model parameters. Let's start by importing the useful modules for our task, along with the data.

# In[ ]:


#import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
#we will use regular expression for passenger names
import re 
from sklearn import svm
from sklearn.model_selection import train_test_split

#import data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#we put both data frames in a list to modify all data easily
data = [train, test]

#data quick check
print(train.head())
print(train.info())
print('\n', test.head())
print(test.info())


# ## Predictive impact of features
# By having a quick look at the data, we see everything was imported properly with `train.head()` and `test.head()` which prints the top of the dataframe, and also that they have a relevant type with `train.info()` and `test.info()`. However, some data are missing (e.g.,  in the "Age", "Cabin" and "Embarked" features for the "train" dataframe) and we will have to deal with this.
# 
# We can check the impact of features on survival by grouping the data according to a particular feature and survival before calculating the mean survival for each group in the feature of interest. Note that we use *fancy indexing* of the dataframe to extract two columns (see [this](https://jakevdp.github.io/PythonDataScienceHandbook/03.02-data-indexing-and-selection.html) for more explanations) and the method`groupby(..., as_index=False)` to prevent the group labels from becoming an index (it's a little detail but I think it makes the result more readable).

# In[ ]:


#check the effect of passenger class on survival rate
print('Survival rate depending passenger class:')
print(train[['Pclass', 'Survived']].groupby('Pclass', as_index=False).mean(), '\n')

#check the effect of passenger sex on survival rate
print('Survival rate depending on passenger sex:')
print(train[['Sex', 'Survived']].groupby('Sex', as_index=False).mean(), '\n')

#check the effect of number of siblings and spouses on survival rate
print('Survival rate depending on number of siblings and spouses:')
print(train[['SibSp', 'Survived']].groupby('SibSp', as_index=False).mean(), '\n')

#check the effect of number of parents and children on survival rate
print('Survival rate depending on number of parents and children:')
print(train[['Parch', 'Survived']].groupby('Parch', as_index=False).mean(), '\n')


# We can see that all the features analyzed so far have a significant impact on survival rate, especially passenger sex. This is because the traditional evacuation rule "women and children first". Before verifying this for age, we need to fill missing values for this feature, since it is available for only 714 passengers out of 891 in the "train" dataset.
# 
# ## Filling missing values
# There are [several strategies to deal with missing values](https://towardsdatascience.com/handling-missing-values-in-machine-learning-part-1-dda69d4f88ca) and I chose to fill them with random values between mean - standard deviation and mean + standard deviation. As for where passengers embarked, we will fill the two missing values with one of the labels, for example 'S'. We will simply ignore the "Cabin" feature because a lot of values are missing for cabin, and this feature is unlikely to generalize well in terms of survival prediction because few passengers occupy each cabin.

# In[ ]:


#fill missing Age values
for dataset in data:
    #calculate mean, standard deviation and number of missing values
    avg = dataset['Age'].mean()
    std = dataset['Age'].std()
    null_count = dataset['Age'].isnull().sum()
    
    #generate random ages centered around the mean
    random_list = np.random.randint(avg - std, avg + std, size=null_count)
    
    #replace missing values with random ages
    dataset['Age'][np.isnan(dataset['Age'])] = random_list
    dataset['Age'] = dataset['Age'].astype(int)

#group Age values into 5 categories and check effect on survival
train['CategoricalAge'] = pd.cut(train['Age'], 5)
age_survival = train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean()

#display barplot of results
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 10))
ax1.bar([0, 1, 2, 3, 4], age_survival['Survived'])
ax1.tick_params(axis='x', color='white')
ax1.set_xticks([0, 1, 2, 3, 4])
ax1.set_xticklabels(list(age_survival['CategoricalAge'].astype(str)), fontsize=15)
ax1.set_yticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], fontsize=16)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.set_ylim(0, .6)
ax1.set_xlabel("Age categories (years)", fontsize=18)
ax1.set_ylabel("Survival rate", fontsize=18)
ax1.set_title("Impact of passenger age on survival", fontsize=18)


#fill missing "Embarked" values and show effect on survival
for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
embarked_survival = train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
ax2.bar([0, 1, 2], embarked_survival['Survived'])
ax2.set_xticks([0, 1, 2])
ax2.tick_params(axis='x', color='white')
ax2.set_xticklabels(['Cherbourg', 'Queenstown', 'Southampton'], fontsize=16)
ax2.set_yticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], fontsize=16)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.set_ylim(0, .6)
ax2.set_xlabel("Boarding location", fontsize=18)
ax2.set_ylabel("Survival rate", fontsize=18)
ax2.set_title("Impact of boarding location on survival", fontsize=18)
plt.show()


# The impact of the "Age" feature on survival confirms that children and teenagers have the highest rate of survival, as suggested earlier. Somehow, the port where passenger boarded also impacts survival.
# 
# Now we do not have missing values anymore, but we still can do a little to engineer new features. We still have to work on the "Fare" feature and we will also engineer features related to wether passengers travelled alone or not.
# 
# ## Feature engineering
# We already engineered a feature containing age categories. Let's do the same on "Fare". Then, we will create a feature indicating if a given passenger travelled alone or not, based on "Parch" (if a passenger travels with parents or children) and "SibSp" (if a passenger travelled with siblings or spouses). Finally, we have the names of passengers, which are not easy to work with in machine learning because they are strings of characters and they are almost unique to every passenger. However, we can extract the *title* of passengers (Mr., Mrs., Dr., etc.) using regular expressions and use it as a new feature. A few titles are rarer than others (e.g. Lady *versus* Mrs.) so we will group them together.

# In[ ]:


#put fare in categories and check effect on survival
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
fare_survival = train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean()

#engineer a feature indicating if a passenger travels alone
for dataset in data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
for dataset in data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
alone_survival = train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

#engineer a feature containing titles
#function to get title from name
def extract_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ''

#extract titles from name and put them in a new feature
for dataset in data:
    dataset['Title'] = dataset['Name'].apply(extract_title)

#replace rarer titles by "Rare"
for dataset in data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Jonkheer',
                                                'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
title_survival = train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

#plot data for the engineered features
fig, ax = plt.subplots(figsize=(10, 8))
ax.bar([0, 1, 2, 3], fare_survival['Survived'])
ax.tick_params(axis='x', color='white')
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(list(fare_survival['CategoricalFare'].astype(str)), fontsize=15)
ax.set_yticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_ylim(0, .6)
ax.set_xlabel("Fare category", fontsize=18)
ax.set_ylabel("Survival rate", fontsize=18)
ax.set_title("Impact of fare on survival", fontsize=18)
plt.show()

fig, ax = plt.subplots()
ax.bar([0, 1], alone_survival['Survived'])
ax.tick_params(axis='x', color='white')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Not alone', 'Alone'], fontsize=15)
ax.set_yticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_ylim(0, .6)
ax.set_ylabel("Survival rate", fontsize=18)
ax.set_title("Impact of being alone on survival", fontsize=18)
plt.show()

fig, ax = plt.subplots(figsize=(10, 8))
ax.bar([0, 1, 2, 3, 4], title_survival['Survived'])
ax.tick_params(axis='x', color='white')
ax.set_xticklabels([''] + list(title_survival['Title']), fontsize=15)
ax.set_yticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_ylim(0, 0.9)
ax.set_xlabel("Title", fontsize=18)
ax.set_ylabel("Survival rate", fontsize=18)
ax.set_title("Impact of title on survival", fontsize=18)
plt.show()


# ## Feature cleaning
# We now have a bunch of features and we need to map them to numerical values. Then, we will remove the features we will not use to train the machine learning model.

# In[ ]:


for dataset in data:
    
    #mapping sex
    dataset['Sex'] = dataset['Sex'].map({'female':0, 'male':1})
    
    #mapping titles
    title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':3, 'Rare':5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
    #mapping embarked
    dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C':1, 'Q':2})
    
    #mapping fare
    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    #mapping age
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4
    
#feature selection
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp','Parch', 'FamilySize'] 
train = train.drop(drop_elements, axis=1)
train = train.drop(['CategoricalFare', 'CategoricalAge'], axis=1)
test_no_id = test.drop(drop_elements, axis=1) #without passenger id

print('Training dataset after feature engineering:\n', train.head(), '\n')
print('Testing dataset after feature engineering:\n', test.head())


# ## Prediction of passenger survival using support vector machines
# I will not comment much on the particular choice of support vector machines (SVM) for the titanic dataset. Compared to other machine learning algorithms, I find them theoretically simple, fast to train, and easy to tune, but you may disagree with me, so I encourage you to write your thoughts in the comments. First, we will split data into training and cross-validation datasets. Then we will perform a grid search to select the best C-value, gamma-value and kernel type. I ommited polynomial kernels because it took a bit too much time to train many SVM models based on polynomial kernels.

# In[ ]:


#prepare data
#X are features
X = train.iloc[:, 1:]

#Y are targets
Y = train.iloc[:, 0]

#split train data into training and cross-validation datasets
X_train, X_cv, Y_train, Y_cv = train_test_split(X, Y, test_size=0.25)

#perform grid search to find the best parameters

#loop over grid of parameters
def grid_search():
    #list of parameters
    C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    kernels = ['linear', 'rbf', 'sigmoid']
    
    #variables to store the results
    best_score = 0
    best_C = None
    best_gamma = None
    best_kernel = None
    
    for C in C_values:
        for gamma in gamma_values:
            for kernel in kernels:
                svc = svm.SVC(C=C, gamma=gamma, kernel=kernel)
                svc.fit(X_train, Y_train)
                score = svc.score(X_cv, Y_cv)
                
                if score > best_score:
                    best_score = score
                    best_C = C
                    best_gamma = gamma
                    best_kernel = kernel
                    
    print('Best parameters give {0:.4%} accuracy'.format(best_score))
    print('C = {0}\ngamma = {1}\nKernel = {2}'.format(best_C, best_gamma, best_kernel))
    
    return best_C, best_gamma, best_kernel


# In[ ]:


#perform grid search
C, gamma, kernel = grid_search()


# Now we can use the best SVM model parameters to train the model and predict survival of passengers.

# In[ ]:


svc = svm.SVC(C=C, gamma=gamma, kernel=kernel, probability=True)
svc.fit(X, Y)
prediction = svc.predict(test_no_id)[:, np.newaxis]
prediction = np.hstack((test['PassengerId'][:, np.newaxis], prediction))
output = pd.DataFrame({'PassengerId':prediction[:, 0], 'Survived':prediction[:, 1]})
print(output.head())
np.savetxt('submission.csv', output, header='PassengerId,Survived', comments='', delimiter=',', fmt='%d')


# In[ ]:




