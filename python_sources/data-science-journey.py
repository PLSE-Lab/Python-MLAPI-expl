#!/usr/bin/env python
# coding: utf-8

# 

# **My Data Science Journey**
# 
# This will be a 28 day journey where I aim to learn something new each day about classification and regression problems. My goal at the end of it is to place in the top 40% of submissions. Looking at the list of featured competitions (past and present), I have chosen 1 competition from each of the 2 catagories based on my interests. On days my willpower falters, my interest in these topics will hopefully help me remain focussed on my goal. 
# 
# * Classification problem:    https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
# * Regression problem:    https://www.kaggle.com/c/elo-merchant-category-recommendation
# 
# edit: have not been updating my kernels due to external datascience commitments! The journey is still on, albeit not on Kaggle. I've been too busy with my machine learning project in school so I've been unable to play on Kaggle :(
# 
# After snooping around on Kaggle for a couple of days, this is what I have gathered so far about core competecies:
# 1. Data visualisation
#     * determine correlation between target and variables
#     * analyse distribution of data
# 2. Data cleaning: 
#     * anomalies
#     * missing data
#     * wrong datatypes
# 3. Data preparation for model
#     * train/test split
#     * data transformation: normalisation, standardisation, log-transform
#     * noise filtering
#     * principal compenent analysis
# 4. Feature selection / Feature engineering
#     * manual feature engineering (domain knowledge)
#     * automated feature engineering (brute force)
#     * feature importance
#     * permutation importanace
#     * feature interaction
# 5. Final model preparation
#     * cross validation
#     * optimise hyperparameters
#     * ensemble methods: stacking, bagging, boosting
#     * evaluation metrics
#     
# Steps 1 and 2 are done concurrently. Likewise for steps 4 and 5!

# **Why do data visualisation?**
# 
# Data visualisation is a method used to summarise our data into informative pictures which helps us make conclusions about our data. 
# 
# So why do we want to make conclusions about our data? 
# 
# The end goal here is to identify features for our training set which we think will best allow our models do predict accurately. Thus, we need to determine the dependencies between features and target (qualitative) and also to determine the strength of these dependencies (quantitative). This will guide us as we explore our data using visualisations.
# 
# **Questions to ask as we go along with data visualisation**:
# 1. What conclusion can I make from this picture?
# 2. How will this conclusion affect the quality of my training set? (which will affect the quality of our model)
# 3. What can do done to solve the problem?
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV, StratifiedShuffleSplit, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, VotingClassifier
import sklearn.ensemble
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **Part 1: Data Visualistion**

# In[ ]:


df_train = pd.read_csv("/kaggle/input/titanic/train.csv")
df_test = pd.read_csv("/kaggle/input/titanic/test.csv")


# In[ ]:


df_train.head()


# In[ ]:


df_train.info()


# In[ ]:


df_train.describe()


# **Insight**
# 
# We see that are some missing values in 'Age'. When there are missing values, we can either drop the column if there are too many missing values or generate values to fill in the blanks. How many missing values are too many? There's no clear answer of course! When in doubt just run two models, with and without the particular feature and then decide. 
# 
# Furthermore, we know age will probably be correlated to survival so we keep it in this case.

# In[ ]:


df_train.describe(include=['O'])


# In[ ]:


df_test.info()


# In[ ]:


df_test.describe()


# In[ ]:


df_test.describe(include='O')


# 
# We first check the distribution of our target variable.
# 

# In[ ]:


total = df_train.Survived.count()
survived = df_train.Survived[df_train.Survived == 1].count() / total * 100
non_survived = df_train.Survived[df_train.Survived == 0].count() / total * 100
print('% survivors = ' + str(survived))
print('% non-survivors = ' + str(non_survived))


# In[ ]:


plt.bar( ['survivors', 'non-survivors'], [survived, non_survived])


# **Insight from bar chart of target variable**
# 
# 1. We see here that we are dealing with an imbalanced dataset. 
# 2. Imbalanced datasets will cause a bias in our model as one class is under-represented by the model.
# 3. Converting this into a balanced dataset or employing methods to mitigate the effects of an imbalanced dataset will probably improve our predictions.  My plan for now is  keep this at the back of my head and come back to this later when I have finished testing my baseline model. 

# In[ ]:


df_train.columns


# We then do a univariate analysis of our input variables.
# 
# **Discrete variables**
# * Nominal variables: Sex, Embarked
# * Ordinal: Pclass
# * Interval based: SibSp, Parch
# 
# **Continuous variables**
# * Age
# * Fare

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=5, figsize = (20,8))
df_train.Sex.value_counts().plot('bar', ax=axes[0], title='Sex')
df_train.Embarked.value_counts().plot('bar', ax=axes[1], title='Embarked')
df_train.Pclass.value_counts().plot('bar', ax=axes[2], title='Pclass')
df_train.SibSp.value_counts().plot('bar', ax=axes[3], title='SibSp')
df_train.Parch.value_counts().plot('bar', ax=axes[4], title='Parch')


# **Insights**
# 
# 1. Majority of people are in Pclass = 3 (3rd class passengers). We also see that majority of people embarked at S. I would start wondering, is there a correlation between the high percentage of people who embarked at port S and the high percentage of 3rd class passengers?
# 
# 2. Considering bar plots of SibSp and Parch together, majority of passengers traveled alone. One could think that a passenger who traveled alone would be able to quickly attempt to escape once the Titanic struck the iceberg while a passenger traveling with family would have to assist the family in escaping. There is possible a correlation between traveling alone and survival.
# 
# 3. Sex could also be correlated with survival. During the time period of the Titanic, chivalry probably still existed culturally so women and children would get priority in getting into life boats as passengers evacuated the sinking ship.

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (20,8))
df_train.Age.plot('hist', bins=20, ax=axes[0], title='Age')
df_train.Fare.plot('hist', bins=20, ax=axes[1], title='Fare')


# **Insights**
# 
# 1. We see a sharp spike at around age = 2, but there's no cause for alarm as this just means there are quite a few infant passengers. Similar to possible correlation between sex and survival, I would think that here is a correlation between age and survival as infants, elderly and women probably got priority in evacuation.
# 
# 2. We see the majority of the fare is around the cheaper end, which agrees which the fact that majority of passengers were 3rd class (3rd class tickets are the cheapest of course). 
# 
# 3. In the histogram for Fares, we see an outlier in the far right end. This really expensive ticket was probably only for a very selective group of individuals. We will decide later whether we will drop this outlier or not.
# 
# 4. Age histogram follows a weibull distribution somewhat and the Fare historgram follows a lognormal distribution.

# Next, we do bivariate analysis.

# In[ ]:


df_train[['Sex', 'Survived']].groupby('Sex').mean()


# This shows the conditional probably: given that a passenger is male/female, what is the probably that the passenger survived?

# In[ ]:


df_train[['Embarked', 'Survived']].groupby('Embarked').mean()


# **Insight**
# 
# We see that passengers who embarked at port S had the lowest survival rate. I would think if mainly 3rd class passengers embarked at port S as previously hypothesised, then it would make sense that their survival rate is lower compared to higher class passengers as they would have gotten priority in evacuation.

# In[ ]:


df_train[['Pclass', 'Survived']].groupby('Pclass').mean()


# **Insight**
# 
# We see a nice direct correlation between survival and passenger class. This would probably be an important feature to prediction survival.

# In[ ]:


df_train[['SibSp', 'Survived']].groupby('SibSp').mean().sort_values('Survived')


# **Insight**
# 
# We see that the passengers who had 1 or 2 SibSp had higher survival rate than those who had none and then beyond that the survival rate fall. Let's think about this. 
# 
# 1. Is it possible that those who had 1 or 2 SibSp would have been able to rely on each other, and through team effort increase their chance for survival? Conversely, those who had no SipSp may have had to struggle for survival alone thus reducing their survival rate.
# 
# 2. In contrast, with 3 or 4 SipSp we see that their survival rate actually decreased. Is it possible that this is too large a group such that coordination is made more difficult and thus survival rate is reduced?

# In[ ]:


df_train[['Parch', 'Survived']].groupby('Parch').mean().sort_values('Survived')


# **Insight**
# 
# We see that passengers traveling with 3 Parch had the highest survival rate. Adding on to the insights from SibSp vs survival rate, although I hypothesised that traveling in too large a group would decrease survival rate, but there can be an exception in the case of parents and children. 
# 
# Let us consider the case of a family with say, a father, a mother and a child. I am inclined to think that the mother and child would have a higher survival rate collectively as they would have gotten priority in evacuation. Well, it is unlikely that a mother and child would be separately evacuated.

# In[ ]:


g = sns.FacetGrid(df_train, col="Survived", height=4, aspect=2)
g.map(plt.hist, "Age", bins=20)


# In[ ]:


plt.figure(figsize=(15,8))
df_train.Age[df_train.Survived == 1].plot('hist')
df_train.Age[df_train.Survived == 0].plot('hist', alpha=0.5)
plt.legend(['survived', 'not survived'])
plt.title('Age')


# From this histogram alone, I can't really extract anything exciting so let's dive in deeper by considering the different categorical varbiales together with age.

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (20,8))
df_train.Age[(df_train.Survived == 1) & (df_train.Sex == 'male')].plot('hist', ax=axes[0])
df_train.Age[(df_train.Survived == 0) & (df_train.Sex == 'male')].plot('hist', ax = axes[0], alpha=0.5, title='Age (Male)')

df_train.Age[(df_train.Survived == 1) & (df_train.Sex == 'female')].plot('hist', ax=axes[1])
df_train.Age[(df_train.Survived == 0) & (df_train.Sex == 'female')].plot('hist', ax = axes[1], alpha=0.5, title='Age (Female)')

fig.legend(['survived', 'not survived'])


# Wow! It is not surprising that a significantly larger number of females than males survived across all ages, with the exception of around ages 6~15, which could be an anormaly. But the overal trend for females seems to be that they had higher survival rate than males regardless of age. The converse is true for the survival rates of males. This ties in well with our hypothesis mentioned above.

# In[ ]:


g = sns.FacetGrid(df_train, col="Survived", height=4, aspect=2)
g.map(plt.hist, "Fare", bins=20)


# In[ ]:


plt.figure(figsize=(15,8))
df_train.Fare[df_train.Survived == 1].plot('hist')
df_train.Fare[df_train.Survived == 0].plot('hist', alpha=0.5)
plt.legend(['survived', 'not survived'])
plt.title('Fare')


# We see that higher paying passengers had higher survival. Let's break it down further according to Pclass.

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize = (20,8))
df_train.Fare[(df_train.Survived == 1) & (df_train.Pclass == 1)].plot('hist', ax=axes[0])
df_train.Fare[(df_train.Survived == 0) & (df_train.Pclass == 1)].plot('hist', ax = axes[0], alpha=0.5, title='Fare (Pclass = 1)')

df_train.Fare[(df_train.Survived == 1) & (df_train.Pclass == 2)].plot('hist', ax=axes[1])
df_train.Fare[(df_train.Survived == 0) & (df_train.Pclass == 2)].plot('hist', ax = axes[1], alpha=0.5, title='Fare (Pclass = 2)')

df_train.Fare[(df_train.Survived == 1) & (df_train.Pclass == 3)].plot('hist', ax=axes[2])
df_train.Fare[(df_train.Survived == 0) & (df_train.Pclass == 3)].plot('hist', ax = axes[2], alpha=0.5, title='Fare (Pclass = 3)')

fig.legend(['survived', 'not survived'])


# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize = (20,8))
df_train.Embarked[(df_train.Survived == 1) & (df_train.Pclass == 1)].value_counts().plot('bar', ax=axes[0])
df_train.Embarked[(df_train.Survived == 0) & (df_train.Pclass == 1)].value_counts().plot('bar', ax = axes[0], color='r', alpha=0.5, title='Embarked (Pclass = 1)')

df_train.Embarked[(df_train.Survived == 1) & (df_train.Pclass == 2)].value_counts().plot('bar', ax=axes[1])
df_train.Embarked[(df_train.Survived == 0) & (df_train.Pclass == 2)].value_counts().plot('bar', ax = axes[1], color='r', alpha=0.5, title='Embarked (Pclass = 2)')

df_train.Embarked[(df_train.Survived == 1) & (df_train.Pclass == 3)].value_counts().plot('bar', ax=axes[2])
df_train.Embarked[(df_train.Survived == 0) & (df_train.Pclass == 3)].value_counts().plot('bar', ax = axes[2], color='r', alpha=0.5, title='Embarked (Pclass = 3)')

fig.legend(['survived', 'not survived'])


# **Insight**
# 
# Our initial hypothesis that port S could be the port where 3rd class passengers mainly board at runs contrary to what is shown here. We see that in all classes, port S is the most popular so it is unlikely that port S is mainly populated by passengers of a certain class. Port S could just be popular due to its geographic reasons and in fact port may not be that important a feature in predicting survival. We will check it out after we build our baseline model.

# In[ ]:


df_train.Fare


# In[ ]:


df_train.boxplot(by='Survived', column = ['Fare'])


# In[ ]:


df_train.boxplot(by='Survived', column = ['Age'])


# These two boxplots show that there is quite a significant number of outliers in our data when we compare 'Fare' with survival. We certainly do not want to discard all these outliers as this will result in sigificant information loss. When deploy our models, I would think that a model that is more robust to outliers will perform better. We will check this later on.

# In[ ]:


df_train.plot.scatter(x = 'Age', y = 'Fare', c='Survived', colormap = 'cool', figsize=(10,10))


# Somehow my x label (Age) is not appearing...
# 
# Personally, I am not well-versed with interpreting scatterplots so in this case here, I wonder what insights could be derived...
# 
# Can anyone reading this provide some clues or resources that might help me? 

# **Conclusion for Part 1: Data Visualisation**
# 
# 1. Exploring numerical statistics from dataset by .describe() and .info()
# 2. Univariate analysis; forming some inital hypotheses.
# 3. Bivariate analysis; conditional probability to probe at underlying relationships and confirming some of the inital hypotheses.
# 4. Exploring higher order relationshis to possibly gain more in-depth insights?
# 
# All in all, we explore the dataset using data visualisation to gain a feel for what features will likely be more important for use in our model later on.

# **Part 2: Data Cleaning**
# 
# Things to look out for:
# 
# 1. Anomalies
# 2. Missing values
# 3. Zero values (is the value really zero or is zero the dafault entry in the event of an info gap)
# 4. Incorrect datatypes
# 
# 

# In[ ]:


df_train[df_train.Pclass == 1].sort_values('Fare')


# **Insight**
# 
# We see that there are some case where Fare is 0. Could the passengers have gotten a free ride onto the Titanic? Possible as these group of people could be the employees (cabin crew, service staff, etc).

# A significant amount of 'age' data is missing in both the test and training dataset. We need to decide how we will deal with this.
# 
# We have a strong feel that age will be an important feature in predicting survival so we prioritise filling in the missing values as opposed to dropping the rows with missing 'age' data.

# **Missing Values**
# 
# 1. training set: Age
# 2. training set: Embarked
# 3. test set: Age
# 4. test set: Fare
# 

# For categaorical variable 'Embarked', we will correct for missing value by using the most frequent value.

# In[ ]:


df_train.Embarked = df_train.Embarked.fillna(df_train.Embarked.value_counts().index[0])


# For 'Age' and 'Fare', we will randomly sample values from the current distribution of values so that we can retain the overall distribution even after filling in the missing values. 

# In[ ]:


missing_age = df_train.Age.isnull()
sample = df_train.Age.dropna().sample(missing_age.sum(), replace=True).values
df_train.loc[missing_age, 'Age'] = sample


# In[ ]:


missing_age2 = df_test.Age.isnull()
sample = df_test.Age.dropna().sample(missing_age2.sum(), replace=True).values
df_test.loc[missing_age2, 'Age'] = sample


# In[ ]:


missing_fare = df_test.Fare.isnull()
sample = df_test.Fare.dropna().sample(missing_fare.sum(), replace=True).values
df_test.loc[missing_fare, 'Fare'] = sample


# When using preprocessing methods, the general consensus is to fit the methods to the train set and apply it on the test set to prevent data leakage. In this case, I would think it is more reasonable to go against the general consensus and actually fill in the missing values in the test set from random samples within the test set itself. 
# 
# In actual fact, I would think that the following two ways would result in a similar end product:
# 
# 1. Fill missing values in test set by random samples from train set.
# 2. Fill missing values in test set by random samples from test set itself.
# 
# This is because the data in train set and test set would have come from a similar distribution before being split and provided to us by Kaggle. We check this by plotting the distribution now.

# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (12,12))

df_train.Age.plot('hist', ax=axes[0,0], title='Age (train set)')
df_test.Age.plot('hist', ax=axes[0,1], title='Age (test set)')

df_train.Fare.plot('hist', ax=axes[1,0], title='Fare (train set)')
df_test.Fare.plot('hist', ax=axes[1,1], title='Fare (test set)')


# **Part 3: Data Preparation for Model**
# 
# We will do the following:
# 
# 1. Apply some form of power transformation to 'Age' and 'Fare' to check if we are able to get a normal distribution.
# 2. Split 'Age' and 'Fare' into bins.
# 4. Convert categorical string variables into numerical variables.
# 5. Deploy baseline model.

# In[ ]:


xform_age = sklearn.preprocessing.PowerTransformer(method='box-cox').fit(np.asarray(df_train.Age).reshape(len(df_train.Age),1))
xform_fare = sklearn.preprocessing.PowerTransformer(method='box-cox').fit(np.asarray(df_train.Fare + 1).reshape(len(df_train.Fare),1))

age_T = xform_age.transform(np.asarray(df_train.Age).reshape(len(df_train.Age),1))
fare_T = xform_fare.transform(np.asarray(df_train.Fare + 1).reshape(len(df_train.Fare),1))

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
ax1.hist(age_T)
ax1.title.set_text('Age (tranformed)')
ax2.hist(fare_T)
ax2.title.set_text('Fare (tranformed)')


# We managed to remove some skewness!
# 
# 
# We added a constant of + 1 to all values in 'Fare' because the box-cox transform does not deal well with the values of 0.

# The next step would be to determine the bin size. I am disinclined to use fix bin widths for either distribution. The reason for this is because there are regions where the data is densely populated and some regions that are sparsely populated. We can attempt to split the bins based on how different age groups of people are defined generally. We do the following:
# 
# * Infant + Toddlers
# * Young adults
# * Adults
# * Everyone else who is older

# In[ ]:


age_T = pd.DataFrame(age_T)
age_T.columns = ['Age']
quantile_age = xform_age.transform(np.asarray([0.1, 12, 20, 40, 80]).reshape(-1,1))
quantile_age = quantile_age.reshape(-1)
age_T['Quantile'] = pd.cut(age_T.Age, bins=quantile_age, labels = False, retbins=False)

fare_T = pd.DataFrame(fare_T)
fare_T.columns = ['Fare']
quantile_fare = xform_fare.transform(np.asarray([1, 32, 100, 350, 601]).reshape(-1,1))
quantile_fare = quantile_fare.reshape(-1)
fare_T['Quantile'] = pd.cut(fare_T.Fare, bins=quantile_fare, labels = False, retbins=False)


# In[ ]:


plt.hist(age_T.Quantile)


# In[ ]:


plt.hist(fare_T.Quantile)


# We now prepare the input features for our baseline model.
# 
# A quick look at what we have so far.

# In[ ]:


df_train.head()


# In[ ]:


input_train = df_train.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Cabin'])
input_train.head()


# In[ ]:


encoder_sex = sklearn.preprocessing.LabelEncoder().fit(input_train.Sex.to_numpy().reshape(len(input_train),1))
sex_encoded = encoder_sex.transform(input_train.Sex.to_numpy().reshape(len(input_train),1))
print(sex_encoded[0:5])


# In[ ]:


encoder_embark = sklearn.preprocessing.OneHotEncoder().fit(input_train.Embarked.to_numpy().reshape(len(input_train),1))
embarked_encoded = encoder_embark.transform(input_train.Embarked.to_numpy().reshape(len(input_train),1)).toarray()
print(encoder_embark.categories_)
print(embarked_encoded[0:5])
encoder_embark.get_feature_names(['Port'])


# In[ ]:


input_Port = pd.DataFrame(embarked_encoded)
input_Port.columns = encoder_embark.get_feature_names(['Port'])


# In[ ]:


input_Sex = pd.DataFrame(sex_encoded)
input_Sex.columns = ['Sex']


# In[ ]:


missing_fare = fare_T.Quantile[fare_T.Quantile.isnull()]


# In[ ]:


fare_T.iloc[missing_fare.index, :]


# In[ ]:


fare_T.fillna(0, inplace=True)


# In[ ]:


fare_T.isnull().sum()


# In[ ]:


input_train.Sex = input_Sex

input_train = input_train.drop(columns=['Embarked'])
input_train[list(encoder_embark.get_feature_names(['Port']))] = input_Port

input_train.Age = age_T.Quantile
input_train.Fare = fare_T.Quantile

input_train.head()


# In[ ]:


input_test = df_test.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Cabin'])

sex_encoded_test = encoder_sex.transform(input_test.Sex.to_numpy().reshape(len(input_test),1))
embarked_encoded_test = encoder_embark.transform(input_test.Embarked.to_numpy().reshape(len(input_test),1)).toarray()

input_Sex_test = pd.DataFrame(sex_encoded_test)
input_Sex_test.columns = ['Sex']
input_Port_test = pd.DataFrame(embarked_encoded_test)

input_test.Sex = input_Sex_test

input_test = input_test.drop(columns=['Embarked'])
input_test[list(encoder_embark.get_feature_names(['Port']))] = input_Port_test

input_age_T = xform_age.transform(input_test.Age.to_numpy().reshape(len(input_test.Age),1))
input_fare_T = xform_fare.transform((input_test.Fare + 1).to_numpy().reshape(len(input_test.Fare),1))

binned_age_T = pd.cut(input_age_T.reshape(len(input_age_T),), bins=quantile_age, labels=[0,1,2,3])  
binned_fare_T = pd.cut(input_fare_T.reshape(len(input_fare_T),), bins=quantile_fare, labels=[0,1,2,3])                                
                                    
input_test.Age = pd.DataFrame(binned_age_T).iloc[:,0]
input_test.Fare = pd.DataFrame(binned_fare_T).iloc[:,0]

input_test.head()


# We check if our binning process was successful

# In[ ]:


idx_age = input_test.Age[input_test.Age.isnull()].index


# In[ ]:


idx_fare = input_test.Fare[input_test.Fare.isnull()].index


# These transformed values are probably out of range of the bins used to fit the train set. We will manually consider these cases and correct them.

# In[ ]:


quantile_age


# In[ ]:


quantile_fare


# In[ ]:


for i in range(len(idx_age)):
    if binned_age_T[idx_age[i]] <= quantile_age[0]:
        input_test.Age[idx_age[i]] = 0
    else: 
        input_test.Age[idx_age[i]] = 3

for i in range(len(idx_fare)):
    if binned_fare_T[idx_fare[i]] <= quantile_fare[0]:
        input_test.Fare[idx_fare[i]] = 0
    else: 
        input_test.Fare[idx_fare[i]] = 3


# In[ ]:


input_train = input_train.astype(int)
input_test = input_test.astype(int)


# Now we can try some baseline models! We start with logistic regression.

# In[ ]:


X = input_train.drop(columns=['Survived'])
y = input_train.Survived


# In[ ]:


logR = LogisticRegression()
logR.fit(X,y)


# In[ ]:


score_logR = logR.score(X,y)
print('LogR score is ' + str(score_logR))


# In[ ]:


input_test


# In[ ]:


logR_coef = pd.DataFrame(logR.coef_.transpose())
logR_coef.columns = ['weights']
logR_coef.insert(0, 'feature', input_test.columns.values)
logR_coef.sort_values('weights')


# Looking at the magnitude of the weights, it seems that SibSp, Age and Parch seem to have a low impact on predicting survival. We shall do feature engineering later on to improve on this.

# In[ ]:


logR_pred = logR.predict(input_test)


# In[ ]:


submit = pd.DataFrame(logR_pred)
submit.columns = ['Survived']

submit.insert(0, 'PassengerId', df_test.PassengerId.values)


# In[ ]:


submit.to_csv('logR_baseline.csv', index = False)


# Next, we try support vector classification.

# In[ ]:


svc = SVC()
svc.fit(X,y)


# In[ ]:


score_svc = svc.score(X,y)
print('SVC score is ' + str(score_svc))


# In[ ]:


svc_pred = svc.predict(input_test)
submit = pd.DataFrame(svc_pred)
submit.columns = ['Survived']
submit.insert(0, 'PassengerId', df_test.PassengerId.values)
submit.to_csv('svc_baseline.csv', index = False)


# We try decision tree next.

# In[ ]:


tree = DecisionTreeClassifier()
tree.fit(X,y)
print(tree)
score_tree = tree.score(X,y)
print('Decision tree score is ' + str(score_tree))

tree_pred = tree.predict(input_test)
submit = pd.DataFrame(tree_pred)
submit.columns = ['Survived']
submit.insert(0, 'PassengerId', df_test.PassengerId.values)
submit.to_csv('tree_baseline.csv', index = False)


# Next, we try k-nearest neighbours.

# In[ ]:


knn = KNeighborsClassifier()
knn.fit(X,y)
print(knn)
score_knn = knn.score(X,y)
print('KNN score is ' + str(score_knn))

knn_pred = knn.predict(input_test)
submit = pd.DataFrame(knn_pred)
submit.columns = ['Survived']
submit.insert(0, 'PassengerId', df_test.PassengerId.values)
submit.to_csv('knn_baseline.csv', index = False)


# Lastly, we try MLPClassifier.

# In[ ]:


mlp = MLPClassifier()
mlp.fit(X,y)
print(mlp)
score_mlp = mlp.score(X,y)
print('MLP score is ' + str(score_mlp))

mlp_pred = mlp.predict(input_test)
submit = pd.DataFrame(knn_pred)
submit.columns = ['Survived']
submit.insert(0, 'PassengerId', df_test.PassengerId.values)
submit.to_csv('mlp_baseline.csv', index = False)


# In[ ]:


summary = pd.DataFrame([score_logR, score_svc, score_tree, score_knn, score_mlp])
summary.columns = ['self score']
baseline = ['logR', 'SVC', 'DecisionT', 'KNN', 'MLP']

summary.insert(0, 'baseline model', baseline)
summary.head()


# From here, we will do feature engineering followed by optimsation of hyper-parameters of our baseline model and lastly try out some ensemble methods. 
# 
# Note that the aim is to improve from our baseline model but not the point where we are overfitting our model to the training data. This is really just one huge optimsation problem were we consider the tradeoff between our model's ability to generalise to unseen data and the amount of overfitting to training data.

# **Part 4: Feature Engineering**

# What we have so far,

# In[ ]:


X


# In[ ]:


def feature_weights(model):
    try: 
        weight = model.coef_
    except AttributeError:
        weight = model.feature_importances_
    finally:
        df_weight = pd.DataFrame(weight.transpose())
        df_weight.columns = ['weights']
        df_weight.insert(0, 'feature', input_test.columns.values)
    return df_weight.sort_values('weights')


# In[ ]:


logR_coef = feature_weights(logR)
logR_coef


# In[ ]:


tree_coef = feature_weights(tree)
tree_coef


# Support vector classification, k-nearest neighbours and multi layer perceptron algorithms have no feature importance attribute.

# From the feature importance of the decision tree model, we see that the feature relating to Ports (Q/C/S) are low individually. This is because of dummy variables created when we used one-hot encoding. We actually have to consider all 3 together for this feature to make sense.

# We try combining SibSp and Parch.

# In[ ]:


X['Family'] = X.SibSp + X.Parch
X.drop(columns=['SibSp', 'Parch'], inplace=True)


# In[ ]:


input_test['Family'] = input_test.SibSp + input_test.Parch
input_test.drop(columns=['SibSp', 'Parch'], inplace=True)


# In[ ]:


X


# Since some of the algorithms have no feature importance API available, we can try the following method to see the effect of a feature on the model's predicitive power.

# In[ ]:


def all_but_one(estimators, X, y, estimator_names):
    score= []
    for i in range(len(estimators)):
        score_all = []
        model = estimators[i].fit(X,y)
        score_base = model.score(X,y)
        col_nam = X.columns.values
        score_all.append(score_base)

        for col in range(len(col_nam)):
            X_remain = X.drop(columns=[col_nam[col]])
            model_remain = estimators[i].fit(X_remain, y)
            score_remain = model_remain.score(X_remain, y)
            score_all.append(score_remain)
        score.append(score_all)
    score = pd.DataFrame(np.asarray(score).transpose())
    col_nam = list('w/o ' + col_nam)
    col_nam.insert(0, 'baseline')
    score.columns = list('self score ' + np.asarray(estimator_names, dtype=object))
    score.insert(0, 'feature', col_nam)
    return score


# In[ ]:


feature_importances = all_but_one([logR, svc, tree, knn, mlp], X.drop(columns=['Port_C','Port_Q','Port_S']), y, ['logR', 'SVC', 'DecisionT', 'KNN', 'MLP'])
feature_importances


# We see that the baseline model has a higher score throughout. This means that we cannot drop any of the current features as they contain information about our target.

# Intuitively, we would think Pclass and Fare would be correlated as tickets for a higher class would have a higher fare. We create a simple interaction term between the 2 as a new feature to see if this gives us more information about whether a passenger survives or not.

# In[ ]:


polyfeat = sklearn.preprocessing.PolynomialFeatures(include_bias=False, interaction_only=True)
interact = polyfeat.fit_transform(X.loc[:,['Pclass', 'Fare']])[:,1]
interact = interact.astype('int')


# In[ ]:


X


# In[ ]:


X_new = X.drop(columns=['Port_C','Port_Q','Port_S']).copy()
X_new.drop(columns=['Pclass', 'Fare'], inplace=True)
X_new.insert(1, 'Pclass_Fare', interact)
X_new


# In[ ]:


feature_importances2 = all_but_one([logR, svc, tree, knn, mlp], X_new, y, ['logR', 'SVC', 'DecisionT', 'KNN', 'MLP'])
feature_importances2


# We see that only the baseline score of Decision Tree model is decreased, the other models remain approximately unaffected. In this case, our new feature did not give us more information about the target. Hence, we drop the new feature.

# **Part 5: Final Model Preparation**
# 
# 1. We first perform cross validation and hyper-parameter optimsation for the models.
# 2. Select the best 3 to create an ensemble.

# In[ ]:


param_logR = {'penalty' : ['l1', 'l2'], 'tol' : [1e-3, 1e-4, 1e-5], 
              'C' : [0.3, 0.6, 1]}
param_svc = {'kernel' : ['rbf', 'linear', 'poly'], 'gamma' : ['auto', 'scale'],
             'C' : [0.3, 0.6, 1],  'tol' : [1e-3, 1e-4, 1e-5]}
param_tree = {'criterion' : ['gini', 'entropy'], 'max_depth' : [2, 4, 6], 'max_leaf_nodes' : [2, 3]}
param_knn =  {'n_neighbors' : [3, 4, 5], 'metric' : ['minkowski', 'euclidean'], 'leaf_size' : [25, 30 ,35]}
param_mlp = {'hidden_layer_sizes' : [3, 6], 'activation' : ['relu', 'logistic'], 'solver' : ['lbfgs', 'sgd', 'adam'], 
             'alpha' : [0.0002, 0.0001, 0.0005], 'learning_rate' : ['constant', 'adaptive'],  'tol' : [1e-3, 1e-4, 1e-5]}


# In[ ]:


rng = 0
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=rng)


# In[ ]:


def optimiser(estimator, param_grid, X_train, y_train, X_test, y_test, rng_instance):
    model = RandomizedSearchCV(estimator, param_distributions=param_grid, scoring='roc_auc', verbose=0,random_state=rng_instance, cv=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    roc_score = roc_auc_score(y_test, y_pred)
    print(model.best_estimator_)
    print(model.best_score_)
    print(roc_score)
    return model.best_estimator_, y_pred


# In[ ]:


tuned_logR ,y_pred_logR = optimiser(logR, param_logR, X_train, y_train, X_test, y_test, 0)
tuned_logR


# In[ ]:


tuned_svc ,y_pred_svc = optimiser(SVC(probability=True), param_svc, X_train, y_train, X_test, y_test, 0)
tuned_svc


# In[ ]:


tuned_tree ,y_pred_tree = optimiser(tree, param_tree, X_train, y_train, X_test, y_test, 0)
tuned_tree


# In[ ]:


tuned_knn ,y_pred_knn = optimiser(knn, param_knn, X_train, y_train, X_test, y_test, 0)
tuned_knn


# In[ ]:


tuned_mlp ,y_pred_mlp = optimiser(MLPClassifier(max_iter=1000), param_mlp, X_train, y_train, X_test, y_test, 0)
tuned_mlp


# **Ensemble Method: Voting**

# In[ ]:


voting = VotingClassifier(estimators=[('logR', tuned_logR), ('svc', tuned_svc), ('tree', tuned_tree), ('mlp', tuned_mlp)], voting='soft')
voting.fit(X_train,y_train)
roc_auc = roc_auc_score(y_test, voting.predict(X_test))
print(roc_auc)
voting_final = voting.fit(X, y)
y_voting = voting_final.predict(input_test)

submit_voting = pd.DataFrame(y_voting)
submit_voting.columns = ['Survived']
submit_voting.insert(0, 'PassengerId', df_test.PassengerId.values)
submit_voting.to_csv('Voting_ensemble.csv', index = False)


# **Ensemble Method: Stacking**

# In[ ]:


def Stacking(estimator, X, y, test, n_fold):
    folds = StratifiedKFold(n_fold,random_state=0)
    train_pred = np.empty((0,1),float)
    test_pred = np.empty((n_fold,test.shape[0]),float)
    for i, (train_idx,val_idx) in enumerate(folds.split(X,y)):
        x_train, x_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx],y.iloc[val_idx]
        estimator.fit(X = x_train, y = y_train)
        train_pred = np.append(train_pred, estimator.predict(x_val))
        test_pred[i] = estimator.predict(test)
    test_pred = np.mean(test_pred, axis=0)
    return pd.DataFrame(test_pred.reshape(-1,1)), pd.DataFrame(train_pred)


# In[ ]:


svc_test, svc_train = Stacking(tuned_svc, X, y, input_test, n_fold=10)
tree_test, tree_train = Stacking(tuned_tree, X, y, input_test, n_fold=10)
#logR_test, logR_train = Stacking(tuned_logR, X, y, input_test, n_fold=10)
#mlp_test, mlp_train = Stacking(tuned_mlp, X, y, input_test, n_fold=10)
knn_test, knn_train = Stacking(tuned_knn, X, y, input_test, n_fold=10)


# In[ ]:


X_stack_train = pd.concat([svc_train, tree_train, knn_train], axis=1)
X_stack_test = pd.concat([svc_test, tree_test, knn_test], axis=1)
X_stack_train.columns = X_stack_test.columns = ['svc', 'tree', 'knn']


# In[ ]:


X_stack_test


# In[ ]:


X_stack_train


# In[ ]:


param_xgb = {'n_estimators' : [500, 1000, 2000, 4000], 'eta' : [0.05, 0.1, 0.2, 0.3] , 'max_depth' : [2, 4, 6, 8], 'min_child_weight' : [1, 2, 3], 'gamma' : [0.3, 0.6, 0.9], 'subsample' : [0.5, 0.8, 1], 
              'colsample_bytree' : [0.6, 0.8, 1], 'lambda' : [0, 0.5, 0.8, 1], 'learning_rate' : [0.05, 0.1, 0.2, 0.3],  'alpha' : [0, 0.5, 0.8, 1]}


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_stack_train, y, test_size=0.1, random_state=rng)


# In[ ]:


tuned_stack, tuned_stack_pred = optimiser(xgb.XGBClassifier(), param_xgb, X_train, y_train, X_test, y_test, rng)


# In[ ]:


tuned_stack.fit(X_stack_train, y)
stack_pred = tuned_stack.predict(X_stack_test)


# In[ ]:


StackingSubmission = pd.DataFrame({ 'PassengerId': df_test.PassengerId,
                            'Survived': stack_pred})
StackingSubmission.to_csv("Stacking_ensemble.csv", index=False)


# **Ensemble Method: Bagging**

# In[ ]:


param_bag = {'base_estimator' : [tuned_logR, tuned_svc, tuned_tree, tuned_mlp], 'n_estimators' : [1000, 1500, 2000, 2500], 'max_samples': [0.25, 0.5, 0.75, 1],
            'max_features' : [0.25, 0.5, 1]}


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=rng)


# In[ ]:


#tuned_bag, tuned_bag_pre = optimiser(BaggingClassifier(), param_bag, X_train, y_train, X_test, y_test, rng)


# In[ ]:


#tuned_bag.fit(X, y)
#bag_pred = tuned_bag.predict(input_test)
#BaggingSubmission = pd.DataFrame({ 'PassengerId': df_test.PassengerId,
                          #  'Survived': bag_pred})
#BaggingSubmission.to_csv("Bagging_Submission.csv", index=False)


# With that, I concludes this notebook for now. Time to try out some past competitions!
