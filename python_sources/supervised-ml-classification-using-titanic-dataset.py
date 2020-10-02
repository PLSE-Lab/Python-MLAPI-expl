#!/usr/bin/env python
# coding: utf-8

# # 1. Import Libraries

# ## Check library versions

# In[ ]:


import sys
print('Python: {}'.format(sys.version))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# seaborn
import seaborn
print('seaborn: {}'.format(seaborn.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import warnings
warnings.simplefilter(action='ignore')


# In[ ]:


sns.set(style='darkgrid')


# # 2. Load dataset

# In[ ]:


train_path = '../input/train.csv'
test_path = '../input/test.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)


# # 3. Summarize data
# 
# Let's take a quick look at the data we have by - 
# 1. Shape of dataframe
# 2. Info of dataset
# 3. Statistics of data
# 4. Peep at data itself
# 5. Missing data

# ## 3.1 Shape of dataset
# 
# The <b>shape</b> attribute of pandas dataframe allow us to quickly see, the <b>number of training examples</b> and the <b>number of features</b> available to us.

# In[ ]:


train_df.shape


# We can see that our training set has <b>891 training examples</b> and <b>11 features + 1 label column</b>.

# ## 3.2 Info of dataset
# 
# The <b>info</b> function of dataframe gives us <b>information of the full dataset column wise</b> - the data types and the missing values in each column etc.

# In[ ]:


train_df.info()


# We can see that we have total <b>11+1 data columns</b>.<br>An interesting thing that we can see here is, columns -<b> Age,Cabin and Embarked</b> - have missing data.<br>In our dataset, we have <b>2 features of float type, 5 features of int type and 5 features that are object (string) type.</b><br><br>
# Definition of each feature is as - 
# <table>
# <tr><th><b>Variable</b></th><th><b>Definition</b></th><th><b>Key</b></th></tr>
# <tr>
# <td>survival</td>
# <td>Survival</td>
# <td>0 = No, 1 = Yes</td>
# </tr>
# <tr>
# <td>pclass</td>
# <td>Ticket class</td>
# <td>1 = 1st, 2 = 2nd, 3 = 3rd</td>
# </tr>
# <tr>
# <td>sex</td>
# <td>Sex</td>
# <td></td>
# </tr>
# <tr>
# <td>Age</td>
# <td>Age in years</td>
# <td></td>
# </tr>
# <tr>
# <td>sibsp</td>
# <td># of siblings / spouses aboard the Titanic</td>
# <td></td>
# </tr>
# <tr>
# <td>parch</td>
# <td># of parents / children aboard the Titanic</td>
# <td></td>
# </tr>
# <tr>
# <td>ticket</td>
# <td>Ticket number</td>
# <td></td>
# </tr>
# <tr>
# <td>fare</td>
# <td>Passenger fare</td>
# <td></td>
# </tr>
# <tr>
# <td>cabin</td>
# <td>Cabin number</td>
# <td></td>
# </tr>
# <tr>
# <td>embarked</td>
# <td>Port of Embarkation</td>
# <td>C = Cherbourg, Q = Queenstown, S = Southampton</td>
# </tr>
# </table>
# 
# (Source: The HTML  for table has been copied from kaggle @ https://www.kaggle.com/c/titanic/data)

# ## 3.3 Statistics of dataset
# 
# The <b>describe</b> function of pandas gives us the <b>statistical description</b> of <b>each column present in our dataframe.</b> A thing to note is that the </b>columns containing only real data</b> are described using this function, <b>categorical features</b> are <b>not</b> present in the result.

# In[ ]:


train_df.describe()


# ## 3.4 Peep at data
# 
# The <b>head</b> function of pandas dataframe allows us to <b>view the dataframe</b>, 5 rows by default

# In[ ]:


train_df.head()


# From the above peep we can see that the Cabin column has some data - <b>NaN</b>, which stands for <b>Not a Number</b>, i.e, those data are missing. We'll fix this issue later.

# ## 3.5 Missing data
# 
# Lets take a brief look at the missing data.<br><br>What we will do is create a dataframe of missing data from our training dataset. This dataframe will give us the total and percent of missing data for each feature.

# In[ ]:


missing_data = train_df.isnull().sum()
percent_missing = round((missing_data / train_df.isnull().count())*100, 2)
missing_df = pd.concat(
    [missing_data.sort_values(ascending=False), percent_missing.sort_values(ascending=False)], 
    axis=1, keys=['Total', 'Percent']
)
missing_df.head(5)


# <b>Cabin</b> feature has <b>687</b> data missing out of total <b>891</b>, which makes about <b>77% of the data missing</b>, it seems like we might need to eliminate this column. But before doing so we'll make sure that we do not drop important data from our dataset. We don't have a lot of data missing from other features, so we'll impute those later.

# # 4. Data Visualization / EDA
# Now, lets visualise and analyse our data, using boxplots, pointplots, violinplots, barplots etc.<br>
# So lets see how our features affected survival in following sequence - 
# 1. Cabin
# 2. Age 
# 3. Sex
# 4. Embarked
# 5. Pclass
# 6. \# Relatives
# 7. Fare

# In[ ]:


train_df.columns


# ## 4.1 Cabin
# 
# Since, we have huge amount of data missing from cabin, it will be difficult to visualise and analyse how cabin effected survival. So, for the sake of simplicity lets see how the presence or absence of cabin data affected survial.

# In[ ]:


temp_df = train_df.copy()
temp_df['Cabin'] = temp_df['Cabin'].fillna('Unknown')

occ_cabins = temp_df['Cabin'].copy()
occ_cabins[occ_cabins != 'Unknown'] = 'Known'
temp_df['Cabin'] = occ_cabins

plt.figure(figsize=(5.5, 5))
sns.barplot(x='Cabin', y='Survived', data=temp_df)
sns.pointplot(x='Cabin', y='Survived', data=temp_df, color='k')


# We can see that the fraction of people who survived most of those peoples' cabin is known.<br>Well, this is not the best intuition but, for now I can think of this only and seeing at the above plot it seems usless to drop this column. If I come up with a better idea I'll update it, but for now lets move with it.

# ## 4.2 Sex
# Now, lets see how <b>gender</b> effected survival.

# In[ ]:


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.barplot(x='Sex', y='Survived', data=train_df)
plt.subplot(1, 2, 2)
sns.violinplot(x='Survived', y='Age', data=train_df, hue='Sex', split=True)


# Seeing the first graph we can infer that greater fraction of females survived the tragedy.

# ## 4.3 Age
# 
# Lets see how the <b>age</b> affected the survival in both male and female.

# In[ ]:


females = train_df[train_df['Sex'] == 'female']
males = train_df[train_df['Sex'] == 'male']

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
ax = sns.distplot(males[males['Survived'] == 1]['Age'].dropna(), bins=10, kde=False, label='survived')
ax = sns.distplot(males[males['Survived'] == 0]['Age'].dropna(), bins=10, kde=False, label='not survived')
ax.legend()
ax.set_title('Male')

plt.subplot(1, 2, 2)
ax = sns.distplot(females[females['Survived'] == 1]['Age'].dropna(), kde=False, label='survived')
ax = sns.distplot(females[females['Survived'] == 0]['Age'].dropna(), kde=False, label='not survived')
ax.legend()
ax.set_title('Female')

plt.show()


# We can see from the above barplots that most of the <b>males</b> that survives were around the age of <b>18-40</b>, whereas most of the <b>females</b> that survived were around the age of <b>15-38</b>

# In[ ]:


plt.figure(figsize=(5.5, 5))
sns.boxplot(x='Survived', y='Age', data=train_df)


# We can generalize the above and say that the <b>people who survived</b>, were around the age of <b>18-38</b>

# ## 4.4 Fare

# In[ ]:


plt.figure(figsize=(5.5, 5))
sns.boxplot(x='Survived', y='Fare', data=train_df)


# The <b>fare of the people who survived</b>, ranged from somewhere about </b>\$10k to \$80k</b> (for about most of them). The plot also indicates that fare payed by the surviving people took a minimum value of \$0k and a maximum value of \$500k.

# ## 4.5 PClass

# In[ ]:


plt.figure(figsize=(5.5, 5))
sns.pointplot(x='Pclass', y='Survived', data=train_df)


# The <b>higher the class</b> (by higher I mean the lower the value of Pclass), the <b>greater fraction</b> of people survived.<br><br>This seems a little obvious correlation though.

# ## 4.6 # Relatives

# In[ ]:


plt.figure(figsize=(5.5, 5))
sns.barplot(x=(train_df['SibSp'] + train_df['Parch']), y=train_df['Survived'])


# The <b>greater fraction of people survived</b> who had <b>1, 2, 4 and somewhat 6 relatives</b> onboard.

# ## 4.7  Embarked

# In[ ]:


plt.figure(figsize=(5.5, 5))
sns.pointplot(x='Embarked', y='Survived', data=train_df)


# <b>Greater fraction</b> of people who survived, <b>embarked at Cherbourg</b>.

# # 5. Data Preprocessing
# 
# Now, lets convert our raw data into some useful information, that can be pluged into out classifiers.<br>
# The steps we'll follow are - 
# 1. Train - Test split
# 2. Handle missing data.
# 3. Handle categorical data.

# ## 5.1 Train-Test split
# 
# Lets <b>split</b> our data into testing and validation set, with validation set containing <b>20% of the data</b>.

# In[ ]:


pass_id = test_df['PassengerId']

X = train_df.drop(['PassengerId'], axis=1)
X_test = test_df.drop(['PassengerId'], axis=1)

y = X['Survived']
X = X.drop('Survived', axis=1)

features_train, features_valid, labels_train, labels_valid = train_test_split(
    X, y, test_size=0.2, random_state=7
)

pre_features_train = features_train.copy()

cols = features_train.columns
cols


# ## 5.2 Missing data
# 
# Now lets impute our missing data, in the following sequence - 
# 1. Cabin
# 2. Age 
# 3. Embarked

# In[ ]:


missing_df


# ### 5.2.1 Cabin
# Well, to be honest, since 77% of data for this feature was missing hence I tried dropping it, but my model performed extremely bad. It turns out that this feature holds a significance in predicting survival.<br>
# So, I imputed the missing values with 'X01', and encoded it as below, just using the deck code, and not the room number (might be!).

# In[ ]:


def missing_cabin(data):
    data['Cabin'].fillna('X01', inplace=True)

    
for data in [features_train, features_valid]:
    missing_cabin(data)


# ### 5.2.2 Age

# In[ ]:


def missing_age(data):
    nan_ages = []
    
    mu = pre_features_train['Age'].mean()
    median = pre_features_train['Age'].median()
    sigma = pre_features_train['Age'].std()

    random_ages = np.random.randint(
            median-sigma, 
            median+sigma,
            data['Age'].isnull().sum()
        )

    nan_ages = data['Age'].copy() 
    nan_ages[nan_ages.isnull()] = random_ages
    data.loc[:, 'Age'] = nan_ages
    

for data in [features_train, features_valid]:
    missing_age(data)


# ### 5.2.3 Embarked

# In[ ]:


impute_embark = SimpleImputer(strategy='most_frequent')

features_train['Embarked'] = impute_embark.fit_transform(features_train['Embarked'].values.reshape(-1, 1))
features_valid['Embarked'] = impute_embark.transform(features_valid['Embarked'].values.reshape(-1, 1))


# In[ ]:


features_train.info()


# In[ ]:


features_train.head()


# ## 5.3 Categorical data
# After imputing missing values, its time to <b>encode categorical data</b> to natural numbers using labelencoder, ordinalencoder and onehot encoder.<br>
# The sequence in which we'll encode our features are as - 
# 1. Cabin
# 2. Gender
# 3. Pclass
# 4. Embarked

# ### 5.3.2 Cabin

# In[ ]:


cabin_data = np.array(features_train['Cabin'])
cabin_data_valid = np.array(features_valid['Cabin'])

cabin_data = pd.DataFrame([x[0] for x in cabin_data], index=features_train.index, columns=['Cabin'])
cabin_data_valid = pd.DataFrame([x[0] for x in cabin_data_valid], index=features_valid.index, columns=['Cabin'])

features_train.drop(['Cabin'], axis=1, inplace=True)
features_valid.drop(['Cabin'], axis=1, inplace=True)

features_train = pd.concat([features_train, cabin_data], axis=1)
features_valid = pd.concat([features_valid, cabin_data_valid], axis=1)

le_cabin = LabelEncoder()
features_train['Cabin'] = le_cabin.fit_transform(features_train['Cabin'])
features_valid['Cabin'] = le_cabin.transform(features_valid['Cabin'])


# ### 5.3.1 Gender

# In[ ]:


le_gender = LabelEncoder()
features_train['Sex'] = le_gender.fit_transform(features_train['Sex'])
features_valid['Sex'] = le_gender.transform(features_valid['Sex'])


# ### 5.3.3 Pclass

# In[ ]:


oe_pclass = OrdinalEncoder(dtype='int64')
features_train['Pclass'] = oe_pclass.fit_transform(features_train['Pclass'].values.reshape(-1, 1))
features_valid['Pclass'] = oe_pclass.transform(features_valid['Pclass'].values.reshape(-1, 1))


# ### 5.3.4 Embarked

# In[ ]:


le_embark = LabelEncoder()
integer_encoded = le_embark.fit_transform(features_train['Embarked'])
integer_encoded_valid = le_embark.transform(features_valid['Embarked'])

oh_embark = OneHotEncoder(handle_unknown='ignore', sparse='False', dtype='int64')
onehot_encoded = pd.DataFrame(
    oh_embark.fit_transform(integer_encoded.reshape(-1, 1)).toarray(), 
    columns=['Embarked_C', 'Embarked_Q', 'Embarked_S'], 
    index=features_train.index
)
onehot_encoded_valid = pd.DataFrame(
    oh_embark.transform(integer_encoded_valid.reshape(-1, 1)).toarray(), 
    columns=['Embarked_C', 'Embarked_Q', 'Embarked_S'], 
    index=features_valid.index
)

features_train = features_train.drop(['Embarked'], axis=1)
features_valid = features_valid.drop(['Embarked'], axis=1)

features_train = pd.concat([features_train, onehot_encoded], axis=1)
features_valid = pd.concat([features_valid, onehot_encoded_valid], axis=1)


# In[ ]:


features_train.head()


# # 6. Feature Engineering
# 
# After we are done with processing our data, now lets work on our features to improve the quality of our model.
# <br>Feature Engineering helps <b>bulid a simpler model and pervents overfitting</b> as well.
# <br>We'll feature engineer in following steps - 
# 1. Feature Scaling
# 2. Feature Addition
# 3. Feature Selection
# <br>Since, we don't have many features we'll skip PCA for this dataset.

# ## 6.1 Feature Scaling
# 
# We will standard scale our data, i.e using our old simple scaling technique - 
# 
# z = (x - $\bar{x}$) / $\sigma$
# 
# where,<br>
# x = feature value<br>
# $\bar{x}$ = mean of features<br>
# $\sigma$ = standard deviation of feature
# 
# <br>We'll scale our features in following sequence - 
# 1. Age
# 2. Fare
# 
# For ease lets define a function that does scaling for us.

# In[ ]:


def scale_feature(data, feature):
    mu = pre_features_train[feature].mean()
    sigma = pre_features_train[feature].std()
    
    data.loc[:, feature] = round((data[feature] - mu) / sigma, 3)


# ### 6.1.1 Age

# In[ ]:


for data in [features_train, features_valid]:
    scale_feature(data, 'Age')


# ### 6.1.2 Fare

# In[ ]:


for data in [features_train, features_valid]:
    scale_feature(data, 'Fare')


# ## 6.2 Feature Addition
# 
# Till now we have been working with the features that we have, but now lets create some new features which might help out classifier give better result.<br>
# 
# We specifically will add - \# Relatives

# ### 6.2.1 # relatives
# 
# We will add a new feaure 'relatives which will gives us the total number of relatives for each person' data.<br><br>
# We'll define it as - <br>
# relatives = Sibsp + Parch

# In[ ]:


def add_rel(data):
    data.loc[:, 'relatives'] = data['SibSp'] + data['Parch']

for data in [features_train, features_valid]:
    add_rel(data)


# ## 6.3 Feature Selection
# 
# Before feature selection, lets drop certain features - name and ticket - which give us relatively less information about survival of that particular persons.<br>
# Though one can argue that a VIP had a more chance for survival, but for now lets drop both the columns.

# In[ ]:


def drop_features(data):
    drop_cols = ['Name', 'Ticket']
    data.drop(drop_cols, axis=1, inplace=True)
    
for data in [features_train, features_valid]:
    drop_features(data)


# Now we have the features that may be of low to high importance to our model.<br><br>Lets find out the <b>importances of the remaining fetures</b> and <b>eliminate</b> the features that are of very low to no importance to us to prevent overfitting of our model.

# In[ ]:


best_features = SelectKBest(k='all')
fit = best_features.fit(features_train, labels_train)

scores_df = pd.DataFrame(data=fit.scores_)
columns_df = pd.DataFrame(data=features_train.columns.values)

feature_scores_df = pd.concat([columns_df,scores_df],axis=1)
feature_scores_df.columns = ['Features','Score']

plt.figure(figsize=(5.5, 5))
sns.barplot(
    x='Score', y='Features', 
    order=feature_scores_df.nlargest(11,'Score')['Features'], 
    data=feature_scores_df, palette=sns.cubehelix_palette(n_colors=4, reverse=True)
)


# Seeing the barplot we can clearly infer that features - relatives, SibSp and Embarked_Q, have very small importance. So, it would be better to drop them, to prevent overfitting our model.

# In[ ]:


def keep_best_features(data):
    drop_cols = ['relatives', 'SibSp', 'Embarked_Q']
    data.drop(drop_cols, axis=1, inplace=True)
    
for data in [features_train, features_valid]:
    keep_best_features(data)


# In[ ]:


features_train.head()


# # 7. Evaluate some algorithms
# 
# After such long cleanign and processing of our data lets build some classifiers, namely - 
# 1. Naive Bayes
# 2. SVC
# 3. Logistic Regression
# 4. KNN
# 5. Decision Tree
# 6. Bagging
# 7. Random Forest
# 8. Adaboost
# 9. XGboost

# In[ ]:


clf = [
    ('GNB', GaussianNB()),
    ('SVC', SVC(C=1000, kernel='rbf', gamma=0.3)),
    ('LReg', LogisticRegression(C=0.5, solver='lbfgs')),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier(min_samples_split=100)),
    ('BAG', BaggingClassifier()),
    ('RF', RandomForestClassifier(n_estimators=100)),
    ('AB', AdaBoostClassifier(n_estimators=100)),
    ('XGB', XGBClassifier(max_depth=10, learning_rate=0.03, n_estimators=100))
]

clf_names = [
    'GaussianNB', 'SVC', 'Logistic Reg', 'KNN', 'Decision Tree', 'Bagging', 'Random Forest', 'AdaBoost', 'XGboost'
]


# # 8. Validation
# 
# After building our classifiers lets use <b>10 fold cross validation</b> to see how well each of them perform in our training data, and select the best out of them.<br><br>
# What 10-fold cross validator does is divide our dataset into 10 folds/groups, keeps one fold to test and trains each classifier on the remainig data. It then selects the other fold for testing and trains on the remaining and goes on until it has used up all folds for testing and the remaining for training.<br>In this way we <b>train as well as test</b> our classifiers on <b>all of the training data</b>.

# In[ ]:


result = []
for model in clf:
    score = cross_val_score(model[1], features_train, labels_train, cv=10)
    result.append(score)
    
result_df = pd.DataFrame({
    'Score' : [round(x.mean(), 3) for x in result],
    'Model' : clf_names
})

result_df = result_df.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')

result_df


# It seems like <b>XGboost</b> does a good job classifing our data, but before selecting a classifier lets see the accuracy distribution for each classifier

# # 9. Select best model

# In[ ]:


plt.figure(figsize = (7, 4))
ax = sns.boxplot(data=pd.DataFrame(np.array(result).transpose(), columns=clf_names))
ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
plt.show()


# It seems like XGboost has a good accuracy range (If you see something different, welcome to the randomness of machine learning, each time we compile we'll get different result, but most of the time XGboost had better results).<br>
# Lets move on using <b>XGBoost Classifier</b>.

# # 10. Predict
# If you remember, we did seperate a <b>validation set</b>. Now, its time to check the <b>accuracy of our selected model on the validation set</b>.

# In[ ]:


clf = XGBClassifier(
    max_depth=5, learning_rate=0.2,
    verbosity=1, silent=None, n_estimators=100,  
    objective='binary:logistic',booster='gbtree'
)

clf.fit(features_train, labels_train)
predict = clf.predict(features_valid)

print(round((accuracy_score(labels_valid, predict)*100), 2))


# <b>81.56% accuracy</b> (you might see some other value, again machine learning, randomness) doesn't seem best, but its kind of good for our model.

# # 11. Performance of model

# In[ ]:


print(accuracy_score(labels_valid, predict))
print(confusion_matrix(labels_valid, predict))
print(classification_report(labels_valid, predict))


# In[ ]:


plt.figure(figsize=(4, 3))
ax = sns.heatmap(confusion_matrix(labels_valid, predict), annot=True)
ax.set_xlabel('Predicted Values')
ax.set_ylabel('True Values')

plt.show()


# # 12. Predict from test data
# 
# Its time, lets process our <b>test dataset and predict using our selected, trained classifier</b>.

# In[ ]:


missing_cabin(X_test)
missing_age(X_test)
X_test['Embarked'] = impute_embark.transform(X_test['Embarked'].values.reshape(-1, 1))

X_test['Sex'] = le_gender.transform(X_test['Sex'])
X_test['Pclass'] = oe_pclass.transform(X_test['Pclass'].values.reshape(-1, 1))

cabin_data = np.array(X_test['Cabin'])
cabin_data = pd.DataFrame([x[0] for x in cabin_data], index=X_test.index, columns=['Cabin'])
X_test.drop(['Cabin'], axis=1, inplace=True)
X_test = pd.concat([X_test, cabin_data], axis=1)
X_test['Cabin'] = le_cabin.transform(X_test['Cabin'])

integer_encoded_test = le_embark.transform(X_test['Embarked'])
onehot_encoded_test = pd.DataFrame(
    oh_embark.transform(integer_encoded_test.reshape(-1, 1)).toarray(), 
    columns=['Embarked_C', 'Embarked_Q', 'Embarked_S'], 
    index=X_test.index
)

X_test.drop(['Embarked'], axis=1, inplace=True)
X_test = pd.concat([X_test, onehot_encoded_test], axis=1)
    
scale_feature(X_test, 'Age')
scale_feature(X_test, 'Fare')
add_rel(X_test)

drop_features(X_test)
keep_best_features(X_test)

predict = clf.predict(X_test)


# In[ ]:


output = pd.DataFrame(
    {
        'PassengerId': pass_id,
        'Survived': predict
    }
)

output.to_csv('gender_submission.csv', index=False)

