#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy.stats as stats

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# <img src="https://unasalahat.files.wordpress.com/2013/06/m2a1_2.jpg">
# 
# Cheers! We will be exploring the red wine quality dataset. Every one loves a glass of wine once in a while, but what goes behind the scenes? We are provided with physiochemical features(the input) and sensory(the output) variables. Let's find what physiochemical features make the wine best! Wine quality ofcourse depends on other factors such as grape types etc. but due to the limitation of the dataset we will just focus on the physiochemical features. So let's begin!

# ## Get the data

# In[ ]:


dataset = pd.read_csv('../input/winequality-red.csv')
dataset.info()


# In[ ]:


dataset.head()


# So we have 11 features and 1 target variable. The dataset is small with just 1599 instances and we do not have any null values (Yay!). Also what I notice is all the features are numerical and they have different scales.
# 
# Now as we know that our target variable **quality** is categorical, let's see how many types of categories we have.

# In[ ]:


dataset.quality.value_counts()


# In[ ]:


plt.figure(figsize=(10, 5))
sns.countplot(x='quality', data=dataset)
plt.xlabel('Quality', fontsize=14)
plt.ylabel('Count', fontsize=14)


# Okay so we can see that the data is unbalanced with wine quality of 5 or 6 the most frequent in our dataset. I want this to be a binary classification project so I'll just set an arbitray cut off later.
# 
# Let's check the distribution first!

# In[ ]:


dataset.describe()


# In[ ]:


dataset.hist(bins=50, figsize=(15, 15))


# Most of the distributions are skewed to the right. Also density and pH show a normal distribution.
# 
# Also I would like to set an arbitrary cutoff for our target variable. As I want this to be a binary classification project, I will classify any wine with a quality 7 or greater as good(1) and the remainder as not good(0).

# In[ ]:


dataset.quality[dataset.quality < 7] = 0
dataset.quality[dataset.quality >= 7] = 1


# In[ ]:


dataset.quality.value_counts()


# In[ ]:


plt.figure(figsize=(10, 5))
sns.countplot(x='quality', data=dataset)
plt.xlabel('Quality', fontsize=14)
plt.ylabel('Count', fontsize=14)


# The data is still unbalanced so when we split into training and test set we must ensure that we maintain this proportion.

# ## Exploratory Data Analysis

# Now let's visualize the data! I think alcohol content would play a big role in the quality of good and not good wines.

# In[ ]:


redwine = dataset.copy()

plt.figure(figsize=(8, 5))
sns.boxplot(x='quality', y='alcohol', data=redwine)
plt.title('Boxplot for alcohol', fontsize=16)
plt.xlabel('Quality', fontsize=14)
plt.ylabel('Alcohol', fontsize=14)


# - We can see that the median of alchol content for 'not good' and good wines are quite well separated. Maybe it would be a good feature for our classification. Let's check the distribution.

# In[ ]:


plt.figure(figsize=(16, 5))

plt.subplot(121)
sns.distplot(redwine[redwine.quality==0]['alcohol'], color='r')
plt.title('Distribution of alcohol for "not good" wine', fontsize=16)
plt.xlabel('Alcohol', fontsize=14)
plt.subplot(122)
sns.distplot(redwine[redwine.quality==1]['alcohol'], color='c')
plt.title('Distribution of alcohol for good wine', fontsize=16)
plt.xlabel('Alcohol', fontsize=14)


# Non-good wines do have few outliers with high alcohol content. But in general we can see that good wines have higher alcohol content.

# I am no expert in wines but after reading the description of **volatile acidity**(the amount of acetic acid in wine, which at too high of levels can lead to an unpleasant, vinegar taste) I am guessing it would have a negative relationship with quality? Let's check it out!

# In[ ]:


plt.figure(figsize=(8, 5))
sns.boxplot(x='quality', y='volatile acidity', data=redwine)
plt.title('Boxplot for volatile acidity', fontsize=16)
plt.xlabel('Quality', fontsize=14)
plt.ylabel('Volatile Acidity', fontsize=14)


# In[ ]:


plt.figure(figsize=(16, 5))

plt.subplot(121)
sns.distplot(redwine[redwine.quality==0]['volatile acidity'], color='r')
plt.title('Distribution of volatile acidity for "not good" wine', fontsize=16)
plt.xlabel('Volatile Acidity', fontsize=14)
plt.subplot(122)
sns.distplot(redwine[redwine.quality==1]['volatile acidity'], color='c')
plt.title('Distribution of volatile acidity for good wine', fontsize=16)
plt.xlabel('Volatile Acidity', fontsize=14)


# YES! we can clearly see that non-good wines have higher volatile acidity!
# 
# Now the term "acid" appears in three of our variables - **fixed acidity**, **volatile acidity**, **citric acid**. Are they correlated?

# In[ ]:


data = redwine.loc[:, ['fixed acidity', 'volatile acidity', 'citric acid']]
ax = sns.PairGrid(data)
ax.map_lower(sns.kdeplot)
ax.map_upper(sns.scatterplot)
ax.map_diag(sns.kdeplot)


# Yup! our guess was right. We can see that **fixed acidity** and **citric acid** have a fairly positive correlation, while **volatile acidity** and **citric acid** have a negative correlation. So during feature selection we might be able to eliminate one of the features(or might not!)

# Now let's do the same for **free sulfur dioxide** and **total sulfur dioxide**.

# In[ ]:


ax = sns.jointplot(x=redwine.loc[:, 'free sulfur dioxide'], y=redwine.loc[:, 'total sulfur dioxide'], kind='reg')
ax.annotate(stats.pearsonr)


# So yes they have a good correlation of 0.67. How are they related to wine quality tho?

# In[ ]:


plt.figure(figsize=(12,5))

plt.subplot(121)
sns.boxplot(x='quality', y='free sulfur dioxide', data=redwine)
plt.title('Boxplot for free sulphur dioxide', fontsize=16)
plt.xlabel('Quality', fontsize=14)
plt.ylabel('Free sulfur Dioxide', fontsize=14)
plt.subplot(122)
sns.boxplot(x='quality', y='total sulfur dioxide', data=redwine)
plt.title('Boxplot for total sulphur dioxide', fontsize=16)
plt.xlabel('Quality', fontsize=14)
plt.ylabel('Total sulfur Dioxide', fontsize=14)


# We can see that non-good wines have a slightly higher content of **free sulfur dioxide** and **total sulfur dioxide** in comparison with good wines. However, good wines do have few outliers.

# Now what about **residual sugar**?(the amount of sugar remaining after fermentation stops). Does the remaining sugar have any impact on the quality of wines?

# In[ ]:


plt.figure(figsize=(8,5))
sns.boxplot(x='quality', y='residual sugar', data=redwine)
plt.title('Boxplot for Residual sugar', fontsize=16)
plt.xlabel('Quality', fontsize=14)
plt.ylabel('Residual Sugar', fontsize=14)


# Uhhhhh no! It doesn't seem to play an important role with the quality of wine. Let's check the same for **chlorides**.

# In[ ]:


plt.figure(figsize=(8,5))
sns.boxplot(x='quality', y='chlorides', data=redwine)
plt.title('Boxplot for Chlorides', fontsize=16)
plt.xlabel('Quality', fontsize=14)
plt.ylabel('Chlorides', fontsize=14)


# Nope! Just like **Residual Sugar** we cannot observe any relation with quality.
# 
# **pH** : describes how acidic or basic a wine is on a scale from 0 (very acidic) to 14 (very basic); most wines are between 3-4 on the pH scale. By checking the histogram of pH we can observe our wines are between 3-4 on the pH scale. But how it is ditributed between good and non-good wines?

# In[ ]:


plt.figure(figsize=(16, 5))

plt.subplot(121)
sns.distplot(redwine[redwine.quality==0]['pH'], color='r')
plt.title('Distribution of pH for "not good" wine', fontsize=16)
plt.xlabel('pH', fontsize=14)
plt.subplot(122)
sns.distplot(redwine[redwine.quality==1]['pH'], color='c')
plt.title('Distribution of pH for good wine', fontsize=16)
plt.xlabel('pH', fontsize=14)


# Hmmmmm! Seems like pH does not have to do anything with the quality of our wines. Also I am thinking whether it does have any relation with our *acid* features.

# In[ ]:


for col, x in zip(('k','c','g'),('fixed acidity', 'volatile acidity', 'citric acid')):
    ax = sns.jointplot(x=x, y='pH', data=redwine, kind='reg', color=col)
    ax.annotate(stats.pearsonr)
    plt.xlabel(x, fontsize=14)
    plt.ylabel('pH', fontsize=14)


# pH does have a good negative correlation with **citric acid** and **fixed acidity**. Let's do the same for **density**(the density of water is close to that of water depending on the percent alcohol and sugar content). 
# 
# The histogram of **density** plotted above shows a normal distribution with a density ranging from 0.99 to 1.0025 which is very well in the range of water. Let's check it out if whether a wine with high/low density has an impact on its quality.

# In[ ]:


plt.figure(figsize=(16, 5))

plt.subplot(121)
sns.distplot(redwine[redwine.quality==0]['density'], color='r')
plt.title('Distribution of density for "not good" wine', fontsize=16)
plt.xlabel('Density', fontsize=14)
plt.subplot(122)
sns.distplot(redwine[redwine.quality==1]['density'], color='c')
plt.title('Distribution of density for good wine', fontsize=16)
plt.xlabel('Density', fontsize=14)


# I don't think so. Also, from the description it says that it is dependent on the percentage of alcohol and sugar content. So from our features I am going to check the relation of **density** with **alcohol**(duh) and **residual sugar**.

# In[ ]:


for col, x in zip(('k','c'),('alcohol', 'residual sugar')):
    ax = sns.jointplot(x=x, y='density', data=redwine, kind='reg', color=col)
    ax.annotate(stats.pearsonr)
    plt.xlabel(x, fontsize=14)
    plt.ylabel('density', fontsize=14)


# **density** does have a slightly good negative correlation with alcohol. Increase in the alcohol content decreases the density of water! Also as this is a binary classification problem we can create swarmplots for our different features and observe if we can see a clear distinction between good and non-good wines. As the features are on different scales I'll transform them to the same scales using standardization and then create a swarmplot

# In[ ]:


# Separating independent and dependent variable
redwine_X = redwine.drop('quality', axis=1)
redwine_y = redwine['quality']
col = redwine_X.columns

# Standardization
sd = StandardScaler()
redwine_X_scaled = sd.fit_transform(redwine_X)
redwine_X_scaled = pd.DataFrame(redwine_X_scaled, columns=col)

# Swarmplot for first 5 features
data = pd.concat([redwine_y, redwine_X_scaled.iloc[:, :5]], axis=1)
data = pd.melt(data, id_vars='quality', var_name='feature', value_name='value')

plt.figure(figsize=(12,6))
sns.swarmplot(x='feature', y='value', hue='quality', data=data)
plt.title('Swarmplot for first 5 features', fontsize=16)
plt.xlabel('Feature', fontsize=14)
plt.ylabel('Value', fontsize=14)


# Although we cannot see a clear distinction, we can still say that **volatile acidity**, **citric acid** would be good for classification. While **fixed acidity**, **residual sugar** and **chlorides** are too mixed up to classify between good and non-good wines. Now let's do the same for last 6 features.

# In[ ]:


# Swarmplot for last 6 features
data = pd.concat([redwine_y, redwine_X_scaled.iloc[:, 5:]], axis=1)
data = pd.melt(data, id_vars='quality', var_name='feature', value_name='value')

plt.figure(figsize=(12,6))
sns.swarmplot(x='feature', y='value', hue='quality', data=data)
plt.title('Swarmplot for last 6 features', fontsize=16)
plt.xlabel('Feature', fontsize=14)
plt.ylabel('Value', fontsize=14)


# Similary, as you may have guessed **sulphates** and **alcohol** are well separated making it easy for classification.
# 
# Now let's look for correlations!

# In[ ]:


plt.figure(figsize=(10, 5))
sns.heatmap(redwine.corr(), annot=True, fmt='.2f')


# Hmmm, there is a fairly good positive correlation between **alcohol** and **quality**. **sulphates** and **citric acid** have a weak positive correlation with **quality**. While **volatile acidity** shows a negative correlation with **quality**.
# 
# What more important to see is whether our features are correlated with each other? For example, **fixed acidity** has a good positive correlation with **density** and negative correlation with **pH**. But it isn't that fairly high. Similarly, we can observe other good correlations between features but none of them show a fairly high correlation, so we decide to keep all of the features for now to predict the quality of wine. 
# 
# ## Creating new features
# 
# Creating new features from existing features is also an important step to check if it improves the accuracy or even correlation with our dependent variable. After little research, I came up with the following new features:
# 
# - **total_acidity** - Sum of fixed and volatile acidity. [**fixed acidity + volatile acidity**]
# - **total_acidity_citric** - I learnt that **citric acid** is a type of titratable or **total acid**, so i thought maybe i should add that too in **total acidity**?. Or else let's just create a new feature for that! [**fixed acidity + volatile acidity + citric acid**]
# - **bound_sulphur_dioxide** - **total sulphur dioxide** is actually a sum of bound(fixed) SO2 and **free sulphur dioxide**. [**total sulphur dioxide - free sulphur dioxide**]
# - **org_minus_final_SG** - $$ \% alcohol(by volume) = \frac{Original Specific Gravity - Final Specific Gravity}{7.36} \times 1000 $$
# 
# *Fun fact: Coke has about the same level of sugar, at 108 g/L, as some of the sweetest dessert wines!*

# In[ ]:


dataset_additional = dataset.copy()

dataset_additional['total_acidity'] = dataset_additional['fixed acidity'] + dataset_additional['volatile acidity']
dataset_additional['total_acidity_citric'] = dataset_additional['total_acidity'] + dataset_additional['citric acid']
dataset_additional['bound_sulphur_dioxide'] = dataset_additional['total sulfur dioxide'] - dataset_additional['free sulfur dioxide']
dataset_additional['org_minus_fail_SG'] = (dataset_additional['alcohol'] * 7.36)/1000


# Now that the additional features are created, lets check the heatmap again.

# In[ ]:


plt.figure(figsize=(12, 6))
sns.heatmap(dataset_additional.corr(), annot=True, fmt='.2f')


# This additional features doesn't seem to correlate well with our dependent variable **quality**, however, something cool to observe is how high **total_acidity** and **total_acidity_citric** is positively correlated with **fixed acidity**. It's good because too much **volatile acidity** can lead to off flavors and aromas. **org_minus_fail_SG** will obviously have same correlations as that of **alcohol**. Because all the four additional features are highly correlated with other features, we will not use them in our classification model.

# ## Prepare the data

# Now lets start our classification modeling process. First with the help of stratified shuffle split lets split into train and test sets to ensure our balance of good and non-good wines is maintained as our data is highly unbalanced.

# In[ ]:


X = dataset.drop('quality', axis=1)
y = dataset['quality']

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(X, y):
    strat_train_set = dataset.loc[train_index]
    strat_test_set = dataset.loc[test_index]


# In[ ]:


strat_train_set['quality'].value_counts()/len(strat_train_set), strat_test_set['quality'].value_counts()/len(strat_test_set)


# In[ ]:


# Check it with the actual dataset proportions
dataset['quality'].value_counts()/len(dataset)


# In[ ]:


X_train = strat_train_set.drop('quality', axis=1)
y_train = strat_train_set['quality']
X_test = strat_test_set.drop('quality', axis=1)
y_test = strat_test_set['quality']


# Now let's *standardize* our dataset as all features are on different scales.

# In[ ]:


sd = StandardScaler()

X_train_scaled = sd.fit_transform(X_train)


# ## Training model using Cross-Validation

# Lets start training our models. We will try several algorithms and see which model gives us the highest accuracy. We will evaluate using cross-validation. The following code performs *K-fold Cross-Validation* where k=10. So basically training set will be splitted into 10 subsets, then the model will train on 9 of the subsets and evaluate on the other subset. It repeats this process 10 times.

# In[ ]:


log_reg = LogisticRegression(random_state=42)
svm_clf = SVC(random_state=42)
tree_clf = DecisionTreeClassifier(random_state=42)
forest_clf = RandomForestClassifier(random_state=42)

X_test_scaled = sd.transform(X_test)

for clf in (log_reg, svm_clf, tree_clf, forest_clf):
    predicted = cross_val_predict(clf, X_train_scaled, y_train, cv=10)
    print(clf.__class__.__name__, ': ', accuracy_score(y_train, predicted))


# So the winner is RandomForest. It gave us an accuracy score of 89.91%. We still have to test it on our test set. 
# 
# ## Fine-Tuning the model
# 
# But first let's fine-tune it a little using *GridSearchCV*! Here I am evaluating 3 x 4 = 12 combinations of *n_estimators* and *max_features* in the first dictionary. Similary, I am trying in the other dictionary but with *bootstrap* set to False.(pasting)

# In[ ]:


param_grid = [
    {'n_estimators': [10, 25, 50, 100, 150], 'max_features': [2,3,6,9]},
    {'bootstrap': [False], 'n_estimators': [10, 25, 50, 100, 150], 'max_features': [2,3,6,9]}
]

forest_clf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(forest_clf, param_grid=param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)


# In[ ]:


grid_search.best_params_


# So it turns out the best parameters for our random forest classifier is to have *max_features* set to 3 and *n_estimators* set to 50. We can also look at the feature importances to see which feature plays an important role in classifying the quality of wine. 

# In[ ]:


attributes = X.columns
sorted(zip(grid_search.best_estimator_.feature_importances_, attributes), reverse=True)


# ## Feature Selection

# As we can see not all features are playing an important role in classification. Therefore we need to identify how many and which features to select. Here we will be using the recursive feature elimination along with cross-validation to see the accuracy as well.

# In[ ]:


# Feature Selection
forest_clf_2 = RandomForestClassifier()
rfecv = RFECV(forest_clf_2, step=1, cv=5, scoring='accuracy')
rfecv.fit(X_train_scaled, y_train)


# In[ ]:


print('Optimal no. of features: ', rfecv.n_features_)
print('Best Features: ', attributes[rfecv.support_])


# [](http://)![](http://)So it turns out 9 features play an important role in classification.

# In[ ]:


# No. of features vs CV scores
plt.xlabel('number of features', fontsize=14)
plt.ylabel('CV Score', fontsize=14)
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)


# 1. * We can observe the peak at 9. Therefore we choose these 9 features and finally test it on the test set.

# ## Evaluate on the Test set

# In[ ]:


attributes_selected = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'total sulfur dioxide', 'density', 'sulphates', 'alcohol']

X_train_selected = strat_train_set[attributes_selected]
X_test_selected = strat_test_set[attributes_selected]

X_train_sel_scaled = sd.fit_transform(X_train_selected)
X_test_sel_scaled = sd.transform(X_test_selected)

grid_search.best_estimator_.fit(X_train_sel_scaled, y_train)


# In[ ]:


y_pred = grid_search.best_estimator_.predict(X_test_sel_scaled)
accuracy_score(y_test, y_pred)


# In[ ]:


confusion_matrix(y_test, y_pred)


# AND WE GOT AN ACCURACY OF 94.06%. Not that of a high accuracy but pretty satisfying! Also looking at the confusion matrix we can see that out of 320 instances in our test set, we made 301 correct predictions and 19 wrong predictions.
# 
# Thank you for looking this kernel. This is my very first kernel and i am still learning. Let me know your thoughts, be open to find any mistakes, or ways/alternatives I could have done better. And lastly, if you do like it then don't forget to have a glass of wine! Cheers!!
