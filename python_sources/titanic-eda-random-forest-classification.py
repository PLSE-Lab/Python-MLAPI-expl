#!/usr/bin/env python
# coding: utf-8

# Hey, I'm glad you're reading this!
# 
# This is my kernel that I wanted to share with you, because it's my first notebook here on Kaggle with few interesting observations and steps. I hope it may help someone when struggling with this competition. I would also appreciate any feedback, maybe if you find some mistakes in my understanding or approach, it will help me improve my score as well!
# 
# I managed to score at most $79.43\%$ with this kernel. It isn't the best score (at the moment it's like top $20\%$ in the rolling leaderboard), but I also don't think it's complicated. Therefore if you are starting with Kaggle competitions or machine learning/data science at all - I think it's a good kernel to start. I tried to explain every single step I make here. Please let me know if something is unclear or I made a mistake (I'm learning as well!).

# ### 1. Read input data
# First things first, we need our data from Kaggle's directories. As you may read in the competition description, the dataset is split into two CSV files: train and test. We will read both and save them to separate DataFrames using Pandas library. DataFrames are arguably the most comfortable data format to work with when preprocessing data we have.

# In[ ]:


import numpy as np
import pandas as pd

df_train = pd.read_csv('/kaggle/input/titanic/train.csv')
df_test  = pd.read_csv('/kaggle/input/titanic/test.csv')

df_train.head()


# ### 2. Count how many values are missing in each column
# We cannot do further processing if there are missing values in any of columns. Depending on a percentage of missing features, we may either fill those cells with appriopriate values or just remove whole column if too much data is missing. In the table below, we display total number of missing features in the dataset (both training and testing part).

# In[ ]:


df_missing = pd.DataFrame(df_train.isna().sum() + df_test.isna().sum(), columns=['Missing'])
df_missing = df_missing.drop('Survived')
df_missing = df_missing.sort_values(by='Missing', ascending=False)
df_missing = df_missing[df_missing.Missing > 0]
df_missing


# ### 3. Drop Cabin column which has definitely too many missing values
# There are 1014 Cabin values missing in the whole Titanic dataset (see the table above). It means it would be difficult to fill missing rows because there is not much data avaiable in our data frames. Imagine a data frame with 1000 rows and filling 997 missing values with median of only 3 values. These values won't make much sense. On the other hand - filling 3 missing values based on information from existing 997 rows? It might work!
# 
# So, we get rid of the Cabin feature as this seems useless for us.

# In[ ]:


df_train, df_test = [x.drop('Cabin', axis=1) for x in [df_train, df_test]]


# ### 4. Visualize the correlation between each column and passengers survival
# For every available column, let's see what is the mean survival rate based on possible column values.
# 
# Here you can see for example that:
# * Usually woman have higher chances to survive than men,
# * Passengers from higher `Pclass` (lower numbers) are more likely to survive than passenger from lower classes,
# * Embarkation port and Parch/SibSp columns might also  be correlated with passengers' survival (we'll describe them soon).
# 
# You can also see that `Fare` feature seems to be [skewed](https://en.wikipedia.org/wiki/Skewness) which is a nice thing to notice.

# In[ ]:


import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 7))

df_train[['Age', 'Survived']].groupby('Age', as_index=True).mean().plot(style='.', ax=axes[0,0])
df_train[['Fare', 'Survived']].groupby('Fare', as_index=True).mean().plot(style='.', ax=axes[0,1])
df_train[['Embarked', 'Survived']].groupby('Embarked', as_index=True).mean().plot(kind='bar', ax=axes[0,2])
df_train[['Parch', 'Survived']].groupby('Parch', as_index=True).mean().plot(kind='bar', ax=axes[0,3])
df_train[['Pclass', 'Survived']].groupby('Pclass', as_index=True).mean().plot(kind='bar', ax=axes[1,0])
df_train[['Sex', 'Survived']].groupby('Sex', as_index=True).mean().plot(kind='bar', ax=axes[1,1])
df_train[['SibSp', 'Survived']].groupby('SibSp', as_index=True).mean().plot(kind='bar', ax=axes[1,2])
# df_train[['Ticket', 'Survived']].groupby('Ticket', as_index=True).mean().plot(kind='bar', ax=axes[1,3])


# ### 5. Check feature correlation
# Another nice way to visualize correlation between features is to use a [HeatMap](https://seaborn.pydata.org/generated/seaborn.heatmap.html) e.g. from seaborn library. In this heatmap values closer to zero mean that there is not much correlation between two columns, higher positive or negative values mean that there is positive/negative correlation.
# 
# Observations:
# * Significant correlation between `Fare` and survival (0.26),
# * Strong positive correlation between `SibSp` and `Parch` columns (0.41).

# In[ ]:


import seaborn as sns

plt.figure(figsize=(8,7))
sns.heatmap(df_train.corr(), annot=True, cmap=plt.cm.Reds)
plt.show()


# ### 6.1. Age
# 
# Now when we already have a brief overview of all features we can start focusing on them one by one. We will start from Age of passengers and we will divide it into four different bins using Pandas [cut](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.cut.html) function. Boundaries between consecutive bins have been chosen experimentally. You can see (if you look closer) that Children and Elder people are a bit more likely to survive than Teens and Adults.
# 
# We will use those bins (categorical variable) in our DataFrame. However, because Random Forest classifier that we are going to use doesn't handle string values, we need to [one-hot encode](https://en.wikipedia.org/wiki/One-hot) all possible categories and store them separately in the frame. This will lead us to have four new columns: Child, Teen, Adult and Elder. On the other hand, we can now remove original Age column, because we don't need it anymore.

# In[ ]:


from sklearn.preprocessing import LabelBinarizer
lb_encoder = LabelBinarizer(sparse_output=False)

age_bins   = [0, 14, 25, 75, 120]
age_labels = ['Child', 'Teen', 'Adult', 'Elder']

# Binning
for ds in [df_train, df_test]:
    ds['Age'].fillna(ds['Age'].median(), inplace=True)
    ds['AgeBin'] = pd.cut(ds['Age'], bins=age_bins, labels=age_labels, include_lowest=True)

g = sns.FacetGrid(df_train, col="AgeBin").map(plt.hist, "Survived")
    
# Encode
lb_encoder.fit(df_train['AgeBin'])
df_train, df_test = [x.join(pd.DataFrame(lb_encoder.transform(x['AgeBin']), columns=age_labels)) for x in [df_train, df_test]]
df_train, df_test = [x.drop(['Age', 'AgeBin'], axis=1) for x in [df_train, df_test]]


# ### 6.2. Fare
# 
# This column is quite similar to Age feature. We also deal with a continuous variable that we want to arrange into several bins, here: Low/Medium/High and Farge fare values will be our bins. There's one additional step for this column though. Before we divide our values, we would like to reduce [data skewness](https://en.wikipedia.org/wiki/Skewness) in Fare column. In order to do this, we will apply a logarithmic scale before (look at the code snippet below).
# 
# If you look closely at diagrams below, you will notice that for each consecutive fare bin, proportion between people who did not survive and those who did changes. It might not be the best diagram to show this, but it seems that the higher the fare (bin) is, the more chances to survive we have.

# In[ ]:


fare_bins   = [0, 2.0, 2.7, 3.4, 6.3]
fare_labels = ['LowFare', 'MedFare', 'HighFare', 'LargeFare']

# Reduce skewness & binning
for ds in [df_train, df_test]:
    ds['Fare'] = ds['Fare'].fillna(ds['Fare'].median())
    ds['Fare'] = ds['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
    ds['FareBin'] = pd.cut(ds['Fare'], bins=fare_bins, labels=fare_labels, include_lowest=True)

g = sns.FacetGrid(df_train, col="FareBin").map(plt.hist, "Survived")
# df_train[['FareBin', 'Survived']].groupby('FareBin', as_index=True).mean().plot(kind='bar')
    
# Encode
lb_encoder.fit(df_train['FareBin'])
df_train, df_test = [x.join(pd.DataFrame(lb_encoder.transform(x['FareBin']), columns=fare_labels)) for x in [df_train, df_test]]
df_train, df_test = [x.drop(['Fare', 'FareBin'], axis=1) for x in [df_train, df_test]]


# ### 6.3. Embarked
# 
# Here it gets a little easier. We have three possible embarkation ports: C, Q, S and only few values missing. We will repeat our procedure, that is:
# * we fill missing values (we can't calculate median this time but we can calculate a [mode](https://en.wikipedia.org/wiki/Mode_(statistics)), which is `S`
# * we encode Embarked column and represent it as one-hot vectors, it results in three different columns - C, Q and S respectively

# In[ ]:


df_train, df_test = [x.fillna('S') for x in [df_train, df_test]]

# Encode 
lb_encoder.fit(df_train['Embarked'])
df_train, df_test = [x.join(pd.DataFrame(lb_encoder.transform(x['Embarked']), columns=lb_encoder.classes_)) for x in [df_train, df_test]]
df_train, df_test = [x.drop('Embarked', axis=1) for x in [df_train, df_test]]


# ### 6.4. Name
# 
# Name column contains a full name of passenger along with a title e.g. Mr, Miss, Master or Col. Names don't seem to be useful for us, but we could definitely try to use titles because they indicate a status somehow. So what we do here is we split Name to retrieve only the title in the first step. Then we try to reduce number of possible titles by grouping similar one e.g. [Mlle](https://en.wikipedia.org/wiki/Mademoiselle_(title)) can be Miss and [Mme](https://en.wikipedia.org/wiki/Madam) can be replaced with Mrs.
# 
# At the end, we also want to group up all titles that don't appear often and replace them with "Rare" title.

# In[ ]:


for ds in [df_train, df_test]:
    ds['Title'] = [x.split(',')[1].split('.')[0].strip() for x in ds['Name']]

    ds['Title'] = ds['Title'].replace(to_replace=['Mlle', 'Ms'], value='Miss')
    ds['Title'] = ds['Title'].replace(to_replace='Mme', value='Mrs')
    ds['Title'] = ds['Title'].apply(lambda i: i if i in ['Mr', 'Mrs', 'Miss', 'Master'] else 'Rare')

print(df_train['Title'].value_counts() + df_test['Title'].value_counts())
print(df_train[['Title', 'Survived']].groupby('Title', as_index=True).mean().sort_values('Survived', ascending=False))

# Encode
lb_encoder.fit(df_train['Title'])
df_train, df_test = [x.join(pd.DataFrame(lb_encoder.transform(x['Title']), columns=lb_encoder.classes_)) for x in [df_train, df_test]]
df_train, df_test = [x.drop(['Name', 'Title'], axis=1) for x in [df_train, df_test]]


# ### 6.5. Pclass
# 
# This one is easy - we just use our [LabelBinarizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html) to convert values of Pclass column to one-hot vectors.

# In[ ]:


pclass_cols = ['UpperClass', 'MiddleClass', 'LowerClass']

lb_encoder.fit(df_train['Pclass'])
df_test  = df_test.join(pd.DataFrame(lb_encoder.transform(df_test['Pclass']), columns=pclass_cols))
df_train = df_train.join(pd.DataFrame(lb_encoder.transform(df_train['Pclass']), columns=pclass_cols))
df_test.drop('Pclass', axis=1, inplace=True)
df_train.drop('Pclass', axis=1, inplace=True)


# ### 6.6. Parch/SibSp
# 
# In this section we handle two columns, which are:
# > sibsp: 	# of siblings / spouses aboard the Titanic 	
# > parch: 	# of parents / children aboard the Titanic
# 
# We will use both values to calculate the size of families for each passenger which is $SibSp + Parch + 1$ (for passenger itself). Then we create three columns depending on the family size i.e. IsAlone, SmallFamily (2 to 4), LargeFamily (5 or more). What's new here is that we use a simple map function instead of LabelBinarizer this time.

# In[ ]:


for ds in [df_train, df_test]:
    df_family = (ds['Parch'] + ds['SibSp'] + 1).astype(int)
    ds.drop(['Parch', 'SibSp'], axis=1, inplace=True)
    ds['IsAlone'] = df_family.map(lambda x: 1 if x == 1 else 0)
    ds['SmallFamily'] = df_family.map(lambda x: 1 if 2 <= x <= 4 else 0)
    ds['LargeFamily'] = df_family.map(lambda x: 1 if x >= 5 else 0)


# ### 6.7. Sex
# 
# Once again we encode a categorical feature.  
# We simply create a column IsFemale where 1 means that passenger is female and 0 obviously is for men.

# In[ ]:


for ds in [df_train, df_test]:
    ds['IsFemale']  = (ds['Sex'] == 'female').astype(int)
    ds.drop('Sex', axis=1, inplace=True)
    
g = sns.FacetGrid(df_train, col="IsFemale").map(plt.hist, "Survived")


# ### 6.8. PassengerId/Ticket
# 
# There are two columns left that we didn't process, because they don't seem that useful. Although, before we drop `PassengerId` column we would like to save it to an array, because we need those indices for creating a submission file at the end of this notebook.

# In[ ]:


test_passenger_ids = np.array(df_test['PassengerId'])

df_test.drop(['PassengerId', 'Ticket'], axis=1, inplace=True)
df_train.drop(['PassengerId', 'Ticket'], axis=1, inplace=True)


# ### 7. See how the data looks like now

# In[ ]:


df_train.info()
df_train.sample(n=5)


# ### 8. Features and labels split
# 
# Now when we've got our features preprocessed, we can split our data frames into training and test set and also into features and labels.  
# Notice that we've got only one label which is `Survived` column and we put it into `y_train` array. 

# In[ ]:


X_train = df_train.loc[:, df_train.columns != 'Survived']
X_test  = np.array(df_test)
y_train = np.array(df_train['Survived'])
print(f"Train features: {X_train.shape}\nTrain labels: {X_test.shape}\nTesting features: {y_train.shape}")


# ### 9. Cross validation
# 
# Here's the final part - we are going to look for a well-performing model, train it and evaluate.
# We use [GridSearch](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) cross validation to search for best [hyper-parameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)) of our classifier (which is [Random Forest](https://en.wikipedia.org/wiki/Random_forest)). In fact the only parameter we are going to change is a number of estimators inside the ensembled forest (that is number of decision trees used underneath).
# 
# After performing cross validation (using 4 folds), we check what is the best (and the mean) score and we use the best estimator to predict values for the whole test set. At last we create a new data frame with passenger indices and predicted values (whether people survived or not) and we save submission file so we can upload it and see our final score. :)

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {'n_estimators': [200, 300, 500, 600, 700, 800], 'random_state': [42]}
grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=4, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print(f"Best score: {round(grid_search.best_score_, 4)}\n"
      f"Mean score: {round(grid_search.cv_results_['mean_test_score'].mean(), 4)}\n"
      f"Top 5 scores:{sorted(grid_search.cv_results_['mean_test_score'], reverse=True)[:5]}\n"
      f"Best params: {grid_search.best_params_}")

best_estimator = grid_search.best_estimator_
best_estimator.fit(X_train, y_train)
y_test = best_estimator.predict(X_test)

df_submission = pd.DataFrame({'PassengerId': test_passenger_ids, 'Survived': y_test})
df_submission.to_csv(r'submission.csv', index=False)
# df_submission.sample(n=10)


# 
