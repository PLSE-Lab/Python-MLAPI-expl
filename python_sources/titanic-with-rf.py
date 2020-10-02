#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train_df = pd.read_csv('../input/train.csv', index_col=0)
test_df = pd.read_csv('../input/test.csv', index_col=0)
gender_sub = pd.read_csv('../input/gender_submission.csv')
combined_df = pd.concat([train_df, test_df])


# In[ ]:


train_df.info(), test_df.info()


# There are few nulls in Embarked in both data set, quite a lot of empty values in Cabin feature and some in Age.

# In[ ]:


train_df.Cabin.isnull().sum()/len(train_df), test_df.Cabin.isnull().sum()/len(test_df)


# Between 77% and 78% is the percentage of nulls for Cabin in our datasets. That high percentage of empty values means no valuable insight could be extracted from this features.

# In[ ]:


combined_df['Age'].fillna(combined_df['Age'].mode().values[0], inplace=True)
combined_df['Embarked'].fillna(combined_df.Embarked.mode().values[0], inplace=True)
combined_df['Fare'].fillna(combined_df.Fare.median(), inplace=True)
combined_df.drop(columns = 'Cabin', inplace=True)


# In[ ]:


combined_df.info()


# In[ ]:


feats = test_df.columns
train_df = combined_df[~combined_df.Survived.isnull()]
test_df = combined_df[combined_df.Survived.isnull()].drop(columns='Survived')
data_sets = [train_df, test_df]


# In[ ]:


train_df.info()
test_df.info()


# # Data visualization

# In[ ]:


sns.countplot(train_df.Survived);
plt.title("Histogram of Survival in training data");


# In[ ]:


train_df.Survived.value_counts()/len(train_df)*100


# 61.6% of data has been classified as 0 - 'Not Survived' and 38.4 as 1 - 'Survived'. The split is reasonable there should be no issues related to imbalanced data for classifier.

# In[ ]:


train_df.dtypes


# If we add SibSp & Parch together the results is number of family member onborad. Sounds like a good new feature.

# In[ ]:


feats = train_df.columns.tolist()
len(train_df.Ticket.unique())


# Feature ticket has a lot of unique values, not uniques are probably a result of one ticket for couples or families/people starting jorney from the same port & same class etc...
# I don't have knowledge that this feature contains any hidden information at this point and decide to exclude it from analysis. PassengerId is unique.
# Feature name is also unique as a whole but it is possible to extract some valuable information from it.

# In[ ]:


titles = ['Mr', 'Mrs', 'Miss', 'Master']


# In[ ]:


def assign_titles(data):
    temp_titles_df = pd.DataFrame(index = data.index)
    temp_titles_df['Title1'] = data['Name'].apply(lambda x: titles[0] if titles[0] in x else None)
    temp_titles_df['Title2'] = data['Name'].apply(lambda x: titles[1] if titles[1] in x else None)
    temp_titles_df['Title3'] = data['Name'].apply(lambda x: titles[2] if titles[2] in x else None)
    temp_titles_df['Title4'] = data['Name'].apply(lambda x: titles[3] if titles[3] in x else None)
    
    def _return_corect_col(row):
        value = "Other"
        if row['Title1']:
            value = row['Title1']
        if row['Title2']:
            value = row['Title2']
        if row['Title3']:
            value = row['Title3']
        if row['Title4']:
            value = row['Title4']
        return value
    
    temp_titles_df['Title'] = temp_titles_df.apply(lambda x : _return_corect_col(x), axis=1)
    
    return pd.merge(data, temp_titles_df[['Title']], left_index=True, right_index=True)['Title']


# In[ ]:


for data in data_sets:
    data['Title'] = assign_titles(data) # engineering Title from feature Name
    data['FamilySize'] = data['SibSp'] + data['Parch'] # new feature as a combination of two others


# In[ ]:


data_sets[0].head()


# Now since we are done with feature engineering I can drop few features:

# In[ ]:


feats_to_drop = ['Ticket', 'Name']
for data in data_sets:
    data.drop(columns=feats_to_drop, inplace=True)


# In[ ]:


cat_feats = train_df.dtypes[train_df.dtypes == 'object'].index.values.tolist()
num_feats = train_df.dtypes[train_df.dtypes != 'object'].index.values.tolist()
cat_feats, num_feats


# ## Visualizing continous features

# In[ ]:


plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.title('Fare')
plt.boxplot(train_df.Fare);

plt.subplot(1,2,2)
plt.title('Age')
plt.boxplot(train_df.Age);


# In[ ]:


print ("Fare median: {}, Fare mean: {} \nAge median: {}, Age mean: {}".format(
    np.median(train_df.Fare), np.mean(train_df.Fare), np.median(train_df.Age), np.mean(train_df.Age)))


# In[ ]:


plt.figure(figsize=(14,5))
plt.subplot(1,2,1)

sns.distplot(train_df[train_df.Survived == 0]['Fare'], label='Not Survived');
sns.distplot(train_df[train_df.Survived == 1]['Fare'], label='Survived');
plt.legend();

plt.subplot(1,2,2)
sns.distplot(train_df[train_df.Survived == 0]['Age'], label='Not Survived');
sns.distplot(train_df[train_df.Survived == 1]['Age'], label='Survived');
plt.legend();


# In[ ]:


a = sns.FacetGrid(train_df, hue = 'Survived', aspect=3)
a.map(sns.kdeplot, 'Age', shade= True );
plt.title("Ladny wykres");


# The distribution of Fare shows that in the range of lowest Fare <0, 40?) there has been more 'Not Survived' examples, for Fares higher than ~50 there is visible margin -> more examples marked as 'Survived'.
# Looking at the Age distrubition, one can see that for lowest values of Age <0,10?) there is more 'Surived' examples, same is visible at range around 30-40.

# ## Categorical features

# In[ ]:


train_df = data_sets[0]

plt.figure(figsize=(16,10))

plt.subplot(2,3,1)
plt.title("Survival by class");
sns.countplot(x='Pclass', hue='Survived', data=train_df);
plt.legend();

plt.subplot(2,3,2)
plt.title("Survival by sex");
sns.countplot(x='Sex', hue='Survived', data=train_df);
plt.legend();

plt.subplot(2,3,3)
plt.title("Survival by Family Size");
sns.countplot(x='FamilySize', hue='Survived', data=train_df);
plt.legend();

plt.subplot(2,3,4)
plt.title("Survival by Embarked");
sns.countplot(x='Embarked', hue='Survived', data=train_df);
plt.legend();

plt.subplot(2,3,5)
plt.title("Survival by Title");
sns.countplot(x='Title', hue='Survived', data=train_df);
plt.legend();


# Analyzing graphs from left to right.
# Pclass has significant impact on dependent variable. The amount of people who survived is increasing with increasing socio-economic status.
# Between male and females, most of male has not survived whereas most of females has survived.
# People who has from 1-3 family members onboard tend to survive more likely.
# More people from those embarked in Cherbourg has survived.
# Looking at Title we can only confirm previous findings that more womens survived than mans.

# In[ ]:


# combined_df['Sex_cat'] = pd.factorize(combined_df.Sex)[0]
# combined_df['Embarked_cat'] = pd.factorize(combined_df.Embarked)[0]


# In[ ]:


train_df.head()


# In[ ]:


corr_df = train_df.corr()
sns.heatmap(corr_df, vmin=-1, vmax=1, center=0,
    cmap= sns.diverging_palette(130, 275, n=200),
    square=True);


# FamilySize is highly correlated with SibSp and Parch - it is not a suprise because it has been built as a combination of them. There is a decent correlation between Pclass and Fare, Survived and Fare (seen on plots), Survived and Pclass (also seen on plots) and few others.

# In[ ]:


corr_df = train_df.corr().abs().unstack().sort_values(ascending=False).reset_index().rename(
    columns={'level_0':'Feature 1', 'level_1':'Feature 2', 0:'Corr coef'})
corr_df.drop(corr_df[corr_df['Corr coef']==1].index, inplace=True)
corr_df = corr_df.iloc[1::2]
corr_df.reset_index(drop=True).iloc[range(0,10),:]


# #### Converting categorical variables to numerical

# In[ ]:


for i in range(0, len(data_sets)):
    data_sets[i] = pd.get_dummies(data_sets[i])


# #### Preparing for modelling

# In[ ]:


train_df, test_df = data_sets
X_train, X_val, Y_train, Y_val = train_test_split(train_df.drop(columns='Survived'), train_df.Survived)
X_test = data_sets[1]


# ## Modelling

# In[ ]:


rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, Y_train)
y_train_pred = rf_clf.predict(X_train)
y_val_pred = rf_clf.predict(X_val)
y_val_pred_prob = rf_clf.predict_proba(X_val)[:,1]
metrics.accuracy_score(Y_train, y_train_pred), metrics.accuracy_score(Y_val, y_val_pred)


# Accuracy score says how many did the classifier labeled correctly out of all samples.
# 96% accuracy score on training test and 83% on validation test.
# Let's see confussion matrix.

# In[ ]:


conf_mat = confusion_matrix(y_val_pred, Y_val)
sns.heatmap(pd.DataFrame(conf_mat, index=['Survived', 'Not Survived'], columns = ['Survived', 'Not Survived']), annot=True, vmin=0);
plt.xlabel('Actual');
plt.ylabel('Predicted');


# In[ ]:


tp, fn, fp, tn = confusion_matrix(y_val_pred, Y_val).ravel()
tp, fn, fp, tn


# In[ ]:


metrics.precision_recall_fscore_support(Y_val, y_val_pred)


# In[ ]:


fpr, tpr, thresholds = roc_curve(Y_val, y_val_pred_prob)
plt.plot(fpr, tpr);
plt.plot([[0,0], [1,1]], linestyle='dashed');
plt.ylabel('True positive rate');
plt.xlabel('False positive rate');
plt.title('AUC is {}'.format(roc_auc_score(Y_val, y_val_pred_prob)));


# #### Hyperparameters optimization with GridSearchCV

# In[ ]:


params = {'n_estimators' : [5, 10, 20, 50, 100, 200], 'criterion' : ['gini', 'entropy'],
          'max_depth': [2, 4, 6, 8, 10, None], 'random_state':[0]}


# In[ ]:


rf = RandomForestClassifier()
clf = GridSearchCV(rf, params, cv=3)
clf.fit(X_train, Y_train)


# In[ ]:


clf.best_params_


# In[ ]:


rf_clf2 = RandomForestClassifier(criterion='entropy', max_depth=8, n_estimators= 100, random_state= 0)
rf_clf2.fit(X_train, Y_train)
y_train_pred = rf_clf2.predict(X_train)
y_val_pred = rf_clf2.predict(X_val)
y_val_pred_prob = rf_clf2.predict_proba(X_val)[:,1]
metrics.accuracy_score(Y_train, y_train_pred), metrics.accuracy_score(Y_val, y_val_pred)


# In[ ]:


fpr, tpr, thresholds = roc_curve(Y_val, y_val_pred_prob)
plt.plot(fpr, tpr);
plt.plot([[0,0], [1,1]], linestyle='dashed');
plt.ylabel('True positive rate');
plt.xlabel('False positive rate');
plt.title('AUC is {}'.format(roc_auc_score(Y_val, y_val_pred_prob)));


# #### Since we have the best hyperparameters for Random Forest we can fit on whole training data and make a prediciton on test df

# In[ ]:


rf_clf3 = RandomForestClassifier(criterion='entropy', max_depth=8, n_estimators= 100, random_state= 0)


# In[ ]:


rf_clf3.fit(train_df.drop(columns='Survived'), train_df.Survived)
y_pred_test = rf_clf3.predict(X_test)


# In[ ]:


submission = pd.DataFrame(y_pred_test, index=X_test.index, columns=['Survived'])
submission.head()

