#!/usr/bin/env python
# coding: utf-8

# # Analysis

# ## Import Libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# ## Read data and take first look

# In[ ]:


train_set = pd.read_csv('../input/train.csv')
test_set = pd.read_csv('../input/test.csv')


# ### Theoretical analysis

# In[ ]:


train_set.head()


# In[ ]:


train_set.describe()


# In[ ]:


train_set.info()


# ### Visual  Analysis

# Plots of survival based on Categorical Columns-
# 
# 1. SibSp
# 2. Sex
# 3. Pclass
# 4. Embarked
# 5. Parch
# 6. Survived (Yes or No)

# In[ ]:


cat_cols = ['Survived', 'Sex', 'Pclass', 'Embarked', 'Parch', 'SibSp']
fig, axs = plt.subplots(2, 3, figsize=(16, 9))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)



for i in range(2):
    for j in range(3):
        c = i * 3 + j
        ax = axs[i][j]
    
        sns.countplot(train_set[cat_cols[c]], hue=train_set['Survived'], ax = ax)
        ax.set_title(cat_cols[c], fontsize=14, fontweight='bold')
        ax.grid()


# Distribution of Age based on Pclass, Sex and Survival

# In[ ]:


bins = np.arange(0, 80, 5)
g = sns.FacetGrid(train_set, row='Sex', col='Pclass', hue='Survived', margin_titles=True, size=3, aspect=1.1)
g.map(sns.distplot, 'Age', kde=False, bins=bins, hist_kws=dict(alpha=0.6))
g.add_legend()
plt.show()  


# Distribution of Age based on Embarked, Sex and Survival

# In[ ]:


bins = np.arange(0, 80, 5)
g = sns.FacetGrid(train_set, row='Sex', col='Embarked', hue='Survived', margin_titles=True, size=3, aspect=1.1)
g.map(sns.distplot, 'Age', kde=False, bins=bins, hist_kws=dict(alpha=0.6))
g.add_legend()
plt.show()  


# Distribution of Fare based on Pclass, Sex and Survival

# In[ ]:


bins = np.arange(0, 550, 50)
g = sns.FacetGrid(train_set, row='Sex', col='Pclass', hue='Survived', margin_titles=True, size=3, aspect=1.1)
g.map(sns.distplot, 'Fare', kde=False, bins=bins, hist_kws=dict(alpha=0.6))
g.add_legend()
plt.show()  


# Distribution of Fare based on Embarked, Sex and Survival

# In[ ]:


bins = np.arange(0, 550, 50)
g = sns.FacetGrid(train_set, row='Sex', col='Embarked', hue='Survived', margin_titles=True, size=3, aspect=1.1)
g.map(sns.distplot, 'Fare', kde=False, bins=bins, hist_kws=dict(alpha=0.6))
g.add_legend()
plt.show()  


# ## Missing Value Analysis

# In[ ]:


def missing_zero_values_table(df):
    zero_val = (df == 0.00).astype(int).sum(axis=0)
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mz_table = pd.concat([zero_val, mis_val, mis_val_percent], axis=1)
    mz_table = mz_table.rename(
    columns = {0 : 'Zero Values', 1 : 'Missing Values', 2 : '% of Total Values'})
    mz_table['Data Type'] = df.dtypes
    mz_table = mz_table[
        mz_table.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns and " + str(df.shape[0]) + " Rows.\n"      
        "There are " + str(mz_table.shape[0]) +
          " columns that have missing values.")
    return mz_table


# In[ ]:


missing_zero_values_table(train_set)


# **Cabin:** More than 3/4th values are missing, so just ignore this variable as imputing might lead to wrong results <br>
# **Age:** Important feature, need to do further analysis to impute the age values for those missing <br>
# **Embarked:** Another importanyt feature and just 2 missing values which can be possibly filled manually with some further analysis

# In[ ]:


missing_zero_values_table(test_set)


# ### Analysis of Emarked

# In[ ]:


train_set[train_set['Embarked'].isna()]


# In[ ]:


train_set[(train_set['Pclass']==1) & (train_set['Embarked']=='Q')]['Sex'].value_counts()


# Mode will be a good measure to impute the two missing values for Embarked in data

# In[ ]:


train_set['Embarked'][61] = 'S'
train_set['Embarked'][829] = 'S'


# ### Analysis of Age

# In[ ]:


train_set[train_set['Age'].isna()]['Sex'].value_counts()


# We can fill up Age based with Median with Categories of Passenger based on Sex, Pclass, Fare, Embarked. But there are a few other variables which can decide age like Family Size and the Title or Salutation. So we will impute the age later after creating those variables in Feature Engineering.

# ### Analysis of Fare

# In[ ]:


test_set[test_set['Fare'].isna()]


# The missing Fare value can be filled with the Median Fare of Pclass 3 and Embarkment port S

# In[ ]:


test_set.at[152, 'Fare'] = np.nanmedian(test_set[(test_set['Pclass']==3) & (test_set['Embarked']=='S')]['Fare'])


# In[ ]:


test_set['Fare'][152]


# ## Feature Engineering

# ### Create the FamilySize Variable

# In[ ]:


train_set['FamilySize'] = train_set['SibSp'] + train_set['Parch'] + 1
test_set['FamilySize'] = test_set['SibSp'] + test_set['Parch'] + 1


# ### Create the Title variable

# For that let's first analyze the structure of names closely

# In[ ]:


train_set['Name']


# Clearly a trend can be seen in the names
# 

# Let's create a function to extract the title based on that trend

# In[ ]:


def extract_title(name):
    return name.split(',')[1].split()[0].strip()


# In[ ]:


train_set['Title'] = train_set['Name'].apply(extract_title)


# In[ ]:


train_set['Title'].value_counts()


# A lot of redundant Titles can still be removed and the values for this variable can be decreased

# Let's create a function to do that

# In[ ]:


def refine_title(title):
    if title in ['Mr.', 'Sir.', 'Major.', 'Dr.', 'Capt.']:
        return 'mr'
    elif title == 'Master.':
        return 'master'
    elif title in ['Miss.', 'Ms.']:
        return 'miss'
    elif title in ['Mrs.', 'Lady.']:
        return 'mrs'
    else:
        return 'other'


# In[ ]:


train_set['Title'] = train_set['Title'].apply(refine_title)


# In[ ]:


train_set['Title'].value_counts()


# Let's apply same functions to the test set

# In[ ]:


test_set['Title'] = test_set['Name'].apply(extract_title)
test_set['Title'] = test_set['Title'].apply(refine_title)


# In[ ]:


test_set['Title'].value_counts()


# In[ ]:


train_set.head()


# ### Creating the bins for Fare variable

# In[ ]:


sns.distplot(train_set['Fare'])


# From the distribution of fare it seems like the bins should be - 
# 1. 10 units from 0 to 100
# 2. 100-200
# 3. 200-300
# 4. 300+

# Let's check the test set also before making final decision

# In[ ]:


sns.distplot(test_set['Fare'])


# The distribution looks very much similar, so let's create the function for binning the Fare

# In[ ]:


fare_bins = [-np.inf, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, np.inf]
fare_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
train_set['FareBin'] = pd.cut(train_set['Fare'], bins=fare_bins, labels=fare_labels)
test_set['FareBin'] = pd.cut(test_set['Fare'], bins=fare_bins, labels=fare_labels)


# In[ ]:


train_set.head()


# In[ ]:


test_set.head()


# ### Fiiling missing values for Age

# We have all the variables now on which Age might depend

# Let's write the script to fill up the age variable

# In[ ]:


def fill_age(df):
    for idx, row in df.iterrows():
        if pd.isnull(row['Age']):
            value = df[ 
                (df['Pclass']==row['Pclass']) & 
                (df['Sex']==row['Sex']) & 
                (df['Embarked']==row['Embarked']) & 
                (df['Title']==row['Title']) & 
                (df['FareBin']==row['FareBin'])
            ]['Age'].median()
            if pd.isnull(value):
                value = df[ 
                (df['Sex']==row['Sex']) & 
                (df['Title']==row['Title']) & 
                (df['FareBin']==row['FareBin'])
            ]['Age'].median()
            if pd.isnull(value):
                value = df[df['Title']==row['Title']]['Age'].median()
            df.at[idx, 'Age'] = value 


# In[ ]:


fill_age(train_set)
fill_age(test_set)


# Let's check the missing values in train and test set now

# In[ ]:


missing_zero_values_table(train_set)


# In[ ]:


missing_zero_values_table(test_set)


# ### Creating bins for Age

# DIstribution of Age

# In[ ]:


sns.distplot(train_set['Age'])


# In[ ]:


sns.distplot(test_set['Age'])


# In[ ]:


train_set['Age'].describe()


# In[ ]:


age_bins = [-np.inf, 10, 20, 30, 40, 50, 60, 70, 80, np.inf]
age_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]
train_set['AgeBin'] = pd.cut(train_set['Age'], bins=age_bins, labels=age_labels)
test_set['AgeBin'] = pd.cut(test_set['Age'], bins=age_bins, labels=age_labels)


# In[ ]:


train_set.head()


# ### Select features to train model 

# In[ ]:


features = ['Pclass', 'Sex', 'Embarked', 'FamilySize', 'Title', 'FareBin', 'AgeBin', 'Fare', 'Age']


# In[ ]:


y_train = train_set['Survived']


# In[ ]:


x_train = train_set[features]
x_test = test_set[features]


# In[ ]:


x_train.head()


# ### Normalize the data using Standard Scaler

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train[['Age', 'Fare', 'FamilySize']] = scaler.fit_transform(x_train[['Age', 'Fare', 'FamilySize']])
x_test[['Age', 'Fare', 'FamilySize']] = scaler.fit_transform(x_test[['Age', 'Fare', 'FamilySize']])


# In[ ]:


x_train.head()


# In[ ]:


x_test.head()


# ### Convert Non-Continuous variables to Categorical Variables

# In[ ]:


x_train['Pclass'] = pd.Categorical(x_train['Pclass'])
x_train['Sex'] = pd.Categorical(x_train['Sex'])
x_train['Embarked'] = pd.Categorical(x_train['Embarked'])
x_train['Title'] = pd.Categorical(x_train['Title'])

x_test['Pclass'] = pd.Categorical(x_test['Pclass'])
x_test['Sex'] = pd.Categorical(x_test['Sex'])
x_test['Embarked'] = pd.Categorical(x_test['Embarked'])
x_test['Title'] = pd.Categorical(x_test['Title'])


# In[ ]:


x_train.info()


# In[ ]:


x_test.info()


# ### Convert Categorical data to dummies

# In[ ]:


x_train = pd.get_dummies(x_train)
x_test = pd.get_dummies(x_test)


# In[ ]:


x_train.head()


# In[ ]:


x_test.head()


# ## Save the data into new files

# In[ ]:


X = x_train.copy()
y = y_train.copy()
test_data = x_test.copy()

X.to_csv('X.csv', index=False, header=True)
test_data.to_csv('test_data.csv', index=False, header=True)
y.to_csv('y.csv', index=False, header=True)


# ### Save the Scaler to use in deployment

# In[ ]:


from sklearn.externals import joblib

joblib.dump(scaler, 'scaler.pkl')


# # Model

# ## Import Libraries

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegressionCV
from xgboost import XGBClassifier


# ## Train-Validation set split

# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=25)


# In[ ]:


X_train.head()


# In[ ]:


y_train.value_counts()


# In[ ]:


y_valid.value_counts()


# The resultant Training and Validation sets after the split look like a true fit for the original data, so we will proceed with these sets

# ## Model Comparison (Validation Set)

# ### Gaussian NB classifier

# In[ ]:


gnb_clf = GaussianNB()
gnb_clf.fit(X_train, y_train)
gnb_prediction = gnb_clf.predict(X_valid)
print(classification_report(y_valid, gnb_prediction))


# ### Support Vector Classifier

# In[ ]:


svc_clf = SVC(kernel='linear')
svc_clf.fit(X_train, y_train)
svc_prediction = svc_clf.predict(X_valid)
print(classification_report(y_valid, svc_prediction))


# ### Decision Tree Classifier

# In[ ]:


tree_clf = DecisionTreeClassifier(max_depth=5)
tree_clf.fit(X_train, y_train)
tree_prediction = tree_clf.predict(X_valid)
print(classification_report(y_valid, tree_prediction))


# ### Random Forest Classifier

# In[ ]:


rf_clf = RandomForestClassifier(n_estimators=200, max_depth=10)
rf_clf.fit(X_train, y_train)
rf_prediction = rf_clf.predict(X_valid)
print(classification_report(y_valid, rf_prediction))


# ### Adaboost Classifier

# In[ ]:


ada_clf = AdaBoostClassifier()
ada_clf.fit(X_train, y_train)
ada_prediction = ada_clf.predict(X_valid)
print(classification_report(y_valid, ada_prediction))


# ### KNN Classifier

# In[ ]:


knn_clf = KNeighborsClassifier(6)
knn_clf.fit(X_train, y_train)
knn_prediction = knn_clf.predict(X_valid)
print(classification_report(y_valid, knn_prediction))


# ### MLPC Classifier

# In[ ]:


mlpc_clf = MLPClassifier(alpha=1, max_iter=5000)
mlpc_clf.fit(X_train, y_train)
mlpc_prediction = mlpc_clf.predict(X_valid)
print(classification_report(y_valid, mlpc_prediction))


# ### Gaussian Process Classifier

# In[ ]:


gp_clf = GaussianProcessClassifier(1.0 * RBF(1.0))
gp_clf.fit(X_train, y_train)
gp_prediction = gp_clf.predict(X_valid)
print(classification_report(y_valid, gp_prediction))


# ### Logistic Regression

# In[ ]:


log_clf = LogisticRegressionCV(cv=5, max_iter=5000)
log_clf.fit(X_train, y_train)
log_prediction = log_clf.predict(X_valid)
print(classification_report(y_valid, log_prediction))
log_prediction_soft = log_clf.predict_proba(X_valid)


# ### XGBoost Classifier

# In[ ]:


xg_clf = XGBClassifier()
xg_clf.fit(X_train, y_train)
xg_prediction = xg_clf.predict(X_valid)
print(classification_report(y_valid, xg_prediction))


# ### Comparison

# In[ ]:


models = [gnb_clf, knn_clf, log_clf, svc_clf, tree_clf, ada_clf, xg_clf, rf_clf, mlpc_clf, gp_clf]
model_names = ['Gaussian NB', 'KNN', 'Logistic Reg', 'SVC', 'Decision Tree', 'Adaboost', 'XGBoost', 
              'Random Forest', 'MLPC', 'Gaussian Process']
accuracies = [np.round(m.score(X_valid, y_valid), 2) for m in models]
result_df = pd.DataFrame({'Model': model_names, 
                         'Accuracy': accuracies}).set_index('Model').sort_values('Accuracy', ascending=False)


# In[ ]:


result_df


# ### Train Final Model

# I will train the final model with Random Forest Classifer based on previous model performance comparison

# Final model will be trained on full data

# In[ ]:


final_model = RandomForestClassifier(max_depth=10, n_estimators=200).fit(X, y)


# In[ ]:


print(final_model.score(X_valid, y_valid))


# In[ ]:


print(final_model.score(X_train, y_train))


# ## Create Kaggle Submission file

# In[ ]:


test_ids = test_set['PassengerId']
prediction = final_model.predict(test_data)
submission = pd.DataFrame({'PassengerId': test_ids,
                          'Survived': prediction})


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('titanic_challenge_submission_vedant511.csv', index=False, header=True)


# ## Save and export the model

# In[ ]:


from sklearn.externals import joblib

joblib.dump(final_model, 'model.pkl')

