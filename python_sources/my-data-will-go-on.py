#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, log_loss


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore")
plt.rcParams['figure.figsize'] = (10, 6)
pd.options.display.max_rows = 50
pd.options.display.max_columns = 100


# ### Functions

# In[ ]:


def plot_categorical_feature_relation(data, feature, hue):
    sns.barplot(data=data, x=hue, y=feature, orient='h', ci=None, ec='black')
    plt.tight_layout()
    plt.show()


# In[ ]:


def plot_numerical_feature_distribution(data, feature, target):
    sns.distplot(data.loc[data[target]==0, feature].dropna(), kde=False, label='0')
    sns.distplot(data.loc[data[target]==1, feature].dropna(), kde=False, label='1')
    plt.legend(title=target, loc='best')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()


# In[ ]:


def get_title(data):
    title_series = data['Name'].str.split('.').str[0].str.split(' ').str[-1]
    title_series.loc[title_series=='Mme'] = 'Mrs'
    title_series.loc[title_series.isin(['Mlle', 'Ms'])] = 'Miss'
    title_series.loc[~(title_series.isin(['Mr', 'Master', 'Mrs', 'Miss']))] = 'Other'
    title_series.name = 'Title'
    return title_series.to_frame()


# In[ ]:


def get_family_size(data):
    family_size_series = (data['SibSp'] + data['Parch'] + 1)
    family_size_series.name = 'FamilySize'
    return family_size_series.to_frame()


# In[ ]:


def get_is_alone(data):
    return pd.DataFrame(np.where(data['SibSp']+data['Parch']==0, 1, 0), index=data.index, columns=['IsAlone'])


# In[ ]:


def mode(data):
    return stats.mode(data)[0]


# In[ ]:


def undersample(X_train, y_train):
    sample_size = y_train.value_counts().min()
    y_train_sampled = pd.concat([y_train[y_train==1].sample(sample_size, random_state=1),
                                y_train[y_train==0].sample(sample_size, random_state=1)], axis=0)                                 
    X_train_sampled = X_train.loc[y_train_sampled.index]
    return X_train_sampled, y_train_sampled


# In[ ]:


def create_grid_search(est, p_grid, X_train, y_train, scr, refit, n=2):
    cv = StratifiedKFold(n_splits=n, shuffle=True, random_state=1)  
    return GridSearchCV(estimator=est, param_grid=p_grid, scoring=scr, n_jobs=4, cv=cv, verbose=0, refit=refit)


# In[ ]:


def compute_nested_score(est, X_train, y_train, scr, n=5):
    cv = StratifiedKFold(n_splits=n, shuffle=True, random_state=1)
    
    # undersampling
    X_train_sampled, y_train_sampled = undersample(X_train, y_train)
    
    # scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sampled.astype(float))
    
    nested_score = pd.DataFrame(cross_validate(est, X=X_train_scaled, y=y_train_sampled, cv=cv, n_jobs=4, scoring=scr, return_train_score=True))
    return {'mean_score': nested_score.mean().to_dict(), 'std_score':nested_score.std().to_dict()}


# In[ ]:


def compare_models(est_dict, X_train, y_train, scr, n=5):
    return {name: compute_nested_score(est, X_train, y_train, scr, n) for name, est in est_dict.items()}


# ### Reading Train and Test Datasets

# In[ ]:


train_raw = pd.read_csv('/kaggle/input/titanic/train.csv', index_col='PassengerId')
test_raw = pd.read_csv('/kaggle/input/titanic/test.csv', index_col='PassengerId')


# In[ ]:


print(train_raw.shape)
train_raw.head()


# In[ ]:


print(test_raw.shape)
test_raw.head()


# In[ ]:


target='Survived'


# ### Exploratory Data Analytics

# In[ ]:


# dtypes and number of non-null values in train data
train_raw.info()


# In[ ]:


# dtypes and number of non-null values in test data
test_raw.info()


# ### Target Variable Distribution
# Target variable distribution is imbalanced.

# In[ ]:


train_raw[target].value_counts()/train_raw.shape[0]


# ### Categorical Features vs Target
# Mean values of target variable is examined with respect to different values in categorical features.
# Embarked and Sex are categorical features. Pclass is ordinal.

# In[ ]:


# Upper class passengers are more likely to survive.
plot_categorical_feature_relation(train_raw, 'Pclass', target)


# In[ ]:


# Female passengers are more likely to survive.
plot_categorical_feature_relation(train_raw, 'Sex', target)


# In[ ]:


# Passengers who embarked at Cherbourg are more likely to survive
plot_categorical_feature_relation(train_raw, 'Embarked', target)


# ### Numerical Features vs Target
# Distribution of the feature is examined with respect to target variable's different values (1/0)

# In[ ]:


sns.pairplot(data=train_raw, vars=['Age', 'SibSp', 'Parch', 'Fare'], hue=target, diag_kind='kde', markers='.', 
             diag_kws={'linewidth':.5}, height=3)
plt.tight_layout()
plt.show()


# In[ ]:


# Some age groups are more likely to survive (e.g. Age<5).
# Some age groups are less likely to survive (e.g. 15<Age<30).
plot_numerical_feature_distribution(train_raw, 'Age', target)


# In[ ]:


# Passengers who do not have any siblings/spouses are less likely to survive.
plot_numerical_feature_distribution(train_raw, 'SibSp', target)


# In[ ]:


# Passengers who do not have any parents/childen are less likely to survive.
plot_numerical_feature_distribution(train_raw, 'Parch', target)


# In[ ]:


# Passengers who pays less fare are less likely to survive
plot_numerical_feature_distribution(train_raw, 'Fare', target)


# ### Creating New Features
# Title, FamilySize and IsAlone features are created.

# In[ ]:


train_titles = get_title(train_raw)
test_titles = get_title(test_raw)


# In[ ]:


train_family_size = get_family_size(train_raw)
test_family_size = get_family_size(test_raw)


# In[ ]:


train_is_alone = get_is_alone(train_raw)
test_is_alone = get_is_alone(test_raw)


# In[ ]:


train = pd.concat([train_raw, train_titles, train_family_size, train_is_alone], axis=1)
test = pd.concat([test_raw, test_titles, test_family_size, test_is_alone], axis=1)


# In[ ]:


# Passengers with Mrs, Miss or Master titles are more likely to survive.
plot_categorical_feature_relation(train, 'Title', target)


# In[ ]:


# Passengers who travel alone are more likely to survive
plot_categorical_feature_relation(train, 'IsAlone', target)


# In[ ]:


# Passengers who have large families are less likely to survive
plot_numerical_feature_distribution(train, 'FamilySize', target)


# #### Correlation Heatmap

# In[ ]:


corr_matrix = train.corr()
masked_corr_matrix = corr_matrix.where(np.tril(np.ones(corr_matrix.shape), k=-1).astype(np.bool))

plt.figure(figsize=(10,10))
sns.heatmap(masked_corr_matrix, cmap='coolwarm', square=True, annot=True, annot_kws={'size':8}, fmt='.2f', cbar=False)
plt.tight_layout()
plt.show()


# In[ ]:


to_drop = ['Ticket', 'Cabin', 'Name', 'Parch', 'SibSp']


# In[ ]:


train = train.drop(to_drop, 1)
test = test.drop(to_drop, 1)


# ### Filling Null Values
# There exists missing values in Age, Fare and Embarked fields. Distribution of those fields are examined with respect to other categorical variables like Pclass, Sex, Title etc. Using this distributions, lookup tables are created and missing values are filled.

# **Age vs Categorical Variables:** Pclass and Title fields are chosen. Sex field is not chosen as it is mostly represented in Title field.
# 

# In[ ]:


train.groupby('Pclass').agg({'Age':'mean'}).sort_values('Age', ascending=False)


# In[ ]:


train.groupby('Sex').agg({'Age':'mean'}).sort_values('Age', ascending=False)


# In[ ]:


train.groupby('Title').agg({'Age':'mean'}).sort_values('Age', ascending=False)


# In[ ]:


train.groupby('Embarked').agg({'Age':'mean'}).sort_values('Age', ascending=False)


# In[ ]:


train.groupby('IsAlone').agg({'Age':'mean'}).sort_values('Age', ascending=False)


# **Fare vs  Categorical Features:** Predictably IsAlone, Pclass and Embarked fields seem related with Fare field.

# In[ ]:


train.groupby('Pclass').agg({'Fare':'mean'}).sort_values('Fare', ascending=False)


# In[ ]:


train.groupby('Sex').agg({'Fare':'mean'}).sort_values('Fare', ascending=False)


# In[ ]:


train.groupby('Title').agg({'Fare':'mean'}).sort_values('Fare', ascending=False)


# In[ ]:


train.groupby('Embarked').agg({'Fare':'mean'}).sort_values('Fare', ascending=False)


# In[ ]:


train.groupby('IsAlone').agg({'Fare':'mean'}).sort_values('Fare', ascending=False)


# **Embarked vs Categorical Features:** Most common Embarked value in all categories is S.
# 

# In[ ]:


train.groupby('Pclass').agg({'Embarked': mode})


# In[ ]:


train.groupby('Sex').agg({'Embarked': mode})


# In[ ]:


train.groupby('Title').agg({'Embarked': mode})


# In[ ]:


train.groupby('IsAlone').agg({'Embarked': mode})


# In[ ]:


# Filling null values in Age with mean Age per Title and Pclass in train data
age_lookup = train.groupby(['Pclass', 'Title'], as_index=False).agg({'Age':'mean'})

train_missing_age_values = pd.merge(train[train['Age'].isnull()][['Pclass', 'Title']], age_lookup, on=['Pclass','Title'], how='left')['Age'].tolist()
train.loc[train['Age'].isnull(), 'Age'] = train_missing_age_values

test_missing_age_values = pd.merge(test[test['Age'].isnull()][['Pclass', 'Title']], age_lookup, on=['Pclass','Title'], how='left')['Age'].tolist()
test.loc[test['Age'].isnull(), 'Age'] = test_missing_age_values


# In[ ]:


# Filling null values in Fare with mean Fare per IsAlone, Pclass and Embarked in train data
fare_lookup = train.groupby(['IsAlone', 'Pclass', 'Embarked'], as_index=False).agg({'Fare':'mean'})

test_missing_fare_values = pd.merge(test[test['Fare'].isnull()][['IsAlone', 'Pclass', 'Embarked']], fare_lookup, on=['IsAlone', 'Pclass', 'Embarked'], how='left')['Fare'].tolist()
test.loc[test['Fare'].isnull(), 'Fare'] = test_missing_fare_values


# In[ ]:


# Filling null values in Embarked with most common Embarked value in train data
test.loc[test['Embarked'].isnull(), 'Embarked'] = 'S'


# In[ ]:


train.head(5)


# In[ ]:


test.head(5)


# Dummy variables are created for categorical features. Pclass is intentionally treated as ordinal variable.

# In[ ]:


train = pd.get_dummies(train, drop_first=True)


# In[ ]:


test = pd.get_dummies(test, drop_first=True)


# In[ ]:


X_train = train.drop([target], 1)
y_train = train[target]
X_test = test


# In[ ]:


X_train.shape


# In[ ]:


y_train.shape


# #### Parameter Optimization and Model Selection
# - Logistic regression, Random Forest, Gradient Boosting and KNN models are used in classification.
# - Parameter optimization and model comparison tasks performed with nested cross validation using grid search and stratified k fold cross validation. 
# - In first step, parameters of each model are tuned and best estimator from each model is compared to each other.
# - Comparison metrics are accuracy, precision, recall and roc_auc.
# - During cross validation, undersampling is applied to data since the target variable has imbalanced distribution. Additionally, features are standardized.

# In[ ]:


lr = LogisticRegression(solver='liblinear')
rf = RandomForestClassifier(random_state=1, n_jobs=4)
gb = GradientBoostingClassifier(random_state=1)
knn = KNeighborsClassifier()

p_grid_lr = [{'penalty': ['l1', 'l2'], 'C': [.5, 1, 2]}]
p_grid_rf = [{'n_estimators': [50, 100, 200], 'min_samples_split': [2, 5, 10]}]
p_grid_gb = [{'n_estimators': [50, 100, 200], 'learning_rate':[.05, .1, .15], 
              'min_samples_split': [2, 5], 'max_depth': [3, 5, 10]}]
p_grid_knn = [{'n_neighbors' : [3, 5, 9]}]

p_grid_list = [p_grid_lr, p_grid_rf, p_grid_gb, p_grid_knn]

est_names = ['LogisticRegression', 'RandomForest', 'GradientBoosting', 'KNearestNeighbours']
est_list = [lr, rf, gb, knn]
est_dict = dict(zip(est_names, est_list))

scoring = {"accuracy": 'accuracy', 'precision': 'precision', 'recall': 'recall', 'roc_auc': 'roc_auc'} 


# In[ ]:


# Estimator dictionary with tuned parameters
grid_search_dict = {name: create_grid_search(est, p_grid, X_train, y_train, scr='roc_auc', refit='roc_auc', n=2) for name, est, p_grid in zip(est_names, est_list, p_grid_list)}


# In[ ]:


# Comparison of estimators with tuned parameters
result = compare_models(grid_search_dict, X_train, y_train, scr=scoring, n=5)


# In[ ]:


# Result of estimator comparison in terms of roc_auc
pd.concat({k: pd.DataFrame(v).unstack().to_frame().T for k, v in result.items()})


# ### Prediction
#  Gradient Boosting and Random Forest gave very similar results. Gradient Boosting is selected arbitrarily as best model. Best model is trained with all train data and test data is predicted.

# In[ ]:


# Undersampling
X_train_sampled, y_train_sampled = undersample(X_train, y_train)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sampled.astype(float))
X_test_scaled = scaler.transform(X_test.astype(float))


# In[ ]:


best_algo = grid_search_dict['GradientBoosting']
best_algo.fit(X_train_scaled, y_train_sampled)
best_params = best_algo.best_params_


# In[ ]:


best_params


# In[ ]:


train_accuracy = accuracy_score(y_train_sampled, best_algo.predict(X_train_scaled))
train_precision = precision_score(y_train_sampled, best_algo.predict(X_train_scaled))
train_recall = recall_score(y_train_sampled, best_algo.predict(X_train_scaled))
train_roc_auc = roc_auc_score(y_train_sampled, best_algo.predict(X_train_scaled))


# In[ ]:


print('Train accuracy is: {}'.format(train_accuracy))
print('Train precision is: {}'.format(train_precision))
print('Train recall is: {}'.format(train_recall))
print('Train roc-auc is: {}'.format(train_roc_auc))


# In[ ]:


pd.Series(best_algo.best_estimator_.feature_importances_, index=X_train_sampled.columns).plot(kind='barh')
plt.xlabel('FeatureImportance')
plt.show()


# In[ ]:


test_predictions = pd.DataFrame(best_algo.predict(X_test_scaled), index=X_test.index, columns=['Predictions'])
test_predictions.head()


# In[ ]:




