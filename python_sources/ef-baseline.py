#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from typing import List, Union, Dict, Optional, Tuple
from datetime import datetime
from os.path import join as pjoin

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# In[ ]:


INPUT_PATH = '../input'
DATA_PATH = pjoin(INPUT_PATH, 'lab34-classification-table')


# In[ ]:


train = pd.read_csv(pjoin(DATA_PATH, 'train.csv'), index_col=0)
test = pd.read_csv(pjoin(DATA_PATH, 'test.csv'), index_col=0)


# In[ ]:


train = train.astype({
    'birth_date': 'datetime64[ns]',
    'contact_date': 'datetime64[ns]',
})
test = test.astype({
    'birth_date': 'datetime64[ns]',
    'contact_date': 'datetime64[ns]',
})


# # Get basic data info

# In[ ]:


train.shape, test.shape


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.describe()


# In[ ]:


test.describe()


# In[ ]:


train.describe(include='object')


# In[ ]:


test.describe(include='object')


# In[ ]:


y = train['y']
y.mean()


# ### Data description
# 
# The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. 
# 
# #### Attribute Information:
# 
# *Input variables:*
# 
#  - birth_date (date)
#  - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
#  - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
#  - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
#  - default: has credit in default? (categorical: 'no','yes','unknown')
#  - housing: has housing loan? (categorical: 'no','yes','unknown')
#  - loan: has personal loan? (categorical: 'no','yes','unknown')
#  
#  - contact: contact communication type (categorical: 'cellular','telephone') 
#  - contact_date (date)
#  
#  - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
#  - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
#  - previous: number of contacts performed before this campaign and for this client (numeric)
#  - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
# 
# *Output variable (desired target):*
# 
#  - y - has the client subscribed a term deposit? (binary: 'yes','no')
# 
# 
# The full description can be found here: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

# # Explore data

# ### Some helpful functions

# In[ ]:


def get_cat_features(X: pd.DataFrame) -> List[str]:
    return X.columns[X.dtypes == 'O'].tolist()


# In[ ]:


def label_distplots(values, labels, kde=True, hist=True):
    sns.distplot(values[labels == 1], kde=kde, hist=hist, label='Label=1', norm_hist=True)
    sns.distplot(values[labels == 0], kde=kde, hist=hist, label='Label=0', norm_hist=True)
    plt.legend();
    
def set_figsize(width, height):
    plt.rcParams['figure.figsize'] = width, height


# ## Explore features distribution

# ### Dates

# In[ ]:


train['contact_date'].min(), train['contact_date'].max()


# In[ ]:


contact_year = (train['contact_date'] - datetime(2000, 1, 1)).dt.days / 365 + 2000
sns.distplot(contact_year)


# In[ ]:


label_distplots(contact_year, y, hist=False)


# In[ ]:


train['birth_date'].min(), train['birth_date'].max()


# In[ ]:


age_when_contact = (train['contact_date'] - train['birth_date']).dt.days / 365
age_when_contact.name = 'age_when_contact'
sns.distplot(age_when_contact)


# In[ ]:


label_distplots(age_when_contact, y, hist=False)


# ### Categorical features

# In[ ]:


cat_features = get_cat_features(train)
for col in cat_features:
    pd.crosstab(train[col], y).plot(kind='bar')


# In[ ]:


pd.pivot_table(
    data=train,
    index='poutcome',
    values='y',
    aggfunc=['mean', 'count'],
).style.bar(color='#339999', vmin=0)


# ### Numerical features

# In[ ]:


train['campaign'].max()


# In[ ]:


sns.violinplot(x='y', y='campaign', data=train.query('campaign < 11'))


# In[ ]:


train['pdays'].value_counts()[:5]


# In[ ]:


sns.violinplot(x='y', y='pdays', data=train)


# In[ ]:


pd.crosstab(train['previous'], y).plot(kind='bar')


# ## Explore feature interaction

# In[ ]:


marital_job_status = pd.pivot_table(
    data=train,
    index='marital',
    columns='job',
    values='y',
    aggfunc='mean',
    fill_value=0.123,
)
marital_job_status.style.background_gradient(cmap='BuGn', axis=1)


# In[ ]:


marital_education_status = pd.pivot_table(
    data=train,
    index='marital',
    columns='education',
    values='y',
    aggfunc='mean',
    fill_value=0.123,
)
marital_education_status.style.background_gradient(cmap='BuGn', axis=1)


# In[ ]:


job_education_status = pd.pivot_table(
    data=train,
    index='job',
    columns='education',
    values='y',
    aggfunc='mean',
    fill_value=0.123,
)
job_education_status.style.background_gradient(cmap='BuGn', axis=1)


# ## Build model

# In[ ]:


from scipy  import sparse

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve


from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from catboost import CatBoostClassifier

from sklearn.manifold import TSNE

import shap
shap.initjs()


# ### Some helpful functions

# #### Transform data

# In[ ]:


def label_encode(
    X: pd.DataFrame, 
    encoders: Optional[Dict[str, LabelEncoder]] = None,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:

    X = X.copy()
    encoders = encoders or {}
    for col in get_cat_features(X):
        if col not in encoders:
            encoder = LabelEncoder().fit(X[col])
            encoders[col] = encoder
        else:
            encoder = encoders[col]
        X[col] = encoder.transform(X[col])
    return X, encoders


def one_hot_encode(
    X: pd.DataFrame, 
    encoders: Optional[Dict[str, OneHotEncoder]] = None,
) -> Tuple[sparse.csr_matrix, Dict[str, OneHotEncoder]]:
    cat_features = get_cat_features(X)
    feature_matrices = []
    encoders = encoders or {}
    for col in X.columns:
        if col in cat_features:
            if col not in encoders:
                encoder = OneHotEncoder().fit(X[[col]])
                encoders[col] = encoder
            else:
                encoder = encoders[col]
            feature_matrix = encoder.transform(X[[col]])
        else:
            feature_matrix = sparse.csr_matrix((
                X[col].values, 
                (
                    np.arange(X.shape[0], dtype=int), 
                    np.zeros(X.shape[0], dtype=int),
                ),
            ))
        feature_matrices.append(feature_matrix)
    features = sparse.hstack(feature_matrices, format='csr')
    return features, encoders 


def scale(
    X: sparse.csr_matrix, 
    scaler: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, StandardScaler]:
    scaler = scaler or StandardScaler()
    X_scaled = scaler.fit_transform(X.toarray())
    return X_scaled, scaler


# #### Represent results

# In[ ]:


def calc_metrics(
    y_true: Union[np.ndarray, pd.Series], 
    pred_proba: Union[np.ndarray, pd.Series], 
    threshold: float = 0.5,
) -> Dict[str, float]:
    res = {}
    pred = np.zeros_like(pred_proba)
    pred[pred_proba > threshold] = 1
    res['accuracy'] = accuracy_score(y_true, pred)
    res['auc'] = roc_auc_score(y_true, pred_proba)
    res['f1'] = f1_score(y_true, pred)
    res['precision'] = precision_score(y_true, pred)
    res['recall'] = recall_score(y_true, pred)
    return res


def get_feature_importances(clf, columns):
    return pd.DataFrame({
        'column': columns,
        'importance': clf.feature_importances_,
    }).sort_values('importance', ascending=False)


def represent_cv_results(gscv):
    cv_results = pd.DataFrame(gscv.cv_results_)
    res = cv_results[['params', 'mean_fit_time', 'mean_train_score', 'mean_test_score']]         .sort_values('mean_test_score', ascending=False)
    return res


# ## Extract features

# In[ ]:


def get_zodiac(birth_date: pd.Series) -> pd.Series:
    zodiacs = pd.read_csv(pjoin(INPUT_PATH, 'zodiac', 'zodiac.csv'), dtype=str)
    birth_day = birth_date.dt.day
    birth_month = birth_date.dt.month
    result = pd.Series(index=birth_date.index)
    
    for _, zodiac in zodiacs.iterrows():
        start_month = int(zodiac['start'][-2:])
        start_day = int(zodiac['start'][:2])
        end_month = int(zodiac['end'][-2:])
        end_day = int(zodiac['end'][:2])
        
        is_it = (
            (birth_month == start_month) & (birth_day >= start_day) |
            (birth_month == end_month) & (birth_day <= end_day)
        )
        result[is_it] = zodiac['name']
        
    return result


# In[ ]:


def extract_features(df: pd.DataFrame):
    X = df.copy()
    X.drop(columns=['birth_date', 'contact_date'], inplace=True)
    
    X['age_when_contact'] = (df['contact_date'] - df['birth_date']).dt.days / 365.25
    X['zodiac'] = get_zodiac(df['birth_date'])
    
    # Maybe some calls were done near birth day and customer was more active
    days_between_bd_and_cd = (df['contact_date'] - df['birth_date']).dt.days
    X['days_between_bd_and_cd'] =  np.minimum(
        days_between_bd_and_cd % 365.25, -days_between_bd_and_cd % 365.25
    )
    
    X['contact_day_of_week'] = df['contact_date'].dt.dayofweek
    
    return X


# In[ ]:


X_train = extract_features(train.drop(columns='y'))
X_train_le, label_encoders = label_encode(X_train)
X_train_ohe, one_hot_encoders = one_hot_encode(X_train)
X_train_ohe_scaled, scaler = scale(X_train_ohe)

y_train = train['y']


# In[ ]:


X_test = extract_features(test)
X_test_le, _ = label_encode(X_test, label_encoders)
X_test_ohe, _ = one_hot_encode(X_test, one_hot_encoders)
X_test_ohe_scaled, _ = scale(X_test_ohe, scaler)


# ## Try to find some clusters in data

# In[ ]:


get_ipython().run_cell_magic('time', '', 'tsne = TSNE(random_state=1)\ntsne_representation = tsne.fit_transform(X_train_ohe_scaled)')


# In[ ]:


plt.scatter(tsne_representation[:, 0], tsne_representation[:, 1], c=y.map({0: 'blue', 1: 'orange'}), s=20);


# In[ ]:


cat_features = get_cat_features(train)

set_figsize(6, len(cat_features) * 5)
fig, axes = plt.subplots(len(cat_features))

for ax, feature in zip(axes, cat_features):
    for val in train[feature].unique():
        mask = train[feature] == val
        ax.scatter(tsne_representation[mask, 0], tsne_representation[mask, 1], label=val, s=20)
    ax.legend()
    ax.set_title(feature)
    
set_figsize(6, 4)


# In[ ]:


housing_unknown_mask = train['housing'] == 'unknown'
loan_unknown_mask = train['loan'] == 'unknown'
loan_and_housing_unknown_mask = housing_unknown_mask & loan_unknown_mask
print(housing_unknown_mask.sum())
print(loan_unknown_mask.sum())
print(loan_and_housing_unknown_mask.sum())


# ## Try different models

# ### Logistic regression

# In[ ]:


lr = LogisticRegression(solver='liblinear', penalty='l2', random_state=1)
gscv_lr = GridSearchCV(
    estimator=lr,
    param_grid={'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10]},
    scoring='roc_auc',
    n_jobs=1,
    cv=StratifiedKFold(n_splits=5, random_state=1),
    refit=True,
    return_train_score=True,
    verbose=True,
)

gscv_lr.fit(X_train_ohe, y_train);


# In[ ]:


represent_cv_results(gscv_lr)


# ### Naive Bayes

# In[ ]:


nb = GaussianNB()
gscv_nb = GridSearchCV(
    estimator=nb,
    param_grid={},
    scoring='roc_auc',
    n_jobs=1,
    cv=StratifiedKFold(n_splits=5, random_state=1),
    refit=True,
    return_train_score=True,
    verbose=True,
)

gscv_nb.fit(X_train_le, y_train);


# In[ ]:


represent_cv_results(gscv_nb)


# ### Neural network

# In[ ]:


nn = MLPClassifier(random_state=1)
gscv_nn = GridSearchCV(
    estimator=nn,
    param_grid={
        'alpha': [1, 0.1],
        'hidden_layer_sizes': [(100, 100), (100,)],
    },
    scoring='roc_auc',
    n_jobs=2,
    cv=StratifiedKFold(n_splits=3, random_state=1),
    refit=True,
    return_train_score=True,
    verbose=True,
)

gscv_nn.fit(X_train_ohe, y_train);


# In[ ]:


represent_cv_results(gscv_nn)


# ### Decision tree

# In[ ]:


dt = DecisionTreeClassifier(max_leaf_nodes=100, random_state=1)
gscv_dt = GridSearchCV(
    estimator=dt,
    param_grid={'max_depth': [4, 5, 6], 'min_samples_leaf': [50, 100, 200]},
    scoring='roc_auc',
    n_jobs=1,
    cv=StratifiedKFold(n_splits=5, random_state=1),
    refit=True,
    return_train_score=True,
    verbose=True,
)

gscv_dt.fit(X_train_le, y_train);


# In[ ]:


represent_cv_results(gscv_dt)


# ### Random forest

# In[ ]:


rf = RandomForestClassifier(random_state=1)
gscv_rf = GridSearchCV(
    estimator=rf,
    param_grid={'max_depth': [3, 4], 'n_estimators': [50, 100, 200]},
    scoring='roc_auc',
    n_jobs=2,
    cv=StratifiedKFold(n_splits=2, random_state=1),
    refit=True,
    return_train_score=True,
    verbose=True,
)


gscv_rf.fit(X_train_le, y_train);


# In[ ]:


represent_cv_results(gscv_rf)


# In[ ]:


rf_clf = gscv_rf.best_estimator_


# In[ ]:


get_feature_importances(rf_clf, X_train_le.columns)


# #### Get brief model explanation by shap

# In[ ]:


rf_explainer = shap.TreeExplainer(rf_clf)
rf_shap_values = rf_explainer.shap_values(X_train_le)


# In[ ]:


shap.summary_plot(rf_shap_values, X_train_le, plot_type='bar')


# **Note:** Feature importances given by model and *shap* are different

# ### Gradient boosting

# **Note:** *CatBoostClassifier* can work with not encoded features, but we'll use label encoded ones to explain model with *shap* later

# In[ ]:


cb = CatBoostClassifier(
    cat_features=get_cat_features(X_train_le),
    eval_metric='AUC',
    random_seed=2,
    nan_mode='Forbidden',
    task_type='CPU',
    verbose=False,
)


gscv_cb = GridSearchCV(
    estimator=cb,
    param_grid={
        'n_estimators': [50, 100], 
        'max_depth': [3, 4],
        
    },
    scoring='roc_auc',
    n_jobs=1,
    cv=StratifiedKFold(n_splits=3, random_state=1),
    refit=True,
    return_train_score=True,
    verbose=True,

)

gscv_cb.fit(X_train_le, y_train);


# In[ ]:


represent_cv_results(gscv_cb)


# In[ ]:


cb_clf = gscv_cb.best_estimator_
get_feature_importances(cb_clf, X_train_le.columns)


# #### Get model explanation by shap

# In[ ]:


cb_explainer = shap.TreeExplainer(cb_clf)
cb_shap_values = cb_explainer.shap_values(X_train_le)


# In[ ]:


# See at random row in data

i = 123
shap.force_plot(
    cb_explainer.expected_value, 
    cb_shap_values[i, :], 
    X_train_le.iloc[i, :],
)


# In[ ]:


# See at all variables at moment

sample = np.random.choice(np.arange(len(X_train_le)), size=100, replace=False)
shap.force_plot(
    cb_explainer.expected_value, 
    cb_shap_values[sample, :], 
    X_train_le.iloc[sample, :],
)


# In[ ]:


# See feature impacts

shap.summary_plot(cb_shap_values, X_train_le)


# In[ ]:


# See mean feature importances

shap.summary_plot(cb_shap_values, X_train_le, plot_type="bar")


# # Make predictions

# Best result gives *catboost* model 

# In[ ]:


final_clf = cb_clf


# In[ ]:


pred_test = final_clf.predict_proba(X_test_le)[:, 1]


# In[ ]:


res = pd.DataFrame({'y': pred_test, 'id': test.index})
res.to_csv('res_baseline.csv', index=False)

