#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import libraries and set desired options
import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack
# !pip install eli5
import eli5
import time
from tqdm import tqdm_notebook
from scipy.sparse import csr_matrix
from itertools import combinations
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import SelectFromModel
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import seaborn as sns
from IPython.display import display_html
from collections import Counter


# In[ ]:


PATH_TO_DATA = '../input'
SEED = 42


# ## Feature engineering

# In[ ]:


PATH_TO_TRAIN = os.path.join(PATH_TO_DATA, 'train_sessions.csv')
PATH_TO_TEST = os.path.join(PATH_TO_DATA, 'test_sessions.csv')
PATH_TO_DICT = os.path.join(PATH_TO_DATA, 'site_dic.pkl')


# In[ ]:


times = ['time%s' % i for i in range(1, 11)]
df = pd.read_csv(PATH_TO_TRAIN, index_col='session_id', parse_dates=times)
df = df.sort_values(by='time1').reset_index(drop=True)
test_df = pd.read_csv(PATH_TO_TEST, index_col='session_id', parse_dates=times)


# Select active days only to speed up the search process.

# In[ ]:


yyyymmdd = 10000 * df.time1.dt.year + 100 * df.time1.dt.month + df.time1.dt.day
full_df = df.loc[yyyymmdd.isin(yyyymmdd[df.target == 1].unique())]


# Split the dataset into training and holdout sets. We will perform CV on the training set, while at the end we will test our results on the holdout set. This way, we may see signs of overfitting locally and establish a good correlation between local score and LB. 

# In[ ]:


holdout_idx = int(len(full_df)*0.9)
holdout_idx


# In[ ]:


y_full = full_df['target'].astype('int')
y_train = y_full.iloc[:holdout_idx]
y_holdout = y_full.iloc[holdout_idx:]

print(y_train.shape, y_holdout.shape)


# Prepare URLs for vectorization.

# In[ ]:


# read site -> id mapping provided by competition organizers 
with open(PATH_TO_DICT, 'rb') as f:
    site2id = pickle.load(f)
# create an inverse id _> site mapping
id2site = {v:k for (k, v) in site2id.items()}
# we treat site with id 0 as "unknown"
id2site[0] = 'unknown'

# Map site integers to URLs
sites = ['site%s' % i for i in range(1, 11)]
# Sites used for creating test predictions
X_full_sites = full_df[sites].fillna(0).astype('int').applymap(lambda x: id2site[x])
X_test_sites = test_df[sites].fillna(0).astype('int').applymap(lambda x: id2site[x])
# Subset of sites used for feature search
X_train_sites = X_full_sites.iloc[:holdout_idx]
X_holdout_sites = X_full_sites.iloc[holdout_idx:]

print(X_train_sites.shape, X_holdout_sites.shape, X_test_sites.shape)


# In[ ]:


def vectorize_sites(train_sites, test_sites, vectorizer_params):
    # Join each row by space and apply an TF-IDF vectorizer to each row
    site_vectorizer = TfidfVectorizer(**vectorizer_params)
    
    train_sessions = train_sites.apply(lambda row: ' '.join(row), axis=1).tolist()
    train_sparse = site_vectorizer.fit_transform(train_sessions)
    
    test_sessions = test_sites.apply(lambda row: ' '.join(row), axis=1).tolist()
    test_sparse = site_vectorizer.transform(test_sessions)
    
    return train_sparse, test_sparse, site_vectorizer


# In[ ]:


get_ipython().run_cell_magic('time', '', "site_vectorizer_params = {\n    'ngram_range': (1, 1), \n    'max_features': 50000, \n    'tokenizer': lambda s: s.split()\n}\nX_train_sites_sparse, X_holdout_sites_sparse, site_vectorizer = vectorize_sites(X_train_sites, X_holdout_sites, site_vectorizer_params)\nX_full_sites_sparse, X_test_sites_sparse, _ = vectorize_sites(X_full_sites, X_test_sites, site_vectorizer_params)\n\nprint(X_train_sites_sparse.shape, X_holdout_sites_sparse.shape, X_test_sites_sparse.shape)")


# We will also vectorize site URL parts such as domains.

# In[ ]:


X_full_parts = X_full_sites.applymap(lambda x: ' '.join(x.split('.')))
X_train_parts = X_train_sites.applymap(lambda x: ' '.join(x.split('.')))
X_holdout_parts = X_holdout_sites.applymap(lambda x: ' '.join(x.split('.')))
X_test_parts = X_test_sites.applymap(lambda x: ' '.join(x.split('.')))


# In[ ]:


get_ipython().run_cell_magic('time', '', "part_vectorizer_params = {\n    'ngram_range': (1, 1), \n    'max_features': 50000, \n    'tokenizer': lambda s: s.split()\n}\nX_train_parts_sparse, X_holdout_parts_sparse, part_vectorizer = vectorize_sites(X_train_parts, X_holdout_parts, part_vectorizer_params)\nX_full_parts_sparse, X_test_parts_sparse, _ = vectorize_sites(X_full_parts, X_test_parts, part_vectorizer_params)\n\nprint(X_train_parts_sparse.shape, X_holdout_parts_sparse.shape, X_test_parts_sparse.shape)")


# Stack both sparse matrices (URLs and URL parts) into one big sparse matrix.

# In[ ]:


def stack_features(*matrices):
    # Stack matrices into a single sparse matrix
    return csr_matrix(hstack(matrices))


# In[ ]:


X_full_sparse = stack_features(X_full_sites_sparse, X_full_parts_sparse)
X_train_sparse = stack_features(X_train_sites_sparse, X_train_parts_sparse)
X_holdout_sparse = stack_features(X_holdout_sites_sparse, X_holdout_parts_sparse)
X_test_sparse = stack_features(X_test_sites_sparse, X_test_parts_sparse)


# Split temporal data.

# In[ ]:


X_full_times = full_df[times]
X_train_times = X_full_times.iloc[:holdout_idx]
X_holdout_times = X_full_times.iloc[holdout_idx:]
X_test_times = test_df[times]

print(X_train_times.shape, X_holdout_times.shape, X_test_times.shape)


# Now let's add some additional features. The more you have here the better, we will filter some of them out later. You can beat **both** baselines just by experimenting with this function.

# In[ ]:


def create_features(sites, times):
    # Perform your feature engineering here
    # For your features, check the following assumptions:
    #   @ Linear relationship
    #   @ Few outliers
    #   @ Multivariate normality
    #   @ No or little multicollinearity
    #   @ No auto-correlation
    #   @ Homoscedasticity
    features = pd.DataFrame(index=sites.index)
    
    # Raw features here
    # For cyclic features try to make them harmonic by mapping onto a circle and using two coordinates
    features['year'] = year = times['time1'].apply(lambda t: t.year).astype(int)
    features['yyyymm'] = times['time1'].apply(lambda t: 100 * t.year + t.month).astype('float64')
    features['yyyymmdd'] = times['time1'].apply(lambda t: 10000 * t.year + 100 * t.month + t.day).astype('float64')
    features['month'] = times['time1'].apply(lambda t: t.month).astype('float64')
    features['day'] = times['time1'].apply(lambda t: t.day).astype('float64')
    features['dayofyear'] = times['time1'].apply(lambda t: t.dayofyear).astype('float64')
    features['dayofweek'] = dayofweek = times['time1'].apply(lambda t: t.dayofweek).astype('float64')
    features['hour'] = hour = times['time1'].apply(lambda ts: ts.hour).astype('float64')
    features['minute'] = times['time1'].apply(lambda ts: ts.minute).astype('float64')
    features['seconds'] = ((times.max(axis=1) - times.min(axis=1)) / np.timedelta64(1, 's')).astype('float64')
    
    # Binary features
    features['is_sunday'] = (dayofweek == 6).astype(int)
    features['is_morning'] = ((hour >= 7) & (hour <= 11)).astype(int)
    features['is_day'] = ((hour >= 12) & (hour <= 18)).astype(int)
    features['is_evening'] = ((hour >= 19) & (hour <= 23)).astype(int)
    features['is_night'] = ((hour >= 0) & (hour <= 6)).astype(int)
    
    # One-hot encoded features here
    features = features.join(pd.get_dummies(features[['year']], columns=['year'], prefix='is_year'))
    features['year'] = features['year'].astype('float64')
    
    return features


# In[ ]:


get_ipython().run_cell_magic('time', '', 'X_full_feat = create_features(X_full_sites, X_full_times)\nX_train_feat = create_features(X_train_sites, X_train_times)\nX_holdout_feat = create_features(X_holdout_sites, X_holdout_times)\nX_test_feat = create_features(X_test_sites, X_test_times)\n\nprint(X_train_feat.shape, X_holdout_feat.shape, X_test_feat.shape)')


# In[ ]:


# Scale float columns
columns_to_scale = X_full_feat.select_dtypes(include='float64').columns

scaler = MinMaxScaler()
X_train_feat[columns_to_scale] = scaler.fit_transform(X_train_feat[columns_to_scale])
X_holdout_feat[columns_to_scale] = scaler.transform(X_holdout_feat[columns_to_scale])
X_full_feat[columns_to_scale] = scaler.fit_transform(X_full_feat[columns_to_scale])
X_test_feat[columns_to_scale] = scaler.transform(X_test_feat[columns_to_scale])


# In[ ]:


X_train_feat.describe().T['max']


# The training set may have a different number of columns than the holdout set - align them.

# In[ ]:


if set(X_train_feat.columns) != set(X_holdout_feat.columns):
    X_train_feat, X_holdout_feat = X_train_feat.align(X_holdout_feat, join='outer', axis=1, fill_value=0)
if set(X_full_feat.columns) != set(X_test_feat.columns):
    X_full_feat, X_test_feat = X_full_feat.align(X_test_feat, join='outer', axis=1, fill_value=0)


# Correct cross-validation scheme for temporal data. 
# 
# We will perform CV on the training data first, then perform validation on the holdout data to get better estimation of public LB score.

# In[ ]:


time_split = TimeSeriesSplit(n_splits=5)

print([(el[0].shape, el[1].shape) for el in time_split.split(y_train)])


# ## Feature selection

# Correlation between features and target gives first hints.

# In[ ]:


print(X_train_feat.corrwith(y_train).abs().sort_values(ascending=False)[:20])


#  You may select the best features based on univariate statistical tests.

# In[ ]:


selector = SelectKBest(f_classif, k=5)
selector.fit(X_train_feat, y_train)

print(X_train_feat.columns[selector.get_support(indices=True)])


# You may also try to select features recursively based on the feature weight coefficients (e.g., linear models) or feature importance (tree-based algorithms).

# In[ ]:


logit = LogisticRegression(C=1, random_state=SEED, solver='liblinear')


# In[ ]:


selector = SelectFromModel(estimator=logit)
selector.fit(X_train_feat, y_train)

print(X_train_feat.columns[selector.get_support(indices=True)])


# Let's try something different: use SFSs to eliminate (or add) features based on a user-defined classifier/regression performance metric.
# 
# http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/#sequential-feature-selector

# In[ ]:


get_ipython().run_cell_magic('time', '', "selectors = {}\nmodes = ['SFS', 'SBS', 'SFFS', 'SBFS']\nfor i in tqdm_notebook(range(len(modes))):\n    mode = modes[i]\n    forward = i in (0, 2)\n    floating = i in (2, 3)\n    selector = SequentialFeatureSelector(logit, k_features='best', forward=forward, floating=floating, \n                                         verbose=False, scoring='roc_auc', cv=time_split, n_jobs=-1)\n    selector.fit(X_train_feat, y_train, custom_feature_names=X_train_feat.columns)\n    selectors[mode] = selector")


# In[ ]:


feature_scores = {}
for selector in selectors.values():
    for subset in selector.subsets_.values():
        feature_names, cv_scores = subset['feature_names'], subset['cv_scores']
        feature_scores[tuple(set(feature_names))] = cv_scores


# In[ ]:


Counter([feature for features in feature_scores.keys() for feature in features])


# In[ ]:


def assess_feature_scores(feature_scores):
    # Format scores
    df = pd.DataFrame(feature_scores).T
    split_cols = ['split_%s'%str(c+1) for c in df.columns]
    df.columns = split_cols
    df['mean'] = df[split_cols].mean(axis=1)
    df['std'] = df[split_cols].std(axis=1)
    df = df.sort_values(by='mean', ascending=False)
    return df


# In[ ]:


assess_feature_scores(feature_scores)


# To avoid overfitting, we want not only to select features based on the highest CV score but also on *the stability of the model*. We see that the model's performance inversely correlates with the number of features, which means that the most of them are just noise.

# In[ ]:


selected_features = ['is_evening', 'is_morning', 'seconds', 'hour']


# After selecting the features of interest, let's remove them iteratively to get the most stable model.

# In[ ]:


def cross_validate(model, X_train, y_train, cv=time_split, scoring='roc_auc'):
    # Cross validate on training set
    return cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

def neg_cv_std(estimator, X, y):
    return -np.std(cross_validate(estimator, X, y))


# In[ ]:


selector = SequentialFeatureSelector(logit, k_features='best', forward=False, floating=True, 
                                     verbose=1, scoring=neg_cv_std, cv=None, n_jobs=-1)
selector.fit(X_train_feat[selected_features], y_train, custom_feature_names=selected_features)


# In[ ]:


selector.k_feature_names_, selector.k_score_


# In[ ]:


selected_features = ['is_evening', 'is_morning', 'seconds']


# ## Evaluation

# It's a good practice to visualize how the model behaves depending on the regularization parameter *C* and the training size.
# 
# Good features lead to stable, narrow curves.

# In[ ]:


X_train = stack_features(X_train_sparse, X_train_feat[selected_features].values)
X_holdout = stack_features(X_holdout_sparse, X_holdout_feat[selected_features].values)

print(X_train.shape, X_holdout.shape)


# In[ ]:


def plot_with_err(x, data, **kwargs):
    mu, std = data.mean(1), data.std(1)
    lines = plt.plot(x, mu, '-', **kwargs)
    plt.fill_between(x, mu - std, mu + std, edgecolor='none',
                     facecolor=lines[0].get_color(), alpha=0.2)


# In[ ]:


def create_validation_curve(X, y):
    Cs = np.logspace(-2, 1, 20)
    time_cv = TimeSeriesSplit(n_splits=5)
    logit = LogisticRegression(random_state=SEED, solver='liblinear')
    train_scores, valid_scores = validation_curve(logit, X, y, 'C', Cs, cv=time_cv, scoring='roc_auc', n_jobs=-1, verbose=1)
    
    plot_with_err(Cs, train_scores, label='training scores')
    plot_with_err(Cs, valid_scores, label='validation scores')
    plt.xlabel('C'); plt.ylabel('ROC AUC')
    plt.legend()
    plt.show()


# In[ ]:


create_validation_curve(X_train, y_train)


# In[ ]:


def create_learning_curve(X, y, C=1):
    train_sizes = np.linspace(0.3, 1, 10)
    time_cv = TimeSeriesSplit(n_splits=5)
    logit = LogisticRegression(C=C, random_state=SEED, solver='liblinear')
    train_sizes, train_scores, valid_scores = learning_curve(
        logit, X, y, train_sizes=train_sizes, cv=time_cv, scoring='roc_auc', n_jobs=-1, verbose=1)
    
    plot_with_err(train_sizes, train_scores, label='training scores')
    plot_with_err(train_sizes, valid_scores, label='validation scores')
    plt.xlabel('Training Set Size'); plt.ylabel('AUC')
    plt.legend()
    plt.show()


# In[ ]:


create_learning_curve(X_train, y_train, C=1)


# Now let's check the weights of the new features to be somewhat intuitively correct.
# 
# If you added some new features and your weights look totally different than for the baseline, this is usually a bad sign.

# In[ ]:


logit.fit(X_train, y_train)


# In[ ]:


def explain_model(model, site_feature_names, new_feature_names=None, top_n_features_to_show=10):
    if new_feature_names is not None:
        all_feature_names = site_feature_names + new_feature_names 
    else: 
        all_feature_names = site_feature_names
    display_html(eli5.show_weights(estimator=model, feature_names=all_feature_names, top=top_n_features_to_show))
    if new_feature_names:
        print('New feature weights:')
        print(pd.Series(model.coef_.flatten()[-len(new_feature_names):], index=new_feature_names).sort_values(ascending=False))


# In[ ]:


site_feature_names = site_vectorizer.get_feature_names() + part_vectorizer.get_feature_names()

explain_model(logit, site_feature_names, new_feature_names=selected_features)


# Finally, if you are confident enough in your features, run the model on the holdout set to get an estimation of the public LB score.

# In[ ]:


def compute_score(model, X, y):
    # Validate on validation set
    y_pred = model.predict_proba(X)[:, 1]
    return roc_auc_score(y, y_pred)


# In[ ]:


compute_score(logit, X_holdout, y_holdout)


# Do not run it very often to not overfit though.

# ## Hyperparameter tuning

# Try to improve the score a bit.

# In[ ]:


# Prepare data for testing
X_full = stack_features(X_full_sparse, X_full_feat[selected_features].values)
X_test = stack_features(X_test_sparse, X_test_feat[selected_features].values)

print(X_full.shape, X_test.shape)


# In[ ]:


param_grid = {
    'C': np.logspace(-2, 1, 20)
}


# In[ ]:


logit_gs = GridSearchCV(estimator=logit, param_grid=param_grid, scoring='roc_auc', 
                        iid=False, n_jobs=-1, cv=time_split, verbose=1, return_train_score=False)


# In[ ]:


logit_gs.fit(X_full, y_full);


# In[ ]:


logit_gs.best_score_, logit_gs.best_params_


# In[ ]:


print(cross_validate(logit, X_full, y_full))
print(cross_validate(logit_gs.best_estimator_, X_full, y_full))


# In[ ]:


# Compare performance against regularization at different splits
logit_gs_scores = pd.DataFrame()
for i in range(5):
    column = 'split%d_test_score'%i
    logit_gs_scores[column] = logit_gs.cv_results_[column]
ax = logit_gs_scores.plot(figsize=(14, 6))
ax.set_xticks(range(len(param_grid['C'])))
ax.set_xticklabels(['%.2f'%c for c in param_grid['C']])
ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5));


# In[ ]:


logit_gs.best_estimator_.fit(X_full, y_full)


# ## Submission

# In[ ]:


# A helper function for writing predictions to a file
def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)
        
def predict_and_submit(model, X_test, file_name='submission.csv'):
    test_pred = model.predict_proba(X_test)[:, 1]
    write_to_submission_file(test_pred, file_name)


# In[ ]:


predict_and_submit(logit_gs.best_estimator_, X_test)


# In[ ]:




