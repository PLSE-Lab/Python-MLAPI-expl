#!/usr/bin/env python
# coding: utf-8

# **1&nbsp;&nbsp;[Introduction](#1)**  
# &nbsp;&nbsp;&nbsp;&nbsp;1.1&nbsp;&nbsp;[Problem definition](#1.1)  
# &nbsp;&nbsp;&nbsp;&nbsp;1.2&nbsp;&nbsp;[Dataset](#1.2)  
# **2&nbsp;&nbsp;[Environment](#2)**  
# &nbsp;&nbsp;&nbsp;&nbsp;2.1&nbsp;&nbsp;[Libraries](#2.1)  
# &nbsp;&nbsp;&nbsp;&nbsp;2.2&nbsp;&nbsp;[Data](#2.2)  
# &nbsp;&nbsp;&nbsp;&nbsp;2.3&nbsp;&nbsp;[Functions](#2.3)  
# **3&nbsp;&nbsp;[Wrangling](#3)**  
# &nbsp;&nbsp;&nbsp;&nbsp;3.1&nbsp;&nbsp;[Relevant features](#3.1)  
# &nbsp;&nbsp;&nbsp;&nbsp;3.2&nbsp;&nbsp;[Feature datatypes](#3.2)  
# &nbsp;&nbsp;&nbsp;&nbsp;3.3&nbsp;&nbsp;[Data export](#3.3)  
# **4&nbsp;&nbsp;[Exploration](#4)**  
# &nbsp;&nbsp;&nbsp;&nbsp;4.1&nbsp;&nbsp;[Univariate analysis](#4.1)  
# &nbsp;&nbsp;&nbsp;&nbsp;4.2&nbsp;&nbsp;[Bivariate analysis](#4.2)  
# **5&nbsp;&nbsp;[Engineering](#5)**  
# &nbsp;&nbsp;&nbsp;&nbsp;5.1&nbsp;&nbsp;[Sample analysis](#5.1)  
# &nbsp;&nbsp;&nbsp;&nbsp;5.2&nbsp;&nbsp;[Feature representation](#5.2)  
# &nbsp;&nbsp;&nbsp;&nbsp;5.3&nbsp;&nbsp;[Feature interactions](#5.3)  
# &nbsp;&nbsp;&nbsp;&nbsp;5.4&nbsp;&nbsp;[Feature transformations](#5.4)  
# &nbsp;&nbsp;&nbsp;&nbsp;5.5&nbsp;&nbsp;[Feature selection](#5.5)  
# &nbsp;&nbsp;&nbsp;&nbsp;5.6&nbsp;&nbsp;[Data export](#5.6)  
# **6&nbsp;&nbsp;[Modelling](#6)**  
# &nbsp;&nbsp;&nbsp;&nbsp;6.1&nbsp;&nbsp;[Baseline models](#6.1)  
# &nbsp;&nbsp;&nbsp;&nbsp;6.2&nbsp;&nbsp;[Ensemble models](#6.2)  
# **7&nbsp;&nbsp;[Evaluation](#7)**  
# &nbsp;&nbsp;&nbsp;&nbsp;7.1&nbsp;&nbsp;[Classifier performance](#7.1)  
# &nbsp;&nbsp;&nbsp;&nbsp;7.2&nbsp;&nbsp;[Feature importance](#7.2)  
# &nbsp;&nbsp;&nbsp;&nbsp;7.3&nbsp;&nbsp;[Learning rate](#7.3)  
# **8&nbsp;&nbsp;[Conclusion](#8)**

# ## 1 Introduction<a id="1"></a>

# ### 1.1 Problem definition<a id="1.1"></a>

# - This notebook will investigate whether it is possible to **predict if a partner will match with their date** based on **dating preferences**, **attribute ratings** and **background information**.
# - The problem can be framed as a supervised, binary classification problem where the model predicts if a partner has accepted or rejected their date.

# ### 1.2 Dataset<a id="1.2"></a>

# - A study conducted by Columbia University explored gender differences in dating preferences.
# - Participants attended a dating event where they had a 4-minute date with every other participant of the opposite sex who attended the same event.
# - The participants decided to accept or reject their partner. If both the participant and partner matched, they received each other's contact information.
# - Participants rated their partners on six personal attributes: attractiveness, sincerity, intelligence, fun, ambition and shared interests.
# - Before and after the event, participants rated their preferences in the six attributes and gave themselves ratings.
# - Other information was collected about the participants' background and preferences.

# ## 2 Environment<a id="2"></a>

# ### 2.1 Libraries<a id="2.1"></a>

# Load libraries into notebook

# In[ ]:


import numpy as np              # arrays
import pandas as pd             # dataframes
import matplotlib.pyplot as plt # graphs
import seaborn as sns           # visualisations
from scipy import stats         # statistics


# In[ ]:


from sklearn.experimental import enable_iterative_imputer # enable experimental imputer
from sklearn.impute import IterativeImputer               # sample imputation
from sklearn import preprocessing                         # encoders, transformations
from sklearn.model_selection import cross_validate        # cross-validation, model evaluation
from sklearn.model_selection import GridSearchCV          # hyper-parameter tuning
from sklearn.linear_model import LogisticRegression       # logistic regression model
from sklearn.svm import SVC                               # support vector machine model
from sklearn.neighbors import KNeighborsClassifier        # k-nearest neighbours model
from sklearn.ensemble import GradientBoostingClassifier   # gradient boosting model
from sklearn.ensemble import VotingClassifier             # voting ensemble model
from sklearn.ensemble import StackingClassifier           # stacking ensemble model


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ### 2.2 Data<a id="2.2"></a>

# Import raw dataset into a dataframe

# In[ ]:


data_raw = pd.read_csv(
    filepath_or_buffer='../input/speed-dating-experiment/Speed Dating Data.csv',
    engine='python'
)


# Get number of rows and columns of raw dataframe

# In[ ]:


data_raw.shape


# Get memory usage of raw dataframe

# In[ ]:


data_raw.memory_usage().sum()


# ### 2.3 Functions<a id="2.3"></a>

# Define a function to plot distribution functions

# In[ ]:


def plot_distribution(data, bins, title, xlabel, ylabel):
    ax = sns.distplot(
        data,
        bins=bins,
        hist_kws={
            "linewidth": 1,
            'edgecolor': 'black',
            'alpha': 1.0
            },
        kde=False
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel);


# Define a function to plot relationship between two features

# In[ ]:


def plot_relationship(x, y, title, xlabel, ylabel):
    ax = sns.barplot(
        x=x,
        y=y,
        orient='h'
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel);


# Define a function to print a feature's mean, standard deviation, skewness and kurtosis

# In[ ]:


def print_moments(title, feature):
    print(title)
    print('Mean: '+'{:>18.2f}'.format(feature.mean()))
    print('Standard deviation: '+'{:.2f}'.format(feature.std()))
    print('Skewness: '+'{:>14.2f}'.format(feature.skew()))
    print('Kurtosis: '+'{:>14.2f}'.format(feature.kurtosis()))


# ## 3 Wrangling<a id="3"></a>

# ### 3.1 Relevant features<a id="3.1"></a>

# Identify relevant features and associated datatypes

# In[ ]:


relevant_features = [
    ['iid', 'int16'],
    ['gender', 'bool'],
    ['wave', 'int16'],
    ['position', 'int16'],
    ['order', 'int16'],
    ['pid', 'int16'],
    ['age_o', 'int16'],
    ['race_o', 'category'],
    ['pf_o_att', 'int16'],
    ['pf_o_sin', 'int16'],
    ['pf_o_int', 'int16'],
    ['pf_o_fun', 'int16'],
    ['pf_o_amb', 'int16'],
    ['pf_o_sha', 'int16'],
    ['dec_o', 'bool'],
    ['attr_o', 'int16'],
    ['sinc_o', 'int16'],
    ['intel_o', 'int16'],
    ['fun_o', 'int16'],
    ['amb_o', 'int16'],
    ['shar_o', 'int16'],
    ['like_o', 'int16'],
    ['prob_o', 'int16'],
    ['met_o', 'bool'],
    ['age', 'int16'],
    ['field_cd', 'category'],
    ['race', 'category'],
    ['imprace', 'int16'],
    ['imprelig', 'int16'],
    ['goal', 'category'],
    ['date', 'int16'],
    ['go_out', 'int16'],
    ['career_c', 'category'],
    ['sports', 'int16'],
    ['tvsports', 'int16'],
    ['exercise', 'int16'],
    ['dining', 'int16'],
    ['museums', 'int16'],
    ['art', 'int16'],
    ['hiking', 'int16'],
    ['gaming', 'int16'],
    ['clubbing', 'int16'],
    ['reading', 'int16'],
    ['tv', 'int16'],
    ['theater', 'int16'],
    ['movies', 'int16'],
    ['concerts', 'int16'],
    ['music', 'int16'],
    ['shopping', 'int16'],
    ['yoga', 'int16'],
    ['exphappy', 'int16'],
    ['expnum', 'int16'],
    ['attr1_1', 'int16'],
    ['sinc1_1', 'int16'],
    ['intel1_1', 'int16'],
    ['fun1_1', 'int16'],
    ['amb1_1', 'int16'],
    ['shar1_1', 'int16'],
    ['attr3_1', 'int16'],
    ['sinc3_1', 'int16'],
    ['fun3_1', 'int16'],
    ['intel3_1', 'int16'],
    ['amb3_1', 'int16'],
    ['dec', 'bool'],
    ['attr', 'int16'],
    ['sinc', 'int16'],
    ['intel', 'int16'],
    ['fun', 'int16'],
    ['amb', 'int16'],
    ['shar', 'int16'],
    ['like', 'int16'],
    ['prob', 'int16'],
    ['met', 'int16'],
    ['match_es', 'int16'],
    ['satis_2', 'int16'],
    ['length', 'int16'],
    ['numdat_2', 'int16']
]


# Create new dataframe containing relevant features

# In[ ]:


data = data_raw[[feature[0] for feature in relevant_features]]


# Get number of rows and columns of new dataframe

# In[ ]:


data.shape


# Get memory usage of new dataframe

# In[ ]:


data.memory_usage().sum()


# ### 3.2 Feature datatypes<a id="3.2"></a>

# Update feature datatypes

# In[ ]:


data = data.astype({feature: datatype if all(data[feature].notna().values) else 'float32' if datatype == 'int16' else datatype for (feature, datatype) in relevant_features})


# Get memory usage of updated dataframe

# In[ ]:


data.memory_usage().sum()


# ### 3.3 Data export<a id="3.3"></a>

# Persist dataframe containing relevant features with appropriate datatypes

# In[ ]:


data.to_csv(
    path_or_buf='./data.csv',
    index=False
)


# ## 4 Exploration<a id="4"></a>

# ### 4.1 Univariate analysis<a id="4.1"></a>

# Get proportion of dates where partner matched with subject

# In[ ]:


partner_accepts = data['dec_o']
round(partner_accepts[partner_accepts == True].count()/partner_accepts.count(),3)


# Plot the distributions of subject attribute ratings from their partners

# In[ ]:


plt.figure(figsize=(16,10))
plt.tight_layout(pad=5.0)

plt.subplot(2,3,1)
plot_distribution(
    data=data['attr_o'],
    bins=np.arange(0, 10, 0.5).tolist(),
    title='Subject\'s attractiveness rating',
    xlabel='Attractiveness rating',
    ylabel='Number of subjects'
)
plt.subplot(2,3,2)
plot_distribution(
    data=data['sinc_o'],
    bins=np.arange(0, 10, 0.5).tolist(),
    title='Subject\'s sincerity rating',
    xlabel='Sincerity rating',
    ylabel='Number of subjects'
)
plt.subplot(2,3,3)
plot_distribution(
    data=data['intel_o'],
    bins=np.arange(0, 10, 0.5).tolist(),
    title='Subject\'s intelligence rating',
    xlabel='Intelligence rating',
    ylabel='Number of subjects'
)
plt.subplot(2,3,4)
plot_distribution(
    data=data['fun_o'],
    bins=np.arange(0, 10, 0.5).tolist(),
    title='Subject\'s fun rating',
    xlabel='Fun rating',
    ylabel='Number of subjects'
)
plt.subplot(2,3,5)
plot_distribution(
    data=data['amb_o'],
    bins=np.arange(0, 10, 0.5).tolist(),
    title='Subject\'s ambition rating',
    xlabel='Ambition rating',
    ylabel='Number of subjects'
)
plt.subplot(2,3,6)
plot_distribution(
    data=data['shar_o'],
    bins=np.arange(0, 10, 0.5).tolist(),
    title='Subject\'s shared interest rating',
    xlabel='Shared interest rating',
    ylabel='Number of subjects'
)


# Calculate the moments of subject attribute ratings from their partners

# In[ ]:


print_moments('Attractiveness rating', data['attr_o'])


# In[ ]:


print_moments('Sincerity rating', data['sinc_o'])


# In[ ]:


print_moments('Intelligence rating', data['intel_o'])


# In[ ]:


print_moments('Fun rating', data['fun_o'])


# In[ ]:


print_moments('Ambition rating', data['amb_o'])


# In[ ]:


print_moments('Shared interest rating', data['shar_o'])


# Get features with highest variance

# In[ ]:


data.std().sort_values(ascending=False).head(10)


# Get features with highest skew

# In[ ]:


abs(data.skew()).sort_values(ascending=False).head(10)


# ### 4.2 Bivariate analysis<a id="4.2"></a>

# Visualise correlation between selected features

# In[ ]:


features_selected = [
    'dec_o',
    'pf_o_att',
    'pf_o_sin',
    'pf_o_int',
    'pf_o_fun',
    'pf_o_amb',
    'pf_o_sha',
    'attr_o',
    'sinc_o',
    'intel_o',
    'fun_o',
    'amb_o',
    'shar_o'
]


# In[ ]:


plt.figure(figsize=(12,10))
cmap = plt.cm.RdBu
mask = np.triu(data[features_selected].astype(float).corr())
sns.heatmap(
    data[features_selected].astype(float).corr(),
    square=True,
    cmap=cmap,
    mask=mask,
    linewidths=0.1,
    vmax=1.0,
    linecolor='white'
);


# Visualise correlation between selected features for men

# In[ ]:


data_men = data[data['gender']==1]

plt.figure(figsize=(12,10))
cmap = plt.cm.RdBu
mask = np.triu(data_men[features_selected].astype(float).corr())
sns.heatmap(
    data_men[features_selected].astype(float).corr(),
    square=True,
    cmap=cmap,
    mask=mask,
    linewidths=0.1,
    vmax=1.0,
    linecolor='white'
);


# Visualise correlation between selected features for women

# In[ ]:


data_women = data[data['gender']==0]

plt.figure(figsize=(12,10))
cmap = plt.cm.RdBu
mask = np.triu(data_women[features_selected].astype(float).corr())
sns.heatmap(
    data_women[features_selected].astype(float).corr(),
    square=True,
    cmap=cmap,
    mask=mask,
    linewidths=0.1,
    vmax=1.0,
    linecolor='white'
);


# Get highest correlated feature pairs

# In[ ]:


correlations = data.corr().abs().unstack().sort_values(ascending=False).drop_duplicates()
correlations = correlations[correlations != 1]
correlations[correlations > 0.6]


# Get highest correlated features with target variable

# In[ ]:


partner_decision_correlations = correlations.loc['dec_o']
partner_decision_correlations[partner_decision_correlations > 0.1]


# ## 5 Engineering<a id="5"></a>

# ### 5.1 Sample analysis<a id="5.1"></a>

# Get proportion of dataframe with missing samples

# In[ ]:


missing_samples_proportion = data.isnull().sum()/len(data)
missing_samples_proportion.sort_values(ascending=False).head(10)


# Drop features with more than 50% missing samples

# In[ ]:


#missing_half_samples = missing_samples_proportion[missing_samples_proportion > 0.5].index.values
#data.drop(columns=missing_half_samples, inplace=True)


# Impute missing samples using iterative imputer

# In[ ]:


imputer = IterativeImputer(
    missing_values=np.nan,
    sample_posterior=True,
    n_nearest_features=5,
    min_value=0,
    max_value=100,
    random_state=0
)
imputer.fit(data)
data_imputed = np.around(imputer.transform(data))
data = pd.DataFrame(data_imputed, columns=data.columns)


# In[ ]:


data = data.astype({feature: datatype if all(data[feature].notna().values) else 'float32' if datatype == 'int16' else datatype for (feature, datatype) in relevant_features})


# ### 5.2 Feature representation<a id="5.2"></a>

# Encode nominal features using one-hot encoding

# In[ ]:


features_nominal = data.dtypes[data.dtypes == 'category'].index.values
data = pd.get_dummies(data, prefix=features_nominal)


# ### 5.3 Feature interactions<a id="5.3"></a>

# Calculate the average attribute ratings for each subject

# In[ ]:


subject_attractiveness_mean = data[['iid', 'attr_o']].groupby(['iid']).mean()['attr_o']
subject_sincerity_mean = data[['iid', 'sinc_o']].groupby(['iid']).mean()['sinc_o']
subject_intelligence_mean = data[['iid', 'intel_o']].groupby(['iid']).mean()['intel_o']
subject_fun_mean = data[['iid', 'fun_o']].groupby(['iid']).mean()['fun_o']
subject_ambition_mean = data[['iid', 'amb_o']].groupby(['iid']).mean()['amb_o']
subject_shared_interest_mean = data[['iid', 'shar_o']].groupby(['iid']).mean()['shar_o']


# Insert average attribute ratings into dataframe

# In[ ]:


data = data.merge(
    right=subject_attractiveness_mean,
    how='inner',
    on='iid'
).rename(columns={
    'attr_o_x': 'attr_o',
    'attr_o_y': 'subject_attractiveness_mean'
})
data = data.merge(
    right=subject_sincerity_mean,
    how='inner',
    on='iid'
).rename(columns={
    'sinc_o_x': 'sinc_o',
    'sinc_o_y': 'subject_sincerity_mean'
})
data = data.merge(
    right=subject_intelligence_mean,
    how='inner',
    on='iid'
).rename(columns={
    'intel_o_x': 'intel_o',
    'intel_o_y': 'subject_intelligence_mean'
})
data = data.merge(
    right=subject_fun_mean,
    how='inner',
    on='iid'
).rename(columns={
    'fun_o_x': 'fun_o',
    'fun_o_y': 'subject_fun_mean'
})
data = data.merge(
    right=subject_ambition_mean,
    how='inner',
    on='iid'
).rename(columns={
    'amb_o_x': 'amb_o',
    'amb_o_y': 'subject_ambition_mean'
})
data = data.merge(
    right=subject_shared_interest_mean,
    how='inner',
    on='iid'
).rename(columns={
    'shar_o_x': 'shar_o',
    'shar_o_y': 'subject_shared_interest_mean'
})


# Calculate difference between subject and partner's ages

# In[ ]:


data['age_difference'] = abs(data['age'] - data['age_o'])


# Calculate difference between subject's attribute ratings and partner's attributes ratings

# In[ ]:


data['attractiveness_difference'] = abs(data['attr'] - data['attr_o'])
data['sincerity_difference'] = abs(data['sinc'] - data['sinc_o'])
data['intelligence_difference'] = abs(data['intel'] - data['intel_o'])
data['fun_difference'] = abs(data['fun'] - data['fun_o'])
data['ambition_difference'] = abs(data['amb'] - data['amb_o'])
data['shared_interest_difference'] = abs(data['shar'] - data['shar_o'])


# ### 5.4 Feature transformations<a id="5.4"></a>

# Scale normal features to zero mean and unit variance

# In[ ]:


features_normal = [
    'attr_o',
    'sinc_o',
    'intel_o',
    'fun_o',
    'amb_o',
    'shar_o',
    'age_difference',
    'attractiveness_difference',
    'sincerity_difference',
    'intelligence_difference',
    'fun_difference',
    'ambition_difference',
    'shared_interest_difference'
]


# In[ ]:


data[features_normal] = data[features_normal].apply(lambda x: preprocessing.scale(x))


# ### 5.5 Feature selection<a id="5.5"></a>

# Drop irrelevant features which contain no information about the target variable 

# In[ ]:


features_no_information = [
    'iid',
    'pid',
    'wave',
    'position',
    'order'
]


# Drop features that are known in the future

# In[ ]:


features_future_information = [
    'dec',
    'dec_o',
    'like',
    'prob',
    'like_o',
    'prob_o'
]


# Drop features that have low variance

# In[ ]:


feature_variances = data.std().sort_values(ascending=True)
features_low_variance = feature_variances[feature_variances < 0.1].index.values.tolist()


# Drop features that have weak correlation with target variable

# In[ ]:


features_weak_correlation = partner_decision_correlations[partner_decision_correlations < 0.1].axes[0].to_list()
features_weak_correlation = list(set(features_weak_correlation) - set(features_future_information) - set(features_no_information))


# Drop features that were used in interaction variables

# In[ ]:


features_interaction = [
    'age',
    'age_o',
]


# In[ ]:


features_remove = features_no_information+features_future_information+features_low_variance+features_weak_correlation+features_interaction
data_model = data.drop(columns=features_remove)


# Get memory usage of model dataframe

# In[ ]:


data_model.memory_usage().sum()


# ### 5.6 Data export<a id="5.6"></a>

# Persist dataframe containing features to be used in model

# In[ ]:


data_model.to_csv(
    path_or_buf='./data_model.csv',
    index=False
)


# ## 6 Modelling<a id="6"></a>

# ### 6.1 Baseline models<a id="6.1"></a>

# Define feature and target variables

# In[ ]:


features = data_model
target = data['dec_o']


# Instantiate, train and tune a linear model

# In[ ]:


parameters = {
    'penalty': ['l2'],
    'solver': ['lbfgs'],
    'C': np.logspace(-4, 4, 20),
    'max_iter': [10000]
}
classifier_lr = LogisticRegression(random_state=0)
classifier_lr = GridSearchCV(
    estimator=classifier_lr,
    param_grid=parameters,
    cv=5,
    verbose=2,
    n_jobs=-1
)
classifier_lr.fit(features, target)
classifier_lr.best_params_


# In[ ]:


classifier_lr = LogisticRegression(
    random_state=0,
    penalty=classifier_lr.best_params_['penalty'],
    solver=classifier_lr.best_params_['solver'],
    C=classifier_lr.best_params_['C'],
    max_iter=classifier_lr.best_params_['max_iter']
)


# Instantiate, train and tune a discriminative model

# In[ ]:


parameters = {
    'kernel': ['rbf'],
    'gamma': [1e-4, 1e-3, 1e-2],
    'C': [1, 10, 100, 1000]
}
classifier_sv = SVC(random_state=0)
classifier_sv = GridSearchCV(
    estimator=classifier_sv,
    param_grid=parameters,
    cv=5,
    verbose=2,
    n_jobs=-1
)
classifier_sv.fit(features, target)
classifier_sv.best_params_


# In[ ]:


classifier_sv = SVC(
    random_state=0,
    kernel=classifier_sv.best_params_['kernel'],
    gamma=classifier_sv.best_params_['gamma'],
    C=classifier_sv.best_params_['C']
)


# Instantiate, train and tune a instance-based model

# In[ ]:


parameters = {
    'n_neighbors': [5,11,19,29],
    'weights': ['uniform', 'distance'],
    'metric': ['minkowski', 'euclidean', 'manhattan']
}
classifier_kn = KNeighborsClassifier()
classifier_kn = GridSearchCV(
    estimator=classifier_kn,
    param_grid=parameters,
    cv=5,
    verbose=2,
    n_jobs=-1
)
classifier_kn.fit(features, target)
classifier_kn.best_params_


# In[ ]:


classifier_kn = KNeighborsClassifier(
    n_neighbors=classifier_kn.best_params_['n_neighbors'],
    weights=classifier_kn.best_params_['weights'],
    metric=classifier_kn.best_params_['metric']
)


# ### 6.2 Ensemble models<a id="6.2"></a>

# Instantiate, train and tune a boosting model

# In[ ]:


parameters = {
    'loss': ['deviance', 'exponential'],
    'learning_rate': [0.05],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'max_features': ['sqrt', 'log2']
}
classifier_gb = GradientBoostingClassifier(random_state=0)
classifier_gb = GridSearchCV(
    estimator=classifier_gb,
    param_grid=parameters,
    cv=5,
    verbose=2,
    n_jobs=-1
)
classifier_gb.fit(features, target)
classifier_gb.best_params_


# In[ ]:


classifier_gb = GradientBoostingClassifier(
    random_state=0,
    loss=classifier_gb.best_params_['loss'],
    learning_rate=classifier_gb.best_params_['learning_rate'],
    n_estimators=classifier_gb.best_params_['n_estimators'],
    max_depth=classifier_gb.best_params_['max_depth'],
    max_features=classifier_gb.best_params_['max_features']
)


# Instantiate and train a voting and stacking model

# In[ ]:


estimators = [
    ('lr', classifier_lr),
    ('sv', classifier_sv),
    ('kn', classifier_kn),
    ('gb', classifier_gb)
]


# In[ ]:


classifier_ve = VotingClassifier(
    estimators=estimators,
    voting='hard'
)


# In[ ]:


classifier_se = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)


# ## 7 Evaluation<a id="7"></a>

# ### 7.1 Classifier performance<a id="7.1"></a>

# Calculate performance metrics for each model

# In[ ]:


metrics = ['accuracy', 'precision', 'recall', 'f1_macro']

for classifier, label in zip(
    [classifier_lr, classifier_sv, classifier_kn, classifier_gb, classifier_ve, classifier_se],
    ['Logistic Regression', 'Support Vector Machine', 'k-Nearest Neighbours', 'Gradient Boosting', 'Voting Ensemble', 'Stacking Ensemble']
):
    print('{}'.format(label))
    scores = cross_validate(
        estimator=classifier,
        X=features,
        y=target,
        scoring=metrics,
        cv=5,
        n_jobs=-1
    )
    for key, value in scores.items():
        print('{:14} {:.3f} +/- {:.3f}'.format(key, value.mean(), value.std()))
    print('\n')


# ### 7.2 Feature importance<a id="7.2"></a>

# Get coefficients of features in linear model

# In[ ]:


labels = features.columns.values
weights = classifier_lr.fit(features,target).coef_[0]


# Get features with highest magnitude

# In[ ]:


top_features = sorted(list(zip(labels,weights)), reverse=True, key = lambda x: abs(x[1]))[0:10]
top_labels = [x[0] for x in top_features]
top_weights = [x[1] for x in top_features]


# Plot features with highest magnitude

# In[ ]:


plt.figure(figsize=(10,8))
plot_relationship(top_weights, top_labels, 'Most significant features in linear model', 'Weight', 'Feature')


# ## 8 Conclusion<a id="8"></a>

# - This notebook explored whether a machine learning model could predict if a partner will match with their date.
# - The best performing classifier was a stacking ensemble model with an accuracy of **~76%** and an F1 score of **~75%**.
# - Based on correlation analysis and weights in the linear model:
#   - **Attractiveness**, **shared interests** and being **fun** were the most significant factors in a partner's decision.
#   - **Medical students** had the highest probability of being matched while **psychologists** and **academics** had the lowest.
# - Further feature engineering such as introducing more interaction features and performing more thorough feature selection could improve the classifier performance.
