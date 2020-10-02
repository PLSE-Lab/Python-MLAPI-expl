#!/usr/bin/env python
# coding: utf-8

# # Featuretools for Good
# 
# In this notebook, we will implement automated feature engineering with [Featuretools](https://docs.featuretools.com/#minute-quick-start) for the Costa Rican Household Poverty Challenge. The objective of this data science for good problem is to predict the poverty of households in Costa Rica. 
# 
# ## Automated Feature Engineering
# 
# Automated feature engineering should be a _default_ part of your data science workflow. Manual feature engineering is limited both by human creativity and time constraints but automated methods have no such constraints. At the moment, Featuretools is the only open-source Python library available for automated feature engineering. This library is extremely easy to get started with and very powerful (as the score from this kernel illustrates). 
# 
# For anyone new to featuretools, check out the [documentation](https://docs.featuretools.com/getting_started/install.html) or an [introductory blog post here.](https://towardsdatascience.com/automated-feature-engineering-in-python-99baf11cc219) 

# In[ ]:


import numpy as np 
import pandas as pd

import featuretools as ft

import warnings
warnings.filterwarnings('ignore')


# We'll read in the data and join the training and testing set together. 

# In[ ]:


# Raw data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
test['Target'] = np.nan

data = train.append(test, sort = True)


# In[ ]:


train_valid = train.loc[train['parentesco1'] == 1, ['idhogar', 'Id', 'Target']].copy()
test_valid = test.loc[test['parentesco1'] == 1, ['idhogar', 'Id']].copy()

submission_base = test[['Id', 'idhogar']]


# ### Data Preprocessing 
# 
# These steps are laid out in the kernel [A Complete Introduction and Walkthrough](https://www.kaggle.com/willkoehrsen/a-complete-introduction-and-walkthrough).  They involve correcting missing values, creating a few features (that Featuretools can build on top of). 

# In[ ]:


mapping = {"yes": 1, "no": 0}

# Fill in the values with the correct mapping
data['dependency'] = data['dependency'].replace(mapping).astype(np.float64)
data['edjefa'] = data['edjefa'].replace(mapping).astype(np.float64)
data['edjefe'] = data['edjefe'].replace(mapping).astype(np.float64)

data[['dependency', 'edjefa', 'edjefe']].describe()


# ## Missing Values

# In[ ]:


data['v18q1'] = data['v18q1'].fillna(0)

# Fill in households that own the house with 0 rent payment
data.loc[(data['tipovivi1'] == 1), 'v2a1'] = 0

# Create missing rent payment column
data['v2a1-missing'] = data['v2a1'].isnull()

# If individual is over 19 or younger than 7 and missing years behind, set it to 0
data.loc[((data['age'] > 19) | (data['age'] < 7)) & (data['rez_esc'].isnull()), 'rez_esc'] = 0

# Add a flag for those between 7 and 19 with a missing value
data['rez_esc-missing'] = data['rez_esc'].isnull()

data.loc[data['rez_esc'] > 5, 'rez_esc'] = 5


# ## Domain Knowledge Feature Construction

# In[ ]:


# Difference between people living in house and household size
data['hhsize-diff'] = data['tamviv'] - data['hhsize']

elec = []

# Assign values
for i, row in data.iterrows():
    if row['noelec'] == 1:
        elec.append(0)
    elif row['coopele'] == 1:
        elec.append(1)
    elif row['public'] == 1:
        elec.append(2)
    elif row['planpri'] == 1:
        elec.append(3)
    else:
        elec.append(np.nan)
        
# Record the new variable and missing flag
data['elec'] = elec
data['elec-missing'] = data['elec'].isnull()

# Remove the electricity columns
# data = data.drop(columns = ['noelec', 'coopele', 'public', 'planpri'])

# Wall ordinal variable
data['walls'] = np.argmax(np.array(data[['epared1', 'epared2', 'epared3']]),
                           axis = 1)

# data = data.drop(columns = ['epared1', 'epared2', 'epared3'])

# Roof ordinal variable
data['roof'] = np.argmax(np.array(data[['etecho1', 'etecho2', 'etecho3']]),
                           axis = 1)
# data = data.drop(columns = ['etecho1', 'etecho2', 'etecho3'])

# Floor ordinal variable
data['floor'] = np.argmax(np.array(data[['eviv1', 'eviv2', 'eviv3']]),
                           axis = 1)
# data = data.drop(columns = ['eviv1', 'eviv2', 'eviv3'])

# Create new feature
data['walls+roof+floor'] = data['walls'] + data['roof'] + data['floor']

# No toilet, no electricity, no floor, no water service, no ceiling
data['warning'] = 1 * (data['sanitario1'] + 
                         (data['elec'] == 0) + 
                         data['pisonotiene'] + 
                         data['abastaguano'] + 
                         (data['cielorazo'] == 0))

# Owns a refrigerator, computer, tablet, and television
data['bonus'] = 1 * (data['refrig'] + 
                      data['computer'] + 
                      (data['v18q1'] > 0) + 
                      data['television'])

# Per capita features
data['phones-per-capita'] = data['qmobilephone'] / data['tamviv']
data['tablets-per-capita'] = data['v18q1'] / data['tamviv']
data['rooms-per-capita'] = data['rooms'] / data['tamviv']
data['rent-per-capita'] = data['v2a1'] / data['tamviv']

# Create one feature from the `instlevel` columns
data['inst'] = np.argmax(np.array(data[[c for c in data if c.startswith('instl')]]), axis = 1)
# data = data.drop(columns = [c for c in data if c.startswith('instlevel')])

data['escolari/age'] = data['escolari'] / data['age']
data['inst/age'] = data['inst'] / data['age']
data['tech'] = data['v18q'] + data['mobilephone']

print('Data shape: ', data.shape)


# ### Remove Squared Variables
# 
# The gradient boosting machine does not need the squared version of variables it if already has the original variables. 

# In[ ]:


data = data[[x for x in data if not x.startswith('SQB')]]
data = data.drop(columns = ['agesq'])
data.shape


# ## Remove Highly Correlated Columns

# In[ ]:


# Create correlation matrix
corr_matrix = data.corr()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.975)]

print(f'There are {len(to_drop)} correlated columns to remove.')
print(to_drop)


# In[ ]:


data = data.drop(columns = to_drop)


# #  Establish Correct Variable Types
# 
# We need to specify the correct variables types:
# 
# 1. Individual Variables: these are characteristics of each individual rather than the household
#     * Boolean: Yes or No (0 or 1)
#     * Ordered Discrete: Integers with an ordering
# 2. Household variables
#     * Boolean: Yes or No
#     * Ordered Discrete: Integers with an ordering
#     * Continuous numeric
# 
# Below we manually define the variables in each category. This is a little tedious, but also necessary.

# In[ ]:


import featuretools.variable_types as vtypes


# In[ ]:


hh_bool = ['hacdor', 'hacapo', 'v14a', 'refrig', 'paredblolad', 'paredzocalo', 
           'paredpreb','pisocemento', 'pareddes', 'paredmad',
           'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisoother', 
           'pisonatur', 'pisonotiene', 'pisomadera',
           'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo', 
           'abastaguadentro', 'abastaguafuera', 'abastaguano',
            'public', 'planpri', 'noelec', 'coopele', 'sanitario1', 
           'sanitario2', 'sanitario3', 'sanitario5',   'sanitario6',
           'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4', 
           'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 
           'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3',
           'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 
           'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 
           'computer', 'television', 'lugar1', 'lugar2', 'lugar3',
           'lugar4', 'lugar5', 'lugar6', 'area1', 'area2', 'v2a1-missing', 'elec-missing']

hh_ordered = [ 'rooms', 'r4h1', 'r4h2', 'r4h3', 'r4m1','r4m2','r4m3', 'r4t1',  'r4t2', 
              'r4t3', 'v18q1', 'tamhog','tamviv','hhsize','hogar_nin','hhsize-diff',
              'elec',  'walls', 'roof', 'floor', 'walls+roof+floor', 'warning', 'bonus',
              'hogar_adul','hogar_mayor','hogar_total',  'bedrooms', 'qmobilephone']

hh_cont = ['v2a1', 'dependency', 'edjefe', 'edjefa', 'meaneduc', 'overcrowding',
          'phones-per-capita', 'tablets-per-capita', 'rooms-per-capita', 'rent-per-capita']


# In[ ]:


ind_bool = ['v18q', 'dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3', 
            'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 
            'parentesco1', 'parentesco2',  'parentesco3', 'parentesco4', 'parentesco5', 
            'parentesco6', 'parentesco7', 'parentesco8',  'parentesco9', 'parentesco10', 
            'parentesco11', 'parentesco12', 'instlevel1', 'instlevel2', 'instlevel3', 
            'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 
            'instlevel9', 'mobilephone', 'rez_esc-missing']

ind_ordered = ['age', 'escolari', 'rez_esc', 'inst', 'tech']

ind_cont = ['escolari/age', 'inst/age']


# The cells below remove any columns that aren't in the data (these may have been removed due to correlation).

# In[ ]:


to_remove = []
for l in [hh_ordered, hh_bool, hh_cont, ind_bool, ind_ordered, ind_cont]:
    for c in l:
        if c not in data:
            to_remove.append(c)


# In[ ]:


for l in [hh_ordered, hh_bool, hh_cont, ind_bool, ind_ordered, ind_cont]:
    for c in to_remove:
        if c in l:
            l.remove(c)


# The three columns not in the above lists are `Id`, `Idhogar`, and `Target`. 

# In[ ]:


len(hh_ordered+hh_bool+hh_cont+ind_bool+ind_ordered+ind_cont) == (data.shape[1] - 3)


# Below we convert the `Boolean` variables to the correct type. 

# In[ ]:


for variable in (hh_bool + ind_bool):
    data[variable] = data[variable].astype('bool')


# Then we convert the float variables.

# In[ ]:


for variable in (hh_cont + ind_cont):
    data[variable] = data[variable].astype(float)


# Finally, the same with the ordinal variables.

# In[ ]:


for variable in (hh_ordered + ind_ordered):
    try:
        data[variable] = data[variable].astype(int)
    except Exception as e:
        print(f'Could not convert {variable} because of missing values.')


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data.dtypes.value_counts().plot.bar(edgecolor = 'k');
plt.title('Variable Type Distribution');


# # EntitySet and Entities
# 
# An `EntitySet` in Featuretools holds all of the tables and the relationships between them. At the moment we only have a single table, but we can create multiple tables through normalization. We'll call the first table `data` since it contains all the information both at the individual level and at the household level.

# In[ ]:


es = ft.EntitySet(id = 'households')
es.entity_from_dataframe(entity_id = 'data', 
                         dataframe = data, 
                         index = 'Id')


# # Normalize Household Table
# 
# Normalization allows us to create another table with one unique row per instance. In this case, the instances are households. The new table is derived from the `data` table and we need to bring along any of the household level variables. Since these are the same for all members of a household, we can directly add these as columns in the household table using `additional_variables`. The index of the household table is `idhogar` which uniquely identifies each household.  
# 
# All of the variable types have already been confirmed.

# In[ ]:


es.normalize_entity(base_entity_id='data', 
                    new_entity_id='household', 
                    index = 'idhogar', 
                    additional_variables = hh_bool + hh_ordered + hh_cont + ['Target'])
es


# ### Table Relationships
# 
# Normalizing the entity automatically adds in the relationship between the parent, `household`, and the child, `ind`. This relationship links the two tables and allows us to create "deep features" by aggregating individuals in each household.

# # Deep Feature Synthesis
# 
# Here is where Featuretools gets to work. Using feature primitives, Deep Feature Synthesis can build hundreds (or 1000s as we will later see) of features from the relationships between tables and the columns in tables themselves. There are two types of primitives, which are operations applied to data:
# 
# * Transforms: applied to one or more columns in a _single table_ of data 
# * Aggregations: applied across _multiple tables_ using the relationships between tables
# 
# We generate the features by calling `ft.dfs`. This build features using any of the applicable primitives for each column in the data. Featuretools uses the table relationships to aggregate features as required. For example, it will automatically aggregate the individual level data at the household level. 

# To start with, we use the default `agg` and `trans` primitives in a call to `ft.dfs`.

# In[ ]:


# Deep Feature Synthesis
feature_matrix, feature_names = ft.dfs(entityset=es, 
                                       target_entity = 'household', 
                                       max_depth = 2, 
                                       verbose = 1, 
                                       n_jobs = -1, 
                                       chunk_size = 100)


# In[ ]:


all_features = [str(x.get_name()) for x in feature_names]
feature_matrix.head()


# In[ ]:


all_features[-10:]


# We need to remove any columns containing derivations of the `Target`. These are created because some of transform primitives might have affected the `Target`.

# In[ ]:


drop_cols = []
for col in feature_matrix:
    if col == 'Target':
        pass
    else:
        if 'Target' in col:
            drop_cols.append(col)
            
print(drop_cols)            
feature_matrix = feature_matrix[[x for x in feature_matrix if x not in drop_cols]]         
feature_matrix.head()


# Most of these features are aggregations we could have made ourselves. However, why go to the trouble if Featuretools can do that for us?

# In[ ]:


feature_matrix.shape


# That one call alone gave us 147 features to train a model! This was only using the default primitives as well. We can use more primitives or write our own to build more features.

# # Feature Selection
# 
# We can do some rudimentary feature selection, removing one of any pair of columns with a correlation greater than 0.99 (absolute value).

# In[ ]:


# Create correlation matrix
corr_matrix = feature_matrix.corr().abs()
# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] >= 0.99)]

print('There are {} columns with >= 0.99 correlation.'.format(len(to_drop)))
to_drop


# In[ ]:


feature_matrix = feature_matrix[[x for x in feature_matrix if x not in to_drop]]


# ### Training and Testing Data

# In[ ]:


train = feature_matrix[feature_matrix['Target'].notnull()].reset_index()
test = feature_matrix[feature_matrix['Target'].isnull()].reset_index()
train.head()


# # Correlations with the target

# In[ ]:


corrs = train.corr()
corrs['Target'].sort_values(ascending = True).head(10)


# In[ ]:


corrs['Target'].sort_values(ascending = True).dropna().tail(10)


# Featuretools has built features with moderate correlations with the `Target`. Although these correlations only show linear relationships, they can still provide an approximation of what features will be "useful" to a machine learning model.

# ## Subset to Relevant Data

# In[ ]:


train = train[train['idhogar'].isin(list(train_valid['idhogar']))]
train.head()


# In[ ]:


test = test[test['idhogar'].isin(list(test_valid['idhogar']))]
test.head()


# ### Labels for Training

# In[ ]:


train_labels = np.array(train.pop('Target')).reshape((-1,))
test_ids = list(test.pop('idhogar'))


# In[ ]:


train, test = train.align(test, axis = 1, join = 'inner')
all_features = list(train.columns)
train.shape


# We'll now get into modeling. The gradient boosting machine implemented in LightGBM usually does well! 

# In[ ]:


get_ipython().run_cell_magic('capture', '', '\n# Visualization\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\nfrom collections import Counter\nfrom sklearn.metrics import f1_score, make_scorer\nfrom sklearn.model_selection import StratifiedKFold\n\nimport lightgbm as lgb')


# ## Custom Evaluation Metric for LightGBM
# 
# This is the F1 Macro score used by the competition. Defining a custom evaluation metric for Light GBM is not exactly straightforward but we can manage.

# In[ ]:


def macro_f1_score(labels, predictions):
    # Reshape the predictions as needed
    predictions = predictions.reshape(len(np.unique(labels)), -1 ).argmax(axis = 0)
    
    metric_value = f1_score(labels, predictions, average = 'macro')
    
    # Return is name, value, is_higher_better
    return 'macro_f1', metric_value, True


# # Modeling with Gradient Boosting Machine
# 
# The hyperparameters used here _have not been optimized_. This is meant only as a first pass at modeling with these features. 

# In[ ]:


from IPython.display import display


# In[ ]:


def model_gbm(features, labels, test_features, test_ids, 
              nfolds = 5, return_preds = False, hyp = None):
    """Model using the GBM and cross validation.
       Trains with early stopping on each fold.
       Hyperparameters probably need to be tuned."""
    
    feature_names = list(features.columns)
    
    hyp_OPTaaS = { 'boosting_type': 'dart',
              'colsample_bytree': 0.9843467236959204,
              'learning_rate': 0.11598629586769524,
              'min_child_samples': 44,
              'num_leaves': 49,
              'reg_alpha': 0.35397370408131534,
              'reg_lambda': 0.5904910774606467,
              'subsample': 0.6299872254632797,
              'subsample_for_bin': 60611}

    # Model hyperparameters
#     params = {'boosting_type': 'dart', 
#               'colsample_bytree': 0.88, 
#               'learning_rate': 0.028, 
#                'min_child_samples': 10, 
#                'num_leaves': 36, 'reg_alpha': 0.76, 
#                'reg_lambda': 0.43, 
#                'subsample_for_bin': 40000, 
#                'subsample': 0.54}

    model = lgb.LGBMClassifier(**hyp_OPTaaS, class_weight = 'balanced',
                               objective = 'multiclass', n_jobs = -1, n_estimators = 10000)
    
    # Using stratified kfold cross validation
    strkfold = StratifiedKFold(n_splits = nfolds, shuffle = True)
    predictions = pd.DataFrame()
    importances = np.zeros(len(feature_names))
    
    # Convert to arrays for indexing
    features = np.array(features)
    test_features = np.array(test_features)
    labels = np.array(labels).reshape((-1 ))
    
    valid_scores = []
    
    # Iterate through the folds
    for i, (train_indices, valid_indices) in enumerate(strkfold.split(features, labels)):
        # Dataframe for 
        fold_predictions = pd.DataFrame()
        
        # Training and validation data
        X_train = features[train_indices]
        X_valid = features[valid_indices]
        y_train = labels[train_indices]
        y_valid = labels[valid_indices]
        
        # Train with early stopping
        model.fit(X_train, y_train, early_stopping_rounds = 100, 
                  eval_metric = macro_f1_score,
                  eval_set = [(X_train, y_train), (X_valid, y_valid)],
                  eval_names = ['train', 'valid'],
                  verbose = 200)
        
        # Record the validation fold score
        valid_scores.append(model.best_score_['valid']['macro_f1'])
        
        # Make predictions from the fold
        fold_probabilitites = model.predict_proba(test_features)
        
        # Record each prediction for each class as a column
        for j in range(4):
            fold_predictions[(j + 1)] = fold_probabilitites[:, j]
            
        fold_predictions['idhogar'] = test_ids
        fold_predictions['fold'] = (i+1)
        predictions = predictions.append(fold_predictions)
        
        importances += model.feature_importances_ / nfolds   
        
        display(f'Fold {i + 1}, Validation Score: {round(valid_scores[i], 5)}, Estimators Trained: {model.best_iteration_}')

    feature_importances = pd.DataFrame({'feature': feature_names,
                                        'importance': importances})
    valid_scores = np.array(valid_scores)
    display(f'{nfolds} cross validation score: {round(valid_scores.mean(), 5)} with std: {round(valid_scores.std(), 5)}.')
    
    # If we want to examine predictions don't average over folds
    if return_preds:
        predictions['Target'] = predictions[[1, 2, 3, 4]].idxmax(axis = 1)
        predictions['confidence'] = predictions[[1, 2, 3, 4]].max(axis = 1)
        return predictions, feature_importances
    
    # Average the predictions over folds
    predictions = predictions.groupby('idhogar', as_index = False).mean()
    
    # Find the class and associated probability
    predictions['Target'] = predictions[[1, 2, 3, 4]].idxmax(axis = 1)
    predictions['confidence'] = predictions[[1, 2, 3, 4]].max(axis = 1)
    predictions = predictions.drop(columns = ['fold'])
    
    # Merge with the base to have one prediction for each individual
    submission = submission_base.merge(predictions[['idhogar', 'Target']], on = 'idhogar', how = 'left').drop(columns = ['idhogar'])
        
    submission['Target'] = submission['Target'].fillna(4).astype(np.int8)
    
    # return the submission and feature importances
    return submission, feature_importances, valid_scores


# We need to make sure the length of the labels matches the length of the training dataset.

# In[ ]:


len(train_labels) == train.shape[0]


# We should also make sure the len of `test_ids` (the `idhogar` of the testing households) is the same as the length of the testing dataset.

# In[ ]:


len(test_ids) == test.shape[0]


# All that's left is to model! The cell below runs the gradient boosting machine model and saves the results. 

# In[ ]:


get_ipython().run_cell_magic('capture', '--no-display', "submission, feature_importances, valid_scores = model_gbm(train, train_labels, test, test_ids, 5)\n\nresults = pd.DataFrame({'version': ['default_5fold'], \n                        'F1-mean': [valid_scores.mean()], \n                        'F1-std': [valid_scores.std()]})")


# I'm not running the GBM with a random seed so the same set of features can produce different cross validation results. A random seed would ensure consistent results, but may have a singificant effect on the predictions.

# ## Feature Importances
# 
# The utility function below plots feature importances and can show us how many features are needed for a certain cumulative level of importance. 

# In[ ]:


def plot_feature_importances(df, n = 15, return_features = False, threshold = None):
    """Plots n most important features. Also plots the cumulative importance if
    threshold is specified and prints the number of features needed to reach threshold cumulative importance.
    Intended for use with any tree-based feature importances. 
    
    Args:
        df (dataframe): Dataframe of feature importances. Columns must be "feature" and "importance".
    
        n (int): Number of most important features to plot. Default is 15.
    
        threshold (float): Threshold for cumulative importance plot. If not provided, no plot is made. Default is None.
        
    Returns:
        df (dataframe): Dataframe ordered by feature importances with a normalized column (sums to 1) 
                        and a cumulative importance column
    
    Note:
    
        * Normalization in this case means sums to 1. 
        * Cumulative importance is calculated by summing features from most to least important
        * A threshold of 0.9 will show the most important features needed to reach 90% of cumulative importance
    
    """
    
    # Sort features with most important at the head
    df = df.sort_values('importance', ascending = False).reset_index(drop = True)
    
    # Normalize the feature importances to add up to one and calculate cumulative importance
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])
    
    plt.rcParams['font.size'] = 12
    plt.style.use('fivethirtyeight')
    # Bar plot of n most important features
    df.loc[:n, :].plot.barh(y = 'importance_normalized', 
                            x = 'feature', color = 'blue', 
                            edgecolor = 'k', figsize = (12, 8),
                            legend = False, linewidth = 2)

    plt.xlabel('Normalized Importance', size = 18); plt.ylabel(''); 
    plt.title(f'Top {n} Most Important Features', size = 18)
    plt.gca().invert_yaxis()
    
    
    if threshold:
        # Cumulative importance plot
        plt.figure(figsize = (8, 6))
        plt.plot(list(range(len(df))), df['cumulative_importance'], 'b-')
        plt.xlabel('Number of Features', size = 16); plt.ylabel('Cumulative Importance', size = 16); 
        plt.title('Cumulative Feature Importance', size = 18);
        
        # Number of features needed for threshold cumulative importance
        # This is the index (will need to add 1 for the actual number)
        importance_index = np.min(np.where(df['cumulative_importance'] > threshold))
        
        # Add vertical line to plot
        plt.vlines(importance_index + 1, ymin = 0, ymax = 1.05, linestyles = '--', colors = 'red')
        plt.show();
        
        print('{} features required for {:.0f}% of cumulative importance.'.format(importance_index + 1, 
                                                                                  100 * threshold))
    if return_features:
        return df


# In[ ]:


plot_feature_importances(feature_importances)


# In[ ]:


submission.to_csv('ft_baseline.csv', index = False)


# In[ ]:


submission['Target'].value_counts().sort_index().plot.bar(color = 'blue');
plt.title('Distribution of Predicted Labels for Individuals', size = 14);


# These shows the predictions on an individual, not household level (we set all individuals to 4 if they did not have a head of household). The distribution is close to what we observe in the training labels, which are provided on the household level.

# In[ ]:


data[data['Target'].notnull()]['Target'].value_counts().sort_index().plot.bar(color = 'blue');
plt.title('Distribution of Labels for Training Individuals', size = 12);


# # Custom Primitive
# 
# To expand the capabilities of featuretools, we can write our own primitives to be applied to the data. We'll write a simple function that finds the range of a numeric column. 
# 
# ### Range Primitive

# In[ ]:


from featuretools.primitives import make_agg_primitive

# Custom primitive
def range_calc(numeric):
    return np.max(numeric) - np.min(numeric)

range_ = make_agg_primitive(function = range_calc,
                            input_types = [ft.variable_types.Numeric], 
                            return_type = ft.variable_types.Numeric)


# We can also make a custom primitive that calculates the correlation coefficient between two columns.
# 
# ### Correlation Primitive

# In[ ]:


def p_corr_calc(numeric1, numeric2):
    return np.corrcoef(numeric1, numeric2)[0, 1]

pcorr_ = make_agg_primitive(function = p_corr_calc,
                            input_types = [ft.variable_types.Numeric, ft.variable_types.Numeric], 
                            return_type = ft.variable_types.Numeric)


# In[ ]:


def s_corr_calc(numeric1, numeric2):
    return spearmanr(numeric1, numeric2)[0]

scorr_ = make_agg_primitive(function = s_corr_calc, 
                           input_types = [ft.variable_types.Numeric, ft.variable_types.Numeric], 
                           return_type = ft.variable_types.Numeric)


# # More Featuretools
# 
# Why stop with 150 features? Let's add in a few more primitives and start creating more. To prevent featuretools from building the exact same features we already have, we can add `drop_exact` and pass in the feature names (as strings using the `get_name` functionality. 

# In[ ]:


get_ipython().run_cell_magic('capture', '', "feature_matrix_add, feature_names_add = ft.dfs(entityset=es, target_entity = 'household', \n                                              agg_primitives = ['min', 'max', 'mean', 'percent_true', 'all', 'any',\n                                                             'sum', 'skew', 'std', range_],\n                                          trans_primitives = [], drop_exact = all_features,\n                                          max_depth = 2, \n                                          verbose = 1, n_jobs = -1, \n                                          chunk_size = 100)")


# In[ ]:


all_features += [str(x.get_name()) for x in feature_names_add]
feature_matrix = pd.concat([feature_matrix, feature_matrix_add], axis = 1)
feature_matrix.shape


# # Post Processing Function
# 
# There are a number of steps after generating the feature matrix so let's put all of these in a function.
# 
# 1. Remove any duplicated columns.
# 2. Replace infinite values with `np.nan`
# 3. Remove columns with a missing percentage above the `missing_threshold`
# 4. Remove columns with only a single unique value.
# 5. Remove one out of every pair of columns with a correlation threshold above the `correlation_threshold`
# 6. Extract the training and testing data along with labels and ids (needed for making submissions)

# In[ ]:


def post_process(feature_matrix,
                 missing_threshold = 0.95, 
                 correlation_threshold = 0.95):
    
    # Remove duplicated features
    start_features = feature_matrix.shape[1]
    feature_matrix = feature_matrix.iloc[:, ~feature_matrix.columns.duplicated()]
    n_duplicated = start_features - feature_matrix.shape[1]
    print(f'There were {n_duplicated} duplicated features.')
    
    feature_matrix = feature_matrix.replace({np.inf: np.nan, -np.inf:np.nan}).reset_index()
    
    # Remove the ids and labels
    ids = list(feature_matrix.pop('idhogar'))
    labels = list(feature_matrix.pop('Target'))
    
    # Remove columns derived from the Target
    drop_cols = []
    for col in feature_matrix:
        if col == 'Target':
            pass
        else:
            if 'Target' in col:
                drop_cols.append(col)
                
    feature_matrix = feature_matrix[[x for x in feature_matrix if x not in drop_cols]] 
    
    # One hot encoding (if necessary)
    feature_matrix = pd.get_dummies(feature_matrix)
    n_features_start = feature_matrix.shape[1]
    print('Original shape: ', feature_matrix.shape)
    
    # Find missing and percentage
    missing = pd.DataFrame(feature_matrix.isnull().sum())
    missing['fraction'] = missing[0] / feature_matrix.shape[0]
    missing.sort_values('fraction', ascending = False, inplace = True)

    # Missing above threshold
    missing_cols = list(missing[missing['fraction'] > missing_threshold].index)
    n_missing_cols = len(missing_cols)

    # Remove missing columns
    feature_matrix = feature_matrix[[x for x in feature_matrix if x not in missing_cols]]
    print('{} missing columns with threshold: {}.'.format(n_missing_cols, missing_threshold))
    
    # Zero variance
    unique_counts = pd.DataFrame(feature_matrix.nunique()).sort_values(0, ascending = True)
    zero_variance_cols = list(unique_counts[unique_counts[0] == 1].index)
    n_zero_variance_cols = len(zero_variance_cols)

    # Remove zero variance columns
    feature_matrix = feature_matrix[[x for x in feature_matrix if x not in zero_variance_cols]]
    print('{} zero variance columns.'.format(n_zero_variance_cols))
    
    # Correlations
    corr_matrix = feature_matrix.corr()

    # Extract the upper triangle of the correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))

    # Select the features with correlations above the threshold
    # Need to use the absolute value
    to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]

    n_collinear = len(to_drop)
    
    feature_matrix = feature_matrix[[x for x in feature_matrix if x not in to_drop]]
    print('{} collinear columns removed with correlation above {}.'.format(n_collinear,  correlation_threshold))
    
    total_removed = n_duplicated + n_missing_cols + n_zero_variance_cols + n_collinear
    
    print('Total columns removed: ', total_removed)
    print('Shape after feature selection: {}.'.format(feature_matrix.shape))
    
    # Extract the ids and labels
    feature_matrix['idhogar'] = ids
    feature_matrix['Target'] = labels
    
    # Extract out training and testing data
    train = feature_matrix[feature_matrix['Target'].notnull()]
    test = feature_matrix[feature_matrix['Target'].isnull()]
    
    # Subset to houses with a head of household
    train = train[train['idhogar'].isin(list(train_valid['idhogar']))]
    test = test[test['idhogar'].isin(list(test_valid['idhogar']))]
    
    # Training labels and testing household ids
    train_labels = np.array(train.pop('Target')).reshape((-1,))
    test_ids = list(test.pop('idhogar'))
    
    # Align the dataframes to ensure they have the same columns
    train, test = train.align(test, join = 'inner', axis = 1)
    
    assert (len(train_labels) == train.shape[0]), "Labels must be same length as number of training observations"
    assert(len(test_ids) == test.shape[0]), "Must be equal number of test ids as testing observations"
    
    return train, train_labels, test, test_ids


# In[ ]:


train, train_labels, test, test_ids = post_process(feature_matrix)


# ## Results After Post-Processing

# In[ ]:


get_ipython().run_cell_magic('capture', '--no-display', "submission, feature_importances, valid_scores = model_gbm(train, train_labels, test, test_ids, 5)\nresults = results.append(pd.DataFrame({'version': ['additional_5fold'], \n                                       'F1-mean': [valid_scores.mean()], \n                                       'F1-std': [valid_scores.std()]}))")


# In[ ]:


plot_feature_importances(feature_importances)


# # Increase number of features

# In[ ]:


get_ipython().run_cell_magic('capture', '', "feature_matrix_add, feature_names_add = ft.dfs(entityset=es, target_entity = 'household', \n                                       agg_primitives = ['min', 'max', 'mean', 'percent_true', 'all', 'any',\n                                                         'sum', 'skew', 'std', range_, pcorr_],\n                                       trans_primitives = [], drop_exact = list(all_features),\n                                       max_depth = 2, max_features = 1000,\n                                       verbose = 1, n_jobs = -1, \n                                       chunk_size = 100)")


# In[ ]:


all_features += [str(x.get_name()) for x in feature_names_add]
feature_matrix = pd.concat([feature_matrix, feature_matrix_add], axis = 1)
train, train_labels, test, test_ids = post_process(feature_matrix)
train.shape


# In[ ]:


get_ipython().run_cell_magic('capture', '--no-display', "\nsubmission, feature_importances, valid_scores = model_gbm(train, train_labels, test, test_ids, 5)\nresults = results.append(pd.DataFrame({'version': ['additionalft_5fold'], \n                                       'F1-mean': [valid_scores.mean()], \n                                       'F1-std': [valid_scores.std()]}))")


# ## Remove Zero Importance Features

# In[ ]:


print('Original training shape', train.shape)
train = train[list(feature_importances.loc[feature_importances['importance'] != 0, 'feature'])]
test = test[train.columns]

print('Training shape after removing zero importance features', train.shape)


# # Add in Divide Primitive
# 
# Next we'll add a `divide` transform primitive into the deep feature synthesis call. At first we'll limit the features to 1000. 

# In[ ]:


get_ipython().run_cell_magic('capture', '', "feature_matrix_add, feature_names_add = ft.dfs(entityset=es, target_entity = 'household', \n                                       agg_primitives = ['min', 'max', 'mean', 'percent_true', 'all', 'any',\n                                                         'sum', 'skew', 'std', range_, pcorr_],\n                                       trans_primitives = ['divide'], drop_contains = list(all_features),\n                                       max_depth = 2, max_features = 1000,\n                                       verbose = 1, n_jobs = -1, \n                                       chunk_size = 1000)")


# In[ ]:


all_features += [str(x.get_name()) for x in feature_names_add]
feature_matrix = pd.concat([feature_matrix, feature_matrix_add], axis = 1)
train, train_labels, test, test_ids = post_process(feature_matrix)
train.shape


# In[ ]:


get_ipython().run_cell_magic('capture', '--no-display', "\nsubmission, feature_importances, valid_scores = model_gbm(train, train_labels, test, test_ids, 5)\nresults = results.append(pd.DataFrame({'version': ['divide1000_5fold'], \n                                       'F1-mean': [valid_scores.mean()], \n                                       'F1-std': [valid_scores.std()]}))\nsubmission.to_csv('divide1000_featuretools.csv', index = False)")


# In[ ]:


plot_feature_importances(feature_importances)


# ## Increase to 1500 features
# 
# 1000 is clearly not enough! Most of these features are highly correlated, but we can still find useful features as evidenced by the feature importances.

# In[ ]:


feature_matrix_add, feature_names_add = ft.dfs(entityset=es, target_entity = 'household', 
                                       agg_primitives = ['min', 'max', 'mean', 'percent_true', 'all', 'any',
                                                         'sum', 'skew', 'std', range_, pcorr_],
                                       trans_primitives = ['divide'], drop_contains = list(all_features),
                                       max_depth = 2, max_features = 1500,
                                       verbose = 1, n_jobs = -1, 
                                       chunk_size = 1000)


# In[ ]:


all_features += [str(x.get_name()) for x in feature_names_add]
feature_matrix = pd.concat([feature_matrix, feature_matrix_add], axis = 1)
train, train_labels, test, test_ids = post_process(feature_matrix)
train.shape


# In[ ]:


get_ipython().run_cell_magic('capture', '--no-display', "\nsubmission, feature_importances, valid_scores = model_gbm(train, train_labels, test, test_ids, 5)\nresults = results.append(pd.DataFrame({'version': ['divide1500_5fold'], \n                                       'F1-mean': [valid_scores.mean()], \n                                       'F1-std': [valid_scores.std()]}))\nsubmission.to_csv('divide1500_featuretools.csv', index = False)")


# In[ ]:


plot_feature_importances(feature_importances)


# ## Go to 2000
# 
# This is getting ridiculous.
# 
# 

# In[ ]:


get_ipython().run_cell_magic('capture', '', "feature_matrix_add, feature_names_add = ft.dfs(entityset=es, target_entity = 'household', \n                                       agg_primitives = ['min', 'max', 'mean', 'percent_true', 'all', 'any',\n                                                         'sum', 'skew', 'std', range_, pcorr_],\n                                       trans_primitives = ['divide'], drop_exact = list(all_features),\n                                       max_depth = 2, max_features = 2000,\n                                       verbose = 1, n_jobs = -1, \n                                       chunk_size = 100)")


# In[ ]:


get_ipython().run_cell_magic('capture', '', 'all_features += [str(x.get_name()) for x in feature_names_add]\nfeature_matrix = pd.concat([feature_matrix, feature_matrix_add], axis = 1)\ntrain, train_labels, test, test_ids = post_process(feature_matrix)\ntrain.shape')


# In[ ]:


print('Total number of features considered: ', len(all_features))


# In[ ]:


get_ipython().run_cell_magic('capture', '--no-display', "\nsubmission, feature_importances, valid_scores = model_gbm(train, train_labels, test, test_ids, 5)\nresults = results.append(pd.DataFrame({'version': ['divide2000_5fold'], \n                                       'F1-mean': [valid_scores.mean()], \n                                       'F1-std': [valid_scores.std()]}))\nsubmission.to_csv('divide2000_featuretools.csv', index = False)")


# In[ ]:


plot_feature_importances(feature_importances)


# # Try Modeling with more folds
# 
# As a final model, we'll increase the number of folds to 10 and see if this results in more stable predictions across folds. It's concerning that there is so much variation between folds, but that is going to happen with a small, imbalanced testing set.

# In[ ]:


get_ipython().run_cell_magic('capture', '--no-display', "submission, feature_importances, valid_scores = model_gbm(train, train_labels, test, test_ids, 10)\nresults = results.append(pd.DataFrame({'version': ['divide2000_10fold'], \n                                       'F1-mean': [valid_scores.mean()], \n                                       'F1-std': [valid_scores.std()]}))\nsubmission.to_csv('divide2000_10fold_featuretools.csv', index = False)")


# # 5000! Features

# In[ ]:


get_ipython().run_cell_magic('capture', '', "feature_matrix_add, feature_names_add = ft.dfs(entityset=es, target_entity = 'household', \n                                       agg_primitives = ['min', 'max', 'mean', 'percent_true', 'all', 'any',\n                                                         'sum', 'skew', 'std', range_, pcorr_],\n                                       trans_primitives = ['divide'], drop_contains = list(all_features),\n                                       max_depth = 2, max_features = 5000,\n                                       verbose = 1, n_jobs = -1, \n                                       chunk_size = 100)")


# In[ ]:


all_features += [str(x.get_name()) for x in feature_names_add]
feature_matrix = pd.concat([feature_matrix, feature_matrix_add], axis = 1)
train, train_labels, test, test_ids = post_process(feature_matrix)
train.shape


# In[ ]:


get_ipython().run_cell_magic('capture', '--no-display', "submission, feature_importances, valid_scores = model_gbm(train, train_labels, test, test_ids, 5)\nresults = results.append(pd.DataFrame({'version': ['divide5000_5fold'], 'F1-mean': [valid_scores.mean()], 'F1-std': [valid_scores.std()]}))\nsubmission.to_csv('divide5000_featuretools.csv', index = False)")


# ### 5000 features with 10 fold modeling

# In[ ]:


get_ipython().run_cell_magic('capture', '--no-display', "submission, feature_importances, valid_scores = model_gbm(train, train_labels, test, test_ids, 10)\nresults = results.append(pd.DataFrame({'version': ['divide5000_10fold'], \n                                       'F1-mean': [valid_scores.mean()], \n                                       'F1-std': [valid_scores.std()]}))\nsubmission.to_csv('divide5000_10fold_featuretools.csv', index = False)")


# In[ ]:


print('Total number of features considered: ', len(all_features))


# # Comparison of Models
# 
# At this point we might honestly ask if there is any benefit to increasing the number of features. Only one way to find out: through data! Let's look at the performance of models so far.

# In[ ]:


results.set_index('version', inplace = True)

results['F1-mean'].plot.bar(color = 'orange', figsize = (8, 6),
                                  yerr = list(results['F1-std']))
plt.title('Model F1 Score Results');
plt.ylabel('Mean F1 Score (with error bar)');


# The cross validation accuracy continues to increase as we add features. I think we should be able to add more features as long as we continue to impose feature selection. The gradient boosting machine seems very good at cutting through the swath of features. Eventually we're probably going to be overfitting to the training data, but the we can address that through regularization and feature selection.

# # Save Data
# 
# We can save the final selected featuretools feature matrix (created with a maximum of 2000 features). This will be used for Bayesian optimization of model hyperparameters. There still might be additional gains to increasing the number of features and/or using different custom primitives. My focus is now going to shift to modeling, but I encourage anyone to keep adjusting the featuretools implementation.

# In[ ]:


feature_matrix = feature_matrix.iloc[:, ~feature_matrix.columns.duplicated()].reset_index()


# In[ ]:


train_ids = list(feature_matrix[(feature_matrix['Target'].notnull()) & (feature_matrix['idhogar'].isin(list(train_valid['idhogar'])))]['idhogar'])


# In[ ]:


print('Train shape before removing zero importance features:', train.shape)
train = train[list(feature_importances.loc[feature_importances['importance'] != 0, 'feature'])]
test = test[train.columns]
print('Train shape after removing zero importance features:', train.shape)


# In[ ]:


train['Target'] = train_labels
test['Target'] = np.nan
train['idhogar'] = train_ids
test['idhogar'] = test_ids
data = train.append(test)

results.to_csv('model_results.csv', index = True)
data.to_csv('ft_5000_important.csv', index = False)


# In[ ]:


print('Final shape of data (with testing joined to training): ', data.shape)


# # Conclusions
# 
# Featuretools certainly can make our job easier for this problem! Adding features continues to improve the validation score with mixed effects on the public leaderboard. The next step is to optimize the model for these features. __Featuretools should be a default part of your data science workflow.__ The tool is incredibly simple to use and delivers considerable value, creating features that we never would have imagined. I look forward to seeing what the community can come up with for this problem! 

# In[ ]:




