#!/usr/bin/env python
# coding: utf-8

# # Titanic Corvus Guide - Part II - Neural Networks Solution
# This notebook is continuation of the [Part 1 Notebook](https://www.kaggle.com/guidant/corvus-part-1-top-15-combining-5-models#5.-Final-Submission), where we used simple Scikit Learn models to write an ensemble solution and get to the top 15%. The question is: can we use the knowledge we already got about the Titanic Dataset and develop a new solution using neural networks? Let's see!
# 
# # 1. Once Again - The Libraries
# Let-me say it again - It's a good practice to start by importing all used libraries at once.

# In[ ]:


# BASIC LIBRARIES
import numpy as np
import pandas as pd
import tensorflow as tf

# SKLEARN TRANSFORMATIONS AND PIPELINES
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# NON-SUPERVISED LEARNING
from sklearn.cluster import DBSCAN

# SUPERVISED LEARNING - NEURAL NETWORKS LIBS
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, AlphaDropout, Input, Dense, concatenate
from keras.utils.vis_utils import plot_model

# SUPERVISED LEARNING - PART 1 LIBS
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

# MODEL SELECTION LIBS
import tensorflow.keras.backend as K

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import TensorBoard, EarlyStopping, LearningRateScheduler
from keras.optimizers import Nadam

# PLOTTING LIBS
import seaborn as sns
import matplotlib as mpl

# ACESSORY LIBRARIES
import re
import os
import dill
import time
import math
import random

from collections import Counter
from IPython.display import Image, FileLinks, display


# In[ ]:


tf.random.set_random_seed(42)
random.seed(42)
np.random.seed(42)


# In[ ]:


data_dir = '/kaggle/input/titanic'
K.clear_session()
get_ipython().run_line_magic('load_ext', 'tensorboard.notebook')


# (To check the TensorBoard, it's necessary to edit this notebook and run the code!)

# In[ ]:


get_ipython().run_line_magic('tensorboard', '--logdir logs')


# # 2. Taking Previous Results
# What do we already have? Simple:
# * 2.A The functions used in the transformations
# * 2.B The transformations used in the pipelines
# * 2.C The feature importance obtained during the Random Forest Algorithm (which can serve as a subside for us)
# 
# # 2.A Functions Used in the Transformations

# ## 2.A.1. Significant Data Transformations

# Functions to separe and classify the titles - Classifying the titles is important to reduce the number of classifications after aggregating the rare occurrences.

# In[ ]:


def get_title(dataframe_in):
    dataframe_in['Title'] = dataframe_in['Name'].apply(lambda X: re.search('[A-Z]{1}[a-z]+\.', X).group(0))
    return dataframe_in

def classify_title(dataframe_in):
    dataframe_in.loc[:, ['Title']] = dataframe_in['Title'].apply(lambda X: X if X in ['Mr.', 'Miss.', 'Mrs.', 'Master.'] else 'Rare')
    return dataframe_in


# Function to extract the family from the "Name" feature.

# In[ ]:


def get_family_name(dataframe_in):
    dataframe_in['FamilyName'] = dataframe_in['Name'].apply(lambda X: X.split(',')[0])
    return dataframe_in


# In[ ]:


def get_cabin_letter(dataframe_in):
    def first_letter_if_exists(str_in):
        if pd.isnull(str_in):
            return '?'
        return str_in[0]
    dataframe_in['Cabin'] = dataframe_in['Cabin'].apply(first_letter_if_exists)
    return dataframe_in


# In[ ]:


def get_cabins_per_family(dataframe_in):
    dataframe_in = get_family_name(dataframe_in)
    dict_cabins_per_family = dict()
    for current_family_name in dataframe_in.FamilyName.unique().tolist():
    
        filter_family_name = (dataframe_in['FamilyName'] == current_family_name)
        filter_known_cabin = (dataframe_in['Cabin'] != '?')
        listCabinsFromFamily = dataframe_in.loc[(filter_family_name) & (filter_known_cabin)].Cabin.unique().tolist()
    
        if len(listCabinsFromFamily) > 0:
            max_v, mode = 0, None
            for curr_cabin, v in Counter(listCabinsFromFamily).items():
                if v > max_v:
                    max_v, mode = v, curr_cabin
            dict_cabins_per_family[current_family_name] = mode
    
    return dict_cabins_per_family


# Functions to relate each family to a cabin. First we define the function to be applied to each row:

# In[ ]:


def get_family_cabin_per_row(df_row, dict_family_in):
    dict_cabins_per_family = dict_family_in
    
    if df_row.FamilyName in dict_cabins_per_family and df_row.Cabin == '?':
        out = dict_cabins_per_family[df_row.FamilyName]
    else:
        out = df_row.Cabin
    return out


# Then apply the function over all the dataframe...

# In[ ]:


def get_family_cabin(df, dict_family_in):
    
    df['Cabin'] = df.apply(lambda X: get_family_cabin_per_row(X, dict_family_in), axis = 1)
    df['Cabin'] = df['Cabin'].fillna('?')
    
    return df


# Separe ticket prefixes

# In[ ]:


def get_ticket_prefix(dataframe_in):
    dataframe_in['Ticket'] = dataframe_in['Ticket'].apply(lambda X: X.split(' ')[0] if len(X.split(' ')) > 1 else '?')
    return dataframe_in


# Get informations about the family structure of each passenger

# In[ ]:


def get_family_info(df_in):
    df_in['FamilyMembers'] = df_in['Parch'] + df_in['SibSp'] + 1
    df_in['Is_Mother'] = np.where((df_in.Title=='Mrs.') & (df_in.Parch >0), 1, 0)
    return df_in


# ## 2.A.2. Missing Data Transformations

# We saw that all the people with a "Master" title were really young. Then, the variance of that group was small and it was possible to separe the algorithm in two partes: (1) filling the missing age of the "Masters" (which can be done easily by simply filling the NA's with the median or average value of the master people) and (2) filling the age missing values of the "Non Masters", which can be done with the same procedure.
# 
# Filling the ages with means and medians to the second case can lead us many mistakes since the variance is not small to that group. That's why we propose to use a clustering algorithm to separe the people in different groups.

# In[ ]:


def fill_age_from_masters(df, strategy_in = 'median'):
    is_master = (df['Title'] == 'Master.')
    imp = SimpleImputer(missing_values = np.nan, strategy = strategy_in)
    df.loc[is_master, 'Age'] = imp.fit_transform(df.loc[is_master][['Age']])
    return df


# In[ ]:


def fill_age_from_non_masters(df, strategy_in = 'median'):
    is_not_master = (df['Title'] != 'Master.')
    imp = SimpleImputer(missing_values = np.nan, strategy = strategy_in)
    df.loc[is_not_master, 'Age'] = imp.fit_transform(df.loc[is_not_master][['Age']])
    return df


# # 2.B. Transformations Used in Pipelines

# ## 2.B.1. Transform Significant Data
# A transformation that represents the extraction of significant variables, defined during the feature exploration step.

# In[ ]:


class TransformerSignificantData(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        return
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        
        out = get_cabin_letter(X)
        out = get_ticket_prefix(out)
        out = get_title(out)
        out = get_family_info(out)
        out = classify_title(out)
        out = get_family_name(out)
        dict_cabins_per_family = get_cabins_per_family(out)
        out = get_family_cabin(out, dict_cabins_per_family)
        
        return out


# ## 2.B.2. Transformer Dummify
# As the name suggests, it's the transformation responsible to the "dummification" of variables by using one-hot encoding and removind one category per feature hot-encoded (to avoid the curse of dimentionality).

# In[ ]:


class TransformerDummify(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        return
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        
        columns_to_dummify = ['Sex', 'Cabin', 'Embarked', 'Title', 'Ticket']
        useless_columns = ['PassengerId', 'Name', 'FamilyName']
        dim_redundant_cols = ['Sex_male', 'Cabin_?', 'Embarked_S', 'Title_Rare', 'Ticket_?']
        
        out_dummies = pd.get_dummies(X[columns_to_dummify], prefix = columns_to_dummify)
        
        out = pd.concat([X, out_dummies], axis = 1)
        out = out.drop(useless_columns + columns_to_dummify + dim_redundant_cols, axis = 1)
        
        return out


# ## 2.B.3. Transformer Missing Data
# Transformer that inputs all the missing data. A special attention was given to the "Age" feature and the procedure of label propagation over DBSCAN clusters was used:

# In[ ]:


class TransformerMissingData(BaseEstimator, TransformerMixin):
    
    def __init__(self, missing_age_masters_strategy='mean', missing_age_non_masters_strategy='cluster'):
        self.missing_age_masters_strategy = missing_age_masters_strategy
        self.missing_age_non_masters_strategy = missing_age_non_masters_strategy
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):

        is_not_master = (X['Title_Master.'] == 0)
        is_master = (X['Title_Master.'] == 1)
        
        imp_non_master = SimpleImputer(missing_values = np.nan, 
                                       strategy = self.missing_age_non_masters_strategy)
        
        imp_master = SimpleImputer(missing_values = np.nan, 
                                   strategy = self.missing_age_masters_strategy)
        
        imp_fare = SimpleImputer(missing_values = np.nan,
                                 strategy = 'mean')
        
        X.loc[is_master, 'Age'] = imp_master.fit_transform(X.loc[is_master, ['Age']])
        X.loc[:, 'Fare'] = imp_fare.fit_transform(X.loc[:, ['Fare']])
        
        if self.missing_age_non_masters_strategy in ['mean', 'median']:
            X.loc[is_not_master, 'Age'] = imp_non_master.fit_transform(X.loc[is_not_master, ['Age']])
            
        elif self.missing_age_non_masters_strategy == 'cluster':
            
            X_without_age = X.drop(['Age', 'Survived'], axis = 1)
            pca_fit_vec = PCA(n_components = 3).fit_transform(X_without_age)
            
            X['ClusterLabel'] = DBSCAN(eps = 0.7, min_samples = 20).fit_predict(pca_fit_vec)
            dict_means_per_cluster = dict()
            for cluster_label in X['ClusterLabel'].tolist():
                dict_means_per_cluster[cluster_label] = X.loc[X['ClusterLabel'] == cluster_label, :]['Age'].mean()
            
            age_list = X['Age'].tolist()
            cluster_list = X['ClusterLabel'].tolist()
            for i, curr_age in enumerate(age_list):
                if np.isnan(curr_age):
                    age_list[i] = dict_means_per_cluster[cluster_label]
            X['Age'] = age_list
            X.drop('ClusterLabel', axis=1, inplace=True)
                
        return X


# And after all these transformations, we just normalize the data:

# In[ ]:


class TransformerNormalize(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        return
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        
        out = X
        out.loc[:, ['Pclass']] = MinMaxScaler().fit_transform(out[['Pclass']])
        out.loc[:, ['Age', 'Fare']] = StandardScaler().fit_transform(out[['Age', 'Fare']])
        
        return out


# # 2.C. Pipelines

# ## 2.C.1. Feature Engineering Pipeline
# The pipeline where the data is preprocessed according to the transformations exposed in the previous step. The hyper-parameters of those transformations can, with this instance, be tunned:

# In[ ]:


feature_engineering_pipeline = Pipeline([
    ('prepare', TransformerSignificantData()),
    ('dummify', TransformerDummify()),
    ('normalize', TransformerNormalize()),
    ('missing', TransformerMissingData())
])


# ## 2.C.2. Train and Test Splitting Procedure

# Let's take the code to split the train / validation set and the testing set as a function. Remember that the training and validating occur in the same set because we are using a Cross Validation K-Fold procedure:

# In[ ]:


def split_cv_test(df_all_processed):
    df_ans = df_all_processed[df_all_processed['Survived'].isnull()]
    df_ans = df_ans.loc[:, df_ans.columns != 'Survived']
    df_cv_and_test = df_all_processed[df_all_processed['Survived'].notnull()]
    df_cv_and_test_X = df_cv_and_test.drop(['Survived'], axis = 1)
    df_cv_and_test_Y = df_cv_and_test['Survived']
    df_cv_X, df_test_X, df_cv_Y, df_test_Y = train_test_split(df_cv_and_test_X,
                                                              df_cv_and_test_Y,
                                                              test_size = 0.1)
    return df_cv_X, df_cv_Y, df_test_X, df_test_Y, df_ans, df_cv_and_test_X, df_cv_and_test_Y


# # 3. Starting the Game - Titanic - A NN Approach
# Applying Preprocessing Pipeline
# Let's read and pre-process our data just like before:

# In[ ]:


df_in = [pd.read_csv(data_dir + '/train.csv'), pd.read_csv(data_dir + '/test.csv')]
df_in = pd.concat(df_in, ignore_index = True, sort = False)
df_in.head()


# In[ ]:


df_in_norm = feature_engineering_pipeline.transform(df_in)
print(df_in_norm.describe())


# # 3.A. Defyning Neural Network Model to Apply
# To create our neural network model, we will use the "Keras Library". The idea here is to test different architectures, enhancing the hyperparameters little by little. It's impossible to GridSearch over all possible neural networks that we can apply. We are going to tune the number of hidden layers and the activation function for three different models:
# 
# * An ordinary Feed Foward Neural Network
# * A Neural Network using a standard DropOut function
# * A Neural Network using an AlphaDropOut function
# 
# The DropOut's will employ a DropOut rate of $50\%$.

# In[ ]:


n_col = len(df_in_norm.columns)
def nn_layers_and_activation(n_hidden_layers, activation_func, dropout=None):
    
    in_layer_args = dict(
        units = n_col - 1,
        activation = activation_func,
        input_dim = n_col - 1,
        kernel_initializer = 'lecun_uniform'
    )
    
    out_layer_args = dict(
        units = 1,
        activation = 'sigmoid',
        input_dim = n_col - 1,
        kernel_initializer = 'lecun_uniform'
    )
    
    model_args = dict(
        loss = 'binary_crossentropy',
        optimizer = 'nadam',
        metrics = ['binary_accuracy']
    )
    
    SelectedDropout = Dropout if dropout == 'Dropout' else AlphaDropout
    out_model = Sequential()
    for i in range(0, n_hidden_layers + 1):
        out_model.add(Dense(**in_layer_args))
        if dropout is not None:
            out_model.add(SelectedDropout(rate=0.5))
        
    out_model.add(Dense(**out_layer_args))
    out_model.compile(**model_args)
    return out_model
    

test_nn = nn_layers_and_activation(1, 'relu')
print(test_nn)


# Notice that:
# * We are defining the number of hidden layers, not the number of total layers ($N_{Total Layers} = N_{Hidden Layers} + 2$)
# * Since the output can get values equal to zero or one, we are defining the output function as a simple sigmoid (generally, the output function is defined by the type of problem - regression or classification - and by the number of outputs - for problems with multiple classes, we use the softmax function and for binary classification, it's common to employ the logistic function as we have done here).
# Let's try to tune our hyperparameters, testing different numbers of $N_{Hidden Layers}$ and creating a pipeline to realize our GridSearch. But, before, let's separe out training and test sets:

# In[ ]:


df_cv_X, df_cv_Y, df_test_X, df_test_Y, df_ans, df_cv_and_test_X, df_cv_and_test_Y = split_cv_test(df_in_norm)
print(df_cv_X.head())


# First, let's try to find the optimal activation function using common values to all the others hyperparameters.

# In[ ]:


wrapped_tune_layers_and_activation = KerasClassifier(build_fn=nn_layers_and_activation, 
                                                     batch_size=64, 
                                                     verbose=0)

nn_pipeline = Pipeline([
    ('tune_layers_and_activation', wrapped_tune_layers_and_activation)
])

def gen_scheduler(eta=0.1, epoch0=100.0):
    def exp_decay_fn(epoch, lr):
        return lr * eta ** (epoch / epoch0)
    return LearningRateScheduler(exp_decay_fn)

dict_params_standard = dict(
    tune_layers_and_activation__n_hidden_layers = [1, 3, 5, 10],
    tune_layers_and_activation__epochs = [100],
    tune_layers_and_activation__activation_func = ['sigmoid', 'relu', 'tanh', 'selu'],
    tune_layers_and_activation__dropout = [None],
    tune_layers_and_activation__callbacks = [[gen_scheduler(epoch0=X), 
                                              EarlyStopping(patience=10, restore_best_weights=False)]\
                                              for X in [1000]]
)
dict_params_dropout = dict(
    tune_layers_and_activation__n_hidden_layers = [1, 3, 5, 10],
    tune_layers_and_activation__epochs = [100],
    tune_layers_and_activation__activation_func = ['sigmoid', 'relu', 'tanh', 'selu'],
    tune_layers_and_activation__dropout = ['Dropout'],
    tune_layers_and_activation__callbacks = [[gen_scheduler(epoch0=X), 
                                              EarlyStopping(patience=10, restore_best_weights=False)]\
                                              for X in [1000]]
)
dict_params_alphadropout = dict(
    tune_layers_and_activation__n_hidden_layers = [1, 3, 5, 10],
    tune_layers_and_activation__epochs = [100],
    tune_layers_and_activation__activation_func = ['sigmoid', 'relu', 'tanh', 'selu'],
    tune_layers_and_activation__dropout = ['AlphaDropout'],
    tune_layers_and_activation__callbacks = [[gen_scheduler(epoch0=X), 
                                              EarlyStopping(patience=10, restore_best_weights=False)]\
                                              for X in [1000]]
)


# In[ ]:


cv_pipeline_standard = GridSearchCV(estimator=nn_pipeline, param_grid=dict_params_standard, 
                                    scoring='accuracy', cv=3, verbose=10, n_jobs=-1)

cv_pipeline_dropout = GridSearchCV(estimator=nn_pipeline, param_grid=dict_params_dropout, 
                                   scoring='accuracy', cv=3, verbose=10, n_jobs=-1)

cv_pipeline_alphadropout = GridSearchCV(estimator=nn_pipeline, param_grid=dict_params_alphadropout, 
                                        scoring='accuracy', cv=3, verbose=10, n_jobs=-1)

print('Fitting model without Dropout')
cv_pipeline_standard.fit(df_cv_X, df_cv_Y)
print('Fitting model with Dropout')
cv_pipeline_dropout.fit(df_cv_X, df_cv_Y)
print('Fitting model with AlphaDropout')
cv_pipeline_alphadropout.fit(df_cv_X, df_cv_Y)


# After tunning the parameters, we have just saved all variables from our session. Now, it's time to check out the optimal neural network we just got.

# In[ ]:


for i, cv_pipeline in enumerate([cv_pipeline_standard, cv_pipeline_dropout, cv_pipeline_alphadropout]):
    list_models = ['No Dropout', 'Std Dropout', 'AlphaDropout']
    
    print('----')
    print(list_models[i] + '- Results:')
    print('----')
    print('Paramaters:')
    dict_params = cv_pipeline.best_estimator_.named_steps['tune_layers_and_activation'].get_params()
    print('Act. Fn: ' + dict_params['activation_func'])
    print('Model: ' + (dict_params['dropout'] if dict_params['dropout'] is not None else 'No Dropout'))
    print('N. Hidden Layers: ' + str(dict_params['n_hidden_layers']))
    
    print('Optimal Exp. Decay:')
    dict_params = cv_pipeline.best_estimator_.named_steps['tune_layers_and_activation'].get_params()
    opt_scheduler = dict_params['callbacks'][0]
    opt_scheduler_fn = opt_scheduler.__dict__['schedule']
    opt_epoch0 = 1 / (math.log10(opt_scheduler_fn(0, 1)) - math.log10(opt_scheduler_fn(1, 1)))
    print(f'Epoch0 = %f' % opt_epoch0)
    
    print('Best Score:')
    print('{0:.0%}'.format(cv_pipeline.best_score_))
    print('\n')


# Let's compare the result along each fold for each one of the models in a BoxPlot:

# In[ ]:


cv_pipeline_standard.cv_results_['split0_test_score']

def get_cv_compared_results(cv_obj_list, cv_obj_names):
    
    obj_cv0 = cv_obj_list[0]
    n_splits = obj_cv0.n_splits_
    param_list = obj_cv0.param_grid.keys()
    dict_df_cv = dict()
    
    for param in param_list:
        dict_df_cv[param.split('__')[1]] = []
        dict_df_cv['OPT__' + param.split('__')[1]] = []
        
    dict_df_cv['Score'] = []
    dict_df_cv['Split'] = []
    dict_df_cv['Model'] = []
    
    df_compared_results = pd.DataFrame(dict_df_cv)
    for idx, obj_cv in enumerate(cv_obj_list):
        
        obj_name = cv_obj_names[idx]
        opt_dict = obj_cv.best_estimator_.named_steps['tune_layers_and_activation'].get_params()
        
        for i, param in enumerate(obj_cv.cv_results_['params']):
            for j in range(0, n_splits):
                
                dict_new_row = dict()
                    
                dict_new_row['Score'] = [obj_cv.cv_results_['split' + str(j) + '_test_score'][i]]
                dict_new_row['Split'] = [j]
                dict_new_row['Model'] = [obj_name]
                
                for param_id, param_val in param.items():
                    
                    param_name = param_id.split('__')[1]
                    
                    dict_new_row[param_name] = [param_val]
                    dict_new_row['OPT__' + param_name] = [opt_dict[param_name]]
                    
            df_compared_results = pd.concat([df_compared_results, pd.DataFrame(dict_new_row)], axis=0, ignore_index=True)

    return df_compared_results
        
    
cv_obj_list = [cv_pipeline_standard, cv_pipeline_dropout, cv_pipeline_alphadropout]
cv_names_list = ['No Dropout', 'Simple Dropout', 'AlphaDropout']

df_compared_results = get_cv_compared_results(cv_obj_list, cv_names_list)
df_compared_results.head()


# In[ ]:


act_fn_used = df_compared_results['activation_func'] == df_compared_results['OPT__activation_func']
df_compared_results_fn_filtered = df_compared_results.loc[act_fn_used, :]
df_compared_results_fn_filtered.head()


# In[ ]:


df_compared_results_fn_filtered.dtypes


# In[ ]:


mpl.rcParams['figure.figsize'] = [10, 10]
sns.set_style("darkgrid")
df_compared_results_fn_filtered['n_hidden_layers_jitter'] = df_compared_results_fn_filtered['n_hidden_layers'].apply(lambda X: X + 0.1 * np.random.normal())
sns.scatterplot(data=df_compared_results_fn_filtered, x='n_hidden_layers_jitter', y='Score', hue='Model',  linewidth=1, s=1000, alpha=0.4, edgecolor='k')


# # 3.B. Prepare a TensorBoard and inspect the Neural Networks in a Separated Validation Set
# The Cross-Validation procedure executes the hyper-parameter tunning over different splits of the same set. For that reason, to inspect the generalization power of our models we are going to look out of that set and inspect the Learning Curves of the $3$ optimal models (without Dropping Out, with DropOut and with AlphaDropout) using the TensorBoard tool. The model fitting will be executed over all the Cross Validation data to inspect the results when we increase the number of samples for each Neural Network.

# In[ ]:


df_submission = pd.read_csv(data_dir + '/test.csv')
df_submission = df_submission.loc[:, ['PassengerId', 'Survived']]

for i, cv_obj in enumerate([cv_pipeline_standard, cv_pipeline_dropout, cv_pipeline_alphadropout]):
    tensor_board_callback = [TensorBoard('./logs/Model-' + str(i + 1), histogram_freq=0, write_graph=False)]
    cbcks_list = cv_obj.best_estimator_.steps[0][1].__dict__['sk_params']['callbacks'] + tensor_board_callback
    cv_obj.best_estimator_.fit(df_cv_X, df_cv_Y, 
                               tune_layers_and_activation__callbacks=cbcks_list,
                               tune_layers_and_activation__validation_data=(df_test_X, df_test_Y))
    
    cv_obj.best_estimator_.fit(df_cv_and_test_X, df_cv_and_test_Y, 
                               tune_layers_and_activation__callbacks=cv_obj.best_estimator_.steps[0][1].__dict__['sk_params']['callbacks'])

    
    df_submission['Survived'] = cv_obj.best_estimator_.predict(df_ans)
    df_submission['Survived'] = df_submission['Survived'].apply(int)
    df_submission.to_csv('Model-' + str(i + 1) + '-submission.csv', index=False)


# # 4. Could We Perform Better? - Testing the Monte Carlo DropOut
# Maybe. When we talk about DropOut, we can have an immediate alternative to improve the model even more: the Monte Carlo Dropout is a technique that can help us to achieve better results in dropping out neural networks. After training the Neural Network using the Dropout, we also drop out some weights randomly during the test $N$ times, obtaining $N$ different models. Then, we blend them in a Soft Voting Model.
# 
# It can be proven that it's equivalent to realize a Bayesian Simulation on our Neural Networks, which is a formal justificative to get almost aways a better result :).

# In[ ]:


with K.learning_phase_scope(1):
    np_mc_probs_per_model = np.stack(cv_pipeline_dropout.best_estimator_.predict(df_ans) for _ in range(100))
    
np_mc_probs = np.mean(np_mc_probs_per_model, axis=0)
np_mc_ans = np.round(np_mc_probs)

df_submission['Survived'] = np_mc_ans
df_submission['Survived'] = df_submission['Survived'].apply(int)
df_submission = df_submission.loc[:, ['PassengerId', 'Survived']]
df_submission.to_csv('Model-MCDropOut-submission.csv', index=False)


# # 5. A Final Try: Using the Results of Part 1
# The best model found during the validation step (when looking at the TensorBoard results) were:
# * The ordinary feed foward neural network using 5 layers and
# * The simple DropOut with 3 layers using the $tanh(x)$ activation function
# The AlphaDropout Neural Network is generally combined with the $selu$ activation function, which make sense with the found results but it's commonly more suitable to Deep Learning problems (which is not our case sice it was not necessary to increase our hidden layers number to a huge quantity, like 100 or even 1000).
# 
# So, our last try will be to develop a Wide and Deep network using the DropOut model [which produced a slightly better result]: the "Deep" part will use 5 layers or 3 layers (depending on the model) and we will add a "Wide" part using the results found in each one of the individual models used in the last notebook (check [the Part 1](https://www.kaggle.com/guidant/corvus-part-1-top-15-combining-5-models#5.-Final-Submission) for more details). 
# 
# Training those models with the optimal parameters found on the previous Notebook...

# ## Previous Optimal Results:
# 
# ---
# * Random Forest
#     * n_estimators: 85 -- FROM [85, 86, 87, 88, 89]
#     * max_depth: 8 -- FROM [3, 4, 5, 6, 7, 8, 9, 10]
# 
# ---
# * SVM
#     * C: 7.0 -- FROM [6.0, 7.0, 8.0]
#     * kernel: rbf -- FROM ['linear', 'poly', 'rbf', 'sigmoid']
#     * gamma: auto -- FROM ['auto', 'scale']
#     * probability: True -- FROM [True]
# 
# ---
# * Logistic Regression
#     * C: 1.0 -- FROM [0.1, 1.0, 10.0]
#     * solver: lbfgs -- FROM ['lbfgs']
# 
# ---
# * Ada Boost
#     * n_estimators: 54 -- FROM [54, 55, 56]
#     * learning_rate: 0.35 -- FROM [0.35, 0.4, 0.45]
# 
# ---
# * XGBTree
#     * eta: 0 -- [0, 1e-05]
#     * gamma: 0.05 -- [0.001, 0.05, 0.1]
#     * max_depth: 3 -- [3, 4, 5]
#     * probability: True -- FROM [True]
# 
# ---

# 

# Let's create dictionaries with the optimal results already found in the previous notebook.

# In[ ]:


dict_rf = dict(
    n_estimators = 85,
    max_depth = 8
)

dict_svm = dict(
    C = 7.0,
    kernel = 'rbf',
    gamma = 'auto',
    probability = True
)

dict_logit = dict(
    C = 1.0,
    solver = 'lbfgs'
)

dict_adab = dict(
    n_estimators = 54,
    learning_rate = 0.35
)

dict_xgboost = dict(
    gamma =0.05,
    max_depth = 3,
    probability = True
)


# Training the model, just like in the first part...

# In[ ]:


rf_model = RandomForestClassifier(**dict_rf).fit(df_cv_and_test_X, df_cv_and_test_Y)
svm_model = SVC(**dict_svm).fit(df_cv_and_test_X, df_cv_and_test_Y)
logit_model = LogisticRegression(**dict_logit).fit(df_cv_and_test_X, df_cv_and_test_Y)
adab_model = AdaBoostClassifier(**dict_adab).fit(df_cv_and_test_X, df_cv_and_test_Y)
xgboost_model = XGBClassifier(**dict_xgboost).fit(df_cv_and_test_X, df_cv_and_test_Y)


# Adding the models' outputs as features to the test and validation sets. The new features will be used in the wide layer of the NN while the other features will be present on the "deep" part.

# In[ ]:


dict_train_X_wide = dict()
dict_valid_X_wide = dict()
dict_ans_wide = dict()

df_list = [df_cv_X, df_test_X, df_ans]
simple_models_names = ['RF', 'SVM', 'LOGISTIC', 'ADABOOST', 'XGBTREE']
for i, mod in enumerate([rf_model, svm_model, logit_model, adab_model, xgboost_model]):
    for j, curr_dict in enumerate([dict_train_X_wide, dict_valid_X_wide, dict_ans_wide]):
        curr_dict[simple_models_names[i]] = [X[0] for X in mod.predict_proba(df_list[j])[:, [0]]]
        
df_cv_X_wide = pd.DataFrame(dict_train_X_wide)
df_test_X_wide = pd.DataFrame(dict_valid_X_wide)
df_ans_wide = pd.DataFrame(dict_ans_wide)


# Time to create the wide and deep model. Previously, we have used the "Sequencial Keras API". This time, we will use the "Functional API":

# In[ ]:


n_wide = len(simple_models_names)
n_deep = len(df_cv_X.columns)

deep_input = Input(shape=(n_deep,))
wide_input = Input(shape=(n_wide,))

opt_n_hidden = cv_pipeline_dropout.best_estimator_.named_steps['tune_layers_and_activation'].get_params()['n_hidden_layers']
opt_act_fn = cv_pipeline_dropout.best_estimator_.named_steps['tune_layers_and_activation'].get_params()['activation_func']

if opt_n_hidden > 0:
    hidden = Dense(n_deep, activation=opt_act_fn, kernel_initializer='lecun_uniform')(deep_input)
    hidden = Dropout(rate=0.5)(hidden)
    for _ in range(opt_n_hidden):
        hidden = Dense(n_deep, activation=opt_act_fn, kernel_initializer='lecun_uniform')(hidden)
        hidden = Dropout(rate=0.5)(hidden)
        
output_layer = Dense(1, activation='sigmoid',  kernel_initializer='lecun_uniform')(concatenate([wide_input, hidden]))

model = Model(inputs=[wide_input, deep_input], output=output_layer)
model.compile(optimizer='nadam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

print(model.summary())


# Let's print the model graph...

# In[ ]:


plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
display(Image(filename='model_plot.png'))


# Finally, let's check the training curves of the model using the TensorFlow backend:

# In[ ]:


tb_wide_and_deep = TensorBoard(log_dir='./logs/wide_and_deep', histogram_freq=1, write_graph=False, write_images=False)
cbcks_list = [tb_wide_and_deep, EarlyStopping(patience=10, restore_best_weights=False)]
model.fit([df_cv_X_wide, df_cv_X], df_cv_Y, validation_data=([df_test_X_wide, df_test_X], df_test_Y), epochs=100, callbacks=cbcks_list, verbose=0)


# Seems to produce similar or even better results. Let's save this model then...of course, we will use all data this time.

# In[ ]:


cbcks_list = [EarlyStopping(patience=10, restore_best_weights=False)]
model.fit([pd.concat([df_cv_X_wide, df_test_X_wide], axis=0, ignore_index=True), 
           pd.concat([df_cv_X, df_test_X], axis=0, ignore_index=True)], 
          pd.concat([df_cv_Y, df_test_Y], axis=0, ignore_index=True), epochs=100, callbacks=cbcks_list, verbose=0)


# In[ ]:


df_submission['Survived'] = model.predict([df_ans_wide, df_ans])
df_submission['Survived'] = df_submission['Survived'].apply(int)
df_submission.to_csv('Model-DropoutWideAndDeep-submission.csv', index=False)


# In[ ]:


FileLinks('.')

