#!/usr/bin/env python
# coding: utf-8

# ### Full credits to the author [Chris Deotte](https://www.kaggle.com/cdeotte) of the [Original notebook](https://www.kaggle.com/cdeotte/one-feature-model-0-930)
# 
# #### I have reorganised and refactored the code for my curiosity and learnings. Added console logs, and re-wrote datastructure to understand how the splits and batching is occuring. Also added two differentCV methods **KFold** and **NestedCV** methods, these are ideal for using with Timeseries-like problems. The focus of the changes in the notebook was to focus a bit more on the training aspects (CV methods), the original notebook covered a lot on analysis and cleaning, please refer to it for those goodies.
# 
# You can read more about **NestedCV** here:
# - https://www.elderresearch.com/blog/nested-cross-validation
# - https://towardsdatascience.com/time-series-nested-cross-validation-76adba623eb9
# - https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html?highlight=nested%20cross%20validation
# 
# Thanks for Marcus for the clean datasets (drift and noise removed), and also [Carlo Lepelaars](https://www.kaggle.com/carlolepelaars) for sharing with me the **54 features**, that were generated fron the `signal` column as part of a Feature Engineering step after cleaning the datasets and before building the model.
# 
# ### This notebook has been forked from [One Feature Model - clean datasets (refactored)](https://www.kaggle.com/neomatrix369/one-feature-model-clean-datasets-refactored/), find other such refactored notebooks [here](https://www.kaggle.com/c/liverpool-ion-switching/discussion/153653).

# # One Feature Model Scores LB 0.930!
# In this notebook, we will explore the Kaggle Ion Comp data and explore a one feature model. The LB result of 0.930 is enlightening.
# 
# Here we manually remove signal drift. Note that it is better to use machine learning to remove drift, but doing it by hand once allows us to understand its nature and build better models later.

# # Load Libraries and Data

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score
import graphviz
from sklearn import tree
from sklearn.model_selection import KFold


# In[ ]:


# prettify plots
plt.rcParams['figure.figsize'] = [20.0, 5.0]


# In[ ]:


get_ipython().system('ls ../input/**')


# In[ ]:


get_ipython().run_cell_magic('time', '', "test = pd.read_csv('../input/54-features-datasets-clean-drift-noise-free/54-features-from-test_clean_removed_drift_noise.csv')")


# In[ ]:


test = test.fillna(0.0)


# In[ ]:


get_ipython().run_cell_magic('time', '', "train = pd.read_csv('../input/54-features-datasets-clean-drift-noise-free/54-features-from-train_clean_removed_drift_noise.csv')")


# In[ ]:


train = train.fillna(0.0)


# In[ ]:


print("train", train.shape)
print("test", test.shape)


# In[ ]:


train.head()


# In[ ]:


test.head()


# # Description of Data
# The training data is recordings in time. At each 10,000th of a second, the strength of the signal was recorded and the number of ion channels open was recorded. It is our task to build a model that predicts the number of open channels from signal at each time step. Furthermore we are told that the data was recorded in batches of 50 seconds. Therefore each 500,000 rows is one batch. The training data contains 10 batches and the test data contains 4 batches. Let's display the number of open channels and signal strength together for each training batch.

# In[ ]:


res = 1000
batch_size=500000
sub_sample_size = batch_size/5
margin=200000

def plot_data(column, column_name):
    plt.figure(figsize=(20,5))
    plt.plot(range(0, column.shape[0], res), column[0::res])
    for i in range(11): plt.plot([i*batch_size,i*batch_size],[-5,12.5],'r')
    for j in range(10): plt.text(j*batch_size+margin,10,str(j+1),size=20)
    plt.xlabel('Row',size=16); plt.ylabel(column_name,size=16); 
    plt.title(f'Training Data {column_name} - 10 batches',size=20)
    plt.show()


# In[ ]:


plot_data(train.signal, 'Signal')


# In[ ]:


plot_data(train.open_channels, 'Open Channels')


# ### Feature columns

# In[ ]:


columns_to_keep_but_are_not_features = [
    'time',
    'batch_index',
    'signal',
    'open_channels'
]

feature_cols = list(set(train.columns) - set(columns_to_keep_but_are_not_features))
print(f"{len(feature_cols)} features: {feature_cols}")


# ### NestedCV implementation

# Below code cells show the smaller building blocks that make up the NestedCV generator (the data structure of the output created by `generate_nested_folds_batch_ranges()` can be wrapped with `enumerate()` and used as an iterator and swapped with other CV methods i.e. `KFold`, etc...

# In[ ]:


def chain_to_previous_range(indices: tuple, folds: int):
    return indices[1] + folds, indices[1] + folds
    
def generate_range_of_indices(indices: tuple, fold_size: int, max_allowed_value: int):
    start = fold_size * indices[0]
    end = fold_size * (indices[1] + 1)
    if abs(max_allowed_value - end) <= 3:
        end = max_allowed_value

    return np.array(range(start, end))


# In[ ]:


def generate_nested_folds_batch_ranges(total_folds_size: int, num_of_training_folds: int = 3,
                                       num_of_validation_folds: int = 1, num_of_test_folds: int = 0):
    total_folds_size = int(total_folds_size)
    total_folds = num_of_training_folds + num_of_validation_folds + num_of_test_folds
    each_fold_size = int(round(total_folds_size / total_folds))

    nested_folds_indices = []
    min_training_index = 0
    max_training_index = total_folds - num_of_validation_folds - num_of_test_folds
    for max_training_index_this_fold in range(min_training_index, max_training_index):
        training_indices = (min_training_index, max_training_index_this_fold)
        validation_indices = chain_to_previous_range(training_indices, num_of_validation_folds)
        test_indices = (0, 0)
        if num_of_test_folds > 0:
            test_indices = chain_to_previous_range(validation_indices, num_of_test_folds)
        nested_folds_indices.append([training_indices, validation_indices, test_indices])

    nested_batch_indices = []
    for each_nested_fold_indices in nested_folds_indices:
        training_indices = each_nested_fold_indices[0]
        validation_indices = each_nested_fold_indices[1]
        test_indices = each_nested_fold_indices[2]

        indices = [
            generate_range_of_indices(training_indices, each_fold_size, total_folds_size),
            generate_range_of_indices(validation_indices, each_fold_size, total_folds_size),
        ]
        if num_of_test_folds > 0:
            indices.append(
                generate_range_of_indices(test_indices, each_fold_size, total_folds_size),
            )
        nested_batch_indices.append(indices)

    return nested_batch_indices


# #### Example usage of the NestCV generator function

# In[ ]:


print("training,      validation,        test")
generate_nested_folds_batch_ranges(train.shape[0], 5)


# In[ ]:


print("training,      validation,        test")
generate_nested_folds_batch_ranges(train.shape[0], 5, 1, 1)


# ### Define the CV to be used for training: KFold and NestedCV

# In[ ]:


training_folds = 5


# In[ ]:


def get_kfold_enumerator(dataset: pd.DataFrame, folds: int = training_folds):
    return enumerate(KFold(n_splits=folds).split(dataset))


# In[ ]:


def get_nestedcv_enumerator(dataset: pd.DataFrame, folds: int = training_folds):
    return enumerate(generate_nested_folds_batch_ranges(dataset.shape[0], folds))


# #### _Removed cells about **Reflection**, **Correlation Between Signal and Open Channels**, Test Data, and Remove Training Data Drift, see [original notebook](https://www.kaggle.com/cdeotte/one-feature-model-0-930) by [Chris Deotte](https://www.kaggle.com/cdeotte/)._

# # Make Five Simple Models
# We will make one model for each different type of signal we observed above.

# In[ ]:


def train_with_cross_validation(params, model, X_train_, y_train_, cv_enumerator=get_nestedcv_enumerator):
    total_f1_macro_score = 0.0
    models = []
    best_model = None
    best_f1_macro_score = 0.0
    
    for fold_index, (training_index, validation_index) in cv_enumerator(X_train_):
        X_training_set = X_train_[training_index]
        y_training_set = y_train_[training_index]
        X_validation_set = X_train_[validation_index]
        y_validation_set = y_train_[validation_index]
        model = model.fit(X_training_set, y_training_set)
        models.append(model)
        predictions = model.predict(X_validation_set)
        f1_macro_score = f1_score(y_validation_set, predictions, average='macro')
        if best_f1_macro_score < f1_macro_score:
            best_f1_macro_score = f1_macro_score
            best_model = model
        print(f'fold {fold_index + 1}: macro f1 validation score: {f1_macro_score}, best macro f1 validation score: {best_f1_macro_score}')
        total_f1_macro_score += f1_macro_score

    return models, best_model, total_f1_macro_score/training_folds

def train_model_by_batch(train_df, feature_cols_, first_batch, second_batch, model_type, 
                         class_names=['0', '1'], params={'max_depth':1}, cv_enumerator=get_nestedcv_enumerator):
    a = batch_size * (first_batch - 1); b = (batch_size * first_batch); 
    c = batch_size * (second_batch - 1); d = (batch_size * second_batch)
    left_batch = train_df[a:b];     right_batch = train_df[a:b];

    X_train = np.concatenate([left_batch[feature_cols_].values, right_batch[feature_cols_].values]).reshape((-1,len(feature_cols)))
    y_train = np.concatenate([left_batch.open_channels.values, left_batch.open_channels.values]).reshape((-1,1))
    
    print(f'Training model {model_type} channel')
    model = tree.DecisionTreeClassifier(**params)
    models, best_model, f1_macro_score = train_with_cross_validation(params, model, X_train, y_train, cv_enumerator=cv_enumerator)
    print(f'model {model_type}, average macro f1 validation score = {f1_macro_score}')
    
    tree_graph = tree.export_graphviz(best_model, out_file=None, max_depth = 10, impurity = False, 
                                      feature_names = feature_cols_, class_names = class_names, rounded = True, filled= True)
    return models, f1_macro_score, graphviz.Source(tree_graph) 


# ### Define the Macro F1 function to be used during training

# In[ ]:


import numpy as np
from sklearn.metrics import f1_score
from tensorflow.keras.callbacks import Callback


def macro_f1(y_true, y_pred):
    """
    The Macro F1 metric used in this competition
    :param y_true: The ground truth labels given in the dataset
    :param y_pred: Our predictions
    :return: The Macro F1 Score
    """
    return f1_score(y_true, y_pred, average="macro", labels=np.unique(y_true))


# In[ ]:


nestedcv_f1_macro_scores = []
kfold_f1_macro_scores = []


# ## 1 Slow Open Channel

# In[ ]:


get_ipython().run_cell_magic('time', '', "nestedcv_clf1s, f1_macro_score, graph = train_model_by_batch(train, feature_cols, 1, 2, '1s', cv_enumerator=get_nestedcv_enumerator)\nnestedcv_f1_macro_scores.append(f1_macro_score)\ngraph")


# In[ ]:


get_ipython().run_cell_magic('time', '', "kfold_clf1s, f1_macro_score, graph = train_model_by_batch(train, feature_cols, 1, 2, '1s', cv_enumerator=get_kfold_enumerator)\nkfold_f1_macro_scores.append(f1_macro_score)\ngraph")


# ## 1 Fast Open Channel

# In[ ]:


get_ipython().run_cell_magic('time', '', "nestedcv_clf1f, f1_macro_score, graph = train_model_by_batch(train, feature_cols, 3, 7, '1f', cv_enumerator=get_nestedcv_enumerator)\nnestedcv_f1_macro_scores.append(f1_macro_score)\ngraph")


# In[ ]:


get_ipython().run_cell_magic('time', '', "kfold_clf1f, f1_macro_score, graph = train_model_by_batch(train,feature_cols, 3, 7, '1f', cv_enumerator=get_kfold_enumerator)\nkfold_f1_macro_scores.append(f1_macro_score)\ngraph")


# ## 3 Open Channels

# In[ ]:


get_ipython().run_cell_magic('time', '', 'print("Training using NestedCV cross-validation method")\nnestedcv_clf3, f1_macro_score, graph = train_model_by_batch(train, feature_cols, 4, 8, \'3\', \n                                                            class_names=[\'0\',\'1\',\'2\',\'3\'], params={\'max_leaf_nodes\': 4}, \n                                                            cv_enumerator=get_nestedcv_enumerator)\nnestedcv_f1_macro_scores.append(f1_macro_score)\ngraph')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'print("Training using KFold cross-validation method")\nkfold_clf3, f1_macro_score, graph = train_model_by_batch(train, feature_cols, 4, 8, \'3\', \n                                                         class_names=[\'0\',\'1\',\'2\',\'3\'], params={\'max_leaf_nodes\': 4}, \n                                                         cv_enumerator=get_kfold_enumerator)\nkfold_f1_macro_scores.append(f1_macro_score)\ngraph')


# ## 5 Open Channels

# In[ ]:


get_ipython().run_cell_magic('time', '', 'print("Training using NestedCV cross-validation method")\nnestedcv_clf5, f1_macro_score, graph = train_model_by_batch(train, feature_cols, 6, 9, \'5\', \n                                                            class_names=[\'0\',\'1\',\'2\',\'3\',\'4\',\'5\'], params={\'max_leaf_nodes\': 6}, \n                                                            cv_enumerator=get_nestedcv_enumerator)\nnestedcv_f1_macro_scores.append(f1_macro_score)\ngraph')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'print("Training using KFold cross-validation method")\nkfold_clf5, f1_macro_score, graph = train_model_by_batch(train, feature_cols, 6, 9, \'5\', \n                                                            class_names=[\'0\',\'1\',\'2\',\'3\',\'4\',\'5\'], params={\'max_leaf_nodes\': 6}, \n                                                           cv_enumerator=get_kfold_enumerator)\nkfold_f1_macro_scores.append(f1_macro_score)\ngraph')


# ## 10 Open Channels

# **Note:** This channel produce low Macro F1 scores (in the ranges of`0.78nnnn`) using the configuration mentioned in the [original notebook](https://www.kaggle.com/cdeotte/one-feature-model-0-930). So a set of hyperparameters were tried and the one that worked best was passing in `params={'max_leaf_nodes': 255}` to the `DecisionTreeClassifier`.
# 
# Other results from using different values for `max_leaf_nodes` are as below:
# ```
# Original: max_leaf_nodes: 8 => scores: 0.78nnnnn (no folds)
# max_leaf_nodes:     100 => oof score: 0.843581 (5 folds)
# max_leaf_nodes:     250 => oof score: 0.871766 (5 folds)
# max_leaf_nodes:     255 => oof score: 0.871819 (5 folds) -- best so far, but you may get other values for this
# max_leaf_nodes:     500 => oof score: 0.87133  (5 folds)
# max_leaf_nodes:   1_000 => oof score: 0.870259 (5 folds)
# max_leaf_nodes:   2_000 => oof score: 0.864435 (5 folds)
# max_leaf_nodes:   4_000 => oof score: 0.861501 (5 folds)
# max_leaf_nodes:   5_000 => oof score: 0.859787  (5 folds)
# max_leaf_nodes:   5_000 => F1 Macro Score: 0.9025314007624331 (no folds)
# max_leaf_nodes:  10_000 => oof score: 0.827114 (5 folds)
# max_leaf_nodes:  10_000 => F1 Macro Score: 0.9177560091220058 (no folds)
# max_leaf_nodes:  25_000 => oof score: 0.815225 (5 folds)
# max_leaf_nodes:  25_000 => F1 Macro Score: 0.9435734746197743 (bit away from overfit zone) (no folds)
# max_leaf_nodes:  50_000 => oof score: 0.78893 (5 folds)
# max_leaf_nodes:  50_000 => F1 Macro Score: 0.9687715940722476 (nearing overfit zone) (no folds)
# max_leaf_nodes: 100_000 => F1 Macro Score: 1.0 (fully in overfit zone) (no folds)
# ```
# 
# Feel free to play with this parameter to improve the scores further, although the scores seem to slowly plateau and go down past the **255**-**300** range.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'print("Training using NestedCV cross-validation method")\nnestedcv_clf10, f1_macro_score, graph = train_model_by_batch(train, feature_cols, 5, 10, \'10\', \n                                                             class_names=[str(x) for x in range(11)], params={\'max_leaf_nodes\': 255}, \n                                                             cv_enumerator=get_nestedcv_enumerator)\nnestedcv_f1_macro_scores.append(f1_macro_score)\ngraph')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'print("Training using KFold cross-validation method")\nkfold_clf10, f1_macro_score, graph = train_model_by_batch(train, feature_cols, 5, 10, \'10\', \n                                                             class_names=[str(x) for x in range(11)], params={\'max_leaf_nodes\': 255}, \n                                                             cv_enumerator=get_kfold_enumerator)\nkfold_f1_macro_scores.append(f1_macro_score)\ngraph')


# #### _Removed cells about **Reflection**, **Test Data**, and **Remove Test Data Drift**, see [original notebook](https://www.kaggle.com/cdeotte/one-feature-model-0-930) by [Chris Deotte](https://www.kaggle.com/cdeotte/)._

# # Predict Test
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', "nestedcv_sub = pd.read_csv('../input/liverpool-ion-switching/sample_submission.csv')\nkfold_sub = nestedcv_sub.copy()")


# In[ ]:


"""
Training Batches mapped to sub-model types 
1,  2 ==>  1 Slow Open Channel
3,  7 ==>  1 Fast Open Channel
4,  8 ==>  3 Open Channels
6,  9 ==>  5 Open Channels
5, 10 ==> 10 Open Channels
"""

f1_macro_scores = nestedcv_f1_macro_scores
nestedcv_params = [
    [ (0, 1), "Subsample A",     "Model 1s (1 Slow Open Channel)", nestedcv_clf1s, f1_macro_scores[0]],
    [ (1, 2), "Subsample B",     "Model 3  (3 Open Channels)",     nestedcv_clf3,  f1_macro_scores[2]],
    [ (2, 3), "Subsample C",     "Model 5  (5 Open Channels)",     nestedcv_clf5,  f1_macro_scores[3]],
    [ (3, 4), "Subsample D",     "Model 1s (1 Slow Open Channel)", nestedcv_clf1s, f1_macro_scores[0]],
    [ (4, 5), "Subsample E",     "Model 1f (1 Fast Open Channel)", nestedcv_clf1f, f1_macro_scores[1]],
    [ (5, 6), "Subsample F",     "Model 10 (10 Open Channels)",    nestedcv_clf10, f1_macro_scores[4]],
    [ (6, 7), "Subsample G",     "Model 5  (5 Open Channels)",     nestedcv_clf5,  f1_macro_scores[3]],
    [ (7, 8), "Subsample H",     "Model 10 (10 Open Channels)",    nestedcv_clf10, f1_macro_scores[4]],
    [ (8, 9), "Subsample I",     "Model 1s (1 Slow Open Channel)", nestedcv_clf1s, f1_macro_scores[0]],
    [ (9,10), "Subsample J",     "Model 3  (3 Open Channels)",     nestedcv_clf3,  f1_macro_scores[2]],
    [(10,20), "Batches 3 and 4", "Model 1s (1 Slow Open Channel)", nestedcv_clf1s, f1_macro_scores[0]]
]

f1_macro_scores = kfold_f1_macro_scores
kfold_params = [
    [ (0, 1), "Subsample A",     "Model 1s (1 Slow Open Channel)", kfold_clf1s, f1_macro_scores[0]],
    [ (1, 2), "Subsample B",     "Model 3  (3 Open Channels)",     kfold_clf3,  f1_macro_scores[2]],
    [ (2, 3), "Subsample C",     "Model 5  (5 Open Channels)",     kfold_clf5,  f1_macro_scores[3]],
    [ (3, 4), "Subsample D",     "Model 1s (1 Slow Open Channel)", kfold_clf1s, f1_macro_scores[0]],
    [ (4, 5), "Subsample E",     "Model 1f (1 Fast Open Channel)", kfold_clf1f, f1_macro_scores[1]],
    [ (5, 6), "Subsample F",     "Model 10 (10 Open Channels)",    kfold_clf10, f1_macro_scores[4]],
    [ (6, 7), "Subsample G",     "Model 5  (5 Open Channels)",     kfold_clf5,  f1_macro_scores[3]],
    [ (7, 8), "Subsample H",     "Model 10 (10 Open Channels)",    kfold_clf10, f1_macro_scores[4]],
    [ (8, 9), "Subsample I",     "Model 1s (1 Slow Open Channel)", kfold_clf1s, f1_macro_scores[0]],
    [ (9,10), "Subsample J",     "Model 3  (3 Open Channels)",     kfold_clf3,  f1_macro_scores[2]],
    [(10,20), "Batches 3 and 4", "Model 1s (1 Slow Open Channel)", kfold_clf1s, f1_macro_scores[0]]
]


# ### Ensembling sub-models using Geometric mean

# In[ ]:


def ensemble_by_geometric_mean(sets_of_predictions,
                               number_of_predictions_per_set: int,
                               min_label_value: int,
                               max_label_value: int) -> np.ndarray:
    result = np.ones(number_of_predictions_per_set)
    for index, each_set_of_predictions in enumerate(sets_of_predictions):
        result *= each_set_of_predictions
    result = result ** (1 / len(sets_of_predictions))
    
    return np.nan_to_num(result, nan=min_label_value, posinf=max_label_value, neginf=min_label_value)
    
def predict_using(models, data):
    predictions = []
    if isinstance(models, list):
        for each_model in models:
            predictions.append(each_model.predict(data))
        return ensemble_by_geometric_mean(predictions, len(data), 0, 10)
    else:
        return np.round(models.predict(data))
    
def create_prediction(reference_dataframe, feature_cols, results_dataframe, params):
    total_score = 0.0
    for each_param in params:
        begin_index, end_index = each_param[0]
        start_batch = int(sub_sample_size * begin_index)
        end_batch = int(sub_sample_size * end_index)
        batch_or_sample_models = each_param[3]
        f1_macro_score = each_param[4]
        X_batch = reference_dataframe[feature_cols]
        X_batch = X_batch.iloc[start_batch:end_batch].values.reshape((-1,len(feature_cols)))
        results_dataframe.iloc[start_batch:end_batch, 1] = predict_using(batch_or_sample_models, X_batch)
        print(f"Predicting for {each_param[1]} ({start_batch} to {end_batch}) of submission with predictions from {each_param[2]} with a F1 Macro score of {f1_macro_score}")
        total_score = total_score + f1_macro_score

    print()
    average_f1_macro_score = total_score/len(params)
    print(f"Average F1 Macro across the {len(params)} subsamples/batches: {average_f1_macro_score}")
    results_dataframe.open_channels = results_dataframe.open_channels.astype(int)
    return results_dataframe, average_f1_macro_score


# #### Ensembled models through NestedCV cross-validation

# In[ ]:


get_ipython().run_cell_magic('time', '', 'nestedcv_sub, nestedcv_average_f1_macro_score = create_prediction(test, feature_cols, nestedcv_sub, nestedcv_params)')


# #### Ensembled models through KFold cross-validation

# In[ ]:


get_ipython().run_cell_magic('time', '', 'kfold_sub, kfold_average_f1_macro_score = create_prediction(test, feature_cols, kfold_sub, kfold_params)')


# # Display Test Predictions

# In[ ]:


res = 1000
letters = ['A','B','C','D','E','F','G','H','I','J']

def plot_results(reference_dataframe, results_dataframe):
    plt.figure(figsize=(20,5))
    plt.plot(range(0,reference_dataframe.shape[0],res),results_dataframe.open_channels[0::res])
    for i in range(5): plt.plot([i*batch_size,i*batch_size],[-5,12.5],'r')
    for i in range(21): plt.plot([i*sub_sample_size, i*sub_sample_size],[-5,12.5],'r:')
    for k in range(4): plt.text(k*batch_size + (batch_size/2),10,str(k+1),size=20)
    for k in range(10): plt.text(k*sub_sample_size + 40000,7.5,letters[k],size=16) # 
    plt.title('Test Data Predictions',size=16)
    plt.show()


# #### Plot results for NestedCV cross-validation

# In[ ]:


plot_results(test, nestedcv_sub)


# In[ ]:


nestedcv_sub.describe()


# In[ ]:


print(nestedcv_sub.open_channels.describe())
nestedcv_sub.open_channels.hist()


# In[ ]:


nestedcv_sub


# In[ ]:


nestedcv_sub[100000:200000]


# #### Ensembled models through KFold cross-validation

# In[ ]:


plot_results(test, kfold_sub)


# In[ ]:


kfold_sub.describe()


# In[ ]:


print(kfold_sub.open_channels.describe())
kfold_sub.open_channels.hist()


# In[ ]:


kfold_sub


# In[ ]:


kfold_sub[100000:200000]


# #### Saving submission results for NestedCV cross-validation

# In[ ]:


get_ipython().run_cell_magic('time', '', "!rm sub*nestedcv*.csv || true\nsubmission_filename = f'submission-1-54-features-nestedcv-DecisionTree-f1-macro.csv'\nnestedcv_sub.to_csv(submission_filename, index=False, float_format='%0.4f')\nprint(f'Saved {submission_filename} with Macro F1 validation score of {nestedcv_average_f1_macro_score}')\n!ls sub*nestedcv*.csv")


# #### Saving submission results for KFold cross-validation

# In[ ]:


get_ipython().run_cell_magic('time', '', "!rm sub*kfold*.csv || true\nsubmission_filename = f'submission-2-54-features-kfold-DecisionTree-f1-macro.csv'\nnestedcv_sub.to_csv(submission_filename, index=False, float_format='%0.4f')\nprint(f'Saved {submission_filename} with Macro F1 validation score of {kfold_average_f1_macro_score}')\n!ls sub*kfold*.csv")

