#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install colour')
get_ipython().system('pip install git+https://github.com/jundongl/scikit-feature/')


# In[2]:


get_ipython().system('jupyter nbextension enable --py widgetsnbextension')
get_ipython().run_line_magic('pylab', 'inline')

import ipywidgets as widgets
from ipywidgets import interact, interact_manual
from IPython.display import HTML

import os
import time
import json
import pickle
import random
import copy
import tqdm 

import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter

import colorlover as cl
from plotly import figure_factory as FF
from plotly.offline import iplot, plot, init_notebook_mode, download_plotlyjs
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from colour import Color
import cufflinks as cf
import missingno as msno
import seaborn as sns

import shap
from skfeature.function.information_theoretical_based import MRMR
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import median_absolute_error, r2_score, mean_squared_error, explained_variance_score, accuracy_score, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

from sklearn.manifold import TSNE
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import cpu_count

import catboost as cb
import xgboost as xgb
import lightgbm as lgbm
from sklearn.svm import SVC
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

import keras
from keras.optimizers import SGD
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings("ignore")

init_notebook_mode(connected=True)
cf.go_offline()
cf.set_config_file(world_readable=True, theme='white', offline=True)
# HTML(cl.to_html(cl.scales['9']))
csv_path = '../input/'
json_path = '../input/jsondata/'


# In[3]:


def error_typer(y, prediction):
    if y == prediction:
        if bool(y): return 'true pos'
        else: return 'true neg'
    if bool(y): return 'false pos'
    else: return 'false neg'

    
def mean(x):
    return sum(x) / len(x)


def onehot(data, onehot_columns, drop=True):
    one_hot = []
    for column in onehot_columns:
        one_hot.append(pd.get_dummies(data[column], prefix=column))
    if drop:
        one_hot.append(data.drop(onehot_columns, axis=1))
    else:
        one_hot.append(data)
    return pd.concat(one_hot, axis=1)


def augment_data(x,y,t=2):
    xs,xn = [],[]
    for i in range(t):
        mask = y>0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xs.append(x1)

    for i in range(t//2):
        mask = y==0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xn.append(x1)

    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x,xs,xn])
    y = np.concatenate([y,ys,yn])
    return x,y

def balance_classes(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    min_class = df[label_column].value_counts().idxmin()
    min_class_size = df[label_column].value_counts().min()
    to_concat = []
    for _class in df[label_column].value_counts().index:
        to_concat.append(df[df[label_column] == _class].sample(min_class_size))
    df = pd.concat(to_concat)
    df = df.sample(frac=1).reset_index(drop=True)
    return df


def correlation_matrix(dataset: pd.DataFrame):
    colors = convert_colorscale_format(gradient_generator('#43C6AC', '#191654', 500))
    corr_matrix = dataset.corr()
    corr_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    
    data = [go.Heatmap(z=corr_matrix.values, x=dataset.columns.tolist(), y=dataset.columns.tolist(), colorscale=colors)]
    fig = go.Figure(data=data, layout=go.Layout(height=800, width=800, xaxis={'autorange':'reversed'}))
    iplot(fig)
    
# -------------------------------------------------------------------
# Color tools

def gradient_generator(color1, color2, n):
    first = Color(color1)
    second = Color(color2)
    return [str(color) for color in list(first.range_to(second, n))]


def convert_colorscale_format(colorscale):
    plotly_colorscale = []
    for index, sec_value in enumerate(np.linspace(0, 1, len(colorscale))):
        plotly_colorscale.append([sec_value, str(colorscale[index])])
    return plotly_colorscale


# -------------------------------------------------------------------
# Memory optimization tools

def optimize_memory(data: pd.DataFrame) -> pd.DataFrame:
    for column, _type in data.dtypes.items():
        if _type == np.int64:
            if data[column].min() >= 0:
                data[column] = pd.to_numeric(data[column], downcast='unsigned')
            else:
                data[column] = pd.to_numeric(data[column], downcast='integer')
        elif _type == np.float64:
            data[column] = pd.to_numeric(data[column], downcast='float')
    return data

def memory_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)


# -------------------------------------------------------------------
# File tools

def create_download_link(data, filename = "data.csv"):  
    data.to_csv(filename, index=False)
    html = '<a href={filename}>Download CSV file</a>'
    html = html.format(title=title,filename=filename)
    return HTML(html)


# In[4]:


def ts_features(ts) -> pd.Series:
    regressor = Ridge(alpha=.1)
    ts = pd.DataFrame(ts)
    ts = ts.reset_index(drop=True)
    X_train = ts.index.values.reshape(-1, 1)
    result = dict()
    for column in ['radiant_gold', 'dire_gold', 'player_gold']:
        Y_train = ts[column].values
        regressor.fit(X_train, Y_train)
        result[f'regressor_{column}'] = regressor.coef_[0]
        methdos = {'skew': ts[column].skew, 'kurt': ts[column].kurt, 'min': ts[column].min, 
                   'max': ts[column].max, 'std': ts[column].std, 'mean': ts[column].mean, 
                   'var': ts[column].var, 'qua': ts[column].quantile, 'std': ts[column].std, 
                   'med': ts[column].median}
        for method_name, method in methdos.items():
            result[f'{column}_{method_name}'] = method()
    return pd.Series(result)


def preprocess_ts(path=f'{json_path}dota2_skill_train.jsonlines', total_tqdm=100):
    collector = list()
    chunks = pd.read_json(path, orient='records', lines=True, chunksize=1000)
    for chunk in tqdm.tqdm_notebook(chunks, total=total_tqdm):
        json_data = chunk[['series', 'damage_targets', 'radiant_heroes', 'id', 'item_purchase_log', 'final_items', 'dire_heroes', 'level_up_times']]
        json_data = json_data.set_index('id')
        json_data = json_data.merge(json_data['series'].apply(ts_features), left_index=True, right_index=True)
        json_data = json_data.drop('series', axis=1)
        collector.append(json_data)
    return pd.concat(collector, axis=0)


def encode_roles(roles):
    result = []
    possible_roles = ['Carry', 'Disabler', 'Durable', 'Escape', 'Initiator', 'Jungler', 'Nuker', 'Pusher', 'Support']
    for role in possible_roles:
        if role in roles:
            result.append(1)
        else:
            result.append(0)
    return pd.Series(result, index=['Carry', 'Disabler', 'Durable', 
                                    'Escape', 'Initiator', 'Jungler', 
                                    'Nuker', 'Pusher', 'Support'])

heroes = pd.read_csv(csv_path + 'dota2_heroes.csv')[['hero_id', 'roles', 'localized_name']]
stats = pd.read_json(csv_path + 'heroes_data.json').T.reset_index().rename({'index': 'localized_name'}, axis=1)
heroes = heroes.merge(stats, on='localized_name')
heroes = heroes.drop('localized_name', axis=1).set_index('hero_id')
heroes = heroes.merge(heroes['roles'].apply(encode_roles), left_index=True, right_index=True).drop('roles', axis=1)
heroes = heroes.rename({column: str(column) + '_player' for column in heroes.columns}, axis=1)

items = pd.read_csv(csv_path + 'dota2_items.csv', index_col='item_id')
items['qual'] = items['qual'].fillna('recipe')
items = items.reset_index()

# ===============================================================================

def string_grouper(row):
    if 'neutral' in row:
        return 'neutrals'
    if 'hero' in row:
        return 'heroes'
    return 'others'

def convert_damage_info(row):
    return pd.Series(row).groupby(string_grouper).sum()

def team_setup(row):
    team = row[f'{row["player_team"]}_heroes']
    return heroes[['Carry', 'Disabler', 'Durable', 'Escape', 'Initiator', 'Jungler', 'Nuker', 'Pusher', 'Support']].loc[team].sum() 

def final_items(final_items):
    result = {'component': 0, 'secret_shop': 0, 'consumable': 0, 'recipe': 0, 'common': 0, 'rare': 0, 'epic': 0, 'artifact': 0, 'summary_cost': 0}
    for item in final_items:
        if item != 0:
            result[items.loc[item]['qual']] += 1
            result['summary_cost'] += items.loc[item]['cost']
    return pd.Series(result)

# start = time.time()
# json_data = pd.concat([data['skilled'], pd.read_csv(csv_path + 'json_data_train.csv', index_col='id')], axis=1)
# rnn_data = json_data[['ability_upgrades', 'level_up_times', 'item_purchase_log', 'skilled']]
# json_data = json_data.drop(['ability_upgrades', 'level_up_times', 'item_purchase_log'], axis=1)
# print('Preprocessing is ready!')

# json_data['damage_targets'] = json_data['damage_targets'].apply(eval)
# json_data = json_data.merge(json_data['damage_targets'].apply(convert_damage_info), left_index=True, right_index=True).drop('damage_targets', axis=1)
# print('damage_targets are ready!')

# json_data['dire_heroes'], json_data['radiant_heroes'] = json_data['dire_heroes'].apply(eval), json_data['radiant_heroes'].apply(eval)
# json_data = json_data.merge(json_data.apply(team_setup, axis=1), left_index=True, right_index=True).drop(['dire_heroes', 'radiant_heroes'], axis=1)
# print('heroes are ready!')

# json_data['final_items'] = json_data['final_items'].apply(eval)
# json_data = json_data.merge(json_data['final_items'].apply(final_items), left_index=True, right_index=True).drop('final_items', axis=1)
# print('final_items are ready!')
# print(round(time.time() - start, 2))


# In[5]:


json_data = pd.read_csv(csv_path + 'prep_json_train.csv', index_col='id').drop(['team_gold_min', 'player_gold_min', 'skilled', 'player_team'], axis=1)
test_json_data = pd.read_csv(csv_path + 'prep_json_test.csv', index_col='id').drop(['team_gold_min', 'player_gold_min', 'player_team'], axis=1)

data = pd.read_csv(csv_path + 'dota2_skill_train.csv', index_col='id')
data = data[data['winner_team'] != 'other']
data = data.replace({'radiant': 1, 'dire': 0})
data['won'] = (data['player_team'] == data['winner_team']).astype(np.int8)
data = data.merge(heroes, on='hero_id', right_index=True)
data = pd.concat([json_data, data], axis=1)
data['neutrals'] = data['neutrals'].fillna(data['neutrals'].max())
data = data.dropna()

test_data = pd.read_csv(csv_path + 'dota2_skill_test.csv', index_col='id')
test_data = test_data.replace({'radiant': 1, 'dire': 0})
test_data['won'] = (test_data['player_team'] == test_data['winner_team']).astype(np.int8)
test_data = test_data.merge(heroes, on='hero_id', right_index=True)
test_data = pd.concat([test_json_data, test_data], axis=1)
test_data['neutrals'] = test_data['neutrals'].fillna(test_data['neutrals'].max())
test_data['others'] = test_data['others'].fillna(test_data['others'].max())
test_data = test_data.fillna(0)


# In[6]:


columns = data.drop(['skilled', 'player_team', 'winner_team'], axis=1).columns.tolist()

# @interact
def dist(balance=['Yes', 'No'], show_test_data=['No', 'Yes'], column=columns):
    df = data[['skilled', column]]
    if balance == 'Yes':
        df = balance_classes(df, 'skilled')
    plot_data = df.groupby('skilled')[column].apply(lambda data: data.reset_index(drop=True)).unstack().T
    if show_test_data == 'Yes':
        plot_data = plot_data.reset_index(drop=True)
        plot_data = pd.concat([plot_data, test_data[column].reset_index(drop=True).rename('test')], axis=1)
    plot_data.iplot('hist', width=0.1, title=f'{column} distribution, 0 mode: {plot_data[0].mode().iloc[0]}; 1 mode: {plot_data[1].mode().iloc[0]}')


# In[7]:


def find_labels(data, unique=10):
    label_columns = list()
    for column in data.columns:
        if data[column].unique().shape[0] <= unique:
            label_columns.append(column)
    return label_columns
    
    
def mhe1(train, test, categorical_features, targets, drop=True):
    for label in tqdm.tqdm_notebook(categorical_features, desc='MHE processing'):
        means = train.groupby(label).mean()
        for target in targets:
            train[f'mhe_{label}_{target}'] = train[label].map(means[target])
            test[f'mhe_{label}_{target}'] = test[label].map(means[target])
            test = test.fillna(train[target].mean())
    if drop:
        train, test = train.drop(categorical_features, axis=1), test.drop(categorical_features, axis=1)  # if test without labels
    return train, test


def mhe2(train, test, categorical_features, targets, drop=True):  # Works worst on test set!!! P.S. Don`t know why
    for label in tqdm.tqdm_notebook(categorical_features, desc='MHE processing'):
        cumsum = train.groupby(label).cumsum()
        cumcount = train.groupby(label).cumcount()
        for target in targets:
            train[f'mhe_{label}_{target}'] = (cumsum[target] - train[target]) / cumcount
            test[f'mhe_{label}_{target}'] = (cumsum[target] - train[target]) / cumcount
    if drop:
        train, test = train.drop(categorical_features, axis=1), test.drop(categorical_features, axis=1)  # if test without labels
    return train, test

def mhe3(train, test, categorical_features, targets, drop=True):
    train_columns, test_columns = list(), list()
    for column in categorical_features:
        for target in targets:
            train_series, test_series = target_enoder(train[column], test[column], train[target])
            train_columns.append(train_series)
            test_columns.append(test_series)
    train_columns, test_columns = pd.concat(train_columns, axis=1), pd.concat(test_columns, axis=1)
    if drop:
        train, test = train.drop(categorical_features, axis=1), test.drop(categorical_features, axis=1)  # if test without labels
    train, test = pd.concat([train, train_columns], axis=1), pd.concat([test, test_columns], axis=1)
    return train, test


def target_enoder(trn_series=None, tst_series=None, target=None, min_samples_leaf=1, smoothing=1, noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior  
    """ 
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)


def interactions_builder(data, columns, calc_collinearity=False):
    bad_interactions = list()
    for i in tqdm.tqdm_notebook(range(len(columns)), desc='Building interactions'):
        for j in range(i + 1, len(columns)):
            interactions = [
                (data[columns[i]] + data[columns[j]]).rename(f'interaction_{columns[i]}_plus_{columns[j]}'),
                (data[columns[i]] * data[columns[j]]).rename(f'interaction_{columns[i]}_mult_{columns[j]}'),
                (data[columns[i]] - data[columns[j]]).rename(f'interaction_{columns[i]}_minus_{columns[j]}'),
                (data[columns[j]] - data[columns[i]]).rename(f'interaction_{columns[j]}_minus_{columns[i]}'),
                (data[columns[i]] / data[columns[j]]).rename(f'interaction_{columns[i]}_div_{columns[j]}'),
                (data[columns[j]] / data[columns[i]]).rename(f'interaction_{columns[j]}_div_{columns[i]}'),
            ]
            interactions = pd.concat(interactions, axis=1)
            if calc_collinearity:
                bad_interactions += collinear_features(interactions, 0.6)
            data = pd.concat([data, interactions], axis=1)
    if calc_collinearity:
        return data, bad_interactions
    return data
#             columns_interactions.append(data[[columns[i], columns[j]]])
#     resulting_data = list()
#     with ThreadPoolExecutor() as executor:
#         for data_piece in executor.map(calc_interaction, columns_interactions):
#             resulting_data.append(data_piece)
#     print(resulting_data)
#     return pd.concat(resulting_data, axis=1)


def best_features(data, label_column, k=40, method='chi', numeric=False):
    if k > data.columns.shape[0] - 1:
        k = data.columns.shape[0] - 1 
        
    if method == 'mrmr':  # Extremly slow!
        print('Start selecting features')
        data = data.sample(500)
        idx, _, _ = MRMR.mrmr(data.drop('skilled', axis=1).values, 
                              data['skilled'].astype(int).values, n_selected_features=k)
        feature_list = data.columns[idx].tolist()
        print(f'Selected {k} best features using mrmr')

    elif method == 'chi':
        data, target = data.drop(label_column, axis=1), data[label_column]
        selector = SelectKBest(chi2, k=k).fit(data, target.astype(int))
        mask = selector.get_support()
        feature_list = [feature for passed, feature in zip(mask, data.columns) if passed]
    else:
        raise (ValueError, "Only 'chi' and 'mrmr' are possible values for method")
    
    if not numeric:
        feature_list += [label_column, ]
            
    return list(set(feature_list))


def normalize(train, test, cat_features):
    train = train.replace([np.inf, -np.inf], np.nan).fillna(0)
    test = test.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    min_max_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()
    if cat_features:
        train_cat, test_cat = train[cat_features], test[cat_features]
        train, test = train.drop(cat_features, axis=1), test.drop(cat_features, axis=1)
    if train.columns.shape[0] == test.columns.shape[0]:
        train = pd.DataFrame(min_max_scaler.fit_transform(standard_scaler.fit_transform(train)), 
                             columns=train.columns, index=train.index)
    else:
        label_column = list(set(train.columns.tolist()) - set(test.columns.tolist()))[0]
        target = train[label_column]
        train = train.drop(label_column, axis=1)
        train = pd.concat([target, 
                           pd.DataFrame(min_max_scaler.fit_transform(standard_scaler.fit_transform(train)), 
                                        columns=train.columns, index=train.index)], axis=1)
    test = pd.DataFrame(min_max_scaler.transform(standard_scaler.transform(test)), 
                        columns=test.columns, index=test.index)
    if cat_features:
        train, test = pd.concat([train, train_cat], axis=1), pd.concat([test, test_cat], axis=1)
    return train, test


def collinear_features(data, highest_corr=0.7):
    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    return [column for column in upper.columns if any(upper[column] > highest_corr)]


# In[8]:


def preprocess(train, test, label_column='skilled', cat_features=[], encoding_type='mhe3', mhe_targets=[], drop_encoded=True, n_features=100, make_features=0, highest_corr=0.7):
    train, test = normalize(train, test, cat_features)
    
    numeric_features = [column for column in train.columns if train[column].unique().shape[0] > 559]
    numeric_features = best_features(train[numeric_features + [label_column,]], label_column, k=make_features, numeric=True)
    if not cat_features:
        cat_features = [column for column in train.columns if train[column].unique().shape[0] < 120]
        cat_features = best_features(train[cat_features], label_column, k=20)
        cat_features.remove(label_column)
        print(f'Found categorical features: {", ".join(cat_features)}')

#   ---------------------------------------------------------------------------------------------------------------------------
#   Interactions building
    if make_features:
        target = train[label_column]
        train, bad_interactions = interactions_builder(train.drop(label_column, axis=1), numeric_features, calc_collinearity=True)
        train = pd.concat([target, train.drop(bad_interactions, axis=1)], axis=1)
        if label_column in test.columns:
            test = pd.concat([test[label_column], interactions_builder(test.drop(label_column, axis=1), numeric_features, calc_collinearity=False)], axis=1)
        else:
            test = interactions_builder(test, numeric_features, calc_collinearity=False)
        test = test.drop(bad_interactions, axis=1)
        train, test = normalize(train, test, cat_features)

#   -----------------------------------------------------------------------------------------
#   Removing colliearity
    to_drop = collinear_features(train.drop(cat_features + [label_column, ], axis=1), highest_corr)
    train, test = train.drop(to_drop, axis=1), test.drop(to_drop, axis=1)

#   -----------------------------------------------------------------------------------------
#   Selecting best features
    best = best_features(train.drop(cat_features, axis=1), label_column, n_features)
    train = train[best + cat_features]
    if label_column not in test.columns:
        best.remove(label_column)
    test = test[best + cat_features]
    
#   -----------------------------------------------------------------------------------------
#   Encoding
    if 'mhe' in encoding_type:
        if not mhe_targets:
            mhe_targets = [label_column,]
        mhes = {'mhe1' : mhe1, 'mhe2': mhe2, 'mhe3': mhe3}
        train, test = mhes[encoding_type](train, test, cat_features, mhe_targets, drop_encoded)

    elif encoding_type == 'ohe':
        train = onehot(train, cat_features, drop_encoded)
        test = onehot(test, cat_features, drop_encoded)
    else:
        pass

    return train, test    


def cross_validate_models(df: pd.DataFrame, label_column: str, models: list, prepocess_function, preprocess_args={}, classification: bool=True, folds_number=5, round_results=3):
    names_counter = Counter()
    kf = KFold(n_splits=folds_number, shuffle=True, random_state=42)
    colors = gradient_generator('#C471EC', '#1CBDE9', folds_number)
    validation_data = {name: [[], []] for name in models.keys()}
    trained_models = {name: list() for name in models.keys()}
    plots = dict()
    for model in models:
        name = model.__class__.__name__
        name = f'{name}_{names_counter[name]}'
        names_counter[name] += 1
        n = 0
        data = [go.Scatter(x=[0., 1.], y=[0., 1.], line=go.scatter.Line(color='#d1d1d1'), marker=go.scatter.Marker(size=1), name='Worst'), ]
        for train_index, val_index in tqdm.tqdm_notebook(kf.split(df), total=folds_number, desc=f'{name} folds'):
            
            train, val = df.iloc[train_index].reset_index(drop=True), df.iloc[val_index].reset_index(drop=True)
            train, val = prepocess_function(train, val, label_column, **preprocess_args)
            X_train, Y_train = train.drop(label_column, axis=1), train[label_column]
            X_val, Y_val = val.drop(label_column, axis=1), val[label_column]
            if type(model) == cb.CatBoostRegressor or type(model) == cb.CatBoostClassifier:
                model.fit(
                    X_train, Y_train,
                    use_best_model=True,
                    eval_set=cb.Pool(X_val, Y_val),
                    logging_level="Silent",  # 'Silent', 'Verbose', 'Info', 'Debug'
                    plot=True)
            elif type(model) == Sequential:
                X_train, Y_train = X_train.values, Y_train.values
                X_val, Y_val = X_val.values, Y_val.values
                model.fit(X_train, Y_train, batch_size=32, epochs=100, verbose=1, validation_data=(X_val, Y_val), 
                          callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.001, patience=20, mode='auto', baseline=None, restore_best_weights=True), ])
            else:
                model.fit(X_train, Y_train)

            prediction = pd.Series(model.predict(X_val))
            trained_models[name].append((model, X_val.columns))
            
            if classification:
                fpr, tpr, thresholds = roc_curve(Y_val, model.predict_proba(X_val)[:, 1])
                _auc = round(auc(fpr, tpr), round_results)
                _acc = round(accuracy_score(Y_val, prediction), round_results)
                data.append(go.Scatter(x=fpr, y=tpr, line=go.scatter.Line(color=colors[n], smoothing=0., shape='spline', ), 
                                       marker=go.scatter.Marker(size=1), name=f'auc: {_auc}; acc: {_acc}'))
                validation_data[name][0].append(_auc)
                validation_data[name][1].append(_acc)

            else:
                raise NotImplemented
#               TODO add data append
            n += 1
    
        if classification:
            plots[name] = data
            
    grids = [[[{}]], [[{}, {}]], [[{}, {}], [{'colspan': 2, 'rowspan': 2}, None], [None, None]], 
             [[{}, {}], [{}, {}]], [[{}, {}], [{}, {}], [{'colspan': 2, 'rowspan': 2}, None], [None, None]]]        
    plots_n = len(plots.keys())
    if plots_n > 4:
        grid = list()
        for i in range(plots_n//4):
            grid.append([{}, {}, {}, {}])
        if plots_n % 4 != 0:
            grid.append([{} for i in range(plots_n % 4)] + [None for i in range(4 - plots_n % 4)])
    else:
        grid = grids[plots_n-1]
    
    ordered_plots = sorted(plots, key=lambda x: sum(validation_data[x][0]))
    subplot_names = [f'{name} AUC: {round(mean(validation_data[name][0]), round_results)}; ACC: {round(mean(validation_data[name][1]), round_results)}' for name in ordered_plots]
    fig = tools.make_subplots(rows=len(grid), cols=len(grid[0]), specs=grid, subplot_titles=subplot_names)
    fig['layout'].update(title=f'Cross validation on {folds_number} folds', height=500 * len(grid), width=500 * len(grid[0]) + 100)
    n = 0
    for i in range(len(grid)):
        for j in range(len(grid[i]) - grid[i].count(None)):
            for plot in plots[ordered_plots[n]]:
                fig.append_trace(plot, i+1, j+1)
            n += 1
    iplot(fig)
    return trained_models


# In[10]:


def dense_nn(n_features):
    model = Sequential()
    model.add(Dense(n_features, activation='relu'))
    model.add(Dense(128, activation='relu', ))
    model.add(Dense(128, activation='relu', kernel_initializer='normal'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


class CatWrapper:
    def __init__(self, model, target, cat, init_params={}, fit_params={}):
        self.model = model
        self.cat_name = cat
        self.target_name = target
        self.init_params = init_params
        self.fit_params = fit_params
        self.models = dict()
        
    def fit(self, X, Y):
        X[self.cat_name] = X[self.cat_name].astype(int)
        data = pd.concat([X, Y], axis=1)
        total = data[self.cat_name].unique().shape[0]
        data = data.groupby(self.cat_name)
        for cat, _data in tqdm.tqdm_notebook(data, total=total, desc='Fitting'):
            self.models[cat] = self.model(**self.init_params)
            self.models[cat].fit(_data.drop(self.target_name, axis=1), _data[self.target_name], **self.fit_params)
    
    def cat_predictor(self, X, proba):
        X[self.cat_name] = X[self.cat_name].astype(int)
        total = X[self.cat_name].unique().shape[0]
        indexes = X.index
        X = X.groupby(self.cat_name)
        Y = list()
        for cat, data in tqdm.tqdm_notebook(X, total=total, desc='Predicting'):
            if proba:
                Y.append(pd.DataFrame(self.models[cat].predict_proba(data), index=data.index, columns=[0, 1]))
            else:
                Y.append(pd.Series(self.models[cat].predict(data), index=data.index).rename(self.target_name).astype(int))
        return pd.concat(Y).reindex(indexes)
   
    def predict(self, X):
        return self.cat_predictor(X, False)
    
    def predict_proba(self, X):
        return self.cat_predictor(X, True)
        


# In[ ]:


# models = cross_validate_models(data, 'skilled', 
#                         [
#                           dense_nn(data.shape[1]),
#                           cb.CatBoostClassifier(task_type='GPU', iterations=1000, eval_metric='Accuracy'),
#                           SGDClassifier(loss='modified_huber'),
#                           KNeighborsClassifier(),
#                           RandomForestClassifier(),
#                           GaussianNB(),
#                           LogisticRegression(),
#                           SVC(probability=True)
#                           GaussianProcessClassifier()
#                         ], 
#                          preprocess,      
#                         preprocess_args={'n_features': 80, 'cat_features': [], 'encoding_type': 'mhe3', 'make_features': 0, 'highest_corr': 0.99}
# )


# In[ ]:


columns = sorted(data.drop(['skilled', 'player_team', 'winner_team'], axis=1).columns.tolist())

# @interact
def interactions_view(column1=columns, column2=columns, do=['mul', 'div', 'sum', 'min'], balance=['Yes', 'No']):
    if column1 != column2:
        column = f'{column1}_{do}_{column2}'
        df = data[['skilled', column1, column2]]
        df[[column1, column2]] = df[[column1, column2]].astype(float)
        if do == 'mul':
            new_series = df[column1] * df[column2]
        elif do == 'div':
            new_series = df[column1] / df[column2]
        elif do == 'sum':
            new_series = df[column1] + df[column2]
        elif do == 'min':
            new_series = df[column1] - df[column2]
        df = df.drop([column1, column2], axis=1)
        df[column] = new_series
    else:
        column = column1
        df = data[['skilled', column]]
    if balance == 'Yes':
        df = balance_classes(df, 'skilled')
    plot_data = df.groupby('skilled')[column].apply(lambda data: data.reset_index(drop=True)).unstack().T
    plot_data.iplot('hist', width=0.1, title=f'{column} distribution, 0 mode: {plot_data[0].mode().iloc[0]}; 1 mode: {plot_data[1].mode().iloc[0]}')


# In[41]:


mhe_targets = ['skilled', 'avg_gpm_x16', 'regressor_team_gold', 'avg_xpm_x16', 'duration', 'team_fight_participation', 'last_hits']

train, test = preprocess(data, test_data, n_features=120, make_features=0, highest_corr=1, 
                         encoding_type='mhe3', drop_encoded=False, mhe_targets=mhe_targets, cat_features=['hero_id'])
train, val = train_test_split(train, test_size=0.1)
X_train, X_val, Y_train, Y_val = train.drop('skilled', axis=1), val.drop('skilled', axis=1), train['skilled'], val['skilled']


# In[28]:


model = cb.CatBoostClassifier(
    iterations=1000, task_type='GPU', eval_metric='Accuracy', 
    bagging_temperature=0.429285352299552, 
    border_count=254, 
    fold_permutation_block_size=790, 
    depth=7,
#     learning_rate=0.11861766957101182, random_strength=0.19180753287299276#, 
#     sampling_unit=0, leaf_estimation_method=0., leaf_estimation_backtracking=2,
)

model.fit(
    cb.Pool(X_train, Y_train),
    eval_set=cb.Pool(X_val, Y_val),
    early_stopping_rounds=500,
    use_best_model=True,
    verbose=False,
    plot=True,
)


# In[ ]:


pd.Series(model._feature_importance, index=X_train.columns).sort_values().iplot('bar')


# In[ ]:


preds = pd.DataFrame({'id': test.index, 'skilled': model.predict(test).astype(int)})
create_download_link(preds)


# In[ ]:





# # Experimental part starts now! 

# In[110]:


# Builds predictions by layers
# if there are more then one model in last layer the voting will be used
class Stacker:
    def __init__(self, folds_number=5, probas=False, random_selection=False):
        self.layers = list()
        self.probas = probas
        self.folds_number = folds_number
        self.random_selection = random_selection
        self.target_name = 'target'
        self.hidden_single_model = False
        
    def add(self, models: list, drop_fitting_data=False):
        if self.hidden_single_model:
            raise ValueError('Passed layer with one model before. Only last layer may have one model.')
        if not self.layers and drop_fitting_data:
            raise ValueError('Can`t drop data without any layers before!')
        if not models:
            raise ValueError('Can`t add layer without models')

        counter = Counter()
        layer = {'models': [], 'drop_fitting_data': drop_fitting_data}
        
        if len(models) == 1:
            self.hidden_single_model = True
            
        for model in models:
            if type(model) != dict:
                inited_model = model()
                model_name = inited_model.__class__.__name__
                counter[model_name] += 1
                model_name = f'{model_name}_{counter[model_name]}'
                layer['models'].append({'model': inited_model, 'name': model_name, 'fit_params': {}})
            else:
                try:
                    keys = model.keys()
                    if 'fit_params' not in keys:
                        model['fit_params'] = dict()
                    if 'init_params' not in keys:
                        model['init_params'] = dict()
                    
                    inited_model = model['model'](**model['init_params'])
                    model_name = inited_model.__class__.__name__
                    counter[model_name] += 1
                    model_name = f'{model_name}_{counter[model_name]}'
                    formed_model = {'model': inited_model, 'name': model_name, 'fit_params': model['fit_params']}
                    if 'val_arg' in keys:
                        formed_model['val_arg'] = model['val_arg']
                    layer['models'].append(formed_model)

                except KeyError:
                    raise ValueError('You should pass or list of models classes or dicts with keys "model", "init_params", "fit_params", "val_arg" as argument models.                                      \n Example: [SVC, CatBoostClassifier] or \n[{"model": SVC, "init_params": {},"fit_params": {}},                                      {"model": CatBoostClassifier(), "init_params": {}, "fit_params": {"learning_rate": 0.01}, "val_arg": "eval_set"}]')
        self.layers.append(layer)
                
    def fit(self, X: pd.DataFrame, Y: pd.Series):
        self.target_name = Y.name
        layer_number = 0
        for layer in tqdm.tqdm_notebook(self.layers, desc='Layers'):
            models_amount = len(layer['models'])
            if layer_number == len(self.layers) - 1:   # Last layer
                if layer['drop_fitting_data']:
                    X = self.drop_data_columns(X)
                for model_data in tqdm.tqdm_notebook(layer['models'], desc='Last layer training'):    
                    self.model_train(model_data, X, Y)
                    
            elif layer_number < len(self.layers) - 1:  # Hidden or Starting layer
                layer_predictions = list()
                for model_data in tqdm.tqdm_notebook(layer['models'], desc='Layer training'):
                    layer_X = X
                    if layer['drop_fitting_data']:
                        layer_X = self.drop_data_columns(layer_X)
                    predictions = self.stack_training(model_data, layer_X, Y)
                    layer_predictions.append(predictions)
                layer_predictions = pd.concat(layer_predictions, axis=1)
                X = pd.concat([X, layer_predictions], axis=1)
            layer_number += 1
    
    def stack_training(self, model_data, X, Y):
        kf = KFold(n_splits=self.folds_number, shuffle=True, random_state=42)
        predictions = list()
        for train_index, test_index in tqdm.tqdm_notebook(kf.split(X), total=self.folds_number, desc=f'{model_data["name"]} training on {self.folds_number} folds'):
            X_train, X_test, Y_train, Y_test = X.iloc[train_index], X.iloc[test_index], Y.iloc[train_index], Y.iloc[test_index]
            self.model_train(model_data, X_train, Y_train)  #  FIXME Works when model retrains on each fit call
            predictions.append(self.predictions_for_stack(model_data, X_test))
        self.model_train(model_data, X, Y, save_columns=True)
        return pd.concat(predictions)
        
    def model_train(self, model_data, X, Y, save_columns=False):
        X, X_val, Y, Y_val = train_test_split(X, Y, test_size=0.2)
        if 'val_arg' in model_data.keys():
            model_data['fit_params'][model_data['val_arg']] = (X_val, Y_val)
        model_data['model'].fit(X, Y, **model_data['fit_params'])
        if save_columns:
            model_data['columns'] = X.columns
    
    @staticmethod
    def drop_data_columns(X, *args):
        columns = [column for column in X.columns if '_prediction' in column]
        if args:
            return [X[columns], ] + [x[columns] for x in args]
        return X[columns]
        
    def predictions_for_stack(self, model_data, X):
        if self.probas:
            try:
                return pd.Series(model_data['model'].predict_proba(X)[:, 1], index=X.index).rename(f"{model_data['name']}_prediction")
            except:
                print(f'{model_data["model_name"]} doesn`t support predict_proba! predict will be used instead.')
        res = pd.Series(model_data['model'].predict(X), index=X.index).rename(f"{model_data['name']}_prediction")
        return res

    def layer_predictions(self, layer, X):
        if layer['drop_fitting_data']:
            X = self.drop_data_columns(X)
        
        if len(layer['models']) == 1:
            model_data = layer['models'][0]
            return self.predictions_for_stack(model_data, X)
        
        predictions = list()
        for model_data in layer['models']:
            predictions.append(self.predictions_for_stack(model_data, X))
        return pd.concat(predictions, axis=1)
    
    def predict(self, X):
        layer_index = 0
        for layer in self.layers:
            if layer_index == len(self.layers) - 1:
                if len(layer['models']) == 1:  #  Last model
                    return self.layer_predictions(layer, X).rename(self.target_name)
                else:                          #  Voiting
                    predictions = self.layer_predictions(layer, X).mean(axis=1).round().astype(int).rename(self.target_name)
                    return predictions
            else:  #  Hidden layers
                predictions = self.layer_predictions(layer, X)
                X = pd.concat([X, predictions], axis=1)
            layer_index += 1

# st = Stacker()
# {'model': CatWrapper, 'init_params': {'model': KNeighborsClassifier, 'target': 'skilled', 'cat': 'hero_id'}}
# st.add([{'model': cb.CatBoostClassifier, 
#          'init_params': dict(iterations=1000, task_type='GPU', eval_metric='Accuracy', bagging_temperature=0.429285352299552, border_count=254, fold_permutation_block_size=790,),
#          'fit_params': dict(early_stopping_rounds=500, use_best_model=True, verbose=False, plot=False,),
#          'val_arg': 'eval_set'
#         }, RandomForestClassifier, LogisticRegression])
# st.add([LogisticRegression], False)
# st.add([GaussianNB, GaussianNB])
# st.add([RandomForestClassifier, LogisticRegression, GaussianNB])
# st.fit(X_train, Y_train)


# In[ ]:


def objective(space):
    X_train, X_val, Y_train, Y_val = train_test_split(train.drop('skilled', axis=1), train['skilled'], test_size=0.2)
    model = cb.CatBoostClassifier(iterations=1000, task_type='GPU', eval_metric='Accuracy', **space)
    model.fit(
        cb.Pool(X_train, Y_train),
        eval_set=cb.Pool(X_val, Y_val),
        early_stopping_rounds=200,
        use_best_model=True,
        verbose=False,
        plot=False,
    )
    loss = - accuracy_score(model.predict(X_val), Y_val)
    return {'loss': loss, 'params': space, 'status': STATUS_OK}

    
space = {
    'learning_rate':                hp.uniform('learning_rate', 0.05, 0.4),
    'depth':                        hp.quniform('depth', 5, 12, 1),
#     'random_seed':                  hp.quniform('random_seed', 1, 1000, 1),
#     'l2_leaf_reg':                  hp.quniform('l2_leaf_reg', 1, 100, 1),  # including makes worst
#     'bootstrap_type':               hp.choice('bootstrap_type', ['Poisson', 'Bayesian', 'Bernoulli', 'No']),
    'random_strength':              hp.uniform('random_strength', 0.01, 0.9),
    'bagging_temperature':          hp.uniform('bagging_temperature', 0, 1),
#     'subsample':                    hp.uniform('subsample', 0.001, 1),  # Poisson or Bernoulli should be active
    'border_count':                 hp.choice('border_count', [128, 254]),
    'sampling_unit':                hp.choice('sampling_unit', ['Object', 'Group']),
#     'one_hot_max_size':             hp.quniform('one_hot_max_size', 2, 120, 2),
#     'has_time':                     hp.choice('has_time', [True, False])
#     'rsm':                          hp.uniform('rsm', 0.0001, 1),
#     'nan_mode':                     hp.choice('nan_mode', ['Forbidden', 'Min', 'Max'])
    'fold_permutation_block_size':  hp.quniform('fold_permutation_block_size', 0, 1000, 10),
    'leaf_estimation_method' :      hp.choice('leaf_estimation_method', ['Newton', 'Gradient']),
    'leaf_estimation_backtracking': hp.choice('leaf_estimation_backtracking', ['No', 'AnyImprovement', 'Armijo']),
#     'fold_len_multiplier':          hp.uniform('fold_len_multiplier', 1.5, 10),  # Can slowdown
#     'approx_on_full_history':       hp.choice('approx_on_full_history', [True, False]),  # Can slowdown
#     'class_weights':                [1, 1.52],
#   'scale_pos_weight':             1.52,
#     'boosting_type':                hp.choice('boosting_type', ['Ordered', 'Plain']),
#     'allow_const_label':            hp.choice('allow_const_label', [True, False])
}

tpe_algorithm = tpe.suggest
bayes_trials = Trials()
best = fmin(fn = objective, space=space, algo=tpe.suggest, max_evals=1000, trials=bayes_trials)

with open('cb_params.json', 'w', encoding='utf8') as file:
    json.dump(best, file, ensure_ascii=False, indent=2)


# In[ ]:


train, test = preprocess(data, test_data, cat_features=['hero_id'], drop_encoded=False, make_features=0, highest_corr=1)
train, val = train_test_split(train, test_size=0.2)
X_train, X_val, Y_train, Y_val = train.drop('skilled', axis=1), val.drop('skilled', axis=1), train['skilled'], val['skilled']
model = CatWrapper(SVC, 'skilled', 'hero_id', {'probability': False})
model.fit(X_train, Y_train)


# In[ ]:


accuracy_score(Y_val, model.predict(X_val))
# model.predict_proba(X_val).iplot('hist', barmode='stack')


# In[ ]:


ts_data = pd.read_csv(csv_path+'rnn_train.csv', index_col='id')
ts_data['hero_id'] = data['hero_id']
ts_data['ability_upgrades'] = ts_data['ability_upgrades'].apply(eval)
ts_data['level_up_times'] = ts_data['level_up_times'].apply(eval)
ts_data['item_purchase_log'] = ts_data['item_purchase_log'].apply(eval)

items_ohe = OneHotEncoder(sparse=False).fit(items['item_id'].values.reshape(-1, 1))
items_columns = [f'item_id_{i}' for i in items['item_id']]

abilities_ohe = OneHotEncoder(sparse=False)

def convert_items(items):
    items = pd.DataFrame(items)
    transformed = items_ohe.transform(items['item_id'].astype(int).values.reshape(-1, 1)).reshape(items.shape[0], -1)
    return pd.concat([items['time'], pd.DataFrame(transformed, columns=items_columns)], axis=1)


# In[ ]:


new_ts = dict()
for hero, _data in tqdm.tqdm_notebook(ts_data.groupby('hero_id', as_index=False)):
    _data['item_purchase_log'] = _data['item_purchase_log'].apply(convert_items)
    _data['ability_upgrades'] new_ts[hero]_data['ability_upgrades'].apply(pd.Series).apply(lambda x: pd.get_dummies(x, prefix='abi'), axis=1)
    new_ts[hero] = _data
    
with open('train_timeseries.pickle', 'wb') as file:
    pickle.dump(new_ts, file)
    
html = '<a href={train_timeseries.pickle}>Download Pickle file</a>'
html = html.format(title=title,filename=filename)
HTML(html)


# In[ ]:


match_data = pd.DataFrame(ts_data.iloc[0]['item_purchase_log'])
# pd.get_dummies(match_data['item_id'].astype(int), prefix='item_id')

