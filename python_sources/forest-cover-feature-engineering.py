#!/usr/bin/env python
# coding: utf-8

# # Helpful notebooks
#  
#  https://www.kaggle.com/jakelj/basic-ensemble-model
#  
#  https://www.kaggle.com/c/ieee-fraud-detection/discussion/108575
#  
#  https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python
#  
#  https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
#  
#  https://www.kaggle.com/shahules/tackling-class-imbalance
# 

# In[ ]:


# Setting Up

import os
import random
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import seaborn as sns

from imblearn.over_sampling import RandomOverSampler, SMOTE

from lightgbm import LGBMClassifier

from mlxtend.classifier import StackingCVClassifier

from matplotlib import pyplot as plt

from plotly import express as px

from sklearn.ensemble import AdaBoostClassifier, IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="whitegrid")

random_state = 1
random.seed(random_state)
np.random.seed(random_state)


# In[ ]:


# Loading Data

train = pd.read_csv('../input/learn-together/train.csv', index_col='Id')
test = pd.read_csv('../input/learn-together/test.csv', index_col='Id')

X_train = train.drop('Cover_Type', axis='columns')
y_train = train['Cover_Type']

X_test = test.copy()


# In[ ]:


numericals = sorted(col for col in X_train.columns if 'Soil_Type' not in col and 'Wilderness_Area' not in col)

X_train['Soil_Type'] = sum(i * X_train['Soil_Type{}'.format(i)] for i in range(1, 41))
X_train['Wilderness_Area'] = sum(i * X_train['Wilderness_Area{}'.format(i)] for i in range(1, 5))


# # Cover_Type one vs rest

# In[ ]:


plot_features = ['Horizontal_Distance_To_Hydrology', 
                 'Horizontal_Distance_To_Roadways', 
                 'Horizontal_Distance_To_Fire_Points']

colors = sns.color_palette('deep')
sample = train.copy()

for cover in [1,2,3,4,5,6,7]:
    rest = list(set([1,2,3,4,5,6,7]) - set([cover]))
    
    sample['Cover_Type'] = train['Cover_Type'].copy()
    sample['Cover_Type'] = sample['Cover_Type'].replace(rest, 0)
    
    fig = plt.figure(figsize=(16, 12))
    palette = ['lavender', colors[cover]]
    
    for i in range(3):
        fig.add_subplot(3, 3, i+1)
        ax = sns.scatterplot(x='Elevation', 
                             y=plot_features[i], 
                             data=sample, 
                             hue='Cover_Type',
                             marker='+',
                             palette=palette)

    plt.tight_layout()
    plt.show()


# # Cover_Type pairs

# In[ ]:


plot_features = ['Horizontal_Distance_To_Hydrology', 
                 'Horizontal_Distance_To_Roadways', 
                 'Horizontal_Distance_To_Fire_Points']

colors = sns.color_palette('deep')
class_pairs = [(a, b) for a in range(1, 8) for b in range(a+1, 8)]

for a, b in class_pairs:
    sample = train[(train['Cover_Type'] == a) | (train['Cover_Type'] == b)]
    
    fig = plt.figure(figsize=(16, 12))
    palette = list(colors[i] for i in [a, b])
    
    for i in range(3):
        fig.add_subplot(3, 3, i+1)
        ax = sns.scatterplot(x='Elevation', 
                             y=plot_features[i],
                             data=sample, 
                             hue='Cover_Type', 
                             marker='+',
                             palette=palette)

    plt.tight_layout()
    plt.show()


# In[ ]:


# Get data for a Wilderness_Area

def get_wa_data(wa, X, y=None):
    X_wa = X[X['Wilderness_Area{}'.format(wa)] == 1]
    
    if y is not None:
        y_wa = y.loc[X_wa.index]
        return X_wa, y_wa
    
    return X_wa, _


# # Soil_Type associations

# In[ ]:


fig = plt.figure(figsize=(16, 20))

for wa in range(1, 5):
    X_wa, y_wa = get_wa_data(wa, X_train, y_train)
    
    soil_covers = pd.crosstab(X_wa['Soil_Type'], y_wa)
    
    missing_soils = set(range(1, 41)) - set(soil_covers.index)
          
    for soil in missing_soils:
        soil_covers.loc[soil] = 0
    
    soil_covers = soil_covers.sort_index()
    
    fig.add_subplot(1, 4, wa)
    ax = sns.heatmap(soil_covers, annot=True, cmap='BuPu', fmt='d',
                     xticklabels=soil_covers.columns, yticklabels=soil_covers.index, 
                     cbar_kws={'fraction':0.03}) 
    ax.set(xlabel='Cover Type', ylabel='Soil Type', title='Wilderness Area {}'.format(wa))
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

plt.tight_layout()
plt.show()


# # Soil_Type boxplots

# In[ ]:


for feature in numericals:
    fig = plt.figure(figsize=(16, 4))
    ax = sns.boxplot(x='Soil_Type', y=feature, data=X_train)
    plt.tight_layout()
    plt.show()


# # Numerical correlations

# In[ ]:


def plot_correlations(X, y=None, label=False):
    numericals = [col for col in X.columns if 
                  'Soil_Type' not in col and 
                  'Wilderness_Area' not in col]

    numericals = sorted(numericals)
    
    if y is None:
        corr = X[numericals].corr()
        
    else:
        # Optional target correlations
        X_cl = X[numericals].copy()
        classes = sorted(y.unique().tolist())
        
        for cl in classes:
            y_cl = (y==cl).astype(np.uint8).rename('Cover_Type' + str(cl))
            X_cl = pd.concat([X_cl, y_cl], axis='columns')
            
        corr = X_cl.corr()
        
    corr = np.around(corr, 1)
    
    # Place correlations in four bins
    corr_bin = corr.abs()
    corr_bin = corr_bin.where(corr_bin > 0.30, 0.0)

    corr_bin = corr_bin.where((corr_bin <= 0.30) | 
                              (corr_bin > 0.50), 0.50)

    corr_bin = corr_bin.where((corr_bin <= 0.50) | 
                              (corr_bin > 0.70), 0.70)

    corr_bin = corr_bin.where(corr_bin <= 0.70, 1.0)
    
    # Show binned correlation plot
    annot = corr if label else False
        
    fig = plt.figure(figsize=(12, 12))
    ax = sns.heatmap(corr_bin, annot=annot, linewidths=1, square=True,
                     cmap='BuPu', cbar_kws={'shrink':0.5})
    plt.show()


# In[ ]:


plot_correlations(X_train, y_train, label=True)


# # Correlation plots

# In[ ]:


fig = plt.figure(figsize=(14, 12))

fig.add_subplot(331)
sns.regplot(x='Hillshade_9am', y='Hillshade_3pm', data=X_train, scatter_kws={'alpha':0.05})

fig.add_subplot(332)
sns.regplot(x='Hillshade_9am', y='Aspect', data=X_train, scatter_kws={'alpha':0.05})

fig.add_subplot(333)
sns.regplot(x='Hillshade_Noon', y='Slope', data=X_train, scatter_kws={'alpha':0.05})

fig.add_subplot(334)
sns.regplot(x='Hillshade_Noon', y='Hillshade_3pm', data=X_train, scatter_kws={'alpha':0.05})

fig.add_subplot(335)
sns.regplot(x='Aspect', y='Hillshade_3pm', data=X_train, scatter_kws={'alpha':0.05})

fig.add_subplot(336)
sns.regplot(x='Elevation', y='Horizontal_Distance_To_Roadways', data=X_train, scatter_kws={'alpha':0.05})

fig.add_subplot(337)
sns.regplot(x='Horizontal_Distance_To_Fire_Points', y='Horizontal_Distance_To_Roadways', data=X_train, scatter_kws={'alpha':0.05})

fig.add_subplot(338)
sns.regplot(x='Horizontal_Distance_To_Hydrology', y='Vertical_Distance_To_Hydrology', data=X_train, scatter_kws={'alpha':0.05})

plt.tight_layout()
plt.show()


# # Cover_Type by Wilderness Area

# In[ ]:


area_covers = pd.crosstab(y_train, X_train['Wilderness_Area'])


fig = plt.figure(figsize=(8, 7))
ax = sns.heatmap(area_covers, annot=True, cmap='BuPu', fmt='d', square=True,
                 xticklabels=area_covers.columns, yticklabels=area_covers.index,
                 cbar_kws={'fraction':0.05})
ax.set(ylabel='Cover Type', xlabel='Wilderness Area')
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
plt.show()


# # Elevation

# In[ ]:


# Plot feature distribution by Wilderness_Area

def plot_distribution_wa(X, feature):
    fig = plt.figure(figsize=(16, 3))
    
    for wa in range(1, 5):
        fig.add_subplot(1, 4, wa)
        
        wilderness_area = 'Wilderness_Area{}'.format(wa)
        feature_wa = X[feature][X[wilderness_area] == 1]
        ax = sns.distplot(feature_wa)
        ax.set(title=wilderness_area)

    plt.tight_layout()
    plt.show()
    
    
# Plot feature distribution by Wilderness_Area and Cover_Type

def plot_distribution_wa_ct(X, y, feature):
    fig = plt.figure(figsize=(16, 3))
    colors = sns.color_palette('deep')
    
    for wa in range(1, 5):
        fig.add_subplot(1, 4, wa)
        
        wilderness_area = 'Wilderness_Area{}'.format(wa)
        cover_type_wa = y[X[wilderness_area] == 1]

        ctypes = sorted(cover_type_wa.unique().tolist())
        
        for ct in ctypes:
            feature_wa_ct = X[feature][(X[wilderness_area] == 1) & (y == ct)]
            ax = sns.kdeplot(feature_wa_ct, color=colors[ct-1], shade=False, label=ct)
        ax.legend()
        ax.set(title=wilderness_area)

    plt.tight_layout()
    plt.show()
  
    
# Plot feature boxplots by Wilderness_Area and Cover_Type

def boxplot_wa_ct(X, y, feature):
    fig = plt.figure(figsize=(16, 4))
    colors = sns.color_palette('deep')
    palette = dict((ct, colors[ct-1]) for ct in range(1, 8))
    
    ax_min = X[feature].min() - 5
    ax_max = X[feature].max() + 5
    
    for wa in range(1, 5):
        fig.add_subplot(1, 4, wa)
        
        wilderness_area = 'Wilderness_Area{}'.format(wa)
        feature_wa = X[feature][X[wilderness_area] == 1]
        cover_type_wa = y[X[wilderness_area] == 1]

        ax = sns.boxplot(x=cover_type_wa, y=feature_wa, palette=palette)
        ax.set(ylim=(ax_min, ax_max), title=wilderness_area)

    plt.tight_layout()
    plt.show()

    
# Plot feature distributions, violinplots and boxplots

def plot_wa_graphs(X, y, feature):
    plot_distribution_wa(X, feature)
#     plot_distribution_wa_ct(X, y, feature)
    boxplot_wa_ct(X, y, feature)


# In[ ]:


plot_wa_graphs(X_train, y_train, 'Elevation')


# # Aspect

# In[ ]:


plot_wa_graphs(X_train, y_train, 'Aspect')


# # Slope

# In[ ]:


plot_wa_graphs(X_train, y_train, 'Slope')


# # Horizontal_Distance_To_Hydrology

# In[ ]:


plot_wa_graphs(X_train, y_train, 'Horizontal_Distance_To_Hydrology')


# # Vertical_Distance_To_Hydrology

# In[ ]:


plot_wa_graphs(X_train, y_train, 'Vertical_Distance_To_Hydrology')


# # Horizontal_Distance_To_Roadways

# In[ ]:


plot_wa_graphs(X_train, y_train, 'Horizontal_Distance_To_Roadways')


# # Hillshade_9am

# In[ ]:


plot_wa_graphs(X_train, y_train, 'Hillshade_9am')


# # Hillshade_Noon

# In[ ]:


plot_wa_graphs(X_train, y_train, 'Hillshade_Noon')


# # Hillshade_3pm

# In[ ]:


plot_wa_graphs(X_train, y_train, 'Hillshade_3pm')


# # Horizontal_Distance_To_Fire_Points

# In[ ]:


plot_wa_graphs(X_train, y_train, 'Horizontal_Distance_To_Fire_Points')


# # Feature importances by Wilderness_Area

# In[ ]:


clf = RandomForestClassifier(n_estimators=100,
                             random_state=random_state,
                             n_jobs=-1)


# In[ ]:


# Plot feature importances by Wilderness_Area

def plot_importances_wa(clf, X, y):
    importances = pd.DataFrame()
    for wa in range(1, 5):
        X_wa, y_wa = get_wa_data(wa, X, y)
        
        if wa == 1:
            importances = pd.DataFrame(columns=X_wa.columns)
            
        clf = clf.fit(X_wa, y_wa)
        importances.loc[wa] = clf.feature_importances_
    
    fig = plt.figure(figsize=(16, 40))
    ax_max = max(importances.max()) + 0.01
    
    for i, col in enumerate(importances.columns):
        fig.add_subplot(15, 4, i+1)
        ax = sns.barplot(x=[1,2,3,4], y=col, data=importances)
        ax.set(xlabel='Wilderness Area', ylabel='Importance', title=col)
        ax.set(ylim=(0, ax_max))
    
    plt.tight_layout()
    plt.show()
    
    X_train = train.drop('Cover_Type', axis='columns')


# In[ ]:


X_train = train.drop('Cover_Type', axis='columns')
plot_importances_wa(clf, X_train, y_train)


# In[ ]:


def feature_importances(clf, X, y):
    clf = clf.fit(X, y)
    
    importances = pd.DataFrame({'Features': X.columns, 
                                'Importances': clf.feature_importances_})
    
    importances.sort_values(by=['Importances'], axis='index', ascending=False, inplace=True)

    fig = plt.figure(figsize=(14, 4))
    sns.barplot(x='Features', y='Importances', data=importances)
    plt.xticks(rotation='vertical')
    plt.show()


# ## Initial cross-validation accuracy and confusion matrix

# In[ ]:


def cv_performance(clf, X, y):
    # Get cross-validation scores
    scores = []
    predictions = pd.Series(dtype=y.dtype)

    splits = StratifiedKFold(n_splits=5).split(X, y)

    for trn_index, val_index in splits:
        X_trn = X.iloc[trn_index, :]
        y_trn = y.iloc[trn_index]

        X_val = X.iloc[val_index, :]
        y_val = y.iloc[val_index]

        clf = clf.fit(X_trn, y_trn)
        pred = clf.predict(X_val)
        
        pred = pd.Series(pred, index=y_val.index)
        predictions = pd.concat([predictions, pred])

        score = accuracy_score(y_val, pred)
        scores.append(score)

    predictions = predictions.sort_index()

    print('Cross-validation accuracy: {:.3f} {}'.format(np.mean(scores), 
                                                        np.around(scores, 3)))
    
    # Plot confusion matrix
    classes = sorted(y.unique().tolist())

    conf_mat = confusion_matrix(y, predictions)
    norm_conf_mat = np.around(conf_mat / conf_mat.sum(axis=1), 2)

    fig = plt.figure(figsize=(14, 6))

    fig.add_subplot(1, 2, 1)
    ax = sns.heatmap(norm_conf_mat, annot=True, cmap='OrRd', 
                     xticklabels=classes, yticklabels=classes)
    ax.set(xlabel='Predicted Class', ylabel='True Class', title='Normalized')


    fig.add_subplot(1, 2, 2)
    ax = sns.heatmap(conf_mat, annot=True, fmt='d', cmap='OrRd', 
                     xticklabels=classes, yticklabels=classes)
    ax.set(xlabel='Predicted Class', ylabel='True Class', title ='Counts')

    plt.tight_layout()
    plt.show()


# In[ ]:


cv_performance(clf, X_train, y_train)


# * Confusion between classes 1 and 2 > 0.2
# * Confusion between classes 3 and 6 > 0.1
# * Confusion between all other classes < 0.1

# # Adding new features

# In[ ]:


# https://www.kaggle.com/jakelj/basic-ensemble-model  (Some useful features)
# https://www.kaggle.com/c/ieee-fraud-detection/discussion/108575  (Feat Eng techniques)

def add_features(X_trn_, X_tst_):
    X_trn = X_trn_.copy()
    X_tst = X_tst_.copy()
    
    X = pd.concat([X_trn, X_tst])
        
    X['Aspect'] = X['Aspect'] % 360
    X['Aspect_120'] = (X['Aspect'] + 120) % 360


    X['Hydro_Elevation_sum'] = X[['Elevation',
                                  'Vertical_Distance_To_Hydrology']
                                 ].sum(axis='columns')
    
    X['Hydro_Elevation_diff'] = X[['Elevation',
                                   'Vertical_Distance_To_Hydrology']
                                  ].diff(axis='columns').iloc[:, [1]]

    X['Hydro_Euclidean'] = np.sqrt(X['Horizontal_Distance_To_Hydrology']**2 +
                                   X['Vertical_Distance_To_Hydrology']**2)

    X['Hydro_Manhattan'] = (X['Horizontal_Distance_To_Hydrology'] +
                            X['Vertical_Distance_To_Hydrology'].abs())
    
    
    X['Hydro_Distance_sum'] = X[['Horizontal_Distance_To_Hydrology',
                                 'Vertical_Distance_To_Hydrology']
                                ].sum(axis='columns')

    X['Hydro_Distance_diff'] = X[['Horizontal_Distance_To_Hydrology',
                                  'Vertical_Distance_To_Hydrology']
                                 ].diff(axis='columns').iloc[:, [1]]
    
    X['Hydro_Fire_sum'] = X[['Horizontal_Distance_To_Hydrology',
                             'Horizontal_Distance_To_Fire_Points']
                            ].sum(axis='columns')

    X['Hydro_Fire_diff'] = X[['Horizontal_Distance_To_Hydrology',
                              'Horizontal_Distance_To_Fire_Points']
                             ].diff(axis='columns').iloc[:, [1]].abs()

    X['Hydro_Fire_mean'] = X[['Horizontal_Distance_To_Hydrology',
                              'Horizontal_Distance_To_Fire_Points']
                             ].mean(axis='columns')

    X['Hydro_Fire_median'] = X[['Horizontal_Distance_To_Hydrology',
                                'Horizontal_Distance_To_Fire_Points']
                               ].median(axis='columns')
                               
    X['Hydro_Road_sum'] = X[['Horizontal_Distance_To_Hydrology',
                             'Horizontal_Distance_To_Roadways']
                            ].sum(axis='columns')

    X['Hydro_Road_diff'] = X[['Horizontal_Distance_To_Hydrology',
                              'Horizontal_Distance_To_Roadways']
                             ].diff(axis='columns').iloc[:, [1]].abs()

    X['Hydro_Road_mean'] = X[['Horizontal_Distance_To_Hydrology',
                              'Horizontal_Distance_To_Roadways']
                             ].mean(axis='columns')

    X['Hydro_Road_median'] = X[['Horizontal_Distance_To_Hydrology',
                                'Horizontal_Distance_To_Roadways']
                               ].median(axis='columns')
    
    X['Road_Fire_sum'] = X[['Horizontal_Distance_To_Roadways',
                            'Horizontal_Distance_To_Fire_Points']
                           ].sum(axis='columns')

    X['Road_Fire_diff'] = X[['Horizontal_Distance_To_Roadways',
                             'Horizontal_Distance_To_Fire_Points']
                            ].diff(axis='columns').iloc[:, [1]].abs()

    X['Road_Fire_mean'] = X[['Horizontal_Distance_To_Roadways',
                             'Horizontal_Distance_To_Fire_Points']
                            ].mean(axis='columns')

    X['Road_Fire_median'] = X[['Horizontal_Distance_To_Roadways',
                               'Horizontal_Distance_To_Fire_Points']
                              ].median(axis='columns')
    
    X['Hydro_Road_Fire_mean'] = X[['Horizontal_Distance_To_Hydrology',
                                   'Horizontal_Distance_To_Roadways',
                                   'Horizontal_Distance_To_Fire_Points']
                                  ].mean(axis='columns')

    X['Hydro_Road_Fire_median'] = X[['Horizontal_Distance_To_Hydrology',
                                     'Horizontal_Distance_To_Roadways',
                                     'Horizontal_Distance_To_Fire_Points']
                                    ].median(axis='columns')

    
    # Compute Soil_Type number from Soil_Type binary columns
    X['Soil_Type'] = sum(i * X['Soil_Type{}'.format(i)] for i in range(1, 41))
    
    # For all 40 Soil_Types, 1=rubbly, 2=stony, 3=very stony, 4=extremely stony, 0=?
    stoneyness = [4, 3, 1, 1, 1, 2, 0, 0, 3, 1, 
                  1, 2, 1, 0, 0, 0, 0, 3, 0, 0, 
                  0, 4, 0, 4, 4, 3, 4, 4, 4, 4, 
                  4, 4, 4, 4, 1, 4, 4, 4, 4, 4]
    
    # Replace Soil_Type number with "stoneyness" value
    X['Stoneyness'] = X['Soil_Type'].replace(range(1, 41), stoneyness)
    
    rocks = [1, 0, 1, 1, 1, 1, 0, 0, 0, 1,
             1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 
             0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 
             0, 1, 1, 1, 1, 1, 1, 0, 0, 1]
    
    X['Rocks'] = X['Soil_Type'].replace(range(1, 41), rocks)
    
    soil_count = X['Soil_Type'].value_counts().to_dict()
    X['Soil_count'] = X['Soil_Type'].map(soil_count)
    
    soil_elevations = X.groupby('Soil_Type')['Elevation'].median().to_dict()
    X['Soil_Elevation'] = X['Soil_Type'].map(soil_elevations)
    
    soil_aspects = X.groupby('Soil_Type')['Aspect'].median().to_dict()
    X['Soil_Aspect'] = X['Soil_Type'].map(soil_aspects)
    
    soil_slopes = X.groupby('Soil_Type')['Slope'].median().to_dict()
    X['Soil_Slope'] = X['Soil_Type'].map(soil_slopes)
    
    soil_hhydros = X.groupby('Soil_Type')['Horizontal_Distance_To_Hydrology'].median().to_dict()
    X['Soil_H_Hydro'] = X['Soil_Type'].map(soil_hhydros)
    
    soil_vhydros = X.groupby('Soil_Type')['Vertical_Distance_To_Hydrology'].median().to_dict()
    X['Soil_V_Hydro'] = X['Soil_Type'].map(soil_vhydros)
    
    soil_roads = X.groupby('Soil_Type')['Horizontal_Distance_To_Roadways'].median().to_dict()
    X['Soil_Road'] = X['Soil_Type'].map(soil_roads)
    
    soil_fires = X.groupby('Soil_Type')['Horizontal_Distance_To_Fire_Points'].median().to_dict()
    X['Soil_Fire'] = X['Soil_Type'].map(soil_fires)
    
    
    X = X.drop(['Soil_Type'], axis='columns')
            
    X_trn = X.loc[X_trn.index, :]
    X_tst = X.loc[X_tst.index, :]
        
    return X_trn, X_tst


# In[ ]:


X_train = train.drop('Cover_Type', axis='columns')
X_test = test.copy()

X_train, X_test = add_features(X_train, X_test)


# ## Feature importances with added features

# In[ ]:


feature_importances(clf, X_train, y_train)


# ## Cross-validation performance with added features

# In[ ]:


cv_performance(clf, X_train, y_train)


# # Removing low importance features

# Low importance Soil_Type features all have low variance, being almost entirely made up of zeros (or ones). The frequency of the mode is close to 100% of data size, whether the mode is 0 or 1.

# In[ ]:


# Plotting mode frequencies as % of data size
n_rows = X_train.shape[0]
mode_frequencies = [X_train[col].value_counts().iat[0]for col in X_train.columns]
mode_frequencies = 100.0 * np.asarray(mode_frequencies) / n_rows

mode_df = pd.DataFrame({'Feature': X_train.columns, 
                        'Mode_Frequency': mode_frequencies})

mode_df.sort_values(by=['Mode_Frequency'], axis='index', ascending=True, inplace=True)

fig = plt.figure(figsize=(14, 4))
ax = sns.barplot(x='Feature', y='Mode_Frequency', data=mode_df, color='b')   
plt.ylabel('Mode Frequency %')
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:


# Drop features with high mode frequencies

def drop_unimportant(X_):
    X = X_.copy()
    
    n_rows = X.shape[0]
    hi_freq_cols = []
    
    for col in X.columns:
        mode_frequency = X[col].value_counts().iat[0] 
        
        if mode_frequency > (n_rows - 10):
            hi_freq_cols.append(col)
            
    X = X.drop(hi_freq_cols, axis='columns')
    
    return X


# In[ ]:


X_train = drop_unimportant(X_train)


# In[ ]:


cv_performance(clf, X_train, y_train)


# ## New feature correlations

# In[ ]:


plot_correlations(X_train)


# ## Backward feature elimination

# Manual feature elimination - iteratively eliminate some features that have high inter-correlations with other features, ensuring that cv score does not get worse in the process.

# In[ ]:


def drop_correlated(X_):
    X = X_.copy()
    
    drop_cols = ['Hillshade_max', 'Hillshade_mean', 'Hillshade_median', 
                 'Hillshade_min', 'Hillshade_sum', 'Hydro_Elevation_sum',
                 'Hydro_Distance_sum', 'Hydro_Distance_diff', 'Hydro_Fire_mean', 
                 'Hydro_Fire_median', 'Hydro_Manhattan', 'Hydro_Road_Fire_mean', 
                 'Hydro_Road_Fire_median', 'Hydro_Road_mean', 'Hydro_Road_median', 
                 'Road_Fire_mean', 'Road_Fire_median', 'Soil_Aspect', 'Soil_Slope', 
                 'Soil_H_Hydro', 'Soil_V_Hydro', 'Soil_Road', 'Soil_Fire',
                 'Stoneyness', 'Rocks', 'Aspect_120','Aspect', 'Slope']
    
    drop_cols += ['Soil_Type{}'.format(i) for i in range(1, 41)]
        
    drop_cols = [col for col in drop_cols if col in X.columns]
    
    X = X.drop(drop_cols, axis='columns')
    
    return X


# In[ ]:


X_train = drop_correlated(X_train)


# In[ ]:


plot_correlations(X_train, y_train)


# In[ ]:


feature_importances(clf, X_train, y_train)


# In[ ]:


cv_performance(clf, X_train, y_train)


# ## Notes
# * Cross-validation accuracy improved from 78.0% to 81.8% (3.8% up).
# *  Confusion between classes 1 and 2 is high (initially 0.24 / 0.25).
# *  Adding new features has not improved the confusion by much.

# # Focusing on Wilderness_Area1

# ## [Wilderness_Area1] :: Data Preparation

# In[ ]:


# Reload data
X_train = train.drop('Cover_Type', axis='columns')
X_test = test.copy()


# In[ ]:


X_train_wa, y_train_wa = get_wa_data(1, X_train, y_train)
X_test_wa, _ = get_wa_data(1, X_test)


# ## [Wilderness_Area1] :: Initial performance

# In[ ]:


feature_importances(clf, X_train_wa, y_train_wa)


# In[ ]:


cv_performance(clf, X_train_wa, y_train_wa)


# ## [Wilderness_Area1] :: Performance after adding features

# In[ ]:


X_train_wa, X_test_wa = add_features(X_train_wa, X_test_wa)


# In[ ]:


feature_importances(clf, X_train_wa, y_train_wa)


# In[ ]:


cv_performance(clf, X_train_wa, y_train_wa)


# ## [Wilderness_Area1] :: Performance after feature removal

# In[ ]:


def drop_features(X_):
    X = X_.copy()
    
    X = drop_unimportant(X)
    X = drop_correlated(X)
    
    return X


# In[ ]:


X_train_wa = drop_features(X_train_wa)


# In[ ]:


feature_importances(clf, X_train_wa, y_train_wa)


# In[ ]:


cv_performance(clf, X_train_wa, y_train_wa)


# ## [Wilderness_Area1] :: Up-sampling to balance classes

# Classes are imbalanced in Wilderness_Area1

# In[ ]:


def plot_class_counts(y):
    fig = plt.figure(figsize=(8, 4))
    sns.countplot(y)
    plt.show()


# In[ ]:


plot_class_counts(y_train_wa)


# In[ ]:


# https://www.kaggle.com/shahules/tackling-class-imbalance

# Up-sample minority classes

def upsample(X_, y_):
    X = X_.copy()
    y = y_.copy()
    
    max_samples = y.value_counts().iat[0]
    classes = y.unique().tolist()
    sampling_strategy = dict((clas, max_samples) for clas in classes)

    sampler = SMOTE(sampling_strategy=sampling_strategy,
                    random_state=random_state)

    x_columns = X.columns.tolist()
    X, y = sampler.fit_resample(X, y)
    X = pd.DataFrame(X, columns=x_columns)
    y = pd.Series(y)
    
    return X, y


# In[ ]:


X_train_wa, y_train_wa = upsample(X_train_wa, y_train_wa)


# Classes become equally balanced after up-sampling

# In[ ]:


plot_class_counts(y_train_wa)


# ## [Wilderness_Area1] :: Performance after up-sampling

# In[ ]:


# Modify cv_performance() to include upsampling

def cv_performance(clf, X, y, resample=False):
    # Get cross-validation scores
    scores = []
    predictions = pd.Series(dtype=y.dtype)

    splits = StratifiedKFold(n_splits=5).split(X, y)

    for trn_index, val_index in splits:
        X_trn = X.iloc[trn_index, :]
        y_trn = y.iloc[trn_index]

        X_val = X.iloc[val_index, :]
        y_val = y.iloc[val_index]
        
        if resample:
            X_trn, y_trn = upsample(X_trn, y_trn)

        clf = clf.fit(X_trn, y_trn)
        pred = clf.predict(X_val)
        
        if resample:
            # Calculate scores on original samples only
            pred = pred[:len(y_val)]
            
        pred = pd.Series(pred, index=y_val.index)
        predictions = pd.concat([predictions, pred])

        score = accuracy_score(y_val, pred)
        scores.append(score)

    predictions = predictions.sort_index()

    print('Cross-validation accuracy: {:.3f} {}'.format(np.mean(scores), 
                                                        np.around(scores, 3)))
    
    # Plot confusion matrix
    classes = sorted(y.unique().tolist())

    conf_mat = confusion_matrix(y, predictions)
    norm_conf_mat = np.around(conf_mat / conf_mat.sum(axis=1), 2)

    fig = plt.figure(figsize=(14, 6))

    fig.add_subplot(1, 2, 1)
    ax = sns.heatmap(norm_conf_mat, annot=True, cmap='OrRd', 
                     xticklabels=classes, yticklabels=classes)
    ax.set(xlabel='Predicted Class', ylabel='True Class', title='Normalized')


    fig.add_subplot(1, 2, 2)
    ax = sns.heatmap(conf_mat, annot=True, fmt='d', cmap='OrRd', 
                     xticklabels=classes, yticklabels=classes)
    ax.set(xlabel='Predicted Class', ylabel='True Class', title ='Counts')

    plt.tight_layout()
    plt.show()


# In[ ]:


# Reload Wilderness_Area1 data
X_train_wa, y_train_wa = get_wa_data(1, X_train, y_train)
X_test_wa, _ = get_wa_data(1, X_test)

X_train_wa, X_test_wa = add_features(X_train_wa, X_test_wa)

X_train_wa = drop_features(X_train_wa)


# In[ ]:


cv_performance(clf, X_train_wa, y_train_wa, resample=True)


# ## Notes
#  * Cross-validation accuracy for Wilderness Area 1 improved from 74.7% to 78.8% (4.1% up).
#  * Confusion between classes 1 and 2 is high (0.22 / 0.23)
#  * Top 5 important Soil Types: 30, 38, 29, 12, 23

# # Focusing on Wilderness_Area2

# ## [Wilderness_Area2] :: Initial performance

# In[ ]:


X_train_wa, y_train_wa = get_wa_data(2, X_train, y_train)
X_test_wa, _ = get_wa_data(2, X_test)


# In[ ]:


plot_class_counts(y_train_wa)


# In[ ]:


feature_importances(clf, X_train_wa, y_train_wa)


# In[ ]:


cv_performance(clf, X_train_wa, y_train_wa)


# ## [Wilderness_Area2] :: Performance after processing

# In[ ]:


X_train_wa, X_test_wa = add_features(X_train_wa, X_test_wa)

X_train_wa = drop_features(X_train_wa)


# In[ ]:


feature_importances(clf, X_train_wa, y_train_wa)


# In[ ]:


cv_performance(clf, X_train_wa, y_train_wa, resample=True)


# ## Notes
# * Cross-validation accuracy for Wilderness Area 2 improved.
# * Confusion between classes 1 and 2 is high.
# * Class 2 has a lower count than class 1 and is harder to predict.

# # Focusing on Wilderness_Area3

# ## [Wilderness_Area3] :: Initial performance

# In[ ]:


X_train_wa, y_train_wa = get_wa_data(3, X_train, y_train)
X_test_wa, _ = get_wa_data(3, X_test)


# In[ ]:


plot_class_counts(y_train_wa)


# In[ ]:


feature_importances(clf, X_train_wa, y_train_wa)


# In[ ]:


cv_performance(clf, X_train_wa, y_train_wa)


# ## [Wilderness_Area3] :: Performance after processing

# In[ ]:


X_train_wa, X_test_wa = add_features(X_train_wa, X_test_wa)

X_train_wa = drop_features(X_train_wa)


# In[ ]:


feature_importances(clf, X_train_wa, y_train_wa)


# In[ ]:


cv_performance(clf, X_train_wa, y_train_wa, resample=True)


# ## Notes
# * Cross-validation accuracy for Wilderness Area 3 improved.
# * Confusion between classes 1 and 2 is high.

# # Focusing on Wilderness_Area4

# ## [Wilderness_Area4] :: Initial performance

# In[ ]:


X_train_wa, y_train_wa = get_wa_data(4, X_train, y_train)
X_test_wa, _ = get_wa_data(4, X_test)


# In[ ]:


plot_class_counts(y_train_wa)


# In[ ]:


feature_importances(clf, X_train_wa, y_train_wa)


# In[ ]:


cv_performance(clf, X_train_wa, y_train_wa)


# ## [Wilderness_Area4] :: Performance after processing

# In[ ]:


X_train_wa, X_test_wa = add_features(X_train_wa, X_test_wa)

X_train_wa = drop_features(X_train_wa)


# In[ ]:


feature_importances(clf, X_train_wa, y_train_wa)


# In[ ]:


cv_performance(clf, X_train_wa, y_train_wa, resample=True)


# ## Notes
# * Class 2 has a very low count compared to the others and is the hardest to predict.
# * Confusion between classes 3 and 6 is noticeable.
# * Unlike other Wilderness Areas, importances of Elevation & Hydro_Elevation_diff do not dwarf all other feature importances.
