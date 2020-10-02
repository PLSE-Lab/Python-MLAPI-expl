#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

import warnings
warnings.simplefilter('ignore')

import gc

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Load and prepare data

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ntrain_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv', index_col='TransactionID')\ntrain_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv', index_col='TransactionID')\ntrain = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)\ndel train_transaction, train_identity\ngc.collect()")


# In[ ]:


train = train.fillna(-999)


# In[ ]:


many_same_values_columns_train = [c for c in train.columns if train[c].value_counts(normalize=True).values[0] > 0.9]
columns_to_drop = many_same_values_columns_train
columns_to_drop.remove('isFraud')
train = train.drop(columns_to_drop, axis=1)
gc.collect()


# In[ ]:


# From https://www.kaggle.com/pavelvpster/ieee-fraud-eda-lightgbm-baseline

from sklearn.preprocessing import LabelEncoder

def encode_categorial_features_fit(df, columns_to_encode):
    encoders = {}
    for c in columns_to_encode:
        if c in df.columns:
            encoder = LabelEncoder()
            encoder.fit(df[c].astype(str).values)
            encoders[c] = encoder
    return encoders

def encode_categorial_features_transform(df, encoders):
    out = pd.DataFrame(index=df.index)
    for c in encoders.keys():
        if c in df.columns:
            out[c] = encoders[c].transform(df[c].astype(str).values)
    return out

categorial_features_columns = [
    'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21',
    'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31',
    'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38',
    'DeviceType', 'DeviceInfo', 'ProductCD', 'P_emaildomain', 'R_emaildomain',
    'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
    'addr1', 'addr2',
    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
    'P_emaildomain_vendor', 'P_emaildomain_suffix', 'P_emaildomain_us',
    'R_emaildomain_vendor', 'R_emaildomain_suffix', 'R_emaildomain_us'
]

categorial_features_encoders = encode_categorial_features_fit(train, categorial_features_columns)
temp = encode_categorial_features_transform(train, categorial_features_encoders)
columns_to_drop = list(set(categorial_features_columns) & set(train.columns))
train = train.drop(columns_to_drop, axis=1).merge(temp, how='left', left_index=True, right_index=True)
del temp
gc.collect()


# In[ ]:


# From https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
#        else:
#            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

train = reduce_mem_usage(train)


# In[ ]:


y_train = train['isFraud'].copy()
x_train = train.drop('isFraud', axis=1)

del train

gc.collect()


# In[ ]:


x_train.shape


# ## Feature selection

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
import lightgbm as lgb


# In[ ]:


percent = 25 # percent of dataset to use for feature selection

x_train_train, x_train_valid, y_train_train, y_train_valid = train_test_split(x_train, y_train, test_size=1.0-percent/100.0, random_state=42)

del x_train
del y_train

gc.collect()


# In[ ]:


params = {
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': False,
    'boost_from_average': True,
    'num_threads': 4,
    
    'num_leaves': 500,
    # 'min_data_in_leaf': 25,
    'max_depth': -1,
    'learning_rate': 0.01
}

# n_jobs=-1 causes out of memory
feature_selector = RFECV(lgb.LGBMClassifier(**params), step=10, scoring='roc_auc', cv=5, verbose=1)
feature_selector.fit(x_train_train, y_train_train)
print('Features selected:', feature_selector.n_features_)


# In[ ]:


selected_features = [f for f in x_train_train.columns[feature_selector.ranking_ == 1]]

selected_features


# ## Compare score (all features vs selected features)

# In[ ]:


from sklearn.metrics import roc_auc_score


# In[ ]:


percent = 25 # percent of dataset to use for validation

_, x_train_valid_part, _, y_train_valid_part = train_test_split(x_train_valid, y_train_valid, test_size=percent/100.0, random_state=42)


# In[ ]:


lgb_model = lgb.LGBMClassifier(**params).fit(x_train_train, y_train_train)
y_all_features = lgb_model.predict_proba(x_train_valid_part)[:,1]
score_all_features = roc_auc_score(y_train_valid_part, y_all_features)
print('Score / all features:', score_all_features)


# In[ ]:


y_selected_features = feature_selector.estimator_.predict_proba(x_train_valid_part[selected_features])[:,1]
score_selected_features = roc_auc_score(y_train_valid_part, y_selected_features)
print('Score / selected features:', score_selected_features)


# In[ ]:


print('Score change:', score_selected_features-score_all_features)


# ## Compare feature importance and feature selection results

# In[ ]:


feature_importance_df = pd.concat([
    pd.Series(x_train_train.columns),
    pd.Series(lgb_model.feature_importances_)], axis=1)
feature_importance_df.columns = ['featureName', 'importance']

feature_importance_df['selected'] = feature_importance_df['featureName'].map(lambda x: x in selected_features)


# In[ ]:


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(feature_importance_df.sort_values(by=['importance'], ascending=False))


# ## Confusion matrix

# In[ ]:


# From https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# In[ ]:


classes = pd.Series([0,1])

plot_confusion_matrix(y_train_valid_part, y_all_features.round(), classes=classes, normalize=True,
                      title='Confusion matrix / all features')

plt.show()


# In[ ]:


plot_confusion_matrix(y_train_valid_part, y_selected_features.round(), classes=classes, normalize=True,
                      title='Confusion matrix / selected features')

plt.show()


# *) Lower values for 1-0 and 0-1 mean better performance

# ## ROC curve

# In[ ]:


# From https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
#  and https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr_all_features, tpr_all_features, _ = roc_curve(y_train_valid_part, y_all_features.round())
roc_auc_all_features = auc(fpr_all_features, tpr_all_features)

fpr_selected_features, tpr_selected_features, _ = roc_curve(y_train_valid_part, y_selected_features.round())
roc_auc_selected_features = auc(fpr_selected_features, tpr_selected_features)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr_all_features, tpr_all_features, 'g', label = 'AUC (all features) = %0.2f' % roc_auc_all_features)
plt.plot(fpr_selected_features, tpr_selected_features, 'b', label = 'AUC (selected features) = %0.2f' % roc_auc_selected_features)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

