#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[ ]:


train_df.head()


# In[ ]:


features = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']


# In[ ]:


total = len(train_df)
(train_df.groupby('Survived')['PassengerId'].count() / total).plot.bar()


# A slightly skewed distribution.

# In[ ]:


print("nulls: {0}".format(train_df['Pclass'].isnull().sum() / len(train_df)))
pd.crosstab(train_df['Pclass'], train_df['Survived'], margins=True)


# In[ ]:


print("nulls: {0}".format(train_df['Sex'].isnull().sum() / len(train_df)))
pd.crosstab(train_df['Sex'], train_df['Survived'], margins=True)


# In[ ]:


print("nulls: {0}".format(train_df['Age'].isnull().sum() / len(train_df)))


# In[ ]:


train_df.boxplot(column='Age', by="Survived")


# As this variable presents NaNs let's create a dummy to inform the model this.

# In[ ]:


train_df['NaN_Age'] = train_df['Age'].isnull()
features.append('NaN_Age')
pd.crosstab(train_df['NaN_Age'], train_df['Survived'], margins=True)


# In[ ]:


print("nulls: {0}".format(train_df['SibSp'].isnull().sum() / len(train_df)))
train_df.boxplot(column='SibSp', by="Survived")


# In[ ]:


print("nulls: {0}".format(train_df['Parch'].isnull().sum() / len(train_df)))
pd.crosstab(train_df['Parch'], train_df['Survived'], margins=True)


# In[ ]:


print("nulls: {0}".format(train_df['Fare'].isnull().sum() / len(train_df)))
train_df.boxplot(column='Fare', by="Survived")


# In[ ]:


print("nulls: {0}".format(train_df['Embarked'].isnull().sum() / len(train_df)))
pd.crosstab(train_df['Embarked'], train_df['Survived'], margins=True)


# As this variable presents NaNs let's create a dummy to inform the model this.

# In[ ]:


train_df['NaN_Embarked'] = train_df['Embarked'].isnull()
features.append('NaN_Embarked')
pd.crosstab(train_df['Survived'], train_df['NaN_Embarked'], margins=True)


# In[ ]:


print("nulls: {0}".format(train_df['Cabin'].isnull().sum() / len(train_df)))
#pd.crosstab(train_df['Cabin'], train_df['Survived'], margins=True)


# A lot of NaN... 
# The Cabin is almost an ID, so we will not include it on the model.

# In[ ]:


train_df['NaN_Cabin'] = train_df['Cabin'].isnull()
features.append('NaN_Cabin')
pd.crosstab(train_df['NaN_Cabin'], train_df['Survived'], margins=True)


# # Feature Engineering

# In[ ]:


# maybe the deck of the passanger can add some information
train_df['Deck'] = train_df['Cabin'].apply(lambda x: x[0] if type(x) == str else '')
pd.crosstab(train_df['Deck'], train_df['Survived'], margins=True)
features.append('Deck')


# In[ ]:


train_df['NaN_Deck'] = train_df['Deck'].isnull()
features.append('NaN_Deck')
pd.crosstab(train_df['NaN_Deck'], train_df['Survived'], margins=True)


# # Modeling

# In[ ]:


from catboost import CatBoostClassifier, Pool, cv
from sklearn.metrics import accuracy_score


# In[ ]:


train_df[features].head()


# In[ ]:


X = X.fillna(-999)


# In[ ]:


import h2o


# In[ ]:


h2o.init()


# In[ ]:


train_df[features].to_csv('train.csv', index=False)


# In[ ]:


h2o_df = h2o.upload_file('train.csv')


# In[ ]:


categorical_vars = ['Survived', 'Pclass', 'Sex',
                    'Embarked', 'NaN_Age', 'NaN_Embarked', 
                    'NaN_Cabin', 'Deck']
train, test = h2o_df.split_frame(ratios=[0.75])
for col in categorical_vars:
    train[col] = train[col].asfactor()
    test[col] = test[col].asfactor()


# In[ ]:


len(train), len(test)


# In[ ]:


response_var = 'Survived'
model_features = [col for col in features if col not in [response_var]]


# In[ ]:


from h2o.estimators.random_forest import H2ORandomForestEstimator


# In[ ]:


naive_rf_model = H2ORandomForestEstimator()


# In[ ]:


naive_rf_model.train(x=features,
                     y=response_var,
                     training_frame=train,
                     validation_frame=test)


# In[ ]:


naive_rf_model.varimp_plot()


# Almost all variables contributes.

# In[ ]:


# Aux functions to metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import itertools
import matplotlib.pyplot as plt

def find_optimal_cutoff(fpr, tpr, threshold):
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i),
                        'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    return list(roc_t['threshold'])[0]

def plot_roc_curve(y_true, y_pred):
    fpr, tpr, thresholds_test = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    fig, ax1 = plt.subplots()
    lw = 2.
    ax1.plot(fpr, tpr, color='darkorange', lw=lw)
    ax1.scatter(fpr, tpr, color='red', lw=lw)

    ax1.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax1.set_xlim([-0.01, 1.05])
    ax1.set_ylim([-0.01, 1.05])

    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC curve (area = %0.2f)' % roc_auc)

    fig.tight_layout()
    plt.show()
    threshold_test = find_optimal_cutoff(fpr, tpr, thresholds_test)
    print (pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': thresholds_test}).query("threshold=={}".format(threshold_test)).iloc[-1])

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:


preds_train = naive_rf_model.predict(train).as_data_frame()
aux = train.as_data_frame()
preds_train['y_true'] = aux['Survived']
preds_train.columns = ['y_pred', 'p_survived', 'p_died', 'y_true']


# In[ ]:


plot_roc_curve(preds_train['y_true'],
               preds_train['p_died'])


# In[ ]:


cnf_matrix = confusion_matrix(preds_train['y_true'],
                              preds_train['y_pred'])
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix,
                      classes=['Died', 'Survived'],
                      normalize=True,
                      title='Normalized confusion matrix')


# ## Metrics on test set

# In[ ]:


preds_test = naive_rf_model.predict(test).as_data_frame()
aux = test.as_data_frame()
preds_test['y_true'] = aux['Survived']
preds_test.columns = ['y_pred', 'p_survived', 'p_died', 'y_true']


# In[ ]:


plot_roc_curve(preds_test['y_true'], preds_test['p_died'])


# In[ ]:


cnf_matrix = confusion_matrix(preds_test['y_true'], preds_test['y_pred'])
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix,
                      classes=['Died', 'Survived'],
                      normalize=True,
                      title='Normalized confusion matrix')


# ## Grid search :)

# In[ ]:


from h2o.grid.grid_search import H2OGridSearch

hyper_parameters = {
    'ntrees': [5, 10, 50, 100, 150],
    'max_depth':[5, 10, 20, 50, 100],
    'balance_classes' : [True, False]
}

grid_search = H2OGridSearch(H2ORandomForestEstimator(),
                            hyper_params=hyper_parameters)

grid_search.train(x=features,
                  y=response_var,
                  training_frame=train,
                  validation_frame=test)


# Maybe we can achieve better results with different #trees and #depth. Also balance classes can improve the performance, as the distribution is skewed.

# In[ ]:


# Configs for Pandas output
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:,.15f}'.format

grid_search.sorted_metric_table().head()


# In[ ]:


# get the better model
grid_search_model = h2o.get_model(grid_search.sorted_metric_table().head().values[0][-2])


# In[ ]:


grid_search_model.varimp_plot()


# Almost all variables contributes. It is expect that Sex and Age are the most heavy contributors, as child and women goes first.

# ## Metrics on train set

# In[ ]:


preds_train = grid_search_model.predict(train).as_data_frame()
aux = train.as_data_frame()
preds_train['y_true'] = aux['Survived']
preds_train.columns = ['y_pred', 'p_survived', 'p_died', 'y_true']


# In[ ]:


plot_roc_curve(preds_train['y_true'], preds_train['p_died'])


# In[ ]:


cnf_matrix = confusion_matrix(preds_train['y_true'], preds_train['y_pred'])
plot_confusion_matrix(cnf_matrix,
                      classes=['Died', 'Survived'],
                      normalize=True,
                      title='Normalized confusion matrix')


# ## Metrics on test set

# In[ ]:


preds_test = grid_search_model.predict(test).as_data_frame()
aux = test.as_data_frame()
preds_test['y_true'] = aux['Survived']
preds_test.columns = ['y_pred', 'p_survived', 'p_died', 'y_true']


# In[ ]:


plot_roc_curve(preds_test['y_true'], preds_test['p_died'])


# In[ ]:


cnf_matrix = confusion_matrix(preds_test['y_true'], preds_test['y_pred'])
plot_confusion_matrix(cnf_matrix,
                      classes=['Died', 'Survived'],
                      normalize=True,
                      title='Normalized confusion matrix')


# In[ ]:


from h2o.automl import H2OAutoML
aml = H2OAutoML(max_runtime_secs = 60 * 5) # run for 5 minutes
aml.train(x=features,
          y=response_var,
          training_frame=train,
          leaderboard_frame=test)

# View the AutoML Leaderboard
lb = aml.leaderboard


# In[ ]:


lb


# ## Metrics on test set

# In[ ]:


preds_test = aml.leader.predict(test).as_data_frame()
aux = test.as_data_frame()
preds_test['y_true'] = aux['Survived']
preds_test.columns = ['y_pred', 'p_survived', 'p_died', 'y_true']


# In[ ]:


plot_roc_curve(preds_test['y_true'],
               preds_test['p_died'])


# In[ ]:


cnf_matrix = confusion_matrix(preds_test['y_true'], preds_test['y_pred'])
plot_confusion_matrix(cnf_matrix,
                      classes=['Died', 'Survived'],
                      normalize=True,
                      title='Normalized confusion matrix')


# # Predict on test
# 

# In[ ]:


test_df['Deck'] = test_df['Cabin'].apply(lambda x: x[0] if type(x) == str else '')
test_df['NaN_Cabin'] = test_df['Cabin'].isnull()
test_df['NaN_Embarked'] = test_df['Embarked'].isnull()
test_df['NaN_Age'] = test_df['Age'].isnull()
test_df['NaN_Deck'] = test_df['Deck'].isnull()


# In[ ]:


test_df[[x for x in features if x != 'Survived']].to_csv('test.csv', index=False)
h2o_df = h2o.upload_file('test.csv')
test_preds =  aml.leader.predict(h2o_df).as_data_frame()


# In[ ]:


submission = pd.DataFrame({'PassengerId' : test_df['PassengerId'], 
                           'Survived': test_preds['predict']})
submission.to_csv('submission.csv',
                  index=False)


# In[ ]:


get_ipython().system('ls submission.csv')


# # Conclusions
# 
# EDA shows some variables with high NaN rates, create a dummy var for this variables help the model to generalize.
# Grid search help us to test diferent set of parameters with few work.
# H2O facilitate our live as he deals with the categorization (no need of a dummy for each category) and gives the Grid Search to help
# 
