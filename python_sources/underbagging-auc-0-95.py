#!/usr/bin/env python
# coding: utf-8

# # Random Forest and Undersampling

# In this notebook, we used Undersampling and RandomForest for developing a model for the prediction backorder problem.
# 
# Done so far:
# -----
# 1. Preprocessing - Convert categorical to binaries, normalization for quantity related attributes and imputation of missing values;
# 2. Undersampling - Random undersampling for classes balancing;
# 3. Model selection - Hyper-parameters tuning using grid search and cross-validation. Kappa is adopted as score;
# 5. Metrics - Accuracy 89.3%, Precision 6.9%, Recall 89.3%;
# 6. Feature importance analysis.
# 
# Next steps:
# ---
# 1. Combine undersampling with oversampling?

# In[ ]:


import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

pd.options.mode.chained_assignment = None

get_ipython().run_line_magic('matplotlib', 'inline')
random_state = 1


# ### Preprocessing

# Explaining some of the strategies used in the preprocessing:
# 
# - Slice the dataset in half, cutting off items with no forecast or sales in the past 3 months. This strategy allows us to reduce 50% the original dataset, loosing only 300 items
# - Binaries were converted from strings ('Yes' and 'No') to 1 and 0.
# - The attributes related to quantities were normalized (std dev equal to 1) per row. Therefore, parts with different order of magnitudes are approximated. For example: 1 unit of a expensive machine may be different from 1 unit of a screw, but if we standard deviate all the quantities we have, we can get a better proportion of equivalence between those items.
# - Missing values for lead_time and perf_month_avg were replaced using series median and mean. 

# In[ ]:


def process(df):
    df = df.loc[(df["forecast_3_month"] > 0) | (df["sales_3_month"] > 0)]
    # Convert to binaries
    for col in ['potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk',
               'stop_auto_buy', 'rev_stop', 'went_on_backorder']:
        df[col] = (df[col] == 'Yes').astype(int)
    # Normalisation
    from sklearn.preprocessing import normalize
    qty_related = ['national_inv', 'in_transit_qty', 'forecast_3_month', 
                   'forecast_6_month', 'forecast_9_month', 'min_bank',
                   'local_bo_qty', 'pieces_past_due', 'sales_1_month', 'sales_3_month', 
                   'sales_6_month', 'sales_9_month',]
    df[qty_related] = normalize(df[qty_related], axis=1)
    # Imput missing values 
    from sklearn.preprocessing import Imputer
    df['lead_time'] = Imputer(strategy='median').fit_transform(
                                    df['lead_time'].values.reshape(-1, 1))
    for col in ['perf_6_month_avg', 'perf_12_month_avg']:    
        df[col] = Imputer(missing_values=-99).fit_transform(
                                    df[col].values.reshape(-1, 1))
    return df


# In[ ]:


train = process(pd.read_csv('../input/Kaggle_Training_Dataset.csv'))
train.hist(figsize=(12,12), alpha=0.8, grid=False)
plt.show()


# Note that in the histogram, even though in preprocessing we sliced some parts considered as inactive (no forecast or sales within 3 months), we stil

# ### Data load

# In[ ]:


X = train.drop(['sku','went_on_backorder'], axis=1)
y = train['went_on_backorder']
print('Original dataset shape {}'.format(Counter(y)))


# ### Undersampling

# In[ ]:


from imblearn.under_sampling import RandomUnderSampler
X_train, y_train = RandomUnderSampler(random_state=random_state, ratio=0.025).fit_sample(X, y)
print('Resampled dataset shape {}'.format(Counter(y_train)))


# ### Model selection

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=random_state)
parameters = {
#    'n_estimators': [30, 50],
    'max_depth': [10, 20, 30],
    'max_features': [3, 4, 5,], # feature bagging
#    'samples_leaf': [1, 2, 3,],
#    "bootstrap": [False], #True
#    "criterion": ["entropy", "gini"],
}

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, cohen_kappa_score
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
gs = GridSearchCV(clf, parameters, cv=kf, n_jobs=-1, verbose=1, 
                  scoring=make_scorer(cohen_kappa_score))
gs.fit(X_train, y_train)

print("Best score: %0.3f" % gs.best_score_)
print("Best parameters set:")
best_parameters = gs.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))


# ### Results
# Testing the model generalization capacity in the test dataset.

# In[ ]:


test = process(pd.read_csv('../input/Kaggle_Test_Dataset.csv'))
X_test = test.drop(['sku','went_on_backorder'], axis=1)
y_test = test['went_on_backorder']
y_pred = gs.predict(X_test)

from sklearn.metrics import classification_report
print (classification_report(y_test, y_pred, digits=3))

from sklearn.metrics import accuracy_score, precision_score, recall_score
print ("\tAS:\t %.3f\n\tPS:\t %.3f\n\tRC:\t %.3f\n" % (accuracy_score(y_test, y_pred), 
                precision_score(y_test, y_pred), recall_score(y_test, y_pred)))


# ### Confusion matrix

# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(100*cm/float(cm.sum()))
ax = sns.heatmap(df_cm.round(2), annot=True, cmap='Greys', fmt='g', linewidths=1)
ax.set_title("Confusion Matrix - per 100 predictions")
ax.set_xlabel('Predicted', fontsize=16)
ax.set_ylabel('True', fontsize=16, rotation=90)
plt.show()


# ### Feature importances

# In[ ]:


fig, ax = plt.subplots(figsize=(12,7))
plt.barh(np.arange(X.shape[1]), gs.best_estimator_.feature_importances_, alpha=0.8)
ax.set_yticks(np.arange(X.shape[1]))
ax.set_yticklabels(X.columns.values)
plt.show()

