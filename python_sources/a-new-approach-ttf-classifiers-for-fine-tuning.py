#!/usr/bin/env python
# coding: utf-8

# ## Classifier Accuracy Across TTF Range
# ---
# 
# The purpose of this simple kernel is to evaluate how accurately a series of LightGBM classifiers can predict that `time_to_failure` lies within a specified range, and how this accuracy is affected by the size of TTF. Which are the easier and harder parts of the data to successfully predict? More importantly, I wanted to investigate a completely new approach to this competition. 
# 
# So far, every kernel I have seen has worked on the principal of minimising MAE within their ML model, in accordance with the competition objectives. Is it possible that by reframing the nature of our target, we can indirectly achieve better results?
# 
# For simplicity and reproducibility, this kernel uses Andrew's Features only. You may find better results with your own data.

# In[ ]:


import numpy as np 
import pandas as pd 
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#I took these from a private kernel of mine but you can add Andrew's Features yourself
train = pd.read_csv('../input/andrews-features/train_X.csv')
test = pd.read_csv('../input/andrews-features/test_X.csv')
y_train = pd.read_csv('../input/andrews-features/train_y.csv').values.flatten()


# This quick section identifies the 'overlaps' between earthquake periods so these segments can be removed. It also creates a sequential column for the earthquake number that you can use for subsetting.

# In[ ]:


#remove overlapping segments
def sequential(df, col='time_to_failure', newcol='quake_id', overlap='remove_idx'):
    df[newcol] = np.zeros(len(df))
    df[overlap] = np.zeros(len(df))
    for i in range(1, len(df)):
        if df.loc[i, col] > df.loc[i-1, col]:
            df.loc[i, newcol] = df.loc[i-1, newcol] + 1
            df.loc[i, overlap] = 1
        else:
            df.loc[i, newcol] = df.loc[i-1, newcol]
    return(df)   

train['time_to_failure'] = y_train
train = sequential(train)
print(train.quake_id.describe())
print('Total number of overlapping segments: ', train.remove_idx.sum())


# 17 distinct quakes and 16 overlapping segments, as expected. The overlapping rows can be removed, and the array of quake ids kept. You can repeat this process for your own work or just copy the subsetting arrays from this kernel.

# In[ ]:


#keep only non-overlapping segments
keep_index = train.loc[train.remove_idx != 1, :].index
train = train.iloc[keep_index].reset_index(drop=True)
y_train = y_train[keep_index]

#save quake ids as numpy array and remove unnecessary columns
quake_ids = train['quake_id'].values
np.save('quake_ids.npy', quake_ids)
np.save('keep_index.npy', keep_index)
train.drop(['remove_idx', 'time_to_failure', 'quake_id'], axis=1, inplace=True)


# Now we can run a 5-fold classifier to identify whether TTF lies in a given range of 0.5 seconds. We'll run this function in a loop so the probability of TTF is evaluated for each row across the entire TTF range in train.csv 

# In[ ]:


N_FOLD = 5
SEP = 0.5

folds = KFold(n_splits=N_FOLD, shuffle=True, random_state=42)
feature_importance = np.zeros(len(train.columns))

train_preds = pd.DataFrame()
test_preds = pd.DataFrame()

def ttf_classifier(threshold, X, X_test, df, test_df, y=y_train, sep=SEP, feature_importance=feature_importance):
    
    #y == 1 if TTF lies in specific range
    y = np.logical_and(y >= threshold, y < threshold + sep)
    models = []
    oof = []
    
    for train_index, test_index in folds.split(y):
        
        X_train, X_val = X.iloc[train_index], X.iloc[test_index]
        y_train, y_val = y[train_index], y[test_index]
        
        #make and fit simple classifier model
        model = lgb.LGBMRegressor(n_estimators = 50000, n_jobs = -1)
        model.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric='auc',
          verbose=0, early_stopping_rounds=500)
        
        models.append(model)
        oof.append(roc_auc_score(y_val, model.predict(X_val)))
        feature_importance += model.feature_importances_/N_FOLD
    
    preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    #predictions for train and test with these models
    #column names lie halfway through the range, i.e. for y==1 at 3s to 4s, column name is '3.5'
    for model in models:
        preds += model.predict(X)
        preds /= len(models)
        df[str(threshold + sep/2)] = preds
        
        test_preds += model.predict(X_test)
        test_preds /= len(models)
        test_df[str(threshold + sep/2)] = test_preds
    
    #return the AUC for the combined classifiers at the target range            
    return(np.asarray(oof).mean())


# I'll be trying with a range of 0.5 seconds but this is easily modifiable for your own work. 

# In[ ]:


auc_af = []
thresh = []
#round maximum value down to smallest multiple of 0.5
MAX_Y = np.floor(y_train.max()*2)/2

for i in np.arange(0, MAX_Y, SEP):
    auc_af.append(ttf_classifier(i, X=train, X_test=test, df=train_preds, test_df=test_preds))
    thresh.append(i)   


# First let's take a look at the feature importances to see if anything is radically different for the classifiers:

# In[ ]:


feature_importance /= len(auc_af)
feat_df = pd.DataFrame({'feature' : train.columns,
                       'importance' : feature_importance}).sort_values('importance', ascending=False)

plt.figure(figsize=(16, 32));
sns.barplot(x='importance', y='feature', data=feat_df.iloc[:, :])
plt.title('Mean Feature Importance')
plt.show()


# These look similar to the feature importances Andrew's Features data usually produce. 
# 
# How well do the classifiers work at each 1s interval of `time_to_failure`? 

# In[ ]:


auc_af_df = pd.DataFrame({'threshold' : np.asarray(thresh) + SEP/2,
                      'auc' : np.asarray(auc_af)})

auc_af_df.plot(kind='line', x = 'threshold', y='auc', figsize=(15, 6))
plt.title('AUC for predicting TTF lies in bin of width 1s')
plt.xlabel('TTF')


# It looks like high TTF values are the easiest to identify. This dips around the central values, and the AUC climbs back up as TTF approaches 0, although it trails off at the end. At the extreme end of TTF there just aren't enough samples to reliably make predictions.
# 
# My current understanding of the data leads me to conclude:
# 
# * Extremely low TTF, less than 1s, is harder to predict due to the gap between the slip/earthquake event and the measure of failure occuring. The acoustic profile of the system after the quake at this stage is similar to that at a high-TTF.
# * The 6-8s range is the most challenging for models due to the earthquake segments that contain 'micro-quakes' at this time
# 
# Now we can evaluate how well the classifiers actually performed:

# In[ ]:


#return column name with maximum probability for each row for train and tets
train_preds['Pred'] = train_preds.apply(lambda x: x.argmax(), axis=1)
train_preds['Pred'] = train_preds['Pred'].astype('float16')

test_preds['Pred'] = test_preds.apply(lambda x: x.argmax(), axis=1)
test_preds['Pred'] = test_preds['Pred'].astype('float16')


# In[ ]:


train_preds['idx'] = train_preds.index
train_preds['ttf'] = y_train
ax = train_preds.plot(kind='scatter',x='idx', y='Pred', figsize=(15, 6), s=1.5, color='b')
train_preds.plot(kind='scatter',x='idx',y='ttf', figsize=(15, 6), s=0.75, color='r', ax=ax)
plt.ylabel('TTF')
plt.title('Predictions/Actual')
plt.show()

print('MAE for classifier: ', mean_absolute_error(y_train, train_preds['Pred'].values))


# The model evaluation graph looks awful! But the model itself is surprisingly (suspiciously) accurate. The graph above actually misrepresents how well the classifiers performed, since the accurate predictions are clumped together and the outliers are visually more obvious.
# 
# Most competitors have found that their LB score was significantly better than their train-CV score. Sadly this is not the case for the classifiers, and you can expect an LB score of around 2 with this approach. Still, this method may have its uses. For example, it was very good at identifying the upper range of `time_to_failure`, and could be used in conjunction with your other models, which in my experience are more conservative at predicting either a very high or very low TTF. 

# In[ ]:


sub = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv')
sub['time_to_failure'] = test_preds['Pred'].values
sub.to_csv('utterly_dreadful_classifier_submission.csv', index=False)


# ### TTF Inequality Classifiers
# ---
# What if instead we wanted to separate the data into two distinct ranges and run different models/algorithms on them? How accurately can we predict whether a segment is above or below a certain threshold? An even simpler variant of the above code can be used:

# In[ ]:


def ttf_ineq_classifier(threshold, X=train, y=y_train):
    
    y = y >= threshold
    models = []
    oof = []
    
    for train_index, test_index in folds.split(y):
        
        X_train, X_val = X.iloc[train_index], X.iloc[test_index]
        y_train, y_val = y[train_index], y[test_index]
        
        model = lgb.LGBMRegressor(n_estimators = 50000, n_jobs = -1)
        model.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric='auc',
          verbose=0, early_stopping_rounds=500)
        models.append(model)
        
        oof.append(roc_auc_score(y_val, model.predict(X_val)))
                           
    return(np.asarray(oof).mean())


# In[ ]:


SEP=0.1
auc_ineq = []
thresh = []
for i in np.arange(SEP, MAX_Y-1, SEP):
    auc_ineq.append(ttf_ineq_classifier(i))
    thresh.append(i)
    
auc_ineq_df = pd.DataFrame({'threshold' : np.asarray(thresh),
                      'auc' : np.asarray(auc_ineq)})

auc_ineq_df.plot(kind='line', x = 'threshold', y='auc', figsize=(15, 6))
plt.title('AUC for predicting TTF > threshold')


# This corroborates the earlier findings that TTF is easiest to classify at its highest values. This also provides a more nuanced view of the TTF < 2s zone. While the basic classifiers found it hardest to identify TTF smaller than 2, once *extremely* close to failure, it is in fact a lot easier to identify. Strangely, the wrapper for LGBMClassifier worked far worse than the wrapper for LGBMRegressor. With the objective set to 'auc' this distinction largely vanishes, so this may be a product of the different hyperparameters they start with. 
# 
# Hopefully these results will be of some use to you - remember that this kernel only used Andrew's Features so your own engineered data may yield more helpful results. Good luck!
