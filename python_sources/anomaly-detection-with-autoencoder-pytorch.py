#!/usr/bin/env python
# coding: utf-8

# # Anomaly Detection with AutoEncoder (pytorch)
# Hi! I'm new to kaggle, and this is my first competition in my life. 
# If you notice the better way about my implementation please tell me! :)
# 
# In past fraud detection competition, some people used auto encoder approach to detect anomalous for fraud data.
# There are categorical variables in the case, however, I thought the same approach can be applied to the task.
# So I just tried it.
# 
# TODO:
# - Current implementation is incorrect. training should be done by using only non fraud data.
# 
# https://www.kaggle.com/jdoz22/detecting-fraud-using-an-autoencoder-and-pytorch?scriptVersionId=2802227
# https://www.kaggle.com/artgor/eda-and-models/notebook

# In[ ]:


import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import seaborn as sns
import collections

from tqdm import tqdm, tqdm_notebook


from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score)
from sklearn.metrics import accuracy_score, precision_score
                            
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from bokeh.plotting import figure, output_notebook, show, ColumnDataSource
from bokeh.models import HoverTool, NumeralTickFormatter
from bokeh.palettes import Set3_12
from bokeh.transform import jitter

import gc
gc.enable()


# In[ ]:


# Training epochs
epochs=10


# # Preprocessing

# In[ ]:


train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')
test_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')

train_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')
test_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')

sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')

train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)

print(train.shape)
print(test.shape)


# In[ ]:


train.head()


# ## Drop columns and Standard Scaling

# In[ ]:


# Drop columns
def dropper(column_name, train, test):
    train = train.drop(column_name, axis=1)
    test = test.drop(column_name, axis=1)
    return train, test

del_columns = ['TransactionDT']
for col in del_columns:
    train, test = dropper(col, train, test)

def scaler(scl, column_name, data):
    data[column_name] = scl.fit_transform(data[column_name].values.reshape(-1,1))
    return data

scl_columns = ['TransactionAmt', 'card1', 'card3', 'card5', 'addr1', 'addr2']
for col in scl_columns:
    train = scaler(StandardScaler(), col, train)
    test = scaler(StandardScaler(), col, test)


# In[ ]:


train.head()


# In[ ]:


"""
#TODO: Learning should be done by using non fraud data
train = train[train['isFraud'] == 0]
train_fraud = train[train['isFraud'] == 1].copy()
"""


# # AutoEncoder
# ## Preprocessing

# In[ ]:


y_train = train['isFraud'].copy()
del train_transaction, train_identity, test_transaction, test_identity

# Drop target
X_train = train.drop('isFraud', axis=1)
#X_train_fraud = train_fraud.drop('isFraud', axis=1)
X_test = test.copy()

del train, test
    
# TODO: change methods
# Fill in NaNs
X_train = X_train.fillna(-999)
#X_train_fraud = X_train_fraud.fillna(-999)
X_test = X_test.fillna(-999)

# TODO: change to Label Count Endocing
# Label Encoding
for f in X_train.columns:
    if X_train[f].dtype=='object' or X_test[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X_train[f].values) + list(X_test[f].values)) #+ list(X_train_fraud[f].values))
        X_train[f] = lbl.transform(list(X_train[f].values))
        #X_train_fraud[f] = lbl.transform(list(X_train_fraud[f].values)) 
        X_test[f] = lbl.transform(list(X_test[f].values)) 
        
gc.collect()


# In[ ]:


print(X_train.head())
#print(X_train_fraud.head())
print(X_test.head())


# In[ ]:


"""
    params:
        data : data desired to be split
        ratio : validation ratio for split
        
    output:
        train_data, validation_data
"""

def splitter(data, ratio=0.2):
    num = int(ratio*len(data))
    return data[num:], data[:num]

X_train, X_val = splitter(X_train)
y_train, y_val = splitter(y_train)

# Check number of data
print(len(X_train), len(X_val), len(y_train), len(y_val))


# In[ ]:


xtr = torch.FloatTensor(X_train.values)
xts = torch.FloatTensor(X_test.values)
# X_val: validation data for isFraud == 0
xvl = torch.FloatTensor(X_val.values) 
# X_train_fraud: validation data for isFraud == 1
#xvt = torch.FloatTensor(X_train_fraud.values)

xdl = DataLoader(xtr,batch_size=1000)
tdl = DataLoader(xts,batch_size=1000)
vdl = DataLoader(xvl,batch_size=1000)
#fdl = DataLoader(xvt,batch_size=1000)

print(len(X_train.values), len(X_test.values), len(X_val.values)) #, len(X_train_fraud))
gc.collect()


# In[ ]:


class AutoEncoder(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.lin1 = nn.Linear(length,20)
        self.lin2 = nn.Linear(20,10)
        self.lin7 = nn.Linear(10,20)
        self.lin8 = nn.Linear(20,length)
        
        self.drop2 = nn.Dropout(0.05)
        
        self.lin1.weight.data.uniform_(-2,2)
        self.lin2.weight.data.uniform_(-2,2)
        self.lin7.weight.data.uniform_(-2,2)
        self.lin8.weight.data.uniform_(-2,2)

    def forward(self, data):
        x = F.tanh(self.lin1(data))
        x = self.drop2(F.tanh(self.lin2(x)))
        x = F.tanh(self.lin7(x))
        x = self.lin8(x)
        return x
    
def score(x):
    y_pred = model(V(x))
    x1 = V(x)
    return loss(y_pred,x1).item()


# In[ ]:


model = AutoEncoder(len(X_train.columns))
loss=nn.MSELoss()
learning_rate = 1e-2
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)
print(model)

# Utilize a named tuple to keep track of scores at each epoch
model_hist = collections.namedtuple('Model','epoch loss val_loss')
model_loss = model_hist(epoch = [], loss = [], val_loss = [])


# In[ ]:


def train(epochs, model, model_loss):
    try: c = model_loss.epoch[-1]
    except: c = 0
    for epoch in tqdm_notebook(range(epochs),position=0, total = epochs):
        losses=[]
        dl = iter(xdl)
        for t in range(len(dl)):
            # Forward pass: compute predicted y and loss by passing x to the model.
            xt = next(dl)
            y_pred = model(V(xt))
            
            l = loss(y_pred,V(xt))
            losses.append(l)
            optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model parameters
            l.backward()

            # Calling the step function on an Optimizer makes an update to its parameters
            optimizer.step()
            
        val_dl = iter(tdl)
        val_scores = [score(next(val_dl)) for i in range(len(val_dl))]
        
        model_loss.epoch.append(c+epoch)
        model_loss.loss.append(l.item())
        model_loss.val_loss.append(np.mean(val_scores))
        print(f'Epoch: {epoch}   Loss: {l.item():.4f}    Val_Loss: {np.mean(val_scores):.4f}')

train(model=model, epochs=epochs, model_loss=model_loss)


# ## Check Loss/Validation Loss

# In[ ]:


x = np.linspace(0, epochs-1, epochs)
print(model_loss.loss)
print(model_loss.val_loss)
print(x)
plt.plot(x, model_loss.loss, label="loss")
plt.legend()
plt.show()

plt.plot(x, model_loss.val_loss, label="val_loss")
plt.legend()
plt.show()

gc.collect()


# ## Validation

# In[ ]:


# Iterate through the dataloader and get predictions for each batch of the test set.
p = iter(vdl)
preds = np.vstack([model(V(next(p))).cpu().data.numpy() for i in range(len(p))])

# Create a pandas DF that shows the Autoencoder MSE vs True Labels
error_nonfraud = np.mean(np.power((X_val-preds),2), axis=1)
"""
p = iter(fdl)
preds = np.vstack([model(V(next(p))).cpu().data.numpy() for i in range(len(p))])
error_fraud = np.mean(np.power((X_train_fraud-preds),2), axis=1)

pd.DataFrame(error_fraud)
"""
error_df = pd.DataFrame(data = {'error':error_nonfraud,'true':y_val})

error_df.groupby('true')['error'].describe().reset_index()


# ### ROC_AUC

# In[ ]:


fpr, tpr, thresholds = roc_curve(error_df.true, error_df.error)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label='ROC curve (area = {})'.format(roc_auc))
plt.legend()
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)


# In[ ]:


temp_df = error_df[error_df['true'] == 0]
threshold = temp_df['error'].mean() + temp_df['error'].std()
print(f'Threshold: {threshold:.3f}')


# ### Precision Recall F1-Score

# In[ ]:


y_pred = [1 if e > threshold else 0 for e in error_df.error.values]
print(classification_report(error_df.true.values,y_pred))


# ### Plot Precision Recall

# In[ ]:


conf_matrix = confusion_matrix(error_df.true, y_pred)

sns.set(font_scale = 1.2)
plt.figure(figsize=(10, 10))
sns.heatmap(conf_matrix, xticklabels=['Not Fraud','Fraud'], yticklabels=['Not Fraud','Fraud'], annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


# ### Plot Precision Recall for each thresholds
# Changing the threshold from threshold_min to threshold_max.

# In[ ]:


plt.figure(figsize=(12, 12))
m = []
threshold_min = threshold * 0.99
threshold_max = threshold * 1.01

for thresh in np.linspace(threshold_min, threshold_max):
    y_pred = [1 if e > thresh else 0 for e in error_df.error.values]
    conf_matrix = confusion_matrix(error_df.true, y_pred)
    m.append((conf_matrix,thresh))
    
count = 0
for i in range(3):
    for j in range(3):
        plt.subplot2grid((3, 3), (i, j))
        sns.heatmap(m[count][0], xticklabels=['Not Fraud','Fraud'], yticklabels=['Not Fraud','Fraud'], annot=True, fmt="d");
        plt.title(f"Threshold - {m[count][1]:.3f}")
        plt.ylabel('True class')
        plt.xlabel('Predicted class')
        plt.tight_layout()
        count += 1
plt.show()


# In[ ]:


# Iterate through the dataloader and get predictions for each batch of the test set.
p = iter(tdl)
preds = np.vstack([model(V(next(p))).cpu().data.numpy() for i in range(len(p))])

# Create a pandas DF that shows the Autoencoder MSE vs True Labels
error = np.mean(np.power((X_test-preds),2), axis=1)


# In[ ]:


def min_max_normalization(x):
    x_min = x.min()
    x_max = x.max()
    x_norm = (x-x_min) / (x_max-x_min)
    return x_norm

# min max normalization
#error_df = pd.DataFrame(data={'isFraud':min_max_normalization(error)})
error_df = pd.DataFrame(data={'isFraud':error})

print("Num data: " + str(len(error_df)))
print("Beyond threshold num data: " + str(len(error_df[error_df['isFraud'] > threshold])))
#error_df[error_df['isFraud'] > threshold]

x_min = 3600000
x_max = 4200000
plt.hlines(threshold, x_min, x_max, "black")
plt.plot(error_df, alpha=0.3)
plt.show()


# In[ ]:


error_df = pd.DataFrame(data={'isFraud':min_max_normalization(error)})
error_df.head()


# In[ ]:


sample_submission['isFraud'] = error_df
sample_submission.to_csv('simple_autoencoder.csv')


# # To be continued
# 
# # Feature Engineering
# ## Day/Hour features from TransactionDT

# In[ ]:


# Making day/hour features. See the kernel below:
# https://www.kaggle.com/c/ieee-fraud-detection/discussion/100400#latest-579480
"""
def make_day_feature(df, offset=0, tname='TransactionDT'):
    # found a good offset is 0.58
    days = df[tname] / (3600*24)        
    encoded_days = np.floor(days-1+offset) % 7
    return encoded_days

def make_hour_feature(df, tname='TransactionDT'):
    hours = df[tname] / (3600)        
    encoded_hours = np.floor(hours) % 24
    return encoded_hours

# check train
print(make_day_feature(train, offset=0.58).value_counts().head(10))
print(make_hour_feature(train).value_counts().head(10))

train['TransactionDay'] = make_day_feature(train, offset=0.58)
train['TransactionHour'] = make_hour_feature(train)
test['TransactionDay'] = make_day_feature(test, offset=0.58)
test['TransactionHour'] = make_hour_feature(test)

"""


# In[ ]:


"""
# Delete columns
drop_true = False
if(drop_true):
    drop_col = ['V300', 'V309', 'V111', 'C3', 'V124', 'V106', 'V125', 'V315', 'V134', 'V102', 'V123', 'V316', 'V113', 'V136', 'V305', 'V110', 'V299', 'V289', 'V286', 'V318', 'V103', 'V304', 'V116', 'V298', 'V284', 'V293', 'V137', 'V295', 'V301', 'V104', 'V311', 'V115', 'V109', 'V119', 'V321', 'V114', 'V133', 'V122', 'V319', 'V105', 'V112', 'V118', 'V117', 'V121', 'V108', 'V135', 'V320', 'V303', 'V297', 'V120']
    X_train.drop(drop_col,axis=1, inplace=True)
    X_test.drop(drop_col, axis=1, inplace=True)
    X_train.head()
    
"""


# In[ ]:


"""
correlation_matrix = X_train.corr()
fig = plt.figure(figsize=(12,9))
sns.heatmap(correlation_matrix,vmax=0.8,square = True)

plt.show()

"""


# # Training
# 
# DAYS OF RESEARCH BROUGHT ME TO THE CONCLUSION THAT I SHOULD SIMPLY SPECIFY `tree_method='gpu_hist'` IN ORDER TO ACTIVATE GPU (okay jk, took me an hour to figure out, but I wish XGBoost documentation was more clear about that).

# In[ ]:


"""
from sklearn.cluster import KMeans
KMeans(n_clusters=2).fit(X_train)

clf = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=9,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    missing=-999,
    tree_method='gpu_hist'  # THE MAGICAL PARAMETER
)
"""


# ## KFold and validation

# In[ ]:


"""
def get_avg_auc_kfold(X_train, y_train, X_test, clf, NFOLDS=2, shuffle=True):
    kf = KFold(n_splits=NFOLDS, shuffle=shuffle)
    y_preds = np.zeros(X_test.shape[0])
    y_oof = np.zeros(X_train.shape[0])
    score = 0
  
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):    
        X_tr, X_vl = X_train.iloc[tr_idx, :], X_train.iloc[val_idx, :]
        y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        clf.fit(X_tr, y_tr)
        y_pred_train = clf.predict_proba(X_vl)[:,1]
        y_oof[val_idx] = y_pred_train
        print("FOLD: ",fold,' AUC {}'.format(roc_auc_score(y_vl, y_pred_train)))
        score += roc_auc_score(y_vl, y_pred_train) / NFOLDS
        y_preds+= clf.predict_proba(X_test)[:,1] / NFOLDS
    
        del X_tr, X_vl, y_tr, y_vl
        gc.collect()
    print (">> Avg AUC: ", score)
    return score

executeKFold = False
if(executeKFold):
    print(get_avg_auc_kfold(X_train, y_train, X_test, clf))
    
"""


# # Train
# ## training and visualizing all importances of futures

# In[ ]:


"""
# Training
%time clf.fit(X_train, y_train)

# Get xgBoost importances
importance_dict = {}
for import_type in ['weight', 'gain', 'cover']:
    importance_dict['xgBoost-'+import_type] = clf.get_booster().get_score(importance_type=import_type)
    
# MinMax scale all importances
importance_df = pd.DataFrame(importance_dict).fillna(0)
importance_df = pd.DataFrame(
    preprocessing.MinMaxScaler().fit_transform(importance_df),
    columns=importance_df.columns,
    index=importance_df.index
)

# Create mean column
importance_df['mean'] = importance_df.mean(axis=1)

# Plot the feature importances
importance_df.sort_values('mean').head(50).plot(kind='bar', figsize=(20, 7))

"""


# In[ ]:


"""
sample_submission['isFraud'] = clf.predict_proba(X_test)[:,1]
sample_submission.to_csv('simple_xgboost.csv')
"""

