#!/usr/bin/env python
# coding: utf-8

# # Introduction: Deep Learning with Embedding Layers
# 
# 
# This notebook is intended for those who want an introduction into Embedding Layers with Keras. I choosed not to focus on describing the preprocessing nor the different methods, to merge all the table, but rather to focus more specificaly on how to get started in Embedding.
# 
# Embedding is a technique used to encode categorical features like One-Hot encoding or target encoding, it is a bit more difficult to implement but keras allow us to create a model pretty easily.
# 
# Embeddings help to generalize better when the data is sparse and statistics is unknown. Thus, it is especially useful for datasets with lots of high cardinality features, where other methods tend to overfit.
# 
# Why should we use Entity Embedding instead of One-Hot Encoding ? There are mutiple reasons for that :
# 
# *  One-Hot encoded vectors are high-dimensional and sparse. In this dataset we have a feature that represent an organization type (denoted: ORGANIZATION_TYPE) of 58 distinct value . This means that, when using one-hot encoding, this feature will be represented by a vector containing 58 integers. And 57 of these integers are zeros. In a big dataset or in NLP ( Natural Language Processing) when you have more than 2000 outcomes for a feature, this approach is not computationally efficient.
# 
# 
# * The vectors of each embedding get updated while training the neural network. This allows us to visualize relationships between words or more generally speaking categories, but also between everything that can be turned into a vector through an embedding layer. Please look at the image below  that show how similarities between categories can be found in a multi-dimensional space.
# 
# ![](https://cdn-images-1.medium.com/max/1000/1*sXNXYfAqfLUeiDXPCo130w.png)

# In[208]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import gc
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Merge, merge, Reshape, Dropout, Input, Flatten, Concatenate
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[209]:


def display_roc_curve(y_, oof_preds_, folds_idx_):
    # Plot ROC curves
    plt.figure(figsize=(6,6))
    scores = [] 
    for n_fold, (_, val_idx) in enumerate(folds_idx_):  
        # Plot the roc curve
        fpr, tpr, thresholds = roc_curve(y_.iloc[val_idx], oof_preds_[val_idx])
        score = roc_auc_score(y_.iloc[val_idx], oof_preds_[val_idx])
        scores.append(score)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (n_fold + 1, score))
    
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    fpr, tpr, thresholds = roc_curve(y_, oof_preds_)
    score = roc_auc_score(y_, oof_preds_)
    plt.plot(fpr, tpr, color='b',
             label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
             lw=2, alpha=.8)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Embedding Neural Network ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    
def display_precision_recall(y_, oof_preds_, folds_idx_):
    # Plot ROC curves
    plt.figure(figsize=(6,6))
    
    scores = [] 
    for n_fold, (_, val_idx) in enumerate(folds_idx_):  
        # Plot the roc curve
        fpr, tpr, thresholds = roc_curve(y_.iloc[val_idx], oof_preds_[val_idx])
        score = average_precision_score(y_.iloc[val_idx], oof_preds_[val_idx])
        scores.append(score)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='AP fold %d (AUC = %0.4f)' % (n_fold + 1, score))
    
    precision, recall, thresholds = precision_recall_curve(y_, oof_preds_)
    score = average_precision_score(y_, oof_preds_)
    plt.plot(precision, recall, color='b',
             label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
             lw=2, alpha=.8)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Embedding Neural Network Recall / Precision')
    plt.legend(loc="best")
    plt.tight_layout()
    
    plt.show()


# In[210]:


def preprocessing(input_dir, debug=False):
    # No target encoding
    num_rows = 10000 if debug else None
    
    print('Preprocessing started.')
    print('Bureau_Balance')
    buro_bal = pd.read_csv(input_dir + 'bureau_balance.csv', nrows=num_rows)
    
    buro_counts = buro_bal[['SK_ID_BUREAU', 'MONTHS_BALANCE']].groupby('SK_ID_BUREAU').count()
    buro_bal['buro_count'] = buro_bal['SK_ID_BUREAU'].map(buro_counts['MONTHS_BALANCE'])
    
    avg_buro_bal = buro_bal.groupby('SK_ID_BUREAU').mean()
    
    avg_buro_bal.columns = ['avg_buro_' + f_ for f_ in avg_buro_bal.columns]
    del buro_bal
    gc.collect()
    
    print('Bureau')
    buro_full = pd.read_csv(input_dir + 'bureau.csv', nrows=num_rows)

    gc.collect()
    
    buro_full = buro_full.merge(right=avg_buro_bal.reset_index(), how='left', on='SK_ID_BUREAU', suffixes=('', '_bur_bal'))
    
    nb_bureau_per_curr = buro_full[['SK_ID_CURR', 'SK_ID_BUREAU']].groupby('SK_ID_CURR').count()
    buro_full['SK_ID_BUREAU'] = buro_full['SK_ID_CURR'].map(nb_bureau_per_curr['SK_ID_BUREAU'])
    
    avg_buro = buro_full.groupby('SK_ID_CURR').mean()
    
    del buro_full
    gc.collect()
    
    print('Previous_Application')
    prev = pd.read_csv(input_dir + 'previous_application.csv', nrows=num_rows)
    
    prev_cat_features = [
        f_ for f_ in prev.columns if prev[f_].dtype == 'object'
    ]
    
    
    nb_prev_per_curr = prev[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    prev['SK_ID_PREV'] = prev['SK_ID_CURR'].map(nb_prev_per_curr['SK_ID_PREV'])
    
    avg_prev = prev.groupby('SK_ID_CURR').mean()
    del prev
    gc.collect()
    
    print('POS_CASH_Balance')
    pos = pd.read_csv(input_dir + 'POS_CASH_balance.csv', nrows=num_rows)
    
    
    nb_prevs = pos[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    pos['SK_ID_PREV'] = pos['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])
    
    avg_pos = pos.groupby('SK_ID_CURR').mean()
    
    del pos, nb_prevs
    gc.collect()
    
    print('Credit_Card_Balance')
    cc_bal = pd.read_csv(input_dir + 'credit_card_balance.csv', nrows=num_rows)

    nb_prevs = cc_bal[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    cc_bal['SK_ID_PREV'] = cc_bal['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])
    
    avg_cc_bal = cc_bal.groupby('SK_ID_CURR').mean()
    avg_cc_bal.columns = ['cc_bal_' + f_ for f_ in avg_cc_bal.columns]
    
    del cc_bal, nb_prevs
    gc.collect()
    
    print('Installments_Payments')
    inst = pd.read_csv(input_dir + 'installments_payments.csv', nrows=num_rows)
    nb_prevs = inst[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    inst['SK_ID_PREV'] = inst['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])
    
    avg_inst = inst.groupby('SK_ID_CURR').mean()
    avg_inst.columns = ['inst_' + f_ for f_ in avg_inst.columns]
    
    print('Train/Test')
    data = pd.read_csv(input_dir + 'application_train.csv', nrows=num_rows)
    test = pd.read_csv(input_dir + 'application_test.csv', nrows=num_rows)
    print('Shapes : ', data.shape, test.shape)
        
    data = data.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
    
    data = data.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
    
    data = data.merge(right=avg_pos.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_pos.reset_index(), how='left', on='SK_ID_CURR')
    
    data = data.merge(right=avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')
    
    data = data.merge(right=avg_inst.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_inst.reset_index(), how='left', on='SK_ID_CURR')
    
    del avg_buro, avg_prev
    gc.collect()
    
    print('Preprocessing done.')

    return data, test


# # Prepare the data

# In[211]:


train, test = preprocessing('../input/', debug=False)


# There is **307511** lines in the train file and **48744** lines in the test files. We have 121 differents features ( I'm deliberating excluding **SK_ID_CURR** which act as an ID and the **TARGET** variable)

# In[212]:


# Drop the target and the ID
X_train, y_train = train.iloc[:,2:], train.TARGET
X_test = test.iloc[:,1:]


# # Variable Type

# In[213]:


col_vals_dict = {c: list(X_train[c].unique()) for c in X_train.columns if X_train[c].dtype == object}


# In[214]:


nb_numeric   = len(X_train.columns) - len(col_vals_dict)
nb_categoric = len(col_vals_dict)
print('Number of Numerical features:', nb_numeric)
print('Number of Categorical features:', nb_categoric)


# # Label encode the categorical features

# In[215]:


# Store the labels of each features
col_vals_dict = {c: list(X_train[c].unique()) for c in X_train.columns if X_train[c].dtype == object}


# In[216]:


# Generator to parse the cat
generator = (c for c in X_train.columns if X_train[c].dtype == object)

# Label Encoder
for c in generator:
    lbl = LabelEncoder()
    lbl.fit(list(X_train[c].values) + list(X_test[c].values))
    X_train[c] = lbl.transform(list(X_train[c].values))
    X_test[c] = lbl.transform(list(X_test[c].values))


# # Create the network

# In order to create our embedding model we need to have a look at the spatiality of the cat features. We choose here to use Embedding only on cat features that present more than 2 outcomes otherwise it is count as a numeric value (0 or 1).

# In[217]:


embed_cols = []
len_embed_cols = []
for c in col_vals_dict:
    if len(col_vals_dict[c])>2:
        embed_cols.append(c)
        len_embed_cols.append(len(col_vals_dict[c]))
        print(c + ': %d values' % len(col_vals_dict[c])) #look at value counts to know the embedding dimensions
        
print('\n Number of embed features :', len(embed_cols))


# We are including 13 features **out of 16 categorical features** into our Embedding.
# 
# We can see that our features have a reatively small number of outcomes except for **OCCUPATION_TYPE** and **ORGANIZATION_TYPE** which will be represented in a high dimensional spaces in our Embedding.
# 
# The first layer of our network is the embedding layer with the size of 3 "CODE_GENDER". The embedding-size defines the dimensionality in which we map the categorical variables (in a 3D spaces for instance). One good rule of thumb to use for the output is : 
# 
# **embedding size = min(50, number of categories/2)**

# In[218]:


def build_embedding_network(len_embed_cols):
    
    model_out = []
    model_in  = []
    
    for dim in len_embed_cols:
        input_dim = Input(shape=(1,), dtype='int32')
        embed_dim = Embedding(dim, dim//2, input_length=1)(input_dim)
        embed_dim = Dropout(0.25)(embed_dim)
        embed_dim = Reshape((dim//2,))(embed_dim)
        model_out.append(embed_dim)
        model_in.append(input_dim)
    
    input_num = Input(shape=(176,), dtype='float32')
    outputs = Concatenate(axis=1)([*model_out, input_num])
    
    outputs = (Dense(128))(outputs) 
    outputs = (Activation('relu'))(outputs)
    outputs = (Dropout(.35))(outputs)
    outputs = (Dense(64))(outputs)
    outputs = (Activation('relu'))(outputs)
    outputs = (Dropout(.15))(outputs)
    outputs = (Dense(32))(outputs) 
    outputs = (Activation('relu'))(outputs)
    outputs = (Dropout(.15))(outputs)
    outputs = (Dense(1))(outputs)
    outputs = (Activation('sigmoid'))(outputs)
    
    model = Model([*model_in, input_num], outputs)

    model.compile(loss='binary_crossentropy', optimizer='adam')
    
    return model


# In order for keras to know which features are going to be included into the Embedding layers we need to create a list containing for each feature the corresponding numpy array (**13** in total for us). The last element of the list will be our numerical features (**173**) and the categorical features that we decided not to include in the Embedding (**3**) for a total of **176** distinct features.
# 

# In[219]:


def preproc(X_train, X_val, X_test):

    input_list_train = []
    input_list_val = []
    input_list_test = []
    
    #the cols to be embedded: rescaling to range [0, # values)
    for c in embed_cols:
        raw_vals = np.unique(X_train[c])
        val_map = {}
        for i in range(len(raw_vals)):
            val_map[raw_vals[i]] = i       
        input_list_train.append(X_train[c].map(val_map).values)
        input_list_val.append(X_val[c].map(val_map).fillna(0).values)
        input_list_test.append(X_test[c].map(val_map).fillna(0).values)
        
    #the rest of the columns
    other_cols = [c for c in X_train.columns if (not c in embed_cols)]
    input_list_train.append(X_train[other_cols].values)
    input_list_val.append(X_val[other_cols].values)
    input_list_test.append(X_test[other_cols].values)
    
    return input_list_train, input_list_val, input_list_test


# Let us go more specifically into this function : 

# In[220]:


proc_X_train_f, proc_X_val_f, proc_X_test_f = preproc(X_train, X_train, X_test)
print('Length of the list:', len(proc_X_train_f))


# In[221]:


proc_X_train_f


# In[222]:


print(proc_X_train_f[12].shape)


# This list will be passed into the network. It is composed of 14 numpy arrays containing our categorical features that are going throught the Embedding Layers (**13 layers**). The last element of the list is a numpy array composed of the **173 numerics features added to the 3 categorical features that have at most 2 distinct outcomes**. 

# In[223]:


del proc_X_train_f, proc_X_val_f, proc_X_test_f
gc.collect()


# # Prepare the data
# 
# In neural networks, it is a best practice to scale input data before use. Data scaling
# makes the training of the network faster, memory efficient and yield accurate
# forecast results. Neural networks only work with data usually between a specified range (1 to 1 or 0 to 1), it makes it necessary then that data is scaled down and normalized. 
# 
# Scaling can be as simple as taking the ratios (reciprocal normalization), computing the differences
# (range normalization) or multiplicative normalization.
# Normalization ensures that data is roughly uniformly distributed between the network inputs
# and the outputs.

# In[224]:


# Select the numeric features
num_cols = [x for x in X_train.columns if x not in embed_cols]


# Impute missing values in order to scale
X_train[num_cols] = X_train[num_cols].fillna(value = 0)
X_test[num_cols] = X_test[num_cols].fillna(value = 0)

# Fit the scaler only on train data
scaler = MinMaxScaler().fit(X_train[num_cols])
X_train.loc[:,num_cols] = scaler.transform(X_train[num_cols])
X_test.loc[:,num_cols] = scaler.transform(X_test[num_cols])


# # Train the network

# In[225]:


K = 5
runs_per_fold = 1
n_epochs = 250
patience = 10

cv_aucs   = []
full_val_preds = np.zeros(np.shape(X_train)[0])
y_preds = np.zeros((np.shape(X_test)[0],K))

kfold = StratifiedKFold(n_splits = K,  
                            shuffle = True, random_state=1)

for i, (f_ind, outf_ind) in enumerate(kfold.split(X_train, y_train)):

    X_train_f, X_val_f = X_train.loc[f_ind].copy(), X_train.loc[outf_ind].copy()
    y_train_f, y_val_f = y_train[f_ind], y_train[outf_ind]
    
    X_test_f = X_test.copy()
    
    
    # Shuffle data
    idx = np.arange(len(X_train_f))
    np.random.shuffle(idx)
    X_train_f = X_train_f.iloc[idx]
    y_train_f = y_train_f.iloc[idx]
    
    #preprocessing
    proc_X_train_f, proc_X_val_f, proc_X_test_f = preproc(X_train_f, X_val_f, X_test_f)
    
    #track oof prediction for cv scores
    val_preds = 0
    
    for j in range(runs_per_fold):
    
        NN = build_embedding_network(len_embed_cols)

        # Set callback functions to early stop training and save the best model so far
        callbacks = [EarlyStopping(monitor='val_loss', patience=patience)]

        NN.fit(proc_X_train_f, y_train_f.values, epochs=n_epochs, batch_size=4096, verbose=1,callbacks=callbacks,validation_data=(proc_X_val_f, y_val_f))
        
        val_preds += NN.predict(proc_X_val_f)[:,0] / runs_per_fold
        y_preds[:,i] += NN.predict(proc_X_test_f)[:,0] / runs_per_fold
        
    full_val_preds[outf_ind] += val_preds
        
    cv_auc  = roc_auc_score(y_val_f.values, val_preds)
    cv_aucs.append(cv_auc)
    print ('\nFold %i prediction cv AUC: %.5f\n' %(i,cv_auc))
    
print('Mean out of fold AUC: %.5f' % np.mean(cv_auc))
print('Full validation AUC: %.5f' % roc_auc_score(y_train.values, full_val_preds))


# In[226]:


folds_idx = [(trn_idx, val_idx) for trn_idx, val_idx in kfold.split(X_train, y_train)]
display_roc_curve(y_=y_train, oof_preds_=full_val_preds, folds_idx_=folds_idx)


# In[227]:


display_precision_recall(y_=y_train, oof_preds_=full_val_preds, folds_idx_=folds_idx)


# In[228]:


test['TARGET'] = np.mean(y_preds, axis=1)
test = test[['SK_ID_CURR', 'TARGET']]
out_df = pd.DataFrame({'SK_ID_CURR': test['SK_ID_CURR'], 'TARGET': test['TARGET']})
out_df.to_csv('nn_embedding_submission.csv', index=False)


# We can see that the model is performing well with an **AVG AUC of 0.75 on CV5 out-of-fold ** and an **AUC of 0.748 on LB**. 
# 
# However I'm having difficulties to perform as well as Boosted Trees like XGBoost, Lgbm and Catboost. If anyone have any hint on how to improve this kernel, please let me know.
