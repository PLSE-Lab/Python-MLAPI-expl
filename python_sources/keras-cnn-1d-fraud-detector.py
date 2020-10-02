#!/usr/bin/env python
# coding: utf-8

# In this competition a lot has been said and written about feature engineering...and the likely winner and top scores will definitely be achieved by heavy feature engineering.
# 
# But curious to see how far we can get with a minimum of feature engineering. So I created this notebook with just some basic preparation, feature selection, count encoding and NaN elimination
# 
# I'll use a simple 1D Convolutional Neural Network to showcase whats possible.

# In[ ]:


import numpy as np
import pandas as pd
import gc
import time
import random
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from keras import backend as K
from keras.models import Sequential, load_model
from keras.optimizers import Adam, Nadam
from keras.initializers import glorot_uniform, lecun_uniform
from keras.layers import Dense, Conv1D, Flatten, MaxPool1D, Dropout, Activation, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Random Seed
seed = 12345
np.random.seed(seed)
random.seed(seed)

# Constants
epochs = 25
batch_size = 1024
number_of_folds = 6


# Next I'll define 2 lists. One with categorical features and one with 'low-score' features. I made a simple LightGBM setup in which I determined the AUC score per feature. This usually gives a nice impression. The list below is a list of the lowest scoring ones. Very quick and dirty determined but I used this already in multiple competitions and usually it works quitte well.

# In[ ]:


# Categorical features
cat_feats = ['ProductCD', 'addr1', 'addr2', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 
                        'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']

# Low score Features
lowscore_feats =   ['V322','V329','V321','V336','V331','V335','V330','V332','V328','V338','V327','V137','V333','V326','V116','V339','V337',
                    'V334','V114','V115','V163','V298','V162','V142','V141','V325','V129','V138','V161','V100','V296','V112','V105',
                    'V113','V111','V106','V299','V98','V110','V301','V108','V135','V109','V319','V104','V300','V297','V119','V311',
                    'V117','V41','V118','V121','V122','V286','V120','V107','V305']


# Next we load the data and immediately drop the columns we don't need. Alternatively we could create a list of just the columns that we want to load. I prefer the first method as I can now easily modify my list with lowscore_feats.

# In[ ]:


# Load Data
train = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')
labels = train['isFraud']
test = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')

# Drop Columns
test.drop(lowscore_feats, axis = 1, inplace = True)
train.drop(lowscore_feats, axis = 1, inplace = True)
train.drop(['isFraud'], axis=1, inplace = True)

# Summary Shapes
print('====== Dataset Shapes')
print('Train Transaction: {0}'.format(train.shape))
print('Test Transaction: {0}'.format(test.shape))
print('Labels: {0}'.format(labels.shape))


# I'll append the train and test set. Only for the features with less than 10K of missing values will the missing values be replaced by the mean value.

# In[ ]:


# Append Train and Test Datasets
train_len = len(train)
df = train.append(test).reset_index()

# Cleanup
del train, test
gc.collect()

# Impute Mean value for features that have less than 10K NaN values
processed_feats = []
for feat in [f for f in df.columns if f not in ['index', 'TransactionID', 'TransactionDT'] + cat_feats + lowscore_feats]:
    if df[feat].isna().sum() < 10000:
        imputer = SimpleImputer(strategy = 'mean')
        df[feat] = imputer.fit_transform(df[feat].values.reshape(-1, 1))
        processed_feats.append(feat)       


# Next we process TransactionDT into new features 'hour' and 'weekday'.

# In[ ]:


# Process TransactionDT into hour and weekday
df['hour'] = df['TransactionDT'].map(lambda x:(x // 3600) % 24)
df['weekday'] = df['TransactionDT'].map(lambda x:(x // (3600 * 24)) % 7)


# All remaining features will be encoded with their value counts. The nan values are included in this to make sure that we don't have any missing values when running the neural net.

# In[ ]:


# Count encode all categorical features
for feat in cat_feats:
    df[feat] = df[feat].map(df[feat].value_counts(dropna = False))

# Count encode all other remaining 'Numerical' Features
for feat in [f for f in df.columns if f not in ['index', 'TransactionID', 'TransactionDT'] + cat_feats + processed_feats]:
    df[feat] = df[feat].map(df[feat].value_counts(dropna = False))


# As a last processing step I use a MinMaxScaler to scale all the features. For features with a large skew value this will also be corrected.

# In[ ]:


# Get Final Features
feats = [f for f in df.columns if f not in ['index', 'TransactionID', 'TransactionDT'] + lowscore_feats]

# Scale and correct extreme skew
scaler = preprocessing.MinMaxScaler()
for feat in feats:
    # Scale
    df[feat] = scaler.fit_transform(df[feat].values.reshape(-1, 1))
    
    # Correct Skew
    if df[feat].skew() > 10:
        df[feat] = np.log10(df[feat] + 1 - min(0, df[feat].min()))


# In[ ]:


# Split back to train and test dataset  
train = df[:train_len]
test = df[train_len:]

# Final Summary Shapes
print('====== Final Dataset Shapes')
print('Train Transaction: {0}'.format(train[feats].shape))
print('Test Transaction: {0}'.format(test[feats].shape))
print('Labels: {0}'.format(labels.shape))


# Define the EarlyStopping, ModelCheckPoint and the model itself.

# In[ ]:


def EarlyStop(patience):
    return EarlyStopping(monitor = "val_loss",
                          min_delta = 0,
                          mode = "min",
                          verbose = 1, 
                          patience = patience)

def ModelCheckpointFull(model_name):
    return ModelCheckpoint(model_name, 
                            monitor = 'val_loss', 
                            verbose = 1, 
                            save_best_only = True, 
                            save_weights_only = False, 
                            mode = 'min', 
                            period = 1)

# Input Shape
input_shape = train[feats].shape[1]

# Define CNN 1D model
def create_model():
    model = Sequential()
    model.add(Conv1D(96, 2, activation = 'relu', input_shape=(input_shape, 1), kernel_initializer = glorot_uniform(seed = seed)))
    model.add(BatchNormalization())       
    model.add(Dropout(0.25))
    model.add(Conv1D(96, 1, activation = 'relu', kernel_initializer = glorot_uniform(seed = seed)))
    model.add(BatchNormalization())       
    model.add(Flatten())
    model.add(Dropout(0.25))    
    model.add(Dense(96, activation = 'relu', kernel_initializer = glorot_uniform(seed = seed)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(1, activation = 'sigmoid', kernel_initializer = glorot_uniform(seed = seed)))
    model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = 0.001), metrics = ['accuracy'])
    
    return model


# As a final step we run a Stratified KFold and generate the submission file. After some trials I found that about 25 epochs is a good value to use. When using more the validation AUC score will increase but the LB score will start to drop slightly.

# In[ ]:


# Reshape
train = train[feats].values.reshape(-1, input_shape, 1)
test = test[feats].values.reshape(-1, input_shape, 1)

# CV Folds
folds = StratifiedKFold(n_splits = number_of_folds, shuffle = True, random_state = seed)

# Arrays to store predictions
oof_preds = np.zeros(train.shape[0])
sub_preds = np.zeros(test.shape[0])

# Loop through folds
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train, labels)):
    train_x, train_y = train[train_idx], labels.iloc[train_idx]
    valid_x, valid_y = train[valid_idx], labels.iloc[valid_idx]

    print('Running Fold: ' + str(n_fold))

    # CNN 1D model
    model = create_model()
    model.fit(train_x, train_y, 
                validation_data=(valid_x, valid_y), 
                epochs=epochs, 
                batch_size=batch_size, 
                verbose=0,
                callbacks=[EarlyStop(10), ModelCheckpointFull('model.h5')])

    # Delete Model
    del model
    gc.collect()

    # Reload Best Saved Model
    model = load_model('model.h5')

    # OOF Predictions
    oof_preds[valid_idx] = model.predict(valid_x).reshape(-1,)
    
    # Submission Predictions
    predictions = model.predict(test).reshape(-1,)
    sub_preds += predictions / number_of_folds

    # Fold AUC Score
    print('Fold %2d AUC : %.6f' % (n_fold, roc_auc_score(valid_y, oof_preds[valid_idx])))        

    # Cleanup 
    del model, train_x, train_y, valid_y, valid_x
    K.clear_session()
    gc.collect

print('Full AUC score %.6f' % roc_auc_score(labels, oof_preds))

# Generate Submission
submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')
submission['isFraud'] = sub_preds
submission.to_csv('submission.csv', index=False)


# I hope you enjoyed the notebook. Let me know if you have any feedback or questions.
