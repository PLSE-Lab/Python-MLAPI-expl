#!/usr/bin/env python
# coding: utf-8

# ## Keras NN Starter Kernel

# Not looking to seriously compete in this competition so figured I'd at least share the small experiments I was toying with. This kernel serves as just a starter to look at how to handle the various numerical and categorical variables in a NN with Keras. 

# In[ ]:


#  Libraries
import numpy as np 
import pandas as pd 
# Data processing, metrics and modeling
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold,KFold
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, roc_auc_score, f1_score, roc_curve, auc,precision_recall_curve
from sklearn import metrics
from sklearn import preprocessing
# Suppr warning
import warnings
warnings.filterwarnings("ignore")

import itertools
from scipy import interp
# Plots
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import rcParams


# ## Data Loading
# Just the standard loading of the data used in most other kernels. 

# In[ ]:


get_ipython().run_cell_magic('time', '', "train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')\ntest_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')\ntrain_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')\ntest_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')\nsample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')")


# In[ ]:


# merge 
train_df = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test_df = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)

print("Train shape : "+str(train_df.shape))
print("Test shape  : "+str(test_df.shape))


# Dropping time since this likely isnt something we want our model to directly learn from

# In[ ]:


train_df = train_df.reset_index()
test_df = test_df.reset_index()


# In[ ]:


train_df['nulls1'] = train_df.isna().sum(axis=1)
test_df['nulls1'] = test_df.isna().sum(axis=1)


# In[ ]:


train_df = train_df.drop(["TransactionDT"], axis = 1)
test_df = test_df.drop(["TransactionDT"], axis = 1)


# Selecting just the first 53 columns and excluding the synthetic "v" features and other very sparse categoricals like deviceinfo anf deviceid

# In[ ]:


train_df


# In[ ]:


test_df


# In[ ]:


train_df = train_df.iloc[:, :53]
test_df = test_df.iloc[:, :52]


# In[ ]:


# #Based on this great kernel https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
# def reduce_mem_usage(df):
#     start_mem_usg = df.memory_usage().sum() / 1024**2 
#     print("Memory usage of properties dataframe is :",start_mem_usg," MB")
#     NAlist = [] # Keeps track of columns that have missing values filled in. 
#     for col in df.columns:
#         if df[col].dtype != object:  # Exclude strings                       
#             # make variables for Int, max and min
#             IsInt = False
#             mx = df[col].max()
#             mn = df[col].min()
#             # Integer does not support NA, therefore, NA needs to be filled
#             if not np.isfinite(df[col]).all(): 
#                 NAlist.append(col)
#                 df[col].fillna(mn-1,inplace=True)  
                   
#             # test if column can be converted to an integer
#             asint = df[col].fillna(0).astype(np.int64)
#             result = (df[col] - asint)
#             result = result.sum()
#             if result > -0.01 and result < 0.01:
#                 IsInt = True            
#             # Make Integer/unsigned Integer datatypes
#             if IsInt:
#                 if mn >= 0:
#                     if mx < 255:
#                         df[col] = df[col].astype(np.uint8)
#                     elif mx < 65535:
#                         df[col] = df[col].astype(np.uint16)
#                     elif mx < 4294967295:
#                         df[col] = df[col].astype(np.uint32)
#                     else:
#                         df[col] = df[col].astype(np.uint64)
#                 else:
#                     if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
#                         df[col] = df[col].astype(np.int8)
#                     elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
#                         df[col] = df[col].astype(np.int16)
#                     elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
#                         df[col] = df[col].astype(np.int32)
#                     elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
#                         df[col] = df[col].astype(np.int64)    
#             # Make float datatypes 32 bit
#             else:
#                 df[col] = df[col].astype(np.float32)
            
#             # Print new column type

#     # Print final result
#     mem_usg = df.memory_usage().sum() / 1024**2 
#     return df, NAlist


# Freeing up some memory by borrowing from https://www.kaggle.com/c/ieee-fraud-detection/discussion/100021#latest-591871

# In[ ]:


# train_df, NAlist = reduce_mem_usage(train_df)


# In[ ]:


# test_df, NAlist = reduce_mem_usage(test_df)


# In[ ]:


del train_transaction, train_identity, test_transaction, test_identity


# In[ ]:


emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 'optonline.net': 'other', 'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo', 'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink', 'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other', 'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo', 'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other', 'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft', 'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink', 'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other', 'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other', 'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}
us_emails = ['gmail', 'net', 'edu']
#https://www.kaggle.com/c/ieee-fraud-detection/discussion/100499#latest_df-579654
for c in ['P_emaildomain', 'R_emaildomain']:
    train_df[c + '_bin'] = train_df[c].map(emails)
    test_df[c + '_bin'] = test_df[c].map(emails)
    
    train_df[c + '_suffix'] = train_df[c].map(lambda x: str(x).split('.')[-1])
    test_df[c + '_suffix'] = test_df[c].map(lambda x: str(x).split('.')[-1])
    
    train_df[c + '_suffix'] = train_df[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
    test_df[c + '_suffix'] = test_df[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')


# In[ ]:


for c1, c2 in train_df.dtypes.reset_index().values:
    if c2=='O':
        train_df[c1] = train_df[c1].map(lambda x: str(x).lower())
        test_df[c1] = test_df[c1].map(lambda x: str(x).lower())


# ## Numerical and Categorical
# Listing off and categorizing the various variables available to us. We have numerical and categoricals. We will treat both of these slightly differently later

# In[ ]:


numerical = ["TransactionAmt", "nulls1", "dist1", "dist2"] + ["C" + str(i) for i in range(1, 15)] +             ["D" + str(i) for i in range(1, 16)] +             ["V" + str(i) for i in range(1, 340)]
categorical = ["ProductCD", "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2",
               "P_emaildomain_bin", "P_emaildomain_suffix", "R_emaildomain_bin", "R_emaildomain_suffix",
               "P_emaildomain", "R_emaildomain",
              "DeviceInfo", "DeviceType"] + ["id_0" + str(i) for i in range(1, 10)] +\
                ["id_" + str(i) for i in range(10, 39)] + \
                 ["M" + str(i) for i in range(1, 10)]


# We already dropped a lot of these features because in some trial and error it was shown that these caused rapid overfitting for some reason or otherwise introduced unnesecary noise into the data. We will make sure we only list the features we actually have still in the df now

# In[ ]:


numerical = [col for col in numerical if col in train_df.columns]
categorical = [col for col in categorical if col in train_df.columns]


# NN doesn't like nans so we will fill numerical columns with 0's. Previous people have tried plugging in the column means, but upon inspection this didn't seem very reliable because the train and test means for a given column could sometimes be drastically different. Plugging in zeros likely isnt the best. Might be better to plug in the train mean to the test df, but for simplicity I will stick with 0's for now. 

# In[ ]:


def nan2mean(df):
    for x in list(df.columns.values):
        if x in numerical:
            #print("___________________"+x)
            #print(df[x].isna().sum())
            df[x] = df[x].fillna(0)
           #print("Mean-"+str(df[x].mean()))
    return df
train_df=nan2mean(train_df)
test_df=nan2mean(test_df)


# ## Label Encoding
# We will take our categorical features fill the nans and assign them an integer ID per category and write down the number of total categories per column. We'll use this later in an embedding layer of the NN

# In[ ]:


# Label Encoding
category_counts = {}
for f in categorical:
    train_df[f] = train_df[f].replace("nan", "other")
    train_df[f] = train_df[f].replace(np.nan, "other")
    test_df[f] = test_df[f].replace("nan", "other")
    test_df[f] = test_df[f].replace(np.nan, "other")
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_df[f].values) + list(test_df[f].values))
    train_df[f] = lbl.transform(list(train_df[f].values))
    test_df[f] = lbl.transform(list(test_df[f].values))
    category_counts[f] = len(list(lbl.classes_)) + 1
# train_df = train_df.reset_index()
# test_df = test_df.reset_index()


# In[ ]:


from sklearn.preprocessing import StandardScaler


# ## Numerical Scaling
# 
# Now we will do some scaling of the data so that it will be in a more NN friendly format. First we will do log1p for any values that are above 100 and not below 0. This is in order to scale down any numerical variables that might have some extremely high values that screws up the statistics of the standard scaler 
# 
# After that we will pass them through the standard scaler so that the values have a normal mean and std. This makes the NN converge signficantly faster and prevents any blowouts. Feel free to try for yourself by commenting out this cell of code

# In[ ]:


for column in numerical:
    scaler = StandardScaler()
    if train_df[column].max() > 100 and train_df[column].min() >= 0:
        train_df[column] = np.log1p(train_df[column])
        test_df[column] = np.log1p(test_df[column])
    scaler.fit(np.concatenate([train_df[column].values.reshape(-1,1), test_df[column].values.reshape(-1,1)]))
    train_df[column] = scaler.transform(train_df[column].values.reshape(-1,1))
    test_df[column] = scaler.transform(test_df[column].values.reshape(-1,1))


# In[ ]:


target = 'isFraud'


# In[ ]:


#cut tr and val
tr_df, val_df = train_test_split(train_df, test_size = 0.2, random_state = 42, stratify = train_df[target])


# Grabbing the features we want to pass into the neural network

# In[ ]:


def get_input_features(df):
    X = {'numerical':np.array(df[numerical])}
    for cat in categorical:
        X[cat] = np.array(df[cat])
    return X


# ## Neural Network Model Details
# 
# Our neural network will be fairly standard. We will use the embedding layer for categoricals and the numericals will go through feed forward dense layers. 
# 
# We create our embedding layers such that we have as many rows as we had categories and the dimension of the embedding is the log1p + 1 of the number of categories. So this means that categorical variables with very high cardinality will have more dimensions but not signficantly more so the information will still be compressed down to only about 13 dimensions and the smaller number of categories will be only 2-3.
# 
# We will then pass the embeddings through a spatial dropout layer which will drop dimensions within the embedding across batches and then flatten and concatenate. Then we will concatenate this to the numerical features and apply batch norm and then add some more dense layers after. 

# In[ ]:


categorical.remove("card1")


# In[ ]:


category_counts


# In[ ]:


from keras.layers import Concatenate, Input, Dense, Embedding, Flatten, Dropout, BatchNormalization, SpatialDropout1D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model
from keras.optimizers import  Adam
import keras.backend as k
def make_model():
    k.clear_session()

    categorical_inputs = []
    for cat in categorical:
        categorical_inputs.append(Input(shape=[1], name=cat))

    categorical_embeddings = []
    for i, cat in enumerate(categorical):
        categorical_embeddings.append(
            Embedding(category_counts[cat], int(np.log1p(category_counts[cat]) + 1), name = cat + "_embed")(categorical_inputs[i]))

    categorical_logits = Concatenate(name = "categorical_conc")([Flatten()(SpatialDropout1D(.1)(cat_emb)) for cat_emb in categorical_embeddings])
#     categorical_logits = Dropout(.5)(categorical_logits)

    numerical_inputs = Input(shape=[tr_df[numerical].shape[1]], name = 'numerical')
    numerical_logits = Dropout(.1)(numerical_inputs)
  
    x = Concatenate()([
        categorical_logits, 
        numerical_logits,
    ])
#     x = categorical_logits
    x = BatchNormalization()(x)
    x = Dense(200, activation = 'relu')(x)
    x = Dropout(.2)(x)
    x = Dense(100, activation = 'relu')(x)
    x = Dropout(.2)(x)
    out = Dense(1, activation = 'sigmoid')(x)
    

    model = Model(inputs=categorical_inputs + [numerical_inputs],outputs=out)
    loss = "binary_crossentropy"
    model.compile(optimizer=Adam(lr = 0.001), loss = loss)
    return model


# We will then extract the features we actually want to pass to the NN

# In[ ]:


X_train = get_input_features(tr_df)
X_valid = get_input_features(val_df)
X_test = get_input_features(test_df)
y_train = tr_df[target]
y_valid = val_df[target]
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=2, mode='auto', verbose = False)


# In[ ]:


from sklearn.metrics import roc_auc_score


# We will iterate through epochs of the model and save the model weights if the score is an improvement upon previous best roc_auc_scores since this is competition metric. If the NN does not improve upon previous best after 4 epochs we will skip the rest of the training steps to save time. 

# In[ ]:


model = make_model()
best_score = 0
patience = 0
for i in range(100):
    if patience < 4:
        hist = model.fit(X_train, y_train, validation_data = (X_valid,y_valid), batch_size = 1000, epochs = 1, verbose = 1)
        valid_preds = model.predict(X_valid, batch_size = 8000, verbose = True)
        score = roc_auc_score(y_valid, valid_preds)
        print(score)
        if score > best_score:
            model.save_weights("model.h5")
            best_score = score
            patience = 0
        else:
#             patience += 1
            pass


# In[ ]:


model.load_weights("model.h5")


# ## Sanity Checks
# Now we will make sure we loaded in the correction  model that scored favorably and then we will compare prediction statistics between validation and test. These should be relatively close or else we know something might have been off between how we prepared the train and test data for inference

# In[ ]:


valid_preds = model.predict(X_valid, batch_size = 500, verbose = True)
score = roc_auc_score(y_valid, valid_preds)
print(score)


# In[ ]:


predictions = model.predict(X_test, batch_size = 2000, verbose = True)


# In[ ]:


pd.DataFrame(valid_preds).describe()


# In[ ]:


pd.DataFrame(predictions).describe()


# In[ ]:


sample_submission['isFraud'] = predictions
sample_submission.to_csv('submission_IEEE.csv')


# In[ ]:




