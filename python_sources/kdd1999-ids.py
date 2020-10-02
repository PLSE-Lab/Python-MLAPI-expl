#!/usr/bin/env python
# coding: utf-8

# Intrusion Detection using KDD1999 data.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings  
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

seed = 51


# In[ ]:


data = pd.read_csv('/kaggle/input/kdd-cup-1999-data/kddcup.data_10_percent_corrected', header=None)


# In[ ]:


data.columns = [
    'duration',
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment',
    'urgent',
    'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised',
    'root_shell',
    'su_attempted',
    'num_root',
    'num_file_creations',
    'num_shells',
    'num_access_files',
    'num_outbound_cmds',
    'is_host_login',
    'is_guest_login',
    'count',
    'srv_count',
    'serror_rate',
    'srv_serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'same_srv_rate',
    'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'outcome'
]


# In[ ]:


data.sample(10)


# In[ ]:


print(data['num_outbound_cmds'].unique())


# In[ ]:


# Encode a numeric column as zscores
def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()

    if sd is None:
        sd = df[name].std()

    df[name] = (df[name] - mean) / sd
    
# Encode text values to dummy variables(i.e. [1,0,0],[0,1,0],[0,0,1] for red,green,blue)
def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)


# In[ ]:


encode_numeric_zscore(data, 'duration')
encode_text_dummy(data, 'protocol_type')
encode_text_dummy(data, 'service')
encode_text_dummy(data, 'flag')
encode_numeric_zscore(data, 'src_bytes')
encode_numeric_zscore(data, 'dst_bytes')
encode_text_dummy(data, 'land')
encode_numeric_zscore(data, 'wrong_fragment')
encode_numeric_zscore(data, 'urgent')
encode_numeric_zscore(data, 'hot')
encode_numeric_zscore(data, 'num_failed_logins')
encode_text_dummy(data, 'logged_in')
encode_numeric_zscore(data, 'num_compromised')
encode_numeric_zscore(data, 'root_shell')
encode_numeric_zscore(data, 'su_attempted')
encode_numeric_zscore(data, 'num_root')
encode_numeric_zscore(data, 'num_file_creations')
encode_numeric_zscore(data, 'num_shells')
encode_numeric_zscore(data, 'num_access_files')
encode_numeric_zscore(data, 'num_outbound_cmds')
encode_text_dummy(data, 'is_host_login')
encode_text_dummy(data, 'is_guest_login')
encode_numeric_zscore(data, 'count')
encode_numeric_zscore(data, 'srv_count')
encode_numeric_zscore(data, 'serror_rate')
encode_numeric_zscore(data, 'srv_serror_rate')
encode_numeric_zscore(data, 'rerror_rate')
encode_numeric_zscore(data, 'srv_rerror_rate')
encode_numeric_zscore(data, 'same_srv_rate')
encode_numeric_zscore(data, 'diff_srv_rate')
encode_numeric_zscore(data, 'srv_diff_host_rate')
encode_numeric_zscore(data, 'dst_host_count')
encode_numeric_zscore(data, 'dst_host_srv_count')
encode_numeric_zscore(data, 'dst_host_same_srv_rate')
encode_numeric_zscore(data, 'dst_host_diff_srv_rate')
encode_numeric_zscore(data, 'dst_host_same_src_port_rate')
encode_numeric_zscore(data, 'dst_host_srv_diff_host_rate')
encode_numeric_zscore(data, 'dst_host_serror_rate')
encode_numeric_zscore(data, 'dst_host_srv_serror_rate')
encode_numeric_zscore(data, 'dst_host_rerror_rate')
encode_numeric_zscore(data, 'dst_host_srv_rerror_rate')

# we need this because of the problem with num_outbound_cmds = 0 in every row.
data.dropna(inplace=True, axis=1)


# In[ ]:


data.outcome.value_counts()


# Outcomes smurf, neptune, and normal are 98% of the data.

# In[ ]:


# x = data.drop(['outcome'], axis=1)
x_columns = data.columns.drop('outcome')
x = data[x_columns].values
dummies = pd.get_dummies(data['outcome']) # Classification
outcomes = dummies.columns
num_classes = len(outcomes)
y = dummies.values


# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=seed)


# In[ ]:


import tensorflow
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, ELU, Input, Dropout

input = Input(shape=x.shape[1])

m = Dense(64)(input)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(64)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(32)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(16)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

# m = Dense(1, activation='linear')(m)

output = Dense(y.shape[1], activation='softmax')(m)

model = Model(inputs=[input], outputs=[output])

model.summary()


# In[ ]:


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)

rlp = ReduceLROnPlateau(monitor='val_loss', patience=9, verbose=1, factor=0.5, cooldown=5, min_lr=1e-10)


# In[ ]:


history = model.fit(x_train
                    ,y_train
#                     ,validation_data=(x_test,y_test)
                    ,callbacks=[es, rlp]
                    ,verbose=1
                    ,epochs=30
                    , batch_size=512).history


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(20, 14))

ax1.plot(history['loss'], label='Train loss')
# ax1.plot(history['val_loss'], label='Validation loss')
ax1.legend(loc='best')
ax1.set_title('Loss')

ax2.plot(history['acc'], label='Train accuracy')
# ax2.plot(history['val_acc'], label='Validation accuracy')
ax2.legend(loc='best')
ax2.set_title('Accuracy')

plt.xlabel('Epochs')
sns.despine()
plt.show()


# In[ ]:


model.evaluate(x_test, y_test)


# **99%!** Why is such a tiny neural network so accurate?
# 
# Let's take a quick look at the correlation of the features to the outcome.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['outcome'] = le.fit_transform(data['outcome'])
corr = data.corr()
corr.sort_values(["outcome"], ascending = False, inplace = True)
corr.outcome


# In[ ]:


plt.figure(figsize=(20, 12))
sns.heatmap(corr, annot=True)


# **Observation**: I believe due to a lot of processing of the raw data to get it into this dataset, the important feature engineering has already been done and the deep network here is able to quickly pickout the rules needed to classify the outcomes.
