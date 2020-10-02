#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import json
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pylab import rcParams
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score
from numpy.random import seed
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
SEED = 123 #used to help randomly select the data points
DATA_SPLIT_PCT = 0.2
rcParams['figure.figsize'] = 8, 6
LABELS = ["Normal","Break"]


# In[ ]:





# In[ ]:


import time
t1 = time.time()
folder_path = '../input/ieee-fraud-detection/'
train_identity = pd.read_csv(f'{folder_path}train_identity.csv')
test_identity = pd.read_csv(f'{folder_path}test_identity.csv')
train_transaction = pd.read_csv(f'{folder_path}train_transaction.csv')
test_transaction = pd.read_csv(f'{folder_path}test_transaction.csv')
t2 = time.time()
print("Time to parse: %f" % (t2 - t1))

train_df = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
test_df = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')


# In[ ]:


del train_identity,train_transaction,test_identity, test_transaction


# In[ ]:


train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)


# In[ ]:


cat_cols = ["ProductCD","card4","card6","P_emaildomain","R_emaildomain",           "M1","M2","M3","M4","M5","M6","M7","M8","M9",           "id_12","id_15","id_16","id_23","id_27","id_28","id_29","id_30","id_31","id_33","id_34","id_35","id_36","id_37",           "id_38","DeviceType","DeviceInfo"]


# In[ ]:


for f in cat_cols:
    if f in train_df.columns:
        lbl = LabelEncoder()
        lbl.fit(list(train_df[f].values)+ list(test_df[f].values))
        train_df[f] = lbl.transform(list(train_df[f].values))
        test_df[f] = lbl.transform(list(test_df[f].values))


# In[ ]:


train_df.drop(labels=["TransactionDT"],axis=1,inplace=True)
test_df.drop(labels=["TransactionDT"],axis=1,inplace=True)


# In[ ]:


train_df.drop(labels=["TransactionID"],axis=1,inplace=True)
test_df.drop(labels=["TransactionID"],axis=1,inplace=True)


# In[ ]:


#drop the V variables -- Lightgbm of V only got low predictive power
V_cols = []
for cols in train_df.columns:
    if cols.startswith("V"):
        V_cols.append(cols)
        


# In[ ]:


train_df.drop(labels=V_cols,axis=1,inplace=True)
test_df.drop(labels=V_cols,axis=1,inplace=True)


# In[ ]:


train_df.shape


# In[ ]:


df_train, df_valid = train_test_split(train_df, test_size=DATA_SPLIT_PCT, random_state=SEED)
df_train_0 = df_train.loc[df_train['isFraud'] == 0] #this is not fraud
df_train_1 = df_train.loc[df_train['isFraud'] == 1]
df_train_0_x = (df_train_0.drop(['isFraud'], axis=1))
df_train_1_x = (df_train_1.drop(['isFraud'], axis=1))
df_valid_0 = df_valid.loc[df_valid['isFraud'] == 0]
df_valid_1 = df_valid.loc[df_valid['isFraud'] == 1]
df_valid_0_x = (df_valid_0.drop(['isFraud'], axis=1))
df_valid_1_x = (df_valid_1.drop(['isFraud'], axis=1))

scaler = StandardScaler().fit(df_train_0_x)
df_train_0_x_rescaled = scaler.transform(df_train_0_x)
df_valid_0_x_rescaled = scaler.transform(df_valid_0_x)

df_valid_x_rescaled = scaler.transform(df_valid.drop(["isFraud"],axis=1))


# In[ ]:


df_test_x_rescaled = scaler.transform(test_df)


# In[ ]:


df_train_0_x_rescaled.shape


# In[ ]:


nb_epoch = 100
batch_size = 32 #32
input_dim = df_train_0_x_rescaled.shape[1] #num of predictor variables, 
encoding_dim = int(input_dim/2)
hidden_dim = int(encoding_dim / 2)
learning_rate = 1e-5
print(encoding_dim)
print(hidden_dim)
input_layer = Input(shape=(input_dim, ))
#encoder = Dense(encoding_dim, activation="tanh", activity_regularizer=regularizers.l1(learning_rate))(input_layer)
#encoder = Dense(hidden_dim, activation="relu")(encoder)
#decoder = Dense(hidden_dim, activation="tanh")(encoder)
#decoder = Dense(encoding_dim, activation="relu")(decoder)
#decoder = Dense(input_dim, activation="tanh")(decoder)
#autoencoder = Model(inputs=input_layer, outputs=decoder)
#autoencoder.summary()



encoder = Dense(encoding_dim, activation="selu", activity_regularizer=regularizers.l1(learning_rate),                kernel_initializer='lecun_normal')(input_layer)
encoder = Dense(200, activation="relu")(encoder)
decoder = Dense(hidden_dim, activation='selu',kernel_initializer='lecun_normal')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)


# In[ ]:


autoencoder.summary()


# In[ ]:


autoencoder.compile(metrics=['accuracy'],
                    loss='mean_squared_error',
                    optimizer='adam')

cp = ModelCheckpoint(filepath="A_autoencoder_classifier_1.h5",
                               save_best_only=True,
                               verbose=0)
tb = TensorBoard(log_dir='./logs',
                histogram_freq=0,
                write_graph=True,
                write_images=True)

history = autoencoder.fit(df_train_0_x_rescaled, df_train_0_x_rescaled,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(df_valid_0_x_rescaled, df_valid_0_x_rescaled),
                    verbose=10,
                    callbacks=[cp, tb]).history


# In[ ]:


history


# In[ ]:


valid_x_predictions = autoencoder.predict(df_valid_x_rescaled)
mse = np.mean(np.power(df_valid_x_rescaled - valid_x_predictions, 2), axis=1)
error_df = pd.DataFrame({'Reconstruction_error': mse,
                        'True_class': df_valid['isFraud']})
precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df.True_class, error_df.Reconstruction_error)
plt.plot(threshold_rt, precision_rt[1:], label="Precision",linewidth=5)
plt.plot(threshold_rt, recall_rt[1:], label="Recall",linewidth=5)
plt.title('Precision and recall for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Precision/Recall')
#plt.xlim(100, 130)
plt.legend()


# In[ ]:


groups = error_df.groupby('True_class')
fig, ax = plt.subplots()

threshold = 100 #to be fixed!
for name, group in groups:
    ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "Fraud" if name == 1 else "Normal")
ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")


# In[ ]:


error_df.describe()


# In[ ]:


test_x_predictions = autoencoder.predict(df_test_x_rescaled)


# In[ ]:


mse = np.mean(np.power(df_test_x_rescaled - test_x_predictions, 2), axis=1)


# In[ ]:


plt.scatter(np.linspace(0,len(mse),len(mse)),mse)


# In[ ]:


sample = pd.read_csv("../input/ieee-fraud-detection/sample_submission.csv")


# In[ ]:





# In[ ]:


tresholds = [1,2, 100,110]

new_res = []
for tres in tresholds:
    for res in mse:
        if res> tres:
            new_res.append(1)
        else:
            new_res.append(0)
    sample["isFraud"] = new_res 
    new_res = []
    filename = "A_submission_0_" + str(tres) + ".csv"
    sample.to_csv(filename, index=False)


# In[ ]:




