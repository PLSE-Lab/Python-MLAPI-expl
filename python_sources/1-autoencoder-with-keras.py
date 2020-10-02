#!/usr/bin/env python
# coding: utf-8

# Note that the autoencoder code are borrowed from the following notebook: https://github.com/curiousily/Credit-Card-Fraud-Detection-using-Autoencoders-in-Keras/blob/master/fraud_detection.ipynb
# 
# The code used for summary statistics / dtype fixing belongs to ZihaoXu.

# In[ ]:


# important packages to import
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
import pickle

from scipy import stats
from pylab import rcParams
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers

from matplotlib import offsetbox
from matplotlib.ticker import NullFormatter
from sklearn import preprocessing, cross_validation, svm, manifold
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier # Load scikit's random forest classifier library
from sklearn.grid_search import GridSearchCV

from time import time
from datetime import datetime, timedelta
from collections import defaultdict

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


# holistic summary of the given data set. 
# "remove_bad_rowCol" can be turned on to remove non-informative col / row
def holistic_summary(df, remove_bad_rowCol = False, verbose = True):
    # remove non-informative columns
    if(remove_bad_rowCol):
        df = df.drop(df.columns[df.isnull().sum() >= .9 * len(df)], axis = 1)
        df = df.drop(df.index[df.isnull().sum(axis = 1) >= .5* len(df.columns)], axis = 0)
        
    # fix column names:
    df.columns = [c.replace(" ", "_").lower() for c in df.columns]
    
    print('***************************************************************')
    print('Begin holistic summary: ')
    print('***************************************************************\n')
    
    print('Dimension of df: ' + str(df.shape))
    print('Percentage of good observations: ' + str(1 - df.isnull().any(axis = 1).sum()/len(df)))
    print('---------------------------------------------------------------\n')
    
    print("Rows with nan values: " + str(df.isnull().any(axis = 1).sum()))
    print("Cols with nan values: " + str(df.isnull().any(axis = 0).sum()))
    print('Breakdown:')
    print(df.isnull().sum()[df.isnull().sum()!=0])
    print('---------------------------------------------------------------\n')
    
    print('Columns details: ')
    print('Columns with known dtypes: ')
    good_cols = pd.DataFrame(df.dtypes[df.dtypes!='object'], columns = ['type'])
    good_cols['nan_num'] = [df[col].isnull().sum() for col in good_cols.index]
    good_cols['unique_val'] = [df[col].nunique() for col in good_cols.index]
    good_cols['example'] = [df[col][1] for col in good_cols.index]
    good_cols = good_cols.reindex(good_cols['type'].astype(str).str.len().sort_values().index)
    print(good_cols)
    print('\n')
    
    try:
        print('Columns with unknown dtypes:')
        bad_cols = pd.DataFrame(df.dtypes[df.dtypes=='object'], columns = ['type'])
        bad_cols['nan_num'] = [df[col].isnull().sum() for col in bad_cols.index]
        bad_cols['unique_val'] = [df[col].nunique() for col in bad_cols.index]
        bad_cols['example(sliced)'] = [str(df[col][1])[:10] for col in bad_cols.index]
        bad_cols = bad_cols.reindex(bad_cols['example(sliced)'].str.len().sort_values().index)
        print(bad_cols)
    except Exception as e:
        print('No columns with unknown dtypes!')
    print('_______________________________________________________________\n\n\n')
    #if not verbose: enablePrint()
    return df
def memo(df):
    mem = df.memory_usage(index=True).sum()
    print(mem/ 1024**2," MB")


# In[ ]:


trans = pd.read_csv('../input/transactions.csv')
# Memory Reduction
def change_datatype(df):
    int_cols = list(df.select_dtypes(include=['int']).columns)
    for col in int_cols:
        if ((np.max(df[col]) <= 127) and(np.min(df[col] >= -128))):
            df[col] = df[col].astype(np.int8)
        elif ((np.max(df[col]) <= 32767) and(np.min(df[col] >= -32768))):
            df[col] = df[col].astype(np.int16)
        elif ((np.max(df[col]) <= 2147483647) and(np.min(df[col] >= -2147483648))):
            df[col] = df[col].astype(np.int32)
        else:
            df[col] = df[col].astype(np.int64)
            
def change_datatype_float(df):
    float_cols = list(df.select_dtypes(include=['float']).columns)
    for col in float_cols:
        df[col] = df[col].astype(np.float32)

change_datatype(trans)
change_datatype_float(trans)
memo(trans)


# In[ ]:


# fixing dtypes: time and numeric variables
def fix_dtypes(df, time_cols, num_cols):
    
    print('***************************************************************')
    print('Begin fixing data types: ')
    print('***************************************************************\n')
    
    def fix_time_col(df, time_cols):
        for time_col in time_cols:
            df[time_col] = pd.to_datetime(df[time_col], errors = 'coerce', format = '%Y%m%d')
        print('---------------------------------------------------------------')
        print('The following time columns has been fixed: ')
        print(time_cols)
        print('---------------------------------------------------------------\n')

    def fix_num_col(df, num_cols):
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors = 'coerce')
        print('---------------------------------------------------------------')
        print('The following number columns has been fixed: ')
        print(num_cols)
        print('---------------------------------------------------------------\n')
        
    if(len(num_cols) > 0):
        fix_num_col(df, num_cols)
    fix_time_col(df, time_cols)

    print('---------------------------------------------------------------')
    print('Final data types:')
    result = pd.DataFrame(df.dtypes, columns = ['type'])
    result = result.reindex(result['type'].astype(str).str.len().sort_values().index)
    print(result)
    print('_______________________________________________________________\n\n\n')
    return df


# In[ ]:


np.random.seed(47)
samp = trans.sample(frac = .01, replace = False)
train = pd.read_csv('../input/train.csv')
train = train.append(pd.read_csv('../input/train_v2.csv'))
train.index = range(len(train))

test = pd.read_csv('../input/sample_submission_zero.csv')
test = test.append(pd.read_csv('../input/sample_submission_v2.csv'))
test.index = range(len(test))

members = pd.read_csv('../input/members_v3.csv')

samp = samp.merge(train, on = 'msno', how = 'inner')
samp = samp.merge(members, on = 'msno', how = 'inner')

samp.head()


# In[ ]:


samp = fix_dtypes(samp, time_cols = ['transaction_date', 'membership_expire_date', 'registration_init_time'], num_cols = [])
samp['year'] = [d.year for d in samp['transaction_date']]
samp['month'] = [d.month for d in samp['transaction_date']]
samp['day'] = [d.day for d in samp['transaction_date']]
samp['wday'] = [d.weekday() for d in samp['transaction_date']]
samp['transaction_date'] = [d.year + (d.month-1) / 12 + d.day / 365 for d in samp['transaction_date']]
samp['membership_expire_date'] = [d.year + (d.month-1) / 12 + d.day / 365 for d in samp['membership_expire_date']]
samp['registration_init_time'] = [d.year + (d.month-1) / 12 + d.day / 365 for d in samp['registration_init_time']]
memo(samp)


# In[ ]:


from multiprocessing import Pool, cpu_count
import gc; gc.enable()

def transform_df(df):
    df = pd.DataFrame(df)
    df = df.sort_values(by=['date'], ascending=[False])
    df = df.reset_index(drop=True)
    df = df.drop_duplicates(subset=['msno'], keep='first')
    return df

def transform_df2(df):
    df = df.sort_values(by=['date'], ascending=[False])
    df = df.reset_index(drop=True)
    df = df.drop_duplicates(subset=['msno'], keep='first')
    return df

df_iter = pd.read_csv('../input/user_logs.csv', low_memory=False, iterator=True, chunksize=10000000)
last_user_logs = []
i = 0 #~400 Million Records - starting at the end but remove locally if needed
for df in df_iter:
    if i>35: # used to be 35, just testing
        if len(df)>0:
            print(df.shape)
            p = Pool(cpu_count())
            df = p.map(transform_df, np.array_split(df, cpu_count()))   
            df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
            df = transform_df2(df)
            p.close(); p.join()
            last_user_logs.append(df)
            print('...', df.shape)
            df = []
    i+=1

last_user_logs = pd.concat(last_user_logs, axis=0, ignore_index=True).reset_index(drop=True)
last_user_logs = transform_df2(last_user_logs)


# In[ ]:


print(last_user_logs.shape)
print(list(last_user_logs))


# In[ ]:


print("Len before the merge: ", len(samp))
samp = samp.merge(last_user_logs, on = 'msno', how = 'inner')
print("Len after the merge: ", len(samp))


# In[ ]:


print(list(samp))
print("Number of observations: " + str(len(samp)))
samp.head()


# # Visualization of the "churn" class

# In[ ]:


# We see that only the gender column has quite a lot of NANs
df = samp
df = holistic_summary(df)


# In[ ]:


# impute the missing genders randomly
import random
np.random.seed(47)

gender = []
for x in df['gender']:
    if type(x) == np.float:
        gender.append(random.choice(['female', 'male']))
    else:
        gender.append(x)
df['gender'] = gender
df['gender'].isnull().any()


# In[ ]:


sns.set(style = 'white')
sns.countplot(data = df, x = 'is_churn')
sns.despine()


# The dataset is quite imbalanced...

# In[ ]:


print("Churn ratio", len(df[df['is_churn'] == 1])/len(df[df['is_churn'] == 0]))


# In[ ]:


churn = df[df['is_churn'] == 1]
n_churn = df[df['is_churn'] == 0]

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by class')

bins = 50
ax1.hist(churn.actual_amount_paid, bins = bins)
ax1.set_title('Churn')

ax2.hist(n_churn.actual_amount_paid, bins = bins)
ax2.set_title('Not Churn')

plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.show();


# In[ ]:


time_cols = ['wday', 'membership_expire_date']

for t in time_cols:
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    f.suptitle('Time of transaction vs Amount by class')
    ax1.scatter(churn[t], churn['actual_amount_paid'], s = 2,alpha = .25)
    ax1.set_title('Churn')

    ax2.scatter(n_churn[t], n_churn['actual_amount_paid'], s = 2, alpha = .25)
    ax2.set_title('Not Churn')

    plt.xlabel('Time')
    plt.ylabel('Amount')
    plt.show()


# # Keras autoencoder

# In[ ]:


# Drop the 'msno', 'bd' cols since have no value
df = df.drop(['msno', 'bd'], axis = 1)


# In[ ]:


# Encode the gender col to 1 for male and 0 for female
df['gender'] = np.where(df['gender'] == 'male', 1, 0)


# In[ ]:


# Plot the correlation matrix
def corr_plot(df, title = 'Correlation Matrix', annot=False, show = True):
    sns.set(style = 'white')

    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, annot=annot,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title(title)
    if show:
        plt.show()
corr_plot(samp[sorted(list(df))])


# In[ ]:


features = np.array(df[[col for col in df.columns if col != 'is_churn']])
response = np.array(df[['is_churn']])

print(len(features))
print(len(response))


# In[ ]:


features


# Let's first standardize the data and apply PCA.

# In[ ]:


from sklearn import decomposition
from sklearn.preprocessing import StandardScaler

# Fit the scaler to the features and transform
features_std = StandardScaler().fit_transform(features)


# In[ ]:


# Create a pca object with the 20 components as a parameter
pca = decomposition.PCA(n_components=20)

# Fit the PCA and transform the data
features_pca = pca.fit_transform(features_std)


# In[ ]:


features_pca.shape


# In[ ]:


viz = pd.DataFrame(features_pca)[[0,1,2]]
viz.columns = [str(c) for c in viz.columns]
viz['is_churn'] = df['is_churn']
sns.lmplot(data = viz, x = '0', y = '1', fit_reg=False, hue = 'is_churn')
plt.show()


# In[ ]:


# Convert features_pca back to a df and add the is_churn column
features_pca = pd.DataFrame(features_pca)
features_pca['is_churn'] = df['is_churn']

features_std = pd.DataFrame(features_std)
features_std['is_churn'] = df['is_churn']


# In[ ]:


from sklearn.cross_validation import train_test_split
X_train, X_test = train_test_split(features_std, test_size=0.2, random_state=47)
X_train = X_train[X_train['is_churn'] == 0]
X_train = X_train.drop(['is_churn'], axis=1)

y_test = X_test['is_churn']
X_test = X_test.drop(['is_churn'], axis=1)

X_train = X_train.values
X_test = X_test.values


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# ## Building the model

# In[ ]:


input_dim = X_train.shape[1]
encoding_dim = 14


# In[ ]:


input_layer = Input(shape=(input_dim, ))

encoder = Dense(encoding_dim, activation="tanh", 
                activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)

decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)


# In[ ]:


nb_epoch = 100
batch_size = 32

autoencoder.compile(optimizer='adam', 
                    loss='mean_squared_error', 
                    metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath="model.h5",
                               verbose=0,
                               save_best_only=True)
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)

history = autoencoder.fit(X_train, X_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_test, X_test),
                    verbose=1,
                    callbacks=[checkpointer, tensorboard]).history


# In[ ]:


autoencoder = load_model('model.h5')


# # Evaluate the Model

# In[ ]:


plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right');


# In[ ]:


predictions = autoencoder.predict(X_test)


# In[ ]:


mse = np.mean(np.power(X_test - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse,
                        'true_class': y_test})


# In[ ]:


error_df.describe()


# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111)
normal_error_df = error_df[(error_df['true_class']== 0) & (error_df['reconstruction_error'] < 10)]
_ = ax.hist(normal_error_df.reconstruction_error.values, bins=10)
plt.title('Reconstruction error without fraud')
sns.despine()


# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111)
fraud_error_df = error_df[error_df['true_class'] == 1]
_ = ax.hist(fraud_error_df.reconstruction_error.values, bins=10)
plt.title('Reconstruction error with fraud')
sns.despine()


# In[ ]:


from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)


# In[ ]:


fpr, tpr, thresholds = roc_curve(error_df.true_class, error_df.reconstruction_error)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();


# In[ ]:


precision, recall, th = precision_recall_curve(error_df.true_class, error_df.reconstruction_error)
plt.plot(recall, precision, 'b', label='Precision-Recall curve')
plt.title('Recall vs Precision')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()


# In[ ]:


plt.plot(th, precision[1:], 'b', label='Threshold-Precision curve')
plt.title('Precision for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Precision')
plt.show()


# In[ ]:


plt.plot(th, recall[1:], 'b', label='Threshold-Recall curve')
plt.title('Recall for different threshold values')
plt.xlabel('Reconstruction error')
plt.ylabel('Recall')
plt.show()


# # Prediction

# In[ ]:


threshold = 2.9


# In[ ]:


groups = error_df.groupby('true_class')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "churn" if name == 1 else "not churn")
ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
sns.despine()
plt.show();


# In[ ]:


LABELS = ['churn', 'not churn']

y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
conf_matrix = confusion_matrix(error_df.true_class, y_pred)

plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


# In[ ]:




