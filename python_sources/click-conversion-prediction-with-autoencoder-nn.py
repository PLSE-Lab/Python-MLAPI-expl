#!/usr/bin/env python
# coding: utf-8

# ## Updated the kernel with:
# * Feature normalization in training and test (improved accuracy 15%)
# * Slightly bigger autoencoder network
# * Added t-SNE visualization

# In[1]:


# Load libraries

import numpy as np 
import pandas as pd 
import datetime
import os
import time

import gc  # garbage collection

import pickle
from scipy import stats

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from pylab import rcParams
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 12, 5

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers

np.random.seed(5)


# # EDA
# Redoing little bit, see the below great kernel for details:  
# https://www.kaggle.com/yuliagm/talkingdata-eda-plus-time-patterns

# In[3]:


# Load only 500k rows from train data, test data will be loaded later

train = pd.read_csv('../input/train.csv', nrows =500000, parse_dates=['click_time'])


# In[4]:


# Quick check
train.head()


# ## What are the given features in the dataset?

# In[5]:


plt.figure(figsize=(10, 6))
cols = ['ip', 'app', 'device', 'os', 'channel']
uniques = [len(train[col].unique()) for col in cols]
sns.set(font_scale=1.2)
ax = sns.barplot(cols, uniques, log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
            ha="center") 


# ## Attributed class has high imbalance!

# In[6]:


mean = (train.is_attributed.values == 1).mean()
ax = sns.barplot(['App Downloaded (1)', 'Not Downloaded (0)'], [mean, 1-mean])
ax.set(ylabel='Proportion', title='App Downloaded vs Not Downloaded')
for p, uniq in zip(ax.patches, [mean, 1-mean]):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height+0.01,
            '{}%'.format(round(uniq * 100, 2)),
            ha="center")


# Data is highly skewed. < 0.2% of clicks actually downloaded apps.  
# 99.8% accuracy may completely miss to capture what click actually got converted to download an app. 
# 
# ## Feature Engineering: let us develop more features from the given ones
# 
# I followed part of below kernel, did not take all the features:  
# https://www.kaggle.com/nanomathias/feature-engineering-importance-testing

# In[7]:


# extract day, minute, hour, second from the click_time
train['day'] = train['click_time'].dt.day.astype('uint8')
train['hour'] = train['click_time'].dt.hour.astype('uint8')
train['minute'] = train['click_time'].dt.minute.astype('uint8')
train['second'] = train['click_time'].dt.second.astype('uint8')


# In[8]:


# Groupby Aggregation
GROUPBY_AGGREGATIONS = [
    
    # V1 - GroupBy Features #
    #########################    
    # Variance in day, for ip-app-channel
    {'groupby': ['ip','app','channel'], 'select': 'day', 'agg': 'var'},
    # Variance in hour, for ip-app-os
    {'groupby': ['ip','app','os'], 'select': 'hour', 'agg': 'var'},
    # Variance in hour, for ip-day-channel
    {'groupby': ['ip','day','channel'], 'select': 'hour', 'agg': 'var'},
    # Count, for ip-day-hour
    {'groupby': ['ip','day','hour'], 'select': 'channel', 'agg': 'count'},
    # Count, for ip-app
    {'groupby': ['ip', 'app'], 'select': 'channel', 'agg': 'count'},        
    # Count, for ip-app-os
    {'groupby': ['ip', 'app', 'os'], 'select': 'channel', 'agg': 'count'},
    # Count, for ip-app-day-hour
    {'groupby': ['ip','app','day','hour'], 'select': 'channel', 'agg': 'count'},
    # Mean hour, for ip-app-channel
    {'groupby': ['ip','app','channel'], 'select': 'hour', 'agg': 'mean'}, 
    
    # V2 - GroupBy Features #
    #########################
    # Average clicks on app by distinct users; is it an app they return to?
    {'groupby': ['app'], 
     'select': 'ip', 
     'agg': lambda x: float(len(x)) / len(x.unique()), 
     'agg_name': 'AvgViewPerDistinct'
    },
    # How popular is the app or channel?
    {'groupby': ['app'], 'select': 'channel', 'agg': 'count'},
    {'groupby': ['channel'], 'select': 'app', 'agg': 'count'},
    
    # V3 - GroupBy Features                                              #
    # https://www.kaggle.com/bk0000/non-blending-lightgbm-model-lb-0-977 #
    ###################################################################### 
    {'groupby': ['ip'], 'select': 'channel', 'agg': 'nunique'}, 
    {'groupby': ['ip'], 'select': 'app', 'agg': 'nunique'}, 
    {'groupby': ['ip','day'], 'select': 'hour', 'agg': 'nunique'}, 
    {'groupby': ['ip','app'], 'select': 'os', 'agg': 'nunique'}, 
    {'groupby': ['ip'], 'select': 'device', 'agg': 'nunique'}, 
    {'groupby': ['app'], 'select': 'channel', 'agg': 'nunique'}, 
    {'groupby': ['ip', 'device', 'os'], 'select': 'app', 'agg': 'nunique'}, 
    {'groupby': ['ip','device','os'], 'select': 'app', 'agg': 'cumcount'}, 
    {'groupby': ['ip'], 'select': 'app', 'agg': 'cumcount'}, 
    {'groupby': ['ip'], 'select': 'os', 'agg': 'cumcount'}, 
    {'groupby': ['ip','day','channel'], 'select': 'hour', 'agg': 'var'}    
]

# Apply all the groupby transformations
for spec in GROUPBY_AGGREGATIONS:
    
    # Name of the aggregation we're applying
    agg_name = spec['agg_name'] if 'agg_name' in spec else spec['agg']
    
    # Name of new feature
    new_feature = '{}_{}_{}'.format('_'.join(spec['groupby']), agg_name, spec['select'])
    
    # Info
    #print("Grouping by {}, and aggregating {} with {}".format(
    #    spec['groupby'], spec['select'], agg_name
    #))
    
    # Unique list of features to select
    all_features = list(set(spec['groupby'] + [spec['select']]))
    
    # Perform the groupby
    gp = train[all_features].         groupby(spec['groupby'])[spec['select']].         agg(spec['agg']).         reset_index().         rename(index=str, columns={spec['select']: new_feature})
        
    # Merge back to X_total
    if 'cumcount' == spec['agg']:
        train[new_feature] = gp[0].values
    else:
        train = train.merge(gp, on=spec['groupby'], how='left')
        
     # Clear memory
    del gp
    gc.collect()


# In[9]:


# Clicks on app ad before & after

HISTORY_CLICKS = {
    'identical_clicks': ['ip', 'app', 'device', 'os', 'channel'],
    'app_clicks': ['ip', 'app']
}

# Go through different group-by combinations
for fname, fset in HISTORY_CLICKS.items():
    
    # Clicks in the past
    train['prev_'+fname] = train.         groupby(fset).         cumcount().         rename('prev_'+fname)
        
    # Clicks in the future
    train['future_'+fname] = train.iloc[::-1].         groupby(fset).         cumcount().         rename('future_'+fname).iloc[::-1]


# In[10]:


#train.info()


# In[11]:


train = train.drop(['click_time', 'attributed_time'], axis=1)


# In[12]:


train=train.replace(np.nan, 0)


# In[13]:


train.isnull().values.any()


# In[14]:


traincolnames = list(train.columns.values)


# ### Normalize the features for model input

# In[15]:


for v in traincolnames:
    if v!= 'is_attributed':
        train[v] = StandardScaler().fit_transform(train[v].values.reshape(-1, 1))


# In[16]:


train.shape


# In[17]:


train.describe()


# ## Model building with Autoencoder Neural network using Keras
# 
# I shall apply Autoencoder neural network with Keras for this highly skewed dataset.  
# Check my previous Kernel [here](https://www.kaggle.com/mnpathak1/fraud-detection-analysis-with-nn) for similar analysis and references on Fraud transaction analysis.   

# In[18]:


Converted = train[train.is_attributed == 1]
DidNotConvert = train[train.is_attributed == 0]


# In[19]:


Converted.shape


# In[20]:


DidNotConvert.shape


# In[21]:


X_train, X_test = train_test_split(train, test_size=0.2, random_state=5)  # split the train data for training model
X_train = X_train[X_train.is_attributed == 0]    # train on 0 class
X_train = X_train.drop(['is_attributed'], axis=1)   
y_test = X_test['is_attributed']
X_test = X_test.drop(['is_attributed'], axis=1)  
X_train = X_train.values
X_test = X_test.values


# In[22]:


X_train.shape, X_test.shape


# I am using a 9 layer network icluding the input and output layers.  Number of neurons in each layers are 35, 20, 10, 5, 4, 5, 10, 20, 35. Input and output layers have same number of neurons.

# In[23]:


input_dim = X_train.shape[1]
encoding_dim = 20

input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="relu", activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
encoder = Dense(int(encoding_dim / 4), activation="relu")(encoder)
encoder = Dense(int((encoding_dim / 4)-1), activation="relu")(encoder)
decoder = Dense(int(encoding_dim / 4), activation='relu')(encoder)
decoder = Dense(int(encoding_dim / 2), activation='relu')(decoder)
decoder = Dense(int(encoding_dim ), activation='relu')(decoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)


# In[24]:


epoch = 10    # large number of iterations help neural network accuracy
batch_size = 25  # small batch size around 30 is typically good


# In[25]:


autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])


# In[26]:


# If needed to use the model later, save it locally
#checkpointer = ModelCheckpoint(filepath="model3.h5", verbose=0, save_best_only=True)
#tensorboard = TensorBoard(log_dir='./logs3', histogram_freq=0,   write_graph=True, write_images=True)


# In[27]:


history = autoencoder.fit(X_train, X_train,
                    epochs=epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_test, X_test),
                    verbose=1,
                    #callbacks=[checkpointer, tensorboard]
                    ).history


# ## Model evaluation
# 
# ### Loss vs. epoch

# In[28]:


plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right');


# ### Accuracy vs. epoch

# In[29]:


plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right');


# ## Model error and characteristics
# 
# Predict on test data and calculate mse.

# In[30]:


predictions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse, 'Converted': y_test})
error_df.describe()


# In[31]:


predictions.shape


# Distribution of error for non-converted (significant majority of the population).

# In[32]:


fig = plt.figure()
ax = fig.add_subplot(111)
DidNotConvert_error_df = error_df[(error_df['Converted']== 0) & (error_df['reconstruction_error'] )]
_ = ax.hist(DidNotConvert_error_df.reconstruction_error.values, bins=20)


# Distribution of error for all that converted.

# In[33]:


fig = plt.figure()
ax = fig.add_subplot(111)
Converted_error_df = error_df[error_df['Converted'] == 1]
_ = ax.hist(Converted_error_df.reconstruction_error.values, bins=20)


# ### ROC
# 
# Since the dataset is highly skewed (prediction class < 0.2%), ROC is actually not significant for this problem.

# In[34]:


from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)


# In[35]:


fpr, tpr, thresholds = roc_curve(error_df.Converted, error_df.reconstruction_error)
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


# ### Recall vs. precision

# In[36]:


precision, recall, th = precision_recall_curve(error_df.Converted, error_df.reconstruction_error)
plt.plot(recall, precision, 'b', label='Precision-Recall curve')
plt.title('Recall vs Precision')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()


# ### Precision at different threshold

# In[37]:


plt.plot(th, precision[1:], 'b', label='Threshold-Precision curve')
plt.title('Precision for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Precision')
plt.show()


# ### Recall at different threshold

# In[38]:


plt.plot(th, recall[1:], 'b', label='Threshold-Recall curve')
plt.title('Recall for different threshold values')
plt.xlabel('Reconstruction error')
plt.ylabel('Recall')
plt.show()


# ### Error scatterplot showing a threshold

# In[39]:


threshold = 0.5


# In[40]:


groups = error_df.groupby('Converted')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "Converted" if name == 1 else "Did not convert")
ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show();


# ### COnfusion matrix for aparticular threshold

# In[41]:


LABELS = ["Did not convert", "Converted"]
y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
conf_matrix = confusion_matrix(error_df.Converted, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('Actual class')
plt.xlabel('Predicted class')
plt.show()


# Model captures part of click conversion but also labels significant false click conversion. Therefore need to improve the model:
# * Need more training with more number of epochs, more data and CPU/GPU space (I had 340% CPU usage during this trial). I trained the model on 500k data, did not dare to use >1GB training data on my local machine or here. 
# * Better feature selection. I did not add all the features from the previous analysis provided due to memory issue, did not select or remove any particular features.
# * Need to explore different Autoencoder network.
# * Compare with other ML outputs such as logistic regression, RF classifier, GBM.
# 
# Now I am going to prepare the test dataset to apply this model.
# 
# ## Applying features to test data
# 
# Did not do it together with train dataset and taking only 500k rows of test dataset due to memory / timeout issues. 

# In[43]:


test = pd.read_csv('../input/test.csv', nrows=500000, parse_dates=['click_time'])  # I am not using the whole test set
test.head()


# In[44]:


# Check feature counts on test dataset

plt.figure(figsize=(10, 6))
cols = ['ip', 'app', 'device', 'os', 'channel']
uniques = [len(test[col].unique()) for col in cols]
sns.set(font_scale=1.2)
ax = sns.barplot(cols, uniques, log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
           ha="center") 


# ### Get same features on test datset as in the model or train dataset 

# In[45]:


# extract day, minute, hour, second from the click_time
test['day'] = test['click_time'].dt.day.astype('uint8')
test['hour'] = test['click_time'].dt.hour.astype('uint8')
test['minute'] = test['click_time'].dt.minute.astype('uint8')
test['second'] = test['click_time'].dt.second.astype('uint8')


# In[46]:


# Groupby Aggregation
GROUPBY_AGGREGATIONS = [
    
    # V1 - GroupBy Features #
    #########################    
    # Variance in day, for ip-app-channel
    {'groupby': ['ip','app','channel'], 'select': 'day', 'agg': 'var'},
    # Variance in hour, for ip-app-os
    {'groupby': ['ip','app','os'], 'select': 'hour', 'agg': 'var'},
    # Variance in hour, for ip-day-channel
    {'groupby': ['ip','day','channel'], 'select': 'hour', 'agg': 'var'},
    # Count, for ip-day-hour
    {'groupby': ['ip','day','hour'], 'select': 'channel', 'agg': 'count'},
    # Count, for ip-app
    {'groupby': ['ip', 'app'], 'select': 'channel', 'agg': 'count'},        
    # Count, for ip-app-os
    {'groupby': ['ip', 'app', 'os'], 'select': 'channel', 'agg': 'count'},
    # Count, for ip-app-day-hour
    {'groupby': ['ip','app','day','hour'], 'select': 'channel', 'agg': 'count'},
    # Mean hour, for ip-app-channel
    {'groupby': ['ip','app','channel'], 'select': 'hour', 'agg': 'mean'}, 
    
    # V2 - GroupBy Features #
    #########################
    # Average clicks on app by distinct users; is it an app they return to?
    {'groupby': ['app'], 
     'select': 'ip', 
     'agg': lambda x: float(len(x)) / len(x.unique()), 
     'agg_name': 'AvgViewPerDistinct'
    },
    # How popular is the app or channel?
    {'groupby': ['app'], 'select': 'channel', 'agg': 'count'},
    {'groupby': ['channel'], 'select': 'app', 'agg': 'count'},
    
    # V3 - GroupBy Features                                              #
    # https://www.kaggle.com/bk0000/non-blending-lightgbm-model-lb-0-977 #
    ###################################################################### 
    {'groupby': ['ip'], 'select': 'channel', 'agg': 'nunique'}, 
    {'groupby': ['ip'], 'select': 'app', 'agg': 'nunique'}, 
    {'groupby': ['ip','day'], 'select': 'hour', 'agg': 'nunique'}, 
    {'groupby': ['ip','app'], 'select': 'os', 'agg': 'nunique'}, 
    {'groupby': ['ip'], 'select': 'device', 'agg': 'nunique'}, 
    {'groupby': ['app'], 'select': 'channel', 'agg': 'nunique'}, 
    {'groupby': ['ip', 'device', 'os'], 'select': 'app', 'agg': 'nunique'}, 
    {'groupby': ['ip','device','os'], 'select': 'app', 'agg': 'cumcount'}, 
    {'groupby': ['ip'], 'select': 'app', 'agg': 'cumcount'}, 
    {'groupby': ['ip'], 'select': 'os', 'agg': 'cumcount'}, 
    {'groupby': ['ip','day','channel'], 'select': 'hour', 'agg': 'var'}    
]

# Apply all the groupby transformations
for spec in GROUPBY_AGGREGATIONS:
    
    # Name of the aggregation we're applying
    agg_name = spec['agg_name'] if 'agg_name' in spec else spec['agg']
    
    # Name of new feature
    new_feature = '{}_{}_{}'.format('_'.join(spec['groupby']), agg_name, spec['select'])
    
    # Info
    #print("Grouping by {}, and aggregating {} with {}".format(
    #    spec['groupby'], spec['select'], agg_name
    #))
    
    # Unique list of features to select
    all_features = list(set(spec['groupby'] + [spec['select']]))
    
    # Perform the groupby
    gp = test[all_features].         groupby(spec['groupby'])[spec['select']].         agg(spec['agg']).         reset_index().         rename(index=str, columns={spec['select']: new_feature})
        
    # Merge back to X_total
    if 'cumcount' == spec['agg']:
        test[new_feature] = gp[0].values
    else:
        test = test.merge(gp, on=spec['groupby'], how='left')
        
     # Clear memory
    del gp
    gc.collect()


# In[47]:


# Clicks on app ad before & after

HISTORY_CLICKS = {
    'identical_clicks': ['ip', 'app', 'device', 'os', 'channel'],
    'app_clicks': ['ip', 'app']
}

# Go through different group-by combinations
for fname, fset in HISTORY_CLICKS.items():
    
    # Clicks in the past
    test['prev_'+fname] = test.         groupby(fset).         cumcount().         rename('prev_'+fname)
        
    # Clicks in the future
    test['future_'+fname] = test.iloc[::-1].         groupby(fset).         cumcount().         rename('future_'+fname).iloc[::-1]


# In[48]:


test = test.drop(['click_time'], axis=1)


# In[49]:


test=test.replace(np.nan, 0)


# In[50]:


test.isnull().values.any()


# In[51]:


test.shape


# ### Normalize features in test data

# In[52]:


for v in list(test.columns.values):
    if v!= 'click_id':
        test[v] = StandardScaler().fit_transform(test[v].values.reshape(-1, 1))


# In[53]:


test.head()


# ## Predict based on the mmodel with threshold on test data and submission
# 
# I am going to appply the model on the test data and based on the threshold above, I am going to assign if the click was converted to download.

# In[54]:


click_id = test['click_id']


# In[55]:


test = test.drop(['click_id'], axis=1)


# In[56]:


predictions_test = autoencoder.predict(test)


# In[57]:


predictions_test.shape


# In[58]:


mse_test = np.mean(np.power(test - predictions_test, 2), axis=1)
test_error_df = pd.DataFrame({'reconstruction_error': mse_test})
test_error_df.describe()


# In[59]:


y_test = [1 if e > threshold else 0 for e in test_error_df.reconstruction_error.values]


# In[60]:


click_idData = pd.DataFrame(click_id)
y_testData = pd.DataFrame(y_test)
y_testData.columns = ['is_attributed']
result = pd.concat([click_idData, y_testData], axis=1, join_axes=[click_idData.index])
result.head()


# In[61]:


#result.to_csv("TalkingData_Submission_v3.csv",index=False)


# ## Visualizing data with t-SNE analysis
# 
# I am taking all is_attributed=1 and sample from is_attributed=0 in the train data to get the t-SNE visualization.

# In[62]:


from sklearn.manifold import TSNE


# In[63]:


# Set the dataset for t-SNE plot

df2 = train[train.is_attributed == 1]
df2 = pd.concat([df2, train[train.is_attributed == 0].sample(n = 5000)], axis = 0)


# In[64]:


#Scale features to improve the training ability of TSNE.
standard_scaler = StandardScaler()
df2_std = standard_scaler.fit_transform(df2)


# In[65]:


#Set y equal to the target values i.e. is_attributed column and all rows
y = df2.iloc[:,5].values


# In[66]:


tsne = TSNE(n_components=2, random_state=0)
x_test_2d = tsne.fit_transform(df2_std)


# In[67]:


#Build the scatter plot with the two types of transactions.
color_map = {0:'red', 1:'blue'}
plt.figure()
for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x = x_test_2d[y==cl,0], 
                y = x_test_2d[y==cl,1], 
                c = color_map[idx], 
                label = cl)
plt.xlabel('X in t-SNE')
plt.ylabel('Y in t-SNE')
plt.legend(loc='upper left')
plt.title('t-SNE visualization of a sample of train data')
plt.show()


# We see that converted (is_attributed=1) and rest of non-converted clicks are separated in this t-SNE plot. Feature engineeting and normalization seem to have worked.
