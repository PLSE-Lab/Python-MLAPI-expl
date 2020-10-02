#!/usr/bin/env python
# coding: utf-8

# # What's New

# These are some of the simple ideas that were not present in my previous public kernel
# 
# *  A better NN Architecture inspired by DenseNet
# 
# *  A better feature pre-processing method (PowerTransformer) for NN.
# 
# *  Post processing the predicted probabilities
# 
# Some other useful ideas (that i am not including here to avoid making the kernel messy)
# 
# * Resampling the time series
# 
#     I found this to be really useful for training neural networks. You can get a boost of around +0.01 if you decide to use this.
# 
# * Second level modelling
# 
#     You can stack the predictions from nn with other methods like lgb,xgb and get very good boost. 
#     
# * A custom loss
# 
#     You can get some interesting insights from analysing the confusion matrix and design a custom loss accordingly. For example class 90 keeps getting confused with class 42 and 52. You can try to suppress such predictions.
# 
# * Neural Network weight initialization
# 
#     You can try experimenting with different weight initialization schemes. It helps in faster convergance and in finding a better minima.
# 
# 
# 
# 

# # Importing Libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import gc
import os
import matplotlib.pyplot as plt
import seaborn as sns 
import lightgbm as lgb
from catboost import Pool, CatBoostClassifier
import itertools
import pickle, gzip
import glob


# In[ ]:


def lgb_multi_weighted_logloss(y_true, y_preds):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    if len(np.unique(y_true)) > 14:
        classes.append(99)
        class_weight[99] = 2
    y_p = y_preds.reshape(y_true.shape[0], len(classes), order='F')
    
    # Trasform y_true in dummies
    y_ohe = pd.get_dummies(y_true)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1-1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set 
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos
    
    loss = - np.sum(y_w) / np.sum(class_arr)
    return 'wloss', loss, False


# # Aggregating features globally (i.e. considering all passbands) and passband wise.

# In[ ]:


gc.enable()

train = pd.read_csv('../input/PLAsTiCC-2018/training_set.csv')

train['flux_ratio_sq'] = np.power(train['flux'] / train['flux_err'], 2.0)
train['flux_by_flux_ratio_sq'] = train['flux'] * train['flux_ratio_sq']

aggs = {
    'flux': ['min', 'max', 'mean', 'median', 'std','sum','skew'],
    'flux_err': ['min', 'max', 'mean','skew'],
    'detected': [ 'mean', 'std','sum'],
    'flux_ratio_sq':['mean','sum','skew'],
    'flux_by_flux_ratio_sq':['mean','sum','skew'],
}

aggs_global = {
    'mjd': ['size'],
    'flux': ['min', 'max', 'mean', 'median', 'std','sum','skew'],
    'flux_err': ['min', 'max', 'mean', 'median', 'std','sum','skew'],
    'detected': [ 'mean','skew','median','sum'],
    'flux_ratio_sq':['min', 'max', 'mean','sum','skew'],
    'flux_by_flux_ratio_sq':['min', 'max', 'mean','sum','skew'],
}

agg_train_global_feat = train.groupby('object_id').agg(aggs_global)

new_columns = [
    k + '_' + agg for k in aggs.keys() for agg in aggs[k]
]

new_columns_global = [
    k + '_' + agg for k in aggs_global.keys() for agg in aggs_global[k]
]

agg_train_global_feat.columns = new_columns_global

agg_train = train.groupby(['object_id','passband']).agg(aggs)

agg_train = agg_train.unstack()

col_names = []
for col in new_columns:
    for i in range(6):
        col_names.append(col+'_'+str(i))
        
agg_train.columns = col_names
agg_train_global_feat['flux_diff'] = agg_train_global_feat['flux_max'] - agg_train_global_feat['flux_min']
agg_train_global_feat['flux_dif2'] = (agg_train_global_feat['flux_max'] - agg_train_global_feat['flux_min']) / agg_train_global_feat['flux_mean']
agg_train_global_feat['flux_w_mean'] = agg_train_global_feat['flux_by_flux_ratio_sq_sum'] / agg_train_global_feat['flux_ratio_sq_sum']
agg_train_global_feat['flux_dif3'] = (agg_train_global_feat['flux_max'] - agg_train_global_feat['flux_min']) / agg_train_global_feat['flux_w_mean']


# # Computing [Sionkowski's Feature](https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69696#410538)

# In[ ]:


# Legacy code. There are much better ways to compute this but for train set this suffices so 
# i got too lazy to change https://www.kaggle.com/c/PLAsTiCC-2018/discussion/71398
def detected_max(mjd,detected):
    try:     return max(mjd[detected==1]) - min(mjd[detected==1])
    except:  return 0
    
temp = train.groupby('object_id').apply(lambda x:detected_max(x['mjd'],x['detected']))
temp1 = train.groupby(['object_id','passband']).apply(lambda x:detected_max(x['mjd'],x['detected'])).unstack()
temp.columns = ['mjd_global_diff']
temp1.columns = ['mjd_pb0','mjd_pb1','mjd_pb2','mjd_pb3','mjd_pb4','mjd_pb5']
temp = temp.reset_index()
temp1 = temp1.reset_index()


# # Aggregating features only on detected events

# In[ ]:


aggs_det = {
    'flux': ['min','mean', 'max','skew'],
    'flux_ratio_sq':['min','mean', 'max','skew'],
    'flux_by_flux_ratio_sq':['min', 'max','mean','skew'],
}

train_detected =  train[train.detected==1]
temp2 = train_detected.groupby(['object_id']).agg(aggs_det)
       
new_columns_det = [
    k + '_det_' + agg for k in aggs_det.keys() for agg in aggs_det[k]
]

temp2.columns = new_columns_det
temp2['flux_diff_det'] = temp2['flux_det_max'] - temp2['flux_det_min']
temp2['flux_ratio_sq_diff_det'] = temp2['flux_ratio_sq_det_max'] - temp2['flux_ratio_sq_det_min']
temp2['flux_by_flux_ratio_sq_diff_det'] = temp2['flux_by_flux_ratio_sq_det_max'] - temp2['flux_by_flux_ratio_sq_det_min']

del temp2['flux_by_flux_ratio_sq_det_max'],temp2['flux_by_flux_ratio_sq_det_min']
del temp2['flux_ratio_sq_det_max'],temp2['flux_ratio_sq_det_min']
del temp2['flux_det_max'],temp2['flux_det_min']
gc.collect()


# # Merging all the aggregated features

# In[ ]:


meta_train = pd.read_csv('../input/PLAsTiCC-2018/training_set_metadata.csv')

full_train = agg_train.reset_index().merge(
    right=meta_train,
    how='outer',
    on='object_id'
)

full_train = full_train.merge(
    right=agg_train_global_feat,
    how='outer',
    on='object_id'
)

full_train = full_train.merge(
    right=temp,
    how='outer',
    on='object_id'
)

full_train = full_train.merge(
    right=temp1,
    how='outer',
    on='object_id'
)

full_train = full_train.merge(
    right=temp2,
    how='outer',
    on='object_id'
)

if 'target' in full_train:
    y = full_train['target']
    del full_train['target']
classes = sorted(y.unique())

class_weight = {
    c: 1 for c in classes
}
for c in [64, 15]:
    class_weight[c] = 2

print('Unique classes : ', classes)


# In[ ]:


if 'object_id' in full_train:
    oof_df = full_train[['object_id']]
    del full_train['object_id'], full_train['hostgal_specz']
    del full_train['ra'], full_train['decl'], full_train['gal_l'],full_train['gal_b'],full_train['ddf']


# In[ ]:


# # one possible way to do resampling
# train = pd.read_csv('../input/PLAsTiCC-2018/training_set.csv')
# # train = train.sample(frac=0.6)
# train['flux_ratio_sq'] = np.power(train['flux'] / train['flux_err'], 2.0)
# train['flux_by_flux_ratio_sq'] = train['flux'] * train['flux_ratio_sq']
# train.set_index('object_id',inplace=True)
# train_sampled = [train.loc[obj_id,:].sample(frac=random.uniform(0.3,0.7)) for obj_id in train.index.unique()]
# train = pd.concat(train_sampled,axis=0)
# train = train.reset_index()
# train.sort_values( ['object_id', 'mjd'], ascending=True, inplace=True )


# # Eliminating the least useful features based on feature importance

# In[ ]:


useless_cols = [
'flux_max_2',
'flux_median_0',
'flux_median_4',
'flux_err_skew_1',
'flux_err_skew_3',
'detected_mean_4',
'detected_std_3',
'detected_std_4',
'detected_sum_4',
'flux_ratio_sq_mean_4',
'flux_ratio_sq_sum_3',
'flux_ratio_sq_sum_4',
'flux_median',
'flux_err_skew',
'flux_ratio_sq_sum',
'mjd_pb5',
'flux_ratio_sq_det_skew',
]


# # Preprocessing using PowerTransformer

# For neural networks, preprocessing the data is** extremely important**. I tried all the available preprocessing methods available in sklearn and some other methods also. I found PowerTransform and GaussRank Scaler to be the best methods. Power transforms helps to make the data more Gaussian like.  For more details  [refer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html).

# In[ ]:


full_train_new = full_train.drop(useless_cols,axis=1)
from sklearn.preprocessing import PowerTransformer
ss = PowerTransformer()
full_train_ss = ss.fit_transform(np.nan_to_num(full_train_new))


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense,BatchNormalization,Dropout
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from keras.utils import to_categorical
import tensorflow as tf
from keras import backend as K
import keras
from keras import regularizers
from collections import Counter
from sklearn.metrics import confusion_matrix


# # Loss Function
# 
# Focal Loss is generally useful for imbalanced classes . You can find more details [here](https://arxiv.org/abs/1708.02002) . If you use it for training it leads to higher accuracy but the lb score will be slightly lower since we are not directly optimizing the actual objective.
# 

# In[ ]:


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        yc = tf.clip_by_value(y_pred,1e-15,1-1e-15)
        pt_1 = tf.where(tf.equal(y_true, 1), yc, tf.ones_like(yc))
        pt_0 = tf.where(tf.equal(y_true, 0), yc, tf.zeros_like(yc))
        return (-K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0)))
    return focal_loss_fixed


# In[ ]:


def mywloss(y_true,y_pred):  
    yc=tf.clip_by_value(y_pred,1e-15,1-1e-15)
    loss=-(tf.reduce_mean(tf.reduce_mean(y_true*tf.log(yc),axis=0)/wtable))
#     a sample custom loss
#     loss=-2*(tf.reduce_mean(tf.reduce_mean(y_true*tf.log(yc),axis=0)/wtable_new)) \
#     + 0.5*tf.reduce_mean(tf.reduce_mean((1-y_true[:,11])*yc[:,11],axis=0)) + 0.10*tf.reduce_mean(tf.reduce_mean((1-y_true[:,6])*yc[:,6],axis=0))
#     + 0.25*tf.reduce_mean(tf.reduce_mean((1-y_true[:,9])*yc[:,9],axis=0)) + 0.1*tf.reduce_mean(tf.reduce_mean((1-y_true[:,4])*yc[:,4],axis=0))
    return loss


# In[ ]:


def multi_weighted_logloss(y_ohe, y_p):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1-1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set 
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos
    
    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss


# In[ ]:


unique_y = np.unique(y)
class_map = dict()
for i,val in enumerate(unique_y):
    class_map[val] = i
    
orig_class_map = dict()
for i,val in enumerate(unique_y):
    orig_class_map[i] = val
    
y_map = np.zeros((y.shape[0],))
y_map = np.array([class_map[val] for val in y])
y_categorical = to_categorical(y_map)

y_count = Counter(y_map)
wtable = np.zeros((len(unique_y),))
for i in range(len(unique_y)):
    wtable[i] = y_count[i]/y_map.shape[0]


# In[ ]:


def plot_loss_acc(history):
    plt.figure(figsize=(20,7))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'][1:])
    plt.plot(history.history['val_loss'][1:])
    plt.title('model loss')
    plt.ylabel('val_loss')
    plt.xlabel('epoch')
    plt.legend(['Train','Validation'], loc='upper left')
    
    plt.subplot(1,2,2)
    plt.plot(history.history['acc'][1:])
    plt.plot(history.history['val_acc'][1:])
    plt.title('Model Accuracy')
    plt.ylabel('val_acc')
    plt.xlabel('epoch')
    plt.legend(['Train','Validation'], loc='upper left')
    plt.show()


# In[ ]:


from keras.layers import Input,Dense,Conv1D,MaxPool1D,GlobalMaxPooling1D,Add,GlobalAveragePooling1D,Reshape,multiply
from keras.layers.merge import concatenate
from keras.models import Model
from keras import callbacks


# A slightly better keras architecture inspired by DenseNet
# 
#  The last layer has 3 inputs, consisting of the feature maps of all preceding convolutional blocks. 

# In[ ]:


K.clear_session()
def build_model(dropout_rate=0.25,activation='relu'):
    start_neurons = 256
    feat_ip = Input(shape=(full_train_ss.shape[1],), name='feature_ip')
    
    x = BatchNormalization()(feat_ip)
    x = Dense(start_neurons, activation=activation)(x)    
    feat1 = Dropout(rate=dropout_rate)(x)

    x = BatchNormalization()(feat1)
    x = Dense(start_neurons//2, activation=activation)(x)    
    feat2 = Dropout(rate=dropout_rate)(x)
    
    x = BatchNormalization()(feat2)
    x = Dense(start_neurons//4, activation=activation)(x)    
    feat3 = Dropout(rate=dropout_rate)(x)
    
    x = BatchNormalization()(feat3)
    x = Dense(start_neurons//8, activation=activation)(x)    
    feat4 = Dropout(rate=dropout_rate/2)(x)

    feat_concat = concatenate([feat1,feat2,feat3,feat4])
    out = Dense(len(classes), activation='softmax')(feat_concat)

    model = Model(inputs=[feat_ip], outputs=[out])    
    return model   


# # Cosine Annealing for learning rate scheduling

# I found cosine annealing to be slightly better than ReduceLROnPlateau. You can even use it for creating different [snapshots](http://openreview.net/pdf?id=BJYwwY9ll) of the model. 

# In[ ]:


# https://github.com/titu1994/Snapshot-Ensembles
class SnapshotCallbackBuilder:
    def __init__(self, nb_epochs, nb_snapshots, init_lr=0.1):
        self.T = nb_epochs
        self.M = nb_snapshots #for single model it should be 1
        self.alpha_zero = init_lr

    def get_callbacks(self, model_prefix='Model'):

        callback_list = [
            ModelCheckpoint("./keras.model",monitor='val_loss',mode = 'min', save_best_only=True, verbose=0),
            callbacks.LearningRateScheduler(schedule=self._cosine_anneal_schedule)
        ]

        return callback_list

    def _cosine_anneal_schedule(self, t):
        cos_inner = np.pi * (t % (self.T // self.M))  # t - 1 is used when t has 1-based indexing.
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        return float(self.alpha_zero / 2 * cos_out)


# In[ ]:


clfs = []
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
oof_preds = np.zeros((len(full_train_ss), len(classes)))
epochs = 250
batch_size = 200
for fold_, (trn_, val_) in enumerate(folds.split(y_map, y_map)):

    checkPoint = ModelCheckpoint("./keras.model",monitor='val_loss',mode = 'min', save_best_only=True, verbose=0)
    x_train, y_train = full_train_ss[trn_], y_categorical[trn_]
    x_valid, y_valid = full_train_ss[val_], y_categorical[val_]
    
    model = build_model(dropout_rate=0.5,activation='tanh')    
    # Compile model    
    model.compile(loss=mywloss, optimizer='adam', metrics=['accuracy'])
    snapshot = SnapshotCallbackBuilder(nb_epochs=epochs,nb_snapshots=1,init_lr=1e-3)
    history = model.fit(x_train, y_train,
                    validation_data=[x_valid, y_valid], 
                    epochs=epochs,
                    batch_size=batch_size,shuffle=True,verbose=0,callbacks=[checkPoint])       
    
    plot_loss_acc(history)
    
    print('Loading Best Model')
    model.load_weights('./keras.model')
    # # Get predicted probabilities for each class
    oof_preds[val_, :] = model.predict(x_valid,batch_size=batch_size)
    print(multi_weighted_logloss(y_valid, model.predict(x_valid,batch_size=batch_size)))
    clfs.append(model)
    
print('MULTI WEIGHTED LOG LOSS : %.5f ' % multi_weighted_logloss(y_categorical,oof_preds))


# In[ ]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[ ]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_map, np.argmax(oof_preds,axis=-1))
np.set_printoptions(precision=2)


# In[ ]:


test_files1 = glob.glob('../input/create-test-set/*.gz')
sample_sub = pd.read_csv('../input/PLAsTiCC-2018/sample_submission.csv')
class_names = list(sample_sub.columns[1:-1])
del sample_sub


# In[ ]:


# Plot non-normalized confusion matrix
plt.figure(figsize=(7,7))
foo = plot_confusion_matrix(cnf_matrix, classes=class_names,normalize=True,
                      title='Confusion matrix, without normalization')


# # Test Set Predictions

# I had to split my test set into multiple files for easier computation and running on kaggle kernels. If you have sufficient RAM  no need to go through these tribulations.

# In[ ]:


for i_c,fn in enumerate(test_files1):
    full_test1 = pickle.load(gzip.open(fn, 'rb'))
    test2fn = '../input/create-test-set-new-feat/'+ fn.split('/')[-1]
    test3fn = '../input/create-test-set-sionkowski-feat/'+ fn.split('/')[-1]
    test4fn = '../input/create-test-set-detected-feat/'+ fn.split('/')[-1]
#     print(test2fn)
    full_test2 = pickle.load(gzip.open(test2fn, 'rb'))
    full_test3 = pickle.load(gzip.open(test3fn, 'rb'))
    full_test4 = pickle.load(gzip.open(test4fn, 'rb'))
    
    full_test = full_test1.merge(right=full_test2,how='outer',on='object_id')
    full_test = full_test.merge(right=full_test3,how='outer',on='object_id')
    full_test = full_test.merge(right=full_test4,how='outer',on='object_id')

    object_ids = full_test.object_id.values
    full_test = full_test[full_train_new.columns]
    full_test_ss = ss.transform(np.nan_to_num(full_test))
    
#     Make predictions
    preds = None
    for clf in clfs:
        if preds is None:
            preds = clf.predict(full_test_ss,batch_size=batch_size) / folds.n_splits
        else:
            preds += clf.predict(full_test_ss,batch_size=batch_size) / folds.n_splits
    if i_c % 10 == 0:
        print(i_c+1,'files done')
   # Compute preds_99 as the proba of class not being any of the others
    # preds_99 = 0.1 gives 1.769
    preds_99 = np.ones(preds.shape[0])
    for i in range(preds.shape[1]):
        preds_99 *= (1 - preds[:, i])

    # Store predictions
    preds_df = pd.DataFrame(preds, columns=class_names)
    preds_df['object_id'] = object_ids
    preds_df['class_99'] = 0.18 * preds_99 / np.mean(preds_99) 
    
    if i_c == 0:
        preds_df.to_csv('predictions.csv',  header=True, mode='a', index=False)
    else: 
        preds_df.to_csv('predictions.csv',  header=False, mode='a', index=False)
        
    del full_test, preds_df, preds
    gc.collect()


# In[ ]:


z = pd.read_csv('predictions.csv')

print(z.groupby('object_id').size().max())
print((z.groupby('object_id').size() > 1).sum())

z = z.groupby('object_id').mean()

# z.to_csv('single_predictions.csv', index=True)


# 
# # Probability Correction

# 1. For objects with hostgal_specz=0
# 
#  Ideally mean for each extra-galactic class should be exactly 0 because according to hostgal_specz the classes can only be galactic classes
# 
# 2. For objects with hostgal_specz!=0
# 
#  Ideally mean for each galactic class should be exactly 0 because according to hostgal_specz the classes can only be extra galactic classes
# 
# 

# In[ ]:


test_meta_data = pd.read_csv('../input/PLAsTiCC-2018/test_set_metadata.csv')
gal_obj = test_meta_data[test_meta_data.hostgal_photoz==0].object_id
ex_gal_obj = test_meta_data[test_meta_data.hostgal_photoz!=0].object_id

print('Percentage of galactic object in test set based on hostgal_photoz',len(gal_obj)/len(test_meta_data))
print('Percentage of extra galactic object in test set based on hostgal_photoz',
      len(ex_gal_obj)/len(test_meta_data))


# In[ ]:


gal_classes = [ 6, 16, 53, 65, 92]
gal_cls_name = []
for val in gal_classes:
    gal_cls_name.append('class_' + str(val))
    
ex_gal_classes = [15, 42, 52, 62, 64, 67, 88, 90, 95]
ex_gal_cls_name = []
for val in ex_gal_classes:
    ex_gal_cls_name.append('class_' + str(val))


# In[ ]:


final_sub = z.copy()
final_sub.loc[gal_obj,gal_cls_name].describe()


# In[ ]:


final_sub.loc[gal_obj,ex_gal_cls_name] = 0 
final_sub.loc[ex_gal_obj,gal_cls_name] = 0 


# In[ ]:


final_sub.corrwith(z)


# In[ ]:


final_sub.head()


# In[ ]:


final_sub.tail()


# In[ ]:


final_sub.to_csv('nn_sub.csv')

