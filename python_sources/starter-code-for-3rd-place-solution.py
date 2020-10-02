#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


testing = True #Set this to true for submission/False for cross validation
X_train = pd.read_csv('../input/X_train.csv')
y_train = pd.read_csv('../input/y_train.csv')
X_train = pd.merge(X_train,y_train,on='series_id')
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_train['surface'] = le.fit_transform(X_train['surface'])
if(testing):
    X_test = pd.read_csv('../input/X_test.csv')
    X_test['series_id'] = X_test['series_id']+3810
    X_test['group_id'] = 0
    X_test['surface'] = 0
    frames = [X_train,X_test]
    X_train = pd.concat(frames)
    X_train.reset_index(drop=True,inplace=True)


# In[3]:


cols = list(X_train.columns.values)
cols.remove('orientation_W')
cols.insert(3,'orientation_W')
X_train = X_train[cols]


# In[4]:


num_meas = 128
num_series = X_train['series_id'].nunique()


# In[5]:


def q_to_angle(q_val):
    #We assume q_val is in this format: [qw, q1, q2, q3]
    #And the quaternion is normalized
    roll = np.arctan2(2*(q_val[0]*q_val[1] + q_val[2]*q_val[3]),1 - 2*(q_val[1]*q_val[1] + q_val[2]*q_val[2]))
    pitch = np.arcsin(2*(q_val[0]*q_val[2] - q_val[3]*q_val[1]))
    yaw = np.arctan2(2*(q_val[0]*q_val[3] + q_val[1]*q_val[2]),1 - 2*(q_val[2]*q_val[2] + q_val[3]*q_val[3]))
    return np.array([roll, pitch, yaw])


# In[6]:


quat_arr = np.array(X_train[['orientation_W','orientation_X','orientation_Y','orientation_Z']])
euler_arr = np.zeros([quat_arr.shape[0],3])
for n,arr in enumerate(quat_arr):
    euler_arr[n] = q_to_angle(arr)


# In[7]:


X_train['roll'] = euler_arr[:,0]
X_train['pitch'] = euler_arr[:,1]
X_train['yaw'] = euler_arr[:,2]


# In[8]:


cols = list(X_train.columns.values)
cols.remove('group_id')
cols.append('group_id')
cols.remove('surface')
cols.append('surface')
X_train = X_train[cols]


# In[9]:


feat_cols = ['roll','pitch','yaw','angular_velocity_X','angular_velocity_Y','angular_velocity_Z','linear_acceleration_X','linear_acceleration_Y','linear_acceleration_Z']
feat_array = np.array(X_train[feat_cols])
feat_array = np.reshape(feat_array,[num_series,128,len(feat_cols)])
group_array = np.array(X_train['group_id'])
group_array = np.reshape(group_array,[num_series,128])
group_array = group_array[:,0]
target_array = np.array(X_train['surface'])
target_array = np.reshape(target_array,[num_series,128])
target_array = target_array[:,0]


# In[10]:


#Use the first order difference of the following features
#Absolute Orientation features dont make sense to predict surface
delta_cols = ['roll','pitch','yaw']
for dc in delta_cols:
    iia = feat_cols.index(dc)
    np_arr = feat_array[:,:,iia]
    roll_arr = np.copy(np_arr)
    roll_arr[:,1:] = roll_arr[:,:-1]
    np_arr = np_arr - roll_arr
    feat_array[:,:,iia] = np_arr


# In[11]:


#Normalize each 128-pt sample to ensure there is no group related information left in the samples
norm_cols = ['linear_acceleration_X','linear_acceleration_Y','linear_acceleration_Z','angular_velocity_X','angular_velocity_Y','angular_velocity_Z']
for norm in norm_cols:
    iia = feat_cols.index(norm)
    np_arr = feat_array[:,:,iia]
    mean_arr = np.mean(np_arr,1)
    mean_arr = np.expand_dims(mean_arr,1)
    mean_arr = np.repeat(mean_arr,num_meas,1)
    np_arr = np_arr - mean_arr
    feat_array[:,:,iia] = np_arr


# In[12]:


def absfft(x):
    return np.abs(np.fft.rfft(x))

feat_fft_array = np.copy(feat_array[:,:,3:])
feat_fft_array = np.apply_along_axis(absfft,1,feat_fft_array)


# In[13]:


#Further normalization across the entire dataset to ensure NN inputs are zero-mean and unit standard deviation

num_sensor = feat_array.shape[2]
for i in range(num_sensor):
    mean_s = np.mean(feat_array[:,:,i])
    sd_s = np.std(feat_array[:,:,i])
    feat_array[:,:,i] = (feat_array[:,:,i]-mean_s)/sd_s

num_sensor_fft = feat_fft_array.shape[2]
for i in range(num_sensor_fft):
    mean_s = np.mean(feat_fft_array[:,:,i])
    sd_s = np.std(feat_fft_array[:,:,i])
    feat_fft_array[:,:,i] = (feat_fft_array[:,:,i]-mean_s)/sd_s


# In[14]:


from keras.layers import Input,Dense, Dropout, BatchNormalization, SeparableConv1D, Reshape, LSTM, DepthwiseConv2D,AveragePooling2D, CuDNNLSTM, Concatenate
from keras.models import Model
from keras.backend import squeeze
from keras.regularizers import l2
kr = None
num_groups = np.unique(group_array).shape[0]
num_surfaces = np.unique(target_array).shape[0]

def get_net_with_fft_mag_only(dp):
    inputs_t = Input(shape=(128,len(feat_cols)))
    x = SeparableConv1D(32,8,2,'same',depth_multiplier=1,activation='relu',kernel_regularizer=kr)(inputs_t)
    x = Dropout(dp)(x)
    x = SeparableConv1D(64,8,4,'same',depth_multiplier=1,activation='relu',kernel_regularizer=kr)(x)
    x = Dropout(dp)(x)
    x = SeparableConv1D(128,8,4,'same',depth_multiplier=1,activation='relu',kernel_regularizer=kr)(x)
    x = Dropout(dp)(x)
    x = SeparableConv1D(256,8,4,'same',depth_multiplier=1,activation='relu',kernel_regularizer=kr)(x)
    x = Reshape((256,))(x)
    x = Dropout(dp)(x)
    x = Dense(64, activation='relu',kernel_regularizer=kr)(x)
    x = Dropout(dp)(x)
    x = Dense(64, activation='relu')(x)
    
    inputs_f = Input(shape=(feat_fft_array.shape[1],feat_fft_array.shape[2]))
    y = SeparableConv1D(32,8,2,'same',depth_multiplier=1,activation='relu',kernel_regularizer=kr)(inputs_f)
    y = Dropout(dp)(y)
    y = SeparableConv1D(64,8,2,'same',depth_multiplier=1,activation='relu',kernel_regularizer=kr)(y)
    y = Dropout(dp)(y)
    y = SeparableConv1D(128,8,4,'same',depth_multiplier=1,activation='relu',kernel_regularizer=kr)(y)
    y = Dropout(dp)(y)
    y = SeparableConv1D(128,8,4,'same',depth_multiplier=1,activation='relu',kernel_regularizer=kr)(y)
    y = Dropout(dp)(y)
    y = SeparableConv1D(256,8,2,'same',depth_multiplier=1,activation='relu',kernel_regularizer=kr)(y)
    y = Reshape((256,))(y)
    y = Dropout(dp)(y)
    y = Dense(64, activation='relu',kernel_regularizer=kr)(y)
    y = Dropout(dp)(y)
    y = Dense(64, activation='relu')(y)
    
        
    inputs = [inputs_t,inputs_f]
    
    z = Concatenate()([x,y])
    z = Dense(64, activation='relu')(z)
    predictions = Dense(num_surfaces, activation='softmax')(z)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# In[15]:


from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import accuracy_score
depthwise = False
fft_net = True
if(not(testing)):
    gkf = GroupKFold(3)
    train_gen = gkf.split(X=feat_array,groups=group_array)
    preds = np.zeros_like(target_array)
    for train_idx,test_idx in train_gen:
        #Test features
        t_feats = feat_array[train_idx]
        t_feats_fft = feat_fft_array[train_idx]
        
        #Validation features
        v_feats = feat_array[test_idx]
        v_feats_fft = feat_fft_array[test_idx]
        
        t_vals = target_array[train_idx]
        v_vals = target_array[test_idx]
        
        pred_classes = np.zeros([v_vals.shape[0],num_surfaces,5])
        for k in range(5): #5 time averaging to get more stable results
            nnet = get_net_with_fft_mag_only(0.5)
            nnet.fit(x=[t_feats,t_feats_fft],y=t_vals,batch_size=256,epochs=3000,validation_data=([v_feats,v_feats_fft],v_vals),verbose=2)
            pred_classes[:,:,k] = nnet.predict([v_feats,v_feats_fft])
        pred_classes = np.mean(pred_classes,axis=2)
        pred_classes = np.argmax(pred_classes,axis=1)
        preds[test_idx] = pred_classes
        print('Val accuracy: ',accuracy_score(v_vals,pred_classes))
        pred_classes = nnet.predict([t_feats,t_feats_fft])
        pred_classes = np.argmax(pred_classes,axis=1)
        print('Train accuracy: ',accuracy_score(t_vals,pred_classes))
    print('5 Fold accuracy: ', accuracy_score(target_array,preds))
else:
    t_feats = feat_array[:3810]
    t_feats_fft = feat_fft_array[:3810]
    t_vals = target_array[:3810]
    v_feats = feat_array[3810:]
    v_feats_fft = feat_fft_array[3810:]
    pred_classes = np.zeros([v_feats.shape[0],num_surfaces,3])
    for k in range(3):
        nnet = get_net_with_fft_mag_only(0.5)
        nnet.fit(x=[t_feats,t_feats_fft],y=t_vals,batch_size=256,epochs=3000,verbose=0)
        pred_classes[:,:,k] = nnet.predict([v_feats,v_feats_fft])
    pred_classes = np.mean(pred_classes,axis=2)
    pred_classes = list(np.argmax(pred_classes,axis=1))
    pred_classes = [le.inverse_transform([i])[0] for i in pred_classes]
    sub_df = pd.read_csv('../input/sample_submission.csv')
    sub_df['surface'] = pred_classes
    sub_df.to_csv('submission.csv',index=False)

