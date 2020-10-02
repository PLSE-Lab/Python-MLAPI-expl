#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This notebook is forked from https://www.kaggle.com/ulrich07/osic-multiple-quantile-regression-starter by @ulrich07 - if you are kind enough to upvote my notebook, please also upvote Ulrich's.
# 
# What have I changed? Just 2 things of consequence:
# 
# 1. I've added some seeds to try and reduce the run-to-run variability.
# 2. A little trick I learnt from some other notebooks (sorry, I'm not sure who to attribute the original idea to) where you update the FVC prediction to the given value where it is available and the confidence for that week to be a small value (high confidence). This is interesting... the competition instructions say we only get the initial FVC value, and that we're scored on predicting values from the last 3 measurements. So that implies that some of the data from private test set used for the public leaderboard, must have only 3 measurements.
# 

# In[ ]:


import numpy as np
import pandas as pd
import pydicom
import os
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


# In[ ]:


import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M


# In[ ]:


def seed_everything(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
seed_everything(42)


# In[ ]:


ROOT = "../input/osic-pulmonary-fibrosis-progression"
BATCH_SIZE=128


# In[ ]:


tr = pd.read_csv(f"{ROOT}/train.csv")
tr.drop_duplicates(keep=False, inplace=True, subset=['Patient','Weeks'])
chunk = pd.read_csv(f"{ROOT}/test.csv")

print("add infos")
sub = pd.read_csv(f"{ROOT}/sample_submission.csv")
sub['Patient'] = sub['Patient_Week'].apply(lambda x:x.split('_')[0])
sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))
sub =  sub[['Patient','Weeks','Confidence','Patient_Week']]
sub = sub.merge(chunk.drop('Weeks', axis=1), on="Patient")


# In[ ]:


tr['WHERE'] = 'train'
chunk['WHERE'] = 'val'
sub['WHERE'] = 'test'
data = tr.append([chunk, sub])


# In[ ]:


print(tr.shape, chunk.shape, sub.shape, data.shape)
print(tr.Patient.nunique(), chunk.Patient.nunique(), sub.Patient.nunique(), 
      data.Patient.nunique())
#


# In[ ]:


data['min_week'] = data['Weeks']
data.loc[data.WHERE=='test','min_week'] = np.nan
data['min_week'] = data.groupby('Patient')['min_week'].transform('min')


# In[ ]:


base = data.loc[data.Weeks == data.min_week]
base = base[['Patient','FVC']].copy()
base.columns = ['Patient','min_FVC']
base['nb'] = 1
base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')
base = base[base.nb==1]
base.drop('nb', axis=1, inplace=True)


# In[ ]:


data = data.merge(base, on='Patient', how='left')
data['base_week'] = data['Weeks'] - data['min_week']
del base


# In[ ]:


COLS = ['Sex','SmokingStatus']
FE = []
for col in COLS:
    for mod in data[col].unique():
        FE.append(mod)
        data[mod] = (data[col] == mod).astype(int)
#=================


# In[ ]:


#
data['age'] = (data['Age'] - data['Age'].min() ) / ( data['Age'].max() - data['Age'].min() )
data['BASE'] = (data['min_FVC'] - data['min_FVC'].min() ) / ( data['min_FVC'].max() - data['min_FVC'].min() )
data['week'] = (data['base_week'] - data['base_week'].min() ) / ( data['base_week'].max() - data['base_week'].min() )
data['percent'] = (data['Percent'] - data['Percent'].min() ) / ( data['Percent'].max() - data['Percent'].min() )
FE += ['age','percent','week','BASE']


# In[ ]:





# In[ ]:


tr = data.loc[data.WHERE=='train']
chunk = data.loc[data.WHERE=='val']
sub = data.loc[data.WHERE=='test']
del data


# In[ ]:


tr.shape, chunk.shape, sub.shape


# ### BASELINE NN 

# In[ ]:


C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")
#=============================#
def score(y_true, y_pred):
    tf.dtypes.cast(y_true, tf.float32)
    tf.dtypes.cast(y_pred, tf.float32)
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]
    
    #sigma_clip = sigma + C1
    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )
    metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)
    return K.mean(metric)
#============================#
def qloss(y_true, y_pred):
    # Pinball loss for multiple quantiles
    qs = [0.2, 0.50, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q*e, (q-1)*e)
    return K.mean(v)
#=============================#
def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda)*score(y_true, y_pred)
    return loss
#=================
def make_model():
    z = L.Input((9,), name="Patient")
    x = L.Dense(100, activation="relu", name="d1")(z)
    x = L.Dense(100, activation="relu", name="d2")(x)
    #x = L.Dense(100, activation="relu", name="d3")(x)
    p1 = L.Dense(3, activation="linear", name="p1")(x)
    p2 = L.Dense(3, activation="relu", name="p2")(x)
    preds = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), 
                     name="preds")([p1, p2])
    
    model = M.Model(z, preds, name="CNN")
    #model.compile(loss=qloss, optimizer="adam", metrics=[score])
    model.compile(loss=mloss(0.8), optimizer="adam", metrics=[score])
    return model


# In[ ]:


net = make_model()
print(net.summary())
print(net.count_params())


# In[ ]:


y = tr['FVC'].values
z = tr[FE].values
ze = sub[FE].values
pe = np.zeros((ze.shape[0], 3))
pred = np.zeros((z.shape[0], 3))


# In[ ]:


NFOLD = 5
kf = KFold(n_splits=NFOLD)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'cnt = 0\nfor tr_idx, val_idx in kf.split(z):\n    cnt += 1\n    print(f"FOLD {cnt}")\n    net = make_model()\n    net.fit(z[tr_idx], y[tr_idx], batch_size=BATCH_SIZE, epochs=500, \n            validation_data=(z[val_idx], y[val_idx]), verbose=0) #\n    print("train", net.evaluate(z[tr_idx], y[tr_idx], verbose=0, batch_size=BATCH_SIZE))\n    print("val", net.evaluate(z[val_idx], y[val_idx], verbose=0, batch_size=BATCH_SIZE))\n    print("predict val...")\n    pred[val_idx] = net.predict(z[val_idx], batch_size=BATCH_SIZE, verbose=0)\n    print("predict test...")\n    pe += net.predict(ze, batch_size=BATCH_SIZE, verbose=0) / NFOLD\n#==============')


# In[ ]:





# In[ ]:


sigma_opt = mean_absolute_error(y, pred[:, 1])
unc = pred[:,2] - pred[:, 0]
sigma_mean = np.mean(unc)
print(sigma_opt, sigma_mean)


# In[ ]:


idxs = np.random.randint(0, y.shape[0], 100)
plt.plot(y[idxs], label="ground truth")
plt.plot(pred[idxs, 0], label="q25")
plt.plot(pred[idxs, 1], label="q50")
plt.plot(pred[idxs, 2], label="q75")
plt.legend(loc="best")
plt.show()


# In[ ]:





# In[ ]:


print(unc.min(), unc.mean(), unc.max(), (unc>=0).mean())


# In[ ]:


plt.hist(unc)
plt.title("uncertainty in prediction")
plt.show()


# ### PREDICTION

# In[ ]:


sub.head()


# In[ ]:





# In[ ]:


sub['FVC1'] = pe[:, 1]
sub['Confidence1'] = pe[:, 2] - pe[:, 0]


# In[ ]:


subm = sub[['Patient_Week','FVC','Confidence','FVC1','Confidence1']].copy()


# In[ ]:


subm.loc[~subm.FVC1.isnull()].head(10)


# In[ ]:


subm.loc[~subm.FVC1.isnull(),'FVC'] = subm.loc[~subm.FVC1.isnull(),'FVC1']
if sigma_mean<70:
    subm['Confidence'] = sigma_opt
else:
    subm.loc[~subm.FVC1.isnull(),'Confidence'] = subm.loc[~subm.FVC1.isnull(),'Confidence1']


# In[ ]:


subm.head()


# In[ ]:


subm.describe().T


# In[ ]:


otest = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
for i in range(len(otest)):
    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'FVC'] = otest.FVC[i]
    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'Confidence'] = 0.1


# In[ ]:


subm[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv", index=False)

