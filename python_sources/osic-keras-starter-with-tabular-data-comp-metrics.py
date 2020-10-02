#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import random
import pandas as pd
import pydicom
import os
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





# In[ ]:


ROOT = "../input/osic-pulmonary-fibrosis-progression"
#DESIRED_SIZE = 256 # Memory issue
DESIRED_SIZE = 128


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


# ### Quick Image processing

# In[ ]:


def get_images(df, how="train"):
    xo = []
    p = []
    w  = []
    for i in tqdm(range(df.shape[0])):
        patient = df.iloc[i,0]
        week = df.iloc[i,1]
        try:
            img_path = f"{ROOT}/{how}/{patient}/{week}.dcm"
            ds = pydicom.dcmread(img_path)
            im = Image.fromarray(ds.pixel_array)
            im = im.resize((DESIRED_SIZE,DESIRED_SIZE)) 
            im = np.array(im)
            xo.append(im[np.newaxis,:,:])
            p.append(patient)
            w.append(week)
        except:
            pass
    data = pd.DataFrame({"Patient":p,"Weeks":w})
    return np.concatenate(xo, axis=0), data


# In[ ]:





# In[ ]:





# ### BASELINE CNN 

# In[ ]:


from sklearn.linear_model import Ridge, ElasticNet


# In[ ]:


def metric( trueFVC, predFVC, predSTD ):
    
    clipSTD = np.clip( predSTD, 70 , 9e9 )  
    
    deltaFVC = np.clip( np.abs(trueFVC-predFVC), 0 , 1000 )  

    return np.mean( -1*(np.sqrt(2)*deltaFVC/clipSTD) - np.log( np.sqrt(2)*clipSTD ) )
#


# In[ ]:


C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")
#=============================#
def kloss(y_true, y_pred):
    tf.dtypes.cast(y_true, tf.float32)
    tf.dtypes.cast(y_pred, tf.float32)
    sigma = y_pred[:, 1]
    fvc_pred = y_pred[:, 0]
    
    sigma_clip = sigma + C1
    #sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    #delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )
    metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)
    return K.mean(metric)
#=============================#
def kmae(y_true, y_pred):
    tf.dtypes.cast(y_true, tf.float32)
    tf.dtypes.cast(y_pred, tf.float32)
    spread = tf.abs( (y_true[:, 0] -  y_pred[:, 0])  / (y_pred[:, 0] + 1.) )
    #spread = tf.abs( (y_true[:, 0] -  y_pred[:, 0])  / y_true[:, 0] )
    return K.mean(spread)
#=============================#

def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * kloss(y_true, y_pred) + (1 - _lambda)*kmae(y_true, y_pred)
    return loss
#=================
def make_model():
    z = L.Input((9,), name="Patient")
    x = L.Dense(100, activation="relu", name="d1")(z)
    x = L.Dense(100, activation="relu", name="d2")(x)
    #x = L.Dense(100, activation="relu", name="d3")(x)
    preds = L.Dense(2, activation="relu", name="preds")(x)
    
    model = M.Model(z, preds, name="CNN")
    model.compile(loss=mloss(0.99), 
                  optimizer=tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, 
                                                     epsilon=None, decay=0.01, amsgrad=False), 
                  metrics=[kloss]) #, kmae
    #model.compile(loss=kmae, optimizer="adam", metrics=[kloss])
    #model.compile(loss=kloss, optimizer="adam", metrics=[kmae])#
    return model


# In[ ]:





# In[ ]:


#net = make_model()
#print(net.summary())


# In[ ]:


tr.head()


# In[ ]:


y = tr['FVC'].values
z = tr[FE].values
ze = sub[FE].values


# In[ ]:


NFOLD = 5
kf = KFold(n_splits=NFOLD)


# In[ ]:


pe0 = np.zeros((ze.shape[0], 2))
pred0 = np.zeros((z.shape[0], 2))


pe1 = np.zeros((ze.shape[0], 2))
pred1 = np.zeros((z.shape[0], 2))

cnt = 0
for tr_idx, val_idx in kf.split(z):
    cnt += 1
    print(f"FOLD {cnt}")
    print("=====================  NEURAL NET =============================")
    net = make_model()
    net.fit(z[tr_idx], y[tr_idx], batch_size=200, epochs=500, 
            validation_data=(z[val_idx], y[val_idx]), verbose=0) #
    print("train", net.evaluate(z[tr_idx], y[tr_idx], verbose=0, batch_size=500))
    print("val", net.evaluate(z[val_idx], y[val_idx], verbose=0, batch_size=500))
    print("predict val...")
    pred0[val_idx] = net.predict(z[val_idx], batch_size=500, verbose=0)
    print("predict test...")
    pe0 += net.predict(ze, batch_size=500, verbose=0) / NFOLD
    print("=====================  RIDGE REG =============================")
    clf = Ridge(alpha=0.05)
    clf.fit(z[tr_idx], y[tr_idx]) #
    #print("predict val...")
    pred1[val_idx, 0] = clf.predict(z[val_idx])
    pred_std = np.mean(np.abs(y[val_idx] - pred1[val_idx, 0])) * np.sqrt(2)
    pred1[val_idx, 1] = pred_std
    print("val", metric(y[val_idx], pred1[val_idx, 0], pred1[val_idx, 1]))
    #print("predict test...")
    pe1[:, 0] += clf.predict(ze) / NFOLD
    pe1[:, 1] += pred_std / NFOLD    
#==============
pred0[:, 1] = pred0[:, 1] + 70.


# In[ ]:


w = 0.5
pred = (1-w) * pred0 + w * pred1
pe = (1-w) * pe0 + w * pe1
pe[:, 1] = pe1[:, 1]
pred[:, 1] = pred1[:, 1]


# In[ ]:


print("oof neural net", metric(y, pred0[:, 0], pred0[:, 1]))
print("oof ridge", metric(y, pred1[:, 0], pred1[:, 1]))
print("oof ensemble", metric(y, pred[:, 0], pred[:, 1]))


# In[ ]:





# In[ ]:


sigma_opt = mean_absolute_error(y, pred[:, 0])
sigma_mean = np.mean(pred[:, 1])
print(sigma_opt, sigma_mean)


# In[ ]:


plt.plot(y)
plt.plot(pred[:, 0])
#plt.plot(pred[:, 1])
plt.show()


# In[ ]:





# In[ ]:


pred[:, 1].min(), pred[:, 1].max()


# In[ ]:


plt.hist(pred[:, 1])
plt.title("uncertainty in prediction")
plt.show()


# ### PREDICTION

# In[ ]:


sub.head()


# In[ ]:





# In[ ]:


sub['FVC1'] = pe[:, 0]
sub['Confidence1'] = pe[:, 1]


# In[ ]:


subm = sub[['Patient_Week','FVC','Confidence','FVC1','Confidence1']].copy()


# In[ ]:





# In[ ]:


subm.loc[~subm.FVC1.isnull()].head(10)


# In[ ]:


subm.loc[~subm.FVC1.isnull(),'FVC'] = subm.loc[~subm.FVC1.isnull(),'FVC1']
if sigma_mean<70:
    subm['Confidence'] = sigma_opt
else:
    subm.loc[~subm.FVC1.isnull(),'Confidence'] = subm.loc[~subm.FVC1.isnull(),'Confidence1']


# In[ ]:


otest = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
for i in range(len(otest)):
    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'FVC'] = otest.FVC[i]
    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'Confidence'] = 0.1
#


# In[ ]:


subm.head()


# In[ ]:


subm.describe().T


# In[ ]:





# In[ ]:


subm[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv", index=False)


# In[ ]:





# In[ ]:




