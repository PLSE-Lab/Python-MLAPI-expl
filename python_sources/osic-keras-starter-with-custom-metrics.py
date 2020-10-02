#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import pydicom
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import mean_absolute_error


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


tr = data.loc[data.WHERE=='train']
chunk = data.loc[data.WHERE=='val']
sub = data.loc[data.WHERE=='test']
del data


# In[ ]:


tr.shape, chunk.shape, sub.shape


# In[ ]:





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


x, df_tr = get_images(tr, how="train")


# In[ ]:


x.shape, df_tr.shape


# In[ ]:


idx = np.random.randint(x.shape[0])
plt.imshow(x[idx], cmap=plt.cm.bone)
plt.show()


# In[ ]:


df_tr = df_tr.merge(tr, how="left", on=['Patient', 'Weeks'])


# In[ ]:


y = df_tr['FVC'].values
z = df_tr[FE].values


# In[ ]:


z.shape


# ### BASELINE CNN 

# In[ ]:


import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M


# In[ ]:


C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")
#=============================#
def kloss(y_true, y_pred):
    tf.dtypes.cast(y_true, tf.float32)
    tf.dtypes.cast(y_pred, tf.float32)
    sigma = y_pred[:, 1]
    fvc_pred = y_pred[:, 0]
    
    #sigma_clip = sigma + C1
    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )
    metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)
    return K.mean(metric)
#=============================#
def kmae(y_true, y_pred):
    tf.dtypes.cast(y_true, tf.float32)
    tf.dtypes.cast(y_pred, tf.float32)
    spread = tf.abs( (y_true[:, 0] -  y_pred[:, 0])  / y_true[:, 0] )
    #spred = tf.square(y_true, y_pred[:, 0])
    return K.mean(spread)
#=============================#

def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * kloss(y_true, y_pred) + (1 - _lambda)*kmae(y_true, y_pred)
    return loss
#=================
def make_model():
    inp = L.Input((DESIRED_SIZE,DESIRED_SIZE), name="input")
    z = L.Input((9,), name="Patient")
    x = L.Conv1D(50, 4, activation="relu", name="conv1")(inp)
    x = L.MaxPool1D(2, name='pool1')(x)
    
    #x = L.Dropout(0.2)(x)
    x = L.Conv1D(50, 4, activation="relu", name="conv2")(x)
    x = L.MaxPool1D(2, name='pool2')(x)
    
    #x = L.Dropout(0.2)(x)
    x = L.Conv1D(50, 4, activation="relu", name="conv3")(x)
    x = L.MaxPool1D(2, name='pool3')(x)
    
    x = L.Flatten(name="features")(x)
    x = L.Dense(50, activation="relu", name="d1")(x)
    l = L.Dense(10, activation="relu", name="d2")(z)
    x = L.Concatenate(name="combine")([x, l])
    x = L.Dense(50, activation="relu", name="d3")(x)
    preds = L.Dense(2, activation="relu", name="preds")(x)
    
    model = M.Model([inp, z], preds, name="CNN")
    model.compile(loss=mloss(0.5), optimizer="adam", metrics=[kloss])
    #model.compile(loss=kmae, optimizer="adam", metrics=[kloss])
    #model.compile(loss=kloss, optimizer="adam", metrics=[kmae])#
    return model


# In[ ]:


net = make_model()
print(net.summary())


# In[ ]:


x_min = np.min(x)
x_max = np.max(x)
xs = x - x_min / (x_max - x_min)


# In[ ]:


xs.shape, y.shape, x_min


# In[ ]:


net.fit([xs, z], y, batch_size=50, epochs=100) #, validation_split=0.1


# In[ ]:


pred = net.predict([xs, z], batch_size=100, verbose=1)


# In[ ]:


sigma_opt = mean_absolute_error(y, pred[:, 0])
sigma_mean = np.mean(pred[:, 1])
print(sigma_opt, sigma_mean)


# In[ ]:





# In[ ]:


plt.plot(y)
plt.plot(pred[:, 0])
#plt.plot(pred[:, 1])


# In[ ]:


pred[:, 1].min(), pred[:, 1].max()


# In[ ]:


plt.hist(pred[:, 1])
plt.title("uncertainty in prediction")
plt.show()


# ### PREDICTION

# In[ ]:


xe, df_te = get_images(sub, how="test")
df_te = df_te.merge(sub, how="left", on=['Patient', 'Weeks'])


# In[ ]:


x_te = xe - x_min / (x_max - x_min)
ze = df_te[FE].values
pe = net.predict([x_te, ze], batch_size=100, verbose=1)


# In[ ]:


df_te['FVC1'] = pe[:, 0]
df_te['Confidence1'] = pe[:, 1]


# In[ ]:


sub = sub.merge(df_te[['Patient','Weeks','FVC1','Confidence1']], how='left', 
                on=['Patient', 'Weeks'])
#====================================================#


# In[ ]:


sub.head()


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
    subm.loc[subm.FVC1.isnull(),'Confidence'] = sigma_opt
#


# In[ ]:


subm.head()


# In[ ]:


subm.describe().T


# In[ ]:


subm[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv", index=False)


# In[ ]:




