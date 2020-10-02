#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.utils import class_weight
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
import pickle
from keras.layers import Dense, Activation, Dropout, BatchNormalization, Input
from keras.models import Sequential, Model
from keras import optimizers, regularizers, initializers
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K
from keras.optimizers import Adam
import tensorflow as tf
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


NCA1 = 100
NCA2 = 50
DROPRATE = 0.2
EP = 500
BATCH_SIZE = 256
VAL_RATIO = 0.1
TEST_RATIO = 0.1


# ## Loading BBBP dataset
# 
# **BBBP**: Binary labels of blood-brain barrier penetration(permeability).
# 

# In[ ]:


bbbp_df= pd.read_csv('../input/bbbp_descriptors/BBBP_df_revised.csv')
print(bbbp_df.shape)
bbbp_df.head()


# The simplified molecular-input line-entry system (**SMILES**) is a specification in form of a line notation for describing the structure of chemical species using short ASCII strings. SMILES can be converted to molecular structure by using RDKIT module.
# 
# Example: 
# ```python
# from rdkit import Chem
# m = Chem.MolFromSmiles('Cc1ccccc1')
# ```
# 
# Further reading:
# * https://www.rdkit.org/docs/GettingStartedInPython.html

# In[ ]:


bbbp_df['p_np'].value_counts()


# ## Loading molecular descriptors
# 
# Descriptors dataframe contains 1625 molecular descriptors (including 3D descriptors) generated on the NCI database using Mordred python module.
# 
# Further Reading:
# * https://en.wikipedia.org/wiki/Molecular_descriptor
# * https://github.com/mordred-descriptor/mordred

# In[ ]:


bbbp_descriptors_df= pd.read_csv('../input/bbbp_descriptors/BBBP_descriptors_df.csv',low_memory=False)
print(bbbp_descriptors_df.shape)
bbbp_descriptors_df.head()


# In[ ]:


# function to coerce all data types to numeric

def coerce_to_numeric(df, column_list):
    df[column_list] = df[column_list].apply(pd.to_numeric, errors='coerce')


# In[ ]:


coerce_to_numeric(bbbp_descriptors_df, bbbp_descriptors_df.columns)
bbbp_descriptors_df.head()


# In[ ]:


bbbp_descriptors_df = bbbp_descriptors_df.fillna(0)
bbbp_descriptors_df.head()


# ## Scaling and Principal component analysis (PCA) 

# In[ ]:


bbbp_scaler1 = StandardScaler()
bbbp_scaler1.fit(bbbp_descriptors_df.values)
bbbp_descriptors_df = pd.DataFrame(bbbp_scaler1.transform(bbbp_descriptors_df.values),
                                   columns=bbbp_descriptors_df.columns)


# In[ ]:


nca = NCA1
cn = ['col'+str(x) for x in range(nca)]


# In[ ]:


bbbp_transformer1 = KernelPCA(n_components=nca, kernel='rbf', n_jobs=-1)
bbbp_transformer1.fit(bbbp_descriptors_df.values)
bbbp_descriptors_df = pd.DataFrame(bbbp_transformer1.transform(bbbp_descriptors_df.values),
                                   columns=cn)
print(bbbp_descriptors_df.shape)
bbbp_descriptors_df.head()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(bbbp_descriptors_df.values, bbbp_df['p_np'].values.flatten(), 
                                                    test_size=TEST_RATIO, 
                                                    random_state=42,stratify=bbbp_df['p_np'].values.flatten())


# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, 
                                                      test_size=VAL_RATIO, 
                                                      random_state=42,stratify=y_train)


# In[ ]:


def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 


# In[ ]:


def Find_Optimal_threshold(target, predicted):
    target = target.reshape(-1,1)
    predicted = predicted.reshape(-1,1)
    
    rng = np.arange(0.0, 0.99, 0.001)
    f1s = np.zeros((rng.shape[0],predicted.shape[1]))
    for i in range(0,predicted.shape[1]):
        for j,t in enumerate(rng):
            p = np.array((predicted[:,i])>t, dtype=np.int8)
            scoref1 = f1_score(target[:,i], p, average='binary')
            f1s[j,i] = scoref1
            
    threshold = np.empty(predicted.shape[1])
    for i in range(predicted.shape[1]):
        threshold[i] = rng[int(np.where(f1s[:,i] == np.max(f1s[:,i]))[0][0])]
        
    return threshold


# ## Sklearn SVC Model

# In[ ]:


parameters = {'kernel':['sigmoid', 'rbf'], 'C':[1,0.5], 'gamma':[1/nca,1/np.sqrt(nca)],'probability':[True]}
bbbp_svc = GridSearchCV(SVC(random_state=23,class_weight='balanced'), parameters, cv=5, scoring='roc_auc',n_jobs=-1)


# In[ ]:


result = bbbp_svc.fit(X_train, y_train)


# In[ ]:


print(result.best_estimator_)


# In[ ]:


print(result.best_score_)


# In[ ]:


pred = bbbp_svc.predict_proba(X_valid)


# **Probability calibration:**
# 
# Some models can give poor estimates of the class probabilities and some even do not support probability prediction. The *CalibratedClassifierCV*  calibrates the probabilities of a given model, or to add support for probability prediction.
# 
# **Further Reading:**
# * https://scikit-learn.org/stable/modules/calibration.html
# * https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html#sklearn.calibration.CalibratedClassifierCV

# In[ ]:


bbbp_svc_calib = CalibratedClassifierCV(bbbp_svc, cv='prefit')
bbbp_svc_calib.fit(X_valid, y_valid)


# In[ ]:


pred = bbbp_svc_calib.predict_proba(X_valid)
pred = pred[:,1]
pred_svc_t = np.copy(pred)


# In[ ]:


threshold = Find_Optimal_threshold(y_valid, pred)
print(threshold)


# In[ ]:


pred = bbbp_svc_calib.predict(X_test)
f1_score(y_test,pred)


# In[ ]:


pred = bbbp_svc_calib.predict_proba(X_test)
roc_auc_score(y_test,pred[:,1])


# In[ ]:


pred = pred[:,1]
pred_svc = np.copy(pred)
pred[pred<=threshold] = 0
pred[pred>threshold] = 1
svc_score = f1_score(y_test,pred)
print(svc_score)


# In[ ]:


y = np.array(bbbp_descriptors_df.loc[23].values).reshape(1, -1)
result = bbbp_svc.predict(y)
prob = bbbp_svc.predict_proba(y)
print(result)
print(prob)
print(int(prob[:,1]>threshold))


# ## Keras Neural Network Model

# In[ ]:


bbbp_model = Sequential()
bbbp_model.add(Dense(128, input_dim=bbbp_descriptors_df.shape[1], 
                     kernel_initializer='he_uniform'))
bbbp_model.add(BatchNormalization())
bbbp_model.add(Activation('tanh'))
bbbp_model.add(Dropout(rate=DROPRATE))
bbbp_model.add(Dense(64,kernel_initializer='he_uniform'))
bbbp_model.add(BatchNormalization())
bbbp_model.add(Activation('tanh'))
bbbp_model.add(Dropout(rate=DROPRATE))
bbbp_model.add(Dense(32,kernel_initializer='he_uniform'))
bbbp_model.add(BatchNormalization())
bbbp_model.add(Activation('tanh'))
bbbp_model.add(Dropout(rate=DROPRATE))
bbbp_model.add(Dense(1,kernel_initializer='he_uniform',activation='sigmoid'))


# In[ ]:


bbbp_model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])


# In[ ]:


checkpoint = ModelCheckpoint('bbbp_model.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')


# In[ ]:


unique_classes = np.unique(bbbp_df['p_np'].values.flatten())
class_weights = class_weight.compute_class_weight('balanced',unique_classes,
                                                  bbbp_df['p_np'].values.flatten())
class_weights = {unique_classes[0]:class_weights[0],unique_classes[1]:class_weights[1]}


# In[ ]:


hist = bbbp_model.fit(X_train, y_train, 
                      validation_data=(X_valid,y_valid),epochs=EP, batch_size=BATCH_SIZE, 
                      class_weight=class_weights ,callbacks=[checkpoint])


# In[ ]:


plt.ylim(0., 1.0)
plt.plot(hist.epoch, hist.history["loss"], label="Train loss")
plt.plot(hist.epoch, hist.history["val_loss"], label="Valid loss")


# In[ ]:


bbbp_model.load_weights('bbbp_model.h5')


# In[ ]:


pred = bbbp_model.predict(X_valid)
pred_nn_t = np.copy(pred)


# In[ ]:


threshold = Find_Optimal_threshold(y_valid, pred)
print(threshold)


# In[ ]:


pred = bbbp_model.predict(X_test)
pred_nn = np.copy(pred)
roc_auc_score(y_test,pred)


# In[ ]:


pred[pred<=threshold] = 0
pred[pred>threshold] = 1
nn_score = f1_score(y_test,pred)
print(nn_score)


# In[ ]:


prob = bbbp_model.predict(y)
print(prob)
print(int(prob>=threshold))


# ## Gradient Boosting of Keras Model with SVC

# In[ ]:


inp = bbbp_model.input
out = bbbp_model.layers[-2].output
bbbp_model_gb = Model(inp, out)


# In[ ]:


X_train = bbbp_model_gb.predict(X_train)
X_valid = bbbp_model_gb.predict(X_valid)
X_test = bbbp_model_gb.predict(X_test)


# In[ ]:


data = np.concatenate((X_train,X_test,X_valid),axis=0)


# In[ ]:


bbbp_scaler2 = StandardScaler()
bbbp_scaler2.fit(data)
X_train = bbbp_scaler2.transform(X_train)
X_valid = bbbp_scaler2.transform(X_valid)
X_test = bbbp_scaler2.transform(X_test)


# In[ ]:


data = np.concatenate((X_train,X_test,X_valid),axis=0)


# In[ ]:


nca = NCA2


# In[ ]:


bbbp_transformer2 = KernelPCA(n_components=nca, kernel='rbf', n_jobs=-1)
bbbp_transformer2.fit(data)
X_train = bbbp_transformer2.transform(X_train)
X_valid = bbbp_transformer2.transform(X_valid)
X_test = bbbp_transformer2.transform(X_test)


# In[ ]:


nca = X_train.shape[1]
parameters = {'kernel':['sigmoid', 'rbf'], 'C':[1,0.5], 'gamma':[1/nca,1/np.sqrt(nca)],'probability':[True]}
bbbp_svc_gb = GridSearchCV(SVC(random_state=23,class_weight='balanced'), parameters, cv=5, scoring='roc_auc',n_jobs=-1)


# In[ ]:


result = bbbp_svc_gb.fit(X_train, y_train)


# In[ ]:


print(result.best_estimator_)


# In[ ]:


print(result.best_score_)


# In[ ]:


pred = bbbp_svc_gb.predict_proba(X_valid)


# In[ ]:


bbbp_svc_gb_calib = CalibratedClassifierCV(bbbp_svc_gb, cv='prefit')
bbbp_svc_gb_calib.fit(X_valid, y_valid)


# In[ ]:


pred = bbbp_svc_gb_calib.predict_proba(X_valid)
pred = pred[:,1]
pred_svc_gb_t = np.copy(pred)


# In[ ]:


threshold = Find_Optimal_threshold(y_valid, pred)
print(threshold)


# In[ ]:


pred = bbbp_svc_gb_calib.predict(X_test)
f1_score(y_test,pred)


# In[ ]:


pred = bbbp_svc_gb_calib.predict_proba(X_test)
roc_auc_score(y_test,pred[:,1])


# In[ ]:


pred = pred[:,1]
pred_svc_gb = np.copy(pred)
pred[pred<=threshold] = 0
pred[pred>threshold] = 1
svc_gb_score = f1_score(y_test,pred)
print(svc_gb_score)


# In[ ]:


y = np.array(X_train[23,:])
y = y.reshape(-1, nca)
result = bbbp_svc_gb_calib.predict(y)
prob = bbbp_svc_gb_calib.predict_proba(y)
print(result)
print(prob)
print(int(prob[:,1]>=threshold))


# ## Gradient Boosting of Keras Model with XGBoost

# In[ ]:


parameters = {'learning_rate':[0.05,0.1,0.15],'n_estimators':[75,100,125], 'max_depth':[3,4,5],
               'booster':['gbtree','dart'],'reg_alpha':[0.,0.1,0.05],'reg_lambda':[0.,0.1,0.5,1.]}

bbbp_xgb_gb = GridSearchCV(XGBClassifier(random_state=32), parameters, cv=5, scoring='roc_auc',n_jobs=-1)


# In[ ]:


result = bbbp_xgb_gb.fit(X_train, y_train)


# In[ ]:


print(result.best_estimator_)


# In[ ]:


print(result.best_score_)


# In[ ]:


pred = bbbp_xgb_gb.predict_proba(X_valid)


# In[ ]:


bbbp_xgb_gb_calib = CalibratedClassifierCV(bbbp_xgb_gb, cv='prefit')
bbbp_xgb_gb_calib.fit(X_valid, y_valid)


# In[ ]:


pred = bbbp_xgb_gb.predict_proba(X_valid)
pred = pred[:,1]
pred_xgb_gb_t= np.copy(pred)


# In[ ]:


threshold = Find_Optimal_threshold(y_valid, pred)
print(threshold)


# In[ ]:


pred = bbbp_xgb_gb_calib.predict(X_test)
f1_score(y_test,pred)


# In[ ]:


pred = bbbp_xgb_gb_calib.predict_proba(X_test)
roc_auc_score(y_test,pred[:,1])


# In[ ]:


pred = pred[:,1]
pred_xgb_gb = np.copy(pred)
pred[pred<=threshold] = 0
pred[pred>threshold] = 1
xgb_gb_score = f1_score(y_test,pred)
print(xgb_gb_score)


# In[ ]:


result = bbbp_xgb_gb_calib.predict(y)
prob = bbbp_xgb_gb_calib.predict_proba(y)
print(result)
print(prob)
print(int(prob[:,1]>=threshold))


# In[ ]:


pred = (pred_svc_t+pred_nn_t.flatten()+pred_svc_gb_t+pred_xgb_gb_t)/4.


# In[ ]:


threshold = Find_Optimal_threshold(y_valid, pred)
print(threshold)


# In[ ]:


pred = (pred_svc+pred_nn.flatten()+pred_svc_gb+pred_xgb_gb)/4.
pred[pred<=threshold] = 0
pred[pred>threshold] = 1
ave_score = f1_score(y_test,pred)


# ## Saving models, transformer and scaler

# In[ ]:


with open('bbbp_transformer1.pkl', 'wb') as fid:
    pickle.dump(bbbp_transformer1, fid)
with open('bbbp_transformer2.pkl', 'wb') as fid:
    pickle.dump(bbbp_transformer2, fid)
with open('bbbp_scaler1.pkl', 'wb') as fid:
    pickle.dump(bbbp_scaler1, fid)
with open('bbbp_scaler2.pkl', 'wb') as fid:
    pickle.dump(bbbp_scaler2, fid)
with open('bbbp_svc_calib.pkl', 'wb') as fid:
    pickle.dump(bbbp_svc_calib, fid)
with open('bbbp_svc.pkl', 'wb') as fid:
    pickle.dump(bbbp_svc, fid)
with open('bbbp_svc_gb_calib.pkl', 'wb') as fid:
    pickle.dump(bbbp_svc_gb_calib, fid)
with open('bbbp_svc_gb.pkl', 'wb') as fid:
    pickle.dump(bbbp_svc_gb, fid)
with open('bbbp_xgb_gb_calib.pkl', 'wb') as fid:
    pickle.dump(bbbp_xgb_gb_calib, fid)
with open('bbbp_xgb_gb.pkl', 'wb') as fid:
    pickle.dump(bbbp_xgb_gb, fid)


# ## For loading saved model
# 
# ```python
# with open('bbbp_svc.pkl', 'rb') as fid:
#     bbbp_svc = pickle.load(fid)
#  ```

# # Comparision of Results with MoleculeNet results
# 
# http://moleculenet.ai/full-results
# 
# The best ROC AUC score on the test data for the MoleculeNet models is ~70, while the best score obtained in this kernel is ~90 on the test data. 
# 
# **Further Reading:**
# * https://arxiv.org/pdf/1703.00564.pdf

# ## F1 Score Result

# In[ ]:


sns.set(style="whitegrid")
ax = sns.barplot(x=[svc_score,nn_score,svc_gb_score,xgb_gb_score,ave_score],
                 y=['SVC','NN','SVC_GB','XGB_GB','ave'])
ax.set(xlim=(0.75, None))

