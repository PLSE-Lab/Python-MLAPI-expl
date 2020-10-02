#!/usr/bin/env python
# coding: utf-8

# **This kernel uses fgcnn model from deepctr package**
# 
# **fgcnn :** [fgcnn using deepctr](https://deepctr-doc.readthedocs.io/en/v0.7.0/deepctr.models.fgcnn.html)
# 
# code forked from https://www.kaggle.com/siavrez/deepfm-model

# In[ ]:


get_ipython().system('pip install --no-warn-conflicts -q deepctr')


# In[ ]:


from deepctr.inputs import  SparseFeat, DenseFeat, get_feature_names
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam,RMSprop
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks
from tensorflow.keras import utils
from deepctr.models import *
from deepctr.models.fgcnn import FGCNN
from deepctr.models.nffm import NFFM
import tensorflow.keras as keras
import tensorflow as tf
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter('ignore')


# In[ ]:


train = pd.read_csv('../input/cat-in-the-dat-ii/train.csv')
test = pd.read_csv('../input/cat-in-the-dat-ii/test.csv')


# In[ ]:


test["target"] = -1


# In[ ]:


data = pd.concat([train, test]).reset_index(drop=True)


# In[ ]:


data['null'] = data.isna().sum(axis=1)


# In[ ]:


sparse_features = [feat for feat in train.columns if feat not in ['id','target']]

data[sparse_features] = data[sparse_features].fillna('-1', )


# In[ ]:


for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat].fillna("-1").astype(str).values)


# In[ ]:


train = data[data.target != -1].reset_index(drop=True)
test  = data[data.target == -1].reset_index(drop=True)


# In[ ]:


fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique()) for feat in sparse_features]

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)


# In[ ]:


def auc(y_true, y_pred):
    def fallback_auc(y_true, y_pred):
        try:
            return roc_auc_score(y_true, y_pred)
        except:
            return 0.5
    return tf.py_function(fallback_auc, (y_true, y_pred), tf.double)


# In[ ]:


class CyclicLR(keras.callbacks.Callback):

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        K.set_value(self.model.optimizer.lr, self.clr())


# In[ ]:


target = ['target']
N_Splits = 20
Epochs = 10
SEED = 2020


# In[ ]:


oof_pred_deepfm = np.zeros((len(train), ))
y_pred_deepfm = np.zeros((len(test),))


skf = StratifiedKFold(n_splits=N_Splits, shuffle=True, random_state=SEED)
for fold, (tr_ind, val_ind) in enumerate(skf.split(train, train[target])):
    X_train, X_val = train[sparse_features].iloc[tr_ind], train[sparse_features].iloc[val_ind]
    y_train, y_val = train[target].iloc[tr_ind], train[target].iloc[val_ind]
    train_model_input = {name:X_train[name] for name in feature_names}
    val_model_input = {name:X_val[name] for name in feature_names}
    test_model_input = {name:test[name] for name in feature_names}
    model = NFFM(linear_feature_columns, dnn_feature_columns)
    model.compile("adam", "binary_crossentropy", metrics=[auc], )
    es = callbacks.EarlyStopping(monitor='val_auc', min_delta=0.0001, patience=2, verbose=1, mode='max', baseline=None, restore_best_weights=True)
    sb = callbacks.ModelCheckpoint('./nn_model.w8', save_weights_only=True, save_best_only=True, verbose=0)
    clr = CyclicLR(base_lr=0.00001 / 100, max_lr = 0.0001, 
                       step_size= int(1.0*(test.shape[0])/1024) , mode='exp_range',
                       gamma=1., scale_fn=None, scale_mode='cycle')
    history = model.fit(train_model_input, y_train,
                        validation_data=(val_model_input, y_val),
                        batch_size=512, epochs=Epochs, verbose=1,
                        callbacks=[es, sb, clr],)
    model.load_weights('./nn_model.w8')
    val_pred = model.predict(val_model_input, batch_size=512)
    print(f"validation AUC fold {fold+1} : {round(roc_auc_score(y_val, val_pred), 5)}")
    oof_pred_deepfm[val_ind] = val_pred.ravel()
    y_pred_deepfm += model.predict(test_model_input, batch_size=512).ravel() / (N_Splits)
    K.clear_session()


# In[ ]:


print(f"OOF AUC : {round(roc_auc_score(train.target.values, oof_pred_deepfm), 5)}")


# In[ ]:


test_idx = test.id.values
submission = pd.DataFrame.from_dict({
    'id': test_idx,
    'target': y_pred_deepfm
})
submission.to_csv("submission.csv", index=False)
print("Submission file saved!")


# In[ ]:


np.save('oof_pred_deepfm.npy',oof_pred_deepfm)
np.save('y_pred_deepfm.npy',    y_pred_deepfm)

