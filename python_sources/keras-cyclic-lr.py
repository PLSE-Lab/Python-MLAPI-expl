#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
print(os.listdir("../input"))


# In[ ]:


import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold, StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense
import keras
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from keras.layers import Activation, Dense, Dropout, Flatten
from keras import optimizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc
from sklearn.metrics import roc_auc_score


# In[ ]:


from keras.callbacks import Callback
from keras import backend as K

class CyclicLR(Callback):
    def __init__(
            self,base_lr=0.001,
            max_lr=0.006,step_size=2000.,
            mode='triangular',gamma=1.,
            scale_fn=None,scale_mode='cycle'):
        
        super(CyclicLR, self).__init__()

        if mode not in ['triangular', 'triangular2','exp_range']:
            raise KeyError("mode must be one of 'triangular', "
                           "'triangular2', or 'exp_range'")
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2.**(x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** x
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
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) *                 np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) *                 np.maximum(0, (1 - x)) * self.scale_fn(self.clr_iterations)

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

        self.history.setdefault(
            'lr', []).append(
            K.get_value(
                self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)


# In[ ]:


train_df = pd.read_csv('../input/train.csv', index_col=0)
test_df = pd.read_csv('../input/test.csv', index_col=0)


# In[ ]:


train_df.shape


# In[ ]:


X = train_df.drop(['target'],axis=1)
y = train_df['target']


# In[ ]:


X_test = test_df


# In[ ]:


test_df.shape


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_s=scaler.fit_transform(X)
X_test_s=scaler.transform(X_test)


# In[ ]:


X=pd.DataFrame(X_s)
X_test=pd.DataFrame(X_test_s)
X_test.index=test_df.index


# In[ ]:


seed = 7
np.random.seed(seed)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)


# In[ ]:


adam = optimizers.adam
model = Sequential()
model.add(Dense(64, input_shape=(256,1),
                kernel_initializer='normal',
                activation="relu"))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1,activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer='adam',metrics=['accuracy'])

clr = CyclicLR(base_lr=0.001, max_lr=0.006,step_size=400., mode='triangular')


# In[ ]:


preds = []
c = 0
oof_preds = np.zeros((len(X), 1))

for train, valid in cv.split(X, y):
    print("VAL %s" % c)
    X_train = np.reshape(X.iloc[train].values, (-1, 256, 1))
    y_train_ = y.iloc[train].values
    X_valid = np.reshape(X.iloc[valid].values, (-1, 256, 1))
    y_valid = y.iloc[valid].values
    early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=20)
    model.fit(X_train, y_train_, validation_data=(X_valid, y_valid), epochs=200, verbose=2, batch_size=1024,
              callbacks=[early,clr])
    X_test1 = np.reshape(X_test.values, (131073, 256, 1))
    curr_preds = model.predict(X_test1, batch_size=1024)
    oof_preds[valid] = model.predict(X_valid)
    preds.append(curr_preds)
    c += 1
auc = roc_auc_score(y, oof_preds)
print("CV_AUC: {}".format(auc))


# In[ ]:


preds = np.asarray(preds)
preds = preds.reshape((5, 131073))
preds_final = np.mean(preds.T, axis=1)
submission = pd.read_csv('./../input/sample_submission.csv')
submission['target'] = preds_final
submission.to_csv('submission.csv', index=False)

