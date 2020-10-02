#!/usr/bin/env python
# coding: utf-8

# # Few lessons learnt while working on this competetion. Might be helpful for beginners.
# 
# 
# 1) Initially predict hard values 0 or 1 and roc_auc scores ended up in 65 - 70. Came to know that we need to use probabilities for AUC, which increased the score to 86.
# 
# 2) Changing the input data dimension from 1D --> 2D improved score a lot. Got this idea from this [kernal](https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/82863)
# 
# 3) Dimensionality reduction: If we want to visualize 2 variables(2 dimensional) we can do it using any type of plot, 1 variable at x-axis and another in y-axis. But, what if want to visualize 200 variables (200 dimension)? we can use dimensionality reduction techniques like PCA,t-SNE, UMAP for the same. These techniques intelligently summarizes/group information related to multi dimension to the required low dimension. Unfortunately this techniques didn't  help much in this competetion. [PCA](https://www.kaggle.com/sandeep8530/pca-for-santander)
# 
# 4) Cyclelr: Learning rate is one of the important hyperparameters. Varying learning rate helps in fast and effective model. [Reference](https://github.com/bckenstler/CLR) 
# 
# 5) K-fold: Came to know that apart from using it for cross_val_score to know the robustness of the model, it can used to predict values at each fold which improves the score(kind of ensemble).

# # Please upvote, if you find this kernel interesting

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# In[ ]:


import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

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
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("whitegrid")
np.random.seed(697)


# In[ ]:


from keras.callbacks import Callback
from keras import backend as K


class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency.
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.
    # Example for CIFAR-10 w/ batch size 100:
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # References
      - [Cyclical Learning Rates for Training Neural Networks](
      https://arxiv.org/abs/1506.01186)
    """

    def __init__(
            self,
            base_lr=0.001,
            max_lr=0.006,
            step_size=2000.,
            mode='triangular',
            gamma=1.,
            scale_fn=None,
            scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        if mode not in ['triangular', 'triangular2',
                        'exp_range']:
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
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
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


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_s=scaler.fit_transform(X)
X_test_s=scaler.transform(X_test)


# In[ ]:


X=pd.DataFrame(X_s)
X_test=pd.DataFrame(X_test_s)


# In[ ]:


X_test.index=test_df.index


# In[ ]:


seed = 7
np.random.seed(seed)


# In[ ]:


# CROSS VALIDATION
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)


# In[ ]:


adam = optimizers.adam
model = Sequential()
model.add(Dense(64, input_shape=(200,1),
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
    X_train = np.reshape(X.iloc[train].values, (-1, 200, 1))
    y_train_ = y.iloc[train].values
    X_valid = np.reshape(X.iloc[valid].values, (-1, 200, 1))
    y_valid = y.iloc[valid].values
    early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=20)
    model.fit(X_train, y_train_, validation_data=(X_valid, y_valid), epochs=200, verbose=2, batch_size=1024,
              callbacks=[early,clr])
    X_test1 = np.reshape(X_test.values, (200000, 200, 1))
    curr_preds = model.predict(X_test1, batch_size=1024)
    oof_preds[valid] = model.predict(X_valid)
    preds.append(curr_preds)
    c += 1
auc = roc_auc_score(y, oof_preds)
print("CV_AUC: {}".format(auc))


# In[ ]:


preds = np.asarray(preds)
preds = preds.reshape((5, 200000))
preds_final = np.mean(preds.T, axis=1)
submission = pd.read_csv('./../input/sample_submission.csv')
submission['target'] = preds_final
submission.to_csv('submission.csv', index=False)


# # Please upvote, if you find this kernel interesting
