#!/usr/bin/env python
# coding: utf-8

# ## 1. Initialize:

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import math
from scipy.spatial import Voronoi
import numba as nb


# In[ ]:


import lightgbm as lgb


# In[ ]:


import keras
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.utils import plot_model
import keras.backend as K
import tensorflow as tf


# In[ ]:


from keras.layers import Dense,Input,Flatten,concatenate,Dropout,Lambda,BatchNormalization
from keras.models import Model
from keras.losses import binary_crossentropy
from  keras.callbacks import EarlyStopping, ModelCheckpoint
import codecs

from keras.utils import to_categorical
from keras.callbacks import Callback
from sklearn.metrics import f1_score


# In[ ]:


pd.set_option("display.max_columns", 1000)
pd.set_option("display.max_rows", 1000)


# In[ ]:


from kaggle.competitions import nflrush
env = nflrush.make_env()


# ## Functions:

# In[ ]:


# https://www.kaggle.com/anandavati/ngboost-for-nfl/

import numpy.random as np_rnd
import scipy as sp
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import norm as dist

def default_tree_learner(depth=3):
    return DecisionTreeRegressor(
        criterion='friedman_mse',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=depth,
        splitter='best')

class MLE:
    def __init__(self, seed=123):
        pass

    def loss(self, forecast, Y):
        return forecast.nll(Y.squeeze()).mean()

    def grad(self, forecast, Y, natural=True):
        fisher = forecast.fisher_info()
        grad = forecast.D_nll(Y)
        if natural:
            grad = np.linalg.solve(fisher, grad)
        return grad


class CRPS:
    def __init__(self, K=32):
        self.K = K

    def loss(self, forecast, Y):
        return forecast.crps(Y.squeeze()).mean()

    def grad(self, forecast, Y, natural=True):
        metric = forecast.crps_metric()
        grad = forecast.D_crps(Y)
        if natural:
            grad = np.linalg.solve(metric, grad)
        return grad

EPS = 1e-8
class Normal(object):
    n_params = 2

    def __init__(self, params, temp_scale = 1.0):
        self.loc = params[0]
        self.scale = np.exp(params[1] / temp_scale) + 1e-8
        self.var = self.scale #** 2  + 1e-8
        self.shp = self.loc.shape

        self.dist = dist(loc=self.loc, scale=self.scale)

    def __getattr__(self, name):
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    def nll(self, Y):
        return -self.dist.logpdf(Y)

    def D_nll(self, Y_):
        Y = Y_.squeeze()
        D = np.zeros((self.var.shape[0], 2))
        D[:, 0] = (self.loc - Y) / self.var
        D[:, 1] = 1 - ((self.loc - Y) ** 2) / self.var
        return D

    def crps(self, Y):
        Z = (Y - self.loc) / (self.scale + EPS)
        return (self.scale * (Z * (2 * sp.stats.norm.cdf(Z) - 1) +                 2 * sp.stats.norm.pdf(Z) - 1 / np.sqrt(np.pi)))

    def D_crps(self, Y_):
        Y = Y_.squeeze()
        Z = (Y - self.loc) / (self.scale + EPS)
        D = np.zeros((self.var.shape[0], 2))
        D[:, 0] = -(2 * sp.stats.norm.cdf(Z) - 1)
        D[:, 1] = self.crps(Y) + (Y - self.loc) * D[:, 0]
        return D

    def crps_metric(self):
        I = np.c_[2 * np.ones_like(self.var), np.zeros_like(self.var),
                  np.zeros_like(self.var), self.var]
        I = I.reshape((self.var.shape[0], 2, 2))
        I = 1/(2*np.sqrt(np.pi)) * I
        return I #+ 1e-4 * np.eye(2)

    def fisher_info(self):
        FI = np.zeros((self.var.shape[0], 2, 2))
        FI[:, 0, 0] = 1/self.var + 1e-5
        FI[:, 1, 1] = 2
        return FI

    def fisher_info_cens(self, T):
        nabla = np.array([self.pdf(T),
                          (T - self.loc) / self.scale * self.pdf(T)])
        return np.outer(nabla, nabla) / (self.cdf(T) * (1 - self.cdf(T))) + 1e-2 * np.eye(2)

    def fit(Y):
        m, s = sp.stats.norm.fit(Y)
        return np.array([m, np.log(s)])
        #return np.array([m, np.log(1e-5)])


class NGBoost(BaseEstimator):

    def __init__(self, Dist=Normal, Score=MLE(),
                 Base=default_tree_learner, natural_gradient=True,
                 n_estimators=500, learning_rate=0.01, minibatch_frac=1.0,
                 verbose=True, verbose_eval=100, tol=1e-4):
        self.Dist = Dist
        self.Score = Score
        self.Base = Base
        self.natural_gradient = natural_gradient
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.minibatch_frac = minibatch_frac
        self.verbose = verbose
        self.verbose_eval = verbose_eval
        self.init_params = None
        self.base_models = []
        self.scalings = []
        self.tol = tol

    def pred_param(self, X, max_iter=None):
        m, n = X.shape
        params = np.ones((m, self.Dist.n_params)) * self.init_params
        for i, (models, s) in enumerate(zip(self.base_models, self.scalings)):
            if max_iter and i == max_iter:
                break
            resids = np.array([model.predict(X) for model in models]).T
            params -= self.learning_rate * resids * s
        return params

    def sample(self, X, Y, params):
        if self.minibatch_frac == 1.0:
            return np.arange(len(Y)), X, Y, params
        sample_size = int(self.minibatch_frac * len(Y))
        idxs = np_rnd.choice(np.arange(len(Y)), sample_size, replace=False)
        return idxs, X[idxs,:], Y[idxs], params[idxs, :]

    def fit_base(self, X, grads):
        models = [self.Base().fit(X, g) for g in grads.T]
        fitted = np.array([m.predict(X) for m in models]).T
        self.base_models.append(models)
        return fitted

    def line_search(self, resids, start, Y, scale_init=1):
        S = self.Score
        D_init = self.Dist(start.T)
        loss_init = S.loss(D_init, Y)
        scale = scale_init
        while True:
            scaled_resids = resids * scale
            D = self.Dist((start - scaled_resids).T)
            loss = S.loss(D, Y)
            norm = np.mean(np.linalg.norm(scaled_resids, axis=1))
            if not np.isnan(loss) and (loss < loss_init or norm < self.tol) and               np.linalg.norm(scaled_resids, axis=1).mean() < 5.0:
                break
            scale = scale * 0.5
        self.scalings.append(scale)
        return scale

    def fit(self, X, Y, X_val = None, Y_val = None, train_loss_monitor = None, val_loss_monitor = None):

        loss_list = []
        val_loss_list = []
        self.fit_init_params_to_marginal(Y)

        params = self.pred_param(X)
        if X_val is not None and Y_val is not None:
            val_params = self.pred_param(X_val)

        S = self.Score

        if not train_loss_monitor:
            train_loss_monitor = S.loss

        if not val_loss_monitor:
            val_loss_monitor = S.loss

        for itr in range(self.n_estimators):
            _, X_batch, Y_batch, P_batch = self.sample(X, Y, params)

            D = self.Dist(P_batch.T)

            loss_list += [train_loss_monitor(D, Y_batch)]
            loss = loss_list[-1]
            grads = S.grad(D, Y_batch, natural=self.natural_gradient)

            proj_grad = self.fit_base(X_batch, grads)
            scale = self.line_search(proj_grad, P_batch, Y_batch)

            params -= self.learning_rate * scale * np.array([m.predict(X) for m in self.base_models[-1]]).T

            val_loss = 0
            if X_val is not None and Y_val is not None:
                val_params -= self.learning_rate * scale * np.array([m.predict(X_val) for m in self.base_models[-1]]).T
                val_loss = val_loss_monitor(self.Dist(val_params.T), Y_val)
                val_loss_list += [val_loss]
                if len(val_loss_list) > 10 and np.mean(np.array(val_loss_list[-5:])) >                    np.mean(np.array(val_loss_list[-10:-5])):
                    if self.verbose:
                        print(f"== Quitting at iteration / VAL {itr} (val_loss={val_loss:.4f})")
                    break

            if self.verbose and int(self.verbose_eval) > 0 and itr % int(self.verbose_eval) == 0:
                grad_norm = np.linalg.norm(grads, axis=1).mean() * scale
                print(f"[iter {itr}] loss={loss:.4f} val_loss={val_loss:.4f} scale={scale:.4f} "
                      f"norm={grad_norm:.4f}")

            if np.linalg.norm(proj_grad, axis=1).mean() < self.tol:
                if self.verbose:
                    print(f"== Quitting at iteration / GRAD {itr}")
                break

        return self

    def fit_init_params_to_marginal(self, Y, iters=1000):
        try:
            E = Y['Event']
            T = Y['Time'].reshape((-1, 1))[E == 1]
        except:
            T = Y
        self.init_params = self.Dist.fit(T)
        return


    def pred_dist(self, X, max_iter=None):
        params = np.asarray(self.pred_param(X, max_iter))
        dist = self.Dist(params.T)
        return dist

    def predict(self, X):
        dist = self.pred_dist(X)
        return list(dist.loc.flatten())


# In[ ]:


# https://www.kaggle.com/bgmello/neural-networks-feature-engineering-for-the-win
class RAdam(keras.optimizers.Optimizer):
    """RAdam optimizer.
    # Arguments
        learning_rate: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        weight_decay: float >= 0. Weight decay for each param.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
        total_steps: int >= 0. Total number of training steps. Enable warmup by setting a positive value.
        warmup_proportion: 0 < warmup_proportion < 1. The proportion of increasing steps.
        min_lr: float >= 0. Minimum learning rate after warmup.
    # References
        - [Adam - A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)
        - [On The Variance Of The Adaptive Learning Rate And Beyond](https://arxiv.org/pdf/1908.03265v1.pdf)
    """

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., weight_decay=0., amsgrad=False,
                 total_steps=0, warmup_proportion=0.1, min_lr=0., **kwargs):
        learning_rate = kwargs.pop('lr', learning_rate)
        super(RAdam, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            self.weight_decay = K.variable(weight_decay, name='weight_decay')
            self.total_steps = K.variable(total_steps, name='total_steps')
            self.warmup_proportion = K.variable(warmup_proportion, name='warmup_proportion')
            self.min_lr = K.variable(min_lr, name='min_lr')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.initial_weight_decay = weight_decay
        self.initial_total_steps = total_steps
        self.amsgrad = amsgrad

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr

        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations, K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1

        if self.initial_total_steps > 0:
            warmup_steps = self.total_steps * self.warmup_proportion
            decay_steps = K.maximum(self.total_steps - warmup_steps, 1)
            decay_rate = (self.min_lr - lr) / decay_steps
            lr = K.switch(
                t <= warmup_steps,
                lr * (t / warmup_steps),
                lr + decay_rate * K.minimum(t - warmup_steps, decay_steps),
            )

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='m_' + str(i)) for (i, p) in enumerate(params)]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='v_' + str(i)) for (i, p) in enumerate(params)]

        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='vhat_' + str(i)) for (i, p) in enumerate(params)]
        else:
            vhats = [K.zeros(1, name='vhat_' + str(i)) for i in range(len(params))]

        self.weights = [self.iterations] + ms + vs + vhats

        beta_1_t = K.pow(self.beta_1, t)
        beta_2_t = K.pow(self.beta_2, t)

        sma_inf = 2.0 / (1.0 - self.beta_2) - 1.0
        sma_t = sma_inf - 2.0 * t * beta_2_t / (1.0 - beta_2_t)

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)

            m_corr_t = m_t / (1.0 - beta_1_t)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                v_corr_t = K.sqrt(vhat_t / (1.0 - beta_2_t))
                self.updates.append(K.update(vhat, vhat_t))
            else:
                v_corr_t = K.sqrt(v_t / (1.0 - beta_2_t))

            r_t = K.sqrt((sma_t - 4.0) / (sma_inf - 4.0) *
                         (sma_t - 2.0) / (sma_inf - 2.0) *
                         sma_inf / sma_t)

            p_t = K.switch(sma_t >= 5, r_t * m_corr_t / (v_corr_t + self.epsilon), m_corr_t)

            if self.initial_weight_decay > 0:
                p_t += self.weight_decay * p

            p_t = p - lr * p_t

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    @property
    def lr(self):
        return self.learning_rate

    @lr.setter
    def lr(self, learning_rate):
        self.learning_rate = learning_rate

    def get_config(self):
        config = {
            'learning_rate': float(K.get_value(self.learning_rate)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'decay': float(K.get_value(self.decay)),
            'weight_decay': float(K.get_value(self.weight_decay)),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad,
            'total_steps': float(K.get_value(self.total_steps)),
            'warmup_proportion': float(K.get_value(self.warmup_proportion)),
            'min_lr': float(K.get_value(self.min_lr)),
        }
        base_config = super(RAdam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
#from https://www.kaggle.com/davidcairuz/nfl-neural-network-w-softmax
def crps_loss(y_true, y_pred):
    return K.mean(K.square(y_true - K.cumsum(y_pred, axis=1)), axis=1)
def get_model(X_train):
    x = keras.layers.Input(shape=[X_train.shape[1]])
    fc1 = keras.layers.Dense(units=450, input_shape=[X_train.shape[1]])(x)
    act1 = keras.layers.PReLU()(fc1)
    bn1 = keras.layers.BatchNormalization()(act1)
    dp1 = keras.layers.Dropout(0.55)(bn1)
    gn1 = keras.layers.GaussianNoise(0.15)(dp1)
    concat1 = keras.layers.Concatenate()([x, gn1])
    fc2 = keras.layers.Dense(units=600)(concat1)
    act2 = keras.layers.PReLU()(fc2)
    bn2 = keras.layers.BatchNormalization()(act2)
    dp2 = keras.layers.Dropout(0.55)(bn2)
    gn2 = keras.layers.GaussianNoise(0.15)(dp2)
    concat2 = keras.layers.Concatenate()([concat1, gn2])
    fc3 = keras.layers.Dense(units=400)(concat2)
    act3 = keras.layers.PReLU()(fc3)
    bn3 = keras.layers.BatchNormalization()(act3)
    dp3 = keras.layers.Dropout(0.55)(bn3)
    gn3 = keras.layers.GaussianNoise(0.15)(dp3)
    concat3 = keras.layers.Concatenate([concat2, gn3])
    output = keras.layers.Dense(units=199, activation='softmax')(concat2)
    model = keras.models.Model(inputs=[x], outputs=[output])
    return model
def train_model(X_train, y_train, X_val, y_val, batch_size=64, epochs=100):
    model = get_model(X_train)
    model.compile(optimizer=RAdam(warmup_proportion=0.1, min_lr=1e-7), loss=crps_loss)
    er = EarlyStopping(patience=20, min_delta=1e-4, restore_best_weights=True, monitor='val_loss')
    model.fit(X_train, y_train, epochs=epochs, callbacks=[er], validation_data=[X_val, y_val], batch_size=batch_size)
    return model


# In[ ]:


# https://www.kaggle.com/dandrocec/location-eda-with-rusher-features#Let's-build-NN

class CRPSCallback(Callback):
    def __init__(self,validation, predict_batch_size=20, include_on_batch=False):
        super(CRPSCallback, self).__init__()
        self.validation = validation
        self.predict_batch_size = predict_batch_size
        self.include_on_batch = include_on_batch

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_train_begin(self, logs={}):
        if not ('CRPS_score_val' in self.params['metrics']):
            self.params['metrics'].append('CRPS_score_val')

    def on_batch_end(self, batch, logs={}):
        if (self.include_on_batch):
            logs['CRPS_score_val'] = float('-inf')

    def on_epoch_end(self, epoch, logs={}):
        logs['CRPS_score_val'] = float('-inf')
            
        if (self.validation):
            X_valid, y_valid = self.validation[0], self.validation[1]
            y_pred = self.model.predict(X_valid)
            y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
            y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
            val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * X_valid.shape[0])
            val_s = np.round(val_s, 6)
            logs['CRPS_score_val'] = val_s

def get_model2(x_tr,y_tr,x_val,y_val,epochs=100,batch_size=1024):
    inp = Input(shape = (x_tr.shape[1],))
    x = Dense(1024, input_dim=X.shape[1], activation='relu')(inp)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    
    out = Dense(199, activation='softmax')(x)
    model = Model(inp,out)
    model.compile(optimizer=RAdam(warmup_proportion=0.1, min_lr=1e-7), loss=crps_loss)
    #add lookahead
    #lookahead = Lookahead(k=5, alpha=0.5) # Initialize Lookahead
    #lookahead.inject(model) # add into model
    es = EarlyStopping(monitor='CRPS_score_val', 
                       mode='min',
                       restore_best_weights=True, 
                       verbose=1, 
                       patience=10)
    mc = ModelCheckpoint('best_model2.h5',monitor='CRPS_score_val',mode='min',
                                   save_best_only=True, verbose=1, save_weights_only=True)
    steps = x_tr.shape[0]/batch_size
    model.fit(x_tr, y_tr,callbacks=[CRPSCallback(validation = (x_val,y_val)),es,mc], epochs=epochs, batch_size=batch_size)
    model.load_weights("best_model2.h5")
    
    y_pred = model.predict(x_val)
    y_valid = y_val
    y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
    val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * x_val.shape[0])
    crps = np.round(val_s, 6)

    return model,crps

def predict2(x_te, models2, batch_size=1024):
    model_num = len(models2)
    for k,m in enumerate(models2):
        if k==0:
            y_pred = m.predict(x_te,batch_size=batch_size)
        else:
            y_pred+=m.predict(x_te,batch_size=batch_size)
            
    y_pred = y_pred / model_num
    
    return y_pred


# ## 2. train_df:

# In[ ]:


get_ipython().run_cell_magic('time', '', "train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)")


# In[ ]:


train_df.shape


# In[ ]:


train_df.describe(include='all')


# ## 3. Preprocessing:

# In[ ]:


train_df_orig = train_df.copy()


# In[ ]:


train_df.shape


# In[ ]:


stad_dict = {
    'AT&T Stadium' : "AT&T Stadium",'Arrowhead Stadium' : "Arrowhead Stadium",'Bank of America Stadium' : "Bank of America Stadium",
    'Broncos Stadium At Mile High' : "Broncos Stadium At Mile High",'Broncos Stadium at Mile High' : "Broncos Stadium At Mile High",
    'CenturyField' : "CenturyField", 'CenturyLink':"CenturyField", 'CenturyLink Field':"CenturyField",
    'Dignity Health Sports Park':"Dignity Health Sports Park",'Empower Field at Mile High':"Empower Field at Mile High",
    'Estadio Azteca':"Estadio Azteca", 'EverBank Field':"EverBank Field",'Everbank Field': "EverBank Field", 'FedExField':"FedExField",
    'FedexField':"FedExField", 'First Energy Stadium':"First Energy Stadium", 'FirstEnergy':"First Energy Stadium",
    'FirstEnergy Stadium':"First Energy Stadium", 'FirstEnergyStadium':"First Energy Stadium", 'Ford Field':"Ford Field",
    'Gillette Stadium':"Gillette Stadium", 'Hard Rock Stadium':"Hard Rock Stadium", 'Heinz Field':"Heinz Field",
    'Lambeau Field':"Lambeau Field", 'Lambeau field':"Lambeau Field", 'Levis Stadium':"Levis Stadium",
    'Lincoln Financial Field':"Lincoln Financial Field", 'Los Angeles Memorial Coliesum':"Los Angeles Memorial Coliesum",
    'Los Angeles Memorial Coliseum':"Los Angeles Memorial Coliesum", 'Lucas Oil Stadium':"Lucas Oil Stadium",
    'M & T Bank Stadium':"M & T Bank Stadium", 'M&T Bank Stadium':"M & T Bank Stadium", 'M&T Stadium':"M & T Bank Stadium",
    'Mercedes-Benz Dome':"Mercedes-Benz Dome", 'Mercedes-Benz Stadium':"Mercedes-Benz Stadium",
    'Mercedes-Benz Superdome' : "Mercedes-Benz Dome", 'MetLife':"MetLife", 'MetLife Stadium':"MetLife",
    'Metlife Stadium':"MetLife", 'NRG':"NRG", 'NRG Stadium':"NRG", 'New Era Field':"New Era Field",
    'Nissan Stadium':"Nissan Stadium", 'Oakland Alameda-County Coliseum':"Oakland Alameda-County Coliseum",
    'Oakland-Alameda County Coliseum':"Oakland Alameda-County Coliseum", 'Paul Brown Stadium':"Paul Brown Stadium",
    'Paul Brown Stdium':"Paul Brown Stadium", 'Raymond James Stadium':"Raymond James Stadium", 'Soldier Field':"Soldier Field",
    'Sports Authority Field at Mile High':"Sports Authority Field at Mile High", 'State Farm Stadium':"State Farm Stadium",
    'StubHub Center':"StubHub Center", 'TIAA Bank Field':"TIAA Bank Field", 'Tottenham Hotspur Stadium':"Tottenham Hotspur Stadium",
    'Twickenham':"Twickenham", 'Twickenham Stadium':"Twickenham", 'U.S. Bank Stadium':"U.S. Bank Stadium",
    'University of Phoenix Stadium':"University of Phoenix Stadium", 'Wembley Stadium':"Wembley Stadium"
}
stad_type_dict = {
    'Bowl' : "bowl", 'Closed Dome': "closed dome", 'Cloudy' : "outdoor",
    'Dome' : "dome",'Dome, closed' : "closed dome", 'Domed' : "dome",
    'Domed, Open' : "dome",'Domed, closed' : "closed dome",
    'Domed, open' : "dome",'Heinz Field' : "heinz",'Indoor' : "indoor",
    'Indoor, Open Roof' : "indoor open",'Indoor, Roof Closed' : "indoor",
    'Indoor, roof open' : "indoor open",'Indoors' : "indoor",'Open' : "outdoor",
    'Oudoor' : "outdoor",'Ourdoor' : "outdoor",'Outddors' : "outdoor",
    'Outdoor' : "outdoor",'Outdoor Retr Roof-Open' : "retr open",
    'Outdoors' : "outdoor",'Outdor' : "outdoor",'Outside' : "outdoor",
    'Retr. Roof - Closed' : "retr closed",'Retr. Roof - Open' : "retr open",
    'Retr. Roof Closed' : "retr closed",'Retr. Roof-Closed' : "retr closed",
    'Retr. Roof-Open' : "retr open",'Retractable Roof' : "retr closed",
    'indoor' : "indoor"
}
turf_dict = {
    'A-Turf Titan' : "titan", 'Artifical' : "artificial", 'Artificial' : "artificial",
    'DD GrassMaster' : "ddgrand",'Field Turf' : "field turf",'Field turf' : "field turf",
    'FieldTurf' : "field turf",'FieldTurf 360' : "field turf 360",
    'FieldTurf360' : "field turf 360",'Grass' : "grass",'Natural' : "natural",
    'Natural Grass' : "natural",'Natural grass' : "natural",'Naturall Grass' : "natural",
    'SISGrass' : "sisgrass",'Turf' : "field turf",'Twenty Four/Seven Turf' : "24/7 turf",
    'Twenty-Four/Seven Turf' : "24/7 turf",'UBU Speed Series-S5-M' : "ubu speed s5m",
    'UBU Sports Speed S5-M' : "ubu speed s5m",'UBU-Speed Series-S5-M' : "ubu speed s5m",
    'grass' : "grass",'natural grass' : "natural"
}
game_weather_dict = {
    "Breezy": "clear", "Light rain": "rain-light", "Mostly Clear" : "clear",
    "N/A Indoors": "indoor", "Partly cloudy and mild": "sunny", "partly cloudy": "sunny",
    '30% Chance of Rain': "cloudy", 'Clear' : "clear", 'Clear Skies' : "clear",
    'Clear and Cool' : "clear-cool",'Clear and Sunny' : "sunny",
    'Clear and cold' : "clear-cool",'Clear and sunny' : "sunny",
    'Clear and warm' : "sunny",'Clear skies' : "clear",
    'Cloudy' : "cloudy",'Cloudy and Cool' : "cloudy-cool",'Cloudy and cold' : "cloudy-cool",
    'Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.' : "rainy",
    'Cloudy, 50% change of rain' : "cloudy",'Cloudy, Rain' : "rainy",
    'Cloudy, chance of rain' : "cloudy",'Cloudy, fog started developing in 2nd quarter' : "cloudy-cool",
    'Cloudy, light snow accumulating 1-3"': "snow",'Cold' : "cool",'Controlled Climate' : "controlled",
    'Coudy': "cloudy",'Fair': "clear",'Hazy' : "haze",'Heavy lake effect snow' : "snow",
    'Indoor' : "indoor",'Indoors' : "indoor",'Light Rain' : "rain-light",'Mostly Cloudy' : "cloudy",
    'Mostly Coudy' : "cloudy",'Mostly Sunny' : "sunny",'Mostly Sunny Skies' : "sunny",
    'Mostly cloudy' : "cloudy",'Mostly sunny' : "sunny",'N/A (Indoors)' : "indoor",
    'N/A Indoor' : "indoor",'Overcast' : "cloudy",'Partly Cloudy' : "sunny",'Partly Clouidy' : "sunny",
    'Partly Sunny' : "cloudy",'Partly clear' : "cloudy",'Partly cloudy' : "sunny",
    'Partly sunny' : "cloudy",'Party Cloudy' : "sunny",'Rain' : "rainy",'Rain Chance 40%' : "cloudy",
    'Rain likely, temps in low 40s.' : "rainy",'Rain shower' : "rain-light",
    'Rainy' : "rainy",'Scattered Showers' : "rain-light",'Showers' : "rain-light",'Snow' : "snow",
    'Sun & clouds' : "cloudy",'Sunny' : "sunny",'Sunny Skies' : "sunny",'Sunny and clear' : "sunny",
    'Sunny and cold' : "sunny",'Sunny and warm' : "sunny",'Sunny, Windy' : "sunny",
    'Sunny, highs to upper 80s' : "sunny",'T: 51; H: 55; W: NW 10 mph' : "sunny",'cloudy' : "cloudy"
}
wind_speed_dict = {
    "6mph": "6","10mph": "10","9mph": "9","14 Gusting to 24": "14-24",
    "6 mph, Gusts to 10": "6-10","13 MPH": "13","4 MPh": "4",
    "15 gusts up to 25": "15-25","10MPH": "10","10mph": "10",
    "7 MPH": "7","Calm": "0","6 mph": "6","12mph": "12"
}
wind_direc_dict = { 'E' : "e", 'EAST' : "e",'ENE' : "ene",'ESE' : "ese",'East' : "e",
    'East North East' : "ene",'East Southeast' : "ese",'From ESE' : "ese",'From NNE' : "nne",
    'From NNW' : "nnw",'From S' : "s",'From SSE' : "sse",'From SSW' : "ssw",
    'From SW' : "sw",'From W' : "w",'From WSW' : "wsw",'N' : "n",'N-NE' : "nne",
    'NE' : "ne",'NNE' : "nne",'NNW' : "nnw",'NW' : "nw",'North': "n",'North East' : "ne",
    'North/Northwest' : "nne",'NorthEast' : "ne",'Northeast' : "ne",'Northwest' : "nw",
    'S' : "s",'SE' : "se",'SSE' : "sse",'SSW' : "ssw","S-SW": "ssw",'SW' : "sw",'South' : "s",
    'South Southeast' : "sse",'South Southwest' : "ssw",'SouthWest' : "sw",'Southeast' : "se",
    'Southwest' : "sw",'W' : "w",'W-NW' : "wnw",'W-SW' : "wsw",'WNW' : "wnw",'WSW' : "wsw",
    'West' : "w",'West Northwest' : "wnw",'West-Southwest' : "wsw",'from W' : "w",'s' : "s"
}


# In[ ]:


map_abbr = {'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'}
for abb in train_df['PossessionTeam'].unique():
    map_abbr[abb] = abb


# In[ ]:


from sklearn.preprocessing import LabelEncoder
from functools import partial

def label_encode(df, col, lb=None, extra_keys=[]):
    if lb is None: 
        lb = LabelEncoder()
        lb.fit(df[col].unique().tolist() + extra_keys)
    df[col] = lb.transform(df[col])
    return lb

def _find_parts(x):
    if x in [np.nan, None]: return {}
    spl1 = [t.strip() for t in x.split(",")]
    dict_ = {s.split(" ")[1]:np.int(s.split(" ")[0]) for s in spl1}
    return dict_

def handle_offence_defence(df, col):
    cols_add = None
    if col == "DefensePersonnel": cols_add = ['DB', 'DL', 'LB', 'OL', 'RB']
    elif col == "OffensePersonnel": cols_add = ['DB', 'DL', 'LB', 'OL', 'QB', 'RB', 'TE', 'WR']
    else: raise Exception()
    mapper = {el: 0 for el in cols_add}
    #print(mapper.keys())
    for el in mapper.keys():
        df[col + f"_{el}"] = df[col].map(lambda x: _find_parts(x)[el] if el in _find_parts(x).keys() else 0)
        
def preprocess(df, lbs=None, print_=False):
    #for col in df.columns:
    #    if col in ["GameClock", "TimeHandoff", "TimeSnap",
    #               "PlayerBirthDate"]: continue
    #    if df[col].dtype in ["object"]:
    #        print(":::: ", col, " ::::")
    #        print(df[col].nunique())
    #        print(df[col].sort_values().unique())
    #        print("\n\n")
    #yards = df1["Yards"].copy()
    #df1.drop(["Yards"], axis=1, inplace=True)
    #df = pd.concat([df1, df2], axis=0)
    get_lbs = False
    if lbs is None:
        lbs = {}
        get_lbs = True
    
    df['PossessionTeam'] = df['PossessionTeam'].map(map_abbr)
    df['HomeTeamAbbr'] = df['HomeTeamAbbr'].map(map_abbr)
    df['VisitorTeamAbbr'] = df['VisitorTeamAbbr'].map(map_abbr)
    # https://www.kaggle.com/cpmpml/initial-wrangling-voronoi-areas-in-python
    df['ToLeft'] = df.PlayDirection == "left"
    df['IsBallCarrier'] = df.NflId == df.NflIdRusher
    df['TeamOnOffense'] = "home"
    df.loc[df.PossessionTeam != df.HomeTeamAbbr, 'TeamOnOffense'] = "away"
    df['IsOnOffense'] = df.Team == df.TeamOnOffense # Is player on offense?
    df['TeamOnOffense'] = df['TeamOnOffense'] == "home"
    df['YardLine_std'] = 100 - df.YardLine
    df.loc[df.FieldPosition.fillna('') == df.PossessionTeam,  
              'YardLine_std'
             ] = df.loc[df.FieldPosition.fillna('') == df.PossessionTeam,  
              'YardLine']
    df['X_std'] = df.X
    df.loc[df.ToLeft, 'X_std'] = 120 - df.loc[df.ToLeft, 'X'] 
    df['Y_std'] = df.Y
    df.loc[df.ToLeft, 'Y_std'] = 160/3 - df.loc[df.ToLeft, 'Y']
    df['Dir_rad'] = np.mod(90 - df.Dir, 360) * math.pi/180.0
    df['Dir_rad'] = df['Dir_rad'].fillna(df['Dir_rad'].mean())
    df['Dir_std'] = df.Dir_rad
    df.loc[df.ToLeft, 'Dir_std'] = np.mod(np.pi + df.loc[df.ToLeft, 'Dir_rad'], 2*np.pi)
    df['Dir_std'] = df['Dir_std'].fillna(df['Dir_std'].mean())
    df['Orientation_rad'] = np.mod(df.Orientation, 360) * math.pi/180.0
    df.loc[df.Season >= 2018, 'Orientation_rad'
         ] = np.mod(df.loc[df.Season >= 2018, 'Orientation'] - 90, 360) * math.pi/180.0
    df['Orientation_rad'] = df['Orientation_rad'].fillna(df['Orientation_rad'].mean())
    if print_: print("Part I. Done.")
    # https://www.kaggle.com/anandavati/ngboost-for-nfl/
    feat_names = ['Fmap0_r3q0', 'Emap0_r3q0', 'Fmap0_r3q1', 'Emap0_r3q1', 'Fmap0_r3q2', 'Emap0_r3q2', 'Fmap0_r3q3', 'Emap0_r3q3', 'Fmap0_r3q4', 'Emap0_r3q4', 'Fmap0_r3q5', 'Emap0_r3q5', 'Fmap0_r3q6', 'Emap0_r3q6', 'Fmap0_r3q7', 'Emap0_r3q7', 'Fmap0_r10q0', 'Emap0_r10q0', 'Fmap0_r10q1', 'Emap0_r10q1', 'Fmap0_r10q2', 'Emap0_r10q2', 'Fmap0_r10q3', 'Emap0_r10q3', 'Fmap0_r10q4', 'Emap0_r10q4', 'Fmap0_r10q5', 'Emap0_r10q5', 'Fmap0_r10q6', 'Emap0_r10q6', 'Fmap0_r10q7', 'Emap0_r10q7', 'Fmap0_r30q0', 'Emap0_r30q0', 'Fmap0_r30q1', 'Emap0_r30q1', 'Fmap0_r30q2', 'Emap0_r30q2', 'Fmap0_r30q3', 'Emap0_r30q3', 'Fmap0_r30q4', 'Emap0_r30q4', 'Fmap0_r30q5', 'Emap0_r30q5', 'Fmap0_r30q6', 'Emap0_r30q6', 'Fmap0_r30q7', 'Emap0_r30q7', 'Fmap0_r120q0', 'Emap0_r120q0', 'Fmap0_r120q1', 'Emap0_r120q1', 'Fmap0_r120q2', 'Emap0_r120q2', 'Fmap0_r120q3', 'Emap0_r120q3', 'Fmap0_r120q4', 'Emap0_r120q4', 'Fmap0_r120q5', 'Emap0_r120q5', 'Fmap0_r120q6', 'Emap0_r120q6', 'Fmap0_r120q7', 'Emap0_r120q7']
    for col in feat_names: df[col] = np.nan

    @nb.njit()
    def calc(friends_R0, friends_Theta0, enemies_R0, enemies_Theta0, result):
        itr = 0
        Rs = [0, 3, 10, 30, 120]
        for r in range(len(Rs)-1):
            rl, rh = Rs[r], Rs[r+1]
            for t in range(8):
                tl, th = t * np.pi/8, (t+1) * np.pi/8
                result[itr] = (np.sum((rl <= friends_R0) & (friends_R0 < rh) & (tl <= friends_Theta0) & (friends_Theta0 < th)))
                itr+=1
                result[itr] = (np.sum((rl <= enemies_R0) & (enemies_R0 < rh) & (tl <= enemies_Theta0) & (enemies_Theta0 < th)))
                itr+=1
        return result
    def func(gp):
        ball_XY = gp.loc[gp["NflId"]==gp["NflIdRusher"], ["X", "Y"]].values[0]
        friends_rel_XY = gp.loc[gp["Team"]==gp["TeamOnOffense"], ["X", "Y"]].values - ball_XY
        enemy_rel_XY = gp.loc[gp["Team"]!=gp["TeamOnOffense"], ["X", "Y"]].values - ball_XY

        friends_R0 = np.sqrt(np.power(friends_rel_XY[:, 0], 2) + np.power(friends_rel_XY[:, 1], 2))
        enemy_R0 = np.sqrt(np.power(enemy_rel_XY[:, 0], 2) + np.power(enemy_rel_XY[:, 1], 2))

        friends_Theta0 = np.arctan2(friends_rel_XY[:, 0], friends_rel_XY[:, 1])
        enemy_Theta0 = np.arctan2(enemy_rel_XY[:, 0], enemy_rel_XY[:, 1])

        features = np.empty((len(feat_names), ))
        features = calc(friends_R0, friends_Theta0, enemy_R0, enemy_Theta0, features)

        return pd.DataFrame([features] * gp.shape[0], index=gp.index)
    df.loc[:, feat_names] = df.groupby("PlayId").apply(func).values
    if print_: print("Part II. Done.")
    # https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    from functools import partial
    def PolyArea(x,y):
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    def func(mdf, type_=1):
        xy = mdf[['X_std', 'Y_std']].values
        n_points = xy.shape[0]
        offense = mdf.IsOnOffense.values
        vor = Voronoi(xy)
        def_area = 0
        off_area = 0
        for r in range(n_points):
            region = vor.regions[vor.point_region[r]]
            if not -1 in region:
                polygon = [vor.vertices[i] for i in region]
                if offense[r]: off_area += PolyArea(*zip(*polygon))
                else: def_area += PolyArea(*zip(*polygon))
        if type_ == 1: return def_area
        elif type_ == 2: return off_area
    area_def = df.groupby("PlayId").apply(partial(func, type_=1))
    df["DefenceAreaCovered"] = df["PlayId"].map(area_def)
    area_off = df.groupby("PlayId").apply(partial(func, type_=2))
    df["OffenceAreaCovered"] = df["PlayId"].map(area_off)
    df["Area_diff"] = df["DefenceAreaCovered"] - df["OffenceAreaCovered"]
    if print_: print("Part III. Done.")
    # https://www.kaggle.com/scirpus/hybrid-gp-and-nn
    if "Yards" in df.columns: outcomes = df[['GameId','PlayId','Yards']].drop_duplicates()
    def new_X(x_coordinate, play_direction):
        if play_direction == 'left':
            return 120.0 - x_coordinate
        else:
            return x_coordinate
    def new_line(rush_team, field_position, yardline):
        if rush_team == field_position:
            # offense starting at X = 0 plus the 10 yard endzone plus the line of scrimmage
            return 10.0 + yardline
        else:
            # half the field plus the yards between midfield and the line of scrimmage
            return 60.0 + (50 - yardline)
    def euclidean_distance(x1,y1,x2,y2):
        x_diff = (x1-x2)**2
        y_diff = (y1-y2)**2

        return np.sqrt(x_diff + y_diff)
    def back_direction(orientation):
        if orientation > 180.0: return 1
        else: return 0
    def back_features(df):
        carriers = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','NflIdRusher','X','Y','Orientation','Dir','YardLine']]
        carriers['back_from_scrimmage'] = carriers['YardLine'] - carriers['X']
        carriers['back_oriented_down_field'] = carriers['Orientation'].apply(lambda x: back_direction(x))
        carriers['back_moving_down_field'] = carriers['Dir'].apply(lambda x: back_direction(x))
        carriers = carriers.rename(columns={'X':'back_X',
                                            'Y':'back_Y'})
        carriers = carriers[['GameId','PlayId','NflIdRusher','back_X','back_Y','back_from_scrimmage',
                             'back_oriented_down_field','back_moving_down_field']]

        return carriers
    def features_relative_to_back(df, carriers):
        player_distance = df[['GameId','PlayId','NflId','X','Y']]
        player_distance = pd.merge(player_distance, carriers, on=['GameId','PlayId'], how='inner')
        player_distance = player_distance[player_distance['NflId'] != player_distance['NflIdRusher']]
        player_distance['dist_to_back'] = player_distance[['X','Y','back_X','back_Y']]                                                    .apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)

        player_distance = player_distance.groupby(['GameId','PlayId','back_from_scrimmage','back_oriented_down_field',
                                                   'back_moving_down_field'])\
                                         .agg({'dist_to_back':['min','max','mean','std']})\
                                         .reset_index()
        player_distance.columns = ['GameId','PlayId','back_from_scrimmage','back_oriented_down_field','back_moving_down_field',
                                   'min_dist','max_dist','mean_dist','std_dist']

        return player_distance
    def defense_features(df):
        rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Team','X','Y']]
        rusher.columns = ['GameId','PlayId','RusherTeam','RusherX','RusherY']
        defense = pd.merge(df,rusher,on=['GameId','PlayId'],how='inner')
        defense = defense[defense['Team'] != defense['RusherTeam']][['GameId','PlayId','X','Y','RusherX','RusherY']]
        defense.loc[:, 'def_dist_to_back'] = defense[['X','Y','RusherX','RusherY']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)

        defense = defense.groupby(['GameId','PlayId'])                         .agg({'def_dist_to_back':['min','max','mean','std']})                         .reset_index()
        defense.columns = ['GameId','PlayId','def_min_dist','def_max_dist','def_mean_dist','def_std_dist']

        return defense
    def static_features(df):
        static_features = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','X','Y','S','A','Dis','Orientation','Dir',
                                                            'YardLine','Quarter','Down','Distance','DefendersInTheBox']].drop_duplicates()
        static_features['DefendersInTheBox'] = static_features['DefendersInTheBox'].fillna(np.mean(static_features['DefendersInTheBox']))

        return static_features
    def combine_features(df, relative_to_back, defense, rushing, static, deploy=False):
        tdf = pd.merge(relative_to_back,defense,on=['GameId','PlayId'],how='inner')
        tdf = pd.merge(tdf,rushing,on=['GameId','PlayId'],how='inner')
        tdf = pd.merge(tdf,static,on=['GameId','PlayId'],how='inner')

        if not deploy:
            tdf = pd.merge(tdf, outcomes, on=['GameId','PlayId'], how='inner')

        return pd.merge(df.loc[:, ['GameId','PlayId'] + [v for v in df.columns if v not in tdf.columns]], tdf, on=['GameId','PlayId'], how='inner')
    def rusher_features(df):
        rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Dir', 'S', 'A', 'X', 'Y']]
        rusher.columns = ['GameId','PlayId', 'RusherDir', 'RusherS', 'RusherA', 'RusherX', 'RusherY']
        
        radian_angle = (90 - rusher['RusherDir']) * np.pi / 180.0
        v_horizontal = np.abs(rusher['RusherS'] * np.cos(radian_angle))
        v_vertical = np.abs(rusher['RusherS'] * np.sin(radian_angle)) 
        
        rusher['v_horizontal'] = v_horizontal
        rusher['v_vertical'] = v_vertical
        rusher.columns = ['GameId','PlayId', 'RusherDir', 'RusherS','RusherA','RusherX', 'RusherY','v_horizontal', 'v_vertical']
        return rusher
    back_feats = back_features(df)
    rel_back = features_relative_to_back(df, back_feats)
    def_feats = defense_features(df)
    rush_feats = rusher_features(df)
    static_feats = static_features(df)
    if "Yards" in df.columns:
        df = combine_features(df, rel_back, def_feats, rush_feats, static_feats, deploy=False)
    else: df = combine_features(df, rel_back, def_feats, rush_feats, static_feats, deploy=True)
    if print_: print("Part IV. Done.")
    #if get_lbs: lbs["Team"] = label_encode(df, "Team")
    #else: label_encode(df, "Team", lb=lbs["Team"])
    df['Team'] = df['Team'].apply(lambda x: x.strip()=='home')
    # https://www.kaggle.com/anandavati/ngboost-for-nfl/
    # df.Team == df.TeamOnOffense
    df["FriendScore"] = df[["HomeScoreBeforePlay", "VisitorScoreBeforePlay", "Team", "TeamOnOffense"]].apply(lambda x: x[0] if (x[2]==x[3]) else x[1], axis=1)
    df["EnemyScore"] = df[["HomeScoreBeforePlay", "VisitorScoreBeforePlay", "Team", "TeamOnOffense"]].apply(lambda x: x[1] if (x[2]!=x[3]) else x[0], axis=1)
    df = df.drop(["HomeScoreBeforePlay", "VisitorScoreBeforePlay"], axis=1)
    df['PlayDirection'] = df['PlayDirection'].apply(lambda x: x.strip() == 'right')
    # length of arr = 250
    players_not_in_train = ['Darnell Savage', 'David Montgomery', 'Duke Shelley', 'Sebastian Joseph-Day', 'Brian Burns', 'Taylor Rapp', 'Christian Miller', 'Darrell Henderson', 'Travin Howard', 'Dennis Daley', 'Isaiah Mack', 'Jamil Douglas', 'Greedy Williams', 'A.J. Brown', 'Daniel Ekuale', 'Amani Hooker', "D'Ernest Johnson", 'Quincy Williams', 'Juan Thornhill', 'Will Richardson', 'Jawaan Taylor', 'Mecole Hardman', 'Byron Pringle', 'Joey Ivie', 'Gardner Minshew', 'Andrew Wingard', 'Dontavius Russell', 'Ryquell Armstead', 'Marquise Brown', 'Miles Boykin', 'Sam Eguavoen', 'Jonathan Ledbetter', 'Christian Wilkins', 'Justice Hill', 'Michael Deiter', 'Preston Williams', 'Jomal Wiltz', 'Chandler Cox', 'Chris Lammons', 'Patrick Mekari', 'Steven Parker', 'DeShon Elliott', 'James Crawford', 'Chris Lindstrom', 'Kaleb McGary', 'Garrett Bradbury', 'Irv Smith', 'Kris Boyd', 'Alexander Mattison', 'Bisi Johnson', 'Brandon Dillon', 'Jaeden Graham', 'Ed Oliver', 'Darryl Johnson', 'Cody Ford', 'Tommy Sweeney', 'Trevon Wesco', 'Dawson Knox', 'Quinnen Williams', 'Devin Singletary', 'Blake Cashman', 'Derrius Guice', 'Terry McLaurin', 'Kelvin Harmon', 'Montez Sweat', 'Jimmy Moreland', 'Cole Holcomb', 'Miles Sanders', 'Andre Dillard', 'J.J. Arcega-Whiteside', 'Rock Ya-Sin', 'Deon Cain', 'Ben Banogu', 'Parris Campbell', 'Jerry Tillery', 'Bobby Okereke', 'Khari Willis', 'D.K. Metcalf', 'Mike Jordan', 'Damion Willis', 'Bryan Mone', 'Germaine Pratt', 'Drew Sample', 'Ugo Amadi', 'Renell Wren', 'T.J. Hockenson', 'Zach Allen', 'Byron Murphy', 'Nick Bawden', 'Jahlani Tavai', 'Kyler Murray', 'KeeSean Johnson', 'Kevin Strong', 'Ty Johnson', 'Nick Gates', 'Deandre Baker', 'Dexter Lawrence', 'Oshane Ximines', 'Devin Smith', 'Tony Pollard', 'R.J. McIntosh', 'Ryan Connelly', 'Joe Jackson', 'Corey Ballentine', 'Dre Greenlaw', 'Deebo Samuel', 'Vernon Hargreaves III', 'Devin White', 'Nick Bosa', 'Dare Ogunbowale', 'Anthony Nelson', 'Emmanuel Moseley', 'Sean Murphy-Bunting', 'Mike Edwards', 'Kam Kelly', 'Devin Bush', 'Ryan Izzo', 'Isaiah Wynn', 'Chase Winovich', 'Jakobi Meyers', 'Gunner Olszewski', 'Shy Tuttle', 'Erik McCoy', 'Chauncey Gardner-Johnson', 'Alec Ingold', 'Josh Jacobs', 'Foster Moreau', 'Troy Fumagalli', 'Noah Fant', 'Dalton Risner', 'Clelin Ferrell', 'Johnathan Abram', 'Hunter Renfrow', 'Maxx Crosby', 'Justin Hollins', 'Malik Reed', 'Mike Purcell', 'Trayvon Mullen', 'Deionte Thompson', 'Miles Brown', 'Justin Skule', 'Azeez Al-Shaair', 'Elgton Jenkins', 'Rashan Gary', 'Will Redmond', 'Lonnie Johnson', 'Roderick Johnson', 'Tytus Howard', 'Charles Omenihu', 'Cullen Gillaspia', 'Byron Cowart', 'Ken Webster', 'Lano Hill', 'L.J. Collier', 'Benny Snell', 'Mason Rudolph', 'Diontae Johnson', 'Steven Sims', 'Robert Davis', 'Ced Wilson', 'Simeon Thomas', 'Keisean Nixon', 'Keelan Doss', 'Darwin Thompson', 'Andrew Beck', 'Diontae Spencer', "Dre'Mont Jones", 'Greg Gaines', 'Jamil Demby', 'Kendall Sheffield', 'Craig James', 'John Cominsky', 'Luke Falk', 'Kyle Phillips', 'Sheldrick Redwine', 'Mack Wilson', 'Aaron Stinnie', 'Andrew Brown', 'Trent Harris', 'Trysten Hill', 'Shaq Calhoun', 'Allen Lazard', 'Darrius Shepherd', 'Juwann Winfree', 'Kingsley Keke', 'E.J. Speed', 'Jaylon Ferguson', "Hercules Mata'afa", 'Marcus Epps', 'Nathan Meadors', 'Braxton Berrios', 'Will Harris', 'Greg Ward', 'Greg Little', 'Reggie Bonnafon', 'Daniel Jones', 'Scott Miller', 'Darius Slayton', 'Tuzar Skipper', 'Max Scharping', 'Roderic Teamer', 'Daniel Brunskill', 'Zach Gentry', 'Abdullah Anderson', 'J.P. Holtz', 'Nate Davis', 'Daylon Mack', 'Jakob Johnson', 'Tom Kennedy', 'C.J. Moore', 'Ashton Dulin', 'Troymaine Pope', 'Isaiah Prince', 'Andre Patton', 'Nasir Adderley', 'Drue Tranquill', 'Cortez Broughton', 'Trey Pipkins', 'Wes Martin', 'Jon Hilliman', 'Dwayne Haskins', 'Andy Isabella', 'Troy Reeder', 'Jonathan Harris', 'Duke Dawson', 'Fred Brown', 'Deonte Harris', 'Carl Granderson', "Lil'Jordan Humphrey", 'Jay Elliott', 'Jamarco Jones', 'Jalen Thompson', 'Michael Dogbe', 'Stanley Morgan', 'Josiah Tauaefa', 'Chuma Edoga', 'T.J. Edwards', 'Devlin Hodges', 'Duke Williams', 'Ryan Bates', 'Jarrett Stidham', 'Joejuan Williams', 'A.J. Johnson', 'Davontae Harris', 'Brandon Knight', 'Shakial Taylor', 'Marvell Tell', 'Ryan Hunter', 'Khalen Saunders', 'Deon Yelder']
    if get_lbs: lbs["DisplayName"] = label_encode(df, "DisplayName", extra_keys=players_not_in_train)
    else: label_encode(df, "DisplayName", lb=lbs["DisplayName"])
    df.drop(["DisplayName"], axis=1, inplace=True)
    df["GameClock"] = pd.to_datetime(df["GameClock"], format="%M:%S:00")
    df["GameClock_minute"] = df["GameClock"].dt.minute
    df["GameClock_second"] = df["GameClock"].dt.second
    df.drop("GameClock", axis=1, inplace=True)
    if print_: print("Part V. Done.")
    # https://www.kaggle.com/bgmello/neural-networks-feature-engineering-for-the-win
    def new_orientation(angle, play_direction):
        if play_direction == 0:
            new_angle = 360.0 - angle
            if new_angle == 360.0:
                new_angle = 0.0
            return new_angle
        else:
            return angle
    df['Orientation'] = df.apply(lambda row: new_orientation(row['Orientation'], row['PlayDirection']), axis=1)
    df['Dir'] = df.apply(lambda row: new_orientation(row['Dir'], row['PlayDirection']), axis=1)
    df['IsRusher'] = (df['NflId'] == df['NflIdRusher'])
    df['HomePossesion'] = df['PossessionTeam'] == df['HomeTeamAbbr']
    df['Field_eq_Possession'] = df['FieldPosition'] == df['PossessionTeam']
    df['HomeField'] = df['FieldPosition'] == df['HomeTeamAbbr']
    df['YardsLeft'] = df.apply(lambda row: 100-row['YardLine'] if row['HomeField'] else row['YardLine'], axis=1)
    df['YardsLeft'] = df.apply(lambda row: row['YardsLeft'] if row['PlayDirection'] else 100-row['YardsLeft'], axis=1)
    df['X'] = df[['X', 'PlayDirection']].apply(lambda x: x['X'] if x['PlayDirection'] else 120-x['X'], axis=1)
    if "Yards" in df.columns: df.drop(df.index[(df['YardsLeft']<df['Yards']) | (df['YardsLeft']-100>df['Yards'])], inplace=True)
    
    if get_lbs: lbs["PossessionTeam"] = label_encode(df, "PossessionTeam")
    else: label_encode(df, "PossessionTeam", lb=lbs["PossessionTeam"])
    df["FieldPosition"] = df["FieldPosition"].fillna("ZZZ")
    if get_lbs: lbs["FieldPosition"] = label_encode(df, "FieldPosition")
    else: label_encode(df, "FieldPosition", lb=lbs["FieldPosition"])
    df["OffenseFormation"] = df["OffenseFormation"].fillna("ZZZ")
    if get_lbs: lbs["OffenseFormation"] = label_encode(df, "OffenseFormation")
    else: label_encode(df, "OffenseFormation", lb=lbs["OffenseFormation"])
    if print_: print("Part VI. Done.")
    #if get_lbs: lbs["OffensePersonnel"] = label_encode(df, "OffensePersonnel", extra_keys=['7 OL, 1 RB, 1 TE, 0 WR,1 LB',
    #                                                                                                   '2 QB, 6 OL, 1 RB, 1 TE, 1 WR'])
    #else: label_encode(df, "OffensePersonnel", lb=lbs["OffensePersonnel"])
    #if get_lbs: lbs["DefensePersonnel"] = label_encode(df, "DefensePersonnel", extra_keys=['0 DL, 4 LB, 6 DB, 1 RB',
    #                                                                                                     '1 DL, 4 LB, 5 DB, 1 RB',
    #                                                                                                     '2 DL, 4 LB, 4 DB, 1 RB',
    #                                                                                                     '2 DL, 3 LB, 5 DB, 1 RB',
    #                                                                                                     '1 DL, 3 LB, 6 DB, 1 RB',
    #                                                                                                     '3 DL, 4 LB, 3 DB, 1 RB'])
    #else: label_encode(df, "DefensePersonnel", lb=lbs["DefensePersonnel"])
    handle_offence_defence(df, "OffensePersonnel")
    handle_offence_defence(df, "DefensePersonnel")
    df = df.drop(["OffensePersonnel", "DefensePersonnel"], axis=1)
    #if get_lbs: lbs["PlayDirection"] = label_encode(df, "PlayDirection")
    #else: label_encode(df, "PlayDirection", lb=lbs["PlayDirection"])
    if print_: print("Part VII. Done.")
    df["TimeHandoff"] = pd.to_datetime(df["TimeHandoff"], format="%Y-%m-%dT%H:%M:%S.%fZ")
    df["TimeHandoff_minute"] = df["TimeHandoff"].dt.minute
    df["TimeHandoff_second"] = df["TimeHandoff"].dt.second
    #df["TimeHandoff_microsecond"] = df["TimeHandoff"].dt.microsecond
    df["TimeSnap"] = pd.to_datetime(df["TimeSnap"], format="%Y-%m-%dT%H:%M:%S.%fZ")
    df["TimeSnap_minute"] = df["TimeSnap"].dt.minute
    df["TimeSnap_second"] = df["TimeSnap"].dt.second
    df['TimeDelta'] = (df['TimeHandoff'] - df['TimeSnap']).dt.seconds
    #df["TimeSnap_microsecond"] = df["TimeSnap"].dt.microsecond
    if print_: print("Part VIII. Done.")
    df["PlayerHeight"] = df["PlayerHeight"].map(lambda x: np.float(x.replace("-", ".")))
    # https://www.kaggle.com/bgmello/neural-networks-feature-engineering-for-the-win
    df['PlayerBMI'] = 703*(df['PlayerWeight']/(df['PlayerHeight'])**2)
    df["PlayerBirthDate"] = pd.to_datetime(df["PlayerBirthDate"], format="%m/%d/%Y")
    df["PlayerBirthDate_year"] = df["PlayerBirthDate"].dt.year
    df["PlayerBirthDate_month"] = df["PlayerBirthDate"].dt.month
    df["PlayerBirthDate_day"] = df["PlayerBirthDate"].dt.day
    seconds_in_year = 60*60*24*365.25
    df['PlayerAge'] = (df['TimeHandoff']-df['PlayerBirthDate']).dt.seconds / seconds_in_year
    df.drop("PlayerBirthDate", axis=1, inplace=True)
    df.drop(["TimeHandoff", "TimeSnap"], axis=1, inplace=True)
    df["PlayerCollegeName"] = df["PlayerCollegeName"].map(lambda x: re.sub("[\s]+", " ", x.lower().replace("state", "")).strip())
    if get_lbs: lbs["PlayerCollegeName"] = label_encode(df, "PlayerCollegeName", extra_keys=['murray',
                                                                                                         'bemidji',
                                                                                                         'california-davis',
                                                                                                         'mcneese',
                                                                                                         'charleston, w. va.',
                                                                                                         'tarleton',
                                                                                                         'bowling green',
                                                                                                         'bryant',
                                                                                                         'malone',
                                                                                                         'sioux falls'])
    else: label_encode(df, "PlayerCollegeName", lb=lbs["PlayerCollegeName"])
    if get_lbs: lbs["Position"] = label_encode(df, "Position")
    else: label_encode(df, "Position", lb=lbs["Position"])
    if get_lbs: lbs["HomeTeamAbbr"] = label_encode(df, "HomeTeamAbbr")
    else: label_encode(df, "HomeTeamAbbr", lb=lbs["HomeTeamAbbr"])
    if get_lbs: lbs["VisitorTeamAbbr"] = label_encode(df, "VisitorTeamAbbr")
    else: label_encode(df, "VisitorTeamAbbr", lb=lbs["VisitorTeamAbbr"])
    df["Stadium"] = df["Stadium"].map(lambda x: stad_dict[x] if x in stad_dict.keys() else x)
    if get_lbs: lbs["Stadium"] = label_encode(df, "Stadium", extra_keys=[stad_dict[v] for v in ['Dignity Health Sports Park',
                                                                                                             'Empower Field at Mile High',
                                                                                                             'FedexField',
                                                                                                             'Tottenham Hotspur Stadium']])
    else: label_encode(df, "Stadium", lb=lbs["Stadium"])
    if print_: print("Part IX. Done.")
    def func(x):
        x = x.replace("Texas", "TX").replace("Maryland", "Md")             .replace("e North Carolina", "e, NC").replace("North Carolina", "NC")             .replace(". IL", ", IL").replace("r CO", "r, CO").replace("E. ", "East ")             .replace("d Ohio", "d, Ohio")             .replace("Ohio", "OH").replace("e Florida", "e, FL").replace("Florida", "FL")             .replace("Calif.", "CA").replace("FLA", "FL").replace("Fla.", "FL")             .replace("La.", "LA").replace("k NY", "k, NY").replace("a, CA", "a, CSA")             .replace("Pa.", "PA")
        dict_ = {"Chicago":"Chicago, IL", "Cleveland":"Cleveland, OH","Detroit":"Detroit, MI",
                 "London":"London, England", "New Orleans":"New Orleans, LA", "Pittsburgh":"Pittsburgh, PA",
                 "Seattle":"Seattle, WA", "Mexico City":"Mexico City, Mexico"}
        if x in dict_.keys(): x = dict_[x]
        x = x.replace(".", "").lower()
        import re
        x = re.sub("[\s]+", " ", x)
        x = x.replace(", ", ",")
        return x
    df["Location"] = df["Location"].map(func)
    df["Location_place"] = df["Location"].map(lambda x: x.split(",")[0])
    df["Location_state"] = df["Location"].map(lambda x: "ZZZ" if len(x.split(","))==1 else x.split(",")[1])
    df.drop("Location", axis=1, inplace=True)
    if get_lbs: lbs["Location_place"] = label_encode(df, "Location_place")
    else: label_encode(df, "Location_place", lb=lbs["Location_place"])
    if get_lbs: lbs["Location_state"] = label_encode(df, "Location_state")
    else: label_encode(df, "Location_state", lb=lbs["Location_state"])
    df["StadiumType"] = df["StadiumType"].map(lambda x: stad_type_dict[x] if x in stad_type_dict.keys() else x)
    df["StadiumType"] = df["StadiumType"].fillna("ZZZ")
    if get_lbs: lbs["StadiumType"] = label_encode(df, "StadiumType")
    else: label_encode(df, "StadiumType", lb=lbs["StadiumType"])
    df["Turf"] = df["Turf"].map(lambda x: turf_dict[x] if x in turf_dict.keys() else x)
    if get_lbs: lbs["Turf"] = label_encode(df, "Turf")
    else: label_encode(df, "Turf", lb=lbs["Turf"])
    df["Temperature"] = df["Temperature"].map(np.float)
    df.loc[train_df["GameWeather"] == "T: 51; H: 55; W: NW 10 mph",             ["Humidity", "Temperature", "WindSpeed", "WindDirection"]] = np.array([55., 51., 10., "nw"])
    df["GameWeather"] = df["GameWeather"].map(lambda x: game_weather_dict[x] if x in game_weather_dict.keys() else x)
    df["GameWeather"] = df["GameWeather"].fillna("ZZZ")
    if get_lbs: lbs["GameWeather"] = label_encode(df, "GameWeather")
    else: label_encode(df, "GameWeather", lb=lbs["GameWeather"])
    if print_: print("Part X. Done.")
    def func(x):
        try:
            np.int(x)
            return True
        except: return False
    locs = df["WindDirection"].map(func)
    winD = df.loc[locs, "WindDirection"].copy()
    winS = df.loc[locs, "WindSpeed"].copy()
    df.loc[locs, "WindDirection"] = winS
    df.loc[locs, "WindSpeed"] = winD
    
    def func(x, dict_):
        if x in dict_.keys(): return dict_[x]
        else: return x
    df["WindSpeed"] = df["WindSpeed"].map(lambda x: func(x, wind_speed_dict))
    df["WindDirection"] = df["WindDirection"].map(lambda x: func(x, wind_direc_dict))

    def func(x, type_=0):
        if type(x) in [float, np.float, np.float64, np.float32,
                       int, np.int, np.int32, np.int64]:
            if type_==0: return 0
            elif type_==1: return x
        if x in [None, np.nan]: return np.nan
        if "-" in x:
            sp = x.split("-")
            if type_ == 0: return np.float(sp[1]) - np.float(sp[0])
            elif type_ == 1: return np.float(sp[0])
        else:
            if type_==0: return 0
            elif type_==1: return np.float(x)
    df["WindSpeed_diff"] = df["WindSpeed"].map(lambda x: func(x, type_=0))
    df["WindSpeed"] = df["WindSpeed"].map(lambda x: func(x, type_=1))
    df["WindDirection"] = df["WindDirection"].fillna("ZZZ")
    if get_lbs: lbs["WindDirection"] = label_encode(df, "WindDirection")
    else: label_encode(df, "WindDirection", lb=lbs["WindDirection"])
        
    df["WindSpeed"] = df["WindSpeed"].fillna(0)
    df["WindSpeed_diff"] = df["WindSpeed_diff"].fillna(0)
    for col in ["Temperature", "Humidity", "DefendersInTheBox", "Dir", "Orientation"]:
        #print(df[col].astype(np.float).quantile(.5))
        df[col] = df[col].fillna(df[col].astype(np.float).quantile(.5)).astype(np.float)
    if print_: print("Completed.")
    return df, lbs


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_df = train_df_orig.copy()\ntrain_df, label_encoders = preprocess(train_df)')


# In[ ]:


train_df.head()


# In[ ]:


for col in train_df.columns:
    if col in ["OffensePersonnel", "DefensePersonnel"]: continue
    if train_df[col].std() == 0: print("'"+col+"'", end=", ")


# In[ ]:


drop = ['Fmap0_r3q0', 'Fmap0_r3q1', 'Fmap0_r3q2', 'Fmap0_r3q3', 'Fmap0_r3q4', 'Fmap0_r3q5', 'Fmap0_r3q6', 'Fmap0_r3q7', 'Fmap0_r10q0', 'Fmap0_r10q1', 'Fmap0_r10q2', 'Fmap0_r10q3', 'Fmap0_r10q4', 'Fmap0_r10q5', 'Fmap0_r10q6', 'Fmap0_r10q7', 'Fmap0_r30q0', 'Fmap0_r30q1', 'Fmap0_r30q2', 'Fmap0_r30q3', 'Fmap0_r30q4', 'Fmap0_r30q5', 'Fmap0_r30q6', 'Fmap0_r30q7', 'Fmap0_r120q0', 'Fmap0_r120q1', 'Fmap0_r120q2', 'Fmap0_r120q3', 'Fmap0_r120q4', 'Fmap0_r120q5', 'Fmap0_r120q6', 'Fmap0_r120q7', 'Emap0_r120q7', 'DefensePersonnel_RB']
train_df = train_df.drop(drop, axis=1)


# In[ ]:


t = train_df.isnull().sum()
t[t>0]


# In[ ]:


print(train_df.columns.values.tolist())


# In[ ]:


columns = train_df.drop("PlayId", axis=1).columns.values.tolist()
train_df_new = train_df.groupby('PlayId').mean()
train_df_new.columns = columns
train_df = train_df_new.reset_index(drop=True)


# In[ ]:


train_df = train_df.drop(['GameId', 'Team', 'NflId', 'JerseyNumber', 'Season', 'NflIdRusher'], axis=1)


# In[ ]:


columns = train_df.columns.values.tolist()
def chk_corr(df, thresh=0.99, threaded=False):
    """
    Checks for highly correlated features and removes them.
    ---------------------------------------------------------------------
    Parameters:
        df: Dataframe to check for correlation
    Output:
        Return list of removed features/columns.
    """
    if threaded:
        import threading.Threads as t
        import Queue
        
        corr_matrix = pd.DataFrame(index=df.columns, columns=df.columns)
    else:
        corr_matrix = df.corr()
        
        # Taking only the upper triangular part of correlation matrix: (We want to remove only one of corr features)
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    
        # Find index of feature columns with correlation greater than {thresh}
        to_drop = [column for column in upper.columns if any(abs(upper[column]) > thresh)]
        
        print('%d feature are highly correlated.'%len(to_drop))
        return to_drop

to_drop = chk_corr(train_df, thresh=0.99)


# In[ ]:


print(to_drop)


# In[ ]:


train_df = train_df.drop(to_drop + ['Y'], axis=1)


# In[ ]:


X, yards = train_df.drop("Yards", axis=1).copy(), train_df["Yards"].copy()


# In[ ]:


Y = np.zeros(shape=(X.shape[0], 199))
for i, yard in enumerate(yards):
    Y[i, np.int(yard+99):] = np.ones(shape=(1, np.int(100-yard)))


# In[ ]:


print(Y)


# In[ ]:


from sklearn.preprocessing import StandardScaler
ssc = StandardScaler()
X = ssc.fit_transform(X)


# In[ ]:


get_ipython().run_cell_magic('time', '', "# https://www.kaggle.com/bgmello/neural-networks-feature-engineering-for-the-win\nfrom sklearn.model_selection import RepeatedKFold\nrkf = RepeatedKFold(n_splits=5, n_repeats=2)\n\nmodels1 = []\nlosses2, models2, crps_csv2 = [], [], []\nmodels4 = []\nfor tr_idx, vl_idx in rkf.split(X, Y, yards):\n    x_tr, y_tr = X[tr_idx], Y[tr_idx]\n    x_vl, y_vl = X[vl_idx], Y[vl_idx]\n    yards_tr, yards_vl = yards.values[tr_idx], yards.values[vl_idx]\n    \n    # 1st NN\n    # https://www.kaggle.com/bgmello/neural-networks-feature-engineering-for-the-win\n    model = train_model(x_tr, y_tr, x_vl, y_vl, batch_size=128, epochs=50)\n    models1.append(model)\n    \n    # 2nd NN\n    # https://www.kaggle.com/dandrocec/location-eda-with-rusher-features/data#Let's-build-NN\n    model2, crps2 = get_model2(x_tr, y_tr, x_vl, y_vl, batch_size=128, epochs=50)\n    models2.append(model2)\n    crps_csv2.append(crps2)\n    \n    # 1st Boosting (NGBoost)\n    # https://www.kaggle.com/anandavati/ngboost-for-nfl/\n    #ngb = NGBoost(Dist=Normal, Score=CRPS(), verbose=True, learning_rate=0.01, n_estimators=500)\n    #ngb.fit(x_tr, yards_tr, X_val=x_vl, Y_val=yards_vl)\n    \n    # 2nd Boosting (LightGBM)\n    # https://www.kaggle.com/hukuda222/nfl-simple-model-using-lightgbm\n    clf = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.01)\n    clf.fit(x_tr, yards_tr, eval_set=[(x_vl, yards_vl)], early_stopping_rounds=20, verbose=False)\n    models4.append(clf)")


# In[ ]:


np.mean(crps_csv2)


# ## Predict:

# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'for (test_df, sample_prediction_df) in env.iter_test():\n    prepared_df, _ = preprocess(test_df, lbs=label_encoders)\n    prepared_df = prepared_df.drop(drop, axis=1)\n    columns = prepared_df.drop("PlayId", axis=1).columns.values.tolist()\n    prepared_df = prepared_df.groupby(\'PlayId\').mean()\n    prepared_df.columns = columns\n    prepared_df = prepared_df.reset_index(drop=True)\n    prepared_df = prepared_df.drop([\'GameId\', \'Team\', \'NflId\', \'JerseyNumber\', \'Season\', \'NflIdRusher\'], axis=1)\n    prepared_df = prepared_df.drop(to_drop + [\'Y\'], axis=1)\n    yardsleft = prepared_df[\'YardsLeft\'].values\n    prepared_df = ssc.transform(prepared_df)\n    \n    # 1st NN\n    y_pred1 = np.mean([np.cumsum(model.predict(prepared_df), axis=1) for model in models1], axis=0)\n    for i in range(len(yardsleft)):\n        y_pred1[i, :np.int(yardsleft[i])-1] = 0\n        y_pred1[i, np.int(yardsleft[i])+100:] = 1\n    \n    # 2nd NN\n    y_pred2 = predict2(prepared_df, models2, batch_size=1024)\n    y_pred2 = np.cumsum(y_pred2, axis=1)\n    \n    # 1st Boosting (NGBoost)\n    #Q = list(range(-99, 100))\n    #y_pred3 = ngb.pred_dist(prepared_df)\n    #y_pred3 = y_pred3.cdf(Q)\n    #print(type(y_pred3))\n    #print(y_pred3.shape)\n    #print(y_pred3)\n    \n    # 2nd Boosting (LightGBM)\n    y_pred4 = np.zeros(199)        \n    y_pred4_p = np.sum(np.round([model.predict(prepared_df)[0] for model in models4]))/len(models4)\n    y_pred4_p += 99\n    for j in range(199):\n        if j>=y_pred4_p+10: y_pred4[j]=1.0\n        elif j>=y_pred4_p-10: y_pred4[j]=(j+10-y_pred4_p)*0.05\n    \n    y_pred_final = (2/4.*y_pred1+1/4.*y_pred2+1/4.*y_pred4) # +1/12.*y_pred3\n    env.predict(pd.DataFrame(data=y_pred_final.clip(0,1), index=sample_prediction_df.index, columns=sample_prediction_df.columns))')


# In[ ]:


env.write_submission_file()

