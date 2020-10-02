#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import datetime
from kaggle.competitions import nflrush
import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import keras

sns.set_style('darkgrid')
mpl.rcParams['figure.figsize'] = [15,10]


# In[ ]:


train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})


# # Overall analysis

# In[ ]:


train.head()


# - Let's see how PlayId is distribuited

# In[ ]:


train['PlayId'].value_counts()


# As expected, we have 22 of each playid since we have 22 players.
# 
# Let's look at our target variable(Yards).

# In[ ]:


train['Yards'].describe()


# In[ ]:


ax = sns.distplot(train['Yards'])
plt.vlines(train['Yards'].mean(), plt.ylim()[0], plt.ylim()[1], color='r', linestyles='--');
plt.text(train['Yards'].mean()-8, plt.ylim()[1]-0.005, "Mean yards travaled", size=15, color='r')
plt.xlabel("")
plt.title("Yards travaled distribution", size=20);


# # Categorical features

# In[ ]:


cat_features = []
for col in train.columns:
    if train[col].dtype =='object':
        cat_features.append((col, len(train[col].unique())))


# Let's preprocess some of those features.

# ## Offense formation

# In[ ]:


off_form = train['OffenseFormation'].unique()
train['OffenseFormation'].value_counts()


# Since I don't have any knowledge about formations, I am just goig to one-hot encode this feature

# In[ ]:


train = pd.concat([train.drop(['OffenseFormation'], axis=1), pd.get_dummies(train['OffenseFormation'], prefix='Formation')], axis=1)
dummy_col = train.columns


# ## Game Clock

# Game clock is supposed to be a numerical feature.

# In[ ]:


train['GameClock'].value_counts()


# Since we already have the quarter feature, we can just divide the Game Clock by 15 minutes so we can get the normalized time left in the quarter.

# In[ ]:


def strtoseconds(txt):
    txt = txt.split(':')
    ans = int(txt[0])*60 + int(txt[1]) + int(txt[2])/60
    return ans


# In[ ]:


train['GameClock'] = train['GameClock'].apply(strtoseconds)


# In[ ]:


sns.distplot(train['GameClock'])


# ## Player height

# In[ ]:


train['PlayerHeight']


# We know that 1ft=12in, thus:

# In[ ]:


train['PlayerHeight'] = train['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))


# ## Time handoff and snap and Player BirthDate

# In[ ]:


train['TimeHandoff']


# In[ ]:


train['TimeHandoff'] = train['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
train['TimeSnap'] = train['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))


# In[ ]:


train['TimeDelta'] = train.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)


# In[ ]:


train['PlayerBirthDate'] = train['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))


# Let's use the time handoff to calculate the players age

# In[ ]:


seconds_in_year = 60*60*24*365.25
train['PlayerAge'] = train.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)


# In[ ]:


train = train.drop(['TimeHandoff', 'TimeSnap', 'PlayerBirthDate'], axis=1)


# ## Wind Speed and Direction

# In[ ]:


train['WindSpeed'].value_counts()


# We can see there are some values that are not standardized(e.g. 12mph), we are going to remove mph from all our values.

# In[ ]:


train['WindSpeed'] = train['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)


# In[ ]:


train['WindSpeed'].value_counts()


# In[ ]:


#let's replace the ones that has x-y by (x+y)/2
# and also the ones with x gusts up to y
train['WindSpeed'] = train['WindSpeed'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x)
train['WindSpeed'] = train['WindSpeed'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)


# In[ ]:


def str_to_float(txt):
    try:
        return float(txt)
    except:
        return -1


# In[ ]:


train['WindSpeed'] = train['WindSpeed'].apply(str_to_float)


# In[ ]:


train['WindDirection'].value_counts()


# The wind direction won't affect our model much because we are analyzing running plays so we are just going to drop it.

# In[ ]:


train.drop('WindDirection', axis=1, inplace=True)


# ## PlayDirection

# In[ ]:


train['PlayDirection'].value_counts()


# In[ ]:


train['PlayDirection'] = train['PlayDirection'].apply(lambda x: x is 'right')


# ## Team

# In[ ]:


train['Team'] = train['Team'].apply(lambda x: x.strip()=='home')


# ## Game Weather

# In[ ]:


train['GameWeather'].unique()


# We are going to apply the following preprocessing:
#  
# - Lower case
# - N/A Indoor, N/A (Indoors) and Indoor => indoor Let's try to cluster those together.
# - coudy and clouidy => cloudy
# - party => partly
# - sunny and clear => clear and sunny
# - skies and mostly => ""

# In[ ]:


train['GameWeather'] = train['GameWeather'].str.lower()
indoor = "indoor"
train['GameWeather'] = train['GameWeather'].apply(lambda x: indoor if not pd.isna(x) and indoor in x else x)
train['GameWeather'] = train['GameWeather'].apply(lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly') if not pd.isna(x) else x)
train['GameWeather'] = train['GameWeather'].apply(lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isna(x) else x)
train['GameWeather'] = train['GameWeather'].apply(lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)


# In[ ]:


train['GameWeather'].unique()


# Let's now look at the most common words we have in the weather description

# In[ ]:


from collections import Counter
weather_count = Counter()
for weather in train['GameWeather']:
    if pd.isna(weather):
        continue
    for word in weather.split():
        weather_count[word]+=1
        
weather_count.most_common()[:15]


# To encode our weather we are going to do the following map:
#  
# - climate controlled or indoor => 3, sunny or sun => 2, clear => 1, cloudy => -1, rain => -2, snow => -3, others => 0
# - partly => multiply by 0.5
# 
# I don't have any expercience with american football so I don't know if playing in a climate controlled or indoor stadium is good or not, if someone has a good idea on how to encode this it would be nice to leave it in the comments :)

# In[ ]:


def map_weather(txt):
    ans = 1
    if pd.isna(txt):
        return 0
    if 'partly' in txt:
        ans*=0.5
    if 'climate controlled' in txt or 'indoor' in txt:
        return ans*3
    if 'sunny' in txt or 'sun' in txt:
        return ans*2
    if 'clear' in txt:
        return ans
    if 'cloudy' in txt:
        return -ans
    if 'rain' in txt or 'rainy' in txt:
        return -2*ans
    if 'snow' in txt:
        return -3*ans
    return 0


# In[ ]:


train['GameWeather'] = train['GameWeather'].apply(map_weather)


# ## NflId NflIdRusher

# In[ ]:


train['IsRusher'] = train['NflId'] == train['NflIdRusher']


# In[ ]:


train.drop(['NflId', 'NflIdRusher'], axis=1, inplace=True)


# # Baseline model

# Let's drop the categorical features and run a simple random forest in our model

# In[ ]:


train = train.sort_values(by=['PlayId', 'Team', 'IsRusher']).reset_index()


# In[ ]:


train.drop(['GameId', 'PlayId', 'index', 'IsRusher', 'Team', 'Season'], axis=1, inplace=True)


# In[ ]:


cat_features = []
for col in train.columns:
    if train[col].dtype =='object':
        cat_features.append(col)
        
train = train.drop(cat_features, axis=1)


# We are now going to make one big row for each play where the rusher is the last one

# In[ ]:


train.fillna(-999, inplace=True)


# In[ ]:


players_col = []
for col in train.columns:
    if train[col][:22].std()!=0:
        players_col.append(col)


# In[ ]:


X_train = np.array(train[players_col]).reshape(-1, 11*22)


# In[ ]:


play_col = train.drop(players_col+['Yards'], axis=1).columns
X_play_col = np.zeros(shape=(X_train.shape[0], len(play_col)))
for i, col in enumerate(play_col):
    X_play_col[:, i] = train[col][::22]


# In[ ]:


X_train = np.concatenate([X_train, X_play_col], axis=1)
y_train = np.zeros(shape=(X_train.shape[0], 199))
for i,yard in enumerate(train['Yards'][::22]):
    y_train[i, yard+99:] = np.ones(shape=(1, 100-yard))


# In[ ]:


from keras.callbacks import EarlyStopping


# In[ ]:


from keras import backend as K


__all__ = ['RAdam']


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


# In[ ]:


from keras.layers import Layer

class Mish(Layer):
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Mish, self).build(input_shape)

    def call(self, x):
        return x * K.tanh(K.softplus(x))

    def compute_output_shape(self, input_shape):
        return input_shape


# In[ ]:


from keras.callbacks import *

class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
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
    
    # Example
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
    """

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
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
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
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
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

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())


# In[ ]:


import tensorflow as tf


# In[ ]:


def train_model(x_tr, y_tr, x_vl, y_vl):
    inp = keras.layers.Input([X_train.shape[1]])
    x = keras.layers.Dense(units=324)(inp)
    x = keras.layers.BatchNormalization(momentum=0.8, axis=1)(x) 
    x = Mish()(x) 
    x = keras.layers.Dropout(0.35)(x) 
        
    x = keras.layers.Dense(units=512)(x) 
    x = keras.layers.BatchNormalization(momentum=0.8, axis=1)(x) 
    x = Mish()(x) 
    x = keras.layers.Dropout(0.5)(x) 
        
    x = keras.layers.Dense(units=324)(x) 
    x = keras.layers.BatchNormalization(momentum=0.8, axis=1)(x) 
    x = Mish()(x) 
    x = keras.layers.Dropout(0.35)(x)
    
    x = keras.layers.Dense(units=199, activation='sigmoid')(x) 
    
    model = keras.Model(inp, x) 
    er = EarlyStopping(patience=15, min_delta=1e-4, restore_best_weights=True, monitor='val_loss')
    clr = CyclicLR(base_lr=1e-3, max_lr=5*1e-3, step_size = 1000, gamma = 0.99)
    model.compile(optimizer=RAdam(warmup_proportion=0.1, min_lr=1e-5), 
                  loss='mse', 
                  metrics = ['mse']
                 )
    model.fit(x_tr, y_tr, epochs=100, callbacks=[clr, er], validation_data=[x_vl, y_vl])
    return model


# In[ ]:


scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train) 


# In[ ]:


def make_pred(df, sample, env, models):
    df['OffenseFormation'] = df['OffenseFormation'].apply(lambda x: x if x in off_form else np.nan)
    df = pd.concat([df.drop(['OffenseFormation'], axis=1), pd.get_dummies(df['OffenseFormation'], prefix='Formation')], axis=1)
    missing_cols = set( dummy_col ) - set( test.columns )-set('Yards')
    for c in missing_cols:
        df[c] = 0
    df = df[dummy_col]
    df.drop(['Yards'], axis=1, inplace=True)
    df['GameClock'] = df['GameClock'].apply(strtoseconds)
    df['PlayerHeight'] = df['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))
    df['TimeHandoff'] = df['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    df['TimeSnap'] = df['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    df['TimeDelta'] = df.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)
    df['PlayerBirthDate'] = df['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))
    seconds_in_year = 60*60*24*365.25
    df['PlayerAge'] = df.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)
    df['WindSpeed'] = df['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)
    df['WindSpeed'] = df['WindSpeed'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x)
    df['WindSpeed'] = df['WindSpeed'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)
    df['WindSpeed'] = df['WindSpeed'].apply(str_to_float)
    df['PlayDirection'] = train['PlayDirection'].apply(lambda x: x is 'right')
    df['Team'] = df['Team'].apply(lambda x: x.strip()=='home')
    indoor = "indoor"
    df['GameWeather'] = df['GameWeather'].apply(lambda x: indoor if not pd.isna(x) and indoor in x else x)
    df['GameWeather'] = df['GameWeather'].apply(lambda x: x.lower().replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly').replace('clear and sunny', 'sunny and clear').replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)
    df['GameWeather'] = df['GameWeather'].apply(map_weather)
    df['IsRusher'] = df['NflId'] == df['NflIdRusher']
    
    df = df.sort_values(by=['PlayId', 'Team', 'IsRusher']).reset_index()
    df = df.drop(['TimeHandoff', 'TimeSnap', 'PlayerBirthDate', 'WindDirection', 'NflId', 'NflIdRusher', 'GameId', 'PlayId', 'index', 'IsRusher', 'Team', 'Season'], axis=1)
    cat_features = []
    for col in df.columns:
        if df[col].dtype =='object':
            cat_features.append(col)

    df = df.drop(cat_features, axis=1)
    df.fillna(-999, inplace=True)
    X = np.array(df[players_col]).reshape(-1, 11*22)
    play_col = df.drop(players_col, axis=1).columns
    X_play_col = np.zeros(shape=(X.shape[0], len(play_col)))
    for i, col in enumerate(play_col):
        X_play_col[:, i] = df[col][::22]
    X = scaler.transform(np.concatenate([X, X_play_col], axis=1))
    y_pred = np.mean([model.predict(X) for model in models], axis=0)
    for pred in y_pred:
        prev = 0
        for i in range(len(pred)):
            if pred[i]<prev:
                pred[i]=prev
            prev=pred[i]
    
    env.predict(pd.DataFrame(data=y_pred,columns=sample.columns))
    return y_pred


# In[ ]:


from sklearn.model_selection import RepeatedKFold

rkf = RepeatedKFold(n_splits=5, n_repeats=5)


# In[ ]:


from keras import backend as K


# In[ ]:


models = []

for tr_idx, vl_idx in rkf.split(X_train, y_train):
    
    x_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    x_vl, y_vl = X_train[vl_idx], y_train[vl_idx]
    
    model = train_model(x_tr, y_tr, x_vl, y_vl)
    models.append(model)


# In[ ]:


env = nflrush.make_env()


# In[ ]:


for test, sample in tqdm.tqdm(env.iter_test()):
    make_pred(test, sample, env, models)


# In[ ]:


env.write_submission_file()


# # End
# 
# If you reached this far please comment and upvote this kernel, feel free to make improvements on the kernel and please share if you found anything useful!
