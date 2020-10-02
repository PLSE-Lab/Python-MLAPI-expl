#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from kaggle.competitions import nflrush
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import keras

from keras.callbacks import EarlyStopping
from keras import backend as K
from collections import Counter
from sklearn.model_selection import RepeatedKFold

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Embedding, Concatenate, Flatten, BatchNormalization, Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
from keras import metrics


# In[ ]:


train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})


# In[ ]:


def strtoseconds(txt):
    txt = txt.split(':')
    ans = int(txt[0])*60 + int(txt[1]) + int(txt[2])/60
    return ans


# In[ ]:


def fix_play_direction(df):
    df.loc[df['PlayDirection'] == 'left', 'X'] = 120 - df.loc[df['PlayDirection'] == 'left', 'X']
    df.loc[df['PlayDirection'] == 'left', 'Y'] = (160 / 3) - df.loc[df['PlayDirection'] == 'left', 'Y']
    df.loc[df['PlayDirection'] == 'left', 'Orientation'] = np.mod(180 + df.loc[df['PlayDirection'] == 'left', 'Orientation'], 360)
    df.loc[df['PlayDirection'] == 'left', 'Dir'] = np.mod(180 + df.loc[df['PlayDirection'] == 'left', 'Dir'], 360)
    df['FieldPosition'].fillna('', inplace=True)
    df.loc[df['PossessionTeam'] != df['FieldPosition'], 'YardLine'] = 100 - df.loc[df['PossessionTeam'] != df['FieldPosition'], 'YardLine']
    return df

train = fix_play_direction(train)


# In[ ]:


train['GameClock'] = train['GameClock'].apply(strtoseconds)


# In[ ]:


train['PlayerHeight'] = train['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))


# In[ ]:


train['TimeHandoff'] = train['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
train['TimeSnap'] = train['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))


# In[ ]:


train['TimeDelta'] = train.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)


# In[ ]:


train['PlayerBirthDate'] = train['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))


# In[ ]:


seconds_in_year = 60*60*24*365.25
train['PlayerAge'] = train.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)


# In[ ]:


train = train.drop(['TimeHandoff', 'TimeSnap', 'PlayerBirthDate'], axis=1)


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


train.drop('WindDirection', axis=1, inplace=True)


# In[ ]:


train['PlayDirection'] = train['PlayDirection'].apply(lambda x: x is 'right')


# In[ ]:


train['Team'] = train['Team'].apply(lambda x: x.strip()=='home')


# In[ ]:


train['GameWeather'] = train['GameWeather'].str.lower()
indoor = "indoor"
train['GameWeather'] = train['GameWeather'].apply(lambda x: indoor if not pd.isna(x) and indoor in x else x)
train['GameWeather'] = train['GameWeather'].apply(lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly') if not pd.isna(x) else x)
train['GameWeather'] = train['GameWeather'].apply(lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isna(x) else x)
train['GameWeather'] = train['GameWeather'].apply(lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)


# In[ ]:


weather_count = Counter()
for weather in train['GameWeather']:
    if pd.isna(weather):
        continue
    for word in weather.split():
        weather_count[word]+=1
        
weather_count.most_common()[:15]


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


# In[ ]:


train['IsRusher'] = train['NflId'] == train['NflIdRusher']
train.drop(['NflId', 'NflIdRusher'], axis=1, inplace=True)


# In[ ]:


train = train.sort_values(by=['PlayId', 'Team', 'IsRusher']).reset_index()


# In[ ]:


play_features = []
pos_features = ["X", "Y"]
player_features = ["S", "A", "Dis", "Orientation", "Dir", "PlayerHeight", "PlayerWeight", "Position", "PlayerAge", "IsRusher", "Team"]
game_features = ["YardLine", "Quarter", "GameClock", 
"Down", "Distance", "HomeScoreBeforePlay", "VisitorScoreBeforePlay", "DefendersInTheBox", "Temperature", "Humidity", "TimeDelta"]


# In[ ]:


train.columns


# In[ ]:


game_train = train[game_features + ["PlayId"]].groupby("PlayId").max()[game_features].values


# In[ ]:


scaling_factor = 2
X_train = np.zeros((len(train)//22, 120//scaling_factor, 58//scaling_factor, len(player_features)+2), dtype = np.float16)


# In[ ]:


x_field = np.arange(-(120//scaling_factor)/2, (120//scaling_factor)/2, 1)
y_field = np.abs(np.arange(-(58//scaling_factor)/2, (58//scaling_factor)/2, 1))


# In[ ]:


field_array = np.dot(x_field[:, None],y_field[None,:])
field_array = field_array/field_array.max()


# In[ ]:


field_array.shape


# In[ ]:


X_train[:, :, :, -2] = field_array


# In[ ]:


#what our directional field we created looks like. Could probably be improved
plt.imshow(X_train[0, :, :, -2].astype(np.float32).T)


# In[ ]:


le = LabelEncoder()
train["Position"] = le.fit_transform(train["Position"])
train[["X", "Y"]] = train[["X", "Y"]].astype(int)


# In[ ]:


scalers = {}
for feature in player_features:
    if feature == "X" or feature == "Y" or feature == "IsRusher":
        continue
    else:
        ss = StandardScaler()
        train[feature] = ss.fit_transform(train[feature].values.reshape(-1, 1))
        scalers[feature] = ss


# In[ ]:


for i, playid in enumerate(tqdm(range(len(train)))):
    player_df = train.iloc[i, :]
    X_train[i//22, player_df["X"]//scaling_factor, player_df["Y"]//scaling_factor, 0:len(player_features)] += player_df[player_features].astype(np.float16)
    X_train[i//22, player_df["YardLine"]//2, :, -1] = 1


# In[ ]:


X_train[np.isnan(X_train)] = 0
game_train[np.isnan(game_train)] = 0


# In[ ]:


X_train.max()


# In[ ]:


y_train = np.zeros(shape=(train.shape[0]//22, 199))
for i,yard in enumerate(train['Yards'][::22]):
    y_train[i, yard+99:] = np.ones(shape=(1, 100-yard))


# In[ ]:


#viewing all of the different channels we have created. Some of these could be represented in better formats
for i in range(len(player_features)+2):
    plt.imshow(X_train[0, :, :, i].astype(np.float32).T)
    try:
        print(player_features[i])
    except:
        pass
    plt.show()


# In[ ]:


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


from keras.layers import Activation
from keras.utils import get_custom_objects
import tensorflow as tf
class Mish(Activation):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X = Activation('Mish', name="conv1_act")(X_input)
    '''

    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'


def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))


get_custom_objects().update({'Mish': Mish(mish)})


# In[ ]:


def crps(y_true, y_pred):
    return K.mean(K.square(y_true - K.cumsum(y_pred, axis=1)), axis=1)


# In[ ]:


from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, Flatten, BatchNormalization, Concatenate, Dropout
def create_model():
    image_inputs = Input(shape=(X_train.shape[1],X_train.shape[2], X_train.shape[3])) 
    game_inputs = Input(shape=(game_train.shape[1],)) 
    x = BatchNormalization()(image_inputs)
    x = Conv2D(16, (3,3), activation='Mish')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(16, (3,3), activation='Mish')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, (3,3), activation='Mish')(x)
    x = Conv2D(64, (3,3), activation='Mish')(x)
    x = GlobalMaxPooling2D()(x)
    game_feats = BatchNormalization()(game_inputs)
    game_feats = Dense(25, activation = 'Mish')(game_feats)
    x = Concatenate()([x, game_feats])
    x = Dropout(.2)(x)
    out = Dense(199, activation='softmax')(x)
    
    model = Model(inputs = [image_inputs,game_inputs], outputs=out)
    return model


# In[ ]:


def train_model(x_tr, y_tr, x_vl, y_vl):
    model = create_model()
    print(model.summary())
    er = EarlyStopping(patience=5, min_delta=1e-4, restore_best_weights=True, monitor='val_loss')
    model.compile(optimizer=RAdam(warmup_proportion=0.1, min_lr=1e-5), loss=crps, metrics=[metrics.mae])
    model.fit(x_tr, y_tr, epochs=100, callbacks=[er], validation_data=[x_vl, y_vl])
    return model


# In[ ]:


rkf = RepeatedKFold(n_splits=4, n_repeats=1)
models = []

for tr_idx, vl_idx in rkf.split(X_train, y_train):
    
    x_tr, y_tr = X_train[tr_idx].astype(np.float16), y_train[tr_idx].astype(np.float16)
    x_vl, y_vl = X_train[vl_idx].astype(np.float16), y_train[vl_idx].astype(np.float16)
    game_tr, game_vl = game_train[tr_idx], game_train[vl_idx]
    model = train_model([x_tr,game_tr], y_tr, [x_vl,game_vl], y_vl)
    models.append(model)


# In[ ]:


preds = models[0].predict([x_vl,game_vl])


# In[ ]:


plt.plot(preds[0].cumsum())


# In[ ]:


del train, X_train


# In[ ]:


def make_pred(df, sample, env, models):
    train = df
    train = fix_play_direction(train)
    train['GameClock'] = train['GameClock'].apply(strtoseconds)
    train['PlayerHeight'] = train['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))
    train['TimeHandoff'] = train['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    train['TimeSnap'] = train['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    train['TimeDelta'] = train.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)
    train['PlayerBirthDate'] = train['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))
    train['PlayerAge'] = train.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)
    train = train.drop(['TimeHandoff', 'TimeSnap', 'PlayerBirthDate'], axis=1)
    train['WindSpeed'] = train['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)
    train['WindSpeed'] = train['WindSpeed'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x)
    train['WindSpeed'] = train['WindSpeed'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)
    train['WindSpeed'] = train['WindSpeed'].apply(str_to_float)
    train.drop('WindDirection', axis=1, inplace=True)
    train['PlayDirection'] = train['PlayDirection'].apply(lambda x: x is 'right')
    train['Team'] = train['Team'].apply(lambda x: x.strip()=='home')
    train['GameWeather'] = train['GameWeather'].str.lower()
    indoor = "indoor"
    train['GameWeather'] = train['GameWeather'].apply(lambda x: indoor if not pd.isna(x) and indoor in x else x)
    train['GameWeather'] = train['GameWeather'].apply(lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly') if not pd.isna(x) else x)
    train['GameWeather'] = train['GameWeather'].apply(lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isna(x) else x)
    train['GameWeather'] = train['GameWeather'].apply(lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)
    train['GameWeather'] = train['GameWeather'].apply(map_weather)
    train['IsRusher'] = train['NflId'] == train['NflIdRusher']
    train.drop(['NflId', 'NflIdRusher'], axis=1, inplace=True)
    train = train.sort_values(by=['PlayId', 'Team', 'IsRusher']).reset_index()
    X_train = np.zeros((1, 120//scaling_factor, 58//scaling_factor, len(player_features)+2), dtype = np.float16)
    X_train[:, :, :, -2] = field_array
    
    train["Position"] = le.transform(train["Position"])
    train[["X", "Y"]] = train[["X", "Y"]].astype(int)
    for feature in player_features:
        if feature == "X" or feature == "Y" or feature == "IsRusher":
            continue
        else:
            ss = scalers[feature]
            train[feature] = ss.transform(train[feature].values.reshape(-1, 1))
    for row in range(22):
        player_df = train.iloc[row]
        X_train[0, player_df["X"]//scaling_factor, player_df["Y"]//scaling_factor, 0:len(player_features)] = player_df[player_features].astype(np.float16)
        X_train[0, train["YardLine"]//2, :, -1] = 1
    game_train = train[game_features + ["PlayId"]].groupby("PlayId").max()[game_features].values
    game_train[np.isnan(game_train)] = 0
    X_train[np.isnan(X_train)] = 0
    y_pred = np.mean([model.predict([X_train, game_train]) for model in models], axis=0)
    
    y_pred[0] = np.cumsum(y_pred[0], axis=0).clip(0, 1)
    y_pred[:, -1] = np.ones(shape=(y_pred.shape[0], 1))
    y_pred[:, 0] = np.zeros(shape=(y_pred.shape[0], 1))
    
    env.predict(pd.DataFrame(data=y_pred,columns=sample.columns))
    return y_pred


# In[ ]:


env = nflrush.make_env()


# In[ ]:


for test, sample in tqdm(env.iter_test()):
    make_pred(test, sample, env, models)


# In[ ]:


env.write_submission_file()


# In[ ]:




