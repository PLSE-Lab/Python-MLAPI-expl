#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm.notebook import tqdm
import os
import re


# In[ ]:


df_train_image = pd.read_csv("/kaggle/input/trends-image-features-53-100/train_features.csv")
df_test_image = pd.read_csv("/kaggle/input/trends-image-features-53-100/test_features.csv")


# In[ ]:


df = pd.read_csv("/kaggle/input/trends-train-test-creator/train.csv")
test_df =pd.read_csv("/kaggle/input/trends-train-test-creator/test.csv")


# In[ ]:


targets = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']
tab_features = list(set(df.columns) - set(targets)- {'Id', 'is_train', 'IC_20'})
loading_features = [col for col in df.columns if '_vs_' not in col and col not in targets and col!='Id' and col!='is_train']
img_features = list(df_train_image.columns[1:])


# In[ ]:


df = pd.merge(df, df_train_image, on = 'Id', how = 'left')
test_df = pd.merge(test_df, df_test_image, on = 'Id', how = 'left')


# In[ ]:


df = df.fillna(0)


# In[ ]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, Callback, ReduceLROnPlateau
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import MeanSquaredError


# In[ ]:


SEED = 42
from numpy.random import seed
seed(SEED)
tf.random.set_seed(SEED)


# In[ ]:


get_ipython().system(' pip install spektral -q')
get_ipython().system(' pip install keras-self-attention -q')


# In[ ]:


def custom_metric(y_true, y_pred):
    score = 0
    w = [0.3, 0.175, 0.175, 0.175, 0.175]
    y_pred = y_pred * K.abs(K.sign(y_true))
    
    for i in range(5):
        t = K.mean(K.sum(K.abs(y_true[:,i] - y_pred[:,i]), axis=0)/K.sum(y_true[:,i], axis=0))
        score+= w[i] * t
    return score


# In[ ]:


MSE = MeanSquaredError()
def modified_mse(y_true, y_pred):
    y_pred = y_pred * K.abs(K.sign(y_true))
    return MSE(y_true, y_pred)


# In[ ]:


def score(y_true, y_pred):
    score = 0
    w = [0.3, 0.175, 0.175, 0.175, 0.175]
    y_pred = y_pred * np.abs(np.sign(y_true))
    
    for i in range(5):
        t = np.mean(np.sum(np.abs(y_true[:,i] - y_pred[:,i]), axis=0)/np.sum(y_true[:,i], axis=0))
        score+= w[i] * t
    return score


# In[ ]:


df['age_bins'] = pd.cut(x=df['age'], bins=[10, 20, 30, 40, 50, 60, 70, 80, 90], 
                                     labels=['teens','twenties','thirties','forties','fifties','sixties','seventies','eighties'])


# In[ ]:


def graph_utils():
  img_axes = pd.read_csv('../input/trends-assessment-prediction/ICN_numbers.csv')['ICN_number'].values
  axes_to_id = {v:i for i, v in enumerate(img_axes)}

  nodes = []
  for col in df.columns:
    if '_vs_' in col:
      a, b = col.split('_vs_')
      nodes = nodes + [a, b]
  nodes = set(nodes)

  node_to_id = {n:i for i,n in enumerate(nodes)}
  axes = []
  for k, v in node_to_id.items():
    num = re.findall(r'(?<=\().*?(?=\))', k)[0]
    axe_id = axes_to_id[int(num)]
    axes.append(axe_id)

  return node_to_id, axes

node_to_id, axes = graph_utils()


# In[ ]:


def features_to_matrix(row):
  mat = np.diag(np.ones(53))
  for col in df.columns:
    if '_vs_' in col:
        a, b = col.split('_vs_')
        id_a, id_b = node_to_id[a], node_to_id[b]
        mat[id_a][id_b] = row[col]
        mat[id_b][id_a] = row[col]
  return mat


# In[ ]:


from joblib import Parallel, delayed
train_graphs = Parallel(n_jobs=-1)( delayed(features_to_matrix)(r) for i, r in tqdm(df.iterrows(), total=len(df) ))
test_graphs = Parallel(n_jobs=-1)( delayed(features_to_matrix)(r) for i, r in tqdm(test_df.iterrows(), total=len(test_df) ))


# In[ ]:


# train_graphs = np.load('train_graph.npy')
# test_graphs = np.load('test_graph.npy')


# In[ ]:


train_graphs = np.array(train_graphs)
test_graphs = np.array(test_graphs)
train_graphs.shape, test_graphs.shape


# In[ ]:


# np.save('./train_graph.npy', train_graphs)
# np.save('./test_graph.npy', test_graphs)


# In[ ]:


from spektral.layers import GraphConv, ARMAConv
from spektral.utils.convolution import localpooling_filter
from keras_self_attention import SeqSelfAttention


# In[ ]:


class Block(layers.Layer):
    def __init__(self, out_sz, rate=0.1, act='relu'):
        super(Block, self).__init__()
        self.out_sz = out_sz
        self.rate = rate
        self.act = act
        self.bn = layers.BatchNormalization()
        self.drop = layers.Dropout(rate)
        self.l = layers.Dense(out_sz, activation=act)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'out_sz': self.out_sz,
            'rate': self.rate,
            'act': self.act,
        })
        return config
    
    def call(self, x):
        x = self.bn(x)
        x = self.drop(x)
        x = self.l(x)
        return x


# In[ ]:


def create_model(LR=1e-3):
    K.clear_session()
    
    F = 100
    N = 53
    Z = len(tab_features)
    n_layers = 6

    X = layers.Input(shape=(N, F))
    FIL = layers.Input((N, N), sparse=False)
    X1 = layers.Input(shape=(Z, ))
    

    channels = [64, 64, 64, 48, 32, 32]
    tab_layers = [512, 256, 128]

    assert len(channels)==n_layers

    gconv = []
    bn = []
    
    l0 = layers.Dense(5, activation='linear')
    drop = layers.Dropout(0.2)
    act = activations.relu

    tab = keras.Sequential([Block(out_sz = i) for i in tab_layers])


    for i in range(n_layers):
      gconv.append(GraphConv(channels[i], activation='linear', kernel_regularizer=l2(5e-6), use_bias=True))
      bn.append(layers.BatchNormalization())
    
    x, x1, f = X, X1, FIL

    # better features of 53 maps
    for i in range(n_layers):
      x = gconv[i]([x, f])
      x = bn[i](x)
      x = act(x)
      x = drop(x)
      
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.concatenate([x, x1], axis=-1)

    x = tab(x)
    x = l0(x)
    
    model = Model(inputs=[X, X1, FIL], outputs=x)

    optimizer = Adam(learning_rate = LR)
    model.compile(loss = modified_mse, 
                  optimizer = optimizer,
                 metrics = [custom_metric])
    return model


# In[ ]:


create_model().summary()


# In[ ]:


NUM_FOLDS = 10
EPOCHS = 25*2
LATENT_DIM = 100
overal_score = 0

y_oof = np.zeros((len(df), len(targets)))

fold_scores, test_fold_pred = [], []

# kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

for f, (train_ind, val_ind) in enumerate(skf.split(df, df['age_bins'])):
    
    print(f'**************************************************************************************** FOLD {f} ****************************************************************************')

    
    train_g, val_g = train_graphs[train_ind], train_graphs[val_ind]
    
    train_g = localpooling_filter(train_g).astype('f4')
    val_g = localpooling_filter(val_g).astype('f4')

    train_df, val_df = df.iloc[train_ind], df.iloc[val_ind]
    
    train_extra_features = train_df[tab_features].values
    val_extra_features = val_df[tab_features].values

    train_features = train_df[img_features].values.astype('float32').reshape(-1, 53, LATENT_DIM)
    train_features = train_features[:, axes, :]

    
    val_features = val_df[img_features].values.astype('float32').reshape(-1, 53, LATENT_DIM)
    val_feautures = val_features[:, axes, :]


    train_targets = train_df[targets].values.astype('float32')
    val_targets = val_df[targets].values.astype('float32')

    # print(train_g.shape, train_features.shape)
    # print(val_g.shape, val_features.shape)
    # print(train_targets.shape, val_targets.shape)
    # print(train_extra_features.shape, val_extra_features.shape)


    filepath = "model.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_custom_metric', verbose=1, save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_custom_metric', factor=0.01, patience=4, min_lr=1e-4, verbose=1)


    callbacks_list = [checkpoint, reduce_lr]

    model = create_model(LR = 5e-3)


    print('Starting training....')
    model.fit([train_features, train_extra_features, train_g], train_targets, 
                epochs = EPOCHS, 
              batch_size = 32, 
              callbacks = callbacks_list,
              verbose=2,
              validation_data = ([val_features, val_extra_features, val_g], val_targets)
             )

    model.load_weights(filepath)

    val_pred = abs(model.predict([val_features, val_extra_features, val_g]))
    y_oof[val_ind] = np.array(val_pred)
    
    
    metric = score(val_df[targets].values.astype('float32'), val_pred)
    
    print(f'Fold {f} : {metric}')
    fold_scores.append(metric)
    
    # test_graphs = localpooling_filter(test_graphs).astype('f4')

    test_features = test_df[img_features].values.reshape(-1, 53, LATENT_DIM)
    test_features = test_features[:, axes, :]
    test_extra_features = test_df[tab_features].values


    test_pred = abs(model.predict([test_features, test_extra_features, test_graphs]))
    test_fold_pred.append(test_pred)


# In[ ]:


fold_scores = np.array(fold_scores)
for i,f in enumerate(fold_scores):
    print(f'Fold {i} : {f}')
print(f'CV = {np.mean(fold_scores)}')


# In[ ]:


train_oof = pd.DataFrame({'Id': df.Id,
                         targets[0]: y_oof[:, 0],
                          targets[1]: y_oof[:, 1],
                          targets[2]: y_oof[:, 2],
                          targets[3]: y_oof[:, 3],
                          targets[4]: y_oof[:, 4],
                         })
train_oof.to_csv('gcn_train_preds_out.csv', index=False)
train_oof.head(4)


# In[ ]:


test_fold_pred = np.array(test_fold_pred)
test_preds = np.mean(test_fold_pred, axis=0)
# test_preds = np.average(test_fold_pred, axis=0, weights = fold_scores )
test_preds.shape


# In[ ]:


test_oof = pd.DataFrame({'Id': test_df.Id,
                         targets[0]: test_preds[:, 0],
                          targets[1]: test_preds[:, 1],
                          targets[2]: test_preds[:, 2],
                          targets[3]: test_preds[:, 3],
                          targets[4]: test_preds[:, 4],
                         })
test_oof.to_csv('gcn_test_preds_out.csv', index=False)
test_oof.head(4)


# In[ ]:


sub_df = pd.melt(test_df[["Id", "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]], id_vars=["Id"], value_name="Predicted")
sub_df["Id"] = sub_df["Id"].astype("str") + "_" +  sub_df["variable"].astype("str")

sub_df = sub_df.drop("variable", axis=1).sort_values("Id")
assert sub_df.shape[0] == test_df.shape[0]*5
sub_df['Predicted'] = test_preds.flatten()
sub_df.head(10)


# In[ ]:


sub_df.to_csv("submission.csv", index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




