#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
# os.system('pip install pandas==0.24.1')

import pandas as pd
print(pd.__version__)
SEED = 26
LR = 1e-4


# In[ ]:


import warnings
import logging

warnings.filterwarnings("ignore")
warnings.filterwarnings(action="ignore",category=DeprecationWarning)
warnings.filterwarnings(action="ignore",category=FutureWarning)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import math
import gc
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

np.random.seed(SEED)
from tensorflow import set_random_seed

set_random_seed(SEED)

from keras.layers import Dense, Input, Activation
from keras.layers import BatchNormalization,Add,Dropout
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras import callbacks
from keras import backend as K

def read_pickle(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def add_contributions(train, test):
    scalar_coupling_contributions = pd.read_csv(
        f"{DATA_PATH}/scalar_coupling_contributions.csv"
    )
    gc.collect()
    len_train = len(train)
    df = pd.concat([train, test], axis=0, sort=True)
    del train, test
    gc.collect()

    df = pd.merge(
        df,
        scalar_coupling_contributions,
        on=["molecule_name", "atom_index_0", "atom_index_1", "type"],
        how="left",
    )
    # energy = pd.read_csv(f"{DATA_DIR}/potential_energy.csv")
    # df = df.merge(
    #     energy, left_on=["molecule_name"], right_on=["molecule_name"], how="left"
    # )
    # del energy

    train = df.iloc[:len_train, :].reset_index(drop=True)
    test = df.iloc[len_train:, :].reset_index(drop=True)
    del df
    gc.collect()
    return train, test


def select_features_for_type(
        type_,
        imp_file="../input/cached-champs/lgb_feat_importance_full.csv",
        top_n=900,
        imp_threshold=100,
):
    df = pd.read_csv(imp_file)
    df = df[
        (df["type"] == type_)
        & (df["importance"] > 0)
        & (df["importance"] > imp_threshold)
    ]
    df = df.sort_values("importance", ascending=False)
    return list(df["feature"].values[:top_n])


# In[ ]:


print(os.listdir("../input"))
# feats
FILETRAIN1 = "../input/cached-champs/k_cached_processed_train2.pkl"
FILETEST1 = "../input/cached-champs/k_cached_processed_test2.pkl"
FILETRAIN = "../input/cached-champs/k_cached_processed_train_inv_pvals.pkl"
FILETEST = "../input/cached-champs/k_cached_processed_test_inv_pvals.pkl"
epoch_n = 2200
verbose = 1
batch_size = 2048
LINEAR_TOP = 27
LGB_TOP = 17
TARGETS = ['fc', 'sd', 'pso', 'dso']

logger = logging.getLogger('chimps gona win')
logger.setLevel(logging.DEBUG)
import pickle
DATA_PATH = '../input/champs-scalar-coupling'
SUBMISSIONS_PATH = './'
# use atomic numbers to recode atomic names
ATOMIC_NUMBERS = {
    'H': 1,
    'C': 6,
    'N': 7,
    'O': 8,
    'F': 9
}
logger.debug('starting... this is going to be awesome')


# In[ ]:


types_dict = {
    "1JHC": 0,
    "2JHH": 3,
    "1JHN": 1,
    "2JHN": 4,
    "2JHC": 2,
    "3JHH": 6,
    "3JHC": 5,
    "3JHN": 7,
}


atom_features = [
    "atom_2",
    "atom_3",
    "atom_4",
    "atom_5",
    "atom_6",
    "atom_7",
    "atom_8",
    "atom_9",
    # "overth",
]

fc_feats = [
    "fc_preds_type",
    "fc_preds_akira",
    "fc_preds_akira2",
    "fc_preds_akira3",
    "fc_preds_akira4",
]

types_to_run = ["1JHN", "1JHC", "2JHH", "2JHN", "2JHC", "3JHH", "3JHC", "3JHN"]
all_feats = []
for mol_type in types_to_run:
    i = types_dict[mol_type]
    linear_feats = select_features_for_type(
        i, top_n=LINEAR_TOP, imp_threshold=0,
        imp_file="../input/cached-champs/feats_inv_pvals.csv"
    )
    lgb_feats = select_features_for_type(
        i, top_n=LGB_TOP, imp_threshold=0,
    )
    feats = linear_feats + lgb_feats
    all_feats.append(feats)
    # print(i, mol_type, feats)
all_feats = np.unique(list(np.concatenate(all_feats)) + atom_features + fc_feats)
# all_feats = [col for col in all_feats if 'fc_preds' not in col]
print(all_feats)


# In[ ]:


train_dtypes = {
    'molecule_name': 'category',
    'atom_index_0': 'int8',
    'atom_index_1': 'int8',
    'type': 'category',
    'scalar_coupling_constant': 'float32'
}
train_csv = pd.read_csv(f'{DATA_PATH}/train.csv', index_col='id', dtype=train_dtypes)
# train_csv['molecule_index'] = train_csv.molecule_name.str.replace('dsgdb9nsd_', '').astype('int32')
cols = ['molecule_name', 'atom_index_0', 'atom_index_1', 'type']
train_csv = train_csv[cols]
train_csv.head(10)


# In[ ]:


import gc
test_csv = pd.read_csv(f'{DATA_PATH}/test.csv', index_col='id', dtype=train_dtypes)
test_csv['molecule_index'] = test_csv['molecule_name'].str.replace('dsgdb9nsd_', '').astype('int32')
cols = [col for col in cols if 'scalar_coupling_constant' not in col]
test_csv = test_csv[cols]
gc.collect()
print(train_csv.shape)
print(test_csv.shape)
test_csv.head()


# In[ ]:


train_csv, test_csv = add_contributions(train_csv, test_csv)


# In[ ]:


cols = ["molecule_name", "atom_index_0", "atom_index_1"]
train_csv = train_csv.drop(cols, axis=1)
test_csv = test_csv.drop(cols, axis=1)
train_csv.head()


# In[ ]:


# plug extra data
tr = read_pickle(FILETRAIN)
tr = tr.fillna(0)
train_ix = train_csv.index
tr.index = train_ix
train_csv = pd.concat([train_csv, tr], axis=1)
train_csv.index = train_ix
train_csv = train_csv[[col for col in train_csv.columns if col in list(all_feats) + TARGETS + ["type"]]]
del tr


# In[ ]:


te = read_pickle(FILETEST)
te = te.fillna(0)
test_ix = test_csv.index
te.index = test_ix
test_csv = pd.concat([test_csv, te], axis=1)
test_csv.index = test_ix
test_csv = test_csv[[col for col in test_csv.columns if col in list(all_feats) + ["type"]]]
del te, test_ix, train_ix
gc.collect()
print(train_csv.shape)
print(test_csv.shape)
test_csv.head(10)


# In[ ]:


# plug extra data
tr = read_pickle(FILETRAIN1)
tr = tr.fillna(0)[[col for col in tr.columns if col not in train_csv.columns]]
train_ix = train_csv.index
tr.index = train_ix
train_csv = pd.concat([train_csv, tr], axis=1)
train_csv.index = train_ix
del tr
train_csv = train_csv[list(all_feats) + TARGETS + ["type"]]
gc.collect()
te = read_pickle(FILETEST1)
te = te.fillna(0)[[col for col in te.columns if col not in test_csv.columns]]
test_ix = test_csv.index
te.index = test_ix
test_csv = pd.concat([test_csv, te], axis=1)
test_csv.index = test_ix
del te, test_ix, train_ix
gc.collect()
test_csv = test_csv[list(all_feats) + ["type"]]
test_csv.head(10)
print(train_csv.shape)
print(test_csv.shape)
gc.collect()


# In[ ]:


# test_csv = test_csv[list(all_feats)]
# train_csv = train_csv[list(all_feats) + ['scalar_coupling_constant']]
gc.collect()
# train_csv = reduce_mem_usage(train_csv)
# test_csv = reduce_mem_usage(test_csv)
print(len(train_csv.columns), len(np.unique(train_csv.columns)))


# In[ ]:


def create_nn_model(input_shape):
    inp = Input(shape=(input_shape,))
    x = Dense(2048, activation="relu")(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(1024, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    out = Dense(4, activation="linear")(x)  
    model = Model(inputs=inp, outputs=[out])
    return model

def plot_history(history, label):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss for %s' % label)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    _= plt.legend(['Train','Validation'], loc='upper left')
    plt.show()


# In[ ]:


# Set up GPU preferences
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 2} ) 
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6
sess = tf.Session(config=config) 
K.set_session(sess)


# In[ ]:


from datetime import datetime
from sklearn.preprocessing import OneHotEncoder

cv_score = []
cv_score_total = 0

# Set to True if we want to train from scratch.  False will reuse saved models as a starting point.
retrain = True

start_time = datetime.now()
test_prediction = np.zeros(len(test_csv))

class FeatureTransformer:
    def transform(self, dataset, ohe_features=[], continuous_features=[]):
        ohe_df = OneHotEncoder().fit_transform(dataset.loc[:, ohe_features]).toarray()
        skews = dataset.loc[:, continuous_features].skew().to_dict()
        # for column, skew in skews.items():
        # if skew > 1 and 'fc_preds' not in column:
        #     dataset[column] = np.log1p(dataset[column])

        return np.concatenate(
            [
                StandardScaler().fit_transform(dataset.loc[:, continuous_features]),
                ohe_df,
            ],
            axis=1,
        )
    
# types_to_run = ["1JHN"]   # quick test
for mol_type in types_to_run:
    model_name_wrt = "/kaggle/working/molecule_model_%s.hdf5" % mol_type
    model_name_rd = "../input/nncont-seed-25/molecule_model_%s.hdf5" % mol_type

    if mol_type=="3JHC":
        linear = 16
        lgb = 16
    else:
        linear = LINEAR_TOP
        lgb = LGB_TOP
    # n_feats=30
    linear_feats = select_features_for_type(
        types_dict[mol_type], top_n=linear, imp_threshold=0,
        imp_file="../input/cached-champs/feats_inv_pvals.csv"
    )
    lgb_feats = select_features_for_type(
        types_dict[mol_type], top_n=lgb, imp_threshold=0,
    )
    
    distance_features = list(np.unique(linear_feats + lgb_feats + fc_feats))
    input_features = list(np.unique(distance_features + atom_features))
    print(input_features)
    df_train_ = train_csv.loc[train_csv["type"] == mol_type, input_features + TARGETS]
    df_test_ = test_csv.loc[test_csv["type"] == mol_type, input_features]
    msg = f"training {mol_type}, out of {types_to_run}, train shape: {df_train_.shape}, LR: {LR}"
    print(msg)
    logger.debug(msg)
    # Standard Scaler from sklearn does seem to work better here than other Scalers
    input_data = FeatureTransformer().transform(
        dataset=pd.concat(
            [df_train_.loc[:, input_features], df_test_.loc[:, input_features]]
        ),
        ohe_features=atom_features,
        continuous_features=distance_features,  # + fc_feats
    )
    target_data = df_train_.loc[:, TARGETS].values

    # Simple split to provide us a validation set to do our CV checks with
    train_index, cv_index = train_test_split(
        np.arange(len(df_train_)), random_state=SEED, test_size=0.1
    )
    # Split all our input and targets by train and cv indexes
    train_target = target_data[train_index]
    cv_target = target_data[cv_index]
    train_input = input_data[train_index]
    cv_input = input_data[cv_index]
    test_input = input_data[len(df_train_) :, :]

    # Build the Neural Net
    nn_model = create_nn_model(train_input.shape[1])

    # If retrain==False, then we load a previous saved model as a starting point.
    retrain = False
    if not retrain:
        nn_model = load_model(model_name_rd)

    nn_model.compile(loss="mae", optimizer=Adam(lr=LR))

    # Callback for Early Stopping... May want to raise the min_delta for small numbers of epochs
    es = callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.0001,
        patience=50,
        verbose=1,
        mode="auto",
        restore_best_weights=True,
    )
    # Callback for Reducing the Learning Rate... when the monitor levels out for 'patience' epochs, then the LR is reduced
    rlr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=30, min_lr=1e-7, mode="auto", verbose=1
    )
    # Save the best value of the model for future use
    sv_mod = callbacks.ModelCheckpoint(
        model_name_wrt, monitor="val_loss", save_best_only=True, period=1
    )
    history = nn_model.fit(
        train_input,
        [train_target],
        validation_data=(cv_input, [cv_target]),
        callbacks=[es, rlr, sv_mod],
        epochs=epoch_n,
        batch_size=batch_size,
        verbose=0,
    )

    cv_predict = nn_model.predict(cv_input).sum(axis=1)
    plot_history(history, mol_type)
    accuracy = np.mean(np.abs(cv_target.sum(axis=1) - cv_predict))
    print(np.log(accuracy))
    cv_score.append(np.log(accuracy))
    cv_score_total += np.log(accuracy)

    # Predict on the test data set using our trained model
    test_predict = nn_model.predict(test_input).sum(axis=1)

    # for each molecule type we'll grab the predicted values
    test_prediction[test_csv["type"] == mol_type] = test_predict
    K.clear_session()

cv_score_total /= len(types_to_run)


# In[ ]:


submit = pd.read_csv(f'{DATA_PATH}/sample_submission.csv')
def submits(predictions):
    submit["scalar_coupling_constant"] = predictions
    submit.to_csv("/kaggle/working/nnetCont_sub.csv", index=False)
submits(test_prediction)


# In[ ]:


print ('Total training time: ', datetime.now() - start_time)

i=0
for mol_type in types_to_run: 
    print(mol_type,": cv score is ",cv_score[i])
    i+=1
print("total cv score is",cv_score_total)


# In[ ]:


submit = pd.read_csv(f'{DATA_PATH}/sample_submission.csv')
def submits(predictions):
    submit["scalar_coupling_constant"] = predictions
    submit.to_csv(f"/kaggle/working/nnetCont_sub_{round(cv_score_total, 4)}.csv", index=False)
submits(test_prediction)

