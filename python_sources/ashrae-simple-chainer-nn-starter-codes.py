#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import time
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
gc.enable()
device_id = 0  # cpu -> -1, gpu -> 0


# # Load data

# In[ ]:


get_ipython().run_cell_magic('time', '', '# import Dataset to play with it\ntrain_data = pd.read_csv(\'/kaggle/input/ashrae-energy-prediction/train.csv\')\nbuilding = pd.read_csv(\'/kaggle/input/ashrae-energy-prediction/building_metadata.csv\')\nweather_train = pd.read_csv(\'/kaggle/input/ashrae-energy-prediction/weather_train.csv\')\ntrain_data = train_data.merge(building, on=\'building_id\', how=\'left\')\ntrain_data = train_data.merge(weather_train, on=[\'site_id\', \'timestamp\'], how=\'left\')\n\ntest_data = pd.read_csv(\'/kaggle/input/ashrae-energy-prediction/test.csv\')\nweather_test = pd.read_csv(\'/kaggle/input/ashrae-energy-prediction/weather_test.csv\')\ntest_data = test_data.merge(building, on=\'building_id\', how=\'left\')\ntest_data = test_data.merge(weather_test, on=[\'site_id\', \'timestamp\'], how=\'left\')\n\nprint ("Done!")')


# Features that are likely predictive:
# 
# **Buildings**
# * primary_use
# * square_feet
# * year_built
# * floor_count (may be too sparse to use)
# 
# **Weather**
# * time of day
# * holiday
# * weekend
# * cloud_coverage + lags
# * dew_temperature + lags
# * precip_depth + lags
# * sea_level_pressure + lags
# * wind_direction + lags
# * wind_speed + lags
# 
# **Train**
# * max, mean, min, std of the specific building historically
# * number of meters
# * number of buildings at a siteid

# In[ ]:


train_data.isnull().any()


# In[ ]:


#Based on this great kernel https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
def reduce_mem_usage(df):
    start_mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in df.columns:
        if df[col].dtype != object:  # Exclude strings            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",df[col].dtype)            
            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()
            print("min for this col: ",mn)
            print("max for this col: ",mx)
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all(): 
                NAlist.append(col)
                df[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)    
            # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",df[col].dtype)
            print("******************************")
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return df, NAlist


# In[ ]:


train, _ = reduce_mem_usage(train_data)
test, _ = reduce_mem_usage(test_data)


# In[ ]:


del building, weather_train, weather_test
del train_data
del test_data
gc.collect()


# In[ ]:


train_columns = train.columns.tolist()


# # Simple data check

# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.describe()


# In[ ]:


train.dtypes


# In[ ]:


for c in train_columns:
    print(train[c].value_counts())
    print()


# In[ ]:


for c in test.columns:
    print(test[c].value_counts())
    print()


# # Feature preprocessing by sklearn pipeline

# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, QuantileTransformer


# In[ ]:


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, df, y=None):
        # df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.hour
        df["day"] = df["timestamp"].dt.day
        df["weekday"] = df["timestamp"].dt.weekday
        df["month"] = df["timestamp"].dt.month
        return df


# In[ ]:


minmax_features = [
    "year_built",
    "hour",
    "day",
    "weekday",
    "month",
]

minmax_transformer = make_pipeline(
    MinMaxScaler(),
)

numeric_features = [
    "square_feet",
    "air_temperature",
    "cloud_coverage",
    "dew_temperature",
    "floor_count",
]

numeric_transformer = make_pipeline(
    QuantileTransformer(
        n_quantiles=100,
        output_distribution="normal",
        random_state=0,
    ),
)

categorical_features = [
    "primary_use",
    "meter",
    "building_id",
]

categorical_transformer = make_pipeline(
    OrdinalEncoder(),
)

preprocessor = make_pipeline(
    DateFeatureExtractor(),
    ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("minmax", minmax_transformer, minmax_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    ),
)


# In[ ]:


preprocessed_train = preprocessor.fit_transform(train)


# In[ ]:


preprocessed_train[:5, :]


# In[ ]:


target = np.log1p(train[["meter_reading"]].values)


# In[ ]:


target[:5, :]


# In[ ]:


del train
gc.collect()


# # Chainer regresser model

# In[ ]:


import chainer
import chainer.functions as F
import chainer.links as L


# In[ ]:


chainer.print_runtime_info()


# In[ ]:


class MLP(chainer.Chain):

    def __init__(self, n_units=10, n_out=10):
        super(MLP, self).__init__()
        with self.init_scope():
            # embed_id
            self.embed_primary_use = L.EmbedID(16, 2)
            self.embed_meter = L.EmbedID(4, 2)
            self.embed_building_id = L.EmbedID(1449, 6)
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def forward(self, numeric_x, categorical_x):
        # embed layers
        e1 = self.embed_primary_use(categorical_x[:, 0])
        e2 = self.embed_meter(categorical_x[:, 1])
        e3 = self.embed_building_id(categorical_x[:, 2])
        
        # concat all inputs
        x = F.concat((numeric_x, e1, e2, e3), axis=1)
        
        # main layers
        h = F.dropout(F.relu(self.l1(x)), ratio=.1)
        h = F.dropout(F.relu(self.l2(h)), ratio=.1)
        return self.l3(h)


# In[ ]:


def train_and_validate(
    model,
    optimizer,
    train,
    validation,
    n_epoch,
    batchsize,
    device,
):
    # 1. If the device is gpu(>=0), send model to the gpu.
    if device >= 0:
        model.to_gpu(device)

    # 2. Setup optimizer
    optimizer.setup(model)

    # 3. Create iterator from datast
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    validation_iter = chainer.iterators.SerialIterator(
        validation, batchsize, repeat=False, shuffle=False
    )

    # 4. Create Updater/Trainer
    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=device)
    trainer = chainer.training.Trainer(updater, (n_epoch, 'epoch'), out='out')

    # 5. Extend functionalities of trainer
    trainer.extend(chainer.training.extensions.LogReport())
    trainer.extend(
        chainer.training.extensions.Evaluator(
            validation_iter, model, device=device
        ), name='val'
    )
    trainer.extend(
        chainer.training.extensions.PrintReport(
            ['epoch', 'main/loss', 'val/main/loss', 'elapsed_time']
        )
    )
    trainer.extend(
        chainer.training.extensions.PlotReport(
            ['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'
        )
    )

    # 6. Start training
    trainer.run()


# In[ ]:


# preprocessed_train = preprocessed_train[:1000]


# In[ ]:


from sklearn.model_selection import KFold

batchsize = 512
n_epoch = 20
n_splits = 3
seed = 666

kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

models = []
for fold_n, (train_index, valid_index) in enumerate(kf.split(preprocessed_train)):
    gc.collect()
    
    print()
    print('Fold:', fold_n)
    X_train, X_valid = preprocessed_train[train_index, :], preprocessed_train[valid_index, :]
    y_train, y_valid = target[train_index], target[valid_index]
    
    model = MLP(64, 1)
    regresser = L.Classifier(model, lossfun=F.mean_squared_error, accfun=F.mean_squared_error)
    optimizer = chainer.optimizers.Adam()
    
    train_and_validate(
        regresser,
        optimizer,
        chainer.datasets.TupleDataset(
            X_train[:, :len(numeric_features) + len(minmax_features)].astype("f"),
            X_train[:, len(numeric_features) + len(minmax_features):].astype("i"),
            y_train,
        ),
        chainer.datasets.TupleDataset(
            X_valid[:, :len(numeric_features) + len(minmax_features)].astype("f"),
            X_valid[:, len(numeric_features) + len(minmax_features):].astype("i"),
            y_valid,
        ),
        n_epoch,
        batchsize,
        device_id,
    )
    
    models.append(model)


# In[ ]:


del X_train, X_valid, y_train, y_valid, preprocessed_train, target
gc.collect()


# # Inference & Submission

# In[ ]:


test["meter_reading"] = 0.0


# In[ ]:


from tqdm import tqdm

if device_id >= 0:
    import cupy as cp

step_size = 50000

i = 0
res = []
for j in tqdm(range(int(np.ceil(test.shape[0] / 50000)))):
    gc.collect()
    batch = test[train_columns].iloc[i : i + step_size]
    preprocessed_batch = preprocessor.transform(batch)
    
    device = chainer.get_device(device_id)
    preprocessed_batch = device.send(preprocessed_batch)
    
    predictions = []
    with chainer.using_config('train', False):
        for model in models:
            ndarray = model(
                preprocessed_batch[:, :len(numeric_features) + len(minmax_features)].astype("f"),
                preprocessed_batch[:, len(numeric_features) + len(minmax_features):].astype("i"),
            )
            ndarray.to_cpu()
            predictions.append(ndarray.array)
        
    res.append(np.expm1(sum(predictions) / n_splits))
    i += step_size


# In[ ]:


res = np.concatenate(res)


# In[ ]:


submission = pd.read_csv('/kaggle/input/ashrae-energy-prediction/sample_submission.csv')
submission['meter_reading'] = res
submission.loc[submission['meter_reading'] < 0, 'meter_reading'] = 0
submission.to_csv('submission.csv', index=False)
submission

