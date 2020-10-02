#!/usr/bin/env python
# coding: utf-8

# ensembling 3 models
# 
# - BERT fine-tuned model
# - Universal Sentence Encoder + Dense NN
# - Universal Sentence Encoder + ElasticNet

# # Install libraries from kaggle dataset
# 
# First of all, we install `transformers`, `iterstrats` and its dependencies from **Kaggle dataset**, not from PyPI.
# This is because the internet connection is forbidden in this competition rule.

# In[ ]:


get_ipython().system('pip install ../input/sacremoses/sacremoses-master/ > /dev/null')
get_ipython().system('pip install ../input/transformers/transformers-2.3.0/ > /dev/null')
get_ipython().system('pip install ../input/iterative-stratification/iterative-stratification-master/ > /dev/null')


# # Import

# In[ ]:


import re
import random
import logging
from typing import List, Dict, Tuple, Any


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import GroupKFold
import joblib
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, GlobalAveragePooling1D, Lambda
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import tensorflow.keras.backend as K
import tensorflow_hub
import transformers
from transformers import TFBertModel, BertTokenizer
from scipy.stats import spearmanr
from tqdm import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.linear_model import MultiTaskElasticNet


# # Magic word

# In[ ]:


pd.options.display.max_columns = None
tf.get_logger().setLevel(logging.ERROR)

random.seed(31)
np.random.seed(31)
tf.random.set_seed(31)


# # Load CSV

# In[ ]:


train = pd.read_csv('../input/google-quest-challenge/train.csv')
target_cols = pd.read_csv('../input/google-quest-challenge/sample_submission.csv').columns[1:].tolist()
features_train, targets_train = train.drop(columns=target_cols), train.loc[:, target_cols]
del train

features_test = pd.read_csv('../input/google-quest-challenge/test.csv')


# # Reusable class

# In[ ]:


class BaseTransformer(BaseEstimator, TransformerMixin):
    
    def fit(self, x: pd.DataFrame, y = None):
        return self
    
    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return x


class ColumnTransformer(BaseTransformer):
    
    def __init__(self, defs: Dict[str, BaseTransformer]):
        self.defs = defs
    
    def fit(self, x: pd.DataFrame, y: np.ndarray = None):
        for col, transformer in self.defs.items():
            transformer.fit(x[col], y)
        return self
        
    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        xp = x.copy()
        for col, transformer in self.defs.items():
            xp[col] = transformer.transform(x[col])
        return xp
    
    def fit_transform(self, x: pd.DataFrame, y: np.ndarray = None) -> pd.DataFrame:
        xp = x.copy()
        for col, transformer in self.defs.items():
            if hasattr(transformer, 'fit_transform'):
                xp[col] = transformer.fit_transform(x[col], y)
            else:
                xp[col] = transformer.fit(x[col], y).transform(x[col])
        return xp


class WrappedOneHotEncoder(BaseTransformer):
    
    def __init__(self, col: str):
        self.col = col
        self.oe = OneHotEncoder(drop='first', sparse=False)
    
    def fit(self, x: pd.DataFrame, y = None):
        self.oe.fit(x.loc[:, [self.col]])
        return self
    
    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        sparse_matrix = self.oe.transform(x.loc[:, [self.col]])
        columns = ['{0}_onehot_{1}'.format(self.col, i) for i in range(sparse_matrix.shape[1])]
        onehot = pd.DataFrame(sparse_matrix, index=x.index, columns=columns)
        return pd.concat([x, onehot], axis=1)


class WrappedMinMaxScaler(BaseTransformer):
    
    def __init__(self, minmax: Tuple[float, float] = (1e-5, 1-1e-5)):
        self.mms = MinMaxScaler(minmax)
    
    def fit_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        array = self.mms.fit_transform(x)
        return pd.DataFrame(array, index=x.index, columns=x.columns)


# # Reusable functions

# In[ ]:


def colwise_spearmanr(y_true: np.ndarray, y_pred: np.ndarray, cols: List[str]) -> Dict[str, float]:
    return {c: spearmanr(y_true[:, i], y_pred[:, i]).correlation for i, c in enumerate(cols)}


def average_spearmanr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.average([
        spearmanr(y_t, y_p).correlation for y_t, y_p in zip(y_true.T, y_pred.T)
    ])


def mini_batch(l: List, batch_size):
    for i in range(0, len(l), batch_size):
        yield l[i:i + batch_size]


def rank_average(arrays: List[np.ndarray]) -> np.ndarray:
    rank_sum = np.sum([pd.DataFrame(a).rank().values for a in arrays], axis=0)
    return rank_sum / (len(arrays) * arrays[0].shape[0])


def pd_average(df_ys: List[pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame(
        np.average([df_y.values for df_y in df_ys], axis=0),
        columns=df_ys[0].columns,
        index=df_ys[0].index,
    )


def pd_rank_average(df_ys: List[pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame(
        rank_average([df_y.values for df_y in df_ys]),
        columns=df_ys[0].columns,
        index=df_ys[0].index,
    )


# # Transformer

# In[ ]:


class SecondLevelDomainExtracter(BaseTransformer):
    
    def transform(self, s_in: pd.Series) -> pd.Series:
        s = s_in.str.extract(r'(^|.*\.)([^\.]+\.[^\.]+$)').iloc[:, 1]
        s.name = s_in.name
        return s


class UniversalSentenceEncoderEncoder(BaseTransformer):
    
    def __init__(self, col: str, model, batch_size: int = 16):
        self.col = col
        self.model = model
        self.batch_size = batch_size
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        texts = df.loc[:, self.col].str.replace('?', '.').str.replace('!', '.').values
        pbar_total = len(df) // self.batch_size + 1
        pbar_desc = '{0} -> USE'.format(self.col)
        use_features = np.vstack([
            self.model(texts_batch)['outputs'].numpy() for texts_batch in tqdm(
                mini_batch(texts, self.batch_size),
                total=pbar_total,
                desc=pbar_desc
            )
        ])
        columns = ['{0}_use{1}'.format(self.col, i) for i in range(use_features.shape[1])]
        return pd.concat([df, pd.DataFrame(use_features, index=df.index, columns=columns)], axis=1)

    
class DistanceEngineerer(BaseTransformer):
    
    def __init__(self, col1: str, col2: str):
        self.col1 = col1
        self.col2 = col2
    
    @staticmethod
    def extract_matrix(df: pd.DataFrame, base_col: str) -> np.ndarray:
        columns = [c for c in df.columns if re.fullmatch(base_col + r'\d+', c) is not None]
        return df.loc[:, columns].values
    
    def transform(self, df_in: pd.DataFrame) -> pd.DataFrame:
        df = df_in.copy()
        a1 = self.extract_matrix(df, self.col1)
        a2 = self.extract_matrix(df, self.col2)
        assert a1.shape == a2.shape
        l2_distance = np.power(a1 - a2, 2).sum(axis=1)
        cos_distance = (a1 * a2).sum(axis=1)
        df.loc[:, 'l2dist_{0}-{1}'.format(self.col1, self.col2)] = l2_distance
        df.loc[:, 'cosdist_{0}-{1}'.format(self.col1, self.col2)] = cos_distance
        return df


class QATokenizer(BaseTransformer):
    
    def __init__(self, tokenizer, max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def tokenize(self, question_title: str, question_body: str, answer: str) -> Tuple[List[int], List[int], List[int]]:
        question = '{0}[SEP]{1}'.format(question_title, question_body)
        return self.tokenizer.encode_plus(question, answer, max_length=self.max_len, pad_to_max_length=True)
    
    def transform(self, df_in: pd.DataFrame) -> pd.DataFrame:        
        tokenized = [self.tokenize(
            row['question_title'],
            row['question_body'],
            row['answer']
        ) for _, row in df_in.iterrows()]
        input_ids = [d['input_ids'] for d in tokenized]
        attention_mask = [d['attention_mask'] for d in tokenized]
        token_type_ids = [d['token_type_ids'] for d in tokenized]
        
        return pd.DataFrame(np.hstack([input_ids, attention_mask, token_type_ids]), index=df_in.index)


class ColumnDropper(BaseTransformer):
    
    def __init__(self, cols: List[str]):
        self.cols = cols
        
    def transform(self, df_in: pd.DataFrame) -> pd.DataFrame:
        return df_in.drop(columns=self.cols)


# # Estimators

# In[ ]:


class BaseEnsembleCV:
    
    def __init__(self, n_splits: int = 5, verbose: bool = False, name: str = 'base_ecv'):
        self.n_splits = n_splits
        self.verbose = verbose
        self.name = name
    
    def fit_fold(self, x_train, y_train, x_val, y_val, **params):
        raise NotImplementedError()
        
    def save_fold(self, model, i_fold, dist):
        raise NotImplementedError()
    
    def load_fold(self, i_fold, src):
        raise NotImplementedError()
        
    def fit(self, df_x: pd.DataFrame, df_y: pd.DataFrame, groups = None, **params):
        
        x = df_x.values
        y = df_y.values

        if groups is None:
            # thanks to [Neuron Engineer's kernel](https://www.kaggle.com/ratthachat/quest-cv-analysis-on-different-splitting-methods)
            kfold = MultilabelStratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=31).split(x, y)
        else:
            kfold = GroupKFold(n_splits=self.n_splits).split(x, y, groups=groups)
                    
        iterator = enumerate(kfold)
        if self.verbose:
            iterator = tqdm(iterator, total=self.n_splits, desc='k-fold')

        self.models = []
        _y_oof = np.zeros(y.shape)

        for i_fold, (i_train, i_val) in iterator:

            x_train, x_val = x[i_train], x[i_val]
            y_train, y_val = y[i_train], y[i_val]
            model = self.fit_fold(x_train, y_train, x_val, y_val, i_fold, **params)
            self.models.append(model)
            _y_oof[i_val] = model.predict(x_val)
        _score = colwise_spearmanr(y, _y_oof, df_y.columns)
        self.y_oof = pd.DataFrame(_y_oof, index=df_y.index, columns=df_y.columns)
        self.score = pd.Series(_score)            
        self.save()

        return self
    
    def predict(self, df_x: pd.DataFrame) -> pd.DataFrame:
        x = df_x.values
        y = np.average([
            m.predict(x) for m in self.models
        ], axis=0)
        return pd.DataFrame(y, index=df_x.index, columns=target_cols)
    
    def save(self, dist: str = '.'):
        for i_fold, m in enumerate(self.models):
            self.save_fold(m, i_fold, dist)
        self.y_oof.to_csv('{0}/{1}.y_oof.csv'.format(dist, self.name), index=False)
        self.score.to_csv('{0}/{1}.score.csv'.format(dist, self.name), header=True)
    
    def load(self, src: str = '.'):
        self.models = []
        for i_fold in range(self.n_splits):
            self.models.append(self.load_fold(i_fold, src))
        self.y_oof = pd.read_csv('{0}/{1}.y_oof.csv'.format(src, self.name))
        self.score = pd.read_csv('{0}/{1}.score.csv'.format(src, self.name), index_col=0).iloc[:, 0]
        return self


# In[ ]:


class ProgressBar(Callback):
    
    def __init__(self, pbar):
        self.pbar = pbar
        
    def on_epoch_end(self, epoch, logs={}):
        self.pbar.update()
        self.pbar.set_postfix(logs)


# ### Dense NN Estimator

# In[ ]:


class DenseECV(BaseEnsembleCV):
    
    def __init__(self, name: str = 'dense_ecv'):
        super().__init__(name=name)
    
    def fit_fold(self, x_train, y_train, x_val, y_val, i_fold: int, **in_params):
        default_parmas = dict(
            lr=0.0001,
            epochs=50,
            batch_size=32,
        )
        params = {**default_parmas, **in_params}
        K.clear_session()
        model = Sequential([
            Dense(512, input_shape=(x_train.shape[1],)),
            Activation('relu'),
            Dropout(0.2),
            Dense(y_train.shape[1]),
            Activation('sigmoid'),
        ])
        es = EarlyStopping(patience=5)
        model.compile(
            optimizer=Adam(lr=params['lr']),
            loss='binary_crossentropy',
        )
        with tqdm(desc='k-fold {0}/{1}'.format(i_fold+1, self.n_splits), total=params['epochs']) as pbar:
            model.fit(
                x_train, y_train,
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                validation_data=(x_val, y_val),
                callbacks=[es, ProgressBar(pbar)],
                verbose=0,
            )
        return model
    
    def save_fold(self, model, i_fold, dist):
        model.save('{0}/{1}.model_{2}.h5'.format(dist, self.name, i_fold))
    
    def load_fold(self, i_fold, src):
        return load_model('{0}/{1}.model_{2}.h5'.format(src, self.name, i_fold))


# ### MultiTask ElasticNet Estimatorhistory

# In[ ]:


class ElasticNetECV(BaseEnsembleCV):
    
    def __init__(self, verbose: bool = True, name: str = 'elasticnet_ecv'):
        super().__init__(verbose=verbose, name=name)
        
    def fit_fold(self, x_train, y_train, *_, **in_params):
        default_params = dict(
            alpha=0.001,
            l1_ratio=0.5,
            selection='random',
        )
        params = {**default_params, **in_params}
        model = MultiTaskElasticNet(random_state=31, **params)
        return model.fit(x_train, y_train)
    
    def save_fold(self, model, i_fold, dist):
        joblib.dump(model, '{0}/{1}.model_{2}.joblib'.format(dist, self.name, i_fold))
    
    def load_fold(self, i_fold, src):
        return joblib.load('{0}/{1}.model_{2}.joblib'.format(src, self.name, i_fold))


# ### BERT Fine Tuning Estimator

# In[ ]:


class BertFineTuningECV(BaseEnsembleCV):
    
    def __init__(self, pretrained_path: str, seq_len: int = 512, name: str = 'bert_ecv'):
        super().__init__(name=name)
        self.pretrained_path = pretrained_path
        self.seq_len = seq_len
    
    def build_model(self, params) -> Model:
        s = self.seq_len
        with tf.device('/cpu:0'):  # avoid OOM error
            x = Input(self.seq_len * 3, dtype=tf.int32, name='x')
            input_word_ids = Lambda(lambda x: x[:, :s], output_shape=(s,))(x)
            attention_mask = Lambda(lambda x: x[:, s:s*2], output_shape=(s,))(x)
            token_type_ids = Lambda(lambda x: x[:, s*2:], output_shape=(s,))(x)
            bert_layer = TFBertModel.from_pretrained(self.pretrained_path)
            last_hidden_layer, _ = bert_layer([input_word_ids, attention_mask, token_type_ids])
            pooled = GlobalAveragePooling1D()(last_hidden_layer)
            pooled = Dropout(0.2)(pooled)
            y = Dense(30, activation='sigmoid', name='output')(pooled)
            model = Model(inputs=x, outputs=y)
        model.compile(
            optimizer=Adam(lr=params['lr']),
            loss='binary_crossentropy',
        )
        return model

    def fit_fold(self, x_train, y_train, x_val, y_val, i_fold, **in_params):
        default_params = dict(
            lr=0.00003,
            epochs=5,
            batch_size=8,
        )
        params = {**default_params, **in_params}
        es = EarlyStopping(patience=5)
        cp = ModelCheckpoint('{0}.model_{1}.h5'.format(self.name, i_fold), save_best_only=True, save_weights_only=True)

        K.clear_session()        
        model = self.build_model(params)
        h = model.fit(
            x_train, y_train,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            validation_data=(x_val, y_val),
            callbacks=[es, cp],
            verbose=1,
        )
        pd.DataFrame(h.history).to_csv('{0}_history_{1}.csv'.format(self.name, i_fold), index=False)
        return model
    
    def save_fold(self, model, i_fold, dist):
        return
        
    def load_fold(self, i_fold, src):
        K.clear_session()
        model = self.build_model(dict(lr=0.00003))
        model.load_weights('{0}/{1}.model_{2}.h5'.format(src, self.name, i_fold))
        return model


# # Pipeline

# In[ ]:


use_model = tensorflow_hub.load('../input/universalsentenceencoderlarge4')
bert_tokenizer = BertTokenizer.from_pretrained('../input/transformers/bert-base-uncased')


# In[ ]:


use_preprocess = Pipeline(steps=[
    
    # Universal Sentence Encoder
    ('USE_question_title', UniversalSentenceEncoderEncoder('question_title', use_model)),
    ('USE_question_body', UniversalSentenceEncoderEncoder('question_body', use_model)),
    ('USE_answer', UniversalSentenceEncoderEncoder('answer', use_model)),
    
    # distance
    ('distance_use_question_title-question_body', DistanceEngineerer('question_title_use', 'question_body_use')),
    ('distance_use_question_title-answer', DistanceEngineerer('question_title_use', 'answer_use')),
    ('distance_use_question_body-answer', DistanceEngineerer('question_body_use', 'answer_use')),
    
    # one-hot encode & drop columns
    ('onehost_encode_and_drop_columns', Pipeline(steps=[

        # abc.example.com -> example.com
        ('extrace_sld', ColumnTransformer({
            'host': SecondLevelDomainExtracter(),
        })),

        # one-hot encode
        ('onehot_encode_host', WrappedOneHotEncoder('host')),
        ('onehot_encode_category', WrappedOneHotEncoder('category')),

    ]).fit(features_train)),
    
    ('drop_columns', ColumnDropper([
        'qa_id', 'category', 'host', 'question_title', 'question_body', 'question_user_name', 'question_user_page',
        'answer', 'answer_user_name', 'answer_user_page', 'url',
    ])),
])

use_densenn = Pipeline(steps=[
    ('preprocess', use_preprocess),
    ('estimate', DenseECV()),
])

use_elasticnet = Pipeline(steps=[    
    ('preprocess', use_preprocess),
    ('estimate', ElasticNetECV())
])

bert_finetuning = Pipeline(steps=[    
    ('tokenize', QATokenizer(bert_tokenizer)),
    ('estimate', BertFineTuningECV('../input/transformers/bert-base-uncased')),
])


# In[ ]:


question_title = features_train['question_title']


# # BERT Fine Tuning

# In[ ]:


# _ = bert_finetuning.fit(features_train, targets_train, estimate__groups=question_title)
_ = bert_finetuning['estimate'].load('../input/google-quest-challenge-trained-models')


# In[ ]:


bert_finetuning['estimate'].score.mean()


# # UniversalSentenceEncder -> Dense NN

# In[ ]:


# _ = use_densenn.fit(features_train, targets_train, estimate__groups=question_title)
_ = use_densenn['estimate'].load('../input/google-quest-challenge-trained-models')


# In[ ]:


use_densenn['estimate'].score.mean()


# # UniversalSentenceEncder -> ElasticNet

# In[ ]:


# _ = use_elasticnet.fit(features_train, targets_train, estimate__groups=question_title)
_ = use_elasticnet['estimate'].load('../input/google-quest-challenge-trained-models')


# In[ ]:


use_elasticnet['estimate'].score.mean()


# # Ensemble

# In[ ]:


average_spearmanr(
    pd_average([
        bert_finetuning['estimate'].y_oof,
        use_densenn['estimate'].y_oof,
        use_elasticnet['estimate'].y_oof,        
    ]).values,
    targets_train.values
)


# In[ ]:


average_spearmanr(
    pd_rank_average([
        bert_finetuning['estimate'].y_oof,
        use_densenn['estimate'].y_oof,
        use_elasticnet['estimate'].y_oof,        
    ]).values,
    targets_train.values
)


# # Prediction

# In[ ]:


def to_submission(y: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([features_test.loc[:, 'qa_id'], y], axis=1)


# In[ ]:


WrappedMinMaxScaler().fit_transform(pd_average([
    bert_finetuning.predict(features_test),
    use_densenn.predict(features_test),
    use_elasticnet.predict(features_test),
])).pipe(to_submission).to_csv('submission.csv', index=False)

