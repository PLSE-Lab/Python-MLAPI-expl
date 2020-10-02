#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# # For Kernal Mode
get_ipython().system('pip install -q ../input/tensorflow-determinism')
get_ipython().system('pip install -q ../input/huggingfacetokenizers/tokenizers-0.0.11-cp36-cp36m-manylinux1_x86_64.whl')
# !pip install -q ../input/sacremoses
get_ipython().system('pip uninstall --yes pytorch-transformers')
get_ipython().system('pip install -q ../input/huggingface-transformers-master')


# In[ ]:





# In[ ]:


import pandas as pd
import numpy as np
import random
import random, math, time
import os, sys
from pathlib import Path
import pickle

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import bisect

import matplotlib.pyplot as plt
from tqdm import tqdm
# from tqdm.notebook import tqdm

import tensorflow as tf
import tensorflow.keras.backend as K
# https://github.com/NVIDIA/tensorflow-determinism
os.environ['TF_DETERMINISTIC_OPS'] = '1' # TF 2.1
# from tfdeterminism import patch
# patch()

import transformers
from transformers import *

import torch

from scipy.stats import spearmanr
from math import floor, ceil

from bs4 import BeautifulSoup

import gc
gc.enable()

np.set_printoptions(suppress=True)
print('Tensorflow version', tf.__version__)

print('PyTorch version', torch.__version__)

print('Transformers version',
      transformers.__version__)  # Current version: 2.3.0


# In[ ]:


# https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

# tf.config.gpu.set_per_process_memory_fraction(0.85)
# tf.config.gpu.set_per_process_memory_growth(True)

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.2
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)


# In[ ]:


import os
os.listdir("../input")


# In[ ]:





# In[ ]:


rand_seed = 20201120
n_splits = 5

# BERT_PATH = "/workspace/Kaggle/QA/pretrained_models/"
# dataset_folder = Path("/workspace/Kaggle/QA/")
# MODEL_PATH_list = [
#     "/workspace/Kaggle/QA/completed/tf-roberta-base-exp-v7/",
#     "/workspace/Kaggle/QA/completed/bert-base-uncased-exp-v4/",
#     "/workspace/Kaggle/QA/completed/tf-bert-base-cased-exp-v4/",
#     "/workspace/Kaggle/QA/completed/tf-roberta-base-exp-v4/",
#     "/workspace/Kaggle/QA/completed/xlnet-base-cased-exp-v7/",
# ]

BERT_PATH = "../input/"
dataset_folder = Path("../input/google-quest-challenge")
MODEL_PATH_list = [
    "../input/tf-roberta-base-exp-v7/",
    "../input/bert-base-uncased-exp-v4/",
    "../input/tf-bert-base-cased-exp-v4/",
    "../input/tf-roberta-base-exp-v4/",
    "../input/xlnet-base-cased-exp-v7/",
]

# pretrained_model_metadata = [
#     # (pretrained_model_name, is_tf, infer_batch_size, cate_embed_mode)
#     ("tf-roberta-base", True, 32, True),
#     ("bert-base-uncased", True, 32, False),
#     ("tf-bert-base-cased", True, 32, False),
#     ("tf-roberta-base", True, 32, False),
#     ("xlnet-base-cased", True, 32, True),
# ]

pretrained_model_metadata = [
    # (pretrained_model_name, is_tf, infer_batch_size, cate_embed_mode)
    ("tf-roberta-base", True, 56, True),
    ("bert-base-uncased", True, 56, False),
    ("tf-bert-base-cased", True, 56, False),
    ("tf-roberta-base", True, 56, False),
    ("xlnet-base-cased", True, 48, True),
]

model_filename_prefix_list = [
    "tf-roberta-base_exp_cate_embed",
    "bert-base-uncased_exp_split_dense",
    "tf-bert-base-cased_exp_split_dense",
    "tf-roberta-base_exp_split_dense",
    "xlnet-base-cased_exp_cate_embed",
]

MAX_SEQUENCE_LENGTH = 512
max_title_length = 100

# learning_rate = 2e-5
# embeddings_dropout = 0.2
# dense_dropout = 0.1


# In[ ]:


for i, p in enumerate(MODEL_PATH_list):
    prefix = model_filename_prefix_list[i]
    for f in os.listdir(p):
        if f != "dataset-metadata.json":
            print(p+f)
            assert prefix in f


# In[ ]:


df_train = pd.read_csv(dataset_folder / 'train.csv')
df_test = pd.read_csv(dataset_folder / 'test.csv')
df_sub = pd.read_csv(dataset_folder / 'sample_submission.csv')
print('Train shape:', df_train.shape)
print('Test shape:', df_test.shape)


# In[ ]:


output_categories = list(df_train.columns[11:])
# Select only question title, body and answer
input_categories = list(df_train.columns[[1, 2, 5]])

print('\nOutput categories:\n', output_categories)
print('\nInput categories:\n', input_categories)


# In[ ]:





# In[ ]:


# Extract domain
def extract_netloc(x):
    tokens = x.split(".")
    if len(tokens) > 3:
        print(x)
        return ".".join(tokens[:2])
        # looks like meta is a special site, we should keep it
        # https://stackoverflow.com/help/whats-meta
        # the part of the site where users discuss the workings and policies of Stack Overflow rather than discussing programming itself.
        # return tokens[1]
    else:
        return tokens[0]


# TODO: test it
# df_train['netloc'] = df_train['host'].apply(
#     lambda x: extract_netloc(x))
# df_test['netloc'] = df_test['host'].apply(
#     lambda x: extract_netloc(x))

df_train['netloc'] = df_train['host'].apply(lambda x: x.split(".")[0])
df_test['netloc'] = df_test['host'].apply(lambda x: x.split(".")[0])


# In[ ]:





# In[ ]:


def set_all_seeds(rand_seed):
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    os.environ['PYTHONHASHSEED'] = str(rand_seed)
    
    # TF 2.0
    tf.random.set_seed(rand_seed)
    
    # PyTorch
    torch.manual_seed(rand_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# In[ ]:


set_all_seeds(rand_seed)


# In[ ]:





# In[ ]:


# Redirect outputs to console
# import sys
# jupyter_console = sys.stdout
# sys.stdout = open('/dev/stdout', 'w')

# Append to log file
# sys.stdout = open(f"stdout.log", 'a')
# sys.stdout = jupyter_console


# ## Preprocessing Utilities

# In[ ]:


def _convert_to_transformer_inputs(title, question, answer, tokenizer,
                                   max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""
    def return_id(str1, str2, truncation_strategy, length):

        inputs = tokenizer.encode_plus(str1,
                                       str2,
                                       add_special_tokens=True,
                                       max_length=length,
                                       truncation_strategy=truncation_strategy)

        input_ids = inputs["input_ids"]
        input_masks = [1] * len(input_ids)
        input_segments = inputs["token_type_ids"]
        padding_length = length - len(input_ids)
        padding_id = tokenizer.pad_token_id
        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)

        return [input_ids, input_masks, input_segments]
    
    def remove_html_special_symbols(x):
        html_entities = [
            ("&quot;", "\""),
            ("&num;", "#"),
            ("&dollar;", "$"),
            ("&percnt;", "%"),
            ("&amp;", "&"),
            ("&apos;", "'"),
            ("&lpar;", "("),
            ("&rpar;", ")"),
            ("&ast;", "*"),
            ("&plus;", "+"),
            ("&comma;", ","),
            ("&minus;", "-"),
            ("&period;", "."),
            ("&sol;", "/"),
            ("&colon;", ":"),
            ("&semi;", ";"),
            ("&lt;", "<"),
            ("&equals;", "="),
            ("&gt;", ">"),
            ("&quest;", "?"),
            ("&commat;", "@"),
            ("&lsqb;", "["),
            ("&bsol;", "\\"),
            ("&rsqb;", "]"),
            ("&Hat;", "^"),
            ("&lowbar;", "_"),
            ("&grave;", "`"),
            ("&lcub;", "{"),
            ("&verbar;", "|"),
            ("&rcub;", "}"),
            # ("", ""),
        ]
        for (k, v) in html_entities:
            x = str(x.replace(k, v))
        return x

    def remove_latex_and_code_tokens(tokens):
        return [
            x for x in tokens if not (x.startswith("$") or x.startswith("\\"))
        ]

    # Remove extra spaces
    title = remove_html_special_symbols(" ".join(
        remove_latex_and_code_tokens(str(title).split()))).strip()
    question = remove_html_special_symbols(" ".join(
        remove_latex_and_code_tokens(str(question).split()))).strip()
    answer = remove_html_special_symbols(" ".join(
        remove_latex_and_code_tokens(str(answer).split()))).strip()

    # Extract plain text from html
    try:
        soup_q = BeautifulSoup(question)
        question = soup_q.get_text()
    except Exception as e:
        print(e)
        pass

    try:
        soup_a = BeautifulSoup(answer)
        answer = soup_a.get_text()
    except Exception as e:
        print(e)
        pass

    input_ids_q, input_masks_q, input_segments_q = return_id(
        "[CLS] " + title[:max_title_length] + " [SEP] " + question + " [SEP]", None,
        'longest_first', max_sequence_length)

    input_ids_a, input_masks_a, input_segments_a = return_id(
        "[CLS] " + answer + " [SEP]", None, 'longest_first', max_sequence_length)

    return [
        input_ids_q, input_masks_q, input_segments_q, input_ids_a,
        input_masks_a, input_segments_a
    ]


# In[ ]:


def compute_input_arrays(df, columns, tokenizer, max_sequence_length):
    input_ids_q, input_masks_q, input_segments_q = [], [], []
    input_ids_a, input_masks_a, input_segments_a = [], [], []
    for _, instance in tqdm(df[columns].iterrows()):
        t, q, a = instance.question_title, instance.question_body, instance.answer

        ids_q, masks_q, segments_q, ids_a, masks_a, segments_a =         _convert_to_transformer_inputs(t, q, a, tokenizer, max_sequence_length)

        input_ids_q.append(ids_q)
        input_masks_q.append(masks_q)
        input_segments_q.append(segments_q)

        input_ids_a.append(ids_a)
        input_masks_a.append(masks_a)
        input_segments_a.append(segments_a)

    return [
        np.asarray(input_ids_q, dtype=np.int32),
        np.asarray(input_masks_q, dtype=np.int32),
        np.asarray(input_segments_q, dtype=np.int32),
        np.asarray(input_ids_a, dtype=np.int32),
        np.asarray(input_masks_a, dtype=np.int32),
        np.asarray(input_segments_a, dtype=np.int32)
    ]


def compute_output_arrays(df, columns):
    return np.asarray(df[columns])


# In[ ]:


def compute_spearmanr_ignore_nan(trues, preds):
    rhos = []
    for tcol, pcol in zip(np.transpose(trues), np.transpose(preds)):
        rhos.append(spearmanr(tcol, pcol).correlation)
    return np.nanmean(rhos)

def compute_spearmanr(trues, preds):
    rhos = []
    for tcol, pcol in zip(np.transpose(trues), np.transpose(preds)):
        rhos.append(spearmanr(tcol, pcol).correlation)
    return np.mean(rhos)


# In[ ]:


class SpearmanMonitorCallback(tf.keras.callbacks.Callback):
    def __init__(self, valid_data, batch_size=16, fold=None):
        self.valid_inputs = valid_data[0]
        self.valid_outputs = valid_data[1]

        self.batch_size = batch_size
        self.fold = fold

    def on_train_begin(self, logs={}):
        self.valid_predictions = []

    def on_epoch_end(self, epoch, logs={}):
        self.valid_predictions.append(
            self.model.predict(self.valid_inputs, batch_size=self.batch_size))

        rho_val = compute_spearmanr(
            self.valid_outputs, np.average(self.valid_predictions, axis=0))

        print(f" Fold {self.fold+1} Validation Score: {rho_val:.6f}")


# In[ ]:





# ## Create Custom Model

# In[ ]:


def create_model(pretrained_model_name):
    q_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH, ), dtype=tf.int32)
    a_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH, ), dtype=tf.int32)

    q_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH, ), dtype=tf.int32)
    a_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH, ), dtype=tf.int32)

    q_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH, ), dtype=tf.int32)
    a_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH, ), dtype=tf.int32)

    pretrained_model = model_class.from_pretrained(BERT_PATH +
                                                   f"{pretrained_model_name}")

    q_embedding = pretrained_model(q_id,
                                   attention_mask=q_mask,
                                   token_type_ids=q_atn)[0]
    a_embedding = pretrained_model(a_id,
                                   attention_mask=a_mask,
                                   token_type_ids=a_atn)[0]

#     q_embedding = tf.keras.layers.SpatialDropout1D(embeddings_dropout)(
#         q_embedding)
#     a_embedding = tf.keras.layers.SpatialDropout1D(embeddings_dropout)(
#         a_embedding)


    # Get CLS token output
    q = q_embedding[:, 0, :]
    a = a_embedding[:, 0, :]

    q = tf.keras.layers.Dense(256, activation='relu')(q)
    a = tf.keras.layers.Dense(256, activation='relu')(a)
    
    # TODO: Test dense dropout
    # q = tf.keras.layers.Dropout(dense_dropout)(q)
    # a = tf.keras.layers.Dropout(dense_dropout)(a)

    q = tf.keras.layers.Dense(21, activation='sigmoid')(q)
    a = tf.keras.layers.Dense(9, activation='sigmoid')(a)

    x = tf.keras.layers.Concatenate()([q, a])

    model = tf.keras.models.Model(inputs=[
        q_id,
        q_mask,
        q_atn,
        a_id,
        a_mask,
        a_atn,
    ],
                                  outputs=x)

    return model, pretrained_model


# In[ ]:


def create_model_cate_embed(pretrained_model_name, embed_info):
    q_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH, ), dtype=tf.int32)
    a_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH, ), dtype=tf.int32)

    q_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH, ), dtype=tf.int32)
    a_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH, ), dtype=tf.int32)

    q_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH, ), dtype=tf.int32)
    a_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH, ), dtype=tf.int32)

    pretrained_model = model_class.from_pretrained(BERT_PATH +
                                                   f"{pretrained_model_name}")

    # Get last hidden-state from 1st element of output
    if "xlnet" in pretrained_model_name:
        q_embedding = pretrained_model(q_id,
                                       attention_mask=q_mask,
                                       token_type_ids=q_atn)[0]
        a_embedding = pretrained_model(a_id,
                                       attention_mask=a_mask,
                                       token_type_ids=a_atn)[0]
    else:
        q_embedding, q_pooler_output = pretrained_model(q_id,
                                                        attention_mask=q_mask,
                                                        token_type_ids=q_atn)
        a_embedding, a_pooler_output = pretrained_model(a_id,
                                                        attention_mask=a_mask,
                                                        token_type_ids=a_atn)

    # Get CLS token output
    q = q_embedding[:, 0, :]
    a = a_embedding[:, 0, :]

    host_input = tf.keras.Input(shape=(1, ), name="host_input")
    netloc_input = tf.keras.Input(shape=(1, ), name="netloc_input")
    cate_input = tf.keras.Input(shape=(1, ), name="category_input")

    host_embed_info = embed_info["host"]
    host_embed = tf.keras.layers.Embedding(input_dim=host_embed_info[0],
                                           output_dim=host_embed_info[1],
                                           input_length=(1, ))(host_input)

    netloc_embed_info = embed_info["netloc"]
    netloc_embed = tf.keras.layers.Embedding(input_dim=netloc_embed_info[0],
                                             output_dim=netloc_embed_info[1],
                                             input_length=(1, ))(netloc_input)

    cate_embed_info = embed_info["category"]
    cate_embed = tf.keras.layers.Embedding(input_dim=cate_embed_info[0],
                                           output_dim=cate_embed_info[1],
                                           input_length=(1, ))(cate_input)

    host_embed = tf.keras.layers.Reshape(
        target_shape=(host_embed_info[1], ))(host_embed)
    netloc_embed = tf.keras.layers.Reshape(
        target_shape=(netloc_embed_info[1], ))(netloc_embed)
    cate_embed = tf.keras.layers.Reshape(
        target_shape=(cate_embed_info[1], ))(cate_embed)

    embed_concat = tf.keras.layers.Concatenate()(
        [host_embed, netloc_embed, cate_embed])
    embed_concat = tf.keras.layers.Dense(128, activation='relu')(embed_concat)

    # Concatenation
    q_concat = tf.keras.layers.Concatenate()([q, embed_concat])
    # q_concat = tf.keras.layers.Concatenate()([q, host_embed, cate_embed, q_pooler_output])
    q_concat = tf.keras.layers.Dense(256, activation='relu')(q_concat)

    a_concat = tf.keras.layers.Concatenate()([a, embed_concat])
    # a_concat = tf.keras.layers.Concatenate()([a, host_embed, cate_embed, a_pooler_output])
    a_concat = tf.keras.layers.Dense(256, activation='relu')(a_concat)

    # Dense dropout
    # q_concat = tf.keras.layers.Dropout(dense_dropout)(q_concat)
    # a_concat = tf.keras.layers.Dropout(dense_dropout)(a_concat)

    # Use sigmoid for multi-label predictions
    q_concat = tf.keras.layers.Dense(21, activation='sigmoid')(q_concat)
    a_concat = tf.keras.layers.Dense(9, activation='sigmoid')(a_concat)

    x = tf.keras.layers.Concatenate()([q_concat, a_concat])

    model = tf.keras.models.Model(inputs=[
        q_id, q_mask, q_atn, a_id, a_mask, a_atn, host_input, netloc_input,
        cate_input
    ],
                                  outputs=x)

    return model, pretrained_model


# In[ ]:





# ### Split K-Folds by Unique Group

# In[ ]:


set_all_seeds(rand_seed)
gkf = GroupKFold(n_splits=n_splits).split(X=df_train.question_body,
                                          groups=df_train.question_body)
gkf = list(gkf)
len(gkf)


# In[ ]:





# ## Do Inference

# In[ ]:


outputs = compute_output_arrays(df_train, output_categories)


# In[ ]:


def optimize_ranks(preds, unique_labels):
    new_preds = np.zeros(preds.shape)
    for i in range(preds.shape[1]):
        interpolate_bins = np.digitize(preds[:, i],
                                       bins=unique_labels,
                                       right=False)
        
        if len(np.unique(interpolate_bins)) == 1:
            # Use original preds
            new_preds[:, i] = preds[:, i]
        else:
            new_preds[:, i] = unique_labels[interpolate_bins]

    return new_preds


# In[ ]:


y_labels = df_train[output_categories].copy()
y_labels = y_labels.values.flatten()
unique_labels = np.array(sorted(np.unique(y_labels)))
unique_labels


# In[ ]:


denominator = 60
q = np.arange(0, 101, 100 / denominator)
exp_labels = np.percentile(unique_labels, q)
exp_labels


# In[ ]:


infer_start_time = time.time()

all_test_preds = []
all_val_preds = []
all_val_scores = []
all_magic_val_scores = []

gc.collect()

for k, MODEL_PATH in enumerate(MODEL_PATH_list):
#     if k < 2:
#         continue
    
    pretrained_model_name, is_tf, infer_batch_size, cate_embed_mode = pretrained_model_metadata[k]
    model_filename_prefix = model_filename_prefix_list[k]

    print(f"Generating prediction from fine-tuned {pretrained_model_name} ......")
    
    preprocessing_start = time.time()
    if is_tf:
        model_class = TFAutoModel
        tokenizer_class = AutoTokenizer
    else:
        model_class = AutoModel
        tokenizer_class = AutoTokenizer
    tokenizer = tokenizer_class.from_pretrained(BERT_PATH +
                                                f"{pretrained_model_name}")
    inputs = compute_input_arrays(df_train, input_categories, tokenizer,
                                  MAX_SEQUENCE_LENGTH)
    test_inputs = compute_input_arrays(df_test, input_categories,
                                       tokenizer, MAX_SEQUENCE_LENGTH)
    print(f"Time spent on preprocessing: {(time.time()-preprocessing_start)/60:,.2f} minutes")

    test_preds = []
    val_preds = []
    val_scores = []
    magic_val_scores = []

    for i, (train_idx, valid_idx) in enumerate(gkf):
        set_all_seeds(rand_seed)

        print(f"Using {pretrained_model_name} Fold {i+1} {pretrained_model_name} for inference ......")

        fold_start = time.time()

        # Generate validation score
        # TODO: remove this part later
        valid_inputs = [inputs[i][valid_idx] for i in range(len(inputs))]
        valid_outputs = outputs[valid_idx]
        
        if cate_embed_mode:
            # Extra categorical embeddings
            embed_info = {}
            category_features = {}
            def extract_category_ids(train, val, test, c, info):
                le = LabelEncoder()
                le.fit(train[c])
                # Set unknonwn category
                val[c] = val[c].map(lambda s: '<unknown>'
                                    if s not in le.classes_ else s)
                test[c] = test[c].map(lambda s: '<unknown>'
                                      if s not in le.classes_ else s)

                le_classes = le.classes_.tolist()
                bisect.insort_left(le_classes, '<unknown>')
                le.classes_ = le_classes

                train[c + "_label"] = le.transform(train[c])
                val[c + "_label"] = le.transform(val[c])
                test[c + "_label"] = le.transform(test[c])

                no_of_unique_cat = train[c + "_label"].nunique()
                embedding_size = min(np.ceil((no_of_unique_cat) / 2), 50)
                embedding_size = int(embedding_size)
                vocab_size = no_of_unique_cat + 1
                info[c] = (vocab_size, embedding_size)

                print(f"Extracted (vocab_size, embedding_size) for {c}: ({vocab_size}, {embedding_size})")

                return val[c + "_label"], test[c + "_label"]

            host_val, host_test = extract_category_ids(df_train.iloc[train_idx, :].copy(),
                                                       df_train.iloc[valid_idx, :].copy(),
                                                       df_test.copy(),
                                                       "host",
                                                       embed_info)
            netloc_val, netloc_test = extract_category_ids(df_train.iloc[train_idx, :].copy(),
                                                           df_train.iloc[valid_idx, :].copy(),
                                                           df_test.copy(),
                                                           "netloc",
                                                           embed_info)
            cate_val, cate_test = extract_category_ids(df_train.iloc[train_idx, :].copy(),
                                                       df_train.iloc[valid_idx, :].copy(),
                                                       df_test.copy(),
                                                       "category",
                                                       embed_info)

            valid_inputs.append(host_val)
            valid_inputs.append(netloc_val)
            valid_inputs.append(cate_val)

            # Copy test_inputs
            submit_inputs = [np.copy(x) for x in test_inputs]
            submit_inputs.append(host_test)
            submit_inputs.append(netloc_test)
            submit_inputs.append(cate_test)
        
        print("Cleaning session ...")
        K.clear_session()    

        print("Loading pretrained model and weights ...")
        if cate_embed_mode:
            model, pretrained_model = create_model_cate_embed(pretrained_model_name, embed_info)
        else:
            model, pretrained_model = create_model(pretrained_model_name)
            
        model_filename = f"{model_filename_prefix}_fold{i+1}.h5"
        # Load fine-tuned weights
        model.load_weights(MODEL_PATH + model_filename)
        
        fold_val_preds = model.predict(valid_inputs,
                                       batch_size=infer_batch_size)
        rho_val = compute_spearmanr(valid_outputs, fold_val_preds)
        print(f"Fold {i+1} Validation Score: {rho_val:.6f}")
        val_preds.append(fold_val_preds)
        val_scores.append(rho_val)
        
        val_magic_preds = optimize_ranks(fold_val_preds, exp_labels)
        magic_rho_val = compute_spearmanr(valid_outputs, val_magic_preds)
        print(f"Fold {i+1} Magic Validation Score: {magic_rho_val:.6f}")
        magic_val_scores.append(magic_rho_val)

        # Generate test predictions
        if cate_embed_mode:
            test_preds.append(model.predict(submit_inputs,
                                            batch_size=infer_batch_size))
        else:
            test_preds.append(model.predict(test_inputs,
                                            batch_size=infer_batch_size))

        print(f"Time spent on fold {i+1} inference: {(time.time()-fold_start)/60:,.2f} minutes")

        del model, pretrained_model, valid_inputs, valid_outputs, fold_val_preds, rho_val
        gc.collect()

    all_test_preds.append(test_preds)
    all_val_preds.append(val_preds)
    all_val_scores.append(val_scores)
    all_magic_val_scores.append(magic_val_scores)
    
    del tokenizer, inputs, test_inputs
    gc.collect()
    
print(f"Time spent on ensemble inference: {(time.time()-infer_start_time)/60:,.2f} minutes")


# In[ ]:


print(f"Mean Validation Score: {np.mean(all_val_scores):.6f}")
print(f"Mean Magic Validation Score: {np.mean(all_magic_val_scores):.6f}")


# In[ ]:


# Mean Validation Score: 0.396150
# Mean Magic Validation Score: 0.417236


# In[ ]:





# ## Check Ensemble Validation Score

# In[ ]:


def val_ensemble_preds(all_val_preds, weights):
    oof_preds = np.zeros(outputs.shape)
    for i, model_preds in enumerate(all_val_preds):
        for j, (train_idx, valid_idx) in enumerate(gkf):
            tmp = np.vstack(model_preds[j])
            oof_preds[valid_idx] += tmp * weights[i]

    oof_preds /= np.sum(weights)

    return oof_preds


# In[ ]:


with open('ensemble-models-v4-v7.pickle', 'wb') as handle:
    pickle.dump(all_val_preds, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:





# In[ ]:


weights = [1.0, 1.0, 1.0, 1.0, 1.0]
oof_preds = val_ensemble_preds(all_val_preds, weights)
magic_preds = optimize_ranks(oof_preds, exp_labels)
blend_score = compute_spearmanr(outputs, magic_preds)
print(weights, blend_score)

weights = [2.0, 1.0, 1.0, 1.0, 2.0]
oof_preds = val_ensemble_preds(all_val_preds, weights)
magic_preds = optimize_ranks(oof_preds, exp_labels)
blend_score = compute_spearmanr(outputs, magic_preds)
print(weights, blend_score)

weights = [2.0, 1.0, 1.0, 1.0, 1.5]
oof_preds = val_ensemble_preds(all_val_preds, weights)
magic_preds = optimize_ranks(oof_preds, exp_labels)
blend_score = compute_spearmanr(outputs, magic_preds)
print(weights, blend_score)

weights = [1.5, 1.0, 1.0, 1.0, 1.5]
oof_preds = val_ensemble_preds(all_val_preds, weights)
magic_preds = optimize_ranks(oof_preds, exp_labels)
blend_score = compute_spearmanr(outputs, magic_preds)
print(weights, blend_score)


# In[ ]:





# ## Generate Submission File

# In[ ]:


submit_preds = [np.average(x, axis=0) for x in all_test_preds]


# In[ ]:


submit_preds = np.average(submit_preds, weights = [2.0, 1.0, 1.0, 1.0, 1.5],
                          axis=0)  # for weighted average set weights=[...]


# In[ ]:


# Optimize ranks
submit_preds = optimize_ranks(submit_preds, exp_labels)


# In[ ]:


df_sub.iloc[:, 1:] = submit_preds
df_sub.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:




