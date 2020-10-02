#!/usr/bin/env python
# coding: utf-8

# ## About

# This is the kernel (inference only) used in Google QUEST competition. This kernel scored in top 2% of participats, at 24th place out of 1571.
# 
# The [competition's](https://www.kaggle.com/c/google-quest-challenge) goal was to rate question-answer pairs on 30 measures. Question-answer pairs where scrapped from popular websites such as stackoverflow. The 30 measures where highly absract in nature e.g. 'answer_helpful' and where loosely defined so that any lay person may intuitevely evaluate question-answer pairs on all of the measures. The training data was crowdsourced and consisted of about 5000 samples.
# 
# **Key ideas:**
# 
# **Pre-processing**
# * Prepare 2 sets of tokenized inputs. First set containing only question title and question body, second set containing all text data (question title, body and answer)
# 
# **Modelling**
# * Train one model per set of inputs and outputs. For example, train model A on question title and body inputs to predict measures related only to a question (e.g. question_type_instruction). And then train model B on question title, body and answer to predict measures related to question-answer interaction and answer only (e.g. answer_helpful or answer_type_instruction)
# * Use models that utilise different approaches - different pre-trained transformers fitted to textual data (BERT, GPT2, XLNET) as well as decision trees fitted to text's meta-features (LGB, XGB)
# * Use cross-validation and one-cycle training policy for all models, but do that on another virtual machine to avoid submission kernels' time restrictions (2 hours for GPU kernels).
# 
# **Post-processing**
# * Weighting models' predictions individually for each of the response variables. Use softmaxed validation score of each of the models to determine its weight.
# * Dynamically round blended predictions to best fit each of the response variables. As a lot of the response variables had hard distributions, iteratively trying different rounding methods to achieve best validation score had significantly boosted leaderboard score.
# 
# **Acknowledgements**
# * This is long evolved descendant of akensert's public kernel, thanks to him for a good starting code
# * Huggingface transformers library authors for a super easy-to-use library for fitting such a variety of transformer models

# ## Modelling config

# **USE_SAMPLE** - to run quick experiments on small subsample of train data in interactive mode
# 
# **USE_ONLY_SELECTED_FOLDS** - some models where only trained on some of N CV folds e.g. first 3 folds of 10-fold CV to reduce inference time (only 2 hours allowed for GPU kernels)
# selected_folds = in case the parameter above is True, this parameter specifies which folds to use at inference
# 
# **WEIGHTS** - specifies method that will be used to blend models together. Can be 'softmax', 'dynamic' and 'dynamic softamx'. Softmax method uses softmaxed validation score of each model for each response variable (30 of them in total) as weight. The softmax uses beta coefficient (last parameter in the cell below). Beta affects how 'hard' blending weights are - high beta values lead to discounting models with bad validation scores (weights -> 0) while lower beta values lead to all models being included to some extent. Dynamic method just iteractes through a predifined weights permutations and takes whatever leads to the best validation score of belnded predictions. Dynamic softmax iterates through different beta coefficients for each response variable to find the one best suited for each of them.
# 
# **ROUNDa** - switch for whether to use dynamic rounding of predictions in post-processing
# 
# **USE_LGB** - whether to use light gradient boosting model (variant of gradient boosted decision trees) trained on a separate set of features (meta features about text like number of words)
# 
# **USE_XGB** - whether to use xgb (variant of gradient boosted decision trees) trained on the same set of features as LGB model
# 
# **USE_3SPLIT_BERT** - I have attempted to break response variables into 3 groups - group 1 of responses related only to a question, group 2 of responses related to interaction between question and answer and group 3 responses realated to answer only. I have then trained some models individually to predict each of those 3 groups. This, however, did not result an improvement in overall score, so I am not using those models here.
# 
# **USE_TEST_SAFETY_ADJUSTMENTS** - basically allows to throw exceptions whenever final submission predicitons are outside of allowed ranges e.g. more than 1 or less than 0 or have variance of 0.
# 
# **beta** - used for softmax model weighting method as described above

# In[ ]:


USE_SAMPLE = False

USE_ONLY_SELECTED_FOLDS = True
selected_folds = [0,1,2]

WEIGHTS = 'softmax'
ROUND = True
USE_LGB = True
USE_XGB = False
USE_3SPLIT_BERT = False
USE_TEST_SAFETY_ADJUSTMENTS = True

beta = 18.5


# ## Install packages for offline use & import of libraries

# In[ ]:


get_ipython().system('pip install ../input/sacremoses0038 > /dev/null')

import sys
sys.path.insert(0, "../input/tokenizers0011/")


# In[ ]:


get_ipython().system('pip install ../input/transformers241 > /dev/null --no-dependencies')


# In[ ]:


get_ipython().system('pip install ../input/fastparquet/fastparquet-0.3.2-cp36-cp36m-linux_x86_64.whl > /dev/null --no-dependencies')


# In[ ]:


get_ipython().system('pip install ../input/fastparquet/thrift-0.13.0-cp36-cp36m-linux_x86_64.whl > /dev/null --no-dependencies')


# In[ ]:


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import psutil
import gc
import tensorflow as tf
import tensorflow.keras.backend as K
import os
from scipy.stats import spearmanr
from math import floor, ceil
from transformers import *
import nltk

np.set_printoptions(suppress=True)
print(tf.__version__)

pd.set_option('display.max_colwidth', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

import matplotlib
import pickle
COLOR = 'black' #set to white for when working in dark mode
matplotlib.rcParams['text.color'] = COLOR
matplotlib.rcParams['axes.labelcolor'] = COLOR
matplotlib.rcParams['xtick.color'] = COLOR
matplotlib.rcParams['ytick.color'] = COLOR
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


# ## Support functions

# ### Tokenization functions

# In[ ]:


PATH = '../input/google-quest-challenge/'

BERT_PATH = '../input/bert-base-uncased-huggingface-transformer/'
GPT2_PATH = '../input/gpt2-hugginface-pretrained/'
XLNET_PATH = '../input/xlnet-huggingface-pretrained/'

MAX_SEQUENCE_LENGTH = 512

df_train = pd.read_csv(PATH+'train.csv')
df_test = pd.read_csv(PATH+'test.csv')
df_sub = pd.read_csv(PATH+'sample_submission.csv')
print('train shape =', df_train.shape)
print('test shape =', df_test.shape)

output_categories = list(df_train.columns[11:])
input_categories = list(df_train.columns[[1,2,5]])
print('\noutput categories:\n\t', output_categories)
print('\ninput categories:\n\t', input_categories)

if USE_SAMPLE:
    df_train = df_train.iloc[0:round(0.05*df_train.shape[0]),:]
    df_test = df_test.iloc[0:round(0.1*df_test.shape[0]),:]


# In[ ]:


def save_file(var, name):
    pickle.dump(var, open(f"/kaggle/working/{name}.p", "wb"))


# In[ ]:


def _convert_to_transformer_inputs(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""
    
    def return_id(str1, str2, truncation_strategy, length):

        inputs = tokenizer.encode_plus(str1, str2,
            add_special_tokens=True,
            max_length=length,
            truncation_strategy=truncation_strategy)
        
        input_ids =  inputs["input_ids"]
        input_masks = [1] * len(input_ids)
        input_segments = inputs["token_type_ids"]
        padding_length = length - len(input_ids)
        padding_id = tokenizer.pad_token_id if USING_PAD_TOKEN else 0
        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)
        
        return [input_ids, input_masks, input_segments]
    
    input_ids_q, input_masks_q, input_segments_q = return_id(title, question, 'longest_first', max_sequence_length)
    
    input_ids_qa, input_masks_qa, input_segments_qa = return_id(title + ' ' + question, answer, 'longest_first', max_sequence_length)

    input_ids_a, input_masks_a, input_segments_a = return_id(answer, None, 'longest_first', max_sequence_length)
    
    return [input_ids_q, input_masks_q, input_segments_q,
            input_ids_qa, input_masks_qa, input_segments_qa,
            input_ids_a, input_masks_a, input_segments_a]

def compute_input_arrays(df, columns, tokenizer, max_sequence_length):
    input_ids_q, input_masks_q, input_segments_q = [], [], []
    input_ids_qa, input_masks_qa, input_segments_qa = [], [], []
    input_ids_a, input_masks_a, input_segments_a = [], [], []
    for _, instance in tqdm(df[columns].iterrows()):
        t, q, a = instance.question_title, instance.question_body, instance.answer

        ids_q, masks_q, segments_q, ids_qa, masks_qa, segments_qa, ids_a, masks_a, segments_a = _convert_to_transformer_inputs(t, q, a, tokenizer, max_sequence_length)
        
        input_ids_q.append(ids_q)
        input_masks_q.append(masks_q)
        input_segments_q.append(segments_q)

        input_ids_qa.append(ids_qa)
        input_masks_qa.append(masks_qa)
        input_segments_qa.append(segments_qa)

        input_ids_a.append(ids_a)
        input_masks_a.append(masks_a)
        input_segments_a.append(segments_a)
        
    return [np.asarray(input_ids_q, dtype=np.int32), 
            np.asarray(input_masks_q, dtype=np.int32), 
            np.asarray(input_segments_q, dtype=np.int32),
            np.asarray(input_ids_qa, dtype=np.int32), 
            np.asarray(input_masks_qa, dtype=np.int32), 
            np.asarray(input_segments_qa, dtype=np.int32),
            np.asarray(input_ids_a, dtype=np.int32), 
            np.asarray(input_masks_a, dtype=np.int32), 
            np.asarray(input_segments_a, dtype=np.int32)]

def compute_output_arrays(df, columns):
    return np.asarray(df[columns])


# ### Models' loading code

# In[ ]:


xlnetcfg = {'architectures': ['XLNetLMHeadModel'],
 'attn_type': 'bi',
 'bi_data': False,
 'bos_token_id': 0,
 'clamp_len': -1,
 'd_head': 64,
 'd_inner': 3072,
 'd_model': 768,
 'do_sample': False,
 'dropout': 0.1,
 'end_n_top': 5,
 'eos_token_ids': 0,
 'ff_activation': 'gelu',
 'finetuning_task': None,
 'id2label': {0: 'LABEL_0', 1: 'LABEL_1'},
 'initializer_range': 0.02,
 'is_decoder': False,
 'label2id': {'LABEL_0': 0, 'LABEL_1': 1},
 'layer_norm_eps': 1e-12,
 'length_penalty': 1.0,
 'max_length': 20,
 'mem_len': None,
 'model_type': 'xlnet',
 'n_head': 12,
 'n_layer': 12,
 'num_beams': 1,
 'num_labels': 2,
 'num_return_sequences': 1,
 'output_attentions': False,
 'output_hidden_states': False,
 'output_past': True,
 'pad_token_id': 0,
 'pruned_heads': {},
 'repetition_penalty': 1.0,
 'reuse_len': None,
 'same_length': False,
 'start_n_top': 5,
 'summary_activation': 'tanh',
 'summary_last_dropout': 0.1,
 'summary_type': 'last',
 'summary_use_proj': True,
 'temperature': 1.0,
 'top_k': 50,
 'top_p': 1.0,
 'torchscript': False,
 'untie_r': True,
 'use_bfloat16': False,
 'vocab_size': 32000}


# In[ ]:


def compute_spearmanr_ignore_nan(trues, preds):
    rhos = []
    for tcol, pcol in zip(np.transpose(trues), np.transpose(preds)):
        rhos.append(spearmanr(tcol, pcol).correlation)
    return np.nanmean(rhos)

def create_nn_model(output_len, model_type):
    q_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    
    q_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    
    q_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    
    if model_type == 'BERT':
        config = BertConfig() # print(config) to see settings
        config.output_hidden_states = False # Set to True to obtain hidden states
        # caution: when using e.g. XLNet, XLNetConfig() will automatically use xlnet-large config

        # normally ".from_pretrained('bert-base-uncased')", but because of no internet, the 
        # pretrained model has been downloaded manually and uploaded to kaggle. 
        bert_model = TFBertModel.from_pretrained(BERT_PATH+'bert-base-uncased-tf_model.h5', config=config)

        # if config.output_hidden_states = True, obtain hidden states via bert_model(...)[-1]
        q_embedding = bert_model(q_id, attention_mask=q_mask, token_type_ids=q_atn)[0]
    elif model_type == 'GPT2':
#         config = GPT2Config.from_pretrained(GPT2_PATH+'gpt2-tf_model.h5')
        config = GPT2Config()
        # caution: when using e.g. XLNet, XLNetConfig() will automatically use xlnet-large config

        # normally ".from_pretrained('bert-base-uncased')", but because of no internet, the 
        # pretrained model has been downloaded manually and uploaded to kaggle. 
        gpt2_model = TFGPT2Model.from_pretrained(GPT2_PATH+'gpt2-tf_model.h5', config=config)

        # if config.output_hidden_states = True, obtain hidden states via bert_model(...)[-1]
        q_embedding = gpt2_model(q_id)[0]
    elif model_type == 'XLNET':
        config = XLNetConfig.from_dict(xlnetcfg)
#         config = XLNetConfig.from_pretrained(XLNET_PATH+'xlnet-vocab.json')
#         config = XLNetConfig()
        xlnet_model = TFXLNetModel.from_pretrained(XLNET_PATH+'xlnet-tf_model.h5', config=config)
        q_embedding = xlnet_model(q_id, attention_mask=q_mask, token_type_ids=q_atn)[0]
    
    x = tf.keras.layers.GlobalAveragePooling1D()(q_embedding)
    
    x = tf.keras.layers.Dense(output_len, activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs=[q_id, q_mask, q_atn], outputs=x)
    
    return model


# ### Inference functions

# In[ ]:


question_only = ['question_asker_intent_understanding', 'question_body_critical', 'question_conversational', 'question_expect_short_answer', 'question_fact_seeking', 'question_has_commonly_accepted_answer', 'question_interestingness_others', 'question_interestingness_self', 'question_multi_intent', 'question_not_really_a_question', 'question_opinion_seeking', 'question_type_choice', 'question_type_compare', 'question_type_consequence', 'question_type_definition', 'question_type_entity', 'question_type_instructions', 'question_type_procedure', 'question_type_reason_explanation', 'question_type_spelling', 'question_well_written']


# In[ ]:


answer_and_question = ['answer_level_of_information', 'answer_helpful','answer_plausible','answer_relevance','answer_satisfaction']


# In[ ]:


answer_only = ['answer_type_instructions', 'answer_type_procedure', 'answer_type_reason_explanation', 'answer_well_written']


# In[ ]:


AQ_and_AO = answer_and_question + answer_only


# In[ ]:


gkf10 = GroupKFold(n_splits=10).split(X=df_train.question_body, groups=df_train.question_body)
gkf5 = GroupKFold(n_splits=5).split(X=df_train.question_body, groups=df_train.question_body)

common_validation_idx = []
val10 = []
val5 = []
val10_fold0 = None
for fold, (train_idx, valid_idx) in enumerate(gkf10):
    if fold in selected_folds:
        val10 += list(valid_idx)
        if fold == 0:
            val10_fold0 = valid_idx
        
for fold, (train_idx, valid_idx) in enumerate(gkf5):
    if fold in selected_folds:
        val5 += list(valid_idx)
        
common_validation_idx = np.array(list(set(val5).intersection(set(val10))))


# In[ ]:


def predict_nn(train_data, valid_data, test_data, weights, model_type):
    K.clear_session()
    model = create_nn_model(train_data[1].shape[1], model_type)
    model.load_weights(weights)
    trn_preds = np.zeros(train_data[1].shape)
    val_preds = model.predict(valid_data[0])
    print(f'Lengths of test list is {len(test_data)}')
    test_preds = model.predict(test_data) if test_data is not None else None
    
    rho_trn = compute_spearmanr_ignore_nan(train_data[1], trn_preds)
    rho_val = compute_spearmanr_ignore_nan(valid_data[1], val_preds)
    print(f'Score train {rho_trn}; Score validation {rho_val}')
    
    return trn_preds, val_preds, test_preds

def get_cross_fold_preds_nn(input_idx, model_type):
    all_fold_trn_preds, all_fold_val_preds, all_fold_test_preds = [],[],[]
    all_fold_trn_outputs, all_fold_val_outputs = [], []
    
    gkf = GroupKFold(n_splits=n_splits).split(X=df_train.question_body, groups=df_train.question_body)

    for fold, (train_idx, valid_idx) in enumerate(gkf):
        if (USE_ONLY_SELECTED_FOLDS & (fold in selected_folds)) or (not USE_ONLY_SELECTED_FOLDS):
            if MOD_DATA_STUCTURE == '2 split':
                if input_idx == [0,1,2]: output_idx = [i for i, z in enumerate(output_categories) if z in question_only]
                if input_idx == [3,4,5]: output_idx = [i for i, z in enumerate(output_categories) if z in AQ_and_AO]
            if MOD_DATA_STUCTURE == '3 split':
                if input_idx == [0,1,2]: output_idx = [i for i, z in enumerate(output_categories) if z in question_only]
                if input_idx == [3,4,5]: output_idx = [i for i, z in enumerate(output_categories) if z in answer_and_question]
                if input_idx == [6,7,8]: output_idx = [i for i, z in enumerate(output_categories) if z in answer_only]
            
            train_inputs = [inputs[i][train_idx] for i in input_idx]
            train_outputs = outputs[np.array(train_idx)[:,None], output_idx]
            all_fold_trn_outputs.append(train_outputs)

            valid_inputs = [inputs[i][valid_idx] for i in input_idx]
            valid_outputs = outputs[np.array(valid_idx)[:,None], output_idx]
            all_fold_val_outputs.append(valid_outputs)
            
            current_test_input = [test_inputs[i] for i in input_idx]

            print(f'Fold {fold}')
            input_type = None
            if (input_idx == [3,4,5]) & (model_type == 'XLNET'):
                weights_path = f"../input/gq-xlnet-pretrained/XLNET_question_only_fold_{fold}.h5" # this is done intentionally as I have named saved weights wrongly
                input_type = 'question and answer'
            elif (input_idx == [0,1,2]) & (model_type == 'XLNET'):
                weights_path = f"../input/gq-xlnet-pretrained/XLNET_question_answer_fold_{fold}.h5" # this is done intentionally as I have named saved weights wrongly
                input_type = 'question only'
        
            if (input_idx == [3,4,5]) & (model_type != 'XLNET'):
                print(f'Using weights for BERT fold {fold}, question and answer modification')
                weights_path = f"../input/{model_roor_dir}/{model_type}_question_answer_fold_{fold}.h5"
                input_type = 'question and answer'
            elif (input_idx == [0,1,2]) & (model_type != 'XLNET'):
                weights_path = f"../input/{model_roor_dir}/{model_type}_question_only_fold_{fold}.h5"
                input_type = 'question only'
            elif (input_idx == [6,7,8]) & (model_type != 'XLNET'):
                weights_path = f"../input/{model_roor_dir}/{model_type}_answer_only_fold_{fold}.h5"
                input_type = 'answer only'

            trn_preds, val_preds, test_preds = predict_nn((train_inputs, train_outputs),(valid_inputs, valid_outputs), current_test_input, weights_path, model_type)
            all_fold_trn_preds.append(trn_preds)
            all_fold_val_preds.append(val_preds)
            all_fold_test_preds.append(test_preds)

    trn_preds, val_preds = np.concatenate(all_fold_trn_preds), np.concatenate(all_fold_val_preds)
    trn_out, val_out = np.concatenate(all_fold_trn_outputs), np.concatenate(all_fold_val_outputs)
    
    test_preds = np.stack(all_fold_test_preds, axis=2)
    test_preds = np.mean(test_preds, axis=2)
    
    print(f'Finished all folds for {model_type} {input_type}')
    print(test_preds.shape, trn_out.shape, val_preds.shape, val_out.shape, test_preds.shape)
    
    return (trn_preds, trn_out), (val_preds, val_out), test_preds

def get_nn_all_outputs(model_type):
    print('Getting all folds for QUESTION ONLY')
    qonly_trn, qonly_val, qonly_tst = get_cross_fold_preds_nn([0,1,2], model_type)
    print('Getting all folds for QUESTION ANSWER')
    qa_trn, qa_val, qa_tst = get_cross_fold_preds_nn([3,4,5], model_type)
    
    if MOD_DATA_STUCTURE == '3 split':
        print('Getting all folds for ANSWER ONLY')
        ao_trn, ao_val, ao_tst = get_cross_fold_preds_nn([6,7,8], model_type)

        trn = (np.concatenate((qonly_trn[0], qa_trn[0], ao_trn[0]), axis=1), np.concatenate((qonly_trn[1], qa_trn[1], ao_trn[1]), axis=1))
        val = (np.concatenate((qonly_val[0], qa_val[0], ao_val[0]), axis=1), np.concatenate((qonly_val[1], qa_val[1], ao_val[1]), axis=1))
        tst = np.concatenate((qonly_tst, qa_tst, ao_tst), axis=1)
    
    if MOD_DATA_STUCTURE == '2 split':
        trn = (np.concatenate((qonly_trn[0], qa_trn[0]), axis=1), np.concatenate((qonly_trn[1], qa_trn[1]), axis=1))
        val = (np.concatenate((qonly_val[0], qa_val[0]), axis=1), np.concatenate((qonly_val[1], qa_val[1]), axis=1))
        tst = np.concatenate((qonly_tst, qa_tst), axis=1)
    
    print(f'Finsihed entire dataset (qonly and qa) for {model_type}')
    print(trn[0].shape, trn[1].shape, val[0].shape, val[1].shape, tst.shape)
    
    save_file(trn, f'{model_roor_dir}_trn')
    save_file(val, f'{model_roor_dir}_val')
    save_file(tst, f'{model_roor_dir}_tst')
    
    return trn, val, tst


# ## Inference

# ### XLNet

# In[ ]:


print(psutil.cpu_percent())
print(dict(psutil.virtual_memory()._asdict()))
gc.collect()


# Get tokenized inputs for XLNet

# In[ ]:


tokenizer = XLNetTokenizer.from_pretrained('../input/gq-manual-uploads/xlnet tokenizer from colab/')
USING_PAD_TOKEN = False

outputs = compute_output_arrays(df_train, output_categories)
inputs = compute_input_arrays(df_train, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
test_inputs = compute_input_arrays(df_test, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)

save_file(outputs, 'XLNET_outputs')
save_file(inputs, 'XLNET_inputs')
save_file(test_inputs, 'XLNET_test_inputs')


# Load trained model weights and make inferences

# In[ ]:


model_roor_dir = 'gq-xlnet-pretrained'
MOD_DATA_STUCTURE = '2 split'
n_splits = 10

xlnet_trn, xlnet_val, xlnet_tst = get_nn_all_outputs('XLNET')


# In[ ]:


print(psutil.cpu_percent())
print(dict(psutil.virtual_memory()._asdict()))
gc.collect()


# ### BERT

# Get tokenized inputs for BERT

# In[ ]:


tokenizer = BertTokenizer.from_pretrained(BERT_PATH+'bert-base-uncased-vocab.txt')
USING_PAD_TOKEN = True

outputs = compute_output_arrays(df_train, output_categories)
inputs = compute_input_arrays(df_train, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
test_inputs = compute_input_arrays(df_test, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)

save_file(outputs, 'BERT_outputs')
save_file(inputs, 'BERT_inputs')
save_file(test_inputs, 'BERT_test_inputs')


# Make inferences using bert models trained in my most recent attempt.

# In[ ]:


model_roor_dir = 'gq-bert-pretrained'
MOD_DATA_STUCTURE = '2 split'
n_splits = 10

bert_trn, bert_val, bert_tst = get_nn_all_outputs('BERT')


# Make inferences using bert models trained in my pre-last attempt. This one uses 3 individual groups of response variables as described in the beginning of this kernel.

# In[ ]:


model_roor_dir = '3rd-training-2nd-gen-bert-download-from-gdrive'
MOD_DATA_STUCTURE = '3 split'
n_splits = 10
selected_folds = [0]

bert_trn_3split, bert_val_3split, bert_tst_3split = get_nn_all_outputs('BERT')
selected_folds = [0,1,2]


# Make inferences using bert models trained in one of my first attempts. Main differences is that this attempt used constant learning rate as opposed to one-cycle scheduling introduced in latter attempts.

# In[ ]:


model_roor_dir = '2nd-training-1st-gen-bert-download-from-gdrive'
MOD_DATA_STUCTURE = '2 split'
n_splits = 5

bert_trn_5fold, bert_val_5fold, bert_tst_5fold = get_nn_all_outputs('BERT')


# In[ ]:


print(psutil.cpu_percent())
print(dict(psutil.virtual_memory()._asdict()))
gc.collect()


# ### GPT2

# Get tokenized input for GPT2

# In[ ]:


# tokenizer = GPT2Tokenizer.from_pretrained(GPT2_PATH+'gpt2-vocab.json')
tokenizer = GPT2Tokenizer.from_pretrained('../input/gq-manual-uploads/gpt2 config from colab/')
USING_PAD_TOKEN = False

outputs = compute_output_arrays(df_train, output_categories)
inputs = compute_input_arrays(df_train, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
test_inputs = compute_input_arrays(df_test, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)

save_file(outputs, 'GPT2_outputs')
save_file(inputs, 'GPT2_inputs')
save_file(test_inputs, 'GPT2_test_inputs')


# Make inference using GPT2 model trained in most recent attempt.

# In[ ]:


model_roor_dir = 'gq-gpt2-pretrained'
MOD_DATA_STUCTURE = '2 split'
n_splits = 10

gpt2_trn, gpt2_val, gpt2_tst = get_nn_all_outputs('GPT2')


# Make inference using GPT2 model trained in one of my first attempt (as with BERT, this training was done using constant learning rate as opposed to one-cycle scheduling introduced in latter iterations)

# In[ ]:


model_roor_dir = '2nd-training-1st-gen-gpt-download-from-gdrive'
MOD_DATA_STUCTURE = '2 split'
n_splits = 5

gpt2_trn_5fold, gpt2_val_5fold, gpt2_tst_5fold = get_nn_all_outputs('GPT2')


# In[ ]:


print(psutil.cpu_percent())
print(dict(psutil.virtual_memory()._asdict()))
gc.collect()


# ## Inferring using meta features and decision trees

# ### Feature engineering - deriving meta features from text such as number of words, punctuation etc.

# In[ ]:


if USE_LGB:
    def remove_articles(df):
        for i in ['question_title', 'question_body', 'answer']:
            df.loc[:,f'{i}_orig'] = df.loc[:,i]
        for i in ['question_title', 'question_body', 'answer']:
            df.loc[:,i] = df.loc[:,i].apply(lambda x: x.replace(' the ',' ').replace(' a ',' ').replace(' an ',' '))
        return df

    df_train = remove_articles(df_train)
    df_test = remove_articles(df_test)


# In[ ]:


if USE_LGB:
    df_train.loc[:,'q_users_host'] = df_train.apply(lambda x: x.question_user_name + x.host, axis=1)
    df_train.loc[:,'a_users_host'] = df_train.apply(lambda x: x.answer_user_name + x.host, axis=1)
    df_test.loc[:,'q_users_host'] = df_test.apply(lambda x: x.question_user_name + x.host, axis=1)
    df_test.loc[:,'a_users_host'] = df_test.apply(lambda x: x.answer_user_name + x.host, axis=1)

    q_users_train = dict(df_train.q_users_host.value_counts())
    q_users_test = dict(df_test.q_users_host.value_counts())
    a_users_train = dict(df_train.a_users_host.value_counts())
    a_users_test = dict(df_test.a_users_host.value_counts())

    q_users = q_users_train
    for i in q_users:
        if i in q_users_test:
            q_users_train[i] += q_users_test[i]
    for i in q_users_test:
        if i not in q_users:
            q_users[i] = q_users_test[i]

    a_users = a_users_train
    for i in a_users:
        if i in a_users_test:
            a_users_train[i] += a_users_test[i]
    for i in a_users_test:
        if i not in a_users:
            a_users[i] = a_users_test[i]


# In[ ]:


if USE_LGB:

    word_categories = ['adjectives','verbs','nouns','list_maker','digits','modals','posessives','persionals','interjection','direction','past_verb']

    adjectives = ['JJ','JJR','JJS','RB','RBR','RBS']
    verbs =  ['VB','VBD','VBG','VBN','VBP','VBZ']
    nouns = ['NN','NNS','NNP','NNPS']
    list_maker = ['LS']
    digits = ['CD']
    modals = ['MD']
    posessives = ['PPR$', 'POS']
    persionals = ['PRP']
    interjection = ['UH']
    direction = ['TO']
    past_verb = ['VBD','VBN']

    def get_string_stats(df, var):
        df.loc[:, f'{var}_numchars'] = df[var].apply(lambda x: len(x))
        df.loc[:, f'{var}_numwords'] = df[var].apply(lambda x: len(x.split()))
        df.loc[:, f'{var}_exclam_count'] = df[var].apply(lambda x: x.count('!'))
        df.loc[:, f'{var}_question_count'] = df[var].apply(lambda x: x.count('?'))
        df.loc[:, f'{var}_coma_count'] = df[var].apply(lambda x: x.count(','))
        df.loc[:, f'{var}_dot_count'] = df[var].apply(lambda x: x.count('.'))
        df.loc[:, f'{var}_all_punct_count'] = df[f'{var}_question_count']+df[f'{var}_coma_count']+df[f'{var}_exclam_count']
        df.loc[:, f'{var}_all_punct_to_sentences'] = df.loc[:, f'{var}_all_punct_count']/df.loc[:, f'{var}_dot_count']
        df.loc[:, f'{var}_questions_to_sentences'] = df.loc[:, f'{var}_question_count']/df.loc[:, f'{var}_dot_count']
        df.loc[:, f'{var}_questions_to_words'] = df.loc[:, f'{var}_question_count']/df.loc[:, f'{var}_numwords']
        df.loc[:, f'{var}_average_word_len'] = df[f'{var}_numchars']/df[f'{var}_numwords']
        df.loc[:, f'{var}_capital_count'] = df[var].apply(lambda x: sum(1 for c in x if c.isupper()))
        df.loc[:, f'{var}_capital_prop'] = df[f'{var}_capital_count']/df[f'{var}_numwords']
        df.loc[:, f'{var}_other_ref'] = df[var].apply(lambda x: sum([x.count(i) for i in [' it ',' they ', " it's ", ' their ']]))
        df.loc[:, f'{var}_self_ref'] = df[var].apply(lambda x: sum([x.count(i) for i in [' I ',' me ', " mine ", ' my ']]))
        df.loc[:, f'{var}_total_ref'] = df[f'{var}_self_ref'] + df[f'{var}_other_ref']
        df.loc[:, f'{var}_total_ref_prop'] = df.loc[:, f'{var}_total_ref']/df.loc[:, f'{var}_numwords']
        df.loc[:, f'{var}_self_ref_prop'] = df[f'{var}_self_ref']/df[f'{var}_total_ref']
        df.loc[:, f'{var}_other_ref_prop'] = df[f'{var}_other_ref']/df[f'{var}_total_ref']
        df.loc[:, f'{var}_words_per_sentence'] = df[f'{var}_numwords']/df[f'{var}_dot_count']
        df.loc[:, f'{var}_unique_words'] = df[f'{var}'].apply(lambda x: len(set(str(x).split())))
        df.loc[:, f'{var}_unique_words_prop'] = df.loc[:, f'{var}_unique_words']/df.loc[:, f'{var}_numwords']
        new_cols = [f'{var}_total_ref_prop', f'{var}_questions_to_words',f'{var}_questions_to_sentences',f'{var}_all_punct_to_sentences', f'{var}_unique_words_prop', f'{var}_unique_words', f'{var}_numchars',f'{var}_numwords',f'{var}_exclam_count',f'{var}_question_count',f'{var}_coma_count',f'{var}_dot_count',f'{var}_all_punct_count',f'{var}_average_word_len',f'{var}_capital_count',f'{var}_capital_prop',f'{var}_other_ref',f'{var}_self_ref',f'{var}_total_ref',f'{var}_self_ref_prop',f'{var}_other_ref_prop',f'{var}_words_per_sentence']

        for category in word_categories: 
            df.loc[:, f'{var}_{category}'] = 0
            new_cols.append(f'{var}_{category}')

        for idx in tqdm(range(df.shape[0]), total = df.shape[0]):
            tokens = nltk.word_tokenize(df.loc[idx, var])
            tags = nltk.pos_tag(tokens)
            tags = [i[1] for i in tags]
    #         print(idx)
    #         print(tags)
            for category in word_categories:
                count = 0
                for tag_name in globals()[category]:
                    count += tags.count(tag_name)
    #                 print(count)
                df.loc[idx, f'{var}_{category}'] = count/df.loc[idx, f'{var}_numwords'] if df.loc[idx, f'{var}_numwords'] != 0 else 0
    #             print(df.loc[idx, f'{var}_numwords'])
    #         break

        return df, new_cols

    def get_extra_features_and_map(df):
        df, nc1 = get_string_stats(df, 'question_title')
        df, nc2 = get_string_stats(df, 'question_body')
        df, nc3 = get_string_stats(df, 'answer')

        df.loc[:,'q_user_q_count'] = df.q_users_host.apply(lambda x: q_users[x] if x in q_users else 0)
        df.loc[:,'q_user_a_count'] = df.q_users_host.apply(lambda x: a_users[x] if x in a_users else 0)
        df.loc[:,'a_user_a_count'] = df.a_users_host.apply(lambda x: a_users[x] if x in a_users else 0)
        df.loc[:,'a_user_q_count'] = df.a_users_host.apply(lambda x: q_users[x] if x in q_users else 0)
        df.loc[:,'q_user_both_count'] = df.loc[:,'q_user_q_count'] + df.loc[:,'q_user_a_count']
        df.loc[:,'a_user_both_count'] = df.loc[:,'a_user_a_count'] + df.loc[:,'a_user_q_count']

        other_features = []
        df.loc[:,'q_to_a_all_punct_count'] = (df.loc[:,'question_body_all_punct_count']+df.loc[:,'question_title_all_punct_count'])/df.loc[:,'answer_all_punct_count']
        df.loc[:,'q_to_a_numwords'] = (df.loc[:,'question_body_numwords']+df.loc[:,'question_title_numwords'])/df.loc[:,'answer_numwords']
        df.loc[:,'q_to_a_capital_count'] = (df.loc[:,'question_body_capital_count']+df.loc[:,'question_title_capital_count'])/df.loc[:,'answer_capital_count']
        df.loc[:,'q_to_a_unique_words_prop'] = (df.loc[:,'question_body_unique_words_prop']+df.loc[:,'question_title_unique_words_prop'])/df.loc[:,'answer_unique_words_prop']
        df.loc[:,'q_to_a_total_ref'] = (df.loc[:,'question_body_total_ref']+df.loc[:,'question_title_total_ref'])/df.loc[:,'answer_total_ref']
        df.loc[:,'common_words'] = df.apply(lambda x: len(set(x.question_body.split()).intersection(set(x.answer.split()))), axis=1)
        other_features += ['q_to_a_all_punct_count', 'q_to_a_numwords', 'q_to_a_capital_count', 'common_words', 'q_to_a_unique_words_prop', 'q_to_a_total_ref']


        for category in word_categories:
            df.loc[:,f'q_to_a_{category}'] = df.loc[:,f'question_body_{category}']/df.loc[:,f'answer_{category}']
            other_features.append(f'q_to_a_{category}')

        df.loc[:,'spell_words'] = df.loc[:,'question_title'].apply(lambda x: sum(1 for i in x.lower().split() if i in ['spell','spelled','spelt','spelling','write','wrote','written']))
        df.loc[:,'spell_words'] += df.loc[:,'question_body'].apply(lambda x: sum(1 for i in x.lower().split() if i in ['spell','spelled','spelt','spelling','write','wrote','written']))
        df.loc[:,'compare_words'] = df.loc[:,'question_title'].apply(lambda x: sum(1 for i in x.lower().split() if i in ['better','best','worse','nicer']))
        df.loc[:,'compare_words'] += df.loc[:,'question_body'].apply(lambda x: sum(1 for i in x.lower().split() if i in ['better','best','worse','nicer']))
        df.loc[:,'consequence_words'] = df.loc[:,'question_title'].apply(lambda x: sum(1 for i in x.lower().split() if i in ['if','when','will','would']))
        df.loc[:,'consequence_words'] += df.loc[:,'question_body'].apply(lambda x: sum(1 for i in x.lower().split() if i in ['if','when','will','would']))
        other_features.append('spell_words')
        other_features.append('compare_words')
        other_features.append('consequence_words')

        onehots=[]
        for i in df.loc[:,'category'].unique():
            df.loc[:,f'{i}_onehot'] = 0
            df.loc[df.loc[:,'category']==i,f'{i}_onehot'] = 1
            onehots.append(f'{i}_onehot')

        for i in df.loc[:,'host'].unique():
            df.loc[:,f'{i}_H_onehot'] = 0
            df.loc[df.loc[:,'host']==i,f'{i}_H_onehot'] = 1
            onehots.append(f'{i}_H_onehot')

        other_features = other_features+nc1+nc2+nc3+onehots+['q_user_q_count', 'q_user_a_count','a_user_a_count','a_user_q_count','q_user_both_count','a_user_both_count']
        return df, other_features

    df_train, other_features_train = get_extra_features_and_map(df_train)
    df_test, other_features = get_extra_features_and_map(df_test)

    for i in [a for a in other_features if a not in other_features_train]:
        df_train.loc[:,i] = np.zeros(df_train.shape[0])


# In[ ]:


if USE_LGB:
    def get_uids_all(df):
        df.loc[:,'answer_uid'] = df.loc[:,'answer_user_page'].apply(lambda x: int(x.split('/')[-1]))
        df.loc[:,'question_uid'] = df.loc[:,'question_user_page'].apply(lambda x: int(x.split('/')[-1]))
        for idx in range(df.shape[0]):
            split = [i for i in df.loc[idx,'url'].split('/') if i.isdigit()]
            df.loc[idx,'url_uid'] = int(split[-1]) if len(split)>0 else -1
        return df

    df_train = get_uids_all(df_train)
    df_test= get_uids_all(df_test)


# ### Using external datasets to get additional meta features about questions and answers such as upvotes, reputation of authoring users etc.

# In[ ]:


if USE_LGB:
    se_path = "../input/stackexchange-data"
    se_posts = pd.read_parquet(se_path+"/stackexchange_posts.parquet.gzip", engine='fastparquet')

    def get_post_info_se(df):
        new_other_features = []

        new_features = ['Score','ViewCount','AnswerCount','CommentCount','FavoriteCount','Tags']
        df = df.merge(se_posts.loc[:,['Id', 'host', 'AcceptedAnswerId'] + new_features], how='left', left_on=['url_uid', 'host'], right_on=['Id', 'host'], sort=False)
        df.rename({i:'SE_QP_'+i for i in new_features}, inplace=True, axis=1)
        new_other_features += ['SE_QP_'+i for i in new_features]

        return df, new_other_features

    df_train, new_other_features = get_post_info_se(df_train)
    df_test, _ = get_post_info_se(df_test)

    del se_posts
    gc.collect()

    other_features += new_other_features
    other_features.remove('SE_QP_Tags')
    
    # -----------
    
    all_tags = []
    for i in range(df_train.shape[0]):
        if (df_train.SE_QP_Tags.iloc[i] == None) or (pd.isna(df_train.SE_QP_Tags.iloc[i])): continue
        tags = df_train.SE_QP_Tags.iloc[i].replace('<','').split('>')
        all_tags += [t for t in tags if len(t) >0]

    top_tags = list(pd.DataFrame(all_tags).iloc[:,0].value_counts()[0:50].index)
    other_features += [f'tag_{i}' for i in top_tags]

    # --------
    
    for t in top_tags:
        df_train.loc[:,f'tag_{t}'] = 0
        df_test.loc[:,f'tag_{t}'] = 0

    def parse_tags(df):
        for i in range(df.shape[0]):
            if (df.SE_QP_Tags.iloc[i] == None) or (pd.isna(df.SE_QP_Tags.iloc[i])): continue
            tags = df.SE_QP_Tags.iloc[i].replace('<','').split('>')
            tags = [t for t in tags if (len(t) > 0) & (t in top_tags)]
            for t in tags:
                df.loc[i,f'tag_{t}'] = 1
        return df

    df_train = parse_tags(df_train)
    df_test = parse_tags(df_test)

    # -------
    
    se_path = "../input/stackexchange-data"
    se_users = pd.read_parquet(se_path+"/stackexchange_users.parquet.gzip", engine='fastparquet')

    def get_user_info_se(df):
        new_other_features = []

        new_features = ['Reputation','Views','Upvotes','Downvotes']
        df = df.merge(se_users.loc[:,['Id', 'host'] + new_features], how='left', left_on=['question_uid', 'host'], right_on=['Id', 'host'], sort=False)
        df.rename({i:'SE_Q_'+i for i in new_features}, inplace=True, axis=1)
        new_other_features += ['SE_Q_'+i for i in new_features]

        new_features = ['Reputation','Views','Upvotes','Downvotes']
        df = df.merge(se_users.loc[:,['Id', 'host'] + new_features], how='left', left_on=['answer_uid', 'host'], right_on=['Id', 'host'], sort=False)
        df.rename({i:'SE_A_'+i for i in new_features}, inplace=True, axis=1)
        new_other_features += ['SE_A_'+i for i in new_features]

        return df, new_other_features

    df_train, new_other_features = get_user_info_se(df_train)
    df_test, _ = get_user_info_se(df_test)

    del se_users
    gc.collect()

    other_features += new_other_features


# In[ ]:


if USE_LGB:
    import bq_helper
    from bq_helper import BigQueryHelper
    # https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
    stackOverflow = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                       dataset_name="stackoverflow")

    def get_user_info_stackoverflow(df):
        print(df.shape)
        new_other_features = []
        all_q_uids = tuple(df.loc[df.host=='stackoverflow.com','answer_uid'].unique())
        print(len(all_q_uids))

        q_users = f"""SELECT id, display_name, reputation, up_votes, down_votes, views from `bigquery-public-data.stackoverflow.users` WHERE id IN {all_q_uids}"""
        q_users_df = stackOverflow.query_to_pandas_safe(q_users)
        print(q_users_df.shape)

        new_features = ['reputation','up_votes','down_votes','views']
        df = df.merge(q_users_df, left_on='answer_uid', right_on='id', how='left', sort=False)
        df.rename({i:'A_'+i for i in new_features}, inplace=True, axis=1)
        new_other_features += ['A_'+i for i in new_features]

        all_q_uids = tuple(df.loc[df.host=='stackoverflow.com','question_uid'].unique())
        print(len(all_q_uids))

        q_users = f"""SELECT id, display_name, reputation, up_votes, down_votes, views from `bigquery-public-data.stackoverflow.users` WHERE id IN {all_q_uids}"""
        q_users_df = stackOverflow.query_to_pandas_safe(q_users)
        print(q_users_df.shape)

        new_features = ['reputation','up_votes','down_votes','views']
        df = df.merge(q_users_df, left_on='question_uid', right_on='id', how='left', sort=False)
        df.rename({i:'Q_'+i for i in new_features}, inplace=True, axis=1)
        new_other_features += ['Q_'+i for i in new_features]

        print(df.shape)
        return df, new_other_features

    df_train, new_other_features = get_user_info_stackoverflow(df_train)
    df_test, _ = get_user_info_stackoverflow(df_test)
    other_features += new_other_features

    def get_question_info_stackoverflow(df):
        print(df.shape)
        new_other_features = []
        uids_selection = tuple(df.loc[df.host=='stackoverflow.com','url_uid'].dropna().unique())
        print(len(uids_selection))

        query = f"""SELECT id, accepted_answer_id, answer_count, comment_count, favorite_count, score, view_count from `bigquery-public-data.stackoverflow.stackoverflow_posts` WHERE id IN {uids_selection}"""
        query_as_df = stackOverflow.query_to_pandas_safe(query)
        print(query_as_df.shape)

        new_features = ['accepted_answer_id', 'answer_count', 'comment_count', 'favorite_count', 'score', 'view_count']
        df = df.merge(query_as_df, left_on='url_uid', right_on='id', how='left', sort=False)
        df.rename({i:'QPAGE_'+i for i in new_features}, inplace=True, axis=1)
        new_other_features += ['QPAGE_'+i for i in new_features]

        print(df.shape)
        return df, new_other_features

    df_train, new_other_features = get_question_info_stackoverflow(df_train)
    df_test, _ = get_question_info_stackoverflow(df_test)
    other_features += new_other_features


# In[ ]:


if USE_LGB:
    def is_answer_accepted(df):
        for i in df.loc[(df.host=='stackoverflow.com') & ~(df.QPAGE_accepted_answer_id.isna()),:].index.values:
            df.loc[i,'answer_accepted'] = 1 if df.loc[i, 'answer_uid'] == df.loc[i, 'QPAGE_accepted_answer_id'] else 0
        for i in df.loc[(df.host!='stackoverflow.com') & ~(df.AcceptedAnswerId.isna()),:].index.values:
            df.loc[i,'answer_accepted'] = 1 if df.loc[i, 'answer_uid'] == df.loc[i, 'AcceptedAnswerId'] else 0
        return df

    df_train = is_answer_accepted(df_train)
    df_test = is_answer_accepted(df_test)
    other_features.append('answer_accepted')


# ### Specifying LGB and XGB models

# In[ ]:


if USE_LGB:
    class Base_Model(object):

        def __init__(self, train_df, test_df, features, categoricals=[], n_splits=5, verbose=True, target=None, predict_test=True):
            self.train_df = train_df
            self.test_df = test_df
            self.features = features
            self.n_splits = 10
            self.categoricals = categoricals
            self.target = target
            self.cv = self.get_cv()
            self.verbose = verbose
            self.params = self.get_params()
            self.predict_test = predict_test
            self.tst_pred, self.score, self.model, self.val_ys, self.val_preds = self.fit()

        def train_model(self, train_set, val_set):
            raise NotImplementedError

        def get_cv(self):
            cv = GroupKFold(n_splits=self.n_splits)
            return cv.split(X=self.train_df.question_body_orig, groups=self.train_df.question_body_orig)

        def get_params(self):
            raise NotImplementedError

        def convert_dataset(self, x_train, y_train, x_val, y_val):
            raise NotImplementedError

        def convert_x(self, x):
            return x

        def fit(self):
    #         oof_pred = np.zeros((len(unseen_valid), )) if MIX_UP else np.zeros((len(self.train_df), ))
            oof_pred = []
            oof_ys_all = []
            y_pred = np.zeros((len(self.test_df), ))
            for fold, (train_idx, val_idx) in enumerate(self.cv):
                if fold < ACTUAL_FOLDS:
                    x_train, x_val = self.train_df[self.features].iloc[train_idx], self.train_df[self.features].iloc[val_idx]
                    y_train, y_val = self.train_df[self.target][train_idx], self.train_df[self.target][val_idx]

                    train_set, val_set = self.convert_dataset(x_train, y_train, x_val, y_val)

                    model = self.load_model(fold)

                    conv_x_val = self.convert_x(x_val.reset_index(drop=True))
                    preds_all = model.predict(conv_x_val)
                    preds_all = 1/(1 + np.exp(-preds_all))
#                     preds_all = np.round(preds_all, ROUND_PLACES) if ROUND else preds_all
                    oof_pred += preds_all.tolist() 

                    if self.predict_test: 
                        x_test = self.convert_x(self.test_df[self.features])
                        current_test_preds = model.predict(x_test).reshape(y_pred.shape)
                        current_test_preds = 1/(1 + np.exp(-current_test_preds))
                        current_test_preds = current_test_preds/ACTUAL_FOLDS
                        y_pred += current_test_preds #no to list as this is stored as single numpy array

                    if self.verbose:print('Partial score (all) of fold {} is: {}'.format(fold, spearmanr(y_val, preds_all).correlation))

                    oof_ys_all += list(y_val.reset_index(drop=True).values)

            loss_score = spearmanr(oof_ys_all, oof_pred).correlation

            if self.verbose:
                print('Our oof cohen kappa score (all) is: ', loss_score)

            return y_pred, loss_score, model, np.array(oof_ys_all), np.array(oof_pred)

    class Lgb_Model(Base_Model):

        def train_model(self, train_set, val_set):
            verbosity = 100 if self.verbose else 0
            return lgb.train(self.params, train_set, valid_sets=[train_set, val_set], verbose_eval=verbosity)

        def load_model(self, fold):
            model = pickle.load(open(f'../input/gq-lgb/{self.target}/{self.target}_{fold}.p', 'rb'))
            return model

        def convert_dataset(self, x_train, y_train, x_val, y_val):
            train_set = lgb.Dataset(x_train, y_train, categorical_feature=self.categoricals)
            val_set = lgb.Dataset(x_val, y_val, categorical_feature=self.categoricals)
            return train_set, val_set

        def get_params(self):
            params = {'n_estimators':5000,
                    'boosting_type': 'gbdt',
                    'objective': 'cross_entropy_lambda',
#                   'is_unbalance': 'true',
#                     'metric': 'huber',
                    'subsample': 0.75,
                    'subsample_freq': 1,
                    'learning_rate': 0.01,
                    'feature_fraction': 0.8,
                    'max_depth': 150, # was 15
                  'num_leaves': 50,
                    'lambda_l1': 0.1,  
                    'lambda_l2': 0.1,
                    'early_stopping_rounds': 300,
                  'min_data_in_leaf': 1,
                          'min_gain_to_split': 0.01,
                          'max_bin': 400
                        }
            return params
        
    from xgboost import XGBClassifier, XGBRegressor
    import xgboost as xgb

    class Xgb_Model(Base_Model):
        
        def load_model(self, fold):
            model = pickle.load(open(f'../input/gq-xgb/{self.target}/{self.target}_{fold}.p', 'rb'))
            return model

        def train_model(self, train_set, val_set):
            verbosity = 100 if self.verbose else 0
            return xgb.train(self.params, train_set, 
                             num_boost_round=5000, evals=[(train_set, 'train'), (val_set, 'val')], 
                             verbose_eval=verbosity, early_stopping_rounds=100)

        def convert_dataset(self, x_train, y_train, x_val, y_val):
            train_set = xgb.DMatrix(x_train, y_train)
            val_set = xgb.DMatrix(x_val, y_val)
            return train_set, val_set

        def convert_x(self, x):
            return xgb.DMatrix(x)

        def get_params(self):
            params = {               
                'objective':'reg:logistic',
                'n_estimators':5000,
            'max_depth':12,
            'eta':0.05}

            return params


# In[ ]:


print(psutil.cpu_percent())
print(dict(psutil.virtual_memory()._asdict()))
gc.collect()


# In[ ]:


if USE_LGB:
    one_lgb_model = pickle.load(open(f'../input/gq-lgb/question_opinion_seeking/question_opinion_seeking_0.p', 'rb'))
    lgb_pretrained_features = one_lgb_model.feature_name()

    for i in lgb_pretrained_features:
        if i not in other_features:
            print(f'{i} not in other features here, adding zeros')
            df_train.loc[:, i] = np.zeros(df_train.shape[0])
            df_test.loc[:, i] = np.zeros(df_test.shape[0])


# ### Making inferences using LGB and XGB trained models

# LGB

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nif USE_LGB:\n    ACTUAL_FOLDS = 3\n    lgb_val_scores = []\n    n_output_categories = len(output_categories)\n\n    lgb_val_outputs_all = []\n    lgb_val_preds_all = []\n    lgb_tst_preds_all = []\n\n    for idx, i in enumerate(output_categories, 1):\n\n        lgb_model = Lgb_Model(df_train, df_test, lgb_pretrained_features, target=i, verbose=False)\n\n        lgb_val_outputs_all.append(lgb_model.val_ys)\n        lgb_val_preds_all.append(lgb_model.val_preds)\n        lgb_tst_preds_all.append(lgb_model.tst_pred)\n        lgb_val_scores.append(lgb_model.score)\n\n        print(f'{idx}/{n_output_categories}',i, lgb_model.score)\n        \n    save_file(lgb_val_outputs_all, 'lgb_val_outputs_all')\n    save_file(lgb_val_preds_all, 'lgb_val_preds_all')\n    save_file(lgb_tst_preds_all, 'lgb_tst_preds_all')\n    \n    save_file(df_train, 'df_train')\n    save_file(df_test, 'df_test')\nelse:\n    lgb_val_outputs_all = [0]*30\n    lgb_val_preds_all = [0]*30\n    lgb_tst_preds_all = [0]*30")


# In[ ]:


if USE_XGB:
    one_xgb_model = pickle.load(open(f'../input/gq-xgb/question_opinion_seeking/question_opinion_seeking_0.p', 'rb'))
    xgb_pretrained_features = one_xgb_model.feature_names

    for i in xgb_pretrained_features:
        if i not in other_features:
            print(f'{i} not in other features here, adding zeros')
            df_train.loc[:, i] = np.zeros(df_train.shape[0])
            df_test.loc[:, i] = np.zeros(df_test.shape[0])


# XGB

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nif USE_XGB:\n    ACTUAL_FOLDS = 3\n    xgb_val_scores = []\n    n_output_categories = len(output_categories)\n\n    xgb_val_outputs_all = []\n    xgb_val_preds_all = []\n    xgb_tst_preds_all = []\n\n    for idx, i in enumerate(output_categories, 1):\n\n        xgb_model = Xgb_Model(df_train, df_test, xgb_pretrained_features, target=i, verbose=False)\n\n        xgb_val_outputs_all.append(xgb_model.val_ys)\n        xgb_val_preds_all.append(xgb_model.val_preds)\n        xgb_tst_preds_all.append(xgb_model.tst_pred)\n        xgb_val_scores.append(xgb_model.score)\n\n        print(f'{idx}/{n_output_categories}',i, xgb_model.score)\n        \n    save_file(xgb_val_outputs_all, 'xgb_val_outputs_all')\n    save_file(xgb_val_preds_all, 'xgb_val_preds_all')\n    save_file(xgb_tst_preds_all, 'xgb_tst_preds_all')\n    \n    save_file(df_train, 'df_train')\n    save_file(df_test, 'df_test')\nelse:\n    xgb_val_outputs_all = [0]*30\n    xgb_val_preds_all = [0]*30\n    xgb_tst_preds_all = [0]*30")


# In[ ]:


print(psutil.cpu_percent())
print(dict(psutil.virtual_memory()._asdict()))
gc.collect()


# ## Post-processing and submission

# ### Support functions

# Function for dynamic rounding. Iteratively tries different rounding methods like normal, ceil and floor as well as different number of decimal places. Uses validation predicitons and response variables for evaluation.

# In[ ]:


def get_rounding(ys, preds):
    rounding_types = ['Normal', 'Ceil', 'Floor']
    rounding_funcs = [np.round, np.ceil, np.floor]
    dec_places = [1,2,3,4,5]
    score = spearmanr(ys, preds).correlation
    if np.isnan(score): score=-100
    best_result = {'Type':'No rounding','DP':0, 'func':None}
    for r_type, r_func in zip(rounding_types, rounding_funcs):
        for dp in dec_places:
            if r_type == 'Normal':
                rounded_preds = r_func(preds, dp)
                cur_score = spearmanr(ys, rounded_preds).correlation
            else:
                rounded_preds = r_func(preds*(10**dp))/(10**dp)
                cur_score = spearmanr(ys, rounded_preds).correlation
            if np.isnan(cur_score): cur_score = 0
            if cur_score > score:
                score = cur_score
                best_result['Type'] = r_type
                best_result['DP'] = dp
                best_result['func'] = r_func
    return score, best_result


# Functions for dynamic optimisation of weights. I have not used Nelder-Mead simply not to overfit train set and not to end up in local minimum. So I have decided to use simple iteration over a range of weights.

# In[ ]:


from scipy.optimize import minimize
from functools import partial

def inverse_spearman_r(weights, ys, preds):
    mixed_val_preds = np.array([i*w for i,w in zip(preds, weights)]).sum(axis=0)
    score = spearmanr(ys, mixed_val_preds).correlation
    if np.isnan(score): score=-100
    return -score

def optimize_mixing_weights(ys, preds):
    naive_mix = np.array(preds).mean(axis=0)
    score = spearmanr(ys, naive_mix).correlation
    if np.isnan(score): score=-100
    c_dict = {'type':'eq','fun':lambda x: 1-np.sum(x)}
    optim_func = partial(inverse_spearman_r, ys=ys, preds=preds)
    x0 = np.array([1/len(preds)]*len(preds))
    res = minimize(optim_func, x0, method='SLSQP', constraints=c_dict)
    print(f'Best score {res.fun}; weights {res.x}')
    return res.x

from itertools import combinations_with_replacement

def optimize_mixing_weights_guessing(ys, preds):
    variants = [0.0, 0.01, 0.04, 0.1, 0.12, 0.16, 0.18, 0.2, 0.22, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.82, 0.85, 0.88, 0.9, 0.94, 0.95]
    models_n = len(preds)
    weights = [1/models_n]*models_n
    best_weights = weights
    best_score = 0
    for comb in combinations_with_replacement(variants, models_n):
        weights = list(comb)
        if np.sum(weights) != 1: continue
        score = spearmanr(ys, np.array([i*w for i,w in zip(preds, weights)]).sum(axis=0)).correlation
        if np.isnan(score): score=-100
        if score>best_score: best_weights, best_score = weights.copy(), score
    print(f'Best score {best_score}; weights {best_weights}')
    return best_weights


# Because there were slightly different validation splits that I have used to train my models, I had to come up with samples that would be unseen for all the models I am using. This is what this 'common validation idx' is for.

# In[ ]:


# common_validation_idx

val10_common = [np.where(val10==i)[0][0] for i in common_validation_idx]
val5_common = [np.where(val5==i)[0][0] for i in common_validation_idx]


# In[ ]:


val10_fold0_common = [np.where(val10_fold0==i)[0] for i in common_validation_idx]
val10_fold0_common = [i[0] for i in val10_fold0_common if len(i)>0]


# ### Blending

# Iterate through all 30 respoinse variables and create individual blends for each of them. Use dynamic rounding and softmax to come up with the blend that would have best validation score.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nweights_all = []\nweights_for_rounding = []\nscores_all = []\n\ndf_sub = pd.read_csv(PATH+\'sample_submission.csv\')\nbestonly_raw_scores = []\nbest_rounded_scores = []\nbest_weighted_scores = []\n\nif USE_SAMPLE: df_sub=df_sub.iloc[0:round(0.1*df_sub.shape[0]),:]\n\nif USE_LGB:  \n    if len(lgb_val_scores) != 30: \n        USE_LGB=False\n        print(\'Something wrong with LGB, switching it off\')\n    \nfor idx, cat in enumerate(output_categories):\n    lgb_score = spearmanr(xlnet_val[1][:,idx], lgb_val_preds_all[idx]).correlation if USE_LGB else np.nan\n    xgb_score = spearmanr(xlnet_val[1][:,idx], xgb_val_preds_all[idx]).correlation if USE_XGB else np.nan\n    gpt2_score = spearmanr(gpt2_val[1][:,idx], gpt2_val[0][:,idx]).correlation\n    bert_score = spearmanr(bert_val[1][:,idx], bert_val[0][:,idx]).correlation\n    xlnet_score = spearmanr(xlnet_val[1][:,idx], xlnet_val[0][:,idx]).correlation\n    bert_score_3split = spearmanr(bert_val_3split[1][:,idx], bert_val_3split[0][:,idx]).correlation if USE_3SPLIT_BERT else np.nan\n    bert_score_5fold = spearmanr(bert_val_5fold[1][:,idx], bert_val_5fold[0][:,idx]).correlation\n    gpt2_score_5fold = spearmanr(gpt2_val_5fold[1][:,idx], gpt2_val_5fold[0][:,idx]).correlation\n    \n    scores_all.append([lgb_score, xgb_score, gpt2_score, bert_score, xlnet_score, bert_score_3split, bert_score_5fold, gpt2_score_5fold])\n    selected_models_scores = [i for i in [lgb_score, xgb_score, gpt2_score, bert_score, xlnet_score, bert_score_3split, bert_score_5fold, gpt2_score_5fold] if not np.isnan(i)]\n    selected_models_val_preds = [i for i, s in zip([lgb_val_preds_all[idx], xgb_val_preds_all[idx], gpt2_val[0][:,idx], bert_val[0][:,idx], xlnet_val[0][:,idx], bert_val_3split[0][:,idx], bert_val_5fold[0][:,idx], gpt2_val_5fold[0][:,idx]],[lgb_score, xgb_score, gpt2_score, bert_score, xlnet_score, bert_score_3split, bert_score_5fold, gpt2_score_5fold]) if not np.isnan(s)]\n    selected_models_tst_preds = [i for i, s in zip([lgb_tst_preds_all[idx], xgb_tst_preds_all[idx], gpt2_tst[:,idx], bert_tst[:,idx], xlnet_tst[:,idx], bert_tst_3split[:,idx], bert_tst_5fold[:,idx], gpt2_tst_5fold[:,idx]],[lgb_score, xgb_score, gpt2_score, bert_score, xlnet_score, bert_score_3split, bert_score_5fold, gpt2_score_5fold]) if not np.isnan(s)]\n    \n    current_best = np.max(selected_models_scores)\n    bestonly_raw_scores.append(current_best)\n    \n    if USE_LGB  & USE_XGB:\n        val_preds_10and5 = [lgb_val_preds_all[idx][val10_common], xgb_val_preds_all[idx][val10_common], gpt2_val[0][val10_common,idx], bert_val[0][val10_common,idx], xlnet_val[0][val10_common,idx], bert_val_5fold[0][val5_common,idx], gpt2_val_5fold[0][val5_common,idx]]\n        scores10and5 = np.sum([np.exp(i*beta) for i in [lgb_score, xgb_score, gpt2_score, bert_score, xlnet_score, bert_score_5fold, gpt2_score_5fold]])\n        weights10and5 = [np.exp(i*beta)/scores10and5 for i in [lgb_score, xgb_score, gpt2_score, bert_score, xlnet_score, bert_score_5fold, gpt2_score_5fold]]\n    if USE_LGB  & (not USE_XGB):\n        val_preds_10and5 = [lgb_val_preds_all[idx][val10_common], gpt2_val[0][val10_common,idx], bert_val[0][val10_common,idx], xlnet_val[0][val10_common,idx], bert_val_5fold[0][val5_common,idx], gpt2_val_5fold[0][val5_common,idx]]\n        scores10and5 = np.sum([np.exp(i*beta) for i in [lgb_score, gpt2_score, bert_score, xlnet_score, bert_score_5fold, gpt2_score_5fold]])\n        weights10and5 = [np.exp(i*beta)/scores10and5 for i in [lgb_score, gpt2_score, bert_score, xlnet_score, bert_score_5fold, gpt2_score_5fold]]\n    if (not USE_LGB)  & (not USE_XGB):\n        val_preds_10and5 = [gpt2_val[0][val10_common,idx], bert_val[0][val10_common,idx], xlnet_val[0][val10_common,idx], bert_val_5fold[0][val5_common,idx], gpt2_val_5fold[0][val5_common,idx]]\n        scores10and5 = np.sum([np.exp(i*beta) for i in [gpt2_score, bert_score, xlnet_score, bert_score_5fold, gpt2_score_5fold]])\n        weights10and5 = [np.exp(i*beta)/scores10and5 for i in [gpt2_score, bert_score, xlnet_score, bert_score_5fold, gpt2_score_5fold]]\n        \n    val_mix = np.array([i*w for i,w in zip(val_preds_10and5, weights10and5)]).sum(axis=0)\n    common_val = xlnet_val[1][val10_common,idx]\n\n    weights_for_rounding.append(weights10and5)\n    \n    if not ROUND: print(f\'{idx} {cat} scores: LGB: {lgb_score}, GPT2: {gpt2_score}, BERT: {bert_score}, XLNET: {xlnet_score}\')\n    \n    if WEIGHTS == \'softmax\':\n        all_scores = np.sum(np.exp([i*beta for i in selected_models_scores]))\n        weights = [np.exp(i*beta)/all_scores for i in selected_models_scores]\n        best_weighted_scores.append(spearmanr(common_val, np.array([i*w for i,w in zip(val_preds_10and5, weights)]).sum(axis=0)).correlation)\n        weights_all.append(weights)\n        df_sub.iloc[:, idx+1] = np.array([i*w for i,w in zip(selected_models_tst_preds, weights)]).sum(axis=0)\n    if WEIGHTS == \'dynamic\':\n        weights = optimize_mixing_weights_guessing(common_val, val_preds_10and5)\n        weights_all.append(weights)\n        best_weighted_scores.append(spearmanr(common_val, np.array([i*w for i,w in zip(val_preds_10and5, weights)]).sum(axis=0)).correlation)\n        val_mix = np.array([i*w for i,w in zip(val_preds_10and5, weights)]).sum(axis=0)\n        df_sub.iloc[:, idx+1] = np.array([i*w for i,w in zip(selected_models_tst_preds, weights)]).sum(axis=0)\n    if WEIGHTS == \'dynamic softmax\':\n        ds_score = -100\n        best_beta = None\n        for b in np.arange(1,30,0.5):\n            all_scores = np.sum(np.exp([i*b for i in selected_models_scores]))\n            try_weights = [np.exp(i*b)/all_scores for i in selected_models_scores]\n            try_mix = np.array([i*w for i,w in zip(val_preds_10and5, try_weights)]).sum(axis=0)\n            try_score = spearmanr(common_val, try_mix).correlation\n            if try_score > ds_score:\n                best_beta = b\n                ds_score = try_score\n                weights = try_weights.copy()\n                val_mix = try_mix.copy()\n        weights_all.append(weights)\n        best_weighted_scores.append(ds_score)\n        print(f\'Best beta: {best_beta}\')\n        df_sub.iloc[:, idx+1] = np.array([i*w for i,w in zip(selected_models_tst_preds, weights)]).sum(axis=0)\n            \n    if WEIGHTS == \'bestonly\':\n        df_sub.iloc[:, idx+1] = selected_models_tst_preds[np.argmax(selected_models_scores)]\n    \n    if ROUND:\n        best_rounded_score, rounding_method = get_rounding(common_val, val_mix)\n        best_rounded_scores.append(best_rounded_score)\n        print(f"{idx} {cat}: mixed score {best_rounded_score}; {rounding_method[\'Type\']}, {rounding_method[\'DP\']}")\n        \n        unrounded_backup = df_sub.iloc[:, idx+1].copy()\n        if rounding_method[\'Type\'] == \'Normal\':\n            df_sub.iloc[:, idx+1] = rounding_method[\'func\'](df_sub.iloc[:, idx+1], rounding_method[\'DP\'])\n        elif (rounding_method[\'Type\'] == \'Ceil\') or (rounding_method[\'Type\'] == \'Floor\'):\n            df_sub.iloc[:, idx+1] = rounding_method[\'func\'](df_sub.iloc[:, idx+1]*(10**rounding_method[\'DP\']))/(10**rounding_method[\'DP\'])\n        \n    if USE_TEST_SAFETY_ADJUSTMENTS:\n        if df_sub.iloc[:,idx+1].var() == 0:\n            print(\'Test predictions are STILL homogenous, reverting to softmax weights\')\n            all_scores = np.sum(np.exp([i for i in selected_models_scores]))\n            softmax_weights = [np.exp(i)/all_scores for i in selected_models_scores]\n            df_sub.iloc[:, idx+1] = np.array([i*w for i,w in zip(selected_models_tst_preds, softmax_weights)]).sum(axis=0)\n        if df_sub.iloc[:,idx+1].var() == 0:\n            raise Exception(\'var = 0\')\n        if df_sub.iloc[:,idx+1].min() < 0:\n            raise Exception("<0")\n        if df_sub.iloc[:,idx+1].max() > 1:\n            raise Exception(">1")\n        if df_sub.isna().sum().sum() != 0:\n            raise Exception("na in sub")\n           \ndf_sub.to_csv(\'submission.csv\', index=False)')


# In[ ]:


save_file(weights_all, 'weights_all')
save_file(weights_for_rounding, 'weights_all')
save_file(scores_all, 'weights_all')


# ### Do some diagnostic printouts and graphs

# Final submission max, min and shape

# In[ ]:


np.max(df_sub.iloc[:,1:].to_numpy().flatten()), np.min(df_sub.iloc[:,1:].to_numpy().flatten()), df_sub.shape


# Average of best raw scores (unrounded and unmixed), min and max of the same

# In[ ]:


np.mean(bestonly_raw_scores), np.min(bestonly_raw_scores), np.max(bestonly_raw_scores), len(bestonly_raw_scores)


# Average of mixed prediction scores before rounding, min and max of the same

# In[ ]:


np.mean(best_weighted_scores), np.min(best_weighted_scores), np.max(best_weighted_scores), len(best_weighted_scores)


# Average of mixed and rounded prediction scores. This is what I would consider final validation score. It was indeed highly correlated with public leaderboard scores.

# In[ ]:


np.mean(best_rounded_scores), np.min(best_rounded_scores), np.max(best_rounded_scores), len(best_rounded_scores)


# Plotting scores, same ones as above

# In[ ]:


from matplotlib.pyplot import figure
figure(num=None, figsize=(16, 6), dpi=80, facecolor='white', edgecolor='k')
plt.boxplot(weights_all)
plt.title('Weights for all models per output category')
plt.show()


# In[ ]:


figure(num=None, figsize=(16, 6), dpi=80, facecolor='white', edgecolor='k')
scores_all = [np.array(x) for x in scores_all]
plt.boxplot([x[~np.isnan(x)] for x in scores_all])
plt.title('Scores (rank correlation) for all models per output category')
plt.show()


# Printing out individual scores, same ones as above

# In[ ]:


[(c,np.round(s,3)) for c,s in zip(output_categories, scores_all)]


# In[ ]:


[(c,np.round(s,3)) for c,s in zip(output_categories, weights_all)]


# Plotting distributions for each response variable - both predicted on test and the ones in train

# In[ ]:


for i in output_categories:
    fig, ax = plt.subplots(1,2, figsize=(10,3))
    ax[0].hist(df_sub[i])
    ax[0].set_title('Distribution in final predictions')
    ax[1].hist(df_train[i])
    ax[1].set_title('Distribution in test data')
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




