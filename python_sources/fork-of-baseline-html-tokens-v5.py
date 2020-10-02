#!/usr/bin/env python
# coding: utf-8

# This version uses the WTA (Winner takes all) summary both for short and long answers.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

import sys
sys.path.append('../input/bert-baseline-pre-and-post-process/')

import preprocessv5 as preprocess
import postprocessv6 as postprocess
import to_pklv5 as to_pkl
import pkl_to_tfrecordsv5 as pkl_to_tfrecords

import json
import tqdm

import absl

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import os
os.getpid()


# In[ ]:


use_wta = True

on_kaggle_server = os.path.exists('/kaggle')
nq_test_file = '../input/tensorflow2-question-answering/simplified-nq-test.jsonl' 
public_dataset = os.path.getsize(nq_test_file)<20_000_000
private_dataset = os.path.getsize(nq_test_file)>=20_000_000
model_path = '../input/tpu-2020-01-22/'

for k in ['on_kaggle_server','nq_test_file','public_dataset','private_dataset']:
    print(k,globals()[k],sep=': ')


# In[ ]:


model = tf.saved_model.load(model_path)


# In[ ]:


to_pkl.jsonl_to_pkl(source=nq_test_file,output='features.pkl',
                vocab=model_path +'assets/vocab-nq.txt',
                max_contexts=-1,lower_case=True)


# In[ ]:


pkl_to_tfrecords._convert(source='features.pkl',output='all.tfrecords',meta_data='meta_data',shuffle=False,shuffle_size=0,yield_segment_variant='nolabels')


# In[ ]:


def input_fn(input_file_pattern,seq_length=512,batch_size=4):
    def mk_labels(ex):          
        qlen = ex.pop('question_len')
        dlen = ex.pop('data_len')
        input_mask = tf.sequence_mask(dlen,seq_length,dtype=tf.int32)
        ex['input_mask']  = input_mask
        ex['segment_ids'] = tf.minimum(input_mask,1-tf.sequence_mask(qlen,seq_length,dtype=tf.int32))
        return ex

    name_to_features = {
        'input_ids'   : tf.io.FixedLenFeature([seq_length], tf.int64),
        'question_len': tf.io.FixedLenFeature([], tf.int64),
        'data_len'    : tf.io.FixedLenFeature([], tf.int64),
    }
    name_to_features['unique_id']   = tf.io.FixedLenFeature([2], tf.int64)

    def decode(record):
        ex = tf.io.parse_single_example(record, name_to_features)
        for k,v in ex.items():
            if k!='unique_id':
                ex[k] = tf.cast(v,tf.int32)
        return ex

    input_files = tf.io.gfile.glob(input_file_pattern)        
    d = tf.data.TFRecordDataset(input_files)
    d = d.map(decode)
    d = d.batch(batch_size,drop_remainder=False)
    #d = d.map(mk_labels)
    d = d.prefetch(128)
    return d


# In[ ]:


def output_fn():
    def _output_fn(unique_id,model_output,n_keep=100):
        pos_logits,ans_logits,long_mask,short_mask,cross = model_output

        long_span_logits =  pos_logits
        mask = tf.cast(tf.expand_dims(long_mask,-1),long_span_logits.dtype)

        long_span_logits = long_span_logits-10000*mask 
        long_p = tf.nn.softmax(long_span_logits,axis=1)

        short_span_logits = pos_logits
        short_span_logits -= 10000*tf.cast(tf.expand_dims(short_mask,-1),short_span_logits.dtype)
        start_logits,end_logits = short_span_logits[:,:,0],short_span_logits[:,:,1]

        batch_size,seq_length = short_span_logits.shape[0],short_span_logits.shape[1]
        seq = tf.range(seq_length)
        i_leq_j_mask = tf.cast(tf.expand_dims(seq,1)>tf.expand_dims(seq,0),short_span_logits.dtype)
        i_leq_j_mask = tf.expand_dims(i_leq_j_mask,0)

        logits  = tf.expand_dims(start_logits,2)+tf.expand_dims(end_logits,1)+cross
        logits -= 10000*i_leq_j_mask
        logits  = tf.reshape(logits, [batch_size,seq_length*seq_length])
        short_p = tf.nn.softmax(logits)
        indices = tf.argsort(short_p,axis=1,direction='DESCENDING')[:,:n_keep]
        short_p = tf.gather(short_p,indices,batch_dims=1)

        return dict(unique_id = unique_id,
                    ans_logits= ans_logits,
                    long_p    = long_p,
                    short_p   = short_p,
                    short_p_indices = indices)
    return _output_fn


# In[ ]:


d = input_fn('all.tfrecords',batch_size=64) 
if public_dataset:
    d = d.take(3)
if not on_kaggle_server:
    d = tqdm.notebook.tqdm(d)
results = []
output = output_fn() 
for b in d:
    unique_id = b.pop('unique_id').numpy()
    b = [b['data_len'],b['input_ids'],b['question_len']]
    # print(b.keys())
    #pos_logits,ans_logits,mask_0,mask_1 = 
    out_dict = output(unique_id,model(b,training=False))
    for k,v in out_dict.items():
            if isinstance(v,tf.Tensor):
                out_dict[k] = v.numpy()
    results.append(out_dict)

raw_results = postprocess.read_rawresult(results)
#    pos_logits,ans_logits = pos_logits.numpy(),ans_logits.numpy()
#    result = postprocess.to_rawresult(unique_id=unique_id,
#                                      pos_logits=pos_logits,
#                                      ans_logits=ans_logits)
#    raw_results.extend([postprocess.RawResult(*x) for x in zip(*result)])


# In[ ]:


iterator = postprocess.pickle_iter('features.pkl')
if not on_kaggle_server:
    iterator = tqdm.notebook.tqdm(iterator)
records = postprocess.read_features(iterator)


# In[ ]:


examples = postprocess.compute_examples(raw_results,records)


# In[ ]:


#e2p = postprocess.ExampleToProb(keep_threshold=0.1,null_prob_threshold=1e-4)
Summary = postprocess.WTASummary #if use_wta else postprocessv3.ProbSummary
summary = Summary(min_vote_prob=0.1)
predictions = [summary(e) for e in tqdm.notebook.tqdm(examples)]


# In[ ]:


index = pd.read_csv('../input/tensorflow2-question-answering/sample_submission.csv').example_id

submission = postprocess.create_submission_df(predictions,index=index,
                                                long_threshold=0.94 if use_wta else 0.77,
                                                short_threshold=0.94 if use_wta else 0.77 ,
                                                yes_no_threshold=0.6)


# In[ ]:


submission.to_csv('submission.csv')


# In[ ]:


## ! head submission.csv


# In[ ]:


submission.head(10)

