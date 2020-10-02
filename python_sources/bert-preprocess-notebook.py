#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import sys
sys.path.append('../input/bert-joint-baseline/')
import bert_utils
import tokenization

import os
on_kaggle_server = os.path.exists('/kaggle')
if not on_kaggle_server:
    sys.path.append('../input/preprocess/')
import bert_preprocess as preprocess
import json
import collections
import itertools
import tqdm
import tensorflow as tf


# In[ ]:


nq_test_file = '../input/tensorflow2-question-answering/simplified-nq-test.jsonl'


# In[ ]:


get_ipython().run_cell_magic('time', '', "eval_records='nq-test_v1.tfrecords'\nif True or not os.path.exists(eval_records):\n    eval_writer = bert_utils.FeatureWriter(\n        filename=os.path.join(eval_records),\n        is_training=False)\n\n    tokenizer = tokenization.FullTokenizer(vocab_file='../input/bert-joint-baseline/vocab-nq.txt', \n                                           do_lower_case=True)\n\n    features = []\n    convert = bert_utils.ConvertExamples2Features(tokenizer=tokenizer,\n                                                  is_training=False,\n                                                  output_fn=eval_writer.process_feature,\n                                                  collect_stat=False)\n\n    n_examples = 0\n    tqdm_notebook= None\n    for examples in bert_utils.nq_examples_iter(input_file=nq_test_file, \n                                                tqdm=tqdm_notebook,\n                                                is_training=False):\n        for example in examples:\n            n_examples += convert(example)\n\n    eval_writer.close()\n    print('number of test examples: %d, written to file: %d' % (n_examples,eval_writer.num_features))")


# In[ ]:


get_ipython().run_cell_magic('time', '', "eval_records='nq-test_v2.tfrecords'\nif True or not os.path.exists(eval_records):\n    eval_writer = preprocess.FeatureWriter(filename=os.path.join(eval_records),is_training=False)\n\n    tokenizer = tokenization.FullTokenizer(vocab_file='../input/bert-joint-baseline/vocab-nq.txt', \n                                           do_lower_case=True)\n\n    convert = preprocess.JSON2Features(tokenizer=tokenizer)\n\n    tqdm_notebook= None # tqdm.tqdm_notebook\n    for line in preprocess.file_iter(input_file=nq_test_file, tqdm=tqdm_notebook):\n        examples = convert(line)\n        for example in examples:\n            eval_writer.process_feature(example)\n\n    eval_writer.close()\n    print('number of test examples written to file: %d' % eval_writer.num_features)")


# In[ ]:


def read_dataset(file_name):
    raw_data = tf.data.TFRecordDataset(file_name)
    seq_length = 512 
    name_to_features = {
        "unique_id":   tf.io.FixedLenFeature([],           tf.int64),
        "input_ids":   tf.io.FixedLenFeature([seq_length], tf.int64),
        "input_mask":  tf.io.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "token_map":   tf.io.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features=name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.io.parse_single_example(serialized=record, features=name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if name != 'unique_id': #t.dtype == tf.int64:
                t = tf.cast(t, dtype=tf.int32)
            example[name] = t

        return example
    decoded_data = raw_data.map(_decode_record)
    return list(decoded_data)


# In[ ]:


get_ipython().run_line_magic('time', "data1 = read_dataset('nq-test_v1.tfrecords')")
get_ipython().run_line_magic('time', "data2 = read_dataset('nq-test_v2.tfrecords')")


# In[ ]:


def neq(d1,d2,return_key=False):
    if d1.keys()!=d2.keys(): return True
    for k,v in d1.items():
        if (v.numpy()!=d2[k].numpy()).any():
            if return_key:
                return k
            return True
    return False


# In[ ]:


d1_neq_d2 = np.array([neq(d1,d2) for d1,d2 in zip(data1,data2)])


# In[ ]:


idx = d1_neq_d2.nonzero()[0]
idx

