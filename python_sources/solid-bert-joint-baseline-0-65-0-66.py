#!/usr/bin/env python
# coding: utf-8

# It's mostly the same BERT-joint [pipeline](https://github.com/google-research/language/blob/master/language/question_answering/bert_joint/run_nq.py) by Google Research team but with some insights from the [paper](https://arxiv.org/abs/1909.05286) by IBM team "Frustratingly Easy Natural Question Answering"
# 
# Main points:
# 1. BERT-large WWM uncased as an initial checkpoint
# 1. the model is first fine-tuned with SQuAD 2.0 data achieving 85.2% / 82.2% (F1/exact)
# 1. we're returning answer type (null, yes, no, short, long) probabilities as well, that's also used at inference 
# 1. maximal sequence length is increased to a maximum (for BERT) of 512
# 1. maximal query length is lowered to 24 due to short questions in NQ dataset (max 17 words for the dev set). This allows more place for candidate texts
# 1. `doc_stride` is set to 192 following the experiments reported in the paper by IBM
# 1. tokenization is done faster with LRU cache
# 1. the total number of n-best predictions to consider is increased to 20 (fixing a funny [bug](https://github.com/google-research/language/blob/master/language/question_answering/bert_joint/run_nq.py#L1160) in the original code where `n_best_size` is overwritten with a local variable)
# 1. thresholds are tuned twice (see below)
# 
# Actually, we noticed that the metric is not too heavily dependent on score thresholds, small perturbations are fine
# 
# <img src="https://habrastorage.org/webt/ka/no/b0/kanob0kktor4pnmyy5uaqwfyylu.png" width=50% />
# 
# And of course coming up with a right metric was important - actually, it's just `nq_eval` with 1 more line :)
# 
# <img src="https://habrastorage.org/webt/ar/12/g0/ar12g0cea9fnojk_ghhjs2wyyae.png" width=70% />
# 
# **This approach leads to 68.2 / 56.7 / 63.5 dev scores (long/short/all F1), 65/66 public/private LB.**
# 
# This Notebook describes only our best single model, @ddanevskiy is going to outline our whole solution.
# 
# Thi is a short Notebook, most of the code lives in the modified `run_nq` script from [this shared Dataset](https://www.kaggle.com/kashnitsky/bert-wwm-063065-checkpoint) that I use in the Notebook. 
# 

# In[ ]:


import sys
import json
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import time
from contextlib import contextmanager


# **Nice way to report running times**

# In[ ]:


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


# **Paths to pretrained models, configs, data etc.**

# In[ ]:


# input
PATH_TO_DATA = '../input/tensorflow2-question-answering/'
PATH_TO_CUSTOM_BERT = '../input/bert-wwm-063065-checkpoint/'
PATH_TO_CUSTOM_BERT_WEIGHTS = PATH_TO_CUSTOM_BERT +     '20200109_bert_joint_wwm_output/20200109_bert_joint_wwm_output/'
CKPT_NAME = PATH_TO_CUSTOM_BERT_WEIGHTS + '20200109_bert_joint_wwm_output_model.ckpt-15458'
PATH_TO_TF_WHEELS = PATH_TO_CUSTOM_BERT +     'tensorflow_gpu_1_13_1_with_deps_whl/tensorflow_gpu_1_13_1_with_deps_whl/'

# output
OUT_PREDICT_JSON = 'simplified-nq-test-pred.json'


# **Constants**

# In[ ]:


# see comments below
LRU_CACHE_SIZE = 30000
MAX_SEQ_LEN = 512 
DOC_STRIDE = 192           
MAX_QUERY_LEN = 24
# score thresholds are set with the dev set and `nq_eval`
LONG_ANS_THRES = 4.4524
SHORT_ANS_THRES =  7.7251
# I did two iterations of threshold setting with dev set
LONG_ANS_THRES_FINAL = 2.1165
SHORT_ANS_THRES_FINAL =  7.6657


# **Adding `document_tokens` to simplified test data so that we can reuse models trained with originally formatted NQ data.**
# 
# This can be done in the `run_nq.py` of course, just didn't refactor this.

# In[ ]:


def process_test_set_for_bert_joint():
    with open(PATH_TO_DATA + 'simplified-nq-test.jsonl') as f_in,         open('simplified-nq-test_with_doc_tokens.jsonl', 'w') as f_out:
            for i, line in tqdm(enumerate(f_in)):
                json_line = json.loads(line)
                json_line["document_tokens"] = []
                for token in json_line['document_text'].split(' '):
                    json_line["document_tokens"].append({"token":token, 
                                             "start_byte": -1, 
                                             "end_byte": -1, 
                                             "html_token": '<' in token})
                    json_line['annotations'] = []
                    json_line["document_title"] =                         json_line["document_tokens"][0]["token"]

                f_out.write(json.dumps(json_line) + '\n')
             
    get_ipython().system('gzip simplified-nq-test_with_doc_tokens.jsonl')


# **We'll be using TF 1.13.1, so installing it with dependencies from a dataset**

# In[ ]:


def setup_tensorflow_1_13_1():
    # Install `tensorflow-gpu==1.13.1` from pre-downloaded wheels
    get_ipython().system('pip install --no-deps $PATH_TO_TF_WHEELS/*.whl > /dev/null 2>&1')
    # Install custom library google-language + dependencies
    get_ipython().system('cp -r $PATH_TO_CUSTOM_BERT/bert-tensorflow-1.0.1/ . ')
    get_ipython().system('cd bert-tensorflow-1.0.1/bert-tensorflow-1.0.1/; python setup.py install > /dev/null 2>&1')


# **The crucial part - inference**
# 
# Sorry for this ugly mixture of Python and Bash but it's handy :)

# In[ ]:


def run_bert_inference():
    get_ipython().system('python $PATH_TO_CUSTOM_BERT/run_nq.py    --logtostderr             --bert_config_file=$PATH_TO_CUSTOM_BERT/bert_config.json             --vocab_file=$PATH_TO_CUSTOM_BERT/vocab-nq.txt                        --tokenizer_cache_size=$LRU_CACHE_SIZE                                 --max_seq_length=$MAX_SEQ_LEN                                           --doc_stride=$DOC_STRIDE                                                 --max_query_length=$MAX_QUERY_LEN                                         --init_checkpoint=$CKPT_NAME                                               --predict_file=simplified-nq-test_with_doc_tokens.jsonl.gz                  --do_predict                                                                 --output_dir=bert_model_output                                                --output_prediction_file=simplified-nq-test-pred.json                          > /dev/null 2>&1    ')
    
    # cleaning up
    get_ipython().system('rm -rf bert-tensorflow-1.0.1/')
    get_ipython().system('rm simplified-nq-test_with_doc_tokens.jsonl.gz')
    get_ipython().system('rm -rf bert_model_output')


# **Here we account for answer types and tune thresholds twice - before and after that**
# 
# Answer types:
# 
# - 0 - "no-answer" otherwise (null instances)
# - 1 - "yes" for "yes" annotations where the instance contains the long answer span
# - 2 - "no" for "no" annotations where the instance contains the long answer span
# - 3 - "short" for instances that contain all annotated short spans
# - 4 -  "long" when the instance contains the long answer span but there is no short or yes/no answer

# In[ ]:


def postprocess_predictions(pred_json_path=OUT_PREDICT_JSON,
                            long_thres=LONG_ANS_THRES,
                            short_thres=SHORT_ANS_THRES):
    
    empty_answer = {'candidate_index': -1,
                'end_byte': -1,
                'end_token': -1,
                'start_byte': -1,
                'start_token': -1}
    
    with open(pred_json_path) as f:
        pred_json = json.load(f)
    
    pred_json_processed = {'predictions': []}

    for i, entry in enumerate(pred_json['predictions']):

        entry_copy = entry.copy()
        ans_type = entry['answer_type']

        if entry['long_answer_score'] < long_thres:
            entry_copy['long_answer'] = empty_answer
        if entry['short_answers_score'] < short_thres:
            entry_copy['short_answers'] = []

        if ans_type== 0: # null
            entry_copy['long_answer'] = empty_answer
            entry_copy['short_answers'] = [empty_answer]
        elif ans_type == 1: # yes
            entry_copy['yes_no_answer'] = "YES"
            entry_copy['short_answers'] = [empty_answer]
        elif ans_type== 2: # no
            entry_copy['yes_no_answer'] = "NO"
            entry_copy['short_answers'] = [empty_answer]
        elif ans_type == 3: # short
            entry_copy['yes_no_answer'] = "NONE"
        elif ans_type == 4: # long but no short or yes/no
            entry_copy['yes_no_answer'] = "NONE"
            entry_copy['short_answers'] = [empty_answer]

        pred_json_processed['predictions'].append(entry_copy)
        
    return pred_json_processed


# **Convert JSON file with prediction into competition submission CSV file**

# In[ ]:


def form_submission_file(pred_json,
                         long_thres=LONG_ANS_THRES,
                         short_thres=SHORT_ANS_THRES):

    example_ids, preds = [], []

    for entry in pred_json['predictions']:

        example_ids.append(str(entry['example_id']) + '_long')
        example_ids.append(str(entry['example_id']) + '_short')

        score = entry['long_answer_score']
        
        if score >= long_thres:
            long_pred = '{}:{}'.format(entry['long_answer']['start_token'],
                                   entry['long_answer']['end_token'])
            if long_pred == '-1:-1':
                long_pred = ""
        else:
            long_pred = ""
        
        if entry['yes_no_answer'] != "NONE":
            short_pred = entry['yes_no_answer']
        elif score >= short_thres:
            if entry['short_answers']:
                short_pred = '{}:{}'.format(entry['short_answers'][0]['start_token'],
                                   entry['short_answers'][0]['end_token'])
            else:
                short_pred = ""
            if short_pred == '-1:-1':
                short_pred = ""
        else:
            short_pred = ""

        preds.extend([long_pred, short_pred])
    
    sub_df = pd.DataFrame({'example_id':example_ids, 
                           'PredictionString': preds})\
            .sort_values(by='example_id').reset_index(drop=True)
    
    return sub_df  


# **Assemble it all**

# In[ ]:


with timer('ALL'):
    with timer('Processing test set'):
        process_test_set_for_bert_joint()
    with timer('Setting up packages'):
        setup_tensorflow_1_13_1()
    with timer('Running inference'):
        run_bert_inference()
    with timer('Forming final submission file'):
        pred_json_processed = postprocess_predictions(OUT_PREDICT_JSON,
                                                     long_thres=LONG_ANS_THRES,
                                                      short_thres=SHORT_ANS_THRES
                                                     )
        sub_df = form_submission_file(pred_json_processed,
                                      long_thres=LONG_ANS_THRES_FINAL,
                                      short_thres=SHORT_ANS_THRES_FINAL
                                     )
        sub_df.to_csv("submission.csv", index=False)

