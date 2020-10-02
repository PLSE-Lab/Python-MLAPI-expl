#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load in 

# import collections
# import enum
# import numpy as np # linear algebra
# import json
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import re
# import tensorflow.compat.v1 as tf

# # Input data files are available in the "../input/" directory.
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # Any results you write to the current directory are saved as output.


# In[ ]:


# import and flags
import albert_yes_no_fn_builder
import albert_yes_no_modeling
import albert_yes_no_utils
import bert_disjoint_fn_builder
import bert_disjoint_modeling
import bert_disjoint_utils
import create_submission
import json
import tensorflow.compat.v1 as tf
import tokenization
tf.logging.set_verbosity(tf.logging.INFO)
tf.disable_eager_execution()

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

del_all_flags(tf.app.flags.FLAGS)
flags = tf.app.flags

# bert
flags.DEFINE_string("bert_config_file", "/kaggle/input/bert-disjoint/bert_config.json", " ")
flags.DEFINE_string("vocab_file_bert", "/kaggle/input/bert-disjoint/vocab-nq.txt", " ")
flags.DEFINE_string("output_dir_bert", "/kaggle/input/bert-disjoint/", " ")
flags.DEFINE_string("init_checkpoint_bert", "/kaggle/input/bert-disjoint/model.ckpt-15000", " ")
# albert
flags.DEFINE_string("albert_config_file", "/kaggle/input/albert-yes-no/albert_config.json", " ")
flags.DEFINE_string("vocab_file_albert", "/kaggle/input/albert-yes-no/30k-clean.vocab", " ")
flags.DEFINE_string("spm_model_file", "/kaggle/input/albert-yes-no/30k-clean.model", " ")
flags.DEFINE_string("output_dir_albert", "/kaggle/input/albert-yes-no/", " ")
flags.DEFINE_string("init_checkpoint_albert", "/kaggle/input/albert-yes-no/model.ckpt-1206", " ")

flags.DEFINE_string("eval_record_file_bert", "eval_bert.tf_record", " ")
# flags.DEFINE_string("eval_record_file_albert", "eval_albert.tf_record", " ")

flags.DEFINE_integer("max_seq_length", 512, " ")
flags.DEFINE_integer("predict_batch_size", 8, " ")

## Special flags - do not change
flags.DEFINE_string(
    "predict_file", "/kaggle/input/tensorflow2-question-answering/simplified-nq-test.jsonl",
    "NQ json for predictions. E.g., dev-v1.1.jsonl.gz or test-v1.1.jsonl.gz")
flags.DEFINE_boolean("logtostderr", True, "Logs to stderr")
flags.DEFINE_boolean("undefok", True, "it's okay to be undefined")
flags.DEFINE_string('f', '', 'kernel')
flags.DEFINE_string('HistoryManager.hist_file', '', 'kernel')

FLAGS = flags.FLAGS


# In[ ]:


# ------------------ cell for extracting yes no questions ------------------
keyword_list = ["are", "is", "was", "were",
                "can", "could", "will", "would", "should",
                "did", "do", "does", "has", "had", "have"]


def is_yes_no_question(_question_text):
    _text = _question_text.lower()
    return ("true or false" in _text) or (_text.strip().split(" ")[0] in keyword_list)


with open(FLAGS.predict_file, "r") as f, open("yes_no_questions.jsonl", "w") as of:
    for line in f:
        question_text = json.loads(line)["question_text"]
        if is_yes_no_question(question_text):
            of.write(line)


# In[ ]:


# ------------------ cell for bert-disjoint stage: example -> record ------------------
bert_tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file_bert, do_lower_case=True)
eval_examples = bert_disjoint_utils.read_nq_examples(input_file=FLAGS.predict_file)
eval_writer = bert_disjoint_utils.FeatureWriter(filename=FLAGS.eval_record_file_bert, is_training=False)
eval_features = []


def append_feature(feature):
    eval_features.append(feature)
    eval_writer.process_feature(feature)


num_spans_to_ids = bert_disjoint_utils.convert_examples_to_features(examples=eval_examples,
                                                                    tokenizer=bert_tokenizer,
                                                                    output_fn=append_feature)
eval_writer.close()
eval_filename = eval_writer.filename


# In[ ]:


# ------------------ cell for bert-disjoint stage: inference ------------------
bert_config = bert_disjoint_modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
run_config = tf.estimator.RunConfig(model_dir=FLAGS.output_dir_bert)

model_fn = bert_disjoint_fn_builder.model_fn_builder(bert_config=bert_config,
                                                     init_checkpoint=FLAGS.init_checkpoint_bert)
estimator = tf.estimator.Estimator(model_fn=model_fn,
                                   model_dir=FLAGS.output_dir_bert,
                                   config=run_config,
                                   params={"batch_size": FLAGS.predict_batch_size})

predict_input_fn = bert_disjoint_fn_builder.input_fn_builder(input_file=eval_filename,
                                                             seq_length=FLAGS.max_seq_length,
                                                             drop_remainder=False)

tf.logging.info("***** Running predictions *****")
tf.logging.info("  Num orig examples = %d", len(eval_examples))
tf.logging.info("  Num split examples = %d", len(eval_features))
tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
for spans, ids in num_spans_to_ids.items():
    tf.logging.info("  Num split into %d = %d", spans, len(ids))

all_results = []
for result in estimator.predict(predict_input_fn, yield_single_examples=True):
    if len(all_results) % 1000 == 0:
        tf.logging.info("Processing example: %d" % (len(all_results)))
    unique_id = int(result["unique_ids"])
    start_logits = [float(x) for x in result["start_logits"].flat]
    end_logits = [float(x) for x in result["end_logits"].flat]
    all_results.append(bert_disjoint_utils.RawResult(unique_id=unique_id,
                                                     start_logits=start_logits,
                                                     end_logits=end_logits))

candidates_dict = bert_disjoint_utils.read_candidates(FLAGS.predict_file)
eval_features = [tf.train.Example.FromString(r) for r in tf.python_io.tf_record_iterator(eval_filename)]
nq_pred_dict = bert_disjoint_utils.compute_pred_dict(candidates_dict, eval_features, [r._asdict() for r in all_results])
predictions = list(nq_pred_dict.values())

# ------------------ clean up ------------------
del bert_config, run_config, model_fn, estimator, predict_input_fn
del all_results, candidates_dict, eval_features, nq_pred_dict


# In[ ]:


# ------------------ cell for yes-no correction stage: example -> record ------------------
albert_tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file_albert,
                                              do_lower_case=True,
                                              spm_model_file=FLAGS.spm_model_file)
eval_examples = albert_yes_no_utils.read_nq_examples(input_file="yes_no_questions.jsonl")
eval_writer = albert_yes_no_utils.FeatureWriter(filename="yes_no_questions.eval.tf_record",
                                                albert_tokenizer=albert_tokenizer)
n_features, example_to_feature_map, feature_to_example_map =     eval_writer.file_based_convert_examples_to_features_span(eval_examples)


# In[ ]:


# ------------------ cell for yes-no correction stage: inference ------------------
albert_config = albert_yes_no_modeling.AlbertConfig.from_json_file(FLAGS.albert_config_file)
run_config = tf.estimator.RunConfig(model_dir=FLAGS.output_dir_albert)

model_fn = albert_yes_no_fn_builder.model_fn_builder(albert_config=albert_config,
                                                     init_checkpoint=FLAGS.init_checkpoint_albert)

estimator = tf.estimator.Estimator(model_fn=model_fn,
                                   model_dir=FLAGS.output_dir_albert,
                                   config=run_config,
                                   params={"batch_size": FLAGS.predict_batch_size})

predict_input_fn = albert_yes_no_fn_builder.input_fn_builder(input_file="yes_no_questions.eval.tf_record",
                                                             seq_length=FLAGS.max_seq_length,
                                                             drop_remainder=False)

tf.logging.info("  Num features: %d" % n_features)
all_results = []
for result in estimator.predict(predict_input_fn, yield_single_examples=True):
    if len(all_results) % 100 == 0:
        tf.logging.info("Processing example: %d" % (len(all_results)))

    answer_type_logits = [float(x) for x in result["answer_type_logits"].flat]
    all_results.append(answer_type_logits)

yes_no_pred_dict = albert_yes_no_utils.compute_pred_dicts_cls("yes_no_questions.jsonl",
                                                              all_results,
                                                              example_to_feature_map,
                                                              feature_to_example_map)

# ------------------ clean up ------------------
del albert_config, run_config, model_fn, estimator, predict_input_fn
del all_results


# In[ ]:


# ------------------ final cell for creating submission ------------------
create_submission.merge_predictions(predictions, yes_no_pred_dict)
create_submission.create_answers(predictions, "submission.csv")

