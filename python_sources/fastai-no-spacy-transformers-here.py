#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gc
from functools import partial
from pathlib import Path

from fastai.text import *
from fastai.callbacks import *
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

# pd.set_option('display.max_colwidth', 200)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.min_rows', 100)
# pd.set_option('display.max_rows', 100)

home = Path(".")
input_dir = Path("/kaggle/input/google-quest-challenge/")

get_ipython().system('mkdir -p ~/.fastai/models/')
get_ipython().system('cp -R ../input/fastai-wt103-models/* ~/.fastai/models/')
get_ipython().system('cp ~/.fastai/models/wt103-fwd/itos_wt103.pkl ~/.fastai/models/wt103-bwd/itos_wt103.pkl')


# This notebook is to train fastai classifier with transfer learning, i.e. ULMFIT with default Spacy tokenizer.
# 
# I am using AWD-LSTM. Most other notebooks use transformers like BERT or XL, which potentially yield better results, but are slower.
# 
# Sources
# https://www.kaggle.com/melissarajaram/roberta-fastai-huggingface-transformers

# **Changes**
# 
# V52
# Return of backwards model
# 
# V51
# 
# Round of categorical labels
# 
# V48
# 
# Spacy, only fwd
# 
# V32
# 
# replace code blocks
# 
# V31
# 
# SentencePiece + backwards, test dataset uses the same tokenizer
# 
# V30
# 
# Backwards model
# 
# V29
# 
# SentencePiece tokenizer

# What didn't work:
# 
# 1. Rounding. Metrics is a correlation
# 2. Codeblocks tokenizing (replacing entire blocks with a token)
# 3. Removing columns separator

# In[ ]:


# The metric used in the competition
from scipy.stats import spearmanr

class AvgSpearman(Callback):
    def on_epoch_begin(self, **kwargs):
        self.preds = None
        self.target = None
    
    def on_batch_end(self, last_output, last_target, **kwargs):
        if self.preds is None or self.target is None:
            self.preds = last_output.cpu()
            self.target = last_target.cpu()
        else:
            self.preds = np.append(self.preds, last_output.cpu(), axis=0)
            self.target = np.append(self.target, last_target.cpu(), axis=0)
    
    def on_epoch_end(self, last_metrics, **kwargs):
        spearsum = 0
        for col in range(self.preds.shape[1]):
            spearsum += spearmanr(self.preds[:,col], self.target[:,col]).correlation
        res = spearsum / (self.preds.shape[1] + 1)
        return add_metrics(last_metrics, res)


# # Preprocess

# In[ ]:


raw_test = pd.read_csv(input_dir/"test.csv"); raw_test.tail(3)


# In[ ]:


raw_train = pd.read_csv(input_dir/"train.csv"); raw_train.tail(3)


# In[ ]:


# pd.get_dummies(raw_train, columns=class_labels)


# In[ ]:


# just to be sane

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)


# In[ ]:


labels = pd.read_csv(input_dir/"sample_submission.csv").columns[1:].to_list()
assert len(labels) == 30
text_cols = ["question_title", "question_body", "answer"] + ["host", "category", "question_user_name", "question_user_page", "answer_user_page", "answer_user_name"]


# In[ ]:


train_df = raw_train.iloc[np.random.permutation(len(raw_train))]
train_lm_df = raw_train.append(raw_test, ignore_index=True, sort=False)


# # LM
# ## Tokenize code

# In[ ]:


m_code = re.compile(r"(\n(?:[a-z  ][\s\S]*?(?: = |{|\()[\s\S]+?)+?\n)")
code = re.compile(r"(  [\s\S]+?\n){2,}",)


# In[ ]:


# train_lm_df.loc[train_lm_df.category.isin(["STACKOVERFLOW", "TECHNOLOGY"]), ["question_body"]] = train_lm_df[train_lm_df.category.isin(["STACKOVERFLOW", "TECHNOLOGY"])]["question_body"].apply(lambda x: m_code.sub(" xxcodeblock ", x))
# train_lm_df.loc[train_lm_df.category.isin(["STACKOVERFLOW", "TECHNOLOGY"]), ["answer"]] = train_lm_df[train_lm_df.category.isin(["STACKOVERFLOW", "TECHNOLOGY"])]["answer"].apply(lambda x: m_code.sub(" xxcodeblock ", x))

# train_df.loc[train_df.category.isin(["STACKOVERFLOW", "TECHNOLOGY"]), ["question_body"]] = train_df[train_df.category.isin(["STACKOVERFLOW", "TECHNOLOGY"])]["question_body"].apply(lambda x: m_code.sub(" xxcodeblock ", x))
# train_df.loc[train_df.category.isin(["STACKOVERFLOW", "TECHNOLOGY"]), ["answer"]] = train_df[train_df.category.isin(["STACKOVERFLOW", "TECHNOLOGY"])]["answer"].apply(lambda x: m_code.sub(" xxcodeblock ", x))

# raw_test.loc[raw_test.category.isin(["STACKOVERFLOW", "TECHNOLOGY"]), ["question_body"]] = raw_test[raw_test.category.isin(["STACKOVERFLOW", "TECHNOLOGY"])]["question_body"].apply(lambda x: m_code.sub(" xxcodeblock ", x))
# raw_test.loc[raw_test.category.isin(["STACKOVERFLOW", "TECHNOLOGY"]), ["answer"]] = raw_test[raw_test.category.isin(["STACKOVERFLOW", "TECHNOLOGY"])]["answer"].apply(lambda x: m_code.sub(" xxcodeblock ", x))


# In[ ]:


BS = 256


# In[ ]:


# data for the language models
tokenizer = Tokenizer(SpacyTokenizer, 'en')
processor = [TokenizeProcessor(tokenizer=tokenizer, mark_fields=True), NumericalizeProcessor()]

lm_label_list = (TextList.from_df(train_lm_df, ".", text_cols, processor=processor)
                 .split_by_rand_pct(0.1, seed=42)
                 .label_for_lm())

data_lm = lm_label_list.databunch(bs=BS)
data_lm_bwd = lm_label_list.databunch(bs=BS, backwards=True)


# In[ ]:


# data for classifiers
vocab = data_lm.vocab
BSC = 120

clas_label_list = (TextList.from_df(train_df, ".", text_cols, vocab=vocab, processor=processor)
                   .split_by_rand_pct(0.2, seed=42)
                   .label_from_df(cols=labels)
                   .add_test(TextList.from_df(raw_test, ".", text_cols, vocab=vocab, processor=processor)))

data_clas = clas_label_list.databunch(bs=BSC)
data_clas_bwd = clas_label_list.databunch(bs=BSC, backwards=True)


# ## Fine tune LM

# In[ ]:


lr = 1e-02
lr *= BS/48
moms = (0.8, 0.7)
wd=0.1
drop_mult = 0.5

def fit_lm(data, epochs=10, head_epochs=5, prefix="fwd"):
    learn = language_model_learner(data, AWD_LSTM, drop_mult=drop_mult,
                                   metrics=[accuracy, Perplexity()],
                               )
    learn = learn.to_fp16()
    learn.fit_one_cycle(head_epochs, slice(lr), moms=moms, wd=wd)
    learn.unfreeze()
    learn.save(f"{prefix}_lm_learn_1")
    learn = learn.load(f"{prefix}_lm_learn_1")
    learn.fit_one_cycle(epochs, slice(lr/100, lr/2), moms=moms, wd=wd,
                        callbacks=[SaveModelCallback(learn, monitor="perplexity", mode="min", name="best_model"),]
                        )
    learn.save_encoder(f"{prefix}_enc")
    learn.save(f"{prefix}_lm_model")
    return learn


# In[ ]:


# learn = language_model_learner(data_lm, AWD_LSTM, config, drop_mult=1.0,
#                                metrics=[accuracy, Perplexity()],
#                                )
# learn.lr_find()
# learn.recorder.plot()


# In[ ]:


# 1/0
# learn.purge();
# gc.collect()


# In[ ]:


learn = fit_lm(data_lm)


# In[ ]:


learn = fit_lm(data=data_lm_bwd, prefix="bwd")


# # Train Classifier

# In[ ]:


lr = 5e-02
lr *= BSC/48  # Scale learning rate by batch size
moms = (0.8, 0.7)
wd=0.1

def fit(data, prefix="fwd", epochs=20, epochs_1=2):
    learn = text_classifier_learner(data, AWD_LSTM,
                                    pretrained=False,
                                    metrics=[AvgSpearman()],
                                    ).to_fp16()
    learn.load_encoder(f"{prefix}_enc");
    learn.fit_one_cycle(epochs_1, lr, moms=moms, wd=wd)

    learn.freeze_to(-2)
    learn.save("learn")
    learn = learn.load("learn")
    learn.fit_one_cycle(2, slice(lr/(2.6**4),lr), moms=moms, wd=wd)

    learn.freeze_to(-3)
    learn.save("learn")
    learn = learn.load("learn")
    learn.fit_one_cycle(2, slice(lr/2/(2.6**4),lr/2), moms=moms, wd=wd)

    learn.unfreeze()
    learn.save(f"{prefix}_learn")
    learn = learn.load(f"{prefix}_learn")
    learn.fit_one_cycle(epochs, slice(lr/10/(2.6**4),lr/10), moms=moms, wd=wd,
                        callbacks=[SaveModelCallback(learn, monitor="avg_spearman", mode="max", name="best_model")]
                        )
    learn.save(f"{prefix}_learn_4")
    learn = learn.load(f"{prefix}_learn_4")
    return learn


# In[ ]:


learn = fit(data_clas)


# In[ ]:


learn_bwd = fit(data=data_clas_bwd, prefix="bwd")


# # Submission

# In[ ]:


sample_submission = pd.DataFrame(columns=["qa_id"]+labels)


# In[ ]:


def spearm(preds, target):
    spearsum = 0
    for col in range(preds.shape[1]):
        spearsum += spearmanr(preds[:,col], target[:,col]).correlation
    return spearsum / (preds.shape[1] + 1)


# In[ ]:


preds, target = learn.get_preds(DatasetType.Valid, ordered=True)
preds_b, _ = learn_bwd.get_preds(DatasetType.Valid, ordered=True)
spearm((preds+preds_b)/2, target)


# In[ ]:


test_preds, _ = learn.get_preds(DatasetType.Test, ordered=True)
test_preds_b, _ = learn_bwd.get_preds(DatasetType.Test, ordered=True)
preds_avg = (test_preds+test_preds_b)/2


# In[ ]:


sample_submission.loc[:, "qa_id"] = raw_test["qa_id"]
# sample_submission.loc[:, labels] = test_preds
sample_submission.loc[:, labels] = preds_avg
sample_submission.loc[:, labels] = np.clip(sample_submission.loc[:, 1:], 0.00001, 0.999999)


# In[ ]:


sample_submission.to_csv("submission.csv", index=False)


# In[ ]:


sample_submission.tail()


# In[ ]:




