#!/usr/bin/env python
# coding: utf-8

# # Jaccard Expectation Maximization

# That's notebook with implementation for some post process method : Jaccard Expectation Maximization (JEM). 
# 
# Topic with explantions: https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/158613.

# In[ ]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import tokenizers
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# # Data Preparation
# 
# I just take one of my prediction for testing strategy, you can fill this part of notebook with the same data. We need the next ones:
# - oof start/end/selected_text prediction (+ oof tweet text)
# - tokenizer, that you used in training time
# - [optional]: I also use splitter for recover correct indexes for oof prediction

# In[ ]:


def read_train():
    train=pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
    train['text']=train['text'].astype(str)
    train['selected_text']=train['selected_text'].astype(str)
    return train

def read_test():
    test=pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
    test['text']=test['text'].astype(str)
    return test
    
def read_submission():
    test=pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')
    return test

train_df = read_train()
test_df = read_test()
submission_df = read_submission()

train_df = read_train()
test_df = read_test()

# there was one NaN value inside tweets in train_df
assert train_df["text"].isna().sum() <= 1
train_df["text"] = train_df["text"].fillna("")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)
splits = list(skf.split(np.arange(len(train_df)), train_df.sentiment.values))
val_inds_arr = [val_inds for tr_inds, val_inds in splits]
val_inds_arr


# In[ ]:


N_FOLDS = 5

def get_union_df(name="train_prediction", inds_arr=None, agg_f=None):
    """ function for gathering results from each fold (for test with aggregation (agg_f) and for oof without one) """
    df = DataFrame()
    for n_fold in range(N_FOLDS):
        fold_df = (
            pd
            .read_csv("../input/bestoofprediction/{}_{}.csv".format(name, n_fold + 1))
            .drop("Unnamed: 0", axis=1)
        )

        if inds_arr is not None:
            fold_df.index = inds_arr[n_fold]

        df = pd.concat([df, fold_df])

    if agg_f:
        df = df.astype(np.float32)
        df = df.groupby(df.index).agg(agg_f)
        
    return df.sort_index()


# In[ ]:


oof_start_proba = get_union_df(name="validation_start_prediction", inds_arr=val_inds_arr)
oof_end_proba = get_union_df(name="validation_end_prediction", inds_arr=val_inds_arr)

oof_start_proba.shape, oof_end_proba.shape


# In[ ]:


oof_start_proba.head()


# Load tokenizer and get the oof prediction with (oof_start_prediction, oof_end_prediction) tuple:

# In[ ]:


def jaccard(str1, str2): 
    a = set(str(str1).lower().split()) 
    b = set(str(str2).lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def get_pred(start_proba, end_proba, df, tokenizer):
    pred = []
    n_samples = len(start_proba)
    for i in range(n_samples):
        text = df['text'][df.index[i]]
        a, b = np.argmax(start_proba[i]), np.argmax(end_proba[i])
        if a > b: 
            pred_ = text # IMPROVE CV/LB with better choice here
        else:
            cleaned_text = " " + " ".join(text.split())
            encoded_text = tokenizer.encode(cleaned_text)
            pred_ids = encoded_text.ids[a - 2: b - 1]
            pred_ = tokenizer.decode(pred_ids)
        pred += [pred_]

    return pred


# In[ ]:


PATH = '../input/tf-roberta/'
tokenizer = tokenizers.ByteLevelBPETokenizer(
    vocab_file=PATH+'vocab-roberta-base.json', 
    merges_file=PATH+'merges-roberta-base.txt',
    lowercase=True,
    add_prefix_space=True
)


# In[ ]:


train_df["pred_selected_text"] = get_pred(oof_start_proba.values, oof_end_proba.values, train_df, tokenizer)
train_df["jaccard"] = train_df.apply(lambda row: jaccard(row["selected_text"], row["pred_selected_text"]), axis=1)
train_df = train_df.sort_values("jaccard")

train_df.head()


# Also compute the model confidence:

# In[ ]:


train_df = train_df.sort_index()
train_df["confidence"] = 0.5 * (oof_start_proba.max(1) + oof_end_proba.max(1))
train_df.head()


# In[ ]:


oof_score = train_df["jaccard"].mean()
print(f'oof score before optimization: {oof_score:.5f}')


# # JEP Implementation

# Now, let's implement new approach. At first, implement getting **hypo_df** table with next fields:
# - **start** - start index prediction
# - **end** - end index prediction
# - **proba** - prediction probability : $0.5 \cdot (start\_proba + end\_proba)$ (but can use other functions)
# 
# sorted by **proba** in decreasing oreder. For compuatinal effectiveness we can compute only first **beam_size** rows (**beam_size** = 100 by default).

# In[ ]:


def get_hypo_df(start_proba, end_proba, beam_size=100):
    start2top_proba = Series(start_proba).sort_values(ascending=False)[:beam_size]
    end2top_proba   = Series(end_proba  ).sort_values(ascending=False)[:beam_size]

    hypos = []
    for start, start_proba in start2top_proba.items():
        for end, end_proba in end2top_proba.items():
            proba = 0.5 * (start_proba + end_proba)
            hypos += [(start, end, proba)]

    return DataFrame(hypos, columns=["start", "end", "proba"]).sort_values("proba")[::-1]


# Example of **hypo_df** computation for some sample:

# In[ ]:


ind = 0
start_proba = oof_start_proba.values[ind,]
end_proba   = oof_end_proba.values  [ind,]

text_len = max((start_proba != 0).sum(), (end_proba != 0).sum())
start_proba = start_proba[:text_len + 5]
end_proba   =   end_proba[:text_len + 5]

start_proba.shape, end_proba.shape


# In[ ]:


hypo_df = get_hypo_df(start_proba, end_proba, beam_size=10)

hypo_df


# Now implement JEPP with next algorithm (example in topic https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/158613):
# 1. Compute **hypo_df**.
# 2. With **hypo_df** for each row compute jaccard expectation with assumption, that correct selected_text is the one, that corresponds this row.
# 3. Find row that maximize jaccard expectation.

# In[ ]:


def get_jaccard_expectation(start0, end0, hypo_df, encoded_text, tokenizer):
    jaccard_expectation_logit, logit_sum = 0, 0
    selected_text0 = tokenizer.decode(encoded_text[start0 - 2: end0 - 1])
    for start, end, logit in zip(hypo_df["start"], hypo_df["end"], hypo_df["proba"]):
        selected_text  = tokenizer.decode(encoded_text[start - 2: end - 1])
        if (start <= end) and (start >= 2) and (selected_text.strip() != ''):
            jaccard_val = jaccard(selected_text0, selected_text)
            jaccard_expectation_logit += logit * jaccard_val
            logit_sum += logit

    return jaccard_expectation_logit / logit_sum


# In[ ]:


def get_best_selected_text(ind, oof_start_prediction, oof_end_prediction, train_df, tokenizer, beam_size=1):
    text = " " + " ".join(train_df["text"][ind].split())
    encoded_text = tokenizer.encode(text).ids
    text_len = len(encoded_text)
    
    start_proba = oof_start_prediction.values[ind,][:text_len + 5]
    end_proba   = oof_end_prediction.values  [ind,][:text_len + 5]
    hypo_df = get_hypo_df(start_proba, end_proba, beam_size=100)

    max_jaccard_expectation, best_selected_text = 0, ""
    for start0, end0 in zip(hypo_df["start"][:beam_size], hypo_df["end"][:beam_size]):
        selected_text0  = tokenizer.decode(encoded_text[start0 - 2: end0 - 1])
        if (start0 <= end0) and (start0 >= 2) and (selected_text0.strip() != ''):
            jaccard_expectation = get_jaccard_expectation(start0, end0, hypo_df, encoded_text, tokenizer)
            if jaccard_expectation > max_jaccard_expectation:
                max_jaccard_expectation = jaccard_expectation
                best_selected_text = tokenizer.decode(encoded_text[start0 - 2: end0 - 1])

    return max_jaccard_expectation, best_selected_text


# Compute all predictions (**pred_selected_text2**) and corresponding JEM (**confidence2**):

# In[ ]:


N_SAMPLES = 30000

max_jaccard_expectations, selected_texts = [], []
for ind in tqdm(train_df[:N_SAMPLES].index):
    max_jaccard_expectation, selected_text =  get_best_selected_text(ind, oof_start_proba, oof_end_proba, train_df, tokenizer, beam_size=2)
    max_jaccard_expectations += [max_jaccard_expectation]
    selected_texts += [selected_text]


# In[ ]:


used_train_df = train_df[:N_SAMPLES].copy()
used_train_df["pred_selected_text2"] = selected_texts
used_train_df["confidence2"] = max_jaccard_expectations
used_train_df["jaccard2"] = used_train_df.apply(lambda row: jaccard(row["selected_text"], row["pred_selected_text2"]), axis=1)
used_train_df.sort_values("confidence2", ascending=False)


# We can use differen strategies to apply JEM:
# - for samples with small confidence
# - for samples with large confidence2
# - for samples with small confidence2
# - for samples with confidence < **thresh** * confidence2 for some threshold **thresh**
# 
# You can use each of them, but for my case 3rd strategy is the best. Find best treshold:

# In[ ]:


old_jaccards = {}
new_jaccards = {}

for thresh in np.arange(0.1, 1.1, 0.1):
    old_jaccard = used_train_df[used_train_df["confidence2"] < thresh]["jaccard" ].mean()
    new_jaccard = used_train_df[used_train_df["confidence2"] < thresh]["jaccard2"].mean()
    old_jaccards[thresh] = old_jaccard
    new_jaccards[thresh] = new_jaccard


# In[ ]:


plt.figure(figsize=(16, 8))
plt.title("Jaccard Curves")

plt.plot(list(old_jaccards.keys()), list(old_jaccards.values()), label="old jaccard")
plt.plot(list(new_jaccards.keys()), list(new_jaccards.values()), label="new jaccard")
_ = plt.legend()


# In[ ]:


best_thresh = (Series(new_jaccards) - Series(old_jaccards)).idxmax()
best_thresh


# And compute the boost:

# In[ ]:


def get_best_prediction(row):
    if row["confidence2"] < best_thresh:
        return jaccard(row["selected_text"], row["pred_selected_text2"])
    return jaccard(row["selected_text"], row["pred_selected_text"])   

used_train_df["best_jaccard"] = used_train_df.apply(lambda row: get_best_prediction(row), axis=1)


# In[ ]:


old_oof_score = used_train_df["jaccard"].mean()
new_oof_score = used_train_df["best_jaccard"].mean()
print(f'oof score before optimization: {old_oof_score:.5f}')
print(f'oof score after optimization: {new_oof_score:.5f}')


# Boost is not so high, but it's diffrenet for different probability predictions. Hope, it helps.
