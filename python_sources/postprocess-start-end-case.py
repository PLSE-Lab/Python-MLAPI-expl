#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import tokenizers
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


N_FOLDS = 5
LEFT_PAD_LEN = 1 # some internal hyperparameter for my model


# # Abstract
# 
# Lots of public notebooks (for instance, https://www.kaggle.com/khoongweihao/tse2020-roberta-cnn-random-seed-distribution and https://www.kaggle.com/al0kharba/tensorflow-roberta-0-712) used a hypothesis, that if in your prediction start position righter end position, than your prediction for selected text equals text. Indeed, that's not the worse strategy, cause in this kind of prediction we get positive jaccard value for each true selected_text label. But obviously, we can improve this strategy. This notebook suggests one of those.

# # Data Preparation
# 
# I just take one of my prediction for testing strategy, you can fill this part of notebook with the same data. We need the next ones:
# - oof start/end/selected_text prediction (+ oof tweet text)
# - test start/end/selected_text prediction (+ test tweet text)
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

train_df = read_train()
test_df = read_test()

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


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


def get_union_df(name="train_prediction", inds_arr=None, agg_f=None):
    """ function for gathering results from each fold (for test with aggregation (agg_f) and for oof without one) """
    df = DataFrame()
    for n_fold in range(N_FOLDS):
        fold_df = (
            pd
            .read_csv("../input/predictionexample/{}_{}.csv".format(name, n_fold + 1))
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


oof_start_prediction = get_union_df(name="validation_start_prediction", inds_arr=val_inds_arr)
oof_end_prediction = get_union_df(name="validation_end_prediction", inds_arr=val_inds_arr)

oof_start_prediction.shape, oof_end_prediction.shape


# In[ ]:


test_start_prediction = get_union_df(name="test_start_prediction", agg_f="mean")
test_end_prediction = get_union_df(name="test_end_prediction", agg_f="mean")

test_start_prediction.shape, test_end_prediction.shape


# In[ ]:


oof_start_prediction.head()


# In[ ]:


test_start_prediction.head()


# # OOF Prediction
# 
# Load tokenizer and get a oof prediction with (oof_start_prediction, oof_end_prediction):

# In[ ]:


PATH = '../input/tf-roberta/'
tokenizer = tokenizers.ByteLevelBPETokenizer(
    vocab_file=PATH+'vocab-roberta-base.json', 
    merges_file=PATH+'merges-roberta-base.txt', 
    lowercase=True,
    add_prefix_space=True
)


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
            pred_ids = encoded_text.ids[a - LEFT_PAD_LEN: b - LEFT_PAD_LEN + 1]
            pred_ = tokenizer.decode(pred_ids)
        pred += [pred_]

    return pred


# In[ ]:


train_df.head()


# In[ ]:


train_df["pred_selected_text"] = get_pred(oof_start_prediction.values, oof_end_prediction.values, train_df, tokenizer)
train_df["jaccard"] = train_df.apply(lambda row: jaccard(row["selected_text"], row["pred_selected_text"]), axis=1)

train_df.head()


# In[ ]:


oof_score = train_df["jaccard"].mean()
print(f'oof score before optimization: {oof_score:.5f}')


# # Algorithm
# 
# ## Stats & Preparation
# 
# At first take a look, what percent of data was covered with particular start > end case:

# In[ ]:


oof_start_pred = oof_start_prediction.idxmax(1).astype(int)
oof_end_pred   = oof_end_prediction  .idxmax(1).astype(int)

oof_percent = (oof_start_pred > oof_end_pred).mean()
oof_count   = (oof_start_pred > oof_end_pred).sum()
print(f'[oof] start > end percent: {(100 * oof_percent):.4f}% ({oof_count} times)')


# In[ ]:


test_start_pred = test_start_prediction.idxmax(1).astype(int)
test_end_pred   = test_end_prediction  .idxmax(1).astype(int)


test_percent = (test_start_pred > test_end_pred).mean()
test_count   = (test_start_pred > test_end_pred).sum()
print(f'[test] start > end percent: {(100 * test_percent):.4f}% ({test_count} times)')


# It means, that we can improve only about 0.2% of data (for both oof and test prediction). That's tiny part of data, but it can make a difference and improve your third sign in public or private test data (for instance, at this moment score 0.714 has [181, 281] leaderboard range, 0.715 - [115, 180]). Save this part of data seprately:

# In[ ]:


bad_train_df = train_df[oof_start_pred > oof_end_pred]
bad_test_df  = test_df[test_start_pred > test_end_pred]

# as described above, we predict text as selected_case in this case
assert np.all(bad_train_df["text"] == bad_train_df["pred_selected_text"])


# ## Theory
# 
# OK, let's try to implement the next idea for start > end case:
# - Just pick a pair (start, end) as prediction among $start \leq end$ pairs with highest score $f(p_{start}, p_{end}) = \frac{p_{start} + p_{end}}{2}$. That's idea from paper about BERT for SQuAD task: https://arxiv.org/pdf/1810.04805.pdf (4.2 SQuAD v1.1 section) (point of improvement: check other functions $f$).
# - We have a deal with cases, that, honestly, our model process really bad. So, it's possible, that correct pair (start, end) will have not first rank with $f$ function, but 2-3 rank, cause of model confidence absence. So, lets just concatenate some $n\_concat$ top rank pairs (start, end) (yes, we can do that in our prediction!)

# ## Implementation
# 
# Time to coding. At first, count jaccard scores for old approach:

# In[ ]:


old_bad_oof_score = bad_train_df["jaccard"].mean()
print(f'[start > end] oof score before optimization: {old_bad_oof_score:.5f}')


# Then implement new approach:

# In[ ]:


def get_hypo_df(start_proba, end_proba, beam_size=10):
    start2top_proba = Series(start_proba).sort_values(ascending=False)[:beam_size]
    end2top_proba   = Series(end_proba  ).sort_values(ascending=False)[:beam_size]

    hypos = []
    for start, start_proba in start2top_proba.items():
        for end, end_proba in end2top_proba.items():
            proba = 0.5 * (start_proba + end_proba)
            hypos += [(start, end, proba)]

    return DataFrame(hypos, columns=["start", "end", "proba"])

def get_prediction(df, start_prediction, end_prediction, ind, tokenizer, n_concat=1):
    start_proba = start_prediction.loc[ind].values
    end_proba   = end_prediction  .loc[ind].values
    hypo_df = get_hypo_df(start_proba, end_proba)

    pred_inds = hypo_df[hypo_df["start"] <= hypo_df["end"]].sort_values("proba", ascending=False).index
    
    pred_selected_texts = []
    for pred_ind in pred_inds[:n_concat]:
        a, b = hypo_df["start"][pred_ind], hypo_df["end"][pred_ind]

        text = df["text"][ind]
        cleaned_text = " " + " ".join(text.split())
        encoded_text = tokenizer.encode(cleaned_text)
        pred_ids = encoded_text.ids[a - LEFT_PAD_LEN: b - LEFT_PAD_LEN + 1]
        pred_selected_text = tokenizer.decode(pred_ids)
        pred_selected_texts += [pred_selected_text]

    return " ".join(pred_selected_texts)


# In[ ]:


N_CONCATS = np.arange(1, 20)

new_jaccards = []
for n_concat in tqdm(N_CONCATS):
    bad_train_df["new_pred_selected_text"] = bad_train_df.index.map(lambda ind: get_prediction(
        bad_train_df,
        oof_start_prediction,
        oof_end_prediction,
        ind,
        tokenizer,
        n_concat=n_concat
    ))
    bad_train_df["new_jaccard"] = bad_train_df.apply(lambda row: jaccard(row["selected_text"], row["new_pred_selected_text"]), axis=1)

    new_jaccard = bad_train_df["new_jaccard"].mean()
    new_jaccards += [new_jaccard]


# In[ ]:


plt.figure(figsize=(16, 8))

new_jaccards = np.array(new_jaccards)
old_jaccards = new_jaccards * 0 + old_bad_oof_score
plt.plot(N_CONCATS, new_jaccards, label="new jaccard")
plt.plot(N_CONCATS, old_jaccards, label="old jaccard")

plt.legend()
plt.xlabel("# concated predictions")
_ = plt.ylabel("jaccard")


# That's good - we take a boost:

# In[ ]:


res = Series(index=N_CONCATS, data=new_jaccards)

best_n_concat = res.idxmax()
new_bad_oof_score = res.max()

print(f'[start > end] oof score before optimization: {old_bad_oof_score:.5f}')
print(f'[start > end] oof score after optimization : {new_bad_oof_score:.5f} (n_concat={best_n_concat})')


# Find a boost on whole train set:

# In[ ]:


train_df["new_pred_selected_text"] = train_df["pred_selected_text"]
bad_inds = bad_train_df.index
train_df["new_pred_selected_text"].loc[bad_inds] = bad_train_df.index.map(lambda ind: get_prediction(
    bad_train_df,
    oof_start_prediction,
    oof_end_prediction,
    ind,
    tokenizer,
    n_concat=best_n_concat
))

train_df["new_jaccard"] = train_df.apply(lambda row: jaccard(row["selected_text"], row["new_pred_selected_text"]), axis=1)


# In[ ]:


old_score = train_df["jaccard"].mean()
new_score = train_df["new_jaccard"].mean()
print(f'[start > end] oof score before optimization: {old_score:.5f}')
print(f'[start > end] oof score after  optimization: {new_score:.5f}')


# # Inference
# 
# Implement method for test set:

# In[ ]:


test_df.head()


# In[ ]:


test_df["selected_text"] = get_pred(test_start_prediction.values, test_end_prediction.values, test_df, tokenizer)
test_df["selected_text"].loc[bad_test_df.index] = bad_test_df.index.map(lambda ind: get_prediction(
    bad_test_df,
    oof_start_prediction,
    oof_end_prediction,
    ind,
    tokenizer,
    n_concat=best_n_concat
))


# In[ ]:


test_df[["textID", "selected_text"]].loc[bad_test_df.index]


# In[ ]:


test_df[["textID", "selected_text"]].to_csv('submission.csv', index=False)


# # Conclusion
# 
# This was method about improving prediction, when start prediction righter than end prediction. We get small boost on oof prediction, but it can make a difference cause of dense leaderboard. You also can try to achieve best improvement with this notebook in next directions:
# - Use this notebook for your predictions
# - Optimize $f$ function, that we use for getting pair (start, end) score.
# - Use concat method not only for start > end cases.
# - Use prediction dataset (https://www.kaggle.com/koza4ukdmitrij/predictionexample) for your own aims.
# 
# Hope it helps.
