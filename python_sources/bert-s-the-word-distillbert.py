#!/usr/bin/env python
# coding: utf-8

# ## If you like the kernel, consider upvoting it and the associated datasets:
# - https://www.kaggle.com/abhishek/transformers
# - https://www.kaggle.com/abhishek/sacremoses
# - https://www.kaggle.com/abhishek/distilbertbaseuncased
# 
# in case of any questions, feel free to ask.
# **P.S. combining this with bilstm nnet will give you 0.33+**

# In[ ]:


get_ipython().system('pip install ../input/sacremoses/sacremoses-master/ > /dev/null')


# In[ ]:


import os
import sys
import glob
import torch

sys.path.insert(0, "../input/transformers/transformers-master/")
import transformers


# In[ ]:


import numpy as np
import pandas as pd
import math
import catboost as ctb
from tqdm import tqdm_notebook


# In[ ]:


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# In[ ]:


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


# In[ ]:


def fetch_vectors(string_list, batch_size=64):
    # inspired by https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
    DEVICE = torch.device("cuda")
    tokenizer = transformers.DistilBertTokenizer.from_pretrained("../input/distilbertbaseuncased/")
    model = transformers.DistilBertModel.from_pretrained("../input/distilbertbaseuncased/")
    model.to(DEVICE)

    fin_features = []
    for data in tqdm_notebook(chunks(string_list, batch_size)):
        tokenized = []
        for x in data:
            x = " ".join(x.strip().split()[:300])
            tok = tokenizer.encode(x, add_special_tokens=True)
            tokenized.append(tok[:512])

        max_len = 512
        padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized])
        attention_mask = np.where(padded != 0, 1, 0)
        input_ids = torch.tensor(padded).to(DEVICE)
        attention_mask = torch.tensor(attention_mask).to(DEVICE)

        with torch.no_grad():
            last_hidden_states = model(input_ids, attention_mask=attention_mask)

        features = last_hidden_states[0][:, 0, :].cpu().numpy()
        fin_features.append(features)

    fin_features = np.vstack(fin_features)
    return fin_features


# In[ ]:


df_train = pd.read_csv("../input/google-quest-challenge/train.csv").fillna("none")
df_test = pd.read_csv("../input/google-quest-challenge/test.csv").fillna("none")

sample = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")
target_cols = list(sample.drop("qa_id", axis=1).columns)

train_question_body_dense = fetch_vectors(df_train.question_body.values)
train_answer_dense = fetch_vectors(df_train.answer.values)

test_question_body_dense = fetch_vectors(df_test.question_body.values)
test_answer_dense = fetch_vectors(df_test.answer.values)


# In[ ]:


xtrain = np.hstack((train_question_body_dense, train_answer_dense))
xtest = np.hstack((test_question_body_dense, test_answer_dense))


# In[ ]:


for tc in target_cols:
    print(tc)
    clf = ctb.CatBoostRegressor(task_type="GPU")
    clf.fit(xtrain, df_train[tc].values)
    preds = [sigmoid(x) for x in clf.predict(xtest)]
    sample[tc] = preds


# In[ ]:


sample.to_csv("submission.csv", index=False)


# In[ ]:


sample.head()


# In[ ]:




