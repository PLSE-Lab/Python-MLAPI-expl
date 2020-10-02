#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install torch')
get_ipython().system('pip install pytorch_transformers')


# In[ ]:


import torch
import numpy as np
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from pytorch_transformers import BertModel
from pytorch_transformers import BertTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


# In[ ]:


def plot_class(who_pd):
    who_pd        .target        .value_counts()        .plot.bar();
    
def encode_text(text):
    input_ids = torch.tensor([tokenizer.encode(text)])
    return model(input_ids)[0].mean(1)[0].detach().numpy()


# In[ ]:


model = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# In[ ]:


train = pd.read_csv('../input/nlp-getting-started/train.csv')
test = pd.read_csv('../input/nlp-getting-started/test.csv')
submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')


# In[ ]:


test.head()


# In[ ]:


plot_class(train)


# In[ ]:


pd_X_train, pd_X_valid = train_test_split(train, test_size=0.1, stratify=train.target)


# In[ ]:


plot_class(pd_X_train)


# In[ ]:


plot_class(pd_X_valid)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'X_train = np.array(list(pd_X_train.text.apply(encode_text).values))\nX_valid = np.array(list(pd_X_valid.text.apply(encode_text).values))\nX_test = np.array(list(test.text.apply(encode_text).values))')


# In[ ]:


X_train.shape


# In[ ]:


y_train = pd_X_train.target
y_valid = pd_X_valid.target


# In[ ]:


# LogisticRegression(
#     penalty='l2',
#     *,
#     dual=False,
#     tol=0.0001,
#     C=1.0,
#     fit_intercept=True,
#     intercept_scaling=1,
#     class_weight=None,
#     random_state=None,
#     solver='lbfgs',
#     max_iter=100,
#     multi_class='auto',
#     verbose=0,
#     warm_start=False,
#     n_jobs=None,
#     l1_ratio=None,
# )


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nclf_model = LogisticRegression(solver='liblinear', max_iter=2000)\nclf_model = clf_model.fit(X_train, y_train)")


# In[ ]:


print("Accuracyl:", clf_model.score(X_valid, y_valid))


# In[ ]:


clf_model.fit(np.concatenate([X_train, X_valid], axis=0), 
              np.concatenate([y_train, y_valid], axis=-1))

preds = clf_model.predict(X_test)


# In[ ]:


submission.target =  preds
submission.to_csv("submission.csv", index=False)
submission.target.value_counts().plot.bar();


# In[ ]:




