#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
os.listdir('../input')


# In[ ]:


import numpy as np, pandas as pd

f_caps_gru = '../input/capsule-net-with-gru/submission.csv'
f_dual_embed_pl = '../input/submission-dual-embed-pl/submission_dual_embed (2).csv'
f_dual_embed_mish = '../input/bi-gru-lstm-dual-embedding-with-mish/submission.csv'
f_lstm_glove_tta = '../input/improved-lstm-baseline-glove-dropout-trainta/submission.csv'
f_dual_embed_dehyp = '../input/improved-lstm-baseline-bi-lstm-dual-embed-dehyp/submission.csv'
f_lstm_fast = '../input/improved-lstm-baseline-fasttext-dropout/submission.csv'
f_nbsvm = '../input/nb-svm-strong-linear-baseline/submission.csv'
f_bi_post = '../input/bi-post/10fold_lstmpp_am.csv'
f_dpcnn = '../input/dpcnn-wordcloud/10fold_dpcnn_test.csv'
f_dmcnn = '../input/dmcnn-demoji/10fold_dmcnn_am.csv'
f_rcn = '../input/rcn-capsule/10fold_capsule_am.csv'


# In[ ]:


p_caps_gru = pd.read_csv(f_caps_gru)
p_dual_embed_pl = pd.read_csv(f_dual_embed_pl)
p_dual_embed_mish = pd.read_csv(f_dual_embed_mish)
p_lstm_glove_tta = pd.read_csv(f_lstm_glove_tta)
p_dual_embed_dehyp = pd.read_csv(f_dual_embed_dehyp)
p_lstm_fast = pd.read_csv(f_lstm_fast)
p_nbsvm = pd.read_csv(f_nbsvm)
p_bi_post = pd.read_csv(f_bi_post)
p_dpcnn = pd.read_csv(f_dpcnn)
p_dmcnn = pd.read_csv(f_dmcnn)
p_rcn = pd.read_csv(f_rcn)


# Now we can take the average of the label columns.

# In[ ]:


label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
p_res = p_caps_gru.copy()
p_res[label_cols] = (p_caps_gru[label_cols] + p_dual_embed_pl[label_cols] + p_dual_embed_mish[label_cols] + p_lstm_glove_tta[label_cols] + p_dual_embed_dehyp[label_cols] + p_lstm_fast[label_cols] + p_nbsvm[label_cols] + p_bi_post[label_cols] * 3 + p_dpcnn[label_cols] * 3 + p_dmcnn[label_cols] * 3 + p_rcn[label_cols] * 3) / 19


# And finally, create our CSV.

# In[ ]:


p_res.to_csv('submission.csv', index=False)


# As we hoped, when we submit this to Kaggle, we get a great result - much better than the individual scores of the kernels we based off. This is currently the best Kaggle kernel submission that runs within the kernels sandbox!
