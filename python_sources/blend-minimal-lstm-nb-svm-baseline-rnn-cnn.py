#!/usr/bin/env python
# coding: utf-8

# ## Introduction

# There are two very different strong baselines currently in the kernels for this competition:
#     
# - An *LSTM* model, which uses a recurrent neural network to model state across each text, with no feature engineering
# - An *NB-SVM* inspired model, which uses a simple linear approach on top of naive bayes features
# 
# In theory, an ensemble works best when the individual models are as different as possible. Therefore, we should see that even a simple average of these two models gets a good result. Let's try it! First, we'll load the outputs of the models (in the Kaggle Kernels environment you can add these as input files directly from the UI; otherwise you'll need to download them first).

# In[ ]:


import numpy as np, pandas as pd

f_lstm = '../input/improved-lstm-baseline-glove-dropout-lb-0-048/submission.csv'
f_nbsvm = '../input/nb-svm-strong-linear-baseline-eda-0-052-lb/submission.csv'
f_cnnrnn = '../input/keras-cnn-rnn-0-051-lb/baseline.csv'


# In[ ]:


p_lstm = pd.read_csv(f_lstm)
p_nbsvm = pd.read_csv(f_nbsvm)
p_cnnrnn = pd.read_csv(f_nbsvm)


# Now we can take the average of the label columns.

# In[ ]:


label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
p_res = p_lstm.copy()
p_res[label_cols] = (p_cnnrnn[label_cols]*1 + p_nbsvm[label_cols]*3 + p_lstm[label_cols]*6) / 10


# And finally, create our CSV.

# In[ ]:


p_res.to_csv('submission.csv', index=False)


# As we hoped, when we submit this to Kaggle, we get a great result - a 0.44 score, compared to the individual scores of 0.48 and 0.52. This is currently the best Kaggle kernel submission that runs within the kernels sandbox!
