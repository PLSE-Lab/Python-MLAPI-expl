#!/usr/bin/env python
# coding: utf-8

# As it has been pointed out (like in [this amazing kernel](https://www.kaggle.com/friedchips/on-markov-chains-and-the-competition-data) by [Markus F](https://www.kaggle.com/friedchips), the signals in this competition data are simple Markov processes combined with Gaussian noise. So inferring underlying parameters of the Markov process would be benefitial to make a prediction.
# 
# Here I used **the HMM (Hidden Markov Model)** to infer the discrete hidden states from the signal to see if those states are correlated with the open channels. The answer seems yes. This kernel is another demonstration that the signals in this competition data are simple Markov processes and the underlying parameters seem easily reconstructable.

# In[ ]:


import numpy as np
import pandas as pd
from hmmlearn import hmm
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix

# visualize
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
from matplotlib import pyplot
from matplotlib.ticker import ScalarFormatter
sns.set_context("talk")
style.use('fivethirtyeight')


# Let's load the data and split it into the 10 separate measurement sequences. I use cleaned data from [this kernel](https://www.kaggle.com/friedchips/clean-removal-of-data-drift).

# In[ ]:


# load data
df_train = pd.read_csv("../input/data-without-drift/train_clean.csv")
train_time   = df_train["time"].values.reshape(-1,500000)
train_signal = df_train["signal"].values.reshape(-1,500000)
train_opench = df_train["open_channels"].values.reshape(-1,500000)
# df_test = pd.read_csv("../input/data-without-drift/test_clean.csv")
# test_time   = df_test["time"].values.reshape(-1,500000)
# test_signal = df_test["signal"].values.reshape(-1,500000)


# In[ ]:


# sample data for quick test
train_time = train_time[:, ::100]
train_signal = train_signal[:, ::100]
train_opench = train_opench[:, ::100]


# In[ ]:


train_signal.shape


# In[ ]:


# fit HMM and estimate hidden states
hidden_states = train_signal.copy()
hmm_models = []
for i in tqdm(np.arange(train_signal.shape[0])):
    print(f"batch {i} ==================")
    model = hmm.GaussianHMM(n_components=len(np.unique(train_opench[i])), covariance_type="full", n_iter=500)
    model.fit(train_signal[i].reshape(-1, 1))
    hidden_states[i, :] = model.predict(train_signal[i].reshape(-1, 1))
    hmm_models.append(model)


# In[ ]:


# hidden states vs open channels
fig, ax = plt.subplots(5, 2, figsize=(12, 28))
ax = ax.flatten()

for i in np.arange(10):
    cm = confusion_matrix(train_opench[i, :], hidden_states[i, :])
    sns.heatmap(cm, annot=True, lw=1, ax=ax[i])
    ax[i].set_xlabel("hidden states")
    ax[i].set_ylabel("open channels")
    ax[i].set_title(f"batch {i}")
plt.tight_layout()

