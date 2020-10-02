#!/usr/bin/env python
# coding: utf-8

# # Preparation

# Install my forck of the  *shap* package:
# 
# (The PyTorchDeepExplainer from the official master branch needs some tweaking to work)

# In[ ]:


get_ipython().system('pip install https://github.com/ceshine/shap/archive/master.zip')


# In[ ]:


import sys
import gc


# In[ ]:


import torch.nn as nn
import torch
from torch.nn.utils.weight_norm import weight_norm
from sklearn import preprocessing
import shap
import numpy as np
import joblib


# ## Load a Model

# In[ ]:


get_ipython().run_line_magic('ls', '../input/data/cache/model_cache/')


# In[ ]:


MODEL = "../input/data/cache/model_cache/snapshot_PUBG_0.02873547.pth"


# In[ ]:


class MLPModel(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.model = nn.Sequential(
            weight_norm(nn.Linear(num_features, 64)),
            nn.ELU(),
            weight_norm(nn.Linear(64, 64)),
            nn.ELU(),
            weight_norm(nn.Linear(64, 64)),
            nn.ELU(),
            weight_norm(nn.Linear(64, 64)),
            nn.ELU(),          
            weight_norm(nn.Linear(64, 1)),
        )

    def forward(self, input_tensor):
        return torch.clamp(self.model(input_tensor), 0, 1)


# In[ ]:


x_train, features = joblib.load("../input/x_train_dump.jl")


# In[ ]:


DEVICE = "cpu"
model = MLPModel(len(features)).to(DEVICE)
model.load_state_dict(torch.load(MODEL, map_location='cpu'))


# ## Deep Explainer
# 
# Here we only use a small sample (300) to save time:

# In[ ]:


get_ipython().run_cell_magic('time', '', 'e = shap.DeepExplainer(\n        model, \n        torch.from_numpy(\n            x_train[np.random.choice(np.arange(len(x_train)), 10000, replace=False)]\n        ).to(DEVICE))')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'x_samples = x_train[np.random.choice(np.arange(len(x_train)), 300, replace=False)]\nprint(len(x_samples))\nshap_values = e.shap_values(\n    torch.from_numpy(x_samples).to(DEVICE)\n)')


# In[ ]:


shap_values.shape


# ### Shap Values As a Data Frame

# In[ ]:


import pandas as pd
df = pd.DataFrame({
    "mean_abs_shap": np.mean(np.abs(shap_values), axis=0), 
    "stdev_abs_shap": np.std(np.abs(shap_values), axis=0), 
    "name": features
})
df.sort_values("mean_abs_shap", ascending=False)[:10]


# ### Plotting Overall Shap Values

# In[ ]:


shap.summary_plot(shap_values, features=x_samples, feature_names=features)

