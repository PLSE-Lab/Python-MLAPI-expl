#!/usr/bin/env python
# coding: utf-8

# # Factorization machine implemented in PyTorch

# Hi! In this tutotial I want to discuss libFFM algorithm and share my implementation of this algorithm in PyTorch. This tutorial should be considered as an extension to already published tutorial ["Factorization Machines" (Russian)](https://nbviewer.jupyter.org/github/Yorko/mlcourse_open/blob/master/jupyter_russian/tutorials/factorization_machines_sygorbatyuk.ipynb), however, my goal is different: to show that given a paper with mathematical description of a model and PyTorch we can easily implement the model all by ourselves.

# The original context for creation of factorization machines was recommender system: for instance, recommend a movie to a customer based on his ratings to other movies. However, I want to take a look on this algorithm from another point view: Factorization Machines can be considered as an extension of linear model that additionally incorporate information about features interactions (in an efficient way). In linear models we just compute a weighted sum of predictors and do not take into account interactions among features e.g. $x_i^{(k)} x_j^{(k)}$ for two features, where $i, j$ - feature indices and $k$ is an index of object in the trainset. However, the number of pairwise interactions scales quadratically with the number of features, so for 1000 features we get 1000000 interactions. Needless to say, it is computationally inefficient to compute weights for each interactions, moreover a model with a large number of parameters is prone to overfitting. Factorization Machines use an elegant trick: find vectors for each feature and compute interaction weight as a dot product of two those features i.e. we **factorize** interactions weight matrix $W \in \mathbb{R}^{n \times n}$ as a product $VV^{^T}$, where $V \in \mathbb{R}^{n \times k}$.

# ## The FM model definition

# The factorization machine with pairwise interactions is defines as:
# $$\hat{y}(x) = w_0 + \sum_{i=1}^{n}w_i x_i + \sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle \textbf v_i, \textbf v_j \rangle x_i x_j$$

# The first two terms is just a linear model (or, in the Deep Learning lingo, a linear layer). The last term can be expressed as:
# $$\sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle \textbf v_i, \textbf v_j \rangle x_i x_j = 
# \frac{1}{2} \sum_{f=1}^{k} \Big( \big(\sum_{i=1}^{n} v_f^{(i)} x_i \big)^2 - \sum_{i=1}^{n}v_f^{(i) 2} x_i^2 \Big) = 
# \frac{1}{2} \sum_{f=1}^{} \Big( S_{1,f}^2 - S_{2,f} \Big) =
# \frac{1}{2} \Big( S_{1}^2 - S_{2} \Big),
# $$

# where I used $S_1$ and $S_2$ for clarity. [](http://)Suppose we have $M$ training objects, $n$ features and we want to factorize feature interaction with vectors of size $k$ i.e. dimensionality of $v_i$. Let us denote our trainset as $X \in \mathbb{R}^{M \times n}$ , and matrix of $v_i$ (the ith row is $v_i$) as  $V \in \mathbb{R}^{n \times k}$. Also let's denote feature vector for the jth object as $\textbf x_j$. So:<br><br>
# $$
# X = \begin{bmatrix}
# x_1^{(1)} & \dots & x_n^{(1)}\\
#  \vdots \ & \ddots \ & \vdots \\ 
# x_1^{(M)} & \dots & x_n^{(M)} \\
# \end{bmatrix}
# $$
# <br><br>
# $$
# V = \begin{bmatrix}
# v_1^{(1)} & \dots & v_k^{(1)}\\
#  \vdots \ & \ddots \ & \vdots \\ 
# v_1^{(n)} & \dots & v_k^{(n)} \\
# \end{bmatrix}
# $$
# <br>
# The number in brackets indicate the index of the sample for $x$ and the index of feature for $v$.

# Let us rewrite the formula above in matrix form. Our final result should be the matrix of size $M \times 1$. We clearly see $S_1 = \sum_{i=1}^{n} v_f^{(i)} x_i $ is a dot product of feature vector  $\textbf x_j$ (a row of $X$) and the ith column of $V$. If we multiply $X$ and $V$, we get: <br><br>
# $$
# XV = \begin{bmatrix}
# \sum_{i=1}^{n} v_f^{(1)} x_i^{(1)}  & \dots &  \sum_{i=1}^{n} v_f^{(k)} x_i\\
#  \vdots \ & \ddots \ & \vdots \\ 
# \sum_{i=1}^{n} v_f^{(1)} x_i^{(M)} & \dots & \sum_{i=1}^{n} v_f^{(k)} x_i^{(M)} \\
# \end{bmatrix} = 
# \begin{bmatrix}
# S_{1,1}^{(1)}  & \dots &  S_{1,k}^{(1)}\\
#  \vdots \ & \ddots \ & \vdots \\ 
# S_{1,1}^{(M)}  & \dots & S_{1,k}^{(M)} \\
# \end{bmatrix}
# $$

# Hm, looks pretty good. So if square $XV$ element-wise and then find sum of each row, we obtain vector of $S_1^2$ terms for each training sample. Also, if we first square $X$ and $V$ element-wise, then multiply them and finally sum by rows,  we'll get $S_2$ term for each training object. So, conceptually, we can express the final term like this:
# $$
# \hat{\textbf{y}}(X) = \frac{1}{2} ( square(XV) - (square(X) \times square(V) )).sum(rowwise),
# $$

# Let's translate it into PyTorch model!

# ## PyTorch model

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import os
import copy
print(os.listdir("../input"))


# In[ ]:


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# We will test our model on data from recently ended (not really ended, but you know what I mean) [mlcourse.ai: Dota 2 Winner Prediction](https://www.kaggle.com/c/mlcourse-dota2-win-prediction). We won't use all features, only binary indicators of hero_ids for each team. We will try to find if there's any "synergy" among pairs of heroes.

# We have to do two things: add a linear layer and define all matrix operations from the expression above. Addition of a linear layer in straightforward. As to factorization part, we shouldn't forget to register $V$ as a parameter of our model with `nn.Parameter` (otherwise, we won't be able to learn optimal $V$ with gradient descent). What is good about PyTorch that we don't have to bother with finding a derivative of our expression, PyTorch will do that for us!

# In[117]:


class TorchFM(nn.Module):
    def __init__(self, n=None, k=None):
        super().__init__()
        # Initially we fill V with random values sampled from Gaussian distribution
        # NB: use nn.Parameter to compute gradients
        self.V = nn.Parameter(torch.randn(n, k),requires_grad=True)
        self.lin = nn.Linear(n, 1)

        
    def forward(self, x):
        out_1 = torch.matmul(x, self.V).pow(2).sum(1, keepdim=True) #S_1^2
        out_2 = torch.matmul(x.pow(2), self.V.pow(2)).sum(1, keepdim=True) # S_2
        
        out_inter = 0.5*(out_1 - out_2)
        out_lin = self.lin(x)
        out = out_inter + out_lin
        
        return out


# You see it's not that much of code. Let's try our model.

# In[ ]:


# load train data
train_df = pd.read_csv('../input/dota-heroes-binary/dota_train_binary_heroes.csv', index_col='match_id_hash')
test_df = pd.read_csv('../input/dota-heroes-binary/dota_train_binary_heroes.csv', index_col='match_id_hash')
target = pd.read_csv('../input/dota-heroes-binary/train_targets.csv', index_col='match_id_hash')
y = target['radiant_win'].values.astype(np.float32)
y = y.reshape(-1,1)


# In[ ]:


# convert to 32-bit numbers to send to GPU 
X_train = train_df.values.astype(np.float32)
X_test = test_df.values.astype(np.float32)


# In order to train our model we have to define several functions:

# In[118]:


# To compute probalities
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# In[119]:


# for reproducibility
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()


# In[115]:


# main training function
def train_mlp(X, X_test, y, folds, model_class=None, model_params=None, batch_size=128, epochs=1,
              criterion=None, optimizer_class=None, opt_params=None,
#               clr=cyclical_lr(10000),
              device=None):
    
    seed_everything()
    models = []
    scores = []
    train_preds = np.zeros(y.shape)
    test_preds = np.zeros((X_test.shape[0], 1))
    
    X_tensor, X_test, y_tensor = torch.from_numpy(X).to(device), torch.from_numpy(X_test).to(device), torch.from_numpy(y).to(device)
    for n_fold, (train_ind, valid_ind) in enumerate(folds.split(X, y)):
        
        print(f'fold {n_fold+1}')
        
        train_set = TensorDataset(X_tensor[train_ind], y_tensor[train_ind])
        valid_set = TensorDataset(X_tensor[valid_ind], y_tensor[valid_ind])
        
        loaders = {'train': DataLoader(train_set, batch_size=batch_size, shuffle=True),
                   'valid': DataLoader(valid_set, batch_size=batch_size, shuffle=False)}
        
        model = model_class(**model_params)
        model.to(device)
        best_model_wts = copy.deepcopy(model.state_dict())
        
        optimizer = optimizer_class(model.parameters(), **opt_params)
#         scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])
        
        # training cycle
        best_score = 0.
        for epoch in range(epochs):
            losses = {'train': 0., 'valid': 0}
            
            for phase in ['train', 'valid']:
               
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                
                for batch_x, batch_y in loaders[phase]:
                    optimizer.zero_grad()
                    out = model(batch_x)
                    loss = criterion(out, batch_y)
                    losses[phase] += loss.item()*batch_x.size(0)
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        if phase == 'train':
                            loss.backward()
#                             scheduler.step()
                            optimizer.step()

                losses[phase] /= len(loaders[phase].dataset)
            
            # after each epoch check if we improved roc auc and if yes - save model
            with torch.no_grad():
                model.eval()
                valid_preds = sigmoid(model(X_tensor[valid_ind]).cpu().numpy())
                epoch_score = roc_auc_score(y[valid_ind], valid_preds)
                if epoch_score > best_score:
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_score = epoch_score
            
            if ((epoch+1) % 30) == 0:
                print(f'epoch {epoch+1} train loss: {losses["train"]:.3f} valid loss {losses["valid"]:.3f} valid roc auc {epoch_score:.3f}')
        
        # prediction on valid set
        with torch.no_grad():
            model.load_state_dict(best_model_wts)
            model.eval()
            
            train_preds[valid_ind] = sigmoid(model(X_tensor[valid_ind]).cpu().numpy())
            fold_score = roc_auc_score(y[valid_ind], train_preds[valid_ind])
            scores.append(fold_score)
            print(f'Best ROC AUC score {fold_score}')
            models.append(model)

            test_preds += sigmoid(model(X_test).cpu().numpy())
    
    print('CV AUC ROC', np.mean(scores), np.std(scores))
    
    test_preds /= folds.n_splits
    
    return models, train_preds, test_preds


# In[ ]:


folds = KFold(n_splits=5, random_state=17)


# Since our kernel is just a proof of concept, we won't optimize hyperparameters and set high learning rate.

# In[116]:


get_ipython().run_cell_magic('time', '', "MS, train_preds, test_preds = train_mlp(X_train, X_test, y, folds, \n                            model_class=TorchFM, \n                            model_params={'n': X_train.shape[1], 'k': 5}, \n                            batch_size=1024,\n                            epochs=300,\n                            criterion=nn.BCEWithLogitsLoss(),\n                            optimizer_class=torch.optim.SGD, \n                            opt_params={'lr': 0.01, 'momentum': 0.9},\n                            device=DEVICE\n                            )")


# I certainly wouldn't call it a good model, but at least it works. We see that there's indeed some link between teams composition and winning in the match.

# ## Conclusion

# In this tutorial we made our own Factorization Machine with a pinch of linear algebra and autograd magic of PyTorch.
# 
# Good luck with your own experiments!

# ## Sources

# 1. Original paper by Rendle https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
