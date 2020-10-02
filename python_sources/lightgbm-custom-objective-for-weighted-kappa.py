#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# This code in this kernel is for a custom loss function for LightGBM based on the paper https://www.sciencedirect.com/science/article/abs/pii/S0167865517301666.
# 
# William Wu published a kernel, https://www.kaggle.com/wuwenmin/dnn-and-effective-soft-quadratic-kappa-loss, using the same paper with an application to tensorflow.  He asks in this thread, https://www.kaggle.com/c/data-science-bowl-2019/discussion/121606, if anyone has used the weighted kappa loss and this is a what I have used.
# 
# Using this loss performs worse than regression in my limited tests.  It performs slightly better on validation when compared to LightGBM's multiclass objective.
# 
# The kernel is organized as follows:
# 1. Background on the objective function
# 2. Calculations of the gradient and hessian
# 3. The code for custom objective for the scikit learn fit function
# 4. Check derivatives using autograd to see if the gradients and second derivatives match
# 

# ## Background
# William has an excellent summary of the weighted kappa loss in his kernel using the notation from the paper.  This will be using similar notation.  Here is a quick overview of the loss function.  
# 
# $$
# \text{ minimize  } \mathscr{L}=\log (1-\kappa) 
# $$
# where
# $$
# \kappa = 1 - \frac{\mathscr{N}}{\mathscr{D}}
# $$
# 
# The paper redefines the numerator and denominator in terms of softmax output probabilities.
# 
# $\mathscr{N}=\sum_{i, j} \omega_{i, j} O_{i, j}=\sum_{k=1}^{N} \sum_{c=1}^{C} \omega_{t_{k}, c} P_{c}(X_{k})$
# 
# $\mathscr{D}=\sum_{i, j} \omega_{i, j} E_{i, j}=\sum_{i=1}^{C} \hat{N}_{i} \sum_{j=1}^{C}\left(\omega_{i, j} \sum_{k=1}^{N} P_{j}(X_{k})\right)$
# 
# where
# 
# $X_{k}$: input data of the k-th sample
# 
# $E_{i, j}=\frac{N_{i} \sum_{k=1}^{N} P_{j}(X_{k})}{N}= \hat{N}_{i} \sum_{k=1}^{N} P_{j}(X_{k})$
# 
# $N$: number of samples
# 
# $N_i$: number of samples of the i-th class 
# 
# $\hat{N}_{i}=\frac{N_{i}}{N}$
# 
# $t_k$: correct class number for sample k
# 
# $P_{c}\left(X_{k}\right)$: conditional probability that the k-th sample belongs to class c given that the true class is $t_k$
# 
# $w_{ij} = \frac{(i-j)^2}{(C-1)^2}$: C is number classes (same formula for $w_{t_kc}$)
# 

# ## Gradient
# 
# Since we are plugging the output of the LightGBM model and running it through softmax before the loss function, we will need the derivatives with respect to that LightGBM predictions.  
# 
# For the gradient, we need to calculate 
# 
# $$\frac{\partial \mathscr{L}}{\partial a_m}$$ 
# 
# where 
# 
# $m \in \{1, 2,..., C \}$ , C is the number of classes 
# 
# $a_m$: is the output of the LightGBM model
# 
# Here are the partials for the weighted kappa loss:
# 
# $$
# \frac{\partial \mathscr{L}}{\partial s_m} = \frac{1}{\mathscr{N}}\frac{\partial \mathscr{N}}{\partial s_m} - 
# \frac{1}{\mathscr{D}}\frac{\partial \mathscr{D}}{\partial s_m}
# $$
# 
# with 
# 
# $$
# \frac{\partial \mathscr{N}}{\partial s_m(X_k)} = \omega_{{t_k} m}
# $$
# 
# $$
# \frac{\partial \mathscr{D}}{\partial s_m(X_k)} = \sum_{i=1}^{C} \hat{N}_{i}\omega_{i, m}
# $$
# 
# $s_m$: softmax function of m-th class
# 
# The jacobian of the softmax is derived at this link https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/.  The partial derivative for the i-th output for the j-th input is:
# 
# $$
# \frac{\partial s_i}{\partial a_j} = \left\{
#                 \begin{array}{ll}
#                   s_i(1 - s_j) \quad i = j\\
#                   -s_i s_j  \quad i \ne j
#                 \end{array}
#               \right.
# $$
# 
# where
# 
# $s_i$: i-th softmax output
# 
# $a_j$: output of LightGBM and input to softmax function
# 
# Putting the above together we have for each model output $a_{ij}$, batch b, and class m:
# 
# $$
# \frac{\partial \mathscr{L}}{\partial a_{i j}} = \sum_{b=1}^{Batch Size} \sum_{m=1}^{Number Classes} \frac{\partial \mathscr{L}}{\partial s_{bm}} \frac{\partial s_{bm}}{\partial a_{i j}}
# $$
# 
# As a concrete example, for batch size of 2 and 4 classes and starting the index at 0, we have:
# 
# $$
# \frac{\partial \mathscr{L}}{\partial a_{00}} = 
# \frac{\partial \mathscr{L}}{\partial s_{00}} \frac{\partial s_{00}}{\partial a_{00}} + 
# \frac{\partial \mathscr{L}}{\partial s_{01}} \frac{\partial s_{01}}{\partial a_{00}} + 
# \frac{\partial \mathscr{L}}{\partial s_{02}} \frac{\partial s_{02}}{\partial a_{00}} + 
# \frac{\partial \mathscr{L}}{\partial s_{03}} \frac{\partial s_{03}}{\partial a_{00}} + 
# \frac{\partial \mathscr{L}}{\partial s_{10}} \frac{\partial s_{10}}{\partial a_{00}} + 
# \frac{\partial \mathscr{L}}{\partial s_{11}} \frac{\partial s_{11}}{\partial a_{00}} + 
# \frac{\partial \mathscr{L}}{\partial s_{12}} \frac{\partial s_{12}}{\partial a_{00}} + 
# \frac{\partial \mathscr{L}}{\partial s_{13}} \frac{\partial s_{13}}{\partial a_{00}}  
# $$
# 
# where the last 4 terms are 0 since the softmax out is not a function of $a_{00}$.

# ## Hessian
# From my understanding, LightGBM only requires the second derivatives for the ouput predictions and not
# the full hessian.
# 
# This is an indexing nightmare, so I will continue with the concrete example above, where we now need to calculate:
# 
# $$
# \frac{\partial}{\partial a_{00}} \frac{\partial \mathscr{L}}{\partial a_{00}}
# $$
# 
# Using the non zero terms above:
# 
# $$
# \frac{\partial}{\partial a_{00}} (\frac{\partial \mathscr{L}}{\partial s_{00}} \frac{\partial s_{00}}{\partial a_{00}} + 
# \frac{\partial \mathscr{L}}{\partial s_{01}} \frac{\partial s_{01}}{\partial a_{00}} + 
# \frac{\partial \mathscr{L}}{\partial s_{02}} \frac{\partial s_{02}}{\partial a_{00}} + 
# \frac{\partial \mathscr{L}}{\partial s_{03}} \frac{\partial s_{03}}{\partial a_{00}}) = 
# \frac{\partial}{\partial a_{00}} \left(A + B + C + D \right)
# $$
# 
# For the first term:
# $$ 
# \frac{\partial}{\partial a_{00}} A = \frac{\partial}{\partial a_{00}} \left(\frac{\partial \mathscr{L}} {\partial s_{00}} \frac{\partial s_{00}}{\partial a_{00}} \right) = \frac{\partial}{\partial a_{00}} \left( \frac{\partial \mathscr{L}} {\partial s_{00}}\right)\frac{\partial s_{00}}{\partial a_{00}} + \frac{\partial \mathscr{L}} {\partial s_{00}}\frac{\partial}{\partial a_{00}} \left(\frac{\partial s_{00}}{\partial a_{00}} \right)
# $$
# 
# which results in:
# $$
# \frac{\partial}{\partial a_{00}} A = \frac{\partial s_{00}}{\partial a_{00}} \left(\frac{\partial s_{00}}{\partial a_{00}} \frac{\partial^2 \mathscr{L}}{\partial s_{00}^2} + 
# \frac{\partial s_{01}}{\partial a_{00}} \frac{\partial^2 \mathscr{L}}{\partial s_{01} \partial s_{00}} + 
# \frac{\partial s_{02}}{\partial a_{00}} \frac{\partial^2 \mathscr{L}}{\partial s_{02} \partial s_{00}} +
# \frac{\partial s_{03}}{\partial a_{00}} \frac{\partial^2 \mathscr{L}}{\partial s_{03} \partial s_{00}} \right) +
# \frac{\partial \mathscr{L}}{\partial s_{00}} \frac{\partial^2 s_{00}}{\partial a_{00}^2}
# $$
# 
# B, C, and D are similarly derived as well as each model output $a_{ij}$.
# 
# For the second derivative of the loss wrt the softmax, we have:
# 
# $$
# \frac{\partial^2 \mathscr{L}}{\partial s_{bi} \partial s_{bj}} = \frac{\partial \mathscr{D}}{\partial s_{bj}}\frac{\partial \mathscr{D}}{\partial s_{bi}}\frac{1}{\mathscr{D}^2}-\frac{\partial \mathscr{N}}{\partial s_{bj}}\frac{\partial \mathscr{N}}{\partial s_{bi}}\frac{1}{\mathscr{N}^2} 
# $$
# 
# The hessian for the softmax function is mostly zeros since values outside of each batch are unrelated.  Within each batch, the second derivaties we want are:
# 
# $$
# \frac{\partial^2 s_{bi}}{\partial a_{bi}\partial a_{bj}} = \left\{
#                 \begin{array}{ll}
#                   s_{bi}(1 - s_{bi})(1 - 2 s_{bi}) \quad i = j\\
#                   s_{bi} s_{bj}(2s_{bi} - 1)  \quad i \ne j
#                 \end{array}
#               \right.
# $$
# 
# For batch b and classes i and j.

# ## Code for LightGBM function
# 
# A couple of notes. This isn't a convex function as far as I can tell by checking the hessian for positive semi-definiteness.  I could only get results using this by taking the absolute value of the hessian as a hack.  The scikit-learn api gives the predictions and takes the gradient and hessian in column order.

# In[ ]:


import numpy as np
from sklearn.preprocessing import OneHotEncoder


# ### Helper Functions

# In[ ]:


def softmax(x, axis=1):
    # Stable Softmax
    # from https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    y = x- np.max(x, axis=axis, keepdims=True)
    return np.exp(y) / np.sum(np.exp(y), axis=axis, keepdims=True)

def softmax_derivatives(s):
    s = s.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

def softmax_second_derivative(s, C=4):
    d2 = np.zeros((C, C))
    for i in range(C):
        for j in range(C):
            if i == j:
                d2[i, j] = (1. - s[i]) * s[i] * (1. - 2. * s[i])
            else:
                d2[i, j] = s[i] * s[j] * (2. * s[i] - 1.)
    return d2


# In[ ]:


def kappa_grads_lightgbm(y_true, y_pred):
    
    number_classes = 4
    C = number_classes
    labels = y_true 

    batch_size = y_true.shape[0]
    labels_one_hot = OneHotEncoder(categories=
                           [range(C)]*1, sparse=False).fit_transform(y_true.reshape(-1, 1))
    y_pred = np.reshape(y_pred, (batch_size, C), order='F')

    y_pred = softmax(y_pred)
    eps = 1e-12

    wtc = (y_true.reshape(-1, 1) - range(C))**2 / ((C - 1)**2)
    N = np.sum(wtc * y_pred)
    dN_ds = wtc

    Ni = np.sum(labels_one_hot, 0) / batch_size 
    repeat_op = np.tile(np.reshape(range(0, C), [C, 1]), [1, C])
    repeat_op_sq = np.square((repeat_op - np.transpose(repeat_op)))
    wij = repeat_op_sq / ((C - 1) ** 2)
    
    hist_preds = np.sum(y_pred, axis=0)
    D = np.sum(Ni.reshape(-1, 1) * (wij * hist_preds.reshape(1, -1)))
    dD_ds = np.tile(np.dot(wij, Ni), (batch_size, 1))
    
    dL_ds = dN_ds / (N + eps) - dD_ds / (D + eps)

    dL_da = np.zeros_like(dL_ds)
    ds_da = np.zeros((batch_size, C, C))
    for i in range(batch_size):
        ds_da[i] = softmax_derivatives(y_pred[i])
        dL_da[i] = np.dot(ds_da[i], dL_ds[i])

    d2L_da2 = np.zeros_like(dL_da)
    for b in range(batch_size):
        d2s_da2 = softmax_second_derivative(y_pred[b])
        d2N = -np.dot(dN_ds[b].reshape([C, 1]), dN_ds[b].reshape(1, C)) / (N * N + eps)
        d2D = np.dot(dD_ds[b].reshape([C, 1]), dD_ds[b].reshape(1, C)) / (D * D + eps)
        d2L_ds2 = d2N + d2D
        for c in range(C):
            AA = ds_da[b,0,c]*(ds_da[b,0,c] * d2L_ds2[0,0] + 
                                ds_da[b,1,c] * d2L_ds2[0,1] + 
                                ds_da[b,2,c] * d2L_ds2[0,2] +
                                ds_da[b,3,c] * d2L_ds2[0,3]
                                ) + dL_ds[b,0] * d2s_da2[c, 0] 

            BB = ds_da[b,1,c]*(ds_da[b,0,c] * d2L_ds2[1,0] + 
                                ds_da[b,1,c] * d2L_ds2[1,1] + 
                                ds_da[b,2,c] * d2L_ds2[1,2] +
                                ds_da[b,3,c] * d2L_ds2[1,3]
                                ) + dL_ds[b,1] * d2s_da2[c, 1] 

            CC = ds_da[b,2,c]*(ds_da[b,0,c] * d2L_ds2[2,0] + 
                                ds_da[b,1,c] * d2L_ds2[2,1] + 
                                ds_da[b,2,c] * d2L_ds2[2,2] +
                                ds_da[b,3,c] * d2L_ds2[2,3]
                                ) + dL_ds[b,2] * d2s_da2[c, 2] 

            DD = ds_da[b,3,c]*(ds_da[b,0,c] * d2L_ds2[3,0] + 
                                ds_da[b,1,c] * d2L_ds2[3,1] + 
                                ds_da[b,2,c] * d2L_ds2[3,2] +
                                ds_da[b,3,c] * d2L_ds2[3,3]
                                ) + dL_ds[b,3] * d2s_da2[c, 3] 

            d2L_da2[b, c] = AA + BB + CC + DD
    

    return [dL_da.flatten('F'), np.abs(d2L_da2.flatten('F'))]


# ## Check with autograd

# In[ ]:


get_ipython().system('pip install autograd')


# In[ ]:


import autograd.numpy as np
from autograd import grad
from autograd import jacobian
from autograd import hessian


# In[ ]:


# Test data
y_pred = np.array([[ 0.89912265,  0.79084255,  0.32162871, -0.99296229],
       [ 0.81017273,  0.18127493,  0.13865968, -0.32750946],
       [ 1.24011418,  0.94562047, -0.19091468, -0.68148713],
       [-1.10852297, -0.5044101 ,  0.41754522,  2.25403507],
       [ 1.41286477, -0.43752987, -0.34757177, -0.35673979],
       [ 3.19812033,  0.15338679, -1.12337413, -1.27692332],
       [ 2.4100999 ,  0.23849515, -0.79835829, -1.22977774],
       [-1.20118161, -1.21880044,  0.59375328,  3.13776406],
       [ 0.35147176,  0.03808217, -0.14641639, -0.22850701],
       [ 2.8068574 ,  0.30871835, -0.84488727, -1.28116301]])

y_true = np.array([3, 0, 3, 3, 0, 0, 0, 3, 3, 0])


# In[ ]:


def weighted_kappa(y_true, y_pred, C=4):
    
    batch_size = y_true.shape[0]
    labels_one_hot = OneHotEncoder(categories=
                           [range(C)]*1, sparse=False).fit_transform(y_true.reshape(-1, 1))
    y_pred = softmax(y_pred)
    eps = 1e-12

    wtc = (y_true.reshape(-1, 1) - range(C))**2 / ((C - 1)**2)
    N = np.sum(wtc * y_pred)
    Ni = np.sum(labels_one_hot, 0) / batch_size 

    repeat_op = np.tile(np.reshape(range(0, C), [C, 1]), [1, C])
    repeat_op_sq = np.square((repeat_op - np.transpose(repeat_op)))
    wij = repeat_op_sq / ((C - 1) ** 2)

    histp = np.sum(y_pred, axis=0)
    D = np.sum(Ni.reshape(-1, 1) * (wij * histp.reshape(1, -1)))
    
    return np.log(N / (D + eps))


# In[ ]:


kappa_grad = grad(weighted_kappa, 1)
kappa_hess = hessian(weighted_kappa, 1)


# In[ ]:


[custom_grad, custom_hess] = kappa_grads_lightgbm(y_true, y_pred)
autograd_grad = kappa_grad(y_true, y_pred).flatten('F')


# In[ ]:


np.allclose(custom_grad, autograd_grad)


# In[ ]:


autograd_all_hess = kappa_hess(y_true, y_pred)
# just need the diagonals to compare
autograd_diag_hess = np.array([
    autograd_all_hess[i, j, i, j]
    for j in range(autograd_all_hess.shape[1])
    for i in range(autograd_all_hess.shape[0])
    ])


# In[ ]:


# use absolute val of autograd_hess since we used it in the custom loss
np.allclose(custom_hess, np.abs(autograd_diag_hess))


# In[ ]:




