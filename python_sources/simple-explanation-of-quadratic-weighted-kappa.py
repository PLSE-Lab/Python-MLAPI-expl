#!/usr/bin/env python
# coding: utf-8

# <a class="anchor" id="0"></a>
# # **Simple Explanation of Quadratic Weighted Kappa**
# 
# 
# 
# 
# 
# ## **Introduction**
# 
# 
# 
# Kaggle has recently launched a competition - Prostate cANcer graDe Assessment (PANDA) Challenge. 
# We can find the competition description [here](https://www.kaggle.com/c/prostate-cancer-grade-assessment/overview) and the cometition evaluation page can be found [here](https://www.kaggle.com/c/prostate-cancer-grade-assessment/overview/evaluation). 
# 
# 
# 
# Submissions in this competition are scored based on the **Quadratic Weighted Kappa (QWK)**. So, in this notebook, I will try to provide an intuitive explanation of **Quadratic Weighted Kappa (QWK)**. I have tried to find out the meaning of Quadratic Weighted Kappa (QWK). I have found few excellent resources which explains QWK. These are -
# 
# 
# - [CPMP's discussion Fast QWK Computation](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/145105).
# 
# - [Ultra Fast QWK Calc Method](https://www.kaggle.com/cpmpml/ultra-fast-qwk-calc-method)
# 
# - [qwk: cupy vs numpy vs numba](https://www.kaggle.com/jiweiliu/qwk-cupy-vs-numpy-vs-numba)
# 
# - [Understanding the Quadratic Weighted Kappa by reigHns](https://www.kaggle.com/reighns/understanding-the-quadratic-weighted-kappa)
# 
# 
# - The codes have been adapted from Ben Hamner's github repository : https://github.com/benhamner/Metrics
# 
# 
# So, let's get started.
# 

# **I hope you find this notebook useful and your <font color="red"><b>UPVOTES</b></font> would be very much appreciated**
# 
# 

# <a class="anchor" id="0.1"></a>
# # **Table of Contents**
# 
# 
# 1.	[Introduction to Quadraticc Weighted Kappa (QWK)](#1)
# 2.	[How to calculate QWK](#2)
# 3.	[Implement QWK Calculation](#3)
# 4.  [Interpretation of QWK Value](#4)
# 5.  [Conclusion](#5)
# 

# # **1. Introduction to Quadratic Weighted Kappa (QWK)** <a class="anchor" id="1"></a>
# [Table of Contents](#0.1)
# 
# 
# 
# - **Quadratic Weighted Kappa (QWK)** is described on the evaluation page, which we can find [here](https://www.kaggle.com/c/prostate-cancer-grade-assessment/overview/evaluation).
# 
# - **Quadratic Weighted Kappa (QWK)** measures the agreement between two outcomes. We can interpret QWK as the amount of agreement between an algorithm's predictions and true labels.
# 
# - This metric typically varies from 0 (random agreement) to 1 (complete agreement). In the event that there is less agreement than expected by chance, the metric may go below 0 i.e. it may become negative.
# 
# - So, we can present it in tabular form as below -
# 
#     - -1 : Complete disagreement
#     - 0 : Agreement by chance
#     - 0-0.2 ; Poor agreement
#     - 0.2-0.4 : Moderate agreement
#     - 0.4-0.6 : Good agreement
#     - 0.6-0.8 : Very good agreement
#     - 0.8-1 : Perfect agreement 
#     - 1 : Complete agreement   
#     
# 

# # **2. How to calculate QWK** <a class="anchor" id="2"></a>
# 
# [Table of Contents](#0.1)
# 
# 
# The QWK calculation is a 4 step process as described in the evaluation page. These steps are explained below-
# 
# 
# - **Step 1** : First, an N x N histogram matrix O is constructed , such that Oi,j corresponds to the number of predicted labels that have a rating of i (actual) that received a predicted value j.
# 
# 
# - **Step 2** : An N-by-N matrix of weights, w, is calculated based on the difference between actual and predicted values. This matrix is given as 
# 
# 
# $$w_{i,j} = \dfrac{(i-j)^2}{(N-1)^2}$$
# 
# 
# - **Step 3** : An N-by-N histogram matrix of expected ratings, E, is calculated, assuming that there is no correlation between rating scores. This is calculated as the outer product between the actual rating's histogram vector of ratings and the predicted rating's histogram vector of ratings, normalized such that E and O have the same sum.
# 
# 
# - **Step 4** : From these three matrices, the quadratic weighted kappa is calculated as follows -
# 
# 
# $$\kappa = 1 - \dfrac{\sum_{i,j}\text{w}_{i,j}O_{i,j}}{\sum_{i,j}\text{w}_{i,j}E_{i,j}}$$
# 

# # **3. Implement QWK Calculation** <a class="anchor" id="3"></a>
# 
# [Table of Contents](#0.1)
# 
# 
# - Now, we will demonstrate the calculation of QWK.
# 

# ## **Step 1 : Create the N x N histogram matrix O** 
# 
# 
# - To create the N x N histogram matrix O, we have to define the confusion matrix. So, let's do it.

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


# For the purpose of explaination, we will assume are actual and preds labels to be the following.

# In[ ]:


# define the actual and predicted labels
X_actual = pd.Series([1,1,1,2,3,4,4,4,4,4]) 
X_pred   = pd.Series([1,1,1,2,3,4,4,4,4,0]) 


# In[ ]:


# create the histogram matrix O
O = confusion_matrix(X_actual, X_pred)
O


# Now, we will calculate N which is the number of labels.

# In[ ]:


N = len(X_actual)
N


# ## **Step 2** : An N-by-N matrix of weights (w) is calculated
# 
# 
# - Now, an N-by-N matrix of weights, w, is calculated based on the difference between actual and predicted values.

# In[ ]:


w = np.zeros((5,5))
w


# In[ ]:


for i in range(len(w)):
    for j in range(len(w)):
        w[i][j] = float(((i-j)**2)/((N-1)**2))


# In[ ]:


w


# ## **Step 3** : Calculation of N-by-N histogram matrix of expected scores(E)
# 
# 
# - An N-by-N histogram matrix of expected outcomes(E) is calculated assuming that there is no correlation between values.
# 
# - This is calculated as the outer product between the actual histogram vector of outcomes and the predicted histogram vector, normalized such that E and O have the same sum.

# In[ ]:


N = 5


# In[ ]:


# calculation of actual histogram vector
X_actual_hist=np.zeros([N]) 
for i in X_actual: 
    X_actual_hist[i]+=1    

print('Actuals value counts : {}'.format(X_actual_hist))


# In[ ]:


# calculation of predicted histogram vector
X_pred_hist=np.zeros([N]) 
for i in X_pred: 
    X_pred_hist[i]+=1    

print('Predicted value counts : {}'.format(X_pred_hist))


# - Now, we calculate the expected matrix E.
# 
# - Expected matrix (E) is calculated as the outer product between the actual values histogram vector and the predicted values histogram vector.

# In[ ]:


E = np.outer(X_actual_hist, X_pred_hist)
E


# - Now, we normalize E and O.
# 
# - E and O are normalized such that E and O have the same sum.

# In[ ]:


E = E/E.sum()
E.sum()


# In[ ]:


O = O/O.sum()
O.sum()


# - Let's print the matrix E and O.

# In[ ]:


E


# In[ ]:


O


# ## **Step 4** : Final Step : QWK Calculation
# 
# 
# - Now, we come to the final step. In this step, we calculate the **Quadratic Weighted Kappa (QWK)**. It is calculated as follows -
# 
# 
# $$\kappa = 1 - \dfrac{\sum_{i,j}\text{w}_{i,j}O_{i,j}}{\sum_{i,j}\text{w}_{i,j}E_{i,j}}$$
# 

# In[ ]:


Num=0
Den=0

for i in range(len(w)):
    for j in range(len(w)):
        Num+=w[i][j]*O[i][j]
        Den+=w[i][j]*E[i][j]
        
Res = Num/Den
 
QWK = (1 - Res)
print('The QWK value is {}'.format(round(QWK,4)))


# # **4. Interpretation of QWK Value** <a class="anchor" id="4"></a>
# 
# [Table of Contents](#0.1)
# 
# 
# - Our QWK value has found to be 0.6154.
# 
# - It means that the actual and predicted values have very good agreement.

# # **5. Conclusion** <a class="anchor" id="5"></a>
# 
# [Table of Contents](#0.1)
# 
# 
# - In this notebook, we have explained the meaning of **QWK**.
# 
# - We have also demonstrated how to calculate **QWK**.
# 
# - We have taken sample actual and predicted values and found the QWK value to be 0.6154.
# 
# - So, we can conclude that the actual and predicted values have every good agreement between them.

# Thus, we come to the end of this notebook.
# 
# I hope you find this notebook useful and enjoyable.
# 
# Your comments and feedback are most welcome.
# 
# Thank you
# 

# [Go to Top](#0)
