#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/rakesh4real/Applied-Machine-Learning/blob/master/005_Linear_SVM.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# #### Before we begin
#  
# A bit of Linear algebra. Understand parallel lines!
#  
# - Two lines are are parallel if their slopes ( $m$ in $y=mx+c$ ) are same. 
#  
# - So, the only change we can do in eqn. of parallel lines is to $c$ (Done below)
#  
# - Consider line $l_0: y=mx+c$
#  
#     - $ l_1: y = mx + (c+k) $ is parallel to $l_0$ and lies on *top* of it.
#  
#   -  $ l_2: y = mx + (c-k) $ is parallel to $l_0$ and lies *below* it.
#  
# - $k$ is the equal distance of separarion between $l_1$ & $l_0$ and $l_2$ & $l_0$ when we use direction cosines; otherwise it is $\frac{k}{||w||^2}$

# ### Linear SVM
# 
# *We call it SVM because we consider only support vectors(lines made by support datapoints) for classification

# 

# Consider linearly separable data for binary classification. 
#  
# We will find line $l_0: wx+b = 0$ which separates them so that it aligns itself exactly in middle of support vectors $l_1: wx+b+k=0$ and $l_2: wx+b-k=0$ where $\frac{2k}{||w||^2}$ is distance between SVs.
#  
# $$
# \begin{align}
# l_0: wx_i + b = 0 \\
# l_2: wx_i + b = k \\
# l_1: wx_i + b = -k \\
# \end{align}
# $$
#  
# For simplicity, if wanted, we can take $k=1$. Note that $k=1$ doesn't mean distance between SVs is always $2k$ and it is hard coded. Distance is $\frac{2k}{||w||^2}$ It always varies with $W$ during trainig

# ### Loss Function 
#  
# - Let class 1 be denoted by $+1$ and class 0 by $-1$
#  
# - Conditions for correct classification
#     - $wx_i + b > k \implies y=+1 \; \text{i.e class 1}$
#  
#   - $wx_i + b < k \implies y=-1 \; \text{i.e class 0}$
#  
# - Above two conditions can be combined into one equation. For correct classification,
#  
# $$ 
# \begin{align}
# y_i(wx_i + b) > k \\
# \text{(or)} \\
# y_i(z_i) > k \\
# \end{align}
# $$
#  
# - Minimize **Hinge Loss** to find optimal $l_0$
#     - $Loss = 0$ for all correcly classified points i.e if $y_i(z_i)>k$
#  
#   - $Loss$ should linearly increase for all incorrectly classified point i.e if $y_i(z_i)<k \implies Loss = k-y_i(z_i)$ 
#  
#   - Can combine the above two into one eqn $$ Hinge Loss = max(0, \, k-y_i(z_i)) $$
#  
# - Maximize **Margin** to allign $l_0$ correctly. $$ argmax(\frac{k}{||w||^{2}}) \implies argmin(||w||^{2}) $$
#  
# Hence, **Loss Function** is 
#  
# $$ J = \lambda{||w||^{2}} + \frac{1}{m} \sum max(0, k-y(z_i)) $$

# ### Update
#  
# Actual loss-value is not important for updation but it's derivatives are important. As, hinge loss is not contninous, derivative formula changes at non-continous position of loss curve.
#  
# 1.When hinge loss = 0 (i.e when $y(z_i) >= k$)
#  
# $$
# \begin{align}
#  J = \lambda ||w||^{2} \\
#  \implies \frac{dJ}{dw} = 2 \lambda w   \\
#  \implies \frac{dJ}{db} = 0 \\
# \end{align}
# $$ 
#  
# 2. When hinge loss = $k - y(z_i)$ (i.e when $k(y_i) < k$)
#  
# $$
# \begin{align}
#     J = \lambda w^{2} + \frac{1}{m} k - y_i(wx_i) \\
#   \implies \frac{dJ}{dw} = 2w\lambda + \frac{1}{m} x_iy_i \\
#   \implies \frac{dJ}{db} = - \frac{1}{m} y \\
# \end{align}
# $$
#  
# **Update using** 
#  
# $$
# \begin{align}
#  w = w - \alpha \cdot gradient \\
#   b = b - \alpha \cdot gradient
# \end{align}
# $$

# 

# ### Prepare Dataset

# In[ ]:


import numpy as np
from sklearn.datasets import make_blobs
 
X_all, y_all = make_blobs(
    n_samples=200,
    n_features=2,
    centers=2, #classes
    random_state=123
)
 
 
#Note: We are not adding bias column as webwont be using vectors.
 
 
# plot
import matplotlib.pyplot as plt
plt.scatter(x=X_all[:, 0], y=X_all[:, 1], c=y_all,
            cmap=plt.cm.Paired)
plt.show();
 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=123)
 
print(X_train.shape)
print(y_train.shape)
print(X_test.shape, y_test.shape)


# In[ ]:


#TRAINING (difft than ususal)
#1. check if y(z_i) > k or < k
#2. calculate mistakes accordingly. 
#3. Correct them until convergence

#PREDICTING
#3. predict using sign of z_i.
#      y(z_i) cant be used for prediction

#Note
#We are not using vectors


# In[ ]:


class SVM:
    def __init__(self, lr=1e-2, epochs=1000, lambda_val=1e-2, k=1):
        self.lr = lr
        self.epochs = epochs
        self.lambda_val = lambda_val
        self.k = k
        self.w = None
        self.b = None
        
    def fit(self, X_train, y_train):
        n_feats = X_train.shape[1]
        self.w = np.ones(n_feats)
        self.b = np.ones(1)
        # make sure class labels are -1 and +1
        y_train = np.where(y_train==0, -1, 1)
        
        for e in range(self.epochs):
            for i, x_i in enumerate(X_train):
                #1. predict
                k_hat = y_train[i] * (np.dot(x_i, self.w) + self.b)
                correct_classification = k_hat > self.k
                #2. find mistakes
                if correct_classification:
                    dw = 2 * self.lambda_val * self.w
                    db = 0
                else:
                    dw = (2 * self.lambda_val * self.w) - (y_train[i] * x_i) #lazy to change wrong eqn above in latex
                    db = (-1) * (y_train[i])
                #3. correct
                self.w = self.w - self.lr * dw
                self.b = self.b - self.lr * db
        
    def predict(self, X_test):
        return [self._predict(x_q) for x_q in X_test]
        
    def _predict(self, x_q):
        pred = np.dot(x_q, self.w) + self.b
        return 0 if pred<0 else 1


# In[ ]:


clf = SVM()
clf.fit(X_train, y_train)
y_test_preds = clf.predict(X_test)

acc = np.sum(y_test_preds == y_test) / len(y_test)
print(acc) 


# In[ ]:


# Plot results

w = clf.w
b = clf.b
k = clf.k

x_min, x_max = X_train[:,0].min(), X_train[:, 0].max()
xs = np.linspace(x_min, x_max, 3)

# Decision boundary
# wx + b = 0
# => w[0]x1 + w[1]x2 + b = 0
# => y = - (w[1]/w[0])x - (b/w[0])
ys = -1 * (1/w[0]) * (w[1]*xs + b)
plt.plot(xs, ys)

# SVs
# y = mx + b +- (k/w^2)
dist = k / np.sqrt(np.sum(w**2))
sv1 = ys + dist
sv2 = ys - dist
plt.plot(xs, sv1, color='r')
plt.plot(xs, sv2, color='r')

# x_test
plt.scatter(x=X_test[:,0], y=X_test[:,1], c=y_test)
#plt.scatter(x=X_train[:,0], y=X_train[:,1], c=y_train, cmap=plt.cm.Paired)

#Note: from fig. train acc is less than 1.0 but test acc is 1.0 

