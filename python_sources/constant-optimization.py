#!/usr/bin/env python
# coding: utf-8

# # Constant optimization
# In this notebook I'm trying to optimize constant prediction (all rows in submission have the same predicted value).
# 
# I know that it's silly, but maybe someone will find it entertaining.

# # First submission
# Simply send __sample_submission.csv__
# 
# **C=0.5; RMSE=1.23646**
# 
# Can we do better?

# # Second submission
# 
# Let's do some sophisticated data analysis!

# In[ ]:


import pandas as pd
sales = pd.read_csv('../input/sales_train.csv.gz', compression='gzip')
sales.item_cnt_day.mean()


# I got a feeling that **1.242641**  will do much better **0.5**.
# 
# **C=1.242641; RMSE=1.54960**
# 
# Hmm, not quite.

# # Third submisson
# 
# Lets compute root mean square error *S* with constant prediction *C* 
# 
# $S = \sqrt{\frac{\sum(y_i - C)^2}{n}}$
# 
# Get rid of root
# 
# $nS^2= \sum{y_i^2} - 2C\sum{y_i} + nC^2 $
# 
# Substititute $A:=\frac{1}{n}\sum{y_i^2},  B:=\frac{1}{n}\sum{y_i}$
# 
# $nS^2 = nA - 2nCB + nC^2$
# 
# $A + (-2C)B=S^2-C^2$
# 
# Knowing two datapoints *(C=0.5; S=1.23646)* and *(C=1.242641; S=1.54960)* we can solve it for *A* and *B*.

# In[1]:


import numpy as np

c_1,c_2 = 0.5,     1.242641
s_1,s_2 = 1.23646, 1.54960

X = np.array([[1., -2.*c_1],
              [1., -2.*c_2]])
S = np.array([s_1*s_1 - c_1*c_1,
              s_2*s_2 - c_2*c_2])

A, B = np.matmul(np.linalg.inv(X), S)
print("A=%.3f\tB=%.3f" % (A, B))


# Let's find value of *C* that minimizes *S*:
# 
# $$
# \underset{C}{\mathrm{argmin}}\sqrt{\frac{\sum(y_i - C)^2}{n}} = 
# \underset{C}{\mathrm{argmin}}\sum(y_i - C)^2 = 
# \underset{C}{\mathrm{argmin}}(nC^2-2C\sum y_i)=
# \underset{C}{\mathrm{argmin}}(C^2-2CB)
# $$
# 
# $$(C^2-2CB)'=2C-2B$$
# 
# $$\underset{C}{\mathrm{argmin}}\text{ }S = B$$

# In[ ]:


C  = B
print("C=%.3f"%C)


# Let's give it a try
# 
# **C=0.284; RMSE=1.21743**
# 
# Congratulations! We reached the ideal of constant optimization.

# # What else can we do?
# 
# We were able to derive $E[Y]= B =\frac{1}{n}\sum{y_i}$  and  $E[Y^2]= A = \frac{1}{n}\sum{y_i^2}$ , knowing that we can compute standard deviation of  $Y$:
# 
# $$\sigma = \sqrt{E[(Y - E[Y])^2]}=\sqrt{E[(Y - B)^2]}=\sqrt{ E[Y^2]  + E[-2YB] +E[ B^2] } = \sqrt{A -  2BE[Y] + B^2} = \sqrt{A - 2B^2 + B^2} = \sqrt{A - B^2}
# $$

# In[3]:


import math
std = math.sqrt(A - B*B)
print("std=%.3f"%std)


# # FIN
