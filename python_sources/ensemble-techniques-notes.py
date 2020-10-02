#!/usr/bin/env python
# coding: utf-8

# # Ensemble Learning
# 
# `wisdom of crowd`
# if you aggregate the the predictions of grou of predictors(classifier or regressor), we will endedup getting better rediction than individual predictor.
# 
# ensemble = "a group of predictors"
# this technique is called `Ensemble Learning`
# 
# ensemble of Decision Trees = Random Forest

# some of the classification approches using ensemble methods:
# 
# ## Voting Classifiers:
# ### (Hard Voting classifier)
# 
# Aggregate of the predictions of each of the classifiers.
# 
# `Which ever class gets the most votes out of all the predicted classes of all classifiers, is marked as the     
#    predicted class by the Voting classifier`
#    
# [Note] Everytime the classifier you built are performing as `Weak learners`,the voting classifier build out of weak learner always performs significantly better.

# ### Biased coin example
# Lets say, you have a slightly biased coin that gives head 51% of the time (prob of being a heads) and just 49% of times the tails coming up, in every toss.
# 
# if you toss it 1000 times,<br>
# 510 heads<br>
# 490 tails<br>

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# tossing coin in series of 10 attempts, for a total of 1000 times
heads_proba = 0.51

coin_tosses = (np.random.rand(10000,10)< heads_proba).astype(np.int32)
cummulative_head_ratio = np.cumsum(coin_tosses,axis=0) / np.arange(1,10001).reshape(-1,1)


# In[ ]:


plt.figure(figsize=(10,5))
plt.plot(cummulative_head_ratio)
plt.plot([0,10000],[.51,.51],'k--',linewidth =2,label='51%')
plt.plot([0,10000],[.50,.50],'k-',linewidth =2,label='50%')
plt.xlabel("No. of coin tosses", fontsize=12)
plt.ylabel("Heads Ratio", fontsize=12)
plt.axis([0, 10000, 0.42, 0.59])
plt.show()


# ### soft voting classifier
# basically  it take into consideration the class probabilities.<br>
# You basically predict the class that has the highest probability across all the 1000 classifiers.
# 
# [NOTE] for this to happen you need to make sure that each of your predictor has predict_proba() decision making method

# In[ ]:


import warnings
warnings.simplefilter(action='ignore',category='FutureWarning')


# In[ ]:


from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X,y = make_moons(n_samples=500,noise=0.3,random_state=42)
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)


# In[ ]:




