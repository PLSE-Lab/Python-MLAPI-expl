#!/usr/bin/env python
# coding: utf-8

# <img src="https://lh3.googleusercontent.com/-tNe1vwwd_w4/VZ_m9E44C7I/AAAAAAAAABM/5yqhpSyYcCUzwHi-ti13MwovCb_AUD_zgCJkCGAYYCw/w256-h86-n-no/Submarineering.png">

# Dear Colleagues, as you know, for this competition and others, we are allowed to make only two submission per day.
# 
# That is a little frustrating, especially when you have a good idea and you have not the possibility of test it.
# 
# I have spent a lot of time on understanding how log_loss behave over different models and kind of submissions.
# 
# The main purpose of this kernel is provide a workbench in order to make you able to test different submissions accuracy before  be submitted.
# 
# I would like to take into consideration your experience and your input. Please comment if you have something to add.
# 
# I highly recommed this related article: 
# 
# http://www.exegetic.biz/blog/2015/12/making-sense-logarithmic-loss/
# 
# I hope this kernel be useful for you. Please, Vote up.  
# 
# 

# In[ ]:


import os
import numpy as np 
import pandas as pd 
from sklearn.metrics import log_loss 
get_ipython().run_line_magic('pylab', 'inline')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# First thing first@
# # Credits to the following awesome authors and kernels
# 
# Author: QuantScientist    
# File: sub_200_ens_densenet.csv     
# Link: https://www.kaggle.com/solomonk/pytorch-cnn-densenet-ensemble-lb-0-1538     
# 
# 
# Author: wvadim     
# File: sub_TF_keras.csv     
# Link: https://www.kaggle.com/wvadim/keras-tf-lb-0-18     
# 
# 
# Author: Ed Miller    
# File: sub_fcn.csv    
# Link: https://www.kaggle.com/bluevalhalla/fully-convolutional-network-lb-0-193     
# 
# 
# Author: Chia-Ta Tsai    
# File: sub_blend009.csv    
# Link: https://www.kaggle.com/cttsai/ensembling-gbms-lb-203    
# 
# 
# Author: DeveshMaheshwari    
# File: sub_keras_beginner.csv    
# Link: https://www.kaggle.com/devm2024/keras-model-for-beginners-0-210-on-lb-eda-r-d       
# 
# Author: Submarineering    
# 
# Files: submission38.csv , submission43.csv, submission54.csv
# 
# Link : https://www.kaggle.com/submarineering/submission38-lb01448
# 
# ### Without their truly dedicated efforts, this notebook will not be possible.     

# # Data Load

# In[ ]:


# Read different submissions.
out1 = pd.read_csv("../input/statoil-iceberg-submissions/sub_200_ens_densenet.csv", index_col=0)
out2 = pd.read_csv("../input/statoil-iceberg-submissions/sub_TF_keras.csv", index_col=0)
out3 = pd.read_csv("../input/submission38-lb01448/submission38.csv", index_col=0)
out4 = pd.read_csv("../input/submission38-lb01448/submission43.csv", index_col=0)
out5 = pd.read_csv('../input/submarineering-even-better-public-score-until-now/submission54.csv',index_col=0)


# Now , very important to understand the concept. 
# Because 'out5' is the best scored available, I am going to use it to get the labels. Imagine that this labels are the true labels. 
# Of course not,  but knowing the score obtain by the file we can estimate and error and play taking into considaration that error to score the others files. 

# In[ ]:


#getting lables from our best scored file. 
labels = (out5>0.5).astype(int)


# In[ ]:


# out5 score a log_loss of 0.1427 and could be considered also like an error Lerr= 0.1427
# Error produce by itself.
out5err = log_loss(labels, out5)
Lerr =  0.1427
print('out5 Error:', Lerr+out5err)


#  Then, now can be estimate the ranking of all the files and test the accuary based on labels, and its error. 
#  I am going to do it for every file('score already known') available on the kernel. 
# 

# In[ ]:


files = ['out1', 'out2', 'out3', 'out4', 'out5']
ranking = []
for file in files:
    ranking.append(log_loss(labels, eval(file)))


# In[ ]:


results = pd.DataFrame(files, columns=['Files'])
results['Error'] = ranking
results['Lerr'] = Lerr
results['Total_Error'] = results['Error']+ results['Lerr']
results


# In[ ]:


results['Total_Error'].plot(kind='bar')


# Now you can get your own conclusion and compare between files before the submission. 

# LOG_LOSS curve: 
# 
# <img src='http://www.exegetic.biz/static/img/2015/12/log-loss-curve.png'>

# As can be read in the above recommeded article, Log_loss penalize very high when the classifier produce a false positive or viceversa. And the penalty is higher when more extreme is the probability predicted for the class. 
# The article, mention some smoothing method to avoid that extra penalty.
# Let me see, how act clipping over this matter. 
# The file with the worse relative score to our artificial labels, is out2. 
# Can be improved the scored, just appling some clipping ?

# In[ ]:


# As before :
out2err = log_loss(labels, out2) + Lerr
out2err


# In[ ]:


# Apply some clipping : 
OUT2err = log_loss(labels, np.clip(out2, 0.0001, 0.99)) + Lerr
OUT2err


# And Of course, you can. Please, fell free to experiment and share if you discover something interesting. 

# 
# Roboust model is always the key component, stacking only comes last with the promise to surprise, sometimes, in an unpleasant direction@. 
# 
# For more efficient models I highly recommend my engineering features extraction kernels: 
# 
# https://www.kaggle.com/submarineering/submarineering-size-matters-0-75-lb
# 
# https://www.kaggle.com/submarineering/submarineering-objects-isolation-0-75-lb
# 
# https://www.kaggle.com/submarineering/submarineering-what-about-volume-lb-0-45
# 
# Greeting, Subamrineering.
# 
# 
# 

# I hope these lines be useful for your. **Please vote up.** and let your comments.
