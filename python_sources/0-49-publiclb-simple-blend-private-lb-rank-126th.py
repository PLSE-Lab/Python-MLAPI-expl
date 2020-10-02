#!/usr/bin/env python
# coding: utf-8

# Hi all,
# I would like to present a simple method to score 0.49 on Public LB (with rank 126 in Private LB) by only using the outputs of public kernels. There are 4 steps:
# 
# 1) Find good kernels before the leak discovery, or more precisely, find good kernels which are written disregarding the leak. They will score 1.37 or 1.38.
# 
# 2) Blend them.
# 
# 3) Replace leak rows using the test leak table of the 0.56 kernel.
# 
# 4) Enjoy the silver medal.
# 
# 
# I have troubles in directly select the outputs of public kernels. So I downloaded them and then uploaded them into my own dataset with renamed files. Here is the list I used. Special thanks to the authors of the below kernels. 
#  + file 137: from this kernel https://www.kaggle.com/mannyelk/an-honest-approach
#  + file 138_1 to 138_5: from these 5 kernels:
#    https://www.kaggle.com/wentixiaogege/santander-46-features-add-andrew-s-feature-b337d2
#    https://www.kaggle.com/indranilbhattacharya/row-features-xgb-yet-to-be-tuned
#    https://www.kaggle.com/sheboke93/santander-46-features-add-andrew-s-feature
#    https://www.kaggle.com/sggpls/public-sin-lb-1-38
#    https://www.kaggle.com/scirpus/santander-gp-clustering-ii
#    
#  + file test_leak_37: from this kernel https://www.kaggle.com/nulldata/jiazhen-to-armamut-via-gurchetan1000-0-56

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** .5

test = pd.read_csv('../input/santander-value-prediction-challenge/test.csv')
tst_leak = pd.read_csv('../input/santander-public-outputs/test_leak_37.csv')
test['leak'] = tst_leak['compiled_leak']

merge_files = ['137','138_1','138_2','138_3','138_4','138_5']
weights = [3,1,1,1,1,1] # the weights are invented by sense


# In[ ]:


score = pd.read_csv('../input/santander-value-prediction-challenge/sample_submission.csv', usecols=['ID'])

for i in range(len(merge_files)):
    score_temp = pd.read_csv('../input/santander-public-outputs/'+ merge_files[i] +'.csv').rename(columns={'target':'score_'+ str(i)})
    score = pd.merge(score, score_temp, how='left', on='ID')
score.head()    


# In[ ]:


# Compute weighted average
sum_pred = np.zeros(len(score), dtype=float)
for i in range(len(merge_files)):
    sum_pred = sum_pred + list(score['score_'+str(i)].values*weights[i])
    
avg_pred = sum_pred/sum(weights)


# In[ ]:


filesave = "Blend_Finale"
lgsub = pd.DataFrame(avg_pred,columns=["target"])
lgsub['ID'] = score['ID'].values
lgsub['leak'] = tst_leak['compiled_leak']
# Replace leak rows
lgsub.loc[lgsub.leak.notnull(),'target'] = lgsub.loc[lgsub.leak.notnull(), 'leak'] 
# Write output
lgsub[['ID','target']].to_csv(filesave+".csv",index=False,header=True)
lgsub[['ID','target']].head()

