#!/usr/bin/env python
# coding: utf-8

# This kernel is just a simple extension of: https://www.kaggle.com/vzaguskin/kappa-sensitivity-dsbowl?scriptVersionId=26918830

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import cohen_kappa_score
import random
random.seed(42)
np.random.seed(42)


# Prepare ground truth and a bad submission

# In[ ]:


gt = np.random.choice([0,1,2,3], 1000, p=(0.24, 0.14, 0.12, 0.50))
pred = gt.copy()
noise = np.random.choice([- 3,-2, -1, 0, 1, 2, 3], 1000, p=(0.05,0.1, 0.15, 0.4, 0.15, 0.1, 0.05))
pred_noisy = pred + noise
pred_noisy[pred_noisy > 3]  -=3
pred_noisy[pred_noisy < 0] += 3


# initial score:

# In[ ]:


df = pd.DataFrame({'reality':pred, 'pred_noisy': pred_noisy})
df[['reality', 'pred_noisy']].groupby(['reality', 'pred_noisy'])['reality'].count().to_frame('count').reset_index()


# In[ ]:


cohen_kappa_score(gt, pred_noisy, weights='quadratic')


# correct 10 errors at each step and see how result improves

# In[ ]:


for i in range(10):
    errs = np.where(pred_noisy != gt)[0]
    pred_noisy[errs[:10]] = gt[errs[:10]]
    print(cohen_kappa_score(gt, pred_noisy, weights='quadratic'))
    


# # Extension
# It turns out that it matters which variables we fix. Or to be precise - for each prediction it depends how much we were wrong. Predicting 0 instead of 3 is not the same as predicting 1 instead of 2. So once we already improved our kappa to 0.585, let's see what happens next based on what fix we make.

# In[ ]:


df = pd.DataFrame({'reality':pred, 'pred_noisy': pred_noisy})
df[['reality', 'pred_noisy']].groupby(['reality', 'pred_noisy'])['reality'].count().to_frame('count').reset_index()


# ## Fixing zeros

# In[ ]:


print('The original kappa was: ' + str(cohen_kappa_score(gt, df['pred_noisy'], weights='quadratic')))
to_fix = [0,1,2,3]
for h in to_fix:
    for i in range(0,4):
        if i != h:
            errs = np.where((df['pred_noisy'] != df['reality']) & (df['reality'] == h) & (df['pred_noisy'] == i))[0]
            df['new_preds'] = df['pred_noisy']
            df['new_preds'][errs[:10]] = df['reality'][errs[:10]]
            print('Kappa after fixing another ten ' + str(h) + ' missclassified as ' + str(i) + ': ' + str(cohen_kappa_score(gt, df['new_preds'], weights='quadratic')) + '. This is '+str(cohen_kappa_score(gt, df['new_preds'], weights='quadratic') - cohen_kappa_score(gt, df['pred_noisy'], weights='quadratic')) + ' improvement from the original')


# # Conclusion
# As we have seen, if we fix 10 zeros misclassified as 3 or 10 threes misclassified as 0, we can move kappa from 0.585 to over 0.61. It might be also worthwhile to analyze how big boost we get if we do not replace bad prediction by correct one, but if we replace it by a 'less wrong one'
