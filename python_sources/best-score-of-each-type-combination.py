#!/usr/bin/env python
# coding: utf-8

# For https://www.kaggle.com/xwxw2929/keras-neural-net-and-distance-features the CV_score for each type is
# **1JHC : cv score is  -0.4814433**
# 2JHH : cv score is  -2.1609457
# **1JHN : cv score is  -1.0991564**
# 2JHN : cv score is  -2.2432249
# 2JHC : cv score is  -1.605876
# 3JHH : cv score is  -2.1098902
# 3JHC : cv score is  -1.5108138
# 3JHN : cv score is  -2.4077184
# total cv score is -1.702383577823639
# 
# For https://www.kaggle.com/xwxw2929/keras-nn-with-multi-output the  CV_score for each type is
# **1JHC : cv score is  -0.5142987**
# 2JHH : cv score is  -2.0816948
# **1JHN : cv score is  -1.1312212**
# 2JHN : cv score is  -2.160751
# 2JHC : cv score is  -1.520438
# 3JHH : cv score is  -2.0065901
# 3JHC : cv score is  -1.4212722
# 3JHN : cv score is  -2.27407
# total cv score is -1.6387920081615448
# 
# Although it's not an improvement of the totle score, the multi-output seems to be useful to the '1JHC' and '1JHN' type. So I combined these two predictions by the best score of each type.

# In[ ]:


import warnings
warnings.filterwarnings("ignore")
import numpy as np 
import pandas as pd
import os
print(os.listdir("../input"))


# In[ ]:


test = pd.read_csv('../input/champs-scalar-coupling/test.csv')
sub1 = pd.read_csv('../input/keras-neural-net-and-distance-features/submission.csv')
sub2 = pd.read_csv('../input/keras-nn-with-multi-output/submission.csv')
display(test.head(),sub1.head(),sub2.head())


# In[ ]:


sub = pd.DataFrame(columns = ['id','scalar_coupling_constant'])
mol_types1 = ['2JHH','2JHN','2JHC','3JHH', '3JHC', '3JHN']
mol_types2 = ['1JHC', '1JHN']


# In[ ]:


for mol_type in mol_types1:
    index = test[test['type']==mol_type].id
    temp= sub1[sub1['id'].isin(index)]
    sub = pd.concat([sub, temp])

for mol_type in mol_types2:
    index = test[test['type']==mol_type].id
    temp= sub2[sub2['id'].isin(index)]
    sub = pd.concat([sub, temp])


# In[ ]:


sub.shape


# In[ ]:


sub.sort_values(['id'], inplace = True)
sub.head()


# In[ ]:


sub.to_csv("/kaggle/working/submission.csv", index=False)

