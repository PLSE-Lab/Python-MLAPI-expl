#!/usr/bin/env python
# coding: utf-8

# ### As far as we know f1 score is highly dependent from true positive rate. If a class is not present then it will have true positive rate and f1 score both equal to 0, even though all predictions are correct. That could have a stong effect******** on small batches when not all classes are present.

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.metrics import f1_score


# ### Read all true labels from training set to demonstate the idea on them

# In[ ]:


num_classes = 28

df = pd.read_csv('../input/train.csv')

y_true = np.zeros((len(df), num_classes))

for i, row in df.iterrows():
    for lblIndex in row['Target'].split():
        y_true[i][int(lblIndex)] = 1
        
print(y_true.shape)


# ### We can calculate f1 score for all true labels with themselves. This should give f1 equal to 1 since all classes are present in training set.

# In[ ]:


print(f1_score(y_true, y_true, average='macro'))


# ### After that we will calculate f1 score for true labels with themselves for small batches to see the effect

# In[ ]:


for batch_size in [64, 32, 16]:
    print("Batch size:", batch_size, "F1 macro:", f1_score(y_true[:batch_size], y_true[:batch_size], average='macro'))


# In[ ]:




