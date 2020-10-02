#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing modules
import itertools
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Sklearn - Defaults

# In[ ]:


expected = [1,1,0,1,0,0,1,0,0,0]
predicted = [1,0,0,1,0,0,1,1,1,0]
cf =confusion_matrix(expected,predicted)
cf


# **Out of four 1's, 3 are predicted correctly and out of six 0's 4 are predicted correctly.**
# 
# *Note: Actual as Rows and Predicted as columns*

# ### Pandas - CrossTab

# In[ ]:


exp_series = pd.Series(expected)
pred_series = pd.Series(predicted)
pd.crosstab(exp_series, pred_series, rownames=['Actual'], colnames=['Predicted'],margins=True)


# ### Matplotlib - Confusion Matrix Plot 

# In[ ]:


plt.matshow(cf)
plt.title('Confusion Matrix Plot')
plt.colorbar()
plt.xlabel('Precited')
plt.ylabel('Actual')
plt.show();


# ### R like Output

# In[ ]:


confusion_matrix(predicted,expected)


# **Out of four 1's, 3 are predicted correctly and out of six 0's 4 are predicted correctly.**
# 
# *Note: Actual as Columns and Predicted as Rows (like we have in R caret package*

# ### Image plot with values inside cells

# In[ ]:


plt.imshow(cf,cmap=plt.cm.Blues,interpolation='nearest')
plt.colorbar()
plt.title('Confusion Matrix without Normalization')
plt.xlabel('Predicted')
plt.ylabel('Actual')
tick_marks = np.arange(len(set(expected))) # length of classes
class_labels = ['0','1']
tick_marks
plt.xticks(tick_marks,class_labels)
plt.yticks(tick_marks,class_labels)
# plotting text value inside cells
thresh = cf.max() / 2.
for i,j in itertools.product(range(cf.shape[0]),range(cf.shape[1])):
    plt.text(j,i,format(cf[i,j],'d'),horizontalalignment='center',color='white' if cf[i,j] >thresh else 'black')
plt.show();

