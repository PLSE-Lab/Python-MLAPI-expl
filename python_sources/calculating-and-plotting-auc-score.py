#!/usr/bin/env python
# coding: utf-8

# Recently I was looking for an easy way to calculate/plot the ROC-AUC score of my validation predictions and this is the little script I found.
# 
# Here I show a sample use case. We will take the first million lines of the training dataset and generate random predictions for them. Then we will calculate the ROC-AUC score between the "predictions" and the real class of the data. Let's get started!

# In[ ]:


import numpy as np
import pandas as pd

n = 1000000
train = pd.read_csv('../input/train.csv', nrows=n)


# First we will read the target values of the dataset. These are the correct classifications we will test against.

# In[ ]:


y_train = train['is_attributed']


# Next we need to make our predictions. These will usually come from a model. For the purposes of this tutorial though, I will simply create a `Series` of random values that will serve as our 'predictions'.

# In[ ]:


predictions = pd.Series(np.random.rand(y_train.shape[0]))


# And now we will calculate and plot the AUC score of `y_train` and `predictions`:

# In[ ]:


import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)

fpr, tpr, thresholds = roc_curve(y_train, predictions)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();


# Unsurprisingly, the AUC score is terrible for the above, since I merely used randomly generated numbers.
# 
# You can use this snippet by replacing the `y_train` data with your validation set data and `predictions` with the predictions your model generated.
# 
# Hope this helps!
# 
# *Disclaimer: I did not create this script, I only moved it from [an article online](https://medium.com/@curiousily/credit-card-fraud-detection-using-autoencoders-in-keras-tensorflow-for-hackers-part-vii-20e0c85301bd) for ease of use.*
