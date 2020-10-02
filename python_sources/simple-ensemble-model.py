#!/usr/bin/env python
# coding: utf-8

# **tl;dr** - combine you previous submission together to get a higher score.
# 
# Digit recognizer problem can be solved with many tools. Either you choose deep CNN or any of the classic ML approaches, you will probably not hit the top positions with your very first try. 
# 
# If you did the same like me, then after several experiments you have several submission files. 
# 
# In this notebook I will show, how you can use them to create simple [Ensemble Model](https://en.wikipedia.org/wiki/Ensemble_learning). And maybe you will climb up in the ranking with the results of your shiny new EM.  
# 
# So let's dive in. 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# After import of all the necessary modules let's create the **list with the names** of submission files.  
# 
# Please note, that in this example I will use only the original sample solution file and one more dummy example. Just put the names with your own solutions and you are ready to go.

# In[ ]:


files = ['../input/original-submission-sample/sample_submission.csv',
         '../input/example-submission-file/submission.csv']


# We will encode the predicted labels using [one-hot encoding](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) and then merge the results together using numpy.

# In[ ]:


#one hot encoded result
y_final = np.zeros((28000, 10))

for fname in files:
    df1 = pd.read_csv(fname)
    labels = np.array(df1.pop('Label'))
    labels = LabelEncoder().fit_transform(labels)[:, None]
    labels = OneHotEncoder().fit_transform(labels).todense()
    y_final += labels

print(y_final)


# The final step is simple - just pick up the most predicted label from all models. 

# In[ ]:


predictions = np.argmax(y_final, axis=1)
print(predictions)


# Save the file and you are ready to go. 

# In[ ]:


submission = pd.DataFrame(data={'ImageId': (np.arange(len(predictions)) + 1), 'Label': predictions})
submission.to_csv('submission-ensembled-model.csv', index=False)
submission.tail()   


# **If you found this notebook helpful or you just liked it, some upvotes would be very much appreciated ;-) **

# 

# In[ ]:




