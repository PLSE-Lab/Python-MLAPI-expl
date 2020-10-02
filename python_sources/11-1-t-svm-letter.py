#!/usr/bin/env python
# coding: utf-8

# # Importing Packages

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm


# In[ ]:


letters_df = pd.read_csv("../input/letterdata.csv")


# In[ ]:


letters_df.head(10)   # all fields except the target ("letter") are numeric. We do not know the scale. So normalize


# # Build model

# In[ ]:


#Prepare X, y
X, y = letters_df.drop(columns = 'letter'), letters_df.loc[:,'letter'] 


# In[ ]:


#Should always be written in this format else will throw shape warning
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)


# In[ ]:


clf = svm.SVC(gamma=0.025, C=3)    
# gamma is a measure of influence of a data point. It is inverse of distance of influence. C is complexity of the model
# lower C value creates simple hyper surface while higher C creates complex surface


# In[ ]:


clf.fit(X_train , y_train)


# In[ ]:


clf.score(X_test, y_test)


# In[ ]:


#Predict y_pred values given X_test and stack y_test, y_pred
y_pred = clf.predict(X_test)


# In[ ]:


y_grid = (np.column_stack([y_test, y_pred]))


# In[ ]:


print(y_grid)


# In[ ]:


#Display all the letters actual and predicted
pd.set_option('display.max_columns', 26)

pd.crosstab(y_pred, y_test)


# ### 

# In[ ]:


#Lets find all the letters which were incorrectly predicted
unmatched = []
for i in range(len(y_grid)):
    if y_grid[i][0] != y_grid[i][1]:
        unmatched.append(i)


# In[ ]:


y_grid[unmatched]


# In[ ]:


#np.savetxt("Text", y_grid , fmt='%s')

