#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


get_ipython().system('pip install pydotplus')


# # Importing Libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from IPython.display import Image
import pydotplus


# # Loading Data

# In[ ]:


data=pd.read_csv('/kaggle/input/data.csv')
data.head()


# In[ ]:


X= data[['age','bp']]
y=data[['diabetes']]


# # Splitting data into train and test set

# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)


# # Using Decision Tree

# In[ ]:


model1=tree.DecisionTreeClassifier(min_samples_split=10,min_samples_leaf=6,max_depth=5)
model1.fit(X_train,y_train)
y_pred_train=model1.predict(X_train)
print('train:',accuracy_score(y_train,y_pred_train))
y_pred_test=model1.predict(X_test)
print('test:',accuracy_score(y_test,y_pred_test))


# # Thats it, right?
# ### It's so simple to implement a decision tree using sklearn, but how does it actually works?
# * First parameter while creating instance of model is 'criterion'.
# * criterion parameter can take two values 'entropy' or 'gini', and has default value 'gini', have a look at screenshot of sklearn docs.
# 
# ![DT.JPG](attachment:DT.JPG)
# 
# * Basic splits of decision tree takes place based criterion(entropy or gini whichever specified), also it depends on many other patameters like min_samples_split, max_depth etc, but basic splits depends on value of gini/entropy.

# ## What is gini and entropy?
# *Note: Here just formulas are mentioned but things will get more clear as you follow the notebook.
# ### Gini index:-
#    * Gini Index is a metric to measure how often a randomly chosen element would be incorrectly identified.
#    * Gini index is given as below, where P is frequentist probability of class.
#    ![gini.png](attachment:gini.png)
#         
#         
# ### Entropy:-
#    * Entropy is simply a measure of disorder or uncetainty.
#    * Entropy is given as below, where P is frequentist probability of class.
#    ![entropy.png](attachment:entropy.png)
#    
#    
# ### Information gain:-
#    * So we calculate Information gain using gini/entropy and whichever feature or condition gives maximum information gain, split occurs based on that feature or condition
#    * Information gain from X on Y
#    
#    ![IG.png](attachment:IG.png)
#   
# **Note:- While calculating information gain we consider weighted values i.e if we are using entropy as our criterion then we'll use weighted entropy to calculate information gain.

# # Let's calculate Information gain manually and try to create decsion tree.
# ### First lets plot the decision tree of the model we trained.

# In[ ]:


dot_data=tree.export_graphviz(model1,out_file=None,feature_names=X.columns,class_names=str(y['diabetes'].unique()))

graph=pydotplus.graph_from_dot_data(dot_data)

Image(graph.create_png())


# # As split is done based on condition i.e b<=79, it means this condition gives maximum **INFORMATION GAIN**, right? Let's verify.

# # Case 1
# ## Claculation for the above plotted decision tree.(only for 1st split)
# * In the plot of decision tree we can see that first split is done based on condition i.e bp<=79, so lets see how is our class distributed.( zoom in the plot and have a look at root node)
# 

# In[ ]:


bp_lessthanequal_79 = X_train[X_train['bp']<=79.0].shape[0]
bp_greaterthan_79 = X_train.shape[0]-bp_lessthanequal_79

print('Number of samples with bp<=79 :',bp_lessthanequal_79)
print('Number of samples with bp>79 :',bp_greaterthan_79)
print('number of patients having diabetes are : ',(y_train['diabetes']==1).sum())
print('number of patients not having diabetes are : ',(y_train['diabetes']==0).sum())


# ### Calculating Gini 
# ** Note: all calculations are for 1st split
# ![gini1.jpg](attachment:gini1.jpg)
# 
# * You may cross verify the values of calculated gini and in the plotted decision tree.

# ### Weighted Average
# ![wa1.jpg](attachment:wa1.jpg)

# ### Information Gain
# * E(Y) is gini of root node and E(Y|X) is weighted average.
# ![ig1.JPG](attachment:ig1.JPG)
# 
# ### So Information gain in case 1 = 0.3687

# # Case 2
# ## Let's assume we want to split root node based on condition bp<=70.
# * Instead of having condition bp<=79 lets assume that we have split our root node based on condition bp<=70
# * The code below can be skipped, its just for getting values of distribution using which we can plot Decision Tree, If want to skip directly jump to the next tree diagram 

# ## Dataframes with bp<=70 and bp>70

# In[ ]:


bp_lessthanequal_70=X_train[X_train['bp']<=70]
bp_greaterthan_70=X_train[X_train['bp']>70]


# ## Distribution 

# In[ ]:


number_lessthanequal_70=bp_lessthanequal_70.shape[0]
number_bp_greaterthan_70=bp_greaterthan_70.shape[0]

print('number of patients with bp<=70 are : ',number_lessthanequal_70)
print('number of patients with bp>70 are : ',number_bp_greaterthan_70)
print('number of patients having diabetes are : ',(y_train['diabetes']==1).sum())
print('number of patients not having diabetes are : ',(y_train['diabetes']==0).sum())


# ### Positives and Negatives for patients with bp<=70

# In[ ]:


positives_on_left=np.logical_and(X_train['bp']<=70,y_train['diabetes']==1).sum()
negatives_on_left=np.logical_and(X_train['bp']<=70,y_train['diabetes']==0).sum()

print('Number of patients with bp<=70 and have diabetes are : ',positives_on_left)
print('Number of patients with bp>70 and have diabetes are : ',negatives_on_left)


# ### Positives and Negatives for patients with bp>70

# In[ ]:


positives_on_right=np.logical_and(X_train['bp']>70,y_train['diabetes']==1).sum()
negatives_on_right=np.logical_and(X_train['bp']>70,y_train['diabetes']==0).sum()

print('Number of patients with bp<=70 and have diabetes are : ',positives_on_right)
print('Number of patients with bp>70 and have diabetes are : ',negatives_on_right)


# ### Tree based on above calculation
# ![tree1.JPG](attachment:tree1.JPG)

# ### Calculating gini
# ![gini2.JPG](attachment:gini2.JPG)

# ### Weighted average and Information gain
# ![ig2.JPG](attachment:ig2.JPG)
# 
# ### So Infomation Gain in case 2 =0.157

# # Observing Information gained in Case 1 and Case 2
# * We know that split will take place based on condition/feature with maximum Information gain.
# * Just to prove this we considered two cases one with the actual split and other with random condition. 
# * After calculation we can see that case 1 has higher amount of information gain as compared to case 2, That's why sklearn used case 1 for spltting.
# * Also splitting depends on other parameters as well like min_samples_split, min_samples_leaf etc.
# * Similary if we had criterion='entropy' instead of calculation gini we would have calculated entropy, and then information gain.

# ### If this notebook was helpful to you please upvote, it motivates me. Also suggest changes if any.
