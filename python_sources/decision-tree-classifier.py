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


# # Importing libraries

# In[ ]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # loading dataset

# In[ ]:



dataset = pd.read_csv("/kaggle/input/decision-tree-data-set-from-stack-abuse/bill_authentication.csv")


# # data analysis

# In[ ]:



dataset.shape


# In[ ]:


dataset.head()


# # preparing the data

# In[ ]:


X = dataset.drop('Class', axis=1)
y = dataset['Class']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# # training and making predictions

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)


# In[ ]:


y_pred = classifier.predict(X_test)


# # evaluating the algorithm

# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:


from sklearn import metrics 
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# # some intallations for printing the tree

# In[ ]:


pip install graphviz


# In[ ]:


pip install pydotplus


# In[ ]:


pip install --upgrade scikit-learn==0.20.3


# # code for printing the tree

# In[ ]:


from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = ['Variance','Skewness','Curtosis','Entropy'],class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())


# In[ ]:




