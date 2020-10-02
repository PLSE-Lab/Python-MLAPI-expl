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


import pandas as pd
import numpy as np
print(os.listdir("../input"))



# In[ ]:


data = pd.read_csv("../input/data.csv")


# In[ ]:


data


# # Codes Used In this
# **mild fever = 0,
# tiredness  = 1,
# no taste   = 2,
# sore throat= 3,
# no symptoms= 4,
# fever      = 5, 
# dry cough  = 6,
# diff breath= 7,
# sore throat= 8**
# 
# 
# 
# 
# > FOR PRESYMTOMATIC = 0
# > FOR ASYMPTOMATIC  = 1
# > FOR SYMPTOMATIC   = 2
# 
# 

# In[ ]:


import random
a = 0
for i in range(0, len(data)):
    S = data["symptom"][i].split()
    travel = data["visiting Wuhan"][i]
    if ('0' in S):
        if ('6' in S) or ('8' in S) or ('4' in S):
            if travel == 1:
                data["result"][i] = 0
    elif travel == 0 and ('4' in S):
        data["result"][i] = 1
    elif ('5' in S):
        if ('6' in S):
            data["result"][i] = 2
    else:
        symtoms = [0] * 10 + [2] * 4
        choice = random.choice(symtoms)
        data["result"][i] = choice
sym = []
pre = []
asym = []
for i in data['result']:
    if i == 0:
        pre.append(1)
    if i == 1:
        asym.append(1)
    if i == 2:
        sym.append(1)

print(len(sym), len(pre), len(asym))
            


# In[ ]:





# # Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


X = data[["symptom", "age", "visiting Wuhan"]]


# In[ ]:


Y = data['result']
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(X["symptom"])
X["symptom"] = le.transform(X["symptom"])


# In[ ]:


model =LogisticRegression()


# In[ ]:


model.fit(X, Y)


# In[ ]:


data


# In[ ]:


le.transform(['3 6 1'])
predict = model.predict(X)


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.title("Logistic Regression with Score 74.76%")
sns.lineplot(x='result', y=np.arange(0, len(data)), data=data, label="Real Values")
sns.lineplot(x=predict, y=np.arange(0, len(data)), label="Predicted Values")
plt.savefig("LogisticGraph.pdf")


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(Y, predict)*100


# # Random Forest 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


model =RandomForestClassifier()


# In[ ]:


model.fit(X, Y)


# In[ ]:


predict = model.predict(X)


# In[ ]:


accuracy_score(Y, predict)*100


# In[ ]:


import seaborn as sns
plt.title("Random Forest with Score 99.63%")

sns.lineplot(x='result', y=np.arange(0, len(data)), data=data)
sns.lineplot(x=predict, y=np.arange(0, len(data)))
plt.savefig("RANDOMFORESTGRAPH.pdf")


# In[ ]:


estimator = model.estimators_[1]
estimator


# In[ ]:


from sklearn.tree import export_graphviz


# In[ ]:


export_graphviz(estimator, out_file='tree_limited.dot', feature_names = X.columns,
                class_names = ['1', '2' , '3'],
                rounded = True, proportion = False, precision = 2, filled = True)


# In[ ]:


X.columns


# In[ ]:


get_ipython().system('dot -Tpng tree_limited.dot -o modelInAtree.png')


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y, predict)
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize = (16, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=25, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)
plot_confusion_matrix(cm, classes = ['presym', 'asym', 'sym'],
                      title = 'Poverty Confusion Matrix')
plt.savefig("ConfusionMatrixfgraph454.pdf")


# In[ ]:


sym = []
pre = []
asym = []
for i in data['result']:
    if i == 0:
        pre.append(1)
    if i == 1:
        asym.append(1)
    if i == 2:
        sym.append(1)

print(len(sym), len(pre), len(asym))


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


# In[ ]:


model_fc = RandomForestClassifier()
model_fc.fit(X_train, y_train)


# In[ ]:


predictions = model.predict(X_test)


# In[ ]:


accuracy_score(y_test, predictions)*100


# In[ ]:




