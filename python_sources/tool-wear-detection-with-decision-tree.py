#!/usr/bin/env python
# coding: utf-8

# **Tool wear detection**
# 
# Thanks to Sharon Sun for uploading this dataset.I was trying some simple models and found that the Z position has a huge impact in predicting the tool condition. Any specific reason? Or did I miss something? Any suggestions are welcome. Here is my approach:

# In[17]:


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Join experiments in a single dataframe and add tool condition column

# In[7]:


frames = list()
results = pd.read_csv("../input/train.csv")
for i in range(1,19):
    exp = '0' + str(i) if i < 10 else str(i)
    frame = pd.read_csv("../input/experiment_{}.csv".format(exp))
    row = results[results['No'] == i]
    frame['target'] = 1 if row.iloc[0]['tool_condition'] == 'worn' else 0
    frames.append(frame)
df = pd.concat(frames, ignore_index = True)
df.head()


# Using every CNC measurement as an independent observation (as described in overview/content)

# In[23]:


# Transform process name in number
le = LabelEncoder()
le.fit(df['Machining_Process'])
df['Machining_Process'] = le.transform(df['Machining_Process'])
# Create np arrays and split train/test sets
y = np.array(df['target'])
x = df.drop('target', axis=1).values
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123)
# Target distribution (count values)
print("Target distribution - 1 worn; 0 unworn:")
print(df['target'].value_counts())


# Apply Decision Tree model and evaluate accuracy and feature importance

# In[21]:


model = DecisionTreeClassifier()
model.fit(x_train, y_train)
y_score = model.predict(x_test)
print("Trained on {0} observations and scoring with {1} test samples.".format(len(x_train), len(x_test)))
print("Accuracy: {0:0.4f}".format(accuracy_score(y_test, y_score)))
print("F1 Score: {0:0.4f}".format(f1_score(y_test, y_score)))
print("Area under ROC curve: {0:0.4f}".format(roc_auc_score(y_test, y_score)))


# In[25]:


# Feature importances
features = [(df.columns[i], v) for i,v in enumerate(model.feature_importances_)]
features.sort(key=lambda x: x[1], reverse = True)
for item in features[:10]:
    print("{0}: {1:0.4f}".format(item[0], item[1]))


# In[ ]:


# Z Actual Position distribution
df['Z1_ActualPosition'].hist()


# 
