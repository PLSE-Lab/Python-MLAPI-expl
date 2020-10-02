#!/usr/bin/env python
# coding: utf-8

# **This notebook explores two models to predict whether a mushroom is edible or not and then explores and visualizes the dataset's feature importances.**

# In[ ]:


# Load dataset
import pandas as pd

data = pd.read_csv("../input/mushrooms.csv")
data.head()


# In[ ]:


# encode labels

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
for col in data.columns:
    data[col] = le.fit_transform(data[col])
data.head()


# In[ ]:


# split out features and target labels

y = data['class']
X = data.drop(['class'], axis=1)


# In[ ]:


# split out training and test set
from sklearn.model_selection import train_test_split

# Shuffle and split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[ ]:


# apply MLP model
from sklearn.neural_network import MLPClassifier

# fit model
clf = MLPClassifier(hidden_layer_sizes=(100,), solver='adam',warm_start=False, random_state=None)
clf.fit(X_train, y_train)

# assess model accuracy
clf.score(X_test, y_test)


# In[ ]:


# create confusion matrix to illustrate accuracy of predictions

from sklearn.metrics import classification_report,confusion_matrix

y_pred = clf.predict(X_train)
cm = (confusion_matrix(y_train,y_pred))

import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

sns.heatmap(cm, annot=True, fmt="d", xticklabels=["Edible", "Poisonous"],yticklabels=["Edible","Poisonous"])


# In[ ]:


print(classification_report(y_train,y_pred))


# **It is difficult to interpret a Multi-Layer Preceptron model - the weights and biases are not be easily interpretable in relation to feature importance of the model.
# Still, attributes coefs_ and intercepts_ can be used to extract the MLP weights and biases.
# RandomForests are more useful when assessing feature importance:**

# In[ ]:


# fit RandomForest model
from sklearn.ensemble import RandomForestClassifier 

tree = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)

tree.fit(X_train, y_train)

# assess model accuracy
tree.score(X_test, y_test)


# In[ ]:


# extract feature importances
import numpy as np

keyfeat = tree.feature_importances_


# In[ ]:


# rank features

df = pd.DataFrame(keyfeat)
df.index = np.arange(1, len(df) + 1)

featurenames = data.columns
featurenames = pd.DataFrame(data.columns)
featurenames.drop(featurenames.head(1).index, inplace=True)

dfnew = pd.concat([featurenames, df], axis=1)
dfnew.columns = ['featurenames', 'weight']
dfsorted = dfnew.sort_values(['weight'], ascending=[False])
dfsorted.head()


# In[ ]:


# plot feature importances
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(15, 10))

ax = sns.barplot(x=dfsorted['featurenames'], y=dfsorted['weight'])

ax.set(xlabel='feature names', ylabel='weight')

ax.set_title('Feature importances')

for item in ax.get_xticklabels():
    item.set_rotation(50)


# **Conclusions:
# The chart above indicates that odor, gill size and spore print color are the three most important features when determining if a mushroom is edible or poisonous.**

# In[ ]:




