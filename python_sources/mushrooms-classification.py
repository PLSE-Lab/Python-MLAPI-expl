#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

INPUT_FILE = "mushrooms.csv"

def load_data(file=INPUT_FILE, header=True):
    csv_path = os.path.join("", file)
    if header:
        return pd.read_csv(csv_path)
    else:
        return pd.read_csv(csv_path, header=None)


data = load_data(INPUT_FILE)
data.head()


# In[2]:


data.info()


# In[3]:


datacopy = data.copy()


# In[4]:


from sklearn.preprocessing import LabelEncoder, LabelBinarizer

encoders = {}
binarizers = {}
for column in list(data):
    encoder = LabelEncoder()
    encoder.fit(data[column])
    binarizer = LabelBinarizer()
    binarizer.fit(data[column])
    data[column] = encoder.transform(data[column])
    encoders[column] = encoder
    binarizers[column] = binarizer


# In[5]:


data.head()


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
data.hist(bins=50, figsize=(20,15))
plt.show()


# In[7]:


corr_features = data.corr()
corr_features["class"]


# In[8]:


columns_removed = []
columns_removed.append("veil-type")


# In[9]:


data = datacopy.copy()


# In[10]:


from sklearn .model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data["class"]):
    train_features = data.loc[train_index]
    test_features = data.loc[test_index]
    
train_labels = train_features["class"].copy().values
train_features.drop("class", axis=1, inplace=True)

test_labels = test_features["class"].copy().values
test_features.drop("class", axis=1, inplace=True)


# In[11]:


for set in (train_features, test_features):
    for column_removed in columns_removed:
        set.drop([column_removed], axis=1, inplace=True)


# In[12]:


train_features.info()


# In[13]:


import numpy as np

X = []
X_test = []
for column in list(train_features):
    if len(X) == 0:
        X = binarizers[column].transform(train_features[column])
        X_test = binarizers[column].transform(test_features[column])
        continue
        
    X = np.concatenate((X, binarizers[column].transform(train_features[column])), axis=1)
    X_test = np.concatenate((X_test, binarizers[column].transform(test_features[column])), axis=1)
    
Y = binarizers["class"].transform(train_labels).flatten()
Y_test = binarizers["class"].transform(test_labels).flatten()


# In[14]:


print(np.shape(Y), np.shape(Y_test), np.shape(X), np.shape(X_test))


# In[15]:


from sklearn.decomposition import PCA
import numpy as np

def get_dims_variances(min_dim, max_dim, threshold=0.1, capToThreshold=False):
    dims = []
    variances = []
    optimum_dim = min_dim
    for dim in range(min_dim, max_dim):
        pca = PCA(n_components=dim)
        pca.fit(X)
        variance = np.array(pca.explained_variance_ratio_)
        variance = variance.min()
        if threshold < variance:
            optimum_dim = dim
            dims.append(dim)
            variances.append(variance)
        else:
            if capToThreshold:
                break
        
    return dims, variances, optimum_dim


# In[16]:


dims, variances, optimum_dim = get_dims_variances(2,  np.shape(X)[1], 0.01, capToThreshold=True)
print(optimum_dim)
import matplotlib.pyplot as plt
plt.plot(dims, variances)
plt.show()


# In[17]:


print(variances)


# In[18]:


pca = PCA(n_components=optimum_dim)
pca.fit(X)
print(pca.explained_variance_ratio_)
X = pca.transform(X)
X_test = pca.transform(X_test)


# In[19]:


from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median")
imputer.fit(X)

X = imputer.transform(X)
X_test = imputer.transform(X_test)


# In[20]:


from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()
scalar.fit(X)

X = scalar.transform(X)
X_test = scalar.transform(X_test)


# In[21]:


print(Y)
print(binarizers["class"].classes_)


# In[22]:


from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
def plot_roc_curve(clf_sets):
    for clf_set in clf_sets:
        y = clf_set[0]
        y_pred = clf_set[1]
        label = clf_set[2]
        fpr, tpr, thresholds = roc_curve(y, y_pred)
        plt.plot(fpr, tpr, linewidth=1, label=label)
    
    plt.plot([0,1],[0,1],'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="bottom right")
    plt.show()


# In[23]:


# SGD Classifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.base import clone

clf_sets = []
sgd_clf = SGDClassifier(random_state=42)

print("Cross Val Scores on training set\n", cross_val_score(clone(sgd_clf), X, Y, cv=3, scoring="accuracy"))
Y_pred = cross_val_predict(clone(sgd_clf), X, Y, cv=3)
print("confusion_matrix\n", confusion_matrix(Y, Y_pred))
print("f1_score\n", f1_score(Y, Y_pred))
print("precision_score\n", precision_score(Y, Y_pred))
print("recall_score\n", recall_score(Y, Y_pred))
clf_sets.append((Y, Y_pred, "SGDClassifier-train"))

print("\n\nCross Val Scores on testing set\n", cross_val_score(clone(sgd_clf), X_test, Y_test, cv=3, scoring="accuracy"))
Y_test_pred = cross_val_predict(clone(sgd_clf), X_test, Y_test, cv=3)
print("confusion_matrix\n", confusion_matrix(Y_test, Y_test_pred))
print("f1_score\n", f1_score(Y_test, Y_test_pred))
print("precision_score\n", precision_score(Y_test, Y_test_pred))
print("recall_score\n", recall_score(Y_test, Y_test_pred))
clf_sets.append((Y_test, Y_test_pred, "SGDClassifier-test"))

sgd_clf.fit(X, Y)
print("\n\nAccuracy on testing data set\n", sum(Y_test == sgd_clf.predict(X_test)) / len(X_test))

plot_roc_curve(clf_sets)


# In[24]:


# KNeighbors Classifier

from sklearn.neighbors import KNeighborsClassifier 

knn_clf = KNeighborsClassifier()
print("Cross Val Scores on training set\n", cross_val_score(clone(knn_clf), X, Y, cv=3, scoring="accuracy"))
Y_pred = cross_val_predict(clone(knn_clf), X, Y, cv=3)
print("confusion_matrix\n", confusion_matrix(Y, Y_pred))
print("f1_score\n", f1_score(Y, Y_pred))
print("precision_score\n", precision_score(Y, Y_pred))
print("recall_score\n", recall_score(Y, Y_pred))
clf_sets.append((Y, Y_pred, "KNN-train"))

print("\n\nCross Val Scores on testing set\n", cross_val_score(clone(knn_clf), X_test, Y_test, cv=3, scoring="accuracy"))
Y_test_pred = cross_val_predict(clone(knn_clf), X_test, Y_test, cv=3)
print("confusion_matrix\n", confusion_matrix(Y_test, Y_test_pred))
print("f1_score\n", f1_score(Y_test, Y_test_pred))
print("precision_score\n", precision_score(Y_test, Y_test_pred))
print("recall_score\n", recall_score(Y_test, Y_test_pred))
clf_sets.append((Y_test, Y_test_pred, "KNN-test"))

knn_clf.fit(X, Y)
print("\n\nAccuracy on testing data set\n", sum(Y_test == knn_clf.predict(X_test)) / len(X_test))

plot_roc_curve(clf_sets)


# In[25]:


# Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier 

forest_clf = RandomForestClassifier(random_state=42)
print("Cross Val Scores on training set\n", cross_val_score(clone(forest_clf), X, Y, cv=3, scoring="accuracy"))
Y_pred = cross_val_predict(clone(forest_clf), X, Y, cv=3)
print("confusion_matrix\n", confusion_matrix(Y, Y_pred))
print("f1_score\n", f1_score(Y, Y_pred))
print("precision_score\n", precision_score(Y, Y_pred))
print("recall_score\n", recall_score(Y, Y_pred))
clf_sets.append((Y, Y_pred, "RandomForest-train"))

print("\n\nCross Val Scores on testing set\n", cross_val_score(clone(forest_clf), X_test, Y_test, cv=3, scoring="accuracy"))
Y_test_pred = cross_val_predict(clone(forest_clf), X_test, Y_test, cv=3)
print("confusion_matrix\n", confusion_matrix(Y_test, Y_test_pred))
print("f1_score\n", f1_score(Y_test, Y_test_pred))
print("precision_score\n", precision_score(Y_test, Y_test_pred))
print("recall_score\n", recall_score(Y_test, Y_test_pred))
clf_sets.append((Y_test, Y_test_pred, "RandomForest-test"))

forest_clf.fit(X, Y)
print("\n\nAccuracy on testing data set\n", sum(Y_test == forest_clf.predict(X_test)) / len(X_test))

plot_roc_curve(clf_sets)


# In[ ]:




