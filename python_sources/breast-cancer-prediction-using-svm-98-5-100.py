#!/usr/bin/env python
# coding: utf-8

# ## Breast Cancer Predictions using SVM##

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import pandas as pd
import random
import itertools
import seaborn as sns

sns.set(style = 'darkgrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# What is the trick ?
# Since breast cancer is a 3 dimmensional structure, we estimatd a volume factor.
# since the malicous cancers are more dented and concaved , we multiply the radius and area to estimate a maximal volume
# since the data are skewed, we log normalize that synthetic volume number
# 
# second we estimate a concavity number multiplied with compactness
# third value is a synthetic number that tries to estimate the ratio of the tumour volume versus the mean tumour dimmension
# fourth number is a multiplication of three low correlated numbers
# 
# These synthetic values help to determine the difference between the two types of tumours

# In[ ]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:


bc = pd.read_csv('../input/data.csv')
bc.head(1)
new_columns = bc.columns.values; new_columns[18] = 'concavepoints_worst'; bc.columns = new_columns
#Volume worst estimation of cancer create a estimate worst 3D structure
bc = bc.drop("Unnamed: 32",1)
temp = np.log(bc.radius_worst*bc.area_worst)
bc['Volume_ln'] = temp.values
#
temp = np.log(bc.concavepoints_worst*bc.concavity_worst*bc.compactness_worst+1)
bc['Concave_ln'] = temp.values
#cancer fractal- symmetry  divided by volume
temp = -np.log(bc.fractal_dimension_worst*bc.symmetry_worst/np.log(bc.radius_mean*bc.area_mean))
bc['FractVol_ln'] = temp.values
# all unrelated
temp = np.log(bc.radius_worst*bc.perimeter_worst*bc.concavepoints_worst+1)
bc['RaPeCo_ln'] = temp.values


# Scale the data to chart it and allow better predictive power

# In[ ]:


bcs = pd.DataFrame(preprocessing.scale(bc.ix[:,2:36]))
bcs.columns = list(bc.ix[:,2:36].columns)
bcs['diagnosis'] = bc['diagnosis']


# Check for correlations between variables and diagnosis

# In[ ]:


from pandas.tools.plotting import scatter_matrix


# Let's see how each variable breaks down by diagnosis

# In[ ]:


mbc = pd.melt(bcs, "diagnosis", var_name="measurement")
fig, ax = plt.subplots(figsize=(10,5))
p = sns.violinplot(ax = ax, x="measurement", y="value", hue="diagnosis", split = True, data=mbc, inner = 'quartile', palette = 'Set2');
p.set_xticklabels(rotation = 90, labels = list(bcs.columns));


# In[ ]:


sns.swarmplot(x = 'diagnosis', y = 'Volume_ln',palette = 'Set2', data = bcs);
sns.swarmplot(x = 'diagnosis', y = 'RaPeCo_ln',palette = 'Set2', data = bcs);


# In[ ]:


sns.jointplot(x = bc['RaPeCo_ln'], y = bc['Volume_ln'], stat_func=None, color="#4CB391", edgecolor = 'w', size = 6);


# Let's build a SVM to predict malignant or benign tumors

# In[ ]:


X = bcs.ix[:,0:30]

y = bcs['diagnosis']
class_names = list(y.unique())


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=42)


# Model scores very well.  Very slight over-fitting.

# In[ ]:


svc = SVC(kernel = 'linear',C=.1, gamma=10, probability = True)
svc.fit(X,y)
y_pred = svc.fit(X_train, y_train).predict(X_test)
t = pd.DataFrame(svc.predict_proba(X_test))
svc.score(X_train,y_train), svc.score(X_test, y_test)


# In[ ]:


mtrx = confusion_matrix(y_test,y_pred)
np.set_printoptions(precision = 2)

plt.figure()
plot_confusion_matrix(mtrx,classes=class_names,title='Confusion matrix, without normalization')

plt.figure()
plot_confusion_matrix(mtrx, classes=class_names, normalize = True, title='Normalized confusion matrix')

plt.show()


# In[ ]:




