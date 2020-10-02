#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import seaborn as sns               # Provides a high level interface for drawing attractive and informative statistical graphics
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()
from subprocess import check_output

import warnings                                            # Ignore warning related to pandas_profiling
warnings.filterwarnings('ignore') 

def annot_plot(ax,w,h):                                    # function to add data to plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for p in ax.patches:
        ax.annotate('{0:.1f}'.format(p.get_height()), (p.get_x()+w, p.get_height()+h))

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


df = pd.read_csv('../input/creditcard.csv')


# In[3]:


df.head()


# In[4]:


len(df[df['Class']==1]), len(df[df['Class']==0])


# In[5]:


percentage_of_Class_0 = ((df[df['Class']==0].count())/df['Class'].count())*100
percentage_of_Class_1 = ((df[df['Class']==1].count())/df['Class'].count())*100
print(percentage_of_Class_0['Class'],'%')
print(percentage_of_Class_1['Class'],'%')


# In[6]:


ax = sns.countplot('Class',data = df)
annot_plot(ax, 0.08, 1)


# ## Split data into training and test set:

# In[7]:


y = df['Class']
x = df.drop('Class', axis = 1)


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 42)


# In[9]:


from sklearn.preprocessing import StandardScaler
Scaler_X = StandardScaler()
X_train = Scaler_X.fit_transform(X_train)
X_test = Scaler_X.transform(X_test)


# In[10]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)


# In[11]:


lr.score(X_test,y_test)


# In[12]:


from sklearn.metrics import accuracy_score, confusion_matrix
print(accuracy_score(y_pred,y_test))
print(confusion_matrix(y_pred,y_test))


# In[13]:


from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_pred,y_test))
print(confusion_matrix(y_pred,y_test))


# In[14]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# from sklearn import model_selection
# from sklearn.neighbors import KNeighborsClassifier
# 
# #Neighbors
# neighbors = np.arange(0,25)
# 
# #Create empty list that will hold cv scores
# cv_scores = []
# 
# for k in neighbors:
#     k_value = k+1
#     knn = KNeighborsClassifier(n_neighbors = k_value, weights='uniform', p=2, metric='euclidean')
#     kfold = model_selection.KFold(n_splits=10, random_state=123)
#     scores = model_selection.cross_val_score(knn, X_train, y_train, cv=kfold, scoring='accuracy')
#     cv_scores.append(scores.mean()*100)
#     print("k=%d %0.2f (+/- %0.2f)" % (k_value, scores.mean()*100, scores.std()*100))
# 
# optimal_k = neighbors[cv_scores.index(max(cv_scores))]
# print ("The optimal number of neighbors is %d with %0.1f%%" % (optimal_k, cv_scores[optimal_k]))
# plt.plot(neighbors, cv_scores)
# plt.xlabel('Number of Neighbors K')
# plt.ylabel('Train Accuracy')
# plt.show()

# as we can see that we are getting 99% accuracy for this model but this is not the case. Either majority class is overlapping or the minority class is being ignored.

# ## Random under-sampling:

# In[19]:


# Class count
count_class_0, count_class_1 = df.Class.value_counts()

# Divide by class
df_class_0 = df[df['Class'] == 0]
df_class_1 = df[df['Class'] == 1]


# In[20]:


df_class_0_under = df_class_0.sample(count_class_1)
df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)

print('Random under-sampling:')
print(df_test_under.Class.value_counts())
df_test_under.Class.value_counts().plot(kind='bar',title = 'count(Class)')


# In[21]:


y = df_test_under['Class']
x = df_test_under.drop('Class', axis = 1)


# In[22]:


X_train_under, X_test_under, y_train_under, y_test_under = train_test_split(x,y, test_size = 0.20, random_state = 42)

model = XGBClassifier()
model.fit(X_train_under,y_train_under)
y_under_pred = model.predict(X_test_under)

print(accuracy_score(y_under_pred,y_test_under)) 
confusion_matrix(y_under_pred,y_test_under)


# In[23]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train_under,y_train_under)
ylr_under_pred = lr.predict(X_test_under)

acuuracy_score = accuracy_score(y_under_pred,y_test_under)
print(acuuracy_score) 
cm = confusion_matrix(ylr_under_pred,y_test_under)
cm


# ## Random over_sampling:

# In[24]:


df_class_1_over = df_class_1.sample(count_class_0, replace=True)
df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

print('Random over-sampling:')
print(df_test_over.Class.value_counts())

df_test_over.Class.value_counts().plot(kind='bar', title='Count (target)');


# In[25]:


y_over = df_test_over['Class']
X_over = df_test_over.drop('Class', axis = 1)


# In[26]:


X_train_over, X_test_over, y_train_over, y_test_over = train_test_split(X_over,y_over, test_size = 0.20, random_state = 42)
model.fit(X_train_over,y_train_over)
y_over_pred = model.predict(X_test_over)

print(accuracy_score(y_over_pred,y_test_over))
confusion_matrix(y_over_pred, y_test_over)


# In[27]:


lr.fit(X_train_over,y_train_over)
ylr_over_pred = lr.predict(X_test_over)

print(accuracy_score(ylr_over_pred,y_test_over))
confusion_matrix(ylr_over_pred, y_test_over)


# ## Python imbalanced-learn module
# A number of more sophisticated resapling techniques have been proposed in the scientific literature.
# 
# For example, we can cluster the records of the majority class, and do the under-sampling by removing records from each cluster, thus seeking to preserve information. In over-sampling, instead of creating exact copies of the minority class records, we can introduce small variations into those copies, creating more diverse synthetic samples.
# 
# Let's apply some of these resampling techniques, using the Python library imbalanced-learn. It is compatible with scikit-learn and is part of scikit-learn-contrib projects

# In[28]:


import imblearn


# In[29]:


from sklearn.datasets import make_classification

X, y = make_classification(
    n_classes=2, class_sep=1.5, weights=[0.9, 0.1],
    n_informative=3, n_redundant=1, flip_y=0,
    n_features=20, n_clusters_per_class=1,
    n_samples=200, random_state=10
)

df = pd.DataFrame(X)
df['Class'] = y

df.Class.value_counts().plot(kind = 'bar', title = 'count(Class)')


# In[30]:


def plot_2d_space(X, y, label='Classes'):   
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()


# In[31]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X = pca.fit_transform(X)

plot_2d_space(X, y, 'Imbalanced dataset (2 PCA components)')


# ## imblearn.under_sampling: Under-sampling methods

# In[32]:


#Random under-sampling and over-sampling with imbalanced-learn

from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(return_indices = True)
X_rus, y_rus, id_rus = rus.fit_sample(X,y)

print('Removed indexes:', id_rus)

plot_2d_space(X_rus, y_rus, 'Random under-sampling')


# In[33]:


from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler()
X_ros, y_ros = ros.fit_sample(X, y)

print(X_ros.shape[0] - X.shape[0], 'new random picked points')

plot_2d_space(X_ros, y_ros, 'Random over-sampling')


# In[34]:


from imblearn.under_sampling import TomekLinks

tl = TomekLinks(return_indices=True, ratio='majority')
X_tl, y_tl, id_tl = tl.fit_sample(X, y)

print('Removed indexes:', id_tl)

plot_2d_space(X_tl, y_tl, 'Tomek links under-sampling')


# In[35]:


from imblearn.under_sampling import ClusterCentroids

cc = ClusterCentroids(ratio={0: 10})
X_cc, y_cc = cc.fit_sample(X, y)

plot_2d_space(X_cc, y_cc, 'Cluster Centroids under-sampling')


# ## Over-sampling: SMOTE
# SMOTE (Synthetic Minority Oversampling TEchnique) consists of synthesizing elements for the minority class, based on those that already exist. It works randomly picingk a point from the minority class and computing the k-nearest neighbors for this point. The synthetic points are added between the chosen point and its neighbors.

# In[36]:


from imblearn.over_sampling import SMOTE

smote = SMOTE(ratio='minority')
X_sm, y_sm = smote.fit_sample(X, y)

plot_2d_space(X_sm, y_sm, 'SMOTE over-sampling')


# ## Over-sampling followed by under-sampling

# In[37]:


from imblearn.combine import SMOTETomek

smt = SMOTETomek(ratio='auto')
X_smt, y_smt = smt.fit_sample(X, y)

plot_2d_space(X_smt, y_smt, 'SMOTE + Tomek links')


# In[ ]:





# In[ ]:





# In[ ]:





# ## Plotting ROC curve and Precision-Recall curve.
# * I find precision-recall curve much more convenient in this case as our problems relies on the "positive" class being more interesting than the negative class, but as we have calculated the recall precision, I am not going to plot the precision recall curves yet.
# 
# * AUC and ROC curve are also interesting to check if the model is also predicting as a whole correctly and not making many errors

# In[ ]:


# ROC CURVE
lr = LogisticRegression(C = best_c, penalty = 'l1')
y_pred_undersample_score = lr.fit(X_train_under,y_train_under.values.ravel()).decision_function(X_test_under.values)

fpr, tpr, thresholds = roc_curve(y_test_under.values.ravel(),y_pred_under_score)
roc_auc = auc(fpr,tpr)

# Plot ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:




