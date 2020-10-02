#!/usr/bin/env python
# coding: utf-8

# In[1]:


##All General Import Statements
import pandas as pd
import numpy as np
import math
import random
get_ipython().run_line_magic('matplotlib', 'inline')
import random
from matplotlib import pyplot
import os
print(os.listdir("../input"))


# # Anomaly Detection Algorithms: Isolation Forest vs the Rest

# <p>This notebook shows a simplified implementation of the algorithm Isolation Forest and compares its Scikit-learn implementation with other popular anomaly detection algorithms. (KMeans, Local Outlier Factor, One-Class SVM)</p>

# ## Isolation Forests in Python

# The Algorithm has 3 parts:
#     1. Forest
#     2. Isolation Tree
#     3. Evaluation (Path Length)

# ### Common Classes

# The below code defines classes for external and internal nodes

# In[2]:


class ExNode:
    def __init__(self,size):
        self.size=size
        
class InNode:
    def __init__(self,left,right,splitAtt,splitVal):
        self.left=left
        self.right=right
        self.splitAtt=splitAtt
        self.splitVal=splitVal


# ### Forest

# In[3]:


def iForest(X,noOfTrees,sampleSize):
    forest=[]
    hlim=math.ceil(math.log(sampleSize,2))
    for i in range(noOfTrees):
        X_train=df_data.sample(sampleSize)
        forest.append(iTree(X_train,0,hlim))
    return forest


# ### Isolation Tree

# In[4]:


def iTree(X,currHeight,hlim):
    if currHeight>=hlim or len(X)<=1:
        return ExNode(len(X))
    else:
        Q=X.columns
        q=random.choice(Q)
        p=random.choice(X[q].unique())
        X_l=X[X[q]<p]
        X_r=X[X[q]>=p]
        return InNode(iTree(X_l,currHeight+1,hlim),iTree(X_r,currHeight+1,hlim),q,p)


# ### Path Length

# In[5]:


def pathLength(x,Tree,currHeight):
    if isinstance(Tree,ExNode):
        return currHeight
    a=Tree.splitAtt
    if x[a]<Tree.splitVal:
        return pathLength(x,Tree.left,currHeight+1)
    else:
        return pathLength(x,Tree.right,currHeight+1)


# ## Test Run

# Let us now test the algorithm on a test dataset.
# Source: https://www.kaggle.com/dalpozz/creditcardfraud

# In[6]:


df=pd.read_csv("../input/creditcard.csv")
y_true=df['Class']
df_data=df.drop('Class',1)


# Next, we create the forest.

# In[7]:


sampleSize=10000
ifor=iForest(df_data.sample(100000),10,sampleSize) ##Forest of 10 trees


# Next, we select 1000 random datapoints and get their path lengths. The purpose for this is to plot and see if anomalies actually have shorter path lengths.

# In[8]:


posLenLst=[]
negLenLst=[]

for sim in range(1000):
    ind=random.choice(df_data[y_true==1].index)
    for tree in ifor:
        posLenLst.append(pathLength(df_data.iloc[ind],tree,0))
        
    ind=random.choice(df_data[y_true==0].index)
    for tree in ifor:
        negLenLst.append(pathLength(df_data.iloc[ind],tree,0))


# Finally, we plot the path lengths.

# In[9]:


bins = np.linspace(0,math.ceil(math.log(sampleSize,2)), math.ceil(math.log(sampleSize,2)))

pyplot.figure(figsize=(12,8))
pyplot.hist(posLenLst, bins, alpha=0.5, label='Anomaly')
pyplot.hist(negLenLst, bins, alpha=0.5, label='Normal')
pyplot.xlabel('Path Length')
pyplot.ylabel('Frequency')
pyplot.legend(loc='upper left')


# Anomalies do seem to have a lower path length. Not bad for random division!

# #### Notes:

# The above implementation ignores three aspects of the actual algorithm fo the sake of simplicity.
# 1. The average depth needs to be added to the depth once the current length hits the height limit
# 2. The path lengths are not normalized between trees and hence the actual values are used for plotting
# 3. The authors of the paper suggest using kurtosis to select features as a refinement over random selection

# ## Plotting the Data

# Using a technique called T-SNE, we can reduce the dimensions of the data and create a 2D plot. The objective here is to show that distance based anomaly detection methods might not work as well as other techniques on this dataset. This is because the positive cases are not too far away from the normal cases.

# In[10]:


from sklearn.manifold import TSNE


# In[11]:


df_plt=df[df['Class']==0].sample(1000)
df_plt_pos=df[df['Class']==1].sample(20)
df_plt=pd.concat([df_plt,df_plt_pos])
y_plt=df_plt['Class']
X_plt=df_plt.drop('Class',1)


# In[12]:


X_embedded = TSNE(n_components=2).fit_transform(X_plt)


# In[13]:


pyplot.figure(figsize=(12,8))
pyplot.scatter(X_embedded[:,0], X_embedded[:,1], c=y_plt, cmap=pyplot.cm.get_cmap("Paired", 2))
pyplot.colorbar(ticks=range(2))


# ## Time for the Real Fight!

# To keep things even, all of the algorithms are run with their default parameters.

# Let's start by importing the scikit-learn implementations of all 4 algorithms.

# In[14]:


from sklearn.ensemble import IsolationForest


# Next, let's create a train and test dataset.

# In[15]:


df_data.head()


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(df_data, y_true, test_size=0.3, random_state=42)


# Finally, let's create a few helper functions that help with training and testing the models. The preprocess function is not used in this notebook but it might help improve the scores on the KMeans and One Class SVM models.

# <b>Note:</b> The below train and predict functions are designed to output ensemble models (bagged models), with the default size being 5 models. The Isolation Forest and One Class SVM use these functions.

# In[18]:


## Not valid for LOF
def train(X,clf,ensembleSize=5,sampleSize=10000):
    mdlLst=[]
    for n in range(ensembleSize):
        X=df_data.sample(sampleSize)
        clf.fit(X)
        mdlLst.append(clf)
    return mdlLst


# In[19]:


## Not valif for LOF
def predict(X,mdlLst):
    y_pred=np.zeros(X.shape[0])
    for clf in mdlLst:
        y_pred=np.add(y_pred,clf.decision_function(X).reshape(X.shape[0],))
    y_pred=(y_pred*1.0)/len(mdlLst)
    return y_pred


# Finally, let's import some model scoring libraries. Since, we are dealing with a heavily imbalanced dataset, F1 Score is used as a proxy for model performance.

# For more details, refer http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

# In[20]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix,f1_score


# #### Isolation Forest

# In[21]:


alg=IsolationForest(n_estimators=100, max_samples='auto', contamination=0.01,                         max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)


# <b>Note:</b> The magic function timeit does not let us use any variable that is created in the timeit cell. Hence, ecery cell with a timeit magic function will have a corresponding regular cell with the same code.

# In[22]:


get_ipython().run_cell_magic('timeit', '', 'if_mdlLst=train(X_train,alg)')


# In[ ]:


if_mdlLst=train(X_train,alg)


# In[ ]:


get_ipython().run_cell_magic('timeit', '', 'if_y_pred=predict(X_test,if_mdlLst)\nif_y_pred=1-if_y_pred\n\n#Creating class labels based on decision function\nif_y_pred_class=if_y_pred.copy()\nif_y_pred_class[if_y_pred>=np.percentile(if_y_pred,95)]=1\nif_y_pred_class[if_y_pred<np.percentile(if_y_pred,95)]=0')


# In[ ]:


if_y_pred=predict(X_test,if_mdlLst)
if_y_pred=1-if_y_pred

#Creating class labels based on decision function
if_y_pred_class=if_y_pred.copy()
if_y_pred_class[if_y_pred>=np.percentile(if_y_pred,95)]=1
if_y_pred_class[if_y_pred<np.percentile(if_y_pred,95)]=0


# In[ ]:


roc_auc_score(y_test, if_y_pred_class)


# In[ ]:


f1_score(y_test, if_y_pred_class)


# In[ ]:


if_cm=confusion_matrix(y_test, if_y_pred_class)


# In[ ]:


import seaborn as sn
     
df_cm = pd.DataFrame(if_cm,
                  ['True Normal','True Fraud'],['Pred Normal','Pred Fraud'])
pyplot.figure(figsize = (8,4))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 16},fmt='g')# font size


# In[ ]:




