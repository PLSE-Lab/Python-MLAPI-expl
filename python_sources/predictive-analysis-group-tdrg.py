#!/usr/bin/env python
# coding: utf-8

# ![Image](https://www.tilburguniversity.edu/upload/c67b70dd-9a4f-499b-82e0-0aee20dfe12a_jads%20logo.png)
# 
# _____________________________________________________________________________________________________
# <center> Problem statement </center> 
# <center> Preventing and reacting on terrorist attacks is a resource intense operation. Suboptimal resource allocation increases the chances of too little investment in risky areas. </center>
# <br>
#  <center> Introduction </center> 
# <center>In this notebook you can find descriptive statistics and predictive analysis we, a group of students, have made for the course Introduction to Data Science. We made descriptive statistics for attacks over time, the locations on maps, and the effectiveness and composition of different attacks.</center>

# ## **Import data and packages**

# In[1]:


# import additional packages

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.plotly as py
import plotly.graph_objs as go
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')
init_notebook_mode()
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from mpl_toolkits.basemap import Basemap
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from sklearn.svm       import SVC
from sklearn.ensemble  import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import confusion_matrix
import itertools

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[2]:


# Import clean data

data_terrorism = pd.read_csv('../input/globalterrorismdb_0617dist.csv', encoding='ISO-8859-1', usecols=[1,2,7,8,9,13,14,16,26,27,28,34,68,69,71,81,98,100,101,103,104,105])
data_terrorism.info()


# ## **Explore dataset**

# In[3]:


# Create copies of the original dataset for the predictive part

data1 = data_terrorism
data2 = data_terrorism
data3 = data_terrorism
data4 = data_terrorism


# In[4]:


# Data head and drop NaN for nkill and nwound (variables of interest)

data_terrorism.dropna(subset=['nkill'])
data_terrorism.dropna(subset=['nwound'])
data_terrorism.head()


# ## **Predictive analysis**

# In our predictive analysis we will predict whether or not a future terrorist attack will have more deaths than the average amount of deaths of a terrorist attack. 
# 
# Our first step in predicting this value, is creating different datasets, that we will use to test our different models. After this, we have made functions that are helpfull for cross-validation, esting the accuracy and plotting the confusion matrix.
# 
# We have decided to come up with two models with different features. The first one is our most accurate model, that, as the name suggest, is the most accurate without overfitting the model. But we decided to make another model as well: the most applicable. This models features are selected based upon accuracy, but the have to be easy accessible during a terrorist attack. In other words, when a terrorist attack happens, it is hard to know the exact number of wounded people, so using this value to predict if there are more or less deaths than average is a bit useless in practic (whenever you have determined the amounts of wounded, you are likely to have determined the amounts of deaths as well). Therefore, we only used features that can be determined relatalively easily (at least faster than deteremining the amounts of deaths). This also means that the second model can be used by intelligence groups to predict the impact of possible future terrorist attacks, and thus being able to allocate resources better in advance.
# 
# For each model, we will first do a feature selection. Based on this selection, we will use those features in three different settings: 
# 
# * Accuracy Support Vector Machines
# * Random Foress
# * K-Nearest-Neighbors
# 
# Each of those, will give different accuracies. Our last step is to select the model with the highest accuracy (for both the most accurate and most applicable model).
# 
# 
# 

# In[5]:


#Data cleaning for predictive analysis

#Preparing the data for usage/creating new variables
data1 = data1[pd.notnull(data1['nkill'])]

total_nkill1 = data1.nkill.sum()
avg_kill1 = total_nkill1/data1.nkill.count()
data1["avg_nkill"] = avg_kill1
data1["boolean_nkill"] = data1["nkill"] > data1["avg_nkill"]

total_nkill2 = data2.nkill.sum()
avg_kill2 = total_nkill2/data2.nkill.count()

data2["avg_nkill"] = avg_kill2
data2["boolean_nkill"] = data2["nkill"] > data2["avg_nkill"]

total_nkill3 = data3.nkill.sum()
avg_kill3 = total_nkill3/data3.nkill.count()

data3["avg_nkill"] = avg_kill3
data3["boolean_nkill"] = data3["nkill"] > data3["avg_nkill"]

total_nkill4 = data4.nkill.sum()
avg_kill4 = total_nkill4/data4.nkill.count()

data4["avg_nkill"] = avg_kill4
data4["boolean_nkill"] = data4["nkill"] > data4["avg_nkill"]

#deleting NaN values
data1 = data1[pd.notnull(data1['imonth'])]
data1 = data1[pd.notnull(data1['country'])]
data1 = data1[pd.notnull(data1['region'])]
data1 = data1[pd.notnull(data1['vicinity'])]
data1 = data1[pd.notnull(data1['success'])]
data1 = data1[pd.notnull(data1['suicide'])]
data1 = data1[pd.notnull(data1['attacktype1'])]
data1 = data1[pd.notnull(data1['targtype1'])]
data1 = data1[pd.notnull(data1['individual'])]
data1 = data1[pd.notnull(data1['nperps'])]
data1 = data1[pd.notnull(data1['claimed'])]
data1 = data1[pd.notnull(data1['weaptype1'])]
data1 = data1[pd.notnull(data1['nkillter'])]
data1 = data1[pd.notnull(data1['nwound'])]

data2 = data2[pd.notnull(data2['nkill'])]
data2 = data2[pd.notnull(data2['imonth'])]
data2 = data2[pd.notnull(data2['suicide'])]
data2 = data2[pd.notnull(data2['nkillter'])]
data2 = data2[pd.notnull(data2['nwound'])]

data3 = data3[pd.notnull(data3['nkill'])]
data3 = data3[pd.notnull(data3['country'])]
data3 = data3[pd.notnull(data3['region'])]
data3 = data3[pd.notnull(data3['vicinity'])]
data3 = data3[pd.notnull(data3['suicide'])]
data3 = data3[pd.notnull(data3['attacktype1'])]
data3 = data3[pd.notnull(data3['targtype1'])]
data3 = data3[pd.notnull(data3['nperps'])]
data3 = data3[pd.notnull(data3['weaptype1'])]
data3 = data3[pd.notnull(data3['imonth'])]

data4 = data4[pd.notnull(data4['nkill'])]
data4 = data4[pd.notnull(data4['imonth'])]
data4 = data4[pd.notnull(data4['suicide'])]
data4 = data4[pd.notnull(data4['targtype1'])]
data4 = data4[pd.notnull(data4['nperps'])]

#reindexing the resulting data
data1.index = pd.RangeIndex(len(data1.index))
data2.index = pd.RangeIndex(len(data2.index))
data3.index = pd.RangeIndex(len(data3.index))
data4.index = pd.RangeIndex(len(data4.index))

#isolate target data and converting it into 0/1 values
y1 = data1['boolean_nkill']
y1 = y1*1

y2 = data2['boolean_nkill']
y2 = y2*1

y3 = data3['boolean_nkill']
y3 = y3*1

y4 = data4['boolean_nkill']
y4 = y4*1


# identifying data to drop, then dropping it
to_drop1 = ['iyear','nkill','boolean_nkill', 'nwoundte', 'country_txt', 'latitude', 'longitude', 'property', 'propextent', 'avg_nkill']
data1_feat_space = data1.drop(to_drop1,axis=1)

to_drop2 = ['iyear','country', 'region', 'vicinity', 'success', 'attacktype1', 'targtype1','individual', 'nperps', 'claimed', 'weaptype1', 'nwound',  'nkill', 'boolean_nkill', 'nwoundte', 'country_txt', 'latitude', 'longitude', 'property', 'propextent', 'avg_nkill']
data2_feat_space = data2.drop(to_drop2,axis=1)

to_drop3 = ['success', 'individual', 'claimed', 'nkillter', 'nwound', 'iyear','nkill','boolean_nkill', 'nwoundte', 'country_txt', 'latitude', 'longitude', 'property', 'propextent', 'avg_nkill']
data3_feat_space = data3.drop(to_drop3,axis=1)

to_drop4 = ['nwound','nkillter','iyear','country', 'region', 'vicinity', 'success', 'attacktype1', 'individual', 'claimed', 'weaptype1', 'nwound',  'nkill', 'boolean_nkill', 'nwoundte', 'country_txt', 'latitude', 'longitude', 'property', 'propextent', 'avg_nkill']
data4_feat_space = data4.drop(to_drop4,axis=1)


# In[6]:


#Defining helpful functions

def run_cv(X,y,clf_class,**kwargs):
    # Construct a kfolds object
    kf = KFold(len(y),n_folds=5,shuffle=True)
    y_pred = y.copy()
    # Iterate through folds
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[test_index] = clf.predict(X_test)
    return y_pred

def accuracy(y_true,y_pred):
    return np.sum(y_true == y_pred)/float(y_true.shape[0])

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greys,
                          print_confusion_matrix = False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fs = 20
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=fs)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=fs)
    plt.yticks(tick_marks, classes, fontsize=fs)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        statement = "Normalized confusion matrix "
    else:
        statement = 'Confusion matrix, without normalization'

    if print_confusion_matrix:
        print(statement)
        print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=fs)
    plt.xlabel('Predicted label', fontsize=fs)


# **Feature selection: most accurate model**
# 

# In[7]:


# Pull out features for future use
features1 = data1_feat_space.columns
X1 = data1_feat_space.as_matrix().astype(np.float)

# Build a forest and compute the feature importances
forest1 = ExtraTreesClassifier(n_estimators=250,random_state=0)

forest1.fit(X1, y1)
importances1 = forest1.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest1.estimators_],
             axis=0)
indices1 = np.argsort(importances1)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X1.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices1[f], importances1[indices1[f]]))
    
# Plot the feature importances of the forest
plt.figure(figsize=(20,10))
plt.title("Feature importances")
plt.bar(range(X1.shape[1]), importances1[indices1],
       color="r", yerr=std[indices1], align="center")
plt.xticks(range(X1.shape[1]), indices1)
plt.xlim([-1, X1.shape[1]])
plt.show()


# The bar graph above shows the importances of each feature. The first 4 features will be used to create a prediction model. These 4 features are: 13: number of wounded persons of the attack , 12: number of terrorist killed in the attack, 0: the month the attack occurs in, and 5 whether the terrorist(s) committed suicide during the attack. The model based on those 4 features is created below.

# **Predictive analysis: most accurate model**

# As you can see, both the Random Forest and the Accuracy Support Vector Machines have the same accuracy of 0.863. This is slightly higher than the accuracy obtained with K-nearest-neighbors. Therefore, Random Forest and the Accuracy Support Vector Machines are the best fit for our data and our predictive research question.

# In[8]:


# Pull out features for future use
features2 = data2_feat_space.columns
X2 = data2_feat_space.as_matrix().astype(np.float)

#normalization
scaler2 = StandardScaler()
X2 = scaler2.fit_transform(X2)

print("Feature space holds %d observations and %d features" % X2.shape)
print("Unique target labels:", np.unique(y2))

print("Accuracy Support vector machines: %.3f" % accuracy(y2, run_cv(X2,y2,SVC)))
print("Random forest: %.3f" % accuracy(y2, run_cv(X2,y2,RF)))
print("K-nearest-neighbors: %.3f" % accuracy(y2, run_cv(X2,y2,KNN)))
  
y2 = np.array(y2)
class_names2 = np.unique(y2)

confusion_matrices2 = [
    ( "Support Vector Machines", confusion_matrix(y2,run_cv(X2,y2,SVC)) ),
    ( "Random Forest", confusion_matrix(y2,run_cv(X2,y2,RF)) ),
    ( "K-Nearest-Neighbors", confusion_matrix(y2,run_cv(X2,y2,KNN)) ),
]

plt.figure(figsize=(20,10))
for idx in range(3):
    plt.subplot(1,3,idx+1)
    plot_confusion_matrix(confusion_matrices2[idx][1],["False","True"],title=confusion_matrices2[idx][0])


#  **Feature selection - most applicable model**

# As you can see, the Accuracy Support Vector Machines has the highest accuracy of the three classification methods; 0.828. This is slightly higher than the accuracy obtained with K-nearest-neighbors and Random Forest . Therefore, the Accuracy Support Vector Machines is the best fit for our data and our predictive research question.
# 

# In[9]:


# Pull out features for future use
features3 = data3_feat_space.columns
X3 = data3_feat_space.as_matrix().astype(np.float)

# Build a forest and compute the feature importances
forest3 = ExtraTreesClassifier(n_estimators=250,random_state=0)

forest3.fit(X3, y3)
importances3 = forest3.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest3.estimators_],
             axis=0)
indices3 = np.argsort(importances3)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X3.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices3[f], importances3[indices3[f]]))

# Plot the feature importances of the forest
plt.figure(figsize=(20,10))
plt.title("Feature importances")
plt.bar(range(X3.shape[1]), importances3[indices3],
       color="r", yerr=std[indices3], align="center")
plt.xticks(range(X3.shape[1]), indices3)
plt.xlim([-1, X3.shape[1]])
plt.show()


# The bar graph above shows the importances of each feature. The first 4 features will be used to create a prediction model. These 4 features are: 0: the month the attack occurs in, 1: the country the attack occurs in, 7: the number of terrorists executing the attack, and 4 whether the terrorist(s) committed suicide during the attack. The model based on those 4 features is created below.

# **Predictive analysis: most applicable model**

# In[10]:


# Pull out features for future use
features4 = data4_feat_space.columns
X4 = data4_feat_space.as_matrix().astype(np.float)

# Scaling Features
scaler4 = StandardScaler()
X4 = scaler4.fit_transform(X4)

print("Feature space holds %d observations and %d features" % X4.shape)
print("Unique target labels:", np.unique(y4))

#printing the accuracy
print("Accuracy Support vector machines: %.3f" % accuracy(y4, run_cv(X4,y4,SVC)))
print("Random forest: %.3f" % accuracy(y4, run_cv(X4,y4,RF)))
print("K-nearest-neighbors: %.3f" % accuracy(y4, run_cv(X4,y4,KNN)))

#creating confusion matrices
y4 = np.array(y4)
class_names4 = np.unique(y4)

confusion_matrices4 = [
    ( "Support Vector Machines", confusion_matrix(y4,run_cv(X4,y4,SVC)) ),
    ( "Random Forest", confusion_matrix(y4,run_cv(X4,y4,RF)) ),
    ( "K-Nearest-Neighbors", confusion_matrix(y4,run_cv(X4,y4,KNN)) ),
]

plt.figure(figsize=(20,10))
for idx in range(3):
    plt.subplot(1,3,idx+1)
    plot_confusion_matrix(confusion_matrices4[idx][1],["False","True"],title=confusion_matrices4[idx][0])


# ### **Conclusions**

# Being able to determine the consequences of terrorist attacks will improve the accuracy of predictions related to the consequences of possible future attacks. This is important to optimize the risk management related to terrorist attacks, and thus enable better allocation of resources. Optimized risk management increases the odds to prevent future attacks. This will lead to our main problem statement: ** Preventing and reacting on terrorist attacks is a resource intense operation. Suboptimal resource allocation increases the chances of too little investment in risky areas.** In our predictive analysis the main problem was thus: predicting whether or not a future terrorist attack will have more deaths than the average amount of deaths of a terrorist attack. 
# 
# The main conclusions to be made from the **descriptive analysis** are:
# * Although there are focus areas where most terrorist attacks have been committed, these regions are known for its' violence and it is thus not surprising. What is surprising at first sight though is that terrorism attacks (in the broad sense of its word) is more widespread, with attacks in nearly all major countries of the world. 
# * In the past 45 years there have been periods of relative calmness (1970's, and 1997 and 2003), and a steep increase after 2003, this can be due to better reporting of acts of terrorism, or increased terrorism violence.
# * with more than 80 percent of the kills, most people in history have been killed by mostly two weapon / attack types: namely firearms or explosives.
# 
# The main conclusions to be made from the applicable model used in the **predictive analysis** can be summarized as follows: with using the features Suicide, number of terrorists involved (nperps), imonth and the target type we can predict whether an attack will have a higher than average impact on the number of people killed with an accuracy of 83%. To be able to predict the impact of an attack institutions are better able to allocate their resources and determine risky situations. Changing the model towards less practical predictors we are able to increase the accuracy of the model to 86.3%.
# 
