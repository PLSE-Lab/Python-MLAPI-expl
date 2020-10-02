#!/usr/bin/env python
# coding: utf-8

# **Predictive Maintenance
# **Based on kaggle dataset
# 
# https://www.kaggle.com/ludobenistant/predictive-maintenance-1/data
# 
# Step by step approach with pandas data frames, then have a HL view of Dask possibilities, then see NN of the data.

# In[1]:


# Import Libraries needed
import pandas as pd                 #dataframe manipulation
import numpy as np                  #numerical processing of vectors
import matplotlib.pyplot as plt     #plotting
get_ipython().run_line_magic('matplotlib', 'inline')

#import tensorflow as tf
import sklearn
from sklearn import tree
import graphviz
import dask


print("Pandas:\t\t", pd.__version__)
print("Numpy:\t\t", np.__version__)
#print("Tensorflow:\t", tf.__version__)
print("Dask:\t\t", dask.__version__)
print("Scikit-learn:\t", sklearn.__version__)


# In[2]:


df_init = df = pd.read_csv('../input/maintenance_data.csv')


# In[3]:


df.columns


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.describe()


# In[7]:


df.sort_values(by='lifetime', ascending=True).head()


# In[8]:


df.sort_values(by='lifetime', ascending=True).tail()


# In[9]:


plt.bar(df.sort_values('team').team, df.sort_values('lifetime').lifetime)


# In[10]:


df.groupby([df.team, df.broken]).count()


# In[11]:


df.groupby(['team','broken']).agg({'broken': 'count'}).apply(lambda x:100 * x / float(x.sum()))


# In[12]:


show_perc = df.groupby(['team','broken']).agg({'broken': 'count'})
show_perc.apply(lambda x:100 * x / float(x.sum()))


# In[13]:


column = 'provider'
show_perc = df.loc[df['broken'] == 1].groupby([column]).agg({'broken': 'count'})
show_perc.apply(lambda x:round(100 * x / float(x.sum()),2)).rename(columns={"broken": "%"})


# **Decision Tree Classification of existing Data with Scikit-learn
# **Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. 
# 
# http://scikit-learn.org/stable/modules/tree.html

# In[14]:


tree_data = df_init.drop('broken', axis=1)
tree_target = df_init.broken

#workaround replacement strings to integers - DO NOT DO IT LIKE THIS ;-)
try:
    tree_data.replace('TeamA',1, inplace=True)
    tree_data.replace('TeamB',2, inplace=True)
    tree_data.replace('TeamC',3, inplace=True)
    tree_data.replace('Provider1',1, inplace=True)
    tree_data.replace('Provider2',2, inplace=True)
    tree_data.replace('Provider3',3, inplace=True)
    tree_data.replace('Provider4',4, inplace=True)
except:
    pass  

#convert dataframes to arrays
tree_data = tree_data.values
tree_target = tree_target.values
#column names - labels
tree_feature_names = ['lifetime', 'pressureInd', 'moistureInd', 'temperatureInd', 'team', 'provider']
#target names - class
tree_target_names = ['BROKEN!','Operational']

#Tree Classifiers
tree_clf = tree.DecisionTreeClassifier()

#tree_clf.set_params(max_depth=3)

tree_clf = tree_clf.fit(tree_data, tree_target)

tree_clf.get_params()


# In[15]:


#output graph tree
tree_dot_data = tree.export_graphviz(tree_clf, 
                                out_file=None, 
                                feature_names=tree_feature_names,
                                class_names=tree_target_names,
                                filled=True, 
                                rounded=True,
                                special_characters=True) 
graph = graphviz.Source(tree_dot_data) 
graph.render("Maintenance_classification_tree")
#show tree
graph


# In[16]:


df_init.drop('broken', axis=1).columns


# In[17]:


#PREDICTION WITHOUT REGRESSION - 1-->BROKEN 0-->Operational
for t in range(1,5):
    for p in range(1,5):
        print("team\tprov\tlife\tno\tyes")
        for i in range(10,110,10):
            arr = tree_clf.predict_proba([[float(i), 100., 100., 100., t, p]])
            print(t, "\t", p, "\t", i, "\t", round(arr[0][0], 2), "\t", round(arr[0][1], 2))


# In[ ]:





# In[ ]:




