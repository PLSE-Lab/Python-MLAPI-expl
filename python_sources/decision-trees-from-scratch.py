#!/usr/bin/env python
# coding: utf-8

# # ID3 Algorithm

# In the ID3 algorithm, decision trees are calculated using the concept of entropy and information gain.

# #### Entropy can be defined as:
# \begin{align}
# H(S)=-\sum_{i=1}^{N}p_{i}log_{2}p_{i}
# \end{align}

# In[ ]:


import pandas as pd
import numpy as np

# eps for making value a bit greater than 0 later on
eps = np.finfo(float).eps

from numpy import log2 as log


# Creating a dataset,

# In[ ]:


dataset = {'Taste':['Salty','Spicy','Spicy','Spicy','Spicy','Sweet','Salty','Sweet','Spicy','Salty'],
       'Temperature':['Hot','Hot','Hot','Cold','Hot','Cold','Cold','Hot','Cold','Hot'],
       'Texture':['Soft','Soft','Hard','Hard','Hard','Soft','Soft','Soft','Soft','Hard'],
       'Eat':['No','No','Yes','No','Yes','Yes','No','Yes','Yes','Yes']}


# In[ ]:


df = pd.DataFrame(dataset,columns=['Taste','Temperature','Texture','Eat'])
df


# In[ ]:


def find_entropy(df):
    '''
    Function to calculate the entropy of the label i.e. Eat.
    '''
    Class = df.keys()[-1] 
    entropy = 0
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value]/len(df[Class])
        entropy += -fraction*np.log2(fraction)
    return entropy


# In[ ]:


def find_entropy_attribute(df,attribute):
    '''
    Function to calculate the entropy of all features.
    '''
    Class = df.keys()[-1]   
    target_variables = df[Class].unique()  
    variables = df[attribute].unique()
    entropy2 = 0
    for variable in variables:
        entropy = 0
        for target_variable in target_variables:
                num = len(df[attribute][df[attribute]==variable][df[Class] ==target_variable])
                den = len(df[attribute][df[attribute]==variable])
                fraction = num/(den+eps)
                entropy += -fraction*log(fraction+eps)
        fraction2 = den/len(df)
        entropy2 += -fraction2*entropy
    return abs(entropy2)


# In[ ]:


def find_winner(df):
    '''
    Function to find the feature with the highest information gain.
    '''
    Entropy_att = []
    IG = []
    for key in df.keys()[:-1]:
#         Entropy_att.append(find_entropy_attribute(df,key))
        IG.append(find_entropy(df)-find_entropy_attribute(df,key))
    return df.keys()[:-1][np.argmax(IG)]


# In[ ]:


def get_subtable(df, node, value):
    '''
    Function to get a subtable of met conditions.
    
    node: Column name
    value: Unique value of the column
    '''
    return df[df[node] == value].reset_index(drop=True)


# In[ ]:


def buildTree(df,tree=None): 
    '''
    Function to build the ID3 Decision Tree.
    '''
    Class = df.keys()[-1]  
    #Here we build our decision tree

    #Get attribute with maximum information gain
    node = find_winner(df)
    
    #Get distinct value of that attribute e.g Salary is node and Low,Med and High are values
    attValue = np.unique(df[node])
    
    #Create an empty dictionary to create tree    
    if tree is None:                    
        tree={}
        tree[node] = {}
    
   #We make loop to construct a tree by calling this function recursively. 
    #In this we check if the subset is pure and stops if it is pure. 

    for value in attValue:
        
        subtable = get_subtable(df,node,value)
        clValue,counts = np.unique(subtable['Eat'],return_counts=True)                        
        
        if len(counts)==1:#Checking purity of subset
            tree[node][value] = clValue[0]                                                    
        else:        
            tree[node][value] = buildTree(subtable) #Calling the function recursively 
                   
    return tree


# In[ ]:


tree = buildTree(df)


# The tree splits are as follows,

# In[ ]:


import pprint
pprint.pprint(tree)


# Now, for prediction we go through each node of the tree to find the output.

# In[ ]:


def predict(inst,tree):
    '''
    Function to predict for any input variable.
    '''
    #Recursively we go through the tree that we built earlier

    for nodes in tree.keys():        
        
        value = inst[nodes]
        tree = tree[nodes][value]
        prediction = 0
            
        if type(tree) is dict:
            prediction = predict(inst, tree)
        else:
            prediction = tree
            break;                            
        
    return prediction


# In[ ]:


data = {'Taste':'Salty','Temperature':'Cold','Texture':'Hard'}


# In[ ]:


inst = pd.Series(data)


# In[ ]:


prediction = predict(inst,tree)
prediction


# # C4.5

# http://gabrielelanaro.github.io/blog/2016/03/03/decision-trees.html

# In[ ]:


# The input values
x1 = [0, 1, 1, 2, 2, 2,3,2,1]
x2 = [0, 0, 1, 1, 1, 0,2,2,1]
# The class
y = np.array([0, 0, 0, 1, 1, 0,1,0,1])


# In[ ]:


def partition(a):
    return {c: (a==c).nonzero()[0] for c in np.unique(a)}


# In[ ]:


def entropy(s):
    res = 0
    val, counts = np.unique(s, return_counts=True)
    freqs = counts.astype('float')/len(s)
    for p in freqs:
        if p != 0.0:
            res -= p * np.log2(p)
    return res


# In[ ]:


def mutual_information(y, x):

    res = entropy(y)

    # We partition x, according to attribute values x_i
    val, counts = np.unique(x, return_counts=True)
    freqs = counts.astype('float')/len(x)

    # We calculate a weighted average of the entropy
    for p, v in zip(freqs, val):
        res -= p * entropy(y[x == v])

    return res


# In[ ]:


def mutual_information(y, x):

    res = entropy(y)

    # We partition x, according to attribute values x_i
    val, counts = np.unique(x, return_counts=True)
    freqs = counts.astype('float')/len(x)

    # We calculate a weighted average of the entropy
    for p, v in zip(freqs, val):
        res -= p * entropy(y[x == v])

    return res


# In[ ]:


from pprint import pprint

def is_pure(s):
    return len(set(s)) == 1

def recursive_split(x, y):
    # If there could be no split, just return the original set
    if is_pure(y) or len(y) == 0:
        return y

    # We get attribute that gives the highest mutual information
    gain = np.array([mutual_information(y, x_attr) for x_attr in x.T])
    selected_attr = np.argmax(gain)

    # If there's no gain at all, nothing has to be done, just return the original set
    if np.all(gain < 1e-6):
        return y


    # We split using the selected attribute
    sets = partition(x[:, selected_attr])

    res = {}
    for k, v in sets.items():
        y_subset = y.take(v, axis=0)
        x_subset = x.take(v, axis=0)

        res["x_%d = %d" % (selected_attr, k)] = recursive_split(x_subset, y_subset)

    return res

X = np.array([x1, x2]).T


# In[ ]:


c4_5 = recursive_split(X, y)


# In[ ]:


def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))

        if isinstance(value, dict):
            pretty(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))


# In[ ]:


pretty(c4_5)


# # Hellinger Distance

# http://www.pythonexample.com/code/hellinger-distance-decision-tree/
# 
# https://gist.github.com/larsmans/3116927
# 
# http://videolectures.net/ecmlpkdd08_cieslak_ldtf/?q=hellinger%20distance

# Three ways of computing the Hellinger distance between two discrete
# probability distributions using NumPy and SciPy.

# In[ ]:


import time


# In[ ]:


import numpy as np
from scipy.linalg import norm
from scipy.spatial.distance import euclidean
 
_SQRT2 = np.sqrt(2)     # sqrt(2) with default precision np.float64

def hellinger_dist(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / _SQRT2


# In[ ]:


def hellinger_dist(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / _SQRT2


# In[ ]:


# The input values
x1 = [0, 1, 1, 2, 2, 2, 1]
x2 = [0, 0, 1, 1, 1, 0, 1]

# The class
y = np.array([0, 0, 0, 1, 1, 0, 0])


# In[ ]:


# repeat = 1

# p = np.array([.05, .05, .1, .1, .2, .2, .3] * repeat)
# q = np.array([  0,   0,  0, .1, .3, .3 ,.3] * repeat)

# p /= p.sum()
# q /= q.sum()

# hellinger_dist(p=p, q=q)


# In[ ]:


# The input values
x1 = [0, 1, 1, 2, 2, 2, 1]
x2 = [0, 0, 1, 1, 1, 0, 1]

# The class
y = np.array([0, 0, 0, 1, 1, 0, 0])


# In[ ]:


def partition(a):
    return {c: (a==c).nonzero()[0] for c in np.unique(a)}


# In[ ]:


from pprint import pprint

def is_pure(s):
    return len(set(s)) == 1

def recursive_split(x, y):
    
    # If there could be no split, just return the original set
    if is_pure(y) or len(y) == 0:
        return y

    # We get attribute that gives the smallest distance
    distance = np.array([hellinger_dist(y/y.sum(), x_attr/x_attr.sum()) for x_attr in x.T])
    selected_attr = np.argmin(distance)

    # If the distance is very less, nothing has to be done, just return the original set
    if np.all(distance < 1e-6):
        return y

    # We split using the selected attribute
    sets = partition(x[:, selected_attr])

    res = {}
    for k, v in sets.items():
        y_subset = y.take(v, axis=0)
        x_subset = x.take(v, axis=0)

        res["x_%d = %d" % (selected_attr, k)] = recursive_split(x_subset, y_subset)

    return res

X = np.array([x1, x2]).T
hellinger = recursive_split(X, y)


# In[ ]:


def pretty_print(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))


# In[ ]:


pretty_print(hellinger)


# In[ ]:




