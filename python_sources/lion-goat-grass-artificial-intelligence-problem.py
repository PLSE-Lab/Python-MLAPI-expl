#!/usr/bin/env python
# coding: utf-8

# Problem:
#     A Man wants to bring a Lion, a goat, and Grass across the river. The boat is tiny and can only carry one passenger at a time. If he leaves the Lion and the goat alone together, the Lion will eat the goat. If he leaves the goat and the Grass alone together, the goat will eat the Grass.
# How can he bring all three safely across the river?
# ![](https://mark-borg.github.io/img/posts/farmer-wolf-goat-cabbage.png)
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
get_ipython().system('pip install anytree')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from anytree import Node, RenderTree,search
from queue import *
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Creating Tree**

# **Moving Farmer, Lion , Goat , Grass **

# In[ ]:


def moveMan(parent_node):
    child=dict(parent_node.name)
    #east
    if child["man"]==False:
        child["man"]=True
        return child
        
    #west
    if child["man"]==True:
        child["man"]=False
        return child
    else : return None
def moveLion(parent_node):
    child=dict(parent_node.name)
    #east
    if child["man"]==False and child["lion"]==False:
        child["man"]=True
        child["lion"]=True
        return child
        
    #west
    if child["man"]==True and child["lion"]==True:
        child["man"]=False
        child["lion"]=False
        return child
    else : return None
    
    
def moveGoat(parent_node):
    child=dict(parent_node.name)
    #east
    if child["man"]==False and child["goat"]==False:
        child["man"]=True
        child["goat"]=True
        
        
        return child
        
    #west
    if child["man"]==True and child["goat"]==True:
        child["man"]=False
        child["goat"]=False
        return child
    else : return None
    
def moveGrass(parent_node):
    child=dict(parent_node.name)
    #east
    if child["man"]==False and child["grass"]==False:
        child["man"]=True
        child["grass"]=True
        
        
        return child
        
    #west
    if child["man"]==True and child["grass"]==True:
        child["man"]=False
        child["grass"]=False
        return child
    else : return None


# **Constraints which Step is false**

# In[ ]:


def constraints(node):
    if(node["grass"]==True and node["goat"]==True and node["man"]==False):
        node["state"]=False
    if(node["grass"]==False and node["goat"]==False and  node["man"]==True):
        node["state"]=False
    if(node["lion"]==True and node["goat"]==True and  node["man"]==False):
        node["state"]=False
    if(node["lion"]==False and node["goat"]==False and node["man"]==True):
        node["state"]=False
    if(node["lion"]==True and node["goat"]==True and node["man"]==True and node["grass"]==True):
        node["state"]="Goal"
    return node


# **Constraints which Step is repeat**

# In[ ]:


def find_repeat(udo,temp):
    result=0
    for pre, fill, node in RenderTree(udo):
        if ((node.name["goat"] == temp["goat"]) and (node.name["lion"] == temp["lion"])  and (node.name["grass"] == temp["grass"])  and (node.name["man"] == temp["man"])  and (node.name["state"] == temp["state"])):
            result=1
    
    if result==1:
        temp["state"]="repeat"
        print("repeat")
    return temp
    
            


# **Main**

# **Breadth First Search**

# In[ ]:


import time
start = time. time()
Start_point={
            "man":False,
           "goat":False,
           "lion":False,
           "grass":False,
            "state":True
          }

q = Queue(maxsize=0)
udo = Node(Start_point)

q.put(udo)
while(True):
    
    Parent=q.get()
    if((Parent.name["lion"]==True and Parent.name["goat"]==True and Parent.name["grass"]==True and Parent.name["man"]==True )):
        break
    
    
    temp=moveMan(Parent)
    if temp != None:
        temp=find_repeat(udo,temp)
        constraints(temp)
        child1=Node(temp,parent=Parent)
        
        if temp["state"]==True:
            q.put(child1)


    temp=moveLion(Parent)
    if temp != None:
        temp=find_repeat(udo,temp)
        constraints(temp)
        child2=Node(temp,parent=Parent)
        if temp["state"]==True:
            q.put(child2)


    temp=moveGoat(Parent)
    if temp != None:
        temp=find_repeat(udo,temp)
        constraints(temp)
        child3=Node(temp,parent=Parent)
        if temp["state"]==True:
            q.put(child3)


    temp=moveGrass(Parent)
    if temp != None:
        temp=find_repeat(udo,temp)
        constraints(temp)
        child4=Node(temp,parent=Parent)
        if temp["state"]==True:
            q.put( child4)
    
    
    if q.empty()==1:
        break
    

for pre, fill, node in RenderTree(udo):
       print("%s%s" % (pre, node.name))     
        
end = time. time()
print("Time Taken",end - start)


# **Goal state in the tree is the final point where are the objects are True and state == Goal** 

# **Depth First Search**

# In[ ]:


import time
start = time. time()
Start_point={
            "man":False,
           "goat":False,
           "lion":False,
           "grass":False,
            "state":True
          }

q = LifoQueue()
udo = Node(Start_point)

q.put(udo)
while(True):
    
    Parent=q.get()
    if((Parent.name["lion"]==True and Parent.name["goat"]==True and Parent.name["grass"]==True and Parent.name["man"]==True )):
        break
    
    
    temp=moveMan(Parent)
    if temp != None:
        temp=find_repeat(udo,temp)
        constraints(temp)
        child1=Node(temp,parent=Parent)
        
        if temp["state"]==True:
            q.put(child1)


    temp=moveLion(Parent)
    if temp != None:
        temp=find_repeat(udo,temp)
        constraints(temp)
        child2=Node(temp,parent=Parent)
        if temp["state"]==True:
            q.put(child2)


    temp=moveGoat(Parent)
    if temp != None:
        temp=find_repeat(udo,temp)
        constraints(temp)
        child3=Node(temp,parent=Parent)
        if temp["state"]==True:
            q.put(child3)


    temp=moveGrass(Parent)
    if temp != None:
        temp=find_repeat(udo,temp)
        constraints(temp)
        child4=Node(temp,parent=Parent)
        if temp["state"]==True:
            q.put( child4)
    
    
    if q.empty()==1:
        break
    

for pre, fill, node in RenderTree(udo):
       print("%s%s" % (pre, node.name))     
        
end = time. time()
print("Time Taken",end - start)


# In[ ]:




