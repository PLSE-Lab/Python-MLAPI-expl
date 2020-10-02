#!/usr/bin/env python
# coding: utf-8

# # DonorsChoose.org - Recommendation system

# ![](http://www.futureearth.org/sites/default/files/styles/full_width_desktop/public/24188011886_5e41ab137b_k_0.jpg?itok=5e3iwrIK)

# This notebook is my attempt at answering this new challenge <br>
# My goal will be to show many different analysis and approaches. <br> 
# 
# ##### Summary
# - **Data discovery and exploration** using automated classes and graph networks 
#   - Quick exploration using objects
#   - Understanding the relation between files with graph networks
#   - Understanding the donation network (NEW)
# - **Embeddings and recommendation systems** (To be implemented)

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[4]:


import matplotlib.pyplot as plt

# Matplotlib default configuration
plt.style.use('ggplot')
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['grid.color'] = "#d4d4d4"
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['lines.linewidth'] = 2


# # Data discovery and exploration

# ## Helper class for visualization

# Great EDAs are present in other kernels. Here I will mostly focus on **quick exploration** , **production** and **reproducability**. <br>
# For this I define a ``DataExplorer`` class with simple data exploration capabilities<br>
# See below to see it in action.

# In[5]:


class DataExplorer(object):
    def __init__(self,data):
        self.data = data
        print(f"Shape : {self.data.shape}")
        
    def _repr_html_(self):
        return self.data.head()._repr_html_()
    
    #-----------------------------------------------------------------------------------------------
    # Exploration
    
    def explore(self):
        for var in self.data.columns:
            self.explore_column(var)
    
    def explore_column(self,var,max_rows = 10000,threshold_occurences = 0.5,**kwargs):
        print(f">> Exploration of {var}")
        column = self.data[var]
        dtype = column.dtype
        
        if len(column) > max_rows:
            column = column.sample(max_rows)
            
        if dtype == np.float64:
            self.show_distribution(var = var,column = column,**kwargs)
        else:
            if len(column.unique()) / len(column) < threshold_occurences:
                self.show_top_occurences(var = var,column = column,**kwargs)
            else:
                print(f"... Too many occurences, '{var}' is probably an ID")
                
        print("")
    
    #-----------------------------------------------------------------------------------------------
    # Visualizations
    
    def show_distribution(self,var = None,column = None,figsize = (15,4),kind = "hist"):
        if column is None:
            column = self.data[var]
        column.plot(kind = kind,figsize = figsize)
        if var is not None:
            plt.title(var)
        plt.show()
        
    def show_top_occurences(self,var = None,column = None,n = 30,figsize = (15,4),kind = "bar"):
        if column is None:
            column = self.data[var]
            
        column.value_counts().head(n).plot(kind = kind,figsize = figsize)
        if var is not None:
            plt.title(var)
        plt.show()


# ## Loading data
# We load all the files from kernel ``input`` data folder

# In[6]:


resources = pd.read_csv("../input/Resources.csv")
schools = pd.read_csv("../input/Schools.csv")
donors = pd.read_csv("../input/Donors.csv")
donations = pd.read_csv("../input/Donations.csv")
teachers = pd.read_csv("../input/Teachers.csv")
projects = pd.read_csv("../input/Projects.csv")


# ## Exploration of each file independently
# Let's now do a quick exploration of each data file to better understand each one independently

# ### Quick exploration of ``Resources`` file

# We can now make use of the ``DataExplorer`` class to explore the ``Resources.csv`` file quickly<br>
# As the file contains 7M rows, the explorer will only take a sample of the rows to explore faster. 

# In[ ]:


explorer = DataExplorer(resources)
explorer


# In[ ]:


explorer.explore()


# ### Quick exploration of the ``schools`` file

# In[ ]:


explorer = DataExplorer(schools)
explorer


# In[ ]:


explorer.explore()


# ### Quick exploration of ``Donors`` file

# In[ ]:


explorer = DataExplorer(donors)
explorer


# In[ ]:


explorer.explore()


# ### Quick exploration of ``Donations`` file

# In[ ]:


explorer = DataExplorer(donations)
explorer


# In[ ]:


explorer.explore()


# ### Quick exploration of ``Teachers`` file

# In[ ]:


explorer = DataExplorer(teachers)
explorer


# In[ ]:


explorer.explore()


# ### Quick exploration of ``Projects`` file

# In[ ]:


explorer = DataExplorer(projects)
explorer


# In[ ]:


explorer.explore()


# ## Connections between files
# We could study how files are connected manually, but let's use a more data-driven approach using **graph networks**

# For this we will use Graphs and the ``networkx`` library

# In[8]:


import networkx as nx

G = nx.Graph()


# Starting by extracting the columns for each data file

# In[30]:


columns = { file:list(eval(file).columns) for file in ["resources","schools","donors","donations","teachers","projects"]}


# We can build the graph now

# In[31]:


for file_i in columns:
    for file_j in columns:
        if file_i != file_j:
            intersection = set(columns[file_i]).intersection(set(columns[file_j]))
            if len(intersection) >= 1:
                G.add_edge(file_i,file_j,intersection=intersection)


# In[80]:


plt.figure(figsize = (10,10))
pos = nx.spring_layout(G)
nx.draw(G,pos = pos)
_ = nx.draw_networkx_labels(G,pos,font_weight="bold")
_ = nx.draw_networkx_edge_labels(G,pos)
plt.show()


# ***
# # Network of recommendation
# To better understand the dynamics of donations, I will highlight the structure of the network and transaction between donors and projects. <br>
# I recommend those great tutorials to get started with network data : 
# - [Exploring and analyzing network data in Python](http://programminghistorian.github.io/ph-submissions/lessons/published/exploring-and-analyzing-network-data-with-python)
# - [Graph optimization with networks](https://www.datacamp.com/community/tutorials/networkx-python-graph-tutorial)
# - The excellent [Social and Economic networks](https://fr.coursera.org/learn/social-economic-networks) course on Coursera by Matthew O. Jackson

# In[9]:


network = nx.Graph()


# ## Network creation

# Let's take only a subset of the data with only one day

# In[10]:


donations["Donation Received Date"] = pd.to_datetime(donations["Donation Received Date"])
donations.set_index("Donation Received Date",inplace = True)


# In[11]:


subset_donations = donations["2018-01"]
subset_donations.shape

Creating the graph
# In[12]:


for i,row in subset_donations.iterrows():
    donor = row["Donor ID"]
    project = row["Project ID"]
    amount = row["Donation Amount"]
    network.add_node(donor,type = "donor")
    network.add_node(project,type = "project")
    network.add_edge(donor,project,amount = amount)


# In[147]:


print(nx.info(network))


# ## Network exploration

# ### Connectedness and degree

# With that many nodes and edges, it will be impossible to visualize the network unfortunately. <br>
# Thus, let's dive a little deeper. 

# In[124]:


degree = pd.DataFrame(list(network.degree),columns = ["node_id","degree"]).set_index("node_id")
types = pd.DataFrame(pd.Series(nx.get_node_attributes(network,"type")),columns = ["node_type"])
degree = degree.join(types)
degree.head()


# In[130]:


projects_degree = degree.query("node_type=='project'")
donors_degree = degree.query("node_type=='donor'")


# Some basic statistics
# - Around 90% of the donors finance one project, most of the rest finance 2 projects, and some benefactors finance more
# - While 40% of the projects are financed by one donor, 20% by 2 and 20% by 3 and 4 persons 
# 

# In[136]:


(projects_degree["degree"].value_counts()/len(projects_degree)).head(10).plot(figsize = (15,4),label = "project")
(donors_degree["degree"].value_counts()/len(donors_degree)).head(10).plot(label = "donor")
plt.title("Distribution of degree in the network")
plt.xlabel("Node degree")
plt.ylabel("Number of nodes")
plt.legend()
plt.show()


# ### Major components

# In[17]:


from tqdm import tqdm_notebook


# In[ ]:


components = []
for i,component in enumerate(tqdm_notebook(nx.connected_component_subgraphs(network))):
    if len(component.nodes) > 2:
        components.append(component)


# ***
# # Recommendation algorithm
# New libraries for recommendations (**lightfm**, **tensorRec**) have recently been created following creation of new algorithms. <br>
# The evolution of Deep Learning has also enable the use of embeddings to represent complex variables. <br>
# I will take a shot at those libraries and algorithms in this notebook. 

# ***
# # TensorRec
# ![](https://cdn-images-1.medium.com/max/1000/1*7HdQ__6RDdtueu-GMETkng.png)

# In[6]:


import tensorrec


# ## Let's start by reproducing documentation example

# ### Generating fake data

# In[7]:


interactions, user_features, item_features = tensorrec.util.generate_dummy_data(
    num_users=100,
    num_items=150,
    interaction_density=.05
)


# In[13]:


interactions_df = pd.DataFrame(interactions.todense())
interactions_df.shape


# In[18]:


user_features


# In[19]:


item_features


# ### Training the recommendation model

# In[21]:


# Build the model with default parameters
model = tensorrec.TensorRec()


# In[22]:


# Fit the model for 5 epochs
model.fit(interactions, user_features, item_features, epochs=5, verbose=True)


# ### Making a prediction

# In[23]:


# Predict scores and ranks for all users and all items
predictions = model.predict(user_features=user_features,
                            item_features=item_features)
predicted_ranks = model.predict_rank(user_features=user_features,
                                     item_features=item_features)


# In[26]:


predictions.shape


# In[28]:


predicted_ranks.shape

