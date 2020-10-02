#!/usr/bin/env python
# coding: utf-8

# **Some context to start with!**
# -------------------------------
# 
# 
# PageRank is an ingenious algorithm, developed by *Larry and Sergey*, arguably the biggest game-changer in the this world that we live in! Pagerank basically ranks the nodes in a graph structure, based on the linkages between them. An edge shared with an important node makes you important as well, a link with the spam node makes you spam too!
# 
# ## Why here? ##
# 
# In our context, every node is a question in our dataset and an edge represents a question pair. More often than not, an edge is shared by somehow related questions(topically), but may not be semantically equivalent -- This edge is however useful to visualize clusters of topics, and importance of certain nodes.
# 
# Thus, if a question is paired with a rather famous(higher pageranked) question, the question becomes relevant in itself.
# 
# Note: The implementation of a Pagerank in this context was inspired by this [discussion][1] from Krzysztof Dziedzic
# 
# 
#   [1]: https://www.kaggle.com/c/quora-question-pairs/discussion/33664

# Let's get started, shall we!
# ----------------------------

# In[ ]:


import pandas as pd

df_train = pd.read_csv('../input/train.csv').fillna("")


# The small function below computes a dictionary of questions, where each key-value pair is a question and its neighboring questions(in a list). This is necessary before we get along with calculating each question's pagerank! 

# In[ ]:


#Generating a graph of Questions and their neighbors
def generate_qid_graph_table(row):

    hash_key1 = row["qid1"]
    hash_key2 = row["qid2"]
        
    qid_graph.setdefault(hash_key1, []).append(hash_key2)
    qid_graph.setdefault(hash_key2, []).append(hash_key1)

qid_graph = {}
df_train.apply(generate_qid_graph_table, axis = 1); #You should apply this on df_test too. Avoiding here on the kernel.


# ## Cut to the chase! ##
# 
# Without getting into a lot of details, pagerank of a node is defined as the sum of a certain ratio of all its neighbors -- a complete dependence on adjacent vertices. The ratio is basically, the pagerank of the neighbor divided by the degree of the neighbor(edges incident on it). 
# 
# Mathematically speaking,
# PR(n) = PR(n1)/num_neighbors(n1) + ... + PR(n_last)/num_neighbors(n_last)
# 
# However, a damping factor is also induced in this formula, so as to account for how often the edge is to be taken(within the context of a random surfer on the web)
# 
# Thus, 
# **PR(n) = (1-d)/N + d*(PR(n1)/num_neighbors(n1) + ... + PR(n_last)/num_neighbors(n_last))**
# 

# In[ ]:


def pagerank():

    MAX_ITER = 20 #Let me know if you find an optimal iteration number!
    d = 0.85
    
    #Initializing -- every node gets a uniform value!
    pagerank_dict = {i:1/len(qid_graph) for i in qid_graph}
    num_nodes = len(pagerank_dict)
    
    for iter in range(0, MAX_ITER):
        
        for node in qid_graph:    
            local_pr = 0
            
            for neighbor in qid_graph[node]:
                local_pr += pagerank_dict[neighbor]/len(qid_graph[neighbor])
            
            pagerank_dict[node] = (1-d)/num_nodes + d*local_pr

    return pagerank_dict

pagerank_dict = pagerank()


# We initially begin with a uniform pagerank value to all nodes, and with every iteration the pageranks begin to converge. You can also introduce a minimum difference between iterations to ensure convergence.

# ## Getting the pageranks ##
# 
# Finally, a getter function to concatenate the features with the rest of the dataframe.

# In[ ]:


def get_pagerank_value(row):
    return pd.Series({
        "q1_pr": pagerank_dict[row["qid1"]],
        "q2_pr": pagerank_dict[row["qid2"]]
    })

pagerank_feats_train = df_train.apply(get_pagerank_value, axis = 1);


# ## Result ##
# 
# These features gave me a slight 0.002 bump on a 100-nround xgboost, not a magic feature by any means ;)
# 
# Fork away :)
