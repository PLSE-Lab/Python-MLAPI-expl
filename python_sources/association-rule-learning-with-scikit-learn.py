#!/usr/bin/env python
# coding: utf-8

# Created by: Sangwook Cheon
# 
# Date: Dec 24, 2018
# 
# This is step-by-step guide to Association Rule Learning (APL) using scikit-learn, which I created for reference. I added some useful notes along the way to clarify things. This notebook's content is from A-Z Datascience course, and I hope this will be useful to those who want to review materials covered, or anyone who wants to learn the basics of Association Rule Learning.
# 
# ## Content:
# ### 1. Apriori
# ### 2. Eclat

# # Apriori
# Informal definition: "Customer who bought this will also buy..." --> Apriori algorithm figures this out. This is used for optimization of combination of things.
# 
# **Definition of terms**
# 
# 1) Support
# ![i98798](https://i.imgur.com/KeZbncP.png)
# 
# 2) Confidence
# ![i9878868](https://i.imgur.com/ji53xXW.png)
# 
# 3) Lift
# ![i234](https://i.imgur.com/RLrYFwV.png)
# 
# **Steps**
# 
# ![i99](https://i.imgur.com/rNd604p.png)
# 
# Companies like Amazon and Netflix does use this algorithm, but they have more sophisticated custom-made algorithms. But Apriori algorithm is a great standard algorithm for optimization. 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('../input/Market_Basket_Optimisation.csv', header = None) #To make sure the first row is not thought of as the heading
dataset.shape

#Transforming the list into a list of lists, so that each transaction can be indexed easier
transactions = []
for i in range(0, dataset.shape[0]):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

print(transactions[0])


# In[ ]:


from apyori import apriori
# Please download this as a custom package --> type "apyori"
# To load custom packages, do not refresh the page. Instead, click on the reset button on the Console.

rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
# Support: number of transactions containing set of times / total number of transactions
# .      --> products that are bought at least 3 times a day --> 21 / 7501 = 0.0027
# Confidence: Should not be too high, as then this wil lead to obvious rules

#Try many combinations of values to experiment with the model. 

#viewing the rules
results = list(rules)


# In[ ]:


#Transferring the list to a table

results = pd.DataFrame(results)
results.head(5)


# Notice that the rows are sorted by relevance. Top ones have large support, which means the rules are strong.

# # Eclat
# 
# In this model, only Support value is used, which shows how frequent a set of itmes occur. Therefore, Eclat is a simplified version of Apriori model.
