#!/usr/bin/env python
# coding: utf-8

# <h1 style="color:brown">Supervised Learning: Regression

# ## Introduction to Supervised Learning

# ---

# ### Introduction

# Supervised learning. If you are hearing or reading this term for the first time, then it may be completely unclear what it means. Don't worry. In this lab, you will get a comprehensive understanding of supervised learning; and, in the next chapter of the experiment, you will learn to use supervised learning to complete data prediction.

# ### Key Points

# - Review of machine learning<br>
# - Algorithms of machine learning<br>
# - Introduction to supervised learning<br>
# - Classification and regression

# ### Environments<br>
# 

# - Python 3.6

# ---

# ## Review of Machine Learning

# In this session. We will look at the following:<br>
# <br>
# - Introduction to the Machine Learning<br>

# Of the remaining sessions of Machine Learning for Financial Data Course, the first 5 weeks belong to the "traditional machine learning" content. The use of the term "traditional machine learning" is not very accurate, so we use the more academic and accurate term "statistical machine learning".<br>
# <br>
# Statistical machine learning is an interdisciplinary subject of probability theory, statistics, computational theory, optimization methods and computer science. Its main research objective is how to learn from experience and improve the performance of specific algorithms.<br>
# <br>
# At present, what we usually call "machine learning" often actually refers to "statistical machine learning". It can be roughly divided into four categories:<br>
# <br>
# - Supervised learning<br>
# - Unsupervised learning<br>
# - Semi-supervised learning<br>
# - Reinforcement learning

# In the introduction to the course, we are presenting Figure 1 to show the relationship between machine learning, deep learning and artificial intelligence. If you want to include "statistical machine learning" in this figure, it should be placed between the 1980's and 2010's. Of course, this is not to say that after 2010, statistical machine learning didn't develop. On the contrary, areas such as reinforcement learning and unsupervised learning still require unremitting exploration.

# In[ ]:


from IPython.display import Image
import os
Image("../input/week-3-images/intro1.jpg")


# ### Introduction to Supervised Learning

# Above, we introduced that machine learning is usually divided into four categories. These four categories are subdivided into dozens of different machine learning algorithms/methods. See Table 1 for details:

# <div style="color: #999; font-size: 12px; text-align: center;">Table 1: Categorization of machine learning algorithms (not complete).</div><br>
# <table><br>
#   <tr><br>
#     <th>Category</th><br>
#     <th>Specific Method</th><br>
#   </tr><br>
#   <tr><br>
#     <td rowspan="8">Supervised Learning</td><br>
#     <td>Artificial Neural Networks</td><br>
#   </tr><br>
#   <tr><br>
#     <td>Bayesian Network</td><br>
#   </tr><br>
#   <tr><br>
#     <td>Support Vector Machines</td><br>
#   </tr><br>
#   <tr><br>
#     <td>Random Forest</td><br>
#   </tr><br>
#   <tr><br>
#     <td>Logistic Regression</td><br>
#   </tr><br>
#   <tr><br>
#     <td>K-Nearest Neighbor</td><br>
#   </tr><br>
#   <tr><br>
#     <td>Decision Tree</td><br>
#   </tr><br>
#   <tr><br>
#     <td>Hidden Markov Model</td><br>
#   </tr><br>
#   <tr><br>
#     <td rowspan="3">Unsupervised Learning</td><br>
#     <td>Artificial Neural Networks</td><br>
#   </tr><br>
#   <tr><br>
#     <td>Association Rule Learning</td><br>
#   </tr><br>
#   <tr><br>
#     <td>Hierarchical Clustering</td><br>
#   </tr><br>
#   <tr><br>
#     <td rowspan="4">Semi-supervised Learning</td><br>
#     <td>Generative Model</td><br>
#   </tr><br>
#   <tr><br>
#     <td>Low Density Separation</td><br>
#   </tr><br>
#   <tr><br>
#     <td>Joint Training</td><br>
#   </tr><br>
#   <tr><br>
#     <td>Graph Method</td><br>
#   </tr><br>
#   <tr><br>
#     <td rowspan="3">Reinforcement Learning</td><br>
#     <td>Time Difference Learning</td><br>
#   </tr><br>
#   <tr><br>
#     <td>Q Learning</td><br>
#   </tr><br>
#   <tr><br>
#     <td>SARSA</td><br>
#   </tr><br>
# </table>

# So, here comes the question: **What does it mean to supervise a learning? What are the characteristics of the various methods of supervision?**

# ### Definition of Supervised Learning

# For the definition of supervised learning, here is a quote from the famous machine learning expert *Mehryar Mohri* in his monograph *Foundations of Machine Learning*:<br>
# <br>
# >*Supervised learning is the machine learning task of learning a function that maps an input to an output based on example input-output pairs. It infers a function from labeled training data consisting of a set of training examples.*

# Analyze a few of the keywords in this sentence. The `example input-output pairs` actually constitute the **training data set**, while the `input` refers to the **feature variable** in the training data set and the `output` is the **label**. The establishment of `functions` is actually training machine learning prediction models. This sentence is actually a typical machine learning process. **The key to supervised learning is that the training data sets provided here are labeled**.

# ### A Simple Instance

# In order to better understand the above definition of supervised learning, the following is an example of judging the species of flower:<br>
# <br>
# As shown in Figure 2, the training data set gives the petal length (training set feature) of three different flowers. We already know the species of the three flowers A, B, C (labels). Then, for an unknown flower, which species it belongs to (test sample label) can be judged according to its petal length (test sample feature). In the figure below, it is definitely more appropriate to judge the unknown flower as Class B.

# In[ ]:


Image("../input/week-3-images/flowers.jpeg")


# In summary, **the "supervision" is reflected in the fact that the training set has a "label"**. Just like in the figure above, we give flowers of some known species, and for a flower of an unknown species it is possible to compare it according to a feature.

# ### Classification and Regression

# Through the above simple example, you should have gotten a certain impression of "supervised learning". Similar to the problem of identifying categories above, we generally refer to the classification problem of supervised learning. Classification is actually one of the most common problems, such as the species of animals, the species of plants, and the types of various items.

# In addition to the classification problem, there is another very important category in supervised learning, that is, the regression problem, which is what we need to learn this week. First, the regression problem is the same as the classification problem. The training data contains labels, which is also the characteristic feature of supervised learning. The difference is that the classification problem predicts a category and the regression problem predicts a continuous value.

# For example, below are some regression problems which are often resolved:<br>
# <br>
# - Stock price forecast<br>
# - House price forecast<br>
# - Flood water line<br>
# <br>
# For the problems listed above, the target we need to predict is not a category, but a real continuous value.

# <img width='600px' src="image/intro3.svg"></img>

# In[ ]:


Image("../input/week-3-images/flowers2.jpeg")


# Thus, here are some common solutions to the regression problems in machine learning:<br>
# <br>
# - Linear regression method<br>
# - Polynomial regression method<br>
# - Ridge regression and LASSO regression

# On the other side, the methods to solve the classification problems are:<br>
# <br>
# - Support Vector Machines<br>
# - K-nearest neighbor algorithm<br>
# - Decision tree algorithm<br>
# - Random forest algorithm<br>
# - Naive Bayesian algorithm<br>
# - ......

# The methods mentioned above will be learned in the next course. We will start with the algorithm principle, build a predictive model with Python and analyze predictive performance on the actual data set.

# ## Summary

# In this experiment, we reorganized the concept of machine learning and its subdivision categories, and learned about common machine learning algorithms. At the same time, the experiment introduced the supervision study, which will help us during the next two weeks of study. The knowledge points included in this experiment are:<br>
# <br>
# - Review of machine learning<br>
# - Algorithms of machine learning<br>
# - Introduction to supervised learning<br>
# - Classification and regression

# ---

# <div style="color: #999;font-size: 12px;font-style: italic;">**Congratulations! You've come to the end of the Introduction to Supervised Learning Lab.</div>
