#!/usr/bin/env python
# coding: utf-8

# # How to learn Data Science? Use Kaggle

# ### 1. Interact with similar-minded people: Community
# Kaggle is surely one of the best community for novice and expert Data Scientist around the world.
# Undeniably, you can learn *Bayesian Inference* from this amazing [GitHub blog](https://camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/), but after a few hours of learning, you might lose track. What if you can ask related questions to someone who knows Bayesian thinking? Just after typing *Bayesian*, Kaggle gives me the following, so **38 users** might know Bayesian, & I might ask them questions. Or, post Bayesian related questions in Kaggle with a hope that someone will answer me.

# ![image.png](attachment:image.png)

# ### 2. See your improvement: [Kaggle progression](https://www.kaggle.com/progression)
# I am also kinda new in Kaggle, but I'm loving it & thinking to invest more time on Kaggle. When you run more of your Data Science Python or R program on Kaggle, and get upvoted, your account will be progressing from Kaggle novice to contributor -> Expert -> Master -> Grandmaster.
# It's a great feeling to be acknowledged from my learning. Let's grow together. :)

# ### 3. Run models in the same platform 
# While writing this article, I am writing a small Python program (on the same webpage) to explore the possibility of survivals of women compared to men in Titanic disaster. See below: **female** has 74% chances of survival whereas **male** has only 18%. This does make sense since rescue operations prioratize children and women as in Titanic dataset.

# In[ ]:


import pandas, seaborn
data = pandas.read_csv('../input/titanic/train.csv') #data
_ = seaborn.countplot(x='Sex', hue='Survived', data=data)
data.groupby('Sex')['Survived'].mean()


# ### 4. Share 
# Of course, you will run your Data Science project in Kaggle, & share it with us. When you run your code in local machine, nobody sees your outputs!

# ### 5. No installation: Python & R
# **Python & R** are the two MOST dominating programming language in Data Science. I personally dont care about C or whatsoever programming language or your skill on those. Why? Because if I need to learn only one language for my Data Science journey, its Python (or if my need is more statistical toolbox focused, then R). 
# Kaggle is the platform that comes with Python & R preinstalled. Salute to Kaggle.
# 
# Note: I am an enthusiastic user of Kaggle and not affiliated with it by any means.
