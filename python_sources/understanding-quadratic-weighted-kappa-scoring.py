#!/usr/bin/env python
# coding: utf-8

# # Understanding the scoring metric: Cohen's kappa

# In this competition, the goal is to *predict* the scoring catogeries of assessments to match with the *actual* results. The scoring categories are as follows:
# 
#     0 : Never solved
#     1 : Solved in 3+ attempts
#     2 : Solved in 2 attempts
#     3 : Solved in 1 attempt
#     
# Clearly there is a sense that a higher category is a better score. This is encoded with a distance ('weight') metric between scores *actual* score $i$ and *predicted* score $j$: 
# 
# $$ w_{ij} = \frac{1}{(N-1)^2}(i-j)^2, $$
# 
# where $N$ is the total number of samples. This quadratic distance severely penalizes bad predictions:
# 
#     ________________
#     Distance  Weight
#     ----------------
#     0         0
#     1         1
#     2         4
#     3         9
#     -----------------
# 
# The scoring uses a Cohen's weighted kappa metric [(wikipedia)](https://en.wikipedia.org/wiki/Cohen%27s_kappa#Weighted_kappa), which compares the statistics of the actual and predicted results, aiming to give an idea of accuracy while taking into account agreements due to chance. (Eg, if you had knowledge of which category occured most frequently in the *actual*s, you could make a trivial model that always predicted this most common category. If this is by far the most popular category then you'll hit a very good accuracy!)
# 
# Define the number of occurances where the actual score was $i$ and the predicted score $j$ as $n_{ij}$. Then define a joint histogram matrix as 
# 
# $$O_{ij} = n_{ij},$$
# 
# together with an outer product of the *actual* and *predicted* histograms
# 
# $$E_{ij} = \left[\sum_k n_{ik} \right]\left[ \sum_k n_{kj} \right],$$
# 
# then the weighted kappa score is given by
# 
# $$\kappa = 1 - \frac{\sum_{ij} w_{ij} O_{ij}}{\sum_{ij} w_{ij} E_{ij}},$$
# 
# which can be rearranged to give
# 
# $$\kappa = 1 - \frac{\sum_{ij}~ (i-j)^2 n_{ij}}{\sum_{ij} (i-j)^2\left[\sum_k n_{ik}~ \right]\left[ \sum_k n_{kj} \right]}.$$

# ## Maximizing the score
# 
# $\kappa$ typically goes between -1 and +1, with +1 a perfect score, -1 a perfect anticorrelated score, and 0 evidence of no correlation between *actual*s and *predicted*s.
# 
# To maximize $\kappa$ we want to minimize the numerator and maximize the denominator. The numerator is increased by increasing accuracy of the model, with bigger discrepencies being penalized more (eg predicting 0 for an actual of 3 is 9x worse than predicting 2). The denominator measures how similar the *actual* and *predicted* histograms are, the idea being that it accounts for accuracy purely due to chance.

# In[ ]:




