#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Learned from: https://github.com/nicku33/demo/blob/master/Contextual%20Bandit%20Synthetic%20Data%20using%20LinUCB.ipynb


# # Solving Contextual Bandit problems using LinUCB
# A contextual bandit problem is a subset of the full reinforcement learning problem. There is context vector $X_i$ attached to each choice, but no state or state transitions that can be predicted. It's assumed that the contexts are IID.
# 
# The tradeoff between exploration and exploitation is central to bandit problems. The simplest algorithm to decide whether to try out something new or go with what you yeilds the best reward is known as greedy-eepsilon algorithm, where a certain percentage is devoted to exploration.
# 
# However, an improvement in efficient can be gained by using an estimation of variance to quantify how much we do not know about a choice. In contextual problems we have variance along each dimension of D, for each arm.
# 
# # Upper Confidence Bound methods
# For each action we may maintain an expected value, which we assume to be normally distributed, therefore we can also estimate a confidence interval based on teh data we have.
# 
# By having our algorithm choose the higherst of all the actions' confidence intervals (say 0.95), we create a smooth transition between learning and exploiting. This is the essence of a UCB method. It's easy to demonstreate in one dimension with 2 arms.

# In[ ]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
A = np.array([[ 0.70558142,  0.86273594],
       [ 0.68797076,  0.02774431],
       [ 0.11372903,  0.38687036],
       [ 0.43072205,  0.54141448],
       [ 0.39529217,  0.91495635],
       [ 0.39027663,  0.95704016],
       [ 0.5450535,  0.02473527],
       [ 0.12265648,  0.88966732],
       [ 0.59852203,  0.28636077],
       [ 0.85040799,  0.04557076],
       [ 0.56896747,  0.89559782],
       [ 0.85425338,  0.01838715],
       [ 0.80782139,  0.17350079]]);
fg = plt.boxplot(A)


# This boxplot is just a quick visual proxy, but let's calculate mean and s.d. of each arm. However, if we take the mean + 3 standard deviations up as the number which helps us select which arm to pull, we combine expected values and uncedrtainty in one metric.

# In[ ]:


a_mu = np.mean(A, axis=0)
a_sig = np.std(A, axis=0)
a_upper_ci = a_mu + (3*a_sig)
a_upper_ci


# In this case, we choose arm 2. After a while, our mean will settle down and the deviation will shrink as $1/\sqrt{n_a}$ and we may eventually choose 1

# # Adding the context Vector
# When we add in context, we expand the problem space significantly. Naively, we might treat all unique combinations of context as separate problems. However, it's unlikely we'll ever get enough data for that. In stead, we assume that the expected value of an arm is some linear combination of the context $x_i$. In this sense, we are performing an online regression for each possible action, conditional on the context. In this case, we only get to see results for the chosen action, so we can only improve one estimate at a time. We are trying to estimate $\theta_a$ for each action $a$, and the expected value = $x_{i}^{T}\theta_a$.
# 
# This is the expected value of an action $a$ from time step $0$ up to time step $T$, associated with a context $x_{i}$ at the time of recommendation ($t$ or $T$?).
# 

# # Creating the synthetic data
# Let's create some idealized synthetic data, and then some random $\theta_a$. Assume 30 features and 8 possible actions.

# In[ ]:


n = 5000 # number of data points
k = 30 # number of features
n_a = 8 # number of actions
D = np.random.random((n, k)) - 0.5 # our data, or these are the contexts, there are n contexts, each has k features
th = np.random.random((n_a, k)) - 0.5 # our real theta, what we will try to guess (there are 8 arms, and each has 30 features)


# Each action yeilds a reward. In this case, a success probability. We get maximium rewa4d when we always choose the best action.
# Since tehre's no state, we can precompute the best actions ahead of time. Here is the distribution of optimal arms

# In[ ]:


P = D.dot(th.T)
optimal = np.array(P.argmax(axis=1), dtype=int)
plt.title("Distribution of ideal arm choices")
fig = plt.hist(optimal, bins=range(0, n_a))


# In[ ]:


P


# # Solve this with UCB
# Let's solve this with the LinUCB algorithm.
# LinUCB uses the same highest C.I. bound principle as above, but in a multidimensional setting. We maintain, for each arm, a running mean and running covariance matrix. For each new data point, we pass the context vector $x_i$ through the covariance matrix to come up with an estimate of how much value the new information is.
# 
# To simplify let's assume that all our feature vectors are independent, and thus we have covariance matrix which is close to diagonal. Passing $x_i$ through it $x_i\sum^{-1}x_i$ now just becomes $\sum_{d \in D}(x_d\sigma_d)^2$. I.e., it's a sum of the squared variances, weighted by the magnitude of each component in our new data point.
# 
# \begin{equation*}
#     p[a] = a\_mean + a\_upper\_ci
# \end{equation*}
# 
# If the new data point has a strong signal along a dimension we know little about, the aggregate score will be high and we will be more likely to explore.
# 
# Using a tuning parameter $\alpha$, we can adjust our explore/exploit ratio, just like greedy-epsilon.

# # The LinUCB implemenation

# In[ ]:


import pdb
def set_break():
    pdb.set_trace()


# In[ ]:


eps = 0.2
choices = np.zeros(n, dtype=int)
rewards = np.zeros(n)
explore=np.zeros(n)
norms = np.zeros(n)
b = np.zeros_like(th)
A = np.zeros((n_a, k, k))
for a in range(0, n_a):
    A[a] = np.identity(k)
th_hat = np.zeros_like(th) # our temporary feature vectors, our best current guesses
p = np.zeros(n_a)
alph = 0.2

# LINUCB, usign disjoint model
# This is all from Algorithm 1, p 664, "A contextual bandit appraoch..." Li, Langford
for i in range(0, n):
    x_i = D[i] # the current context vector
    for a in range(0, n_a):
        A_inv = np.linalg.inv(A[a]) # we use it twice so cache it.
        th_hat[a] = A_inv.dot(b[a]) # Line 5
        ta = x_i.dot(A_inv).dot(x_i) # how informative is this?
        a_upper_ci = alph * np.sqrt(ta) # upper part of variance interval

        a_mean = th_hat[a].dot(x_i) # current estimate of mean
        p[a] = a_mean + a_upper_ci
    norms[i] = np.linalg.norm(th_hat - th, 'fro') # diagnostic, are we converging?
    #Let's hnot be biased with tiebraks, but add in some random noise
    p = p + (np.random.random(len(p))*0.000001)
    choices[i] = p.argmax() # choose the highest, line 11
    
    # See what kind of result we get
    rewards[i] = th[choices[i]].dot(x_i) # using actual theta to figure out reward
    
    #update the input vector
    A[choices[i]] += np.outer(x_i, x_i)
    b[choices[i]] += rewards[i]*x_i


# In[ ]:


plt.figure(1, figsize=(10, 5))
plt.subplot(121)
plt.plot(norms)
plt.title("Frobeninus norm of estimated theta vs actual")

regret = (P.max(axis=1) - rewards)
plt.subplot(122)
plt.plot(regret.cumsum())
plt.title("Cumulative regret")


# # Analysis
# The two plots above show how we converge to the true linear model parameters, and how much it cost us along the way.
# 

# # Hybrid Models
# It is a good bet that there will be features that are common to all arms, which we need to know how to interpret. For example, if we classify articles into course categories, then information about one article should transfer to another.
# To take these into account, we can addanother term and another set of coefficients $\beta^T z_i$, which we update regardless of what arm we pick.

# # Taking this to production
# ### Can we have multiple boxes serve different requests, so we have load balancing and decentralization?
# Yes. Sicne the A and b variables above are only added to, we can read off the logs of each box into cumulative A and b in shared memory or offline in batch, then distribute them back periodically. The loss of accuracy as a result just leads to some extra exploration vs the single box but this only aids convergence. Upon gettin ght esum for all boxes each box would have a recent sum of pooled information for all models. Optionally each box can add to it's own A and b until the next update.
# 
# Since the model is additive, we are also free to use anything that supported atomic increments ,which are primitives in Java and some data stores.
# 
# ### Can we train up the models off line and avoid cold start?
# I think so. In the algorithm above we choose our initial means as (0, 0, ..., 0). We can get a closer estimate by just running a regression over previous data. Even if it's just similar data (different articles and users, but there will be some features such as article topic that carry over) it's better than startin gfrom 0. It will be very biased toward whaever chosen articles before, but I'll take a biased initial estimate over a random as long as there's no chance of local minima, which our linear model guarantees.
# 
# Also, in the paper they made the point that if hyou run a test with random bandit choices, you can use this to score your own algorithm offline, probabiy useful for feature selection, since the rewards will also be randomly distributed. I put in a request with Yahoo Labs to get the original data and see.
# 
# ### What to use?
# I haven't used it but this seems like Storm's use case in nutshell. However, we could even just publish matrices periodically off of a batch mapreduce or Spark job. It depends on how often articles change. Storm would be more responsive, clearly.
