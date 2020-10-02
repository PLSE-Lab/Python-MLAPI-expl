#!/usr/bin/env python
# coding: utf-8

# ## About
# 
# Every time a neural network finishes passing a batch through the network and generating prediction results, it must decide how to use the difference between the results it got and the values it knows to be true to adjust the weights on the nodes so that the network steps towards a solution. The algorithm that determines that step is known as the **optimization algorithm**.
# 
# This notebook reviews the optimization algorithms available in `keras`. The algorithms are described in the official documentation [here](https://keras.io/optimizers/). In this notebook I go into slightly more detail about how each of the various algorithms are constructed.

# In[ ]:


import keras


# ## SGD
# 
# SGD, or stochastic gradient descent, is the "classical" optimization algorithm. In SGD we compute the gradient of the network loss function with respect to each individual weight in the network. Each forward pass through the network results in a certain parameterized loss function, and we use each of the gradients we've created for each of the weights, multiplied by a certain learning rate, to move our weights in whatever direction its gradient is pointing.
# 
# SGD is the simplest algorithm both conceptually and in terms of its behavior. Given a small enough learning rate, SGD always simply follows the gradient on the cost surface. The new weights generated on each iteration will always be strictly better than old ones from the previous iteration.
# 
# SGD's simplicity makes it a good choice for shallow networks. However, it also means that SGD converges significantly more slowly than other, more advanced algorithms that are also available in `keras`. It is also less capable of escaping locally optimal traps in the cost surface (see the next section). Hence SGD is not used or recommended for use on deep networks.

# In[ ]:


keras.optimizers.SGD


# SGD implements a handful of different parameters.
# 
# ## SGD with [Nesterov] momentum
# 
# **Nesterov momentum** was one of the first innovations that improved optimization algorithm convergence speed.
# 
# Nesterov momentum is an example of a technique that uses **momentum**. Momentum techniques introduce information from past steps into the determination of the current step. In other words, descent in an algorithm using a momentum technique depends not just one the algorithm's current determination, but also on some of the steps it had taken in the recent past.
# 
# While SGD on its own strictly follows the cost surface, to the extent to which the learning rate allows for it, SGD with Nesterov momentum will "roll a ball" along the cost surface. If the algorithm made a lot of steps in a particular direction in the recent past, even if there is a sharp turn in the current gradient step, the algorithm will nevertheless continue somewhat to want to move in that direction, until there are enough accumulated signals to the contrary to make it stop. In this way momentum works just like it does in conventional physics.
# 
# Momentum has two advantages. One is that it helps address one of the major problems with straight SGD: local minima traps. SGD may get stuck at a local minima which is wide enough to push any gradient step back into itself. Momentum allows the learner to jump past the edge of and "escape" that local minima. The other advantage is that momentum techniques allow optimizers to learn more quickly, in particular by making it safer to pick large learning rates. Momentum techniques can overshoot a solution by a lot and that's OK, because like a ball rolling up a hill they will eventually go back the other way.
# 
# How do we apply momentum?
# 
# The simplest way of applying momentum is to, for each iteration of the learner, create a vector that is some kind of decaying average of past steps taken by the algorithm, sum that with the vector for the current gradient, and move in direction of the summed vector.
# 
# Nesterov momentum is a slight modification to this approach that works better in practice. Nesterov momentum takes a decayed average of past steps and steps in that direction *first*. Then we compute the gradient from that new position using our data, performing a "correction". We thus update the weights twice on each iteration: once using momentum and using our gradient algorithm.
# 
# It's not hard to see why Nesterov momentum is better in practice than simple momentum. It gets to use additional information: the gradient on the data at the uncorrected point.
# 
# A good article on this subject which provides some simple mathematical explanations and diagrams is [this one on TowardsDataScience](https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d).
# 
# SGD uses no momentum by default. You can configure it to use momentum or to use Nesterov momentum as follows:

# In[ ]:


keras.optimizers.SGD(momentum=0.01, nesterov=True)


# ## Adagrad
# 
# Adagrad is a more advanced machine learning technique (relative to SGD) which performs gradient descent with a variable learning rate. Node weights which have historically had large gradients are given large gradients, whilst node weights that have historically had small gradients are given small gradients.
# 
# Thus Adagrad is effectively SGD with a per-node **learning rate scheduler** built into the algorithm (for a learning rate scheduler in action see [this notebook](https://www.kaggle.com/residentmario/tuning-your-learning-rate)). Adagrad thus improves on SGD by giving weights historically accurate learning rates, instead of satisfying itself with a single learning rate for all nodes.
# 
# To better understand the properties of Adagrad we need to look at its equation (taken from [this post](http://ruder.io/optimizing-gradient-descent/)):
# 
# $$\theta_{t + 1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \cdot g_t$$
# 
# Where $\theta_t$ is the node weight for generation $t$; $G_t$ is a diagonal matrix containing the squares of all previous gradients; $\epsilon$ is a very small order of e^-8 regularization term that provides numerical stability by preventing division by 0; $g_t$ is the vector of gradients for the current generation; and $\eta$ is the learning rate.
# 
# The magic is in the diagonalized matrix in the fractional denominator, which performs the weight biasing that the algorithm is good for.
# 
# The disadvantage of Adagrad is also in that matrix, however. Every update we ever want to perform will be positive, and so that denominator matrix will always grow in magnitude. Given enough generations it will overwhelm the learning rate and tune actual learning down to nearly nothing, preventing further convergence. This is obviously catastrophic if your algorithm has not converged yet.
# 
# The literature generally recommends leaving the parameters for this optimizer at their default values, and this is a big advantage of `Adagrad`: the default learning rate value of 0.01 has been shown to work well in the vast majority of cases, due to the built in learning rate schedule, so you don't have to think about tuning that hyperparameter.
# 
# Adagrad has been used in e.g. Google's famous cat video recognizer.

# In[ ]:


keras.optimizers.adagrad


# ## Adadelta
# 
# Adadelta is an adaptation of Adagrad that uses momentum techniques to deal with the monotonically decreasing learning rate problem. Adadelta has each gradient update on each weight be a weighted sum of the current gradient and an exponentially decaying average of a limited number (a rolling window) of past gradient updates. Since the gradient denominator is no longer monotonically increasing, the learning rate is more stable, and the algorithm overall is more robust.
# 
# Technically speaking, in the original implementation of Adadelta there wasn't even a learning rate parameter that needed to be set. But `keras` uses a modified version of Adadelta with a learning rate defined for consistency with the rest of the `keras` optimization algorithms. 
# 
# For a more detailed mathematical reference on these two algorithms and pointers to other resources and the original papers see [this StackOverflow answer](https://datascience.stackexchange.com/questions/27676/understanding-the-mathematics-of-adagrad-and-adadelta).

# In[ ]:


keras.optimizers.adadelta


# ## RMSprop
# 
# RMSprop is a correction to Adagrad that was proposed independently of Adadelta but developed at around the same time. It was suggested by Geoffy Hinton in his Coursea course (the classic one that I watched, incidentally). It's equivalent to Adadelta, with one difference: the learning rate is further divided by an exponentially decaying average of all squared gradients, e.g. a global tuning value.
# 
# As with the other adaptive learning rate optimization algorithms the recommendation is to leave the aglorithm hyperparameters at their default settings.

# In[ ]:


keras.optimizers.rmsprop


# ## Adam
# 
# Adam stands for Adaptive Moment Estimation. In addition to storing an exponentially decaying average of past squared gradients like Adadelta and RMSprop, Adam also keeps an exponentially decaying average of past gradients, similar to momentum.
# 
# Mathematically, Adam works thusly. We start by computing estimates of the mean and variance (first and second moments) of the gradient, using the following formula:
# 
# $$m_t = \beta_1 m_{t - 1} + (1 - \beta_1)g_t$$
# $$v_t = \beta_1 v_{t - 1} + (1 - \beta_2)g_t^2$$
# 
# Where $v_t$ is an exponentially decaying average of past squared gradients and $m_t$ is an exponentially decaying average of past gradients. Notice that these are parts of the formulas for RMSprop/Adagrad/Adadelta and Momentum, respectively. $\beta_1$ and $\beta_2$ are the decay rates, which control the relative contribution of past history versus the present gradient. `B_1 = 0.9` and `B_2 = 0.999` by default in `keras`; these values are usually very large, e.g. very heavily biased towards the past as opposed to the present.
# 
# The problem with using these values in an update rule is that they are biased towards 0, since $m$ and $v$ are initialized as zero vectors on the first algorithm run. Adam therefore introduces a further bias correction to its formula:
# 
# $$\hat{m}_t = \frac{m_t}{1 - \beta_1}$$
# $$\hat{v}_t = \frac{v_t}{1 - \beta_2}$$
# 
# We plug these corrected, momentum-using values into the same adaptive learning rate formula used by RMSprop and family:
# 
# $$\theta_{t + 1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
# 
# Thus Adam is basically RMSprop plus momentum. The pathfinding of the Adam algorithm is essentially that of a ball with both momentum and friction. Adam biases the path the agorithm takes towards flat minima in the error surface, slowing down learning when moving along a large gradient.
# 
# Adam is one of the most popular optimization algorithms at present, largely because it provides both the smart learning rate annealing and momentum behaviors of the algorithms we've seen here thus far.

# In[ ]:


keras.optimizers.adam


# ## AdaMax
# 
# Adam, like RMSprop et. al., uses an exponentially decaying weighted average estimate of the variance of the gradient in its formulation. However, strictly speaking, theres nothing that says we have to use the variance.
# 
# The variance is equivalent to the second moment or L2 norm of the gradient. The $L_n$ norm is defined as:
# 
# $$L_1 = g$$
# $$L_2 = \sqrt{g^2}$$
# $$L_3 = \sqrt[3]{g^3}$$
# $$L_n = \sqrt[n]{g ^n}$$
# 
# To learn more about norms refer to [this notebook on L1 versus L2 norms](https://www.kaggle.com/residentmario/l1-norms-versus-l2-norms).
# 
# Norm values larger than 2 aren't very useful because they are not to be numerically stable. But what about...the infinity norm?!?
# 
# $$L_\infty = \sqrt[\infty]{g^\infty}$$
# 
# The infinite norm is numerically stable because it has asymptotically convergent behavior (assuming $g \in [0, 1]$). For reasons unclear to me $v_t$ simplifies to:
# 
# $$v_t = \max{(\beta_2 \cdot v_{t - 1}, |g_t|)}$$
# 
# AdaMax is more robust to gradient update noise than Adam is, and has better numerical stability. Precisely why that is bears further investigation.

# In[ ]:


keras.optimizers.adamax


# ## Nadam
# 
# Nadam is Adam, but with Nesterov momentum instead of ordinary momentum. The advantage of using Nesterov momentum instead of regular momentum is the same as it is in the SGD case.

# In[ ]:


keras.optimizers.nadam


# ## AMSgrad
# 
# AMSgrad is a recent proposed improvement to Adam. It has been observed that for certain datasets, Adam fails to converge to the globally optimal solution, whereas simpler algorithms like SGD do. A paper written just last year proposed that the exponential weights on the terms in the algorithm are the problem.
# 
# The hypothesis is that for certain datasets (e.g. image recognition?), there are a lot of small, less informative gradients punctuated by occassional large, more informative gradients. Adam has an inbuilt tendency to deprioritize the more informative gradients because those gradients are quickly "swallowed" by the exponentially weighting, causing the algorithm to steer past the point of optimality without sufficiently exploring it.
# 
# AMSgrad proposes to solve this issue by computing the current $\hat{v}_t$, but then setting $v_t = \max{(\hat{v}_t, v_{t - 1})}$. In other words, if the exponentially weighted L2 norm for the current gradient position is smaller than the same for the previous gradient position, the previous and larger result is retained. In this way the influence of large gradients is retained, as they "stick around" in more backpropogation passes due to their "stickiness".
# 
# This algorithm may perform better on certain datasets in practice, but has not displaced Adam, due to a lack of verifiability in terms of win on general-purpose datasets.

# In[ ]:


keras.optimizers.adam(amsgrad=True)

