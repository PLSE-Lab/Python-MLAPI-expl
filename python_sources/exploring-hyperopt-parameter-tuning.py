#!/usr/bin/env python
# coding: utf-8

# # Exploring Hyperopt parameter tuning
# 
# Hyperparameter tuning can be a bit of a drag. [GridSearch](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) and [RandomSearch](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) are two basic approaches for automating some aspects of it. GridSearch is quite thorought but on the other hand rigid and slow. Rigid in only exploring specific set of exact parameter values, and slow in trying every combination, which quickly becomes a large set. RandomSearch explores variants faster but not very systematically, and also focuses on a similar strict specification of search-space.
# 
# Another approach is used by optimization frameworks/libraries such as [Hyperopt](http://hyperopt.github.io/hyperopt/). To me, it seems to combine some of the best of both, with random initializations leading to more guided search towards the promising areas. It also uses a more relaxed definition of the search-space in form of distributions vs exact values.
# 
# I wanted to try hyperopt and see how I could make best use of it. The hyperopt website provides a good starting point, and searching on the internets I did find a number of reasonable examples of how people had it set up. But just slapping these parameters and search space configurations around blindly did not make me happy. I would prefer to understand a bit about what I am doing, and what those configurations mean.
# 
# This is what I attempt to explore here. Exploring the search-space of the hyperopt search-space. The different distributions (normal, lognormal, uniform, loguniform, qlognormal, ...) and what they mean, how to configure them. I also take a very brief look at what the algorithm applied by Hyperopt (Tree Parzen Estimator) might be doing with these search space definitions. 
# 
# This has helped me understand the approach better, learn about some distributions, and figure myself a process on how to apply this for model tuning. Of course, I might have missed something, and some deepest details slightly still eluded me. So please do correct or add any useful info in comments or otherwise.
# 
# PS if you get tired of too much jargon and silly exploration of them numbers and distributions, I try to summarize what I learned in the end. Some day I will also add an example there too. Of course I will.
# 

# In[ ]:


import numpy as np
import pandas as pd
import random
import seaborn as sns
import hyperopt
from hyperopt import hp

import os
#currently no input as just exploring the functions with generated data. maybe later I will add an example with some dataset
print(os.listdir("../input"))


# ## Distributions in general

# To use hyperopt, we must describe the search space for it using value distributions for the parameters we want it to explore. These are variants of normal and uniform distributions, and their logarithmic variants. First, a look at those distributions to understand how to use them with hyperopt later.
# 
# ### Normal distribution
# 
# Start with the [standard normal distribution](https://en.wikipedia.org/wiki/Normal_distribution), and how to generate data from such a distribution using Python (sampling the search-space in hyperparameter optimization terminology). The standard normal distribution has a mean of 0 and standard deviation of 1 like so:

# In[ ]:


mean = 0
std=1
#values = []
#for x in range(1000):
#    values.append(random.gauss(mean, std))
#we could do the three rows above to generate the normal (gaussian) distribution or just use numpy:
values = np.random.normal(mean, std, 1000)
df = pd.DataFrame(values, columns=["var1"])
df.hist(bins=100)


# In[ ]:


df.shape


# The above is a histogram with 100 bins for the values. This shows the general shape of a normal distribution as one might expect. A bit edgy, but we can convert it to a [kernel density plot](https://towardsdatascience.com/histograms-and-density-plots-in-python-f6bda88f5ac0) to smoothen it out:

# In[ ]:


import seaborn as sns
sns.set_style('whitegrid')
sns.kdeplot(df["var1"].values, bw=0.5)


# The kernel density plot represents more of a [probability distribution](https://towardsdatascience.com/histograms-and-density-plots-in-python-f6bda88f5ac0) vs the absolute values in the histogram. The above is for the standard normal distribution. So if we use this type of distribution to represent the search-space for hyperopt, it would focus more on the peak, and progressively less on the sides.
# 
# ### Log-normal distribution
# 
# What is a log-normal distribution then? Try to visualize one by generating data in it:

# In[ ]:


mean = 0
std = 1
count = 10000
values = np.random.lognormal(mean, std, count)
df = pd.DataFrame(values, columns=["var1"])
df.hist(bins=100)


# And visualize it as a KDE in a similar way as above for normal distribution:

# In[ ]:


sns.set_style('whitegrid')
sns.kdeplot(df["var1"].values, bw=0.5)


# So it spikes on the left and has a long tail on the right. Seems useful to describe search-spaces where we want to put more focus on the lower part of the distribution. Such as learning rates, which quite often end up at a lower end.
# 
# This is the practical view. But what does log normal distribution mean generally? [Wikipedia](https://en.wikipedia.org/wiki/Log-normal_distribution) says that if a variable X is log-normally distributed, ln(X) should be normally distributed. Let's see:

# In[ ]:


df["var2"] = np.log(df["var1"]) #in numpy log() is same as ln()
sns.kdeplot(df["var2"].values, bw=0.5)


# Well, that certainly shows how the log(x)=normally distributed assumption holds true. I am pretending the slightly longer tail on the left does not exist to simplify things.
# 
# Before I figured this out, I also wanted to see if log normal would mean taking the log of normal distribution, but no. It just moves the distribution center and scales the width:

# In[ ]:


mean = 100.
std = 2.
count = 10000
values = np.log(np.random.normal(mean, std, count))
df = pd.DataFrame(values, columns=["var1"])
df.hist(bins=100)


# ### Uniform distribution
# 
# What about a uniform distribution? This is described in [numpy docs](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.uniform.html) as "any value within the given interval is equally likely to be drawn".
# 
# Lets see:

# In[ ]:


low = 0
high = 10
values = []
for x in range(10000):
    values.append(np.random.uniform(low, high))
df = pd.DataFrame(values, columns=["var1"])
df.hist(bins=100)


# As you might expect, "uniform" picks values uniformly across the range :). With sufficiently large number of samples, I expect it would become very close to a straight line on the top.

# ## Distributions in Hyperopt
# 
# Above is all nice to understand a bit about the distributions in general. What about hyperopt? What do the distributions it can use to explore the search space look like? The distribution functions [listed](https://github.com/hyperopt/hyperopt/wiki/FMin) are:
# 
# - hp.choice(label, options)
# - hp.randint(label, upper)
# - hp.uniform(label, low, high)
# - hp.quniform(label, low, high, q)
# - hp.loguniform(label, low, high)
# - hp.qloguniform(label, low, high, q)
# - hp.normal(label, mu, sigma)
# - hp.qnormal(label, mu, sigma, q)
# - hp.lognormal(label, mu, sigma)
# - hp.qlognormal(label, mu, sigma, q)
# 
# ### Uniform distribution
# 
# First, the uniform distribution, described as [returns a value uniformly between low and high](https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions):

# In[ ]:


low = 0
high = 100
#this is how we define the possible space for hyperopt to explore the variable
space = hp.uniform('something', low ,high)
#and this is how we ask it to provide a single sample from that space (to "sample" the search space)
print (hyperopt.pyll.stochastic.sample(space))


# look, we got a value in the range given (0-100)! Ooh. Now lets try a larger number of samples to see if it matches the example of uniform distribution I made with Numpy above:

# In[ ]:


low = 0
high = 100
values = []
space = hp.uniform('something', low, high)
for x in range(10000):
    values.append(hyperopt.pyll.stochastic.sample(space))
df = pd.DataFrame(values, columns=["var1"])
df.hist(bins=100)


# It does. And the KDE for it:

# In[ ]:


sns.kdeplot(df["var1"].values, bw=2.)


# The KDE looks a bit like Hill Climb Racing. Anyway, I expected uniform() to sample the value space uniformly for the given range. Seems to match. I would suggest to use this if you want uniform (continuous) values across a range (surprise!).
# 
# ### Normal distribution
# 
# Now, a look at normal distribution in hyperopt. This is described as [normally-distributed with mean mu and standard deviation sigma](https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions):

# In[ ]:


mean = 0 #it is the mu
std = 1  #it is the sigma
values = []
space = hp.normal('something', mean, std)
for x in range(1000):
    values.append(hyperopt.pyll.stochastic.sample(space))
df = pd.DataFrame(values, columns=["var1"])
df.hist(bins=100)


# In[ ]:


sns.kdeplot(df["var1"].values, bw=0.5)


# Yes, it generates values that are in line with normal distribution I explored higher above with Numpy.
# 
# ### Log-normal distribution
# 
# And lognormal, described in hyperopt as [a value drawn according to exp(normal(mu, sigma)) so that the logarithm of the return value is normally distributed](https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions):

# In[ ]:


mean = 0
std = 1
values = []
space = hp.lognormal('something', mean, std)
for x in range(1000):
    values.append(hyperopt.pyll.stochastic.sample(space))
df = pd.DataFrame(values, columns=["var1"])
df.hist(bins=100)


# In[ ]:


sns.kdeplot(df["var1"].values, bw=0.5)


# It is very similar to the lognormal I explored far above. Again, I think this type of distribution is great if you want to generate more values on the lower end of the search space. For example, focus hyperparameter search for learning rate on the smaller values, while not ruling out trying a few larger values.
# 
# For a bit more detail, lets look at how it does at the very bottom of the value space:

# In[ ]:


df[df["var1"] < 1].hist(bins=100)


# The height of the bars is different from the above plot of the full graph due to different scale but same bin count. But it does fit the general shape, and gives an idea of how the biggest spike(s) are distributed from the overall generation.

# ### Log-uniform distribution
# 
# Now hyperopt loguniform(), described as [a value drawn according to exp(uniform(low, high)) so that the logarithm of the return value is uniformly distributed](https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions):

# In[ ]:


values = []
space = hp.loguniform('something',0,1)
for x in range(1000):
    values.append(hyperopt.pyll.stochastic.sample(space))
df = pd.DataFrame(values, columns=["var1"])
df.hist(bins=100)


# Or in a different range:

# In[ ]:


values = []
space = hp.loguniform('something',0,5)
for x in range(1000):
    values.append(hyperopt.pyll.stochastic.sample(space))
df = pd.DataFrame(values, columns=["var1"])
df.hist(bins=100)


# In[ ]:


min(df["var1"]) 


# The smallest generated value is close to 1 because exp(0) = 1, and 0 is the min above. That's what I think.

# In[ ]:


sns.kdeplot(df["var1"].values, bw=0.5)


# Better smoothing would require a large bandwidth for the gaussian kernels (see, I sounded smart!).
# 
# Lets apply a log() conversion on the generated distribution from above. This should produce the standard uniform distribution shape:

# In[ ]:


df["var2"] = np.log(df["var1"])
df["var2"].hist(bins=100)
#sns.kdeplot(df["var2"].values, bw=0.5)


# It resembles a uniform distribution as one might expect, would probably benefit from more samples to even it out more. 
# 
# And the density plot:

# In[ ]:


sns.kdeplot(df["var2"].values, bw=0.5)


# Looks like a tooth. But it is actually the uniform distribution smoothed out by KDE. 
# 
# The plots above also illustrate where the actual parameters of 0 and 5 come from. Because the generated values are not in range of 0 to 5 but rather their log is. This is how the distribution is defined after all.
# 
# But I often want to generate values in a specific range, and not reason in my head about how this would effect the exp or log of the values and variables. So, how can we make the generated values in range 0 to 5? To understand, let's start with understanding what is a logarithm:

# In[ ]:


#2 to the power of 3 equals 8 (2*2*2):
np.power(2,3)


# Now, if we take the value 8 and its base 2 logarithm, this should tell use what power should 2 be raised to, to get 8:

# In[ ]:


np.log2(8)


# To recap, hp.loguniform() [returns a value drawn according to exp(uniform(low, high))](https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions). The [exp()](https://docs.scipy.org/doc/numpy/reference/generated/numpy.exp.html) and [log()](https://docs.scipy.org/doc/numpy/reference/generated/numpy.log.html) functions deal with the natural logarithm, which uses the *e* value as its base.
# 
# Raising this *e* to the power of 1 itself should give the value of *e*:

# In[ ]:


np.exp(1)


# If you look that up, it is correct.
# 
# The natural logarithm of *e* is 1:

# In[ ]:


np.log(np.exp(1))


# To get value 1 from exp(x), we can pass the log(1) as x:

# In[ ]:


np.exp(np.log(1))


# Lets try this for a few more values (some rounding issues if not exact):

# In[ ]:


for x in range(1, 10):
    print(np.exp(np.log(x)))


# Since hyperopt uses exp(...) to build its loguniform disributions, lets see what an exp() distribution looks like:

# In[ ]:


values = []
for x in np.arange(0, 10, 0.1):
    values.append(np.exp(x))
df = pd.DataFrame(values, columns=["var1"])
df.hist(bins=100)


# In[ ]:


min(df["var1"])


# The figure illustrates how this is e^x (e to the power of x), starting with smaller values repeating more and the number growing bigger with more multipliers (higher power).
# 
# So the way to limit the loguniform distribution from hyperopt to a given range, is to use minimum and maximum values that give the desired maximum value when put into formula e^x. What is this value then? It is log(x). So to get values in range from 0.1 to 10 from loguniform(), we need min=log(0.1) and max=log(10):

# In[ ]:


values = []
for x in np.linspace(np.log(0.1), np.log(10), 100):
    values.append(np.exp(x))
df = pd.DataFrame(values, columns=["var1"])
df.hist(bins=100)


# Compare this to hyperopt loguniform sampling using the same limits:

# In[ ]:


values = []
space = hp.loguniform('something',np.log(0.1),np.log(10))
for x in range(100):
    values.append(hyperopt.pyll.stochastic.sample(space))
df = pd.DataFrame(values, columns=["var1"])
df.hist(bins=100)


# This is very similar, it is basically the same distribution. The difference comes from my np.exp() version above looping all the values from 0.1 to 10 using exact spacing of 0.1. Hyperopt is randomly sampling the space, so the choice of values is more random. But the overall shape matches as expected.

# When I was looking for insights on how people use hyperopt to optimize hyperparameters, I searched for examples. Some of these used negative values as parameters for the loguniform and lognormal distributions. This confused me because I thought the distributions would be based on log(x). Because its says logXXX in the distribution name. However, as I found, this cannot be true, as a negative x for log(x) gives an error (or a complex number):

# In[ ]:


np.log(-1)


# But since the name does not really mean the values are generated using log(x) but rather exp(x), this is different. A [negative exponent](https://www.mathsisfun.com/algebra/negative-exponents.html) just makes the value the smaller the bigger the negative exponent:

# In[ ]:


np.exp(-1)


# In[ ]:


np.exp(-2)


# In[ ]:


np.exp(-5)


# And this is how it looks when used from hyperopt:

# In[ ]:


values = []
space = hp.loguniform('learning_rate', -5.0, -0.7)
for x in range(1000):
    values.append(hyperopt.pyll.stochastic.sample(space))
df = pd.DataFrame(values, columns=["var1"])
df.hist(bins=100)


# In[ ]:


sns.kdeplot(df["var1"].values, bw=0.05)


# A look at the range it generates:

# In[ ]:


min(df["var1"])


# In[ ]:


max(df["var1"])


# So the above show using -5 to -0.7 as arange, resulting in values in that range of exp(-5) to exp(-0.7). These are approximately what is shown above with the min/max of generated (you can also check by running np.exp(-5) and np.exp(-0.7). 
# 
# This helps me understand those strange examples of tuning parameters I found on the internets with negative log-distribution parameters. However, I find it much more intuitive to just specify the actual value range so I see what I am getting. Something close to these limits would be 0.005 to 0.5. We can do this like so:

# In[ ]:


values = []
space = hp.loguniform('learning_rate', np.log(0.005), np.log(0.5))
for x in range(1000):
    values.append(hyperopt.pyll.stochastic.sample(space))
df = pd.DataFrame(values, columns=["var1"])
df.hist(bins=100)


# So to generate values in a specific range, just use np.log(x) as min/max bounds definitions for the hyperopt hp.logXXX functions.

# ## QLogUniform and other "quantized" distributions

# Besides the distributions above, hyperopt also includes various qXXXX versions of those distributions, called "quantized" distributions:
# 
# - quniform
# - qloguniform
# - qnormal
# - qlognormal
# 
# What does that mean?
# 
# Lets generate some such distributions and see what we get:

# In[ ]:


low = np.log( 10 )
high = np.log( 1000 )
q = 1
values = []
#with np.log(x) can set starting value at x
space = hp.qloguniform('learning_rate', low, high, q)
for x in range(1000):
    values.append(hyperopt.pyll.stochastic.sample(space))
df = pd.DataFrame(values, columns=["var1"])
df.hist(bins=100)


# The above looks similar to the loguniform distributions I played with above. This makes sense, since the hyperopt [description](https://github.com/hyperopt/hyperopt/wiki/FMin) for it gives a formula of "round(exp(uniform(low, high)) / q) * q". So divide and multiply by q=1 as above has no effect.
# 
# Try to explore a bit more by setting q=3 instead:

# In[ ]:


low = 0
high = 10
q = 3
values = []
for x in range(1000):
    r = round(np.random.uniform(low, high) / q) * q
    print(r, end=" ")
    values.append(r)
df = pd.DataFrame(values, columns=["var1"])
df.hist(bins=100)


# So it looks like the "quantized" version means it will produce values in intervals the size of *q*. It seems because of the rounding approach in the formula, the ends of the distribution get fewer "hits" (the bars are shorter). Is this just an artefact of how I created it using numpy above, or also the same for hyperopt? Lets see:
# 

# In[ ]:


values = []
#with np.log(x) can set starting value at x
space = hp.quniform('learning_rate', 0, 10, 3)
for x in range(1000):
    values.append(hyperopt.pyll.stochastic.sample(space))
df = pd.DataFrame(values, columns=["var1"])
df.hist(bins=100)


# Not just my approach, It is very much the same here as well. So the quantized version may not be quite so "uniform" in this type of case, but anyway. I guess it is good to remember this when using the quantized distributions.
# 
# Try with a q = 5 and a log uniform distribution from 10 to 100:

# In[ ]:


values = []
#with np.log(x) can set starting value at x
space = hp.qloguniform('learning_rate', np.log( 10 ), np.log( 100 ), 5)
for x in range(1000):
    values.append(hyperopt.pyll.stochastic.sample(space))
df = pd.DataFrame(values, columns=["var1"])
df.hist(bins=100)


# Matches the expectations from above.
# 
# Try with a fractional *q*:

# In[ ]:


values = []
space = hp.qlognormal('learning_rate', 1, 1, 0.1)
for x in range(1000):
    values.append(hyperopt.pyll.stochastic.sample(space))
df = pd.DataFrame(values, columns=["var1"])
df.hist(bins=100)


# It works, although the tail is quite long. Exploring it a bit:

# In[ ]:


df.min()


# In[ ]:


df.max()


# In[ ]:


df["var1"].clip(0, 30).hist(bins=100)


# In[ ]:


df[df["var1"] < 30].hist(bins=100)


# And then the same fractional test for log-uniform:

# In[ ]:


values = []
#with np.log(x) can set starting value at x
space = hp.qloguniform('learning_rate', np.log( 1 ), np.log( 2 ), 0.1)
for x in range(1000):
    values.append(hyperopt.pyll.stochastic.sample(space))
df = pd.DataFrame(values, columns=["var1"])
df.hist(bins=100)


# Fractionals work just fine.

# ## Moving and inverting the value ranges

# OK, I guess I now have a decent understanding of the distributions and how to use them with hyperopt. But what if I want some distributions that are not directly what it supports? Like focus more on bigger values and not smaller. Lets see:

# In[ ]:


values = []
space = hp.qloguniform('learning_rate', np.log( 10 ), np.log( 1000 ), 1)
for x in range(1000):
    #np.log(1000) above should give a max value generated of 1000, so use it here:
    values.append(1000-hyperopt.pyll.stochastic.sample(space))
df = pd.DataFrame(values, columns=["var1"])
df.hist(bins=100)


# That worked. So more complex equations using the distributions can also be provided to hyperopt.
# 
# But what happens if I give the low end as the bigger value? Nothing, because hyperopt seems to fix the wrong parameter ordering itself:

# In[ ]:


values = []
space = hp.qloguniform('learning_rate', np.log( 1000 ), np.log( 10 ), 1)
for x in range(1000):
    values.append(hyperopt.pyll.stochastic.sample(space))
df = pd.DataFrame(values, columns=["var1"])
df.hist(bins=100)


# ## hp.choice()

# One interesting generator left uncovered still is hp.choice(). This is simply a random choice between elements. I tried and found it can also be used to combine different distributions:

# In[ ]:


space = hp.choice('choice over something', [hp.normal('learning_rate', 0, 1),
                                            20+hp.normal('learning_rate', 0, 1)])
values = []
for x in range(1000):
    values.append(hyperopt.pyll.stochastic.sample(space))
df = pd.DataFrame(values, columns=["var1"])
df.hist(bins=100)


# So in the above I create two identical normal distributions, but move the center of one of them to 20. This "20+..." part is similar to how I did above with "1000-..." to produce a custom distribution.
# 
# As visible in the above, hyperopt now randomly chooses between the two choices when it picks the choice path. Then it proceeds down the path to generate a value of its choice from one of these distributions. 
# 
# This search space definition above has only that one hp.choice() at top level so it always picks that first, the this makes a random choice between the two normal distributions below it.
# 
# For further visualization, I pick a third distribution that has its own independent peak at 1000 but also a longish tail that overlaps with the two normal distributions. How does that look? Like so:
# 

# In[ ]:


space = hp.choice('learning_rate', [hp.normal('learning_rate_1', 0, 20),
                                            200+hp.normal('learning_rate_2', 0, 20),
                                           1000-hp.loguniform('learning_rate_3', np.log(10), np.log(1000))])
values = []
for x in range(1000):
    values.append(hyperopt.pyll.stochastic.sample(space))
df = pd.DataFrame(values, columns=["var1"])
df.hist(bins=100)


# Matches what I expected. They can also overlap (the choices).
# 
# That is mostly the distributions. Finally, what about the overall algorithm used by hyperopt? Let's see:

# ## Tree Parzen Estimator

# If you look up some explanation on how hyperopt works, the algorithm applied is something called a [tree of parzen estimator](https://hyperopt.github.io/hyperopt/#algorithms) (TPE). What does it mean? I actually had a bit of a hard time finding any good and intuitive explanations. Mostly websites just mention the name and do not really give a good explanation.
# 
# Strart with the tree part of it. If you search the internets for "tree parzen estimator", you will get a lot of hits for "tree-structured parzen optimizer". Which seems to be the name of the general technique. Reading the [paper](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf) on the topic (as well as other materials on the internets), it seems that there are actually two parts to this. The tree structure (hence I guess also the hyperopt name "tree of..") and the parzen part.
# 
# There is a nice [response on datascience stackexchange](https://datascience.stackexchange.com/questions/42133/what-makes-a-tree-structured-parzen-estimator-tree-structured) on the tree question. This also references the [original paper](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf) in stating that the configuration space is described as a tree of possible choices. So this is basically the hp.choice() configuration above, allowing to define valid choices in relation to other choices, such as number of hidden units is only valid for a multi-layer neural network. This is the tree part, so something like this from my above examples:
# 
# ```
# -learning_rate
#  |-learning_rate_1 (hp.normal('learning_rate_1', 0, 20)
#  |-learning_rate_2 200+hp.normal('learning_rate_2', 0, 20)
#  |-learning_rate_3 1000-hp.loguniform('learning_rate_3', np.log(10), np.log(1000))
# ```
# 
# The [StackExchange answer](https://datascience.stackexchange.com/questions/42133/what-makes-a-tree-structured-parzen-estimator-tree-structured) has a better example. But the real tree form would then be deeper than that, and maybe make a bit more sense in terms of parameter trees. But that is the tree part as far as I can tell.
# 
# As for the Parzen part, there are various references to [Parzen](https://sebastianraschka.com/Articles/2014_kernel_density_est.html) [Windows](https://en.m.wikipedia.org/wiki/Kernel_density_estimation) on the internets. [This explanation](https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f) seems relatively understandable. 
# 
# To my understanding, TPE uses an approach based on Bayesian probabilities. You start with some random sampling of the search space, see how the objective function (your classifier) does in that space. Collect that information, and use probabilities for the collected hyperparameter values vs the score of the objective function to further guide the search in the search space. 
# 
# And I guess since you provide hyperopt with the probability densities (all the hp.xxxx functions defining the search space), it will be a form of combination of the collected probabilities from the experiments and the defined probability functions. Weight one with the other to get the probability to pick a value from some part of the distribution.
# 
# Over time, this should focus on the more promising parts of the search space. The search would initially put more focus on the higher probability densities we define as input but over time would focus on the better "scoring" parts more. So I think. What do you think?
# 
# 

# # Summary

# I would summarize my learnings from this as a process of looking at plots of the distributions to understand what am I really configuring, and trying a few distributions and see which one matches my expected search space for that parameter. Some personal opinions on the distribution settings (for now, maybe some day I know better..):
# 
# #### Random values:
# 
# - hp.randint(label, upper)
# 
# This is described in [hyperopt docs] as "The semantics of this distribution is that there is no more correlation in the loss function between nearby integer values, as compared with more distant integer values", and being more suitable for values like random seeds. I interpret this description to suggest using randint() when the values are independent and discrete. In other cases the values would be less independent, for example, values closer to a good value would be more likely to be good. In such a case the other functions would serve better. randint() only if there is no relation.
# 
# #### Normal distributions values:
# 
# Variants of the normal distribution with a mean *mu* and standard deviation *sigma*. Use when you want to focus exploration more on a specific part of the parameter value-space. I find it simpler to generate a more normal like distribution and shift or scale it as needed. But that's just me.
# 
# - hp.normal(label, mu, sigma)
# - hp.qnormal(label, mu, sigma, q)
# - hp.lognormal(label, mu, sigma)
# - hp.qlognormal(label, mu, sigma, q) 
# 
# #### Uniform distribution values:
# 
# Variants of uniform distribution ranging from *low* to *high*. Use to get values in a range with equal probability. Like trying number of estimators a classifier uses.
# 
# - hp.uniform(label, low, high)
# - hp.quniform(label, low, high, q)
# - hp.loguniform(label, low, high)
# - hp.qloguniform(label, low, high, q)
# 
# #### Log-versions:
# 
# Distributions where the log of the value is in the given range (of normal distribution or uniform). Use np.log(x) to set specific range values as illustrated further above. Seems handy to focus more on smaller/larger values of the normal/uniform distributions.
# 
# - hp.lognormal(label, mu, sigma)
# - hp.qlognormal(label, mu, sigma, q) 
# - hp.loguniform(label, low, high)
# - hp.qloguniform(label, low, high, q)
# 
# #### Quantized versions:
# 
# These distributions are like their non-q counterparts but with the possible values space in steps size of *q*. Range 0 to 10 with *q*=2 would give options of 0,2,4,6,8,10. Range 0 to 1 with *q*=0.2 would give options of 0.0, 0.2, 0.4, 0.6, 0.8, 1.0. Maybe it is sometimes helpful to focus on fewer values in a larger range, while exploring combinations in relation to those. 
# 
# - hp.quniform(label, low, high, q)
# - hp.qloguniform(label, low, high, q)
# - hp.qnormal(label, mu, sigma, q)
# - hp.qlognormal(label, mu, sigma, q) 
# 
# #### Choice:
# 
# Useful for independent choices of values, such as "gini entropy" vs "information gain" in splitting a decision tree.
# 
# - hp.choice(label, options)
# 
# ## Summary of Summary - My Picks (for now):
# 
# I would perhaps try these:
# 
# - hp.randint(): Not sure if this makes any difference from just randomly picking a value, but I guess it could be interesting to try if hyperopt can make any use of a seed value generated with this function. It also allows hyperopt to play with the value along with other values.
# - hp.uniform(): I like that this gives me exact control over the value range, and is easier to set than playing with the parameters of the normal distribution. The quantized and log versions give some control to also weight this besides the plain normal distribution versions.
# - hp.normal(): Could try the variants of this sometimes to see if it makes any difference in the distributions. Many seem to use it, although I like the uniform variants for simplicity.
# - hp.choice(): Choice of "categorical" configuration parameters such as the "gini entropy" vs "information gain" parameters.
# 
# But this might change at any time as I would learn something when playing with these. In any case, I would just generally think of reasonable value ranges and distibutions for exploring a parameter, plot the hyperopt distributions as I did in this kernel to see what a configuration really means, and go with a sensible looking choice.
# 

# In[ ]:




