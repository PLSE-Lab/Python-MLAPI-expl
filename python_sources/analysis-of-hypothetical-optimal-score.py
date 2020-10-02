#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from math import log
from random import random
from sklearn.metrics import log_loss
import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np

#Begin by importing these modules, which help with the analysis


# The very first question I had when starting this competition was what a reasonable lower bound for log loss should be. It's unclear whether 0.1, 0.01, 0.001, etc are reasonable floors for how well the given data can predict the likelihood of passing, so I figured the best way to gain insight would be to actually analyze the behavior of the scoring function. And what better way to do that than to put it together into a small kernel.

# Begin by considering random labels, and random probabilities assigned to them. Try running the following code block to see what it converges too.

# In[ ]:


Labels = []
Probs = []
amount = 10000

for i in range(amount):
  p = random()
  y = round(random())
  Labels.append(y)
  Probs.append([1-p,p])

print(log_loss(Labels,Probs))


# The result should come out to about 1, but why? We can show that it actually does tend to 1 (in the long run), by evaluating the expected value of log loss: \\(-(y \log(p) + (1-y)\log(1-p))\\), with our chosen parameters:
# 
# $$E\Big{[}\text{LogLoss}\quad \Big{|} \quad p \in \text{Uniform}(0,1), \quad y \in \operatorname{Bern} \left({\frac{1}{2}}\right)\Big{]} $$
# $$ = -\int_{0}^{1} \frac{\log(p)}{2}  \mathrm{dp} -\int_{0}^{1} \frac{\log(1-p)}{2} \mathrm{dp}  = -\int_{0}^{1} \log(p) \mathrm{dp} = 1 $$

# What if we however had perfect knowledge of the probability that we pass? That is to say, we select $p_1$, $p_2$ from $\text{Uniform}(0,1)$, our truth label becomes $p_1 > p_2$, and then we use $p_1$ as our guess. In such a system, with only knowledge of $p_1$, it is always the optimal guess! Try the code below to see how well we can do optimally.

# In[ ]:


Labels = []
Probs = []
amount = 10000

for i in range(amount):
  p1 = random()
  p2 = random()
  Labels.append(p1 > p2)
  Probs.append([1-p1,p1])

print(log_loss(Labels,Probs))


# So it turns out that here, we tend to 0.5! Proof:
# 
# $$E\Big{[}\text{LogLoss}\quad \Big{|} \quad p_1, p_2 \in \text{Uniform}(0,1), \quad y = (p_1 > p_2) \Big{]} $$
# 
# $$ = \int_0^1 \int_0^{p_1} \log(p_1) \mathrm{dp_2} + \int_{p_1}^1 \log(1-p_1) \mathrm{dp_2} \mathrm{dp_1} = \frac{1}{2}$$
# 
# (Excercise: What if instead of being given $p_1$, we were given $p_2$? Could we achieve the same, or even better performance?)
# 
# 
# 

# But in our problem, we deal with far less randomness, where clear cases exist for both pass and fail, and a small to moderate amount of cases fall in a mid probability region. What if we change the probability distribution we sample from to be bimodal with peaks at 0 and 1? Consider a [U-quadratic distribution](https://en.wikipedia.org/wiki/U-quadratic_distribution) in (0,1). Run the following two blocks to see how the error reduces as the shape of our pdf is more u shaped.
# 

# In[ ]:


#Define our custom distribution
class my_pdf(st.rv_continuous):
    def _pdf(self,x):
      return alpha*((x-beta)**2)
    def _cdf(self,x):
      return (alpha/3)*((x-beta)**3 + (beta-0)**3)
    def _ppf(self,x):
      q =  (3*x / alpha) - ((beta)**3)
      return np.sign(q)*(np.abs(q)**(1/3)) + beta

beta = 1/2
alpha = 12
uQuad = my_pdf(a=0, b=1, name='U-Quad')

samples = uQuad.rvs(size=100000)
plt.hist(samples,bins=100);


# In[ ]:


amount = 10000
Labels = []
Probs = []

for i in range(amount):  
  p1 = uQuad.rvs()
  p2 = random()
  Labels.append(int(p1 > p2))
  Probs.append([1-p1,p1])

print(log_loss(Labels,Probs))


# Solving for an exact analytic solution here is left as an excercise to the reader (the author was too lazy to compute it.)

# What if we were incredibly certain in all of our choices (e.g. we pick p = 0.99, and 0.01 only) over a large amount of trials. What does our score look like if we make M mistakes?

# In[ ]:


p = 0.99 #our confidence in our answers. Try changing it around.
amount = 1000
Labels = [random() > 0.5 for _ in range(amount)]
Probs = list(map({True : [1-p,p], False: [p,1-p]}.get, Labels))

maxErrors = amount
LogLosses = [0]*maxErrors
for i in range(maxErrors):
    LogLosses[i] = log_loss(Labels,Probs)
    #On the next itteration, we'll have 1 new mistake:
    Labels[i] = not Labels[i] 

plt.plot(LogLosses)
plt.title('Error Graph')
plt.xlabel('# Mistakes')
plt.ylabel('LogLoss Error')

printErrorAt = [0,1,2,10,20,50,100,250,999]

for i in printErrorAt:
    print('Error with {} mistakes: {}'.format(i,LogLosses[i]))


# Clearly here error grows linerally with the number of mistakes. With absolutely no mistakes, our performance is bounded bellow by $log(p_1)$. But even mislabeling 2% of cases is enough to lose an entire magnitude of accuracy (0.01 to 0.1), and mislabeling 25% of cases does worse than even randomly guessing would!
# 
# This observation has led to the conclussion that we can do no better than the expected likelihood of human errors, and variances between the training data and test data. For instance, if we believe that there is a 1/1000 chance that a human mislabeling error occurs, then any prediction we make needs to assume that it will be penalised at least once in 1000 trials. We can optimize this!

# In[ ]:


ExpectedHumanError = 1/100 #how often do we believe an unforseeable error will occur?
epsilon = 0.00001 #if the above variable is made too small, you may need to decrease epsilon
errors = np.array([-log(p)*(1-ExpectedHumanError) - log(1-p)*(ExpectedHumanError)           for p in np.arange(epsilon,1-epsilon,epsilon)])

minima = np.argmin(errors)*epsilon
#plot the region near the global minima
mask = (np.arange(epsilon,1-epsilon,epsilon) >= minima - 5*(1-minima))      & (np.arange(epsilon,1-epsilon,epsilon) <= minima + 5*(1-minima))
plt.plot(np.arange(epsilon,1-epsilon,epsilon)[mask],errors[mask])

print('Best error: {}\nat p = {}'.format(min(errors),minima))


# It is simple enough to get closed form solution for both the optimal $p$ value, and optimal error given how often we expect there to be a mislabeling in test data (denote this by $h$).
# 
# $$\min\quad -log(p)\cdot(1-h) - log(1-p)\cdot h$$
# We evaluate the derivative at 0.
# $$-\frac{1-h}{p} + \frac{h}{1-p} = 0 \implies p = 1-h$$
# 
# Therefore our expected optimal score is $-log(1-h)\cdot(1-h) - log(h)\cdot h$.
# 
# Lets plot this out now.

# In[ ]:


epsilon = 0.00001
rnge = np.arange(epsilon,1-epsilon,epsilon)
plt.plot(rnge,[-log(1-h)*(1-h)-log(h)*h for h in rnge])
plt.title('Hypothetical Optimal Score')
plt.xlabel('Expected Mislabels')
plt.ylabel('LogLoss Error');


# It's easy to see why the hypothetical score decreases near 0, but why does it decrease as we get closer to 1 as well? The reason for this is that we assumed that we have perfect knowledge of the labels, and we have aprori knowledge of the mislabeling rate. So at high mislabeling rates, our guesses would all just be flipped. At 0.5 mislabeling rate (the highest/worst possible score), everything is random, and the score is just -log(0.5).

# None of this analysis has actually touched the data, and it doesn't really give a numeric value for a hypothetical floor. Still i'd wager that the mislabeling rate is no better than 1/10,000, which by the above calculation yields ~0.001 as a lower bound on how well we can perform.

# In[ ]:


h = 1/10000
print(-log(1-h)*(1-h)-log(h)*h)


# What do you think is the best reasonably attainable score?
