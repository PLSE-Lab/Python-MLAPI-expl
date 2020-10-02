#!/usr/bin/env python
# coding: utf-8

# # Monte Carlo integration
# learned from: https://github.com/afeiguin/comp-phys/blob/master/10_01_montecarlo_integration.ipynb
# ## The idea
# Imagine tha twe want to measure the area of a pond with arbitrary shape. Suppose that this pond is in the middle of a field with known area $A$. If we throw $N$ stones randomly, such that they alnd within the boundaries of the field, and we count the number of stones that fall in the pond $N_{in}$, the area of the pond willbe approximately proportional to the fraction of stones that make a splash, multiplied by $A$:
# \begin{equation}
#     A_{pond} = \frac{N_{in}}{N}A
# \end{equation}
# 
# This simple procedure is an example of the "Monte Carlo" method.
# 
# ## Simple Monte Carlo integration
# More generally, iagine a rectangle of height H in the integration interval $[a, b]$, such that the function $f(x)$ is within its boundaries. Compute $n$ pairs of random numbers $(x_i, y_i)$ such that they are uniformly distributed inside this rectangle. The faction of points that falls within the area contained below $f(x)$, *i.e.,* that satisfy $y_i \leq f(x_i)$ is an estimate of the ratio of the integral of $f(x)$ and the area of the rectangle. Hence, the estimate of the integral will be given by:
# \begin{equation}
# \int_a^bf(x)dx \simeq I(N) = \frac{N_{in}}{N}H(b-a)
# \end{equation}
# 
# Another Monte Carlo procedure is based on the definiton of the average of a function $f(x)$:
# \begin{equation}
# \langle g \rangle = \frac{1}{(b-a)}\int_a^bf(x)dx
# \end{equation}
# 
# In order to determine this average, we sample the value of $f(x)$:
# \begin{equation}
# \langle f \rangle \simeq \frac{1}{N}\sum_{i=1}^Nf(x_i)
# \end{equation}
# 
# Thus:
# \begin{equation}
# \frac{1}{(b-a)}\int_a^bf(x)dx = \frac{1}{N}\sum_{i=1}^Nf(x_i)
# \end{equation}
# 
# Or:
# \begin{equation}
# \int_a^bf(x)dx = \frac{(b-a)}{N}\sum_{i=1}^Nf(x_i)
# \end{equation}
# where the $N$ values $x_i$ are distributed uniformly in the interval $[a, b]$. The integral will be given by:
# \begin{equation}
# I(N) = (b-a)\langle f \rangle = \frac{b-a}{N} \sum_{i=1}^Nf(x_i) \text{   }(1)
# \end{equation}
# 
# ## Monte Carlo error analysis
# The Monte Carlo method clearly yields approximate results. The accuracy depends on the number of values N that we use for the average. A possible measure of the error is the "variance" $\sigma^2$ defined by:
# \begin{equation}
# \sigma^2 = \langle f^2 \rangle - {\langle f \rangle}^2
# \end{equation}
# where
# \begin{equation}
# \langle f \rangle = \frac{1}{N}\sum_{i=1}^N f(x_i)
# \end{equation}
# and
# \begin{equation}
# \label{eq:sigma}
# \langle f^2 \rangle = \frac{1}{N}\sum_{i=1}^N {f(x_i)}^2
# \end{equation}
# 
# The "standard deviation" is $\sigma$. However, we should expect that the error decreases with the number of points $N$, and the quantity $\sigma$ defines by (\ref{eq:sigma}) does not. Hence, this cannot be a good measure of the error.
# 
# Imagine that we perform several measurements of the integral, each of them yielding a resuult $I_n$. These values have been obtained with different sequences of N random numbers. According to the central limit theorem, these values would be normally distributed around a mean $\langle I \rangle$. Suppose that we have a set of M of such measurements $I_n$. A convenient measure of the differences of these measurements is the "standard deviation of the means" $\sigma_M$:
# \begin{equation}
# \sigma^2_M = \langle I^2 \rangle - {\langle f \rangle}^2 \text{   }(2),
# \end{equation}
# where 
# \begin{equation}
# \langle I \rangle = \frac{1}{M}\sum_{n=1}^M I_n  \text{   }(3),
# \end{equation}
# and
# \begin{equation}
# \langle I^2 \rangle = \frac{1}{M}\sum_{n=1}^M I_n^2 \text{   }(4)
# \end{equation}
# It can be proven that
# \begin{equation}
# \sigma_M = \sigma/\sqrt{N} \text{   }(5)
# \end{equation}
# This relation becomes exact in the limit of a very large number of measurements. Note that this expression implies that the error decreases with the square root of th enumber of trials, meaning that if we want to reduce error by a factor 10, we need 100 times more points for the average.
# 
# ## Excercise 10.1: One dimensional integration
# 1. Write a program that implements the "hit and miss" Monte Carlo integration algorithm. Find the estimate $I(N)$ for the integral of:
# \begin{equation}
# f(x) = 4\sqrt{1-x^2}
# \end{equation}
# as a function of $N$, in the interval (0, 1). Choose H = 1, and sample only the x-dependant part $\sqrt{1-x^2}$, and multiply the result by 4. Calculate the difference between $I(N)$ and the exact result $\pi$. This difference is a measure of the error associated with the Monte Carlo estimate. Make a log-log plot of the error as a function of $N$. What is the approximate functional dependence of the error on $N$ for large $N$?
# 2. Estimate the integral of $f(x)$ using the simple Monte Carlo integration by averaging over $N$ points, using integral formula $(3)$, and compute the error as a funciton of N, for N up to 10,000. Determine the approximate functional dependence of the error on $N$ for large $N$. How many trials are necessary to determine $I_N$ to two decimal places?
# 3. Perform 10 measurements I_n(N), with N = 10,000 using different random sequences. Show in a table the values of $I_n$ and $\sigma$ according to the formlulas for integral $(1)$ and sigma $(2)$. Use the formula $(5)$ to estimate the standard deviation of the means, and cmopare to the values obtained from the formula $(2)$ using 100,000 values.
# 4. To verify that your result for the error is independent of the number of sets you used to divide your data, repeat the previous item grouping your results in 20 groups of 5,000 points each.
# 
# ## Exercise 10.2: Importance of randomness
# To examine the efects of a poor random number generator, modify your program to use the linear congruential random number generator using the parameters $a=5, c = 0$ and the seed $x_1 = 1$. Repeat the integral of the previous exercise and compare your results.

# In[ ]:


# Exercise 10.2
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from matplotlib import pyplot


# In[ ]:


x = np.arange(0, 1, 0.02)
fig = pyplot.plot(x, 4*np.sqrt(1-x**2))


# In[ ]:


# Hit and miss Monte Carlo integration
ngroups = 16
N = np.zeros(ngroups)
I = np.zeros(ngroups)
E = np.zeros(ngroups)

n0 = 100
for i in range(ngroups):
    N[i] = n0
    x = np.random.random(n0)
    y = np.random.random(n0)
    I[i] = 0
    Nin = 0
    for j in range(n0):
        if(y[j]<np.sqrt(1-x[j]**2)):
            Nin += 1
    I[i] = 4*float(Nin)/float(n0)
    E[i] = abs(I[i] - np.pi)
    print(n0, Nin, I[i], E[i])
    n0 *= 2

pyplot.plot(N, E, ls='-', c='red',lw =3);
pyplot.plot(N, 0.8/np.sqrt(N), ls='-', c='blue', lw=3)
pyplot.xscale('log')
pyplot.yscale('log')


# In[ ]:


# Simple Monte Carlo Integration
ngroups = 16
N = np.zeros(ngroups)
I = np.zeros(ngroups)
E = np.zeros(ngroups)

n0 = 100
for i in range(ngroups):
    N[i] = n0
    r = np.random.random(n0)
    I[i] = 0.
    for j in range(n0):
        x = r[j]
        I[i] += np.sqrt(1-x**2)
        
    I[i] *=  4./float(n0)
    E[i] = abs(I[i] - np.pi)
    print(n0, I[i], E[i])
    n0 *= 2

pyplot.plot(N, E, ls='-', c='red', lw = 3)
pyplot.plot(N, 0.8/np.sqrt(N), ls='-', c='blue', lw=3)
pyplot.xscale('log')
pyplot.yscale('log')


# In[ ]:


n0 = 100000
I = np.zeros(n0)
r = np.random.random(n0)
for j in range(n0):
    x = r[j]
    I[j] = 4 * np.sqrt(1-x**2)
    
def group_measurements(ngroups):
    global I, n0
    nmeasurements = n0//ngroups;
    for n in range(ngroups):
        Ig = 0.
        Ig2 = 0.
        for i in range (n*nmeasurements, (n+1)*nmeasurements):
            Ig += I[i]
            Ig2 += I[i]**2
        Ig /= nmeasurements
        Ig2 /= nmeasurements
        sigma = Ig2 - Ig**2
        print(Ig, Ig2, sigma)
group_measurements(10)
print("-----------------------------------")
group_measurements(20)
print("-----------------------------------")
group_measurements(1)


# ## Variance reduction
# If the function being integrated does nto fluctuate too much in the interval of integration, and does not differ much from the average value, then the standard Monte Carlo mean-value method should work well with a reasonable number of points. Otherwise, we will find that the variance is very large, meaning that some points will make small contributions, while others will make large contributions to the integeral. If this is the case, the algorithm will be very inefficient. The method can be improved by splitting the function $f(x)$ in two $f(x) = f_1(x) + f_2(x)$, such that the integral of $f_1(x)$ is known, and $f2(x)$ as a small variance. The "variance reduction" technique, consists then in evaluating th eintegral of $f_2(x)$ to obtain:
# \begin{equation}
# \int_a^bf(x)dx = \int_a^bf_1(x)dx + \int_a^bf_2(x)dx = \int_a^bf_1(x)dx + J
# \end{equation}
# 
# ## Importance of Sampling
# Imagine that we want to sample the function $f(x) = e^{-x^2}$ in the interval $[0, 1]$. It is evident that most of our points will fall in the region where value of $f(x)$ is very small, and therefore we will need a large number of values to achieve a decent accuracy. A way to improve the measurement by reducing the variance is obtained by "importance sampling". As the name says, the idea is to sample the regions with large contributions to the integral. For this goal, we introduce a probability distribution $P(x)$ normalized in the interval of integration:
# \begin{equation}
# \int_a^bP(x)dx = 1
# \end{equation}
# 
# Then, we can rewrite the integral of $f(x)$ as:
# \begin{equation}
#     I = \int_a^b\frac{f(x)}{P(x)}P(x)dx
# \end{equation}
# We can evaluate this integral, by sampling according to the probability distribution $P(x)$ and evaluting the sum:
# \begin{equation}
#     I(N) = \frac{1}{N}\sum_{i=1}^N\frac{f(x_i)}{P(x_i)}.
# \end{equation}
# 
# Note that for the uniform case $P(x) = 1/(b-a)$, the expression reduces to the simple Monte Carlo integral.
# 
# We are free to choose $P(x)$ now. We wish to do it in a way to reduce and minimize the variance of the integrand $f(x)/P(x)$. The way to do this is picking a $P(x)$ that mimics $f(x)$ where $f(x)$ is large. If we are able to determine an appropriate $P(x)$, the integrand will be slowly varying, and hence the variance will be reduced. Another consideration is that the generation of points according to the distribution $P(x)$ should be simple task. As an example, let us consider again the integeral
# \begin{equation}
#     I = \int_0^1e^{-x^2}dx.
# \end{equation}
# 
# A reasonable choice for a weight function is $P(x) = Ae^{-x}$, where A is a normalization constant.
# 
# Notice that for $P(x) = f(x)$ the variance is zero! This is known as the zero variance property. There is a catch, though: The probability function $P(x)$ needs to be normalized, implying that inreality, $P(x) = f(x)/\int f(x)dx$, which assumes that we know in advance precisely the integeral that we are trying to calculate!
# 
# 
# ### Exercise 10.3: Importance sampling
# 1. Choose the weight function $P(x) = e^{-x}$ and evaluate the integral
# \begin{equation}
#        \int_0^{\infty}x^{3/2}e^{-x}dx.
# \end{equation}
# 2. Choose $P(x) = e^{-ax}$ and estimate
# \begin{equation}
#     \int_0^{\pi}\frac{dx}{x^2 + cos^2x}
# \end{equation}
# Determine the value of $a$ that  minimizes the variance of the integral

# In[ ]:


pyplot.xlim(0, 10)
pyplot.ylim(0, 1)
x = np.arange(0, 10, 0.1)
pyplot.plot(x, np.exp(-x), label='e^{-x}')
pyplot.plot(x, np.exp(-x**2), label='e^{-x^2}')
pyplot.plot(x, x**1.5*np.exp(-x), label='x^{3/2}e^{-x}')
pyplot.legend()


# In[ ]:


# Trapezoidal integration
def trapezoids(func, xmin, xmax, nmax):
    Isim = func(xmin) + func(xmax)
    h = (xmax-xmin)/nmax
    for i in range(1, nmax):
        x = xmin + i *h
        Isim += 2*func(x)
    Isim *= h/2
    return Isim


# In[ ]:


def f(x):
    return x**1.5*np.exp(-x)


# In[ ]:


print("Trapezoids: ", trapezoids(f, 0., 20., 100000))


# In[ ]:


# Simple Monte arlo integration
n0 = 100000
r = np.random.random(n0)

Itot = np.sum(r**1.5*np.exp(-r))
print("Simple Monte Carlo: ", Itot/n0)


# In[ ]:


# Importance sampling
x = -np.log(r)
Itot = np.sum(x**1.5)
print("Importance Sampling: ", Itot/n0)


# In[ ]:


pyplot.xlim(0, np.pi)
x = np.arange(0, np.pi, 0.05)
pyplot.plot(x, 1./(x**2 + np.cos(x)**2), label='one')
pyplot.plot(x, np.exp(-x), label='two')
pyplot.plot(x, np.exp(-2*x), label='three')
pyplot.plot(x, np.exp(-0.2*x), label='four')
pyplot.legend()


# In[ ]:


# Trapezoidal integration
def g(x):
    return 1./(x**2+np.cos(x)**2)

print("Trapezoids: ", trapezoids(g, 0., np.pi, 1000000))


# In[ ]:


# Simple Monte Carlo integeration
n0 = 1000000
a = np.arange(0.1, 2.1, 0.1)
I = np.arange(0.1, 2.1, 0.1)

r = np.random.random(n0)
I0 = np.sum(1./((r*np.pi)**2 +np.cos(r*np.pi)**2))
print("Simple Monte Carlo: ", I0/n0*np.pi)


# In[ ]:


# importance sampling
print("Importance Sampling:")
x = -np.log(r)
i = 0
for ai in a:
    norm = (1.-np.exp(-ai*np.pi))/ai
    x1 = norm*x/ai
    Itot = 0.
    Nin = 0
    I2 = 0.
    for xi in x1:
        if(xi <= np.pi):
            Nin += 1
            Itot += g(xi)*np.exp(xi*ai)
            I2 += (g(xi)*np.exp(xi*ai))**2
    Itot *= norm
    I2 *= norm
    
    I[i] = Itot/Nin
    i+=1
    print(ai, Itot/Nin, np.sqrt(abs(Itot**2/Nin**2-I2/Nin))/np.sqrt(Nin))
pyplot.plot(a, I, ls='-', marker='o', c = 'red', lw=3)


# ### Exercise 10.4: The Metropolis algorithm
# Use the Metropolis algorithm to sample points according to a distribution and estiamte the integral
# \begin{equation}
#     \int_0^4 x^2 e^{-x}dx
# \end{equation}
# with $P(x) = e^{-x}$ for $0\leq x \leq4$. Plot the number of times the walker is at point $x_0, x_1, x_2, ...$ Is the integrand sampled uniformly? If not, what is the approximte region of $x$ where the integrand is sampled more often?

# In[ ]:


delta = 2
xmin = 0.
xmax = 4.0
def f(x):
    return x**2*np.exp(-x)

def P(x):
    global xmin, xmax
    if(x<xmin or x> xmax):
        return 0.
    return np.exp(-x)

def metropolis(xold):
    global delta
    xtrial = np.random.random()
    xtrial = xold +(2*xtrial-1)*delta
    weight = P(xtrial)/P(xold)
    xnew = xold
    if weight >= 1:
        xnew = xtrial
    elif(weight != 0 ):
        r = np.random.random()
        if(r<=weight):
            xnew = xtrial
    return xnew
xwalker = (xmax + xmin)/2.
for i in range(100000):
    xwalker = metropolis(xwalker)

I0 = 0.
N = 300000
x = np.zeros(N)
x[0] = xwalker
for i in range(1, N):
    for j in range(20):
        xwalker = metropolis(xwalker)
    x[i] = xwalker
    I0 += x[i]**2

binwidth = 0.1
pyplot.hist(x, bins= np.arange(xmin-1, xmax+1, 0.1), normed= True)

print("Trapezoids: ", trapezoids(f, xmin, xmax, 100000))
print("Metropolis: ", I0*(1.0-np.exp(-4.0))/N)


# In[ ]:


fig=pyplot.hist(x**2, bins=np.arange(xmin**2-1, xmax**2+1, 0.1), normed=True)


# # Monte Carlo Methods
# Learned from: https://towardsdatascience.com/monte-carlo-simulations-with-python-part-1-f5627b7d60b0

# ## The Crude Monte Carlo: Implementation
# 1. Get a random input value from the integration range
# 2. Evaluate the integrand
# 3. Repeat Steps 1 and 2 for as long as you like
# 4. Determine the average of all these samples and multiply by the range

# In[ ]:


import numpy as np
import math
import random
from matplotlib import pyplot as plt
from IPython.display import clear_output
PI = 3.1415926
e = 2.71828


# In[ ]:


# uniform random value from a range
def get_rand_number(min_value, max_value):
    range_value = max_value - min_value
    choice = random.uniform(0, 1)
    return min_value + range_value*choice


# In[ ]:


def f_of_x(x):
    """
    The function that we want to integerate over
    """
    return (e**(-1*x))/(1+(x-1)**2)


# ### The algorithm

# In[ ]:


def crude_monte_carlo(num_samples=5000):
    lower_bound = 0
    upper_bound = 50
    
    sum_of_samples = 0
    for i in range(num_samples):
        x = get_rand_number(lower_bound, upper_bound)
        sum_of_samples += f_of_x(x)
    return (upper_bound - lower_bound)*float(sum_of_samples/num_samples)


# In[ ]:


print(crude_monte_carlo(100000))


# ### Determine the Variance of the estimation
# Run the algorithm for several times and find the variance
# $\sigma^2 = <I^2> - <I>^2$, where $I$ is the integral value.
# Or
# \begin{equation}
# \sigma^2 = \mathcal{[}\frac{b-a}{N}\sum_i^N f^2(x_i) \mathcal{]} - \mathcal{[}\sum_j^N\frac{b-a}{N}f(x_j)\mathcal{]}^2
# \end{equation}
# This is the equation for variance we'll use in our simulations. Let's see how to do this in Python.

# In[ ]:


def get_crude_MC_variance(num_samples):
    """
    This function returns the variance of the Crude Monte Carlo.
    Note that the inputed number of samples does not necessarily need to correpsond to the number of samples used in the Monte Carlo Simulation.
    Args:
    - num_samples (int)
    Return:
    - Variance for Crude Monte Carlo approximation of f(x) (float)
    """
    int_max = 5 # this is th emax of our integration range
    # get the average of squares
    running_total = 0
    for i in range(num_samples):
        x = get_rand_number(0, int_max)
        running_total += f_of_x(x)**2
    sum_of_sqs = running_total*int_max/num_samples
    
    # get square of average
    running_total = 0
    for i in range(num_samples):
        x = get_rand_number(0, int_max)
        running_total += f_of_x(x)
    sq_ave = (int_max*running_total/num_samples)**2
    
    return sum_of_sqs - sq_ave


# This implementation of the Crude Monte Carlo gives us a variance of 0.266 which corresponds to an error of 0.005. For a quick, back of the envelop estimate, this isn't bad at all, but what if we need a lot more precision? We could always jsut increase teh nuber of samples, but then our computation time will increase as well. What if, instead of using random sampling, we cleverly sampled from the right distribution of points ... called, *importance sampling*.

# ## Importance Sampling: The Intuition
# Importance sampling is a method for reducing the variance of a Monte Carlo simulation without increasing the number of samples. The idea is that instead of randomly sampling from the whole function, let's just sample from a distribution of points similarly shaped to the function.
# 
# Let's say you have a step function active on the range $[0, 2]$ and inactive from $[2, 6]$. Sampling 10 times might yield estimates like this:

# In[ ]:


def step_f(x):
    return 1 if x<=2 else 0
for i in range(10):
    x = get_rand_number(0, 6)
    print(x, step_f(x))


# In[ ]:


(4.0/10.0)*6


# These samples correspond to a most likely distribution of samples, and yield an integral estimation of 2.4. But, what if instead, we estimate the ratio between our function $f(x)$ and some special weight function $g(x)$ whose value is almost always about 1/2 the value of $f(x)$ for any given x? What if we also bias our samples to appear in teh most acive ranges of our function (which we'll find to minimize the error). You'll see that the average of these ratios is a lot closer the real value of our integral, which is 2. The importance sampling method is used to determine this optimal function $g(x)$.

# ## The Math
# Let's see if we can find a $g(x)$ such that:
# \begin{equation}
#     \frac{f(x)}{g(x)} \simeq k\text{, where k is some constant}
# \end{equation}
# Basically, we want g(x) to look like a scaled version of $f(x)$.
# 
# We'll also need $g(x)$ to satisfy a few criteria:
# 1. $g(x)$ is integrable
# 2. g(x) is non-negative on $[a, b]$
# 3. The indefinite integral of $g(x)$, which we'll call $G(x)$, has a real inverse
# 4. The integral of $g(x)$ in the range $[a, b]$ must equal 1
# 
# In ideal case, $f(x) = k *g(x)$, where $k$ is a constant. However, if $f(x) = k*g(x)$, then $f(x)$ would be integrable and we would have no need to perform a Monte Carlo simulation; we could just solve the problem analytically!
# 
# So, we'll settle for $f(x) \simeq k*g(x)$. We won't get a perfect estimate of course, but you'll find it performs better than our crude estimation from earlier.
# 
# We'll define $G(x)$ as follows, and we'll also perform a change of variables to $r$.
# \begin{equation}
#     G(x) = \int_0^x g(x) d(x) \\
#     r = G(x)
# \end{equation}
# 
# $r$ will be restricted to the range $[0, 1]$.Sicne the integral of $g(x)$ was defined to be 1, $G(x)$ can never be greater than 1, and therefore $r$ can never be greater than one. This is important becasue later, we will randomly sample from $r$ in the range $[0, 1]$ when performing the simulation.
# 
# Using these definitions, we can produce the following estimation:
# \begin{equation}
#     I \simeq \frac{1}{N}\sum_i^N \frac{f(G^{-1}(r_i))}{g(G^{-1}(r_i))}
# \end{equation}
# 
# This sum is what we will be calculating when we perform the Monte Carlo. We'll randomly sample from r in order to produce our estimate.
# 
# Simple right? Don't be intimated if this doesn't make sense at first glance. I intentionally focused o the intuition and breezed through the math quite a bit. If you're confued, or you want more mathematical rigor, check out the resource I talked about earlier until you believe that final equation.
# 
# ## Importance Sampling: Python Implementation.
# Ok, now that we understand the math behind importance sampling, let's go back to our problem from before. Remember, we're trying to estimate the following integral as precisely as we can:
# \begin{equation}
#     I = \int_0^\infty \frac{e^{-x}}{1 + (x - 1)^2} dx
# \end{equation}
# 
# ## Visualizing our problem
# Let's start by generating a template for our $g(x)$ weight function. I'd like to visualize my function $f(x)$, so we'll do that using <code>matplotlib</code>:
# 

# In[ ]:


xs = [float(i/50) for i in range(int(50*PI*2))]
ys = [f_of_x(x) for x in xs]
plt.xlim([0, 6])
plt.ylim([0, 0.5])
plt.plot(xs, ys)
plt.title("f(x)");


# OK, so we can see that our function is mostly active in the rough rangeof $[0, 3-ish]$ and is mostly inactive on the range $[4-ish, \infty]$. So, let's see if we can find a function template that can be parameterized to replicate this quality. Deb's proposes this function:
# \begin{equation}
#     g(x) = Ae^{-\lambda x}
# \end{equation}
# 
# After we find the ideal values for $A$ and $\lambda$, we'll be able to construct this plot of $f(x)$ and our optimal weight function $g(x)$.
# 
# You can see that inmany ways $g(x)$ does n ot ideally replicate the shape of $f(x)$. This is ok. A crude $g(x)$ can still do the marvels for decreasing your estimation variance. Feel free to experiment with other weight function $g(x)$ to see if you can find even better solutions.
# 
# 
# ## Parameterize g(x)
# Before we can perform the simulation, we will need to find the optimal parameters $\lambda$ and $A$. We can find $A(\lambda)$ using the normalization restriction on $g(x)$:
# \begin{equation}
#     \mathcal{1} = \int_0^{\infty} g(x)dx \rightarrow A = \lambda
# \end{equation}
# 
# Now, all we need to do is find the ideal $\lambda$, and we'll have our ideal $g(x)$.
# 
# To do this, let's calculate the variance for different $\lambda$ on the range $[0.05, 3.0]$ in increament of 0.5, and use the $\lambda$ with the lowest variance.
# 
# When using importance sampling, we calculate the variance of the ratio between f(x) and g(x).
# 
# \begin{equation}
#     \sigma^2 = [\frac{1}{N}\sum_i^N\frac{f^2(x_i)}{g^2(x_i)}] - [\sum_j^N\frac{1}{N}\frac{f(x_j)}{g(x_j)}]^2
# \end{equation}
# And we'll want to use this equation to approximate the integral:
# 
# \begin{equation}
#     I \simeq \frac{1}{N} \sum_i^N \frac{f(G^{-1}(r_i))}{g(G^{-1}(r_i))}
# \end{equation}
# 
# We'll recalculate the variance for different $\lambda$, changing the weight function accordingly each time. After, we'll use our optimal $\lambda$ to calcualte the integreal with minimal variance.
# 
# The algorithm will look like this:
# 1. Start at $\lambda = 0.05$
# 2. Calculate the variance
# 3. Increment $\lambda$
# 4. Repeat steps 2 and 3 until you reach the last $\lambda$
# 5. Pick the $\lambda$ with the lowest variance - this is your optimal $\lambda$
# 6. Use importance sampling Monte Carlo with this $\lambda$ to calculate the integral

# In[ ]:


# finding the optimal lambda
def g_of_x(x, A, lamda):
    e = 2.71828
    return A*math.pow(e, -1*lamda*x)

def inverse_G_of_r(r, lamda):
    return (-1 * math.log(float(r)))/lamda

def get_IS_variance(lamda, num_samples):
    """
    This function calculates the variance if a Monte Carlo using importance sampling.
    Args:
    - lamda (float): lambda value of g(x) being tested
    Return:
    - Variance
    """
    A = lamda
    int_max = 5
    # get sum of squares
    running_total = 0
    for i in range(num_samples):
        x = get_rand_number(0, int_max)
        running_total += (f_of_x(x)/g_of_x(x, A, lamda)/g_of_x(x, A, lamda))**2
        
    sum_of_sqs = running_total/num_samples
    
    # get squared average
    running_total = 0
    for i in range(num_samples):
        x = get_rand_number(0, int_max)
        running_total += f_of_x(x)/g_of_x(x, A, lamda)
    sq_ave = (running_total/num_samples)**2
    
    return sum_of_sqs - sq_ave

# get variance as a funciton of lambda by testing many different lambdas
test_lambdas = [i*.05 for i in range (1, 61)]
variances = []
for i, lamda in enumerate(test_lambdas):
    print(f"lambda {i+1}/{len(test_lambdas)}: {lamda}")
    A = lamda
    variances.append(get_IS_variance(lamda, 10000))
    clear_output(wait=True)

optimal_lambda = test_lambdas[np.argmin(np.asarray(variances))]
IS_variance = variances[np.argmin(np.asarray(variances))]

print(f"Optimal Lambda: {optimal_lambda}")
print(f"Optimal Variance: {IS_variance}")
print(f"Error: {(IS_variance/10000)**0.5}")


# ## Run the simulation
# Now, all we ahve to do is run the simulation with our optimized g(x) function, and we're good to go. Here is what it looks like in ode:

# In[ ]:


def importance_sampling_MC(lamda, num_samples):
    A  = lamda
    running_total = 0
    for i in range(num_samples):
        r = get_rand_number(0, 1)
        running_total += f_of_x(inverse_G_of_r(r, lamda=lamda))/g_of_x(inverse_G_of_r(r, lamda=lamda), A, lamda)
    approximation = float(running_total/num_samples)
    return approximation


# In[ ]:


# run simulation
num_samples = 10000
approx = importance_sampling_MC(optimal_lambda, num_samples)
variance = get_IS_variance(optimal_lambda, num_samples)
error = (variance/num_samples)**0.5

# Display results
print(f"Importance samping approximation: {approx}")
print(f"Variance: {variance}")
print(f"Error: {error}")


# # Wrapping things up
# In this tutorial, we learned how to perform Monte Carlo simulations for the estimation of a definite integreal. We used both the crude method and the importance sampling method, and found that importance sampling provded a significant decrease in the error.

# In[ ]:




