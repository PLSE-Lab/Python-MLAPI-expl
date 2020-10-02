#!/usr/bin/env python
# coding: utf-8

# # Gentle Introduction to automatic differentiation

# Automatic differntiation is about computing derivatives of functions encoded as computer programs. In this notebook, we will build a skeleton of a toy autodiff framework in Python, using dual numbers and Python's magic methods.
# 
# ## Tl;dr summary ##
# * Automatic differentiation is a key component in every deep learning framework.
# * The basics of automatic differentiation are not hard.
# * Autograd package is awesome.

# ### Prior art
# There are plenty of resources about automatic differentiation. This notebook comes from my attempt to digest [this great blog post](https://alexey.radul.name/ideas/2013/introduction-to-automatic-differentiation/) by Alexey Radul. There've been plenty others, like this [article](http://www.ams.org/publicoutreach/feature-column/fc-2017-12) from AMS. In terms of code, [this github gist](https://gist.github.com/kmizu/1428717/b7ccee41e1d8ec62fbd2bd64df50bc8cb097d51c) is very much in the spirit as my code, albeit in Scala. Alexey Radul also teamed up with some autograd luminaries to write [a comprehensive survey](https://arxiv.org/abs/1502.05767) on automatic differentiation.

# ## Table of Contents 
# [Section 1. Derivative in a picture](#picture)
# 
# [Section 2. Building toy autodiff](#toy)
# 
#   [2a. Python magic methods](#magic)
#  
# [Section 3. How to build a real autodiff](#real)
# 
# [Section 4. Questions and Exercises.](#qa)

# # <a name="picture"></a>Section 1. Derivative in a picture #

# We will pay some lip service to the notion of derivative - instead of defining it, we will use a picture and an animation to illustrate it.

# Let's define a very simple function $$\text{func}(x)=3x^3-5x^2.$$ 
# The derivative of func can be readily computed to be $$\text{func_der}(x)=9x^2-10x.$$

# In[ ]:


#In Python code:
def func(x):
    return 3 * x ** 3 - 5 * x ** 2

def func_der(x):
    return 9 * x ** 2 - 10 * x


# What does derivative really mean? Let's imagine the graph of the function $\text{func}$. Let's take a point $x=1.5$. We compute the derivative $$\text{der_func}(1.5)= 9 * (1.5)^2 - 15 = 5.25.$$
# This informs us that the slope of the tangent line to the graph of $\text{func}$ is 5.25. Indeed, we can plot this to see.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, rc
from IPython.display import HTML


# In[ ]:


x = np.linspace(0,2,200)
y = func(x)
xprime = np.linspace(1.1,1.9)
yprime = (xprime - 1.5) * func_der(1.5)  + func(1.5)
plt.text(0.5, 2, "$y=3x^3-5x^2$", fontsize=20)
plt.text(0.5,1, "$y_{line}=5.25x-9$", fontsize=16)
plt.axvline(1.5, color='k', linestyle='--',linewidth=1) 
plt.plot(x,y, xprime, yprime, 'r--', [1.5], [func(1.5)], 'ro')


# ### Cauchy's definition of derivative animated ###

# What is the precise definition of the derivative? It involves a limit, so instead of writing this down, let's use an animation to illustrate.

# In[ ]:


#Initial plot to setup the animation
fig, ax = plt.subplots()
ax.set_xlim(( 0, 2))
ax.set_ylim((-4, 4))
_,_,_, point, line = ax.plot(x,y, xprime, yprime, 'r--', [1.5], [func(1.5)], 'ro', [],[], 'ko', [], [], 'k-')
text = ax.text(0.5, 1, "")
ax.text(0.5, 0.65, "derivative 5.25", color="r")


# Watch in the animation below how when the other end point approaches $x=1.5$, the slope of the arc becomes closer to the value given by the derivatvive $\text{der_func}(1.5)=5.25$

# In[ ]:


def init():
    line.set_data([], [])
    point.set_data([], [])
    text.set_text("")
    return (point, line, text)
def animate(i):
    if (i < 45):
        pt = 1.495 - 0.495 * (60 - i) / 60
    elif (i < 75):
        pt = 1.495 - 0.495 * (16.25 - (i-45)/2 ) / 75
    elif (i<80):
        pt = 1.495
    elif (i < 125):
        pt = 1.495 + 0.495 * (140 - i) / 60
    elif (i < 155):
        pt = 1.495 + 0.495 * (16.25 - (i-125)/2 ) / 75
    else:
        pt = 1.505
    x = np.linspace(0.8, 1.99)
    text.set_text("slope of the arc {0:.4f}".format((func(1.5) - func(pt))/(1.5 - pt)))
    y = (x - 1.5) * (func(1.5) - func(pt))/(1.5 - pt) + func(1.5)
    line.set_data(x, y)
    point.set_data([pt], [func(pt)])
    return (point, line, text)
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=160, repeat=True, blit=True)
HTML(anim.to_jshtml())


# ## Computation of derivatives - motivation ##

# So why do we seek to compute derivatives? There are quite a few uses for that
# 
# * Sensitivity analysis
# 
# Derivative is an indicator of a rate of change.
# 
# * Optimization
# 
# Can I find the minimum, maximum value that my code returns? Fastest optimization algorithms are based on computing derivatives.
# * Finding inverses
# 
# What is the inputs for my code that return 5?
# 
# For example, the function <i>square root</i> is the inverse of the function <i>square</i>. Indeed, [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method) to compute the inverse of function $f$ consists of iterating 
# $$
# x_{n+1} = x_{n} - \frac{f(x_n)}{f'(x_n)}
# $$
# until convergence and it uses derivatives.
# 
# * Machine Learning and Deep Learning
# 
# Machine learning training task (mainly supervised) is simply finding parameters that minimize a certain chosen function called loss. In deep learning, the number of parameters can get to billions, hence the need for smart ways to compute derivatives

# # <a name="build"></a> Section 2. Building toy autodiff

# One of the way to approach automatic differentiation is to use dual numbers. What are they?

# ## Dual numbers
# Suppose now $f$ is a nice enough function and $\varepsilon$ is a small number. Then one can write
# $$ f(x + \varepsilon) = f(x) + \varepsilon f'(x) + \varepsilon^2 ... + \varepsilon^3 ... + ....$$
# This is essentially Taylor's theorem.

# ### Stepping outside of math for a moment
# Suppose $\varepsilon$ is so small that $$\varepsilon^2=0$$ then $$f(x + \varepsilon) = f(x) + \varepsilon f'(x).$$
# So armed object like $x + \varepsilon$, we can plug it in to computation of $f$ and then read of the derivative from the coefficient of the $\varepsilon$ term. Simple, isn't it?

# ### Defining dual numbers
# Let's try to define this intuition. Dual number consists of a pair of real numbers, $(val, eps)$. In math notation one would write $val + eps \cdot\varepsilon$. Think of it as a Python object *Dual(val, eps)*. Define this in code:

# In[ ]:


class DualBasic(object):
    def __init__(self, val, eps):
        self.val = val
        self.eps = eps


# # <a name="magic"></a> 2a. Python magic methods

# At the next step, we would like manipulations of dual numbers to look as much as possible as manipulation of regular floats. In order to achieve that, we will use Python's `magic methods`. Magic methods  allow overloading/extending basic operators and functions to new classes by using special names for attributes and methods that serve as hooks for many python standard functions. See this [great tutorial](https://rszalski.github.io/magicmethods/) and the [official documentation](https://docs.python.org/3/reference/datamodel.html#special-method-names).

# For convenience sake, we will add an automatic coversion of ```float``` or ```int``` number ```x``` to ```Dual(x, 0)```. And also add the absolute value function and string representation.

# In[ ]:


class DualBasicEnhanced(object):
    def __init__(self, *args):
        if len(args) == 2:
            value, eps = args
        elif len(args) == 1:
            if isinstance(args[0], (float, int)):
                value, eps = args[0], 0
            else:
                value, eps = args[0].value, args[0].eps
        self.value = value
        self.eps = eps
        
    def __abs__(self):
        return abs(self.value)
    
    def __str__(self):
        return "Dual({}, {})".format(self.value, self.eps)

    def __repr__(self):
        return str(self)


# How should arithmetic work? The arithmetic operations should come from $\varepsilon^2=0$. So we define:
# 
# Addition
# $$(x + a \varepsilon)+(y + b\varepsilon)=(x+y) + (a+b)\varepsilon$$
# Multiplication
# $$(x + a \varepsilon)*(y + b\varepsilon)=xy + (xb+ya)\varepsilon$$

# In[ ]:


#In code:
class DualArith(object):
    def __add__(self, other):
        other = Dual(other)
        return Dual(self.value + other.value, self.eps + other.eps)
    
    def __sub__(self, other):
        other = Dual(other)
        return Dual(self.value - other.value, self.eps - other.eps)
    
    def __mul__(self, other):
        other = Dual(other)
        return Dual(self.value * other.value, self.eps * other.value + self.value * other.eps)


# Very important is the division of two dual numbers. There are two ways to arrive to what should the inverse of a dual number be. I'll give hints in the exercises. For now let's define:
# 
# Division
# $$ \frac{1}{x+a\varepsilon} = \frac{1}{x} - \varepsilon\frac{a}{x^2}$$

# In[ ]:


class DualDiv(object):
        def __truediv__(self, other):
            other = Dual(other)
            if abs(other.value) == 0:
                raise ZeroDivisionError
            else:
                return Dual(self.value / other.value, 
                            self.eps / other.value - self.value / (other.value)**2 * other.eps)


# We have enough to perform some basic computations with our class. Let's bring the all together in one class:

# In[ ]:


class Dual(DualBasicEnhanced, DualArith, DualDiv):
    pass


# __Now to the main point:__
# ## Statement: ##
# Suppose a Python function *piece_of_code* approximates a mathematical function *f*. Then
# 
# 
# <center>*piece_of_code(Dual(x,1))*</center>
# approximates
# <center>
# *Dual(f(x), f'(x))*</center> 

# We'll spend the rest of this section to demonstrate this statement.

# The most important class to verify this is for polynomials. Why? Because a lot of other functions can be approximated by polynomials:
# 
# 
# For example $$e^x \approx \sum_{k=0}^n \frac{1}{k!} x^k.$$ Take $n=4$: 
# $$e^x = 1 + x + \frac{1}{2} x^2 + \frac{1}{6} x^3 + O(x^4).$$

# ### Demonstrate for polynomials ###

# So a derivative of a monomial $x^k$ where $k$ is an integer number is $$(x^k)'=kx^{k-1}.$$

# So derivative of $x^2$ is $2x$ which at $x=3$ is $2*3=6$. With our dual numbers:

# In[ ]:


def square(x):
    return x * x
square(Dual(3,1))


# Great. (Recall that to read off the derivative, you need to look at the second number)
# 
# Let's try the derivavive of $x^3$, which at 2 is:

# In[ ]:


def cube(x):
    return x * x * x
cube(Dual(2,1))


# Fantastic!
# 
# Also important, for  a constant $c$ the derivative of $cx$ is just $c$. So for $c=2$:

# In[ ]:


def by2(x):
    return x * 2
by2(Dual(5,1))


# What about other functions that are not computed with polynomial approximation? Example - square root. Square root is computed by Newton formula. To compute the square root of $y$, we seek the fixed point of the function $$f(x) = \frac12 x + \frac12 \frac yx$$. I.e. the square root of $y$ is $x_0$ such that
# $$x_0 = \frac12 x_0 + \frac12 \frac y{x_0}$$
# How do you compute an approximation to a fixed point? Iterate the function until the results are close to each other. Indeed, if we build a sequence $x_0, x_1, ..., $ by $x_{i+1} = f(x_{i})$ and we have for some $n$ that $x_{n+1}\approx x_n$ then substituting $f(x_n)\approx x_n$ which means that $x_n \approx \sqrt{y}$ which is what we are trying to achieve.

# In[ ]:


EPS = 10E-12 # arbitrary accuracy

def next_iter(xn, ysq):
    return (xn + ysq / xn) * 1/2

def custom_sqrt(ysq):
    xnext, xprev = ysq, 0
    while abs(xnext * xnext - xprev * xprev) > EPS * abs(ysq):
        xnext, xprev = next_iter(xnext, ysq), xnext
    return xnext


# Let's check that our function works as intended

# In[ ]:


custom_sqrt(4)


# What should we expect for the derivative? Recall the computation of the derivative of square root:
# $$ (\sqrt{x})' =(x^\frac{1}{2})'=\frac{1}{2} x^{\frac{1}{2} - 1} = \frac{1}{2 \sqrt{x}}.$$

# In[ ]:


custom_sqrt(Dual(4,1))


# Just the result we were expecting!

# ## <a name="real"></a> Section 3. How to create an automatic differentiation system?

# Unfortunately, this is as far as our framework can go. If we tried this on a real life Python function:

# In[ ]:


from math import sqrt
sqrt(Dual(4,1))


# This is because many Python function are written in C and don't operate well on custom objects such as our Dual.

# ### So how does one implement automatic differentiation?

# - Operator overloading/Templates/Generify - this works well for statically typed languages. Esp. for Scala with its' powerful `implicits` mechanism.
# - Source to source translation - programmatically inspect the code and replace the computations of floats with computations of derivatives. Google's [Tangent](https://github.com/google/tangent) is one such attempt. It uses `Autograd` and `Tensorflow`'s eager mode of computation to achieve its goals.
# - Write dedicated automatic differentiation framework:
#     - We can add hooks for derivatives that we know. For instance $(e^x)'=e^x$, it is inefficient to recompute it. In fact, one can say that our toy autodiff is such a framework with hooks only for arithmetic operations. That's why we had to go to all that trouble to implement a custom square root. 
#     - Can build a function from basic building blocks if our framework is rich enough. Example from `Pytorch` with forward (computation of the function) and backward (computation of the derivative) methods. Also `Tensorflow`, `Theano` and every deep learning framework out there. And of course `Autograd`.
# - `Autograd` goes further than the deep learning frameworks - it swaps numpy with it's own extended version allowing to seamlessly differentiate numpy code without hassle.

# You can use Autograd to compute derivatives of ```custom_sqrt```

# In[ ]:


import autograd
grad_custom_sqrt = autograd.grad(custom_sqrt)
grad_custom_sqrt(4.)


# Of course, there are no miracles here. Trying to differentiate Python's ```math.sqrt``` fill fail:

# In[ ]:


import math
grad_math_sqrt = autograd.grad(math.sqrt)
try:
    grad_math_sqrt(4.)
except:
    import traceback
    traceback.print_exc(limit=1)


# Instead, we should use the wrapper around ```numpy``` to achieve the desired result:

# In[ ]:


import autograd.numpy as np
autograd.grad(np.sqrt)(4.)


# For Pytorch, the situation is similar, we could use our ```custom_sqrt``` to build graph in Pytorch (with version 0.4 API improvements):

# Create a pytorch tensor:

# In[ ]:


import torch
x = torch.tensor(4., requires_grad=True)
x


# Build the graph with ```custom_sqrt``` and differentiate it:

# In[ ]:


graph = custom_sqrt(x)
graph.backward()


# And the derivative is...

# In[ ]:


x.grad


# ## <a name="qa"></a> Section 4. Questions and Exercises.

# 1. How to establish that $ Dual(x,1)=Dual(\frac{1}{x}, -\frac{1}{x^2})$ or in math notation $\frac{1}{x+\varepsilon}=\frac{1}{x}-\frac{1}{x^2}\varepsilon$? There are two ways, in fact. One is to use the derivative of the function $f(x)=\frac{1}{x}$. The more satisfying way is purely algebraic - by solving an equation. We need to find two real numbers $x_1, x_2$ such that: $$ (x+\varepsilon)(x_1 + x_2\varepsilon)= 1 + 0 \cdot \varepsilon. $$
#     
# 1. The approximation $e^x \approx 1 + x + \frac{x^2}{2} + \frac{x^3}{3!}+\frac{x^4}{4!}$ breaks down pretty quickly when $x$ is large enough. To compute the exponential for large $x$, we can employ the following procedure. Use $x$'s binary expansion to write it as a sum of powers of two and a small remainder:
# $$
# x = \sum_{b_i> -5} 2^{b_i} + y,
# $$
# where $b_i$'s are non-zero bits of the binary expansion of $x$. We chose to terminate our expansion at -5 in an arbitrary fashion. We then use $e^{a+b}=e^{a}e^b$ to write 
# $$
# e^x = e^y\big( e^{2^{b_m}}..e^{2^{b_0}}\big).
# $$
# We can then apply the polynomial approximation for $e^{y}$ (in fact $e^{y}\approx 1+y$ works great). Verify that when implemented like this with say $compute\_exp$ then $compute\_exp(Dual(x,1))$ produces the correct derivative.
# 
# 1. a. Is our toy-autodiff approach a "forward mode" or "backward mode" differentiation? See [Wikipedia article](https://en.wikipedia.org/wiki/Automatic_differentiation) for defintions.
# 
#     b. Why would ```custom_sqrt``` be impossible in Tensorflow (or Tensorflow's default, original mode) and definitely not in Theano.
#     
#     c. If you familiar with either of those rewrite a compromise version of ```custom_sqrt```.
# 
# 1. Note how we were always using ```Dual(a,b) * c``` for multiplication by a scalar, which looks a little bit awkward. What is Python magic method do we need to implement to make ```c * Dual(a,b)``` work?
#     
# 1. Prove that for a polynomial $P(x)=\sum a_n x^n$ and your favorite method to compute polynomials the identity $P(Dual(x,1)) = Dual(P(x), P'(x))$ holds. 
#     This will help to explain why the following happens:
# 

# In[ ]:


from sympy.abc import x
def func(var):
    return var * var * var * 3 - var * var * 5
func(Dual(x,1))


# 6.Another interesting point to think about is many-variable functions. After all, neural networks have many millions of parameters. For concreteness, take the two-variable function below and compute it's gradient.

# In[ ]:


def two_var_func(x,y):
    return y/custom_sqrt(x*x + y*y)


# There are two ways to go about this: either compute the directional derivatives or use two-dimensional perturbations correctly. If you don't want to rewrite the ```Dual``` class, you can use ```numpy``` for vector perturbations. Of course at this point our approach will deviate sharply from the deep learning package's way.

# ## The end ##
# ### Hope you enjoyed, comments and questions are welcome. ###
