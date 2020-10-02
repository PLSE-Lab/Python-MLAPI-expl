#!/usr/bin/env python
# coding: utf-8

# ## MATH 152 Lab 2 
# # 
# # Names of team members: Brett Miller, Luke Smith, Garett Morrison, Eric Anton

# In[ ]:


from sympy import *
from sympy.plotting import (plot, plot_parametric,plot3d_parametric_surface, plot3d_parametric_line,plot3d)


# #1 

# In[ ]:


from sympy import *
from sympy.plotting import (plot, plot_parametric,plot3d_parametric_surface, plot3d_parametric_line,plot3d)
#Start of Part A
x = symbols('x')
p1 = plot((x,(x,0,3)),show=False)
p2 = plot((x*E**(1-x/2),(x,0,3)),show=False,line_color='firebrick')
p1.title = "Problem 1"
p1.xlabel = 'X-Axis'
p1.ylabel = 'Y-Axis'
p1.legend = True
p1.append(p2[0])
p1.show()
#%reset -f


# In[ ]:


#Start of Part B
#Intersections = [0,0] & [2,2]
from sympy import *
import sympy as sy
import math
def e():
    return math.e
def f(x): return x
def g(x): return x*e()**(1-x/2)
x = sy.Symbol('x')
def part1(): 
    return sy.integrate(f(x),(x,0,2))
def part2():
    return sy.integrate(g(x),(x,0,2))
def pi():
    return math.pi
def FinalAnswer():
    return (pi()*part1()**2)-(pi()*(part2())**2)
print(abs(FinalAnswer()))
get_ipython().run_line_magic('reset', '-f')


# In[ ]:


#Start of Part C
from sympy import *
import sympy as sy
import math
def e():
    return math.e
def f(x): return x
def g(x): return x*e()**(1-x/2)
x = sy.Symbol('x')
get_ipython().run_line_magic('reset', '-f')


# In[ ]:


#Start of Part D
from sympy import *
import sympy as sy
import math
def e():
    return math.e
def f(x): return x
def g(x): return x*e()**(1-x/2)
x = sy.Symbol('x')
get_ipython().run_line_magic('reset', '-f')


# #2

# In[ ]:


#Start of Part 2(A)
from sympy import *
import sympy as sy
from sympy.plotting import (plot, plot_parametric,plot3d_parametric_surface, plot3d_parametric_line,plot3d)
x = sy.Symbol('x')
h = sy.Symbol('h')
r = sy.Symbol('r')
R = sy.Symbol('R')
#Done setting fixed constants
def f(x):
    return (h/R-r)*(x-r)
def g(x):
    return h
# Setting constants equal to respective values
r = 2 
R = 4
h = 3
# Plotting in first quadrant
p1 = plot((f(x),(x,0,2)),(g(x),(x,0,2)))
get_ipython().run_line_magic('reset', '-f')


# In[ ]:


#Start of Part 2(B)
from sympy import *
import sympy as sy
x = sy.Symbol('x')
h = sy.Symbol('h')
r = sy.Symbol('r')
R = sy.Symbol('R')
#Done setting fixed constants
def f(x):
    return (h/R-r)*(x-r)
def g(x):
    return h
get_ipython().run_line_magic('reset', '-f')


# In[ ]:


#Start of Part 2(C)
from sympy import *
import sympy as sy
x = sy.Symbol('x')
h = sy.Symbol('h')
r = sy.Symbol('r')
R = sy.Symbol('R')
A = sy.Symbol('A') #Will be used as a variable for the x-intercept of f
B = sy.Symbol('B') #Will be used as a variable for the intersection of f & g
#Done setting fixed constants
def f(x):
    return (h/R-r)*(x-r)
def g(x):
    return h
#Objective: Find the volume for region below Line A-B, and above the x-axis
#           rotated about the line x = R (simplify final answer)
get_ipython().run_line_magic('reset', '-f')


# In[ ]:


#Start of Part 2(D)
def my_Answer():
    return """
    Psuedo multi-block comment in python (only other option is to
    comment each line individually.)
    """
my_Answer()
get_ipython().run_line_magic('reset', '-f')


# #3

# In[ ]:


#Start of Problem 3
from sympy import *
import sympy as sy
from sympy.plotting import (plot, plot_parametric,plot3d_parametric_surface, plot3d_parametric_line,plot3d)
import matplotlib.pyplot as plt
import numpy as np
# X and Y Vectors
x = np.array(range(10))
y = 1/2*(1-x**2)

#Creating the plot
plt.plot(x,y,label='y=1/2(1-x**2)')
#Plot title
plt.title('Problem 3 Graph')
#X and Y labels
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
#Grid for the plot
plt.grid(alpha=.4,linestyle='--')
#Adding plot legend
plt.legend()
#Show the plot
plt.show()
get_ipython().run_line_magic('reset', '-f')

