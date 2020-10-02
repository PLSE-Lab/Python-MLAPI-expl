#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import numpy as np
import math
import pandas as pd
from sympy.solvers import solve
from sympy import Symbol, exp
import matplotlib.pyplot as plt

print(os.listdir("../input"))
import warnings
warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.


# **Abstract**
# 
# In this kernel I will cover numerical method with using Numerical Methods for Engineers by chapra. We will cover each topic first theoretically and after that we will examine algorithm of these topics in python.
# 
# <font color='Black'>
# <br>Content:
# * [Truncation Erros and Taylor Series](#1)
# * [Graphical Method](#2)
# * [Bisection Method](#3)
# * [The False Position Method](#4)    
# * [Newton Raphson Method](#5)
# * [Navie Gauss Elemination](#6)
# * [Gauss-Sidel Method](#7)
#     
# <font color='Black'>
# <br>Topic will be cover later
# * Optimization
# * Curve Fitting
# * Numerical Differentiation And Integration
# * Ordinary Differential Equations    

# 
# 

# **Truncation Erros and Taylor Series**
# 
# Truncation errors are those that result from using a approximation in place of an exact matematical produne. For example, derivative of veclocity of a object by finite-divided-difference equation of form
# 
# $\frac{dv}{dt}\cong \frac{\Delta v}{\delta t}= \frac{\nu(t_{j+1})-\nu(t_{j})}{t_{j+1}-t_{j}}(1)$ .
# 
# Now turn to mathematical formulation that is used widely in numerical methods to express functions in an approximate fashion- the Taylor series.
# 
# 
# 

# **The Taylor Series**
# 
# Taylor theorem is of great value in the study of numerical methods. It provides a means to predict function at one point in terms of the function value and its derivatives. In matematical formulation, 
# 
# $f(x)=f(a)+f'(a)(x-a)+\frac{f''(a)(x-a)^{2}}{2!}+.........+\frac{f^{n}(a)(x-a)^{n}}{n!}+R_{n} (1)$
# 
# Where $R_{n}$ is called remainder or truncation error and accounts for all terms from $n+1$ to infinity
# 
# $R_{n}=\int_{a}^{x}\frac{(x-t)^{N}}{n!}f^{n+1}(t)dt (2)$ 
# and truncation error can also have second represantation 
# 
# $R_{n}=\frac{f^{n+1}(\zeta)}{(n+1!)}(x-a)^{n+1} (3)$ 
# where $\zeta$ is a value that lies intarvel (a,x)
# 
# Return to Eq.1 and applied Taylor expansion. Then, term by term examination of Taylor expansion give to perfect estimination about the function's value at given a point.
# 
# $f(x_{i+1})\cong f(x_{i})   (4)$ 
# 
# Eq.4 is a zero order approximation. For better estiminate, term order can be increase. This condition depend on the function which is examine.

# **Algorithm Process for Taylor Series Approximation**

# In[ ]:


N = 5 # Significant Number  

#True Value of the function 
tv = math.exp(-1*(0.2**2)/3)
print("True value: \n{}".format(tv))

#Tolerance of the system - Iteration continue until Tolerance error is neglectable for our algorithm.-
eps=0.5*(10)**(2-N)
print("Tolerance:\n{}".format(eps))


# In[ ]:


# Analytic Defination of function 
def expp(x,term):
    expa=0
    for n in range(term):
        expa=expa+((-1)**n)*(x**(2*n))/((3**n)*math.factorial(n))
    return expa


# In[ ]:


#Print nth order approximation term of Taylar expansion
apt=[] # create empty list for approximation terms
for i in range(0,100): # 0 to 100 term approximation of Taylor expansion
    apt.append(expp(0.2,i)) # Append term approximation to empty list (apt)
# Print fist to fourth term approximation (This process can be extend if users want to)     
print("First term approximation: \n{}".format(apt[1]))
print("Second term approximation: \n{}".format(apt[2]))
print("Third term approximation: \n{}".format(apt[3]))
print("Fourth term approximation: \n{}".format(apt[4]))


# In[ ]:


# Unleash the Itreation's Beast  

apt1=[] #List of Current Value
ept=[] #True Relative error
epa=1 #Approximate Relative Error
itr=[] #Iteration numbers
epa1=[] # List of Approximate Percent Relative Error terms
while epa > eps:
    for j in range(0,15):
        epa=abs((apt[j+1]-apt[j])/apt[j+1])*100 # Calculate Relative Error with current and previous value
        itr.append(j)
        ept.append(abs(((tv-apt[j])/tv)))
        print("Term",j+1)
        print("Approximate relative error",epa)
        print("True relative error",ept[j])
        print("Previous Value",apt[j])
        print("Current Value",apt[j+1])
        apt1.append(apt[j+1])
        epa1.append(epa)
        # if tolerance error bigger than relative error which is calculated in iteration then break the iteration
        if eps > epa :
            break


# In[ ]:


# Create table from variables and numbers in previous iteration
table = pd.DataFrame({'Iteration':itr,'True Relative Error':ept,'Current Value': apt1, 'Approximate relative error':epa1})
print(table.ix[1:]) # The first iteration not count because algorithm has no previous value in inital state
print(table.loc[:,"Current Value"]) # 


# In[ ]:


# Figure of analytic solution and Taylor approximations

x = np.linspace(-1, 2) # range of x-axis
plt.figure(figsize=(15,7))

plt.plot(x, np.exp(-1*(x**2)/3), label="Analytic Solution", color="blue", marker="o", markerfacecolor="red", linewidth=2, markersize=5) # Analytic solution
plt.plot(x, expp(x, 1), label="First Term Approximation", color="orange", marker="o", markerfacecolor="black", linewidth=2, markersize=5) # First term approximation
plt.plot(x, expp(x, 2), label="Second Term Approximation", color="yellow", marker="o", markerfacecolor="green", linewidth=2, markersize=5) # Second term approximation
plt.plot(x, expp(x, 3), label="Third Term Approximation", color="red", marker="o", markerfacecolor="blue", linewidth=2, markersize=5) # Third Term approximation
plt.plot(x, expp(x, 4), label="Fourth Term Approximation", color="green", marker="o", markerfacecolor="red", linewidth=2, markersize=5) # Fourth Term approximation
plt.legend()
plt.show()


# **Conclusion**
# 
# The third order term approximation enought to give value of function which is close to true value came from analytic solution. When torrelance error is decreased(from $0.005$), iteration can be continue and it can be more precise value of function but this process is not necessary because computational time cost( True value : $0.9867551618071957$).

# **Graphical Method**
# 
# A simple method for obtaning an estimate of the root equation $f(x)=0$ to make plot of the function and observe where it crosses the $x$- axis.
# 
# In general same sign lead to no roots or an even numbers of roots oppesite sign lead to odd numbers of roots.
# 
# Graphical techniques are of limited practical value because they are not precise. However,
# graphical methods can be utilized to obtain rough estimates of roots. These estimates
# can be employed as starting guesses for numerical methods

# In[ ]:


# Define the function 
def f(x):
    return x**3-4*x**2+3*x+12


# In[ ]:


# Prediction for x axis. Each prediction done after examine preveous graph. 
x1 = np.linspace(-10, 10)
x2 = np.linspace(-2.5, 2.5)
x3 = np.linspace(-2, -1)
x4 = np.linspace(-1.5, -1)
x5 = np.linspace(-1.3, -1.1)
x6 = np.linspace(-1.28, -1.25)


# In[ ]:


fig = plt.figure(figsize=(20,10))

ax1 = fig.add_subplot(231)
ax1.plot(x1, f(x1), color="blue", marker="o", markerfacecolor="red", linewidth=2, markersize=5)
plt.xlabel("x values")
plt.ylabel("f(x)")

ax2 = fig.add_subplot(232)
ax2.plot(x2, f(x2),color="blue", marker="o", markerfacecolor="red", linewidth=2, markersize=5)
plt.xlabel("x values")
plt.ylabel("f(x)")

ax3 = fig.add_subplot(233)
ax3.plot(x3, f(x3),color="blue", marker="o", markerfacecolor="red", linewidth=2, markersize=5)
plt.xlabel("x values")
plt.ylabel("f(x)")

ax4 = fig.add_subplot(234)
ax4.plot(x4, f(x4), color="blue", marker="o", markerfacecolor="red", linewidth=2, markersize=5)
plt.xlabel("x values")
plt.ylabel("f(x)")

ax5 = fig.add_subplot(235)
ax5.plot(x5, f(x5), color="blue", marker="o", markerfacecolor="red", linewidth=2, markersize=5)
plt.xlabel("x values")
plt.ylabel("f(x)")

ax6 = fig.add_subplot(236)
ax6.plot(x6, f(x6), color="blue", marker="o", markerfacecolor="red", linewidth=2, markersize=5)
plt.xlabel("x values")
plt.ylabel("f(x)")
plt.show()


# In[ ]:


# result of the function with the estimate value from final graph
print(f(-1.253))


# **Conclusion**
# 
# The Graphical method is not precise to decet equation of root but it give to quick and practical guess.  

# **The Bisection Method**
# 
# In general, if f(x) is real and continous interval from $x_{l}$ (lower value in x-axis) to $x_{u}$ (upper value in x-axis) and $f(x_{l})$ $f(x_{u})$ have opposite signs, $f(x_{l})f(x_{u})<0$ then, there is a at least one real root between $x_{l}$ and $x_{u}$.

# In[ ]:


# Define the function
#Same function in graphical method, so methods can be compared
def f(x):
    return x ** 3 - 4 * x ** 2 + 3 * x + 12


# In[ ]:


# Number of significant figures and error criterion
n = 3
eps = 0.5 * (10 ** (2 - n)) # Tolerance Error

# Choose lower and upper value for function
xu = 4
xl = -3

# Create Empty List for root values
xr_l = []  # Empty list of roots
epa_l = []  # Empty list of Approximation Error
itr_l = []  # Empty list of Iteration number

# An estimate of root xr is determined by
xr_l.append((xu + xl) / 2)  # First iteration begin this value


# In[ ]:


# Unleash The Beast
for i in range(0, 30):  # Iterate 0 to 30
    if f(xl) * f(xr_l[i]) < 0:  # Step 1: if this condition hold ;
        xu = xr_l[i]  # Then equal xu(upper value) value to new xr(root value)
        xr = ((xu + xl) / 2)
        xr_l.append(xr)  # append root values to empty list of root
        epa = abs((xr_l[i] - xr_l[i - 1]) / xr_l[i])  # Calculate Tolerance Error from new root value and old root value
        epa_l.append(epa)  # append tolerance error to empty list of Approximation error
        itr_l.append(i)  # append iteration numbers to empty list of iteration list

    if f(xl) * f(xr_l[i]) > 0:  # Step 2: if this condition hold ;
        xl = xr_l[i]  # Then equal xl(lower value) value to new xr(root value)
        xr = ((xu + xl) / 2)
        xr_l.append(xr)  # append root values to empty list of root
        epa = abs(
            (xr_l[i] - xr_l[i - 1]) / xr_l[i]) * 100  # Calculate Tolerance Error from new root value and old root value
        epa_l.append(epa)  # append tolerance error to empty list of Approximation error
        itr_l.append(i)  # append iteration numbers to empty list of iteration list
    # Terminate Criteria for Algorithm
    elif eps > epa != 0:  # Step 3: else if tolerance error's value bigger than(and also not equal zero) approximation
        break             # error's value which is calculated in step 1 or step 2 then stop to iteration


# In[ ]:


table = pd.DataFrame({'Iteration': itr_l, 'Root Value': xr_l[1:], 'Approximation Error': epa_l}) # xr_l[1:] used because xr_l has one more
#term than others list(Remember, xr_l was not begin as empty list.)
print(table)


# In[ ]:


# result of the function with the estimate value from final iteration
print(f(-1.252563))


# **Conclusion**
# 
# Bisection Method result has same precision with Graphical method. However, Graphical methond include human factor(choosing the value from graph with using eyes) and this situation can create mistake for choosing a suitable root.Bisection method use only computer algorithm to obtain more reliable result. 

# **The False-Position Method**
# 
# In this method , $f(x_{l})$ and $f(x_{u})$ is joint by straight line. The intersection of this line with $x-axis$ represent an improved estimate of the root. From the trigonometry in graph, 
# $\frac{f(x_{l})}{x_{l}}=\frac{f(x_{u})}{x_{u}-x_{l}}$. Then rearrange,
# 
# $x_{r} = x_{u}-\frac{f(x_{u})(x_{l}-x_{r})}{f(x_{l})-f(x_{u})}(6)$.
# 
# This is the false-position formula. The value of $x_{r}$ computed with Eq.6 then replaces whichever of two initial guesses, x_{l} or x_{u}, yields a function value with same sign as $f(x_{r})$.
# 
# 

# In[ ]:


# Define Function
def f(x):
    return (np.exp(-1 * 0.5 * x) * (4 - x)) - 2


# Number of significant figures and error criterion
n = 4
eps = 0.5 * 10 ** (2 - n)
print("Percent Tolerance: \n{}".format(eps))


# In[ ]:


# Analytic Solution of the function(this is used for comparing the algorithm's value for function)
x = Symbol('x')
As = solve(exp(-1 * 0.5 * x) * (4 - x) - 2, x)

print("Solution of function: \n{}".format(As))


# In[ ]:


# Graph for guessing to initial value(x lower and x upper)
x = np.linspace(-1, 3)

plt.figure(figsize=(16,8))
plt.plot(x, f(x), marker='o', markerfacecolor="r")
plt.xlabel("x")
plt.ylabel("Function")
plt.show()


# In[ ]:


# Two initial guesses, they choose from graph
x_u = 1  # upper point
x_l = 0  # lower point

# Iteration Algorithm
itr = []  # list of iteration number
x_r = []  # list of root
epa = 1  # approximation error(for start to iteration)
epa1 = []  # list approximation error
for i in range(0, 20):
    if f(x_l) * f(x_u) < 0: # Step 1
        x_r.append(x_u - ((f(x_u) * (x_l - x_u)) / (f(x_l) - f(x_u)))) # Eq.6
        epa = (abs((x_r[i] - x_r[i - 1]) / x_r[i])) * 100 # Approximation Error Calculation
        epa1.append(epa) # Append approximation error values to empty list of approximation error
        itr.append(i) # Append iteration number to empty list of iteration
        x_u = x_r[i]

    if f(x_l) * f(x_u) > 0: # Step 2

        x_r.append(x_u - ((f(x_u) * (x_l - x_u)) / (f(x_l) - f(x_u)))) # Eq.6
        x_l = x_r[i]
        epa = (abs((x_r[i] - x_r[i - 1]) / x_r[i])) * 100 # Approximation Error Calculation

    elif eps > epa != 0.0: # Step 3 : Stop iteration when tolernace error bigger than approximation error
        break


# In[ ]:


table = pd.DataFrame({'Iteration': itr, 'Approximation Percent Error': epa1, ' Estimate': x_r})
print(table.ix[1:]) # table.ix[1:] is used because first row is not includue iteration process


# In[ ]:


# Error of False-position Method
err = (abs(np.asarray(As) - np.asarray(x_r[6])) / np.asarray(As)) * 100 # x_r[6] represent final(and also optimum) value of the iteration
print("Error Percent of Newton-Raphon Method \n{}".format(err))


# **Conculusion**
# 
# The error of False-position method is very tiny but in scientific calculation, this kind of error can not negligible. Moreover, there is a pitfalls of the false-position method, it can give performs poorly for some functions. 
# 
# 

# **Newton-Raphson Method**
# 
# In the previous methods, the root is located within an inverval prescribed by a lower and an upper bound. In constrast, the Newton-Raphson method is based on formulas that require only a single starting value of $x$.
# If the initial guess at the root is $x_{i}$ a tangent can be extended from the point $[x_{i},f(x_{i})]$. The point where this tangent crosses the $x$ axis usually represents an improved estimate of the root.
# 
# Consider a Taylor series expansion around the estimate $x_{i}$ : 
# 
# $f(x_{i+1}=f(x_{i})+f'(x_{i})(x_{i+1}-x_{i})+R_{1} (1)$
# 
# Truncating the series after the first
# derivative term
# 
# $f(x_{i+1}=f(x_{i})+f'(x_{i})(x_{i+1}-x_{i}) (2).$
# 
# The new estimate $x_{i+1}$ is located at point $f(x_{i+1})=0$ then,
# 
# $0=f(x_{i})+f'(x_{i})(x_{i+1}-x_{i}) (3)$ or,
# 
# $x_{i+1}=x_{i}-\frac{f(x_{i})}{f'(x_{i})} (4)$
# 
# Eq.4 is called Newton-Raphson Formula.
# 

# In[ ]:


# Define Function
def f(x):
    return (math.exp(-1 * 0.5 * x) * (4 - x)) - 2


# Derivation of The Function

def fd(x):
    x = Symbol('x')
    f = exp(-1 * 0.5 * x) * (4 - x) - 2
    return f.diff(x)


print("Derivation of Function f(x): \n{}".format(fd(x)))


# In[ ]:


def fd(x):  # fd function define for numerical calculation from analytic solution output
    return -0.5 * (-x + 4) * exp(-0.5 * x) - exp(-0.5 * x)


# In[ ]:


# Initial Guess
x_f = 1

# Iteration Algorithm
itr = []  # list of iteration number
x_s = []  # list of root
epa = 1  # approximation error
epa1 = []  # list approximation error
while epa > eps:
    for i in range(0, 10): # Step 1
        x_s.append(x_f - f(x_f) / fd(x_f)) # Eq.4
        epa = abs((x_s[i] - x_s[i - 1]) / x_s[i]) * 100 # Calculation of Approximation Error
        epa1.append(epa) # Append approximation error to empty list 
        itr.append(i) # Append iteration number to empty list

        x_f = x_s[i]

        if eps > epa != 0.0: # Stop if tolerance error is bigger than approximation error
            break


# In[ ]:


table = pd.DataFrame({'Iteration': itr, 'Approximation Percent Error': epa1, ' Estimate': x_s})
print(table.ix[1:]) # table.ix[1:] is used because first row is not includue iteration process


# In[ ]:


# Error of Newton-Raphson Method
err = (abs(np.asarray(As) - np.asarray(x_s[2])) / np.asarray(As)) * 100
print(" Percent Error of Newton-Raphson Method \n{}".format(err))


# **Conclusion**
# 
# The Newton-Raphson method's error value is acceptable for scientific calculation but same situations, it can performs poorly -specially multiple roots function- 
# 

# **Naive Gauss Elimination**
# 
# Gauss elimination include foward elimination and back substitution to solve linear equation.
# 
# 1. The equations were manipulated to eliminate one of the unknowns from the equations.
# The result of this elimination step was that we had one equation with one unknown.
# 2. Consequently, this equation could be solved directly and the result back-substituted
# into one of the original equations to solve for the remaining unknown.
# 
# This approach is designed to solve a general set of n equation:
# 
# $a_{11}x_{1}+a_{12}x_{2}+.......+a_{1n}x_{n}=b_{1}$
# 
# $a_{21}x_{1}+a_{22}x_{2}+.......+a_{2n}x_{n}=b_{2}  (1)$
#                         
# $a_{n1}x_{1}+a_{n2}x_{2}+.......+a_{nn}x_{n}=b_{n}$
# 
# * Forward Elimination of Unknown
# The fist phase of designed to reduce the system to upper triangular system. The initial step will be to eliminate the fist unknown, $x_{1}$ for the second through the nth equation. To do this multiply the first equation by $a_{21}/a_{11}$ then substract on the second equation.
# 
# $(a_{22}-\frac{a_{21}}{a_{11}})x_{2}+(a_{23}-\frac{a_{21}}{a_{11}}a_{13})x_{3}+.......+(a_{2n}-\frac{a_{21}}{a_{11}}a_{1n})x_{n}=b_{2} (2)$ 
# 
# or Eq.2 can be written more regularly
# 
# $a'_{22}x_{2}+a'_{23}x_{2}+.......+a'_{2n}x_{n}=b'_{2} (3)$
# 
# where the prime indicates that the elements have been changed from their orginal values. These process repeat until to reach nth equation in the system. Then the procedure is continued using remaining pivot equation to obtain the upper triangle form:
# 
# $a_{11}x_{1}+a_{12}x_{2}+.......+a_{1n}x_{n}=b_{1}$
# 
# $             a'_{22}x_{2}+.......+a'_{2n}x_{n}=b'_{2}  (4)$
#                         
# $                                  a_{nn}^{n-1}x_{n}=b_{n}^{n-1}$
# 

# In[ ]:


# Define matrix for coefficient
A = np.array([[-1, -2, 3],
               [5, 2, -1],
               [3, 10, 0.1]])
print("Coefficient Matrix: \n{}".format(A))


# Define matrix right side of equation
B = np.array([[-1],
               [1],
               [2]])
print("Right Side of Equation's Matrix: \n {}".format(B))


# In[ ]:


# Solve Linear System with linalg.solve command for comparison to algorithm's results
sol = np.linalg.solve(A,B)
print("Value of x1, x2, x3 \n{}".format(sol))

# Tolerance error
n = 4  # Number of significant figure
eps = 0.5 * (10 ** (2 - n))


# In[ ]:


# Forward Elimination

for i in range(0, 1): # first column
    for j in range(1, 3): # second row to third row
        B[j] = B[j]-(A[j, i]/A[i, i])*B[i]
        for k in range(2, -1, -1): # third row to first row
            #print(j, i, B)
            A[j, k] = A[j, k]-(A[j, i]/A[i, i])*A[i, k]
            print(j, k, i, A)

for i in range(1, 2): # second column
    for j in range(2, 3): # third row
        B[j] = B[j]-(A[j, i]/A[i, i])*B[i]
        for k in range(2, -1, -1): # third row to first row
            #print(j, i, B)
            A[j, k] = A[j, k]-(A[j, i]/A[i, i])*A[i, k]
            print(j, k, i, A)

print("Right Side of Equation's Matrix after Forward Substation: \n {}".format(B))
print("Coefficient Matrix After Forward Substation: \n{}".format(A))


# In[ ]:


# Back Substation
x3 = B[2]/A[2,2]
x2 = (B[1]-(x3*A[1, 2]))/A[1, 1]
x1 = (B[0]-(x2*A[0, 1]+x3*A[0, 2]))/A[0, 0]

print("Value of x1, x2, x3 \n {} \n {} \n {} ".format(x1, x2, x3))


# In[ ]:


# Error Calculation
erx1 = ((sol[0]-x1)/x1)*100 # Error percent of x1
erx2 = ((sol[1]-x2)/x2)*100 # Error percent of x2
erx3 = ((sol[2]-x3)/x3)*100 # Error percent of x3

print("Percent Error of x1:", erx1)
print("Percent Error of x2:", erx2)
print("Percent Error of x3:", erx3)


# **Conclusion**
# The error is acceptable  for scientific calculation. There is problem which I can not manage over it : When I use the 4x4 matrix, percent error of unknows give do not acceptable results. Therefore I used Gauss-Siedel method for examination the 4x4 matrix's unknowns. 

# **Gauss-Seidel Method**
# 
# Iterative or approximate methods provide an alternative to the elimination methods. Consider we will show this 3 by 3 system written in matrix form :
# 
# $[A]{X}={B}$ $(1)$
# 
# $a_{11}x_{1}+a_{12}x_{2}+a_{13}x_{3}=b_{1}$
# 
# $a_{21}x_{1}+a_{22}x_{2}+a_{23}x_{3}=b_{2}$  $(2)$
# 
# $a_{31}x_{1}+a_{32}x_{2}+a_{33}x_{3}=b_{3}$
# 
# The unknows can be arrange :
# 
# $x_{1}= \frac{b_{1}-a_{12}x_{2}-a_{13}x_{3}}{a_{11}}$   ;  $x_{2}= \frac{b_{2}-a_{21}x_{1}-a_{23}x_{3}}{a_{22}}$   ;  $x_{3}= \frac{b_{3}-a_{31}x_{1}-a_{32}x_{2}}{a_{33}}$  $(3)$
# 
# Then start with initial guess : $x_{1}^{0}=x_{2}^{0}=x_{3}^{0}=0$
# 
# The first iteration : $x_{1}^{1}= \frac{b_{1}}{a_{11}}$   ;   $x_{2}= \frac{b_{2}-a_{21}x_{1}^{1}}{a_{22}}$   ;   $x_{3}= \frac{b_{3}-a_{31}x_{1}^{1}-a_{32}x_{2}^{1}}{a_{33}}$  $(4)$
# 
# And this process will continue until tolerance error's value bigger than approximation error's value.
# 
# *  Improvement of Convergence by Relaxation
# 
# After each new value of $x$ is computed that value is modified by weighted average of the results of the previous and present iterations :
# 
# $x_{i}^{new}=\lambda x_{i}^{new} + (1-\lambda)x_{i}^{old}$ $(5)$
# 
# if $0<\lambda<1$ under relaxation designed to make a nonconvergent system converge.
# 
# if $1<\lambda<2$ over relaxation designed accelerate the convergence system of an already convergent system.
# 

# In[ ]:


# Define matrix for coefficient
cm = np.array([[-1, -2, 3, 10],
               [5, 2, -1, 0.5],
               [3, 10, 0.1, -0.1],
               [4, 4, -20, 4]])
print("Coefficient Matrix: \n{}".format(cm))

# Define matrix right side of equation
rm = np.array([[-1],
               [1],
               [2],
               [-3]])
print("Right Side of Equation's Matrix: \n {}".format(rm))


# In[ ]:


# Summon linalg.solve command for comparison with solution of algorithm
sol = np.linalg.solve(cm, rm)
print("Value of x1, x2, x3, x4: \n{}".format(sol))


# In[ ]:


# Tolerance error
n = 4  # Number of significant figure
eps = 0.5 * (10 ** (2 - n))


# In[ ]:


# Create empty list for unknown
x1 = []
x2 = []
x3 = []
x4 = []
# Define approximation errors of each unknown
epa1 = 1
epa2 = 1
epa3 = 1
epa4 = 1
# Initial guesses for Unknown
x2.append(0.000000000000000001)
x3.append(0.000000000000000001)
x4.append(0.000000000000000001)

itr = [] # List of iteration
epa1_l = [] # List of approximation percent error for x1
epa2_l = [] # List of approximation percent error for x2
epa3_l = [] # List of approximation percent error for x3
epa4_l = [] # List of approximation percent error for x4

# Unleash the beast
for i in range(0, 20): # Iteration 1 to 20

    x1.append((rm[1] - cm[1][1] * x2[i] - cm[1][2] * x3[i] - cm[1][3] * x4[i]) / cm[1][0]) # Second row used
    x2.append((rm[2] - cm[2][0] * x1[i] - cm[2][2] * x3[i] - cm[2][3] * x4[i]) / cm[2][1]) # Third row used
    x3.append((rm[3] - cm[3][0] * x1[i] - cm[3][1] * x2[i] - cm[3][3] * x4[i]) / cm[3][2]) # Fourth row used
    x4.append((rm[0] - cm[0][0] * x1[i] - cm[0][2] * x3[i] - cm[0][1] * x2[i]) / cm[0][3]) # First row used
    epa1 = abs(((x1[i] - x1[i - 1]) / x1[i])) * 100 # x1's approximation error
    epa2 = abs((x2[i] - x2[i - 1]) / x2[i]) * 100 # x2's approximation error
    epa3 = abs((x3[i] - x3[i - 1]) / x3[i]) * 100 # x3's approximation error
    epa4 = abs((x4[i] - x4[i - 1]) / x4[i]) * 100 # x4's approximation error
    itr.append(i)

    epa1_l.append(epa1)
    epa2_l.append(epa2)
    epa3_l.append(epa3)
    epa4_l.append(epa4)
    
    if eps > epa1 != 0.0 and eps > epa2 != 0.0 and eps > epa3 != 0.0 and eps > epa4 != 0.0: # Break to Iteration when approximation 
        #errors of unknowns smaller than tolerance error
        break


# In[ ]:


table = pd.DataFrame({'Iteration': itr, 'x1': x1, 'x2': x2[1:], 'x3': x3[1:], 'x4': x4[1:] }) # Crate table with iteration number and unknowns
print(table)


# In[ ]:


# Error percent of unknowns (between linalg.solve-sol[]- and algorihm -x_i- result )
erx1 = ((sol[0]-x1[max(itr)])/x1[max(itr)])*100 # Error percent of x1 
erx2 = ((sol[1]-x2[max(itr)])/x2[max(itr)])*100 # Error percent of x2
erx3 = ((sol[2]-x3[max(itr)])/x3[max(itr)])*100 # Error percent of x3
erx4 = ((sol[3]-x4[max(itr)])/x4[max(itr)])*100 # Error percent of x4

print("Percent Error of x1:", erx1)
print("Percent Error of x2:", erx2)
print("Percent Error of x3:", erx3)
print("Percent Error of x4:", erx4)


# In[ ]:


#Improvement of convergence by Relaxation
l = 1.5 # Lambda Value

# Unleash the beast
for i in range(0, 20): # Iteration 1 to 20

    x1.append((rm[1] - cm[1][1] * l*x2[i] - cm[1][2] * l*x3[i] - cm[1][3] * l*x4[i]) / cm[1][0]) # Second row used
    x2.append((rm[2] - cm[2][0] * l*x1[i] - cm[2][2] * l*x3[i] - cm[2][3] * l*x4[i]) / cm[2][1]) # Third row used
    x3.append((rm[3] - cm[3][0] * l*x1[i] - cm[3][1] * l*x2[i] - cm[3][3] * l*x4[i]) / cm[3][2]) # Fourth row used
    x4.append((rm[0] - cm[0][0] * l*x1[i] - cm[0][2] * l*x3[i] - cm[0][1] * l*x2[i]) / cm[0][3]) # First row used
    epa1 = abs(((l*x1[i] - l*x1[i - 1]) / l*x1[i])) * 100 # x1's approximation error
    epa2 = abs((l*x2[i] - l*x2[i - 1]) / l*x2[i]) * 100 # x2's approximation error
    epa3 = abs((l*x3[i] - l*x3[i - 1]) / l*x3[i]) * 100 # x3's approximation error
    epa4 = abs((l*x4[i] - l*x4[i - 1]) / l*x4[i]) * 100 # x4's approximation error
    itr.append(i)

    epa1_l.append(epa1)
    epa2_l.append(epa2)
    epa3_l.append(epa3)
    epa4_l.append(epa4)
    if eps > epa1 != 0.0 and eps > epa2 != 0.0 and eps > epa3 != 0.0 and eps > epa4 != 0.0: # Break to Iteration when approximation 
        #errors of unknowns smaller than tolerance error
        break


# In[ ]:


table = pd.DataFrame({'Iteration': itr[9:], 'x1': x1[9:], 'x2': x2[10:], 'x3': x3[10:], 'x4': x4[10:] }) # Crate table with iteration number and unknowns([9:] and [10:] used for writing the second iteration)
print(table)


# **Conclusion**
# 
# the Gauss-Seidel technique's error value is acceptable for scientific calculation and also with $\lambda$ value, iteration number can be decrease. 
