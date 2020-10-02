#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#v2: moved most inline math expressions into equation environments
#v3: added discussion of cylindrical and spherical coordinates, corrected formulas involving charge sign
#v4: typos corrected


# I collect some details concerning the geometry of the helix for charged particles in a constant magnetic field and discuss why the coordinate transformations used in the DBSCAN Benchmark kernel https://www.kaggle.com/mikhailhushchyn/dbscan-benchmark are useful.

# **Cartesian coordinates**
# 
# We fix the following notation:
# 1. Particle rest mass $m_0$ and electric charge $q$.
# 2. $m=\gamma m_0$ with relativistic factor $\gamma$ (considered constant).
# 3. Coordinate system $(x,y,z)$.
# 4. Magnetic field $\vec{B}=Be_z$.
# 5. Initial coordinates $x_0=y_0=z_0=0$ (for simplicity, see comment below).
# 6. Initial velocities $v_{x0}, v_{y0}, v_{z0}$ and initial momenta $p_{x0}, p_{y0}, p_{z0}$ with $p_{i0}=m v_{i0}$ for $i=x,y,z$.
# 7. Longitudinal momentum $p_\parallel = p_{z0}$, transversal momentum 
# \begin{equation*}
# p_\perp = \sqrt{p_{x0}^2+p_{y0}^2},
# \end{equation*}
# and total momentum 
# \begin{equation*}
# p=\sqrt{p_\parallel^2+p_\perp^2}.
# \end{equation*}
# 
# We define:
# \begin{align*}
# \omega &= \frac{qB}{m}\\
# R&= \frac{p_\perp}{qB}
# \end{align*}
# and an angle $\phi_0$ by
# \begin{align*}
# \cos\phi_0&=-\frac{p_{y0}}{p_\perp}\\
# \sin\phi_0&=\frac{p_{x0}}{p_\perp}.
# \end{align*}
# The system of differential equations
# \begin{align*}
# m\ddot{x} &= qB\dot{y}\\
# m\ddot{y} &= -qB\dot{x}\\
# m\ddot{z} &= 0
# \end{align*}
# has the helix solution
# \begin{align*}
# x(t) &= R(\cos(\phi_0-\omega t)-\cos(\phi_0))\\
# y(t) &= R(\sin(\phi_0-\omega t)-\sin(\phi_0))\\
# z(t) &= R\frac{p_\parallel}{p_\perp}\omega t.
# \end{align*}
# If the initial coordinates are non-zero, we just add $x_0$, $y_0$ and $z_0$ to these expressions.
# 
# Projected onto the $x$-$y$-plane, the helix forms a circle of radius $|R|$ with center at the point 
# \begin{equation*}
# -R(\cos\phi_0,\sin\phi_0).
# \end{equation*}
# The motion in $z$-direction is of constant velocity.

# Using
# \begin{align*}
# \cos^2(\alpha)+\sin^2(\alpha)&=1\\
# \cos(\alpha)\cos(\beta)+\sin(\alpha)\sin(\beta)&=\cos(\alpha-\beta),
# \end{align*}
# we have
# \begin{align*}
# x^2+y^2 &= 2R^2(1-\cos(\omega t))\\
# &\approx R^2\left(\omega^2t^2-\frac{1}{12}\omega^4t^4  \right),
# \end{align*}
# where the approximation in the second line follows from Taylor's formula and is valid for small $\omega t$.
# 
# We define a new coordinate
# \begin{equation*}
# z_2=\frac{z}{\sqrt{x^2+y^2}}=\pm\frac{p_\parallel}{p_\perp}\frac{\omega t}{\sqrt{2(1-\cos(\omega t))}},
# \end{equation*}
# which is independent of $R$ and $\phi_0$ and the sign $\pm$ is the sign of the charge $q$.
# 
# For small $\omega t$ we get
# \begin{align*}
# z_2&\approx \frac{p_\parallel}{p_\perp}\frac{1}{\sqrt{1-\frac{1}{12}\omega^2t^2}}\\
# &\approx\frac{p_\parallel}{p_\perp}\left(1+\frac{1}{24}\omega^2t^2\right).
# \end{align*}
# Hence up to first order in $\omega t$ the coordinate $z_2$ is constant for each helix and given by
# \begin{equation*}
# z_2\approx \frac{p_\parallel}{p_\perp}.
# \end{equation*}

# We set
# \begin{align*}
# x_2 &= \frac{x}{\sqrt{x^2+y^2+z^2}}\\
# y_2 &= \frac{y}{\sqrt{x^2+y^2+z^2}}.
# \end{align*}
# Then the expression
# \begin{align*}
# x_2^2+y_2^2 &= \frac{x^2+y^2}{x^2+y^2+z^2}\\
# &=\frac{1}{1+z_2^2}
# \end{align*}
# is independent of $R$ and $\phi_0$.
# 
# For small $\omega t$, where $z_2\approx \frac{p_\parallel}{p_\perp}$, the point $(x_2,y_2)$ lies approximately on a circle of radius
# \begin{equation*}
# \frac{1}{\sqrt{1+\frac{p_\parallel^2}{p_\perp^2}}} = \frac{p_\perp}{\sqrt{p_\parallel^2+p_\perp^2}} = \frac{p_\perp}{p},
# \end{equation*}
# where $p$ is the total momentum.
# 
# Using Taylor expansion for small $\omega t$ we can calculate
# \begin{align*}
# x(t)&\approx R\sin(\phi_0)\omega t\\
# y(t)&\approx -R\cos(\phi_0)\omega t.
# \end{align*}
# This implies
# \begin{align*}
# \lim_{t\rightarrow 0}x_2(t)&=\sin(\phi_0)\frac{p_\perp}{p}\\
# \lim_{t\rightarrow 0}y_2(t)&=-\cos(\phi_0)\frac{p_\perp}{p}.
# \end{align*}
# We see that the (idealized) initial point $(x_2,y_2)$ on the circle of radius $\frac{p_\perp}{p}$ depends on the angle $\phi_0$, even though $x_0=y_0=0$. Hence in the coordinates $x_2,y_2$ the different helices get separated, depending on the angle $\phi_0$.

# **Cylindrical coordinates**
# 
# Cylindrical coordinates $(r,\phi,z)$, with $r\geq 0, \phi\in[0,2\pi]$, are given by
# \begin{align*}
# x &= r\cos\phi\\
# y &= r\sin\phi\\
# z &= z.
# \end{align*}
# We have  already calculated
# \begin{align*}
# r&=\sqrt{x^2+y^2}\\
# &=\sqrt{2R^2(1-\cos(\omega t))}\\
# &=\left|2R\sin\left(\frac{\omega t}{2}\right)\right|,
# \end{align*}
# where we used in the final step the identity 
# \begin{equation*}
# \cos(\omega t) = \cos^2\left(\frac{\omega t}{2}\right)-\sin^2\left(\frac{\omega t}{2}\right).
# \end{equation*}
# 
# Using the identities
# \begin{align*}
# \cos \alpha-\cos \beta &=-2\sin {\frac  {\alpha+\beta}{2}}\sin {\frac  {\alpha-\beta}{2}}\\
# \sin \alpha-\sin \beta &=2\cos \frac{\alpha+\beta}{2}\sin \frac{\alpha-\beta}{2}
# \end{align*}
# we can write
# \begin{align*}
# x(t) &= 2R\sin\left(\phi_0-\frac{1}{2}\omega t\right)\sin(\omega t)\\
# y(t) &= -2R\cos\left(\phi_0-\frac{1}{2}\omega t\right)\sin(\omega t).
# \end{align*}
# It follows that
# \begin{align*}
# \tan\phi &= \frac{y}{x}\\
# &=\frac{-\cos\left(\phi_0-\frac{1}{2}\omega t\right)}{\sin\left(\phi_0-\frac{1}{2}\omega t\right)}\\ 
# &=\frac{\sin\left(\phi_0-\frac{1}{2}\omega t-\frac{\pi}{2}\right)}{\cos\left(\phi_0-\frac{1}{2}\omega t-\frac{\pi}{2}\right)}\\
# &=\tan\left(\phi_0-\frac{1}{2}\omega t-\frac{\pi}{2}\right).
# \end{align*}
# Hence the helix is given in cylindrical coordinates by
# \begin{align*}
# r(t)&=\left|2R\sin\left(\frac{\omega t}{2}\right)\right|\\
# \phi(t)&=\phi_0-\frac{1}{2}\omega t-\frac{\pi}{2}\\
# z(t)&=R\frac{p_\parallel}{p_\perp}\omega t.
# \end{align*}
# Notice that strictly speaking the angle $\phi$ is only defined if $r\neq 0$. However, the expression for $\phi(t)$ above can still be used.
# 
# Up to terms of second order in $\omega t$ we get
# \begin{align*}
# r(t)&\approx \left|R\omega t\right|\\
# \phi(t)&=\phi_0-\frac{1}{2}\omega t-\frac{\pi}{2}\\
# z(t)&=R\frac{p_\parallel}{p_\perp}\omega t.
# \end{align*}

# **Spherical coordinates**
# 
# Spherical coordinates $(r,\phi,\theta)$, with $r\geq 0, \phi\in[0,2\pi], \theta\in[0,\pi]$, are given by
# \begin{align*}
# x &= r\sin\theta\cos\phi\\
# y &= r\sin\theta\sin\phi\\
# z &= r\cos\theta.
# \end{align*}
# Using our calculations above we can describe the helix by
# \begin{align*}
# r(t)&=|R|\sqrt{\left(\frac{p_\parallel}{p_\perp}\right)^2\omega^2t^2+4\sin^2\left(\frac{\omega t}{2}\right)}\\
# \phi(t)&=\phi_0-\frac{1}{2}\omega t-\frac{\pi}{2}\\
# \theta(t)&=\arctan\left(\frac{p_\perp}{p_\parallel}\frac{2\sin\left(\frac{\omega t}{2}\right)}{\omega t}\right).
# \end{align*}
# Up to terms of second order in $\omega t$ we get
# \begin{align*}
# r(t)&\approx \frac{p}{p_\perp}|R\omega t|\\
# \phi(t)&=\phi_0-\frac{1}{2}\omega t-\frac{\pi}{2}\\
# \theta(t)&\approx\arctan\left(\frac{p_\perp}{p_\parallel}\right).
# \end{align*}
# In particular, the angle $\theta$ is approximately constant.
# 

# **Visualization**
# 
# Let's visualize the formulas for $x_2, y_2, z_2$.

# In[ ]:


# Use time parameter s = omega*t

def x(s,R,phi_0):
    return R*(np.cos(phi_0-s)-np.cos(phi_0))

def y(s,R,phi_0):
    return R*(np.sin(phi_0-s)-np.sin(phi_0))

# p_L = p_longitudinal, p_T = p_transversal
def z(s,R,p_T,p_L):
    return R*(p_L/p_T)*s


# In[ ]:


def r1(s,R,phi_0,p_T,p_L):
    return np.sqrt(x(s,R,phi_0)**2+y(s,R,phi_0)**2+z(s,R,p_T,p_L)**2)

def r2(s,R,phi_0):
    return np.sqrt(x(s,R,phi_0)**2+y(s,R,phi_0)**2)


# In[ ]:


def x2(s,R,phi_0,p_T,p_L):
    return x(s,R,phi_0)/r1(s,R,phi_0,p_T,p_L)

def y2(s,R,phi_0,p_T,p_L):
    return y(s,R,phi_0)/r1(s,R,phi_0,p_T,p_L)

def z2(s,R,phi_0,p_T,p_L):
    return z(s,R,p_T,p_L)/r2(s,R,phi_0)


# In[ ]:


# Set some values for radius R and momenta p_L, p_T, used throughout the examples
R = 1
p_L = 100
p_T = 10


# In[ ]:


# Start with a short time interval [0.01, 0.5]
# We do not start in s=0 to avoid dividing by 0 when calculating x2, y2, z2
S = np.linspace(0.01, 0.5, 200)

# Plot (x,y) for 10 different values for phi_0, corresponding to different initial velocity vectors
for phi_0 in np.linspace(0, np.pi/2, 10):
      
    X = x(S,R,phi_0)
    Y = y(S,R,phi_0)
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    
    plt.plot(X,Y)


# In[ ]:


# Plot transformed coordinates (x2,y2) for the same angles
# The transformed curves align on a circle segment
for phi_0 in np.linspace(0, np.pi/2, 10):
    
    X2 = x2(S,R,phi_0,p_T,p_L)
    Y2 = y2(S,R,phi_0,p_T,p_L)
    
    plt.axis("equal")
    plt.xlabel("x2")
    plt.ylabel("y2")
    
    plt.plot(X2,Y2)


# In[ ]:


# If we make the time interval very short, the different starting points, depending on phi_0, become obvious
S = np.linspace(0.01, 0.03, 200)

for phi_0 in np.linspace(0, np.pi/2, 10):
    
    X2 = x2(S,R,phi_0,p_T,p_L)
    Y2 = y2(S,R,phi_0,p_T,p_L)
    
    plt.axis("equal")
    plt.xlabel("x2")
    plt.ylabel("y2")
    
    plt.plot(X2,Y2)


# In[ ]:


S = np.linspace(0.01, 0.5, 200)

# The circle becomes clearer if we let phi_0 run from 0 to 2*pi
for phi_0 in np.linspace(0, 2*np.pi, 10):
    
    X2 = x2(S,R,phi_0,p_T,p_L)
    Y2 = y2(S,R,phi_0,p_T,p_L)
    
    plt.axis("equal")
    plt.xlabel("x2")
    plt.ylabel("y2")
    
    plt.plot(X2,Y2)


# In[ ]:


# If the time interval becomes large, the segments do not align as well on the circle as before

S = np.linspace(0.01, 2, 200)

for phi_0 in np.linspace(0, np.pi/2, 10):
    
    X2 = x2(S,R,phi_0,p_T,p_L)
    Y2 = y2(S,R,phi_0,p_T,p_L)
    
    plt.axis("equal")
    plt.xlabel("x2")
    plt.ylabel("y2")
    
    plt.plot(X2,Y2)


# In[ ]:


for phi_0 in np.linspace(0,2*np.pi,10):
    
    X2 = x2(S,R,phi_0,p_T,p_L)
    Y2 = y2(S,R,phi_0,p_T,p_L)
    
    plt.axis("equal")
    plt.xlabel("x2")
    plt.ylabel("y2")
    
    plt.plot(X2,Y2)


# In[ ]:


# We now plot z2
# Back to the short time interval
S = np.linspace(0.01, 0.5, 200)

# z2 is independent of phi_0, set phi_0 = 0
phi_0 = 0

# Plot the time dependency of z2
# z2 is approximately constant and the quadratic time dependency is apparent
Z2 = z2(S,R,phi_0,p_T,p_L)
plt.xlabel("s")
plt.ylabel("z2")
plt.plot(S,Z2)
plt.show()


# In[ ]:


# With a larger time interval, higher order terms kick in
S = np.linspace(0.01, 5, 200)

phi_0 = 0

Z2 = z2(S,R,phi_0,p_T,p_L)
plt.xlabel("s")
plt.ylabel("z2")
plt.plot(S,Z2)
plt.show()


# In[ ]:


# The denominator of z2 becomes singular at integer multiples of 2*pi
S = np.linspace(0.01, 7, 200)

phi_0 = 0

Z2 = z2(S,R,phi_0,p_T,p_L)
plt.xlabel("s")
plt.ylabel("z2")
plt.plot(S,Z2)
plt.show()

