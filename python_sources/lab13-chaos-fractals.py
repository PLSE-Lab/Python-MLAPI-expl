#!/usr/bin/env python
# coding: utf-8

# Some phenomena may look random but are in fact derived from a deterministic process. This is the study of chaos theory. The process is nonlinear and heavily dependant on the initial value. Even a slight change in the initial state results in completely different subsequent behaviour. Fractals are objects which when zoomed in, has the same structure as the original. That is, it is said to posess self-similarity. This is interesting because these kind of seemingly complex structure can be produced from a very simple prinsiple. Materials for this lab were taken from [here](https://qiita.com/jabberwocky0139/items/33add5b3725204ad377f). We will look first at the Mandelbrot set. It is the set of pairs $(x,y)$ such that the recurrence relation,
# \begin{align*}
# a_{n+1}&=a_n^2-b_n^2+x\\
# b_{n+1}&=2a_n b_n+y\\
# a_0&=b_0=0
# \end{align*}
# does not result in the blowup of the values for $a_n$ and $b_n$ as $n\rightarrow \infty$. Let us calculate this set. 

# In[9]:


import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
N, M = 50, 500
def mandel(X, Y):
    a, b = [0] * 2
    for i in range(N):
        a, b = a**2 - b**2 + X, 2 * a * b + Y
    return a**2 + b**2 < 4
x, y = [np.linspace(-2, 2, M)] * 2
X, Y = np.meshgrid(x, y)
plt.pcolor(X, Y, mandel(X, Y))
plt.show()


# After 50 iterations, if the norm squared of $(a,b)$ is smaller than 4 we consider it to not have blown up. We code 1 for non blow up and 0 for blow up. 0 is for purple and 1 is for yellow. Let's look at (a,b) in more detail. Instead of mapping the values to 0 or 1, we can think of the numbers as is. The variation in color shows the change in the values that the processes converge to. 

# In[25]:


N=10000
def mandel2(X, Y):
    a, b = [0] * 2
    for i in range(N):
        a, b = a**2 - b**2 + X, 2 * a * b + Y
    return a**2 + b**2
x, y = [np.linspace(-2, 2, M)] * 2
X, Y = np.meshgrid(x, y)
plt.pcolor(X, Y, mandel2(X, Y),cmap='RdBu')
plt.show()


# Logistic mapping is another example. For each $a$ we think of the following sequence $x_{n+1}=ax_n(1-x_n)$, which is called a logistic equation. For each $a$ we run this process for 400 steps and take the last 100 numbers. These 100 numbers might all be the same number for some $a$ where the sequence convergences. While for the later $a$'s the sequence might oscillate among a range of values. The plot shows the $a$'s on the horizantal axis and the values of the last 100 values of the sequence on the vertical axis. 

# In[26]:


def logistic(a):
    x = [0.8]
    for i in range(400):
        x.append(a * x[-1] * (1 - x[-1]))
    return x[-100:]

for a in np.linspace(2.0, 4.0, 1000):
    x = logistic(a)
    plt.plot([a]*len(x), x, "c.", markersize=0.1)

plt.show()


# It is said that this mapping is also used to model animal populations. It is interesting to see such pattern out of such simple process. Apparently there is no instability due to the initial value. Please check this. Now we will look at the Duffing equation.
# \begin{align*}
# m\ddot{x}=-\gamma \dot{x}+2ax-4bx^3+F_0\cos(\omega t+\delta).
# \end{align*}
# We can solve an ordinary differential equation using the `odeint` function of `scipy.integrate`. The x axis is for location and the y axis is for velocity.

# In[28]:


from scipy.integrate import odeint, simps
def duffing(var, t, gamma, a, b, F0, omega, delta):
    x_dot = var[1]
    p_dot = -gamma * var[1] + 2 * a * var[0] - 4 * b * var[0]**3 + F0 * np.cos(omega * t + delta)
    return np.array([x_dot, p_dot])
F0, gamma, omega, delta = 10, 0.1, np.pi/3, 1.5*np.pi
a, b = 1/4, 1/2
var, var_lin = [[0, 1]] * 2
t = np.arange(0, 20000, 2*np.pi/omega)
t_lin = np.linspace(0, 100, 10000)
var = odeint(duffing, var, t, args=(gamma, a, b, F0, omega, delta))
var_lin = odeint(duffing, var_lin, t_lin, args=(gamma, a, b, F0, omega, delta))
x, p = var.T[0], var.T[1]
x_lin, p_lin = var_lin.T[0], var_lin.T[1]
plt.plot(x, p, ".", markersize=2)
plt.show()


# A very peculiar pattern has appeared. This is called a strange attractor. If you were to make a plot with more detail and zoomed in, it is said that you can observe self-similarity. Let us see what the trajectory will be for a process with a slightly different initial state. You will see that the location of the pendulum will match completely at some time points while for other time points they will be located at a completely different location. Note that the x axis is time and the y axis is the pendulum location. 

# In[29]:


plt.plot(t_lin, x_lin)
var_lin = odeint(duffing, [0.1, 1], t_lin, args=(gamma, a, b, F0, omega, delta))
x_lin, p_lin = var_lin.T[0], var_lin.T[1]
plt.plot(t_lin, x_lin)
plt.show()


# Now we will look at the 3 body problem. We have three point masses. The Lagrangian of the system is,
# \begin{align*}
# L&=\sum_{i=1}^3 \left[\frac{m_i \dot{q}_i^2}{2}+\sum_{j>i} \frac{Gm_im_j}{r_{ij}}\right]\\
# q_i&=(x_i,y_i)\\
# r_{ij}&=\sqrt{(x_i-x_j)^2-(y_i-y_j)^2}.
# \end{align*}
# We need to solve the Euler-Lagrange equation,
# \begin{align*}
# \frac{d}{dt} \frac{\partial L}{\partial \dot{q}_i}-\frac{\partial L}{\partial q_i}=0,
# \end{align*}
# for this system. The trajectory is very complicated and tangled. 

# In[15]:


def r(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1- y2)**2)**-3
def manybody(var, t, m1, m2, m3, G):
    x1 = var[2]
    px1 = -G * (m2 * r(var[0], var[1], var[4], var[5]) * (var[0] - var[4]) + m3 * r(var[0], var[1], var[8], var[9]) * (var[0] - var[8]))
    y1 = var[3]
    py1 = -G * (m2 * r(var[0], var[1], var[4], var[5]) * (var[1] - var[5]) + m3 * r(var[0], var[1], var[8], var[9]) * (var[1] - var[9]))
    x2 = var[6]
    px2 = -G * (-m1 * r(var[4], var[5], var[0], var[1]) * (var[0] - var[4]) + m3 * r(var[4], var[5], var[8], var[9]) * (var[4] - var[8]))
    y2 = var[7]
    py2 = -G * (-m1 * r(var[4], var[5], var[0], var[1]) * (var[1] - var[5]) + m3 * r(var[4], var[5], var[8], var[9]) * (var[5] - var[9]))
    x3 = var[10]
    px3 = -G * (-m1 * r(var[8], var[9], var[0], var[1]) * (var[0] - var[8]) - m2 * r(var[8], var[9], var[4], var[5]) * (var[4] - var[8]))
    y3 = var[11]
    py3 = -G * (-m1 * r(var[8], var[9], var[0], var[1]) * (var[1] - var[9]) - m2 * r(var[8], var[9], var[4], var[5]) * (var[5] - var[9]))
    return np.array([x1, y1, px1, py1, x2, y2, px2, py2, x3, y3, px3, py3])
m1, m2, m3 = 3, 4, 5
G = 1
var = np.array([0, 4, 0, 0, -3, 0, 0, 0, 0, 0, 0, 0])
t = np.linspace(0, 70, 3e7)
var = odeint(manybody, var, t, args=(m1, m2, m3, G), full_output=False)
plt.plot(var[:, 0][::1000], var[:, 1][::1000], label="1")
plt.plot(var[:, 4][::1000], var[:, 5][::1000], label="2")
plt.plot(var[:, 8][::1000], var[:, 9][::1000], label="3")
plt.show()


# You can checkout the animation of this trajectory [here](https://t.co/hQnWrZD2Vt). Again, thanks to Jabberwocky. This is actually not the best calculation one can do. If you are interested please see [here](http://www.ucolick.org/~laugh/oxide/projects/burrau.html). 
