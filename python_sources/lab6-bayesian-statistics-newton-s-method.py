#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
References
https://qiita.com/tadOne/items/dcff3e52ea4956008519
https://qiita.com/yadoyado128/items/e4a8d8fa23ec76473d5c
http://danielhomola.com/2016/02/09/newtons-method-with-10-lines-of-python/
https://www.wikiwand.com/en/Newton's_method#/Practical_considerations
"""


# In bag A we have 2 red balls and 1 white ball. In bag B there is 1 red ball and 4 white balls. Choose one bag and choose balls from it. Don't know which bag was chosen, just know what balls were chosen. From the chosen balls estimate which bag was chosen. 

# In[ ]:


bagA=0
bagB=1
ballRed=0
ballWhite=1
probBag=[]
probBag.insert(bagA, 1/2)
probBag.insert(bagB, 1/2)
probBag


# In[ ]:


probBall=[]

probBallA=[]
probBallA.insert(ballRed, 2/3)
probBallA.insert(ballWhite, 1/3)
probBall.insert(bagA, probBallA)

probBallB=[]
probBallB.insert(ballRed, 1/4)
probBallB.insert(ballWhite, 3/4)
probBall.insert(bagB, probBallB)

probBall


# In[ ]:


def posterior(ballList, bag):
    if len(ballList) == 1:
        return probBall[bag][ballList[0]] * probBag[bag]
    else:
        return probBall[bag][ballList[0]] * posterior(ballList[1:], bag)

def posteriorA(ballList):
    return posterior(ballList, bagA) / (posterior(ballList, bagA) + posterior(ballList, bagB))


# In[ ]:


balls=[ballWhite]*2
posteriorA(balls)


# In[ ]:


balls=[ballRed]*2
posteriorA(balls)


# In[ ]:


balls=[ballRed]
posteriorA(balls)


# In[ ]:


balls=[ballWhite]
posteriorA(balls)


# In[ ]:


balls=[ballRed]*5
balls.extend([ballWhite]*3)
posteriorA(balls)


# In[ ]:


balls=[ballRed]*5*5
balls.extend([ballWhite]*3*5)
posteriorA(balls)


# In[ ]:


balls=[ballRed]*5*50
balls.extend([ballWhite]*3*50)
posteriorA(balls)


# In[ ]:


balls=[ballRed,ballWhite]
posteriorA(balls)


# In[ ]:


for n in range(5,101,5):
    balls=[ballRed]*n+[ballWhite]*n
    print(posteriorA(balls))


# In[ ]:


import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


experiment=[1,0,1,1,0,1,0,1,0,0]
nheads=sum(experiment)
nsamples=len(experiment)
with pm.Model() as model:
    p=pm.Beta('p',alpha=1,beta=1)
    y=pm.Binomial('y',n=nsamples,p=p,observed=nheads)
    start = {'p': 0.5}
    step = pm.Metropolis()
    trace = pm.sample(10**6, step, start)
pm.traceplot(trace)
plt.show()


# Want to sample from the posterior distribution $f(\theta|x)=\frac{\pi(\theta) f(x|\theta)}{\int \pi(\theta) f(x|\theta) dx}$. Repeatedly sample $\theta$. Sampling by a chain. Specify the initial value to be 0.5.

# In[ ]:


p, bins = np.histogram(trace["p"], bins=10000, density=True)
theta = np.linspace(np.min(bins), np.max(bins), 10000)
print("ML:" + str(nheads / float(nsamples)))
print("MCMC:" + str(np.dot(p, theta) / 10000))


# In[ ]:


bins


# In[ ]:


x=np.random.uniform(-1,1,10**6)
y=np.random.uniform(-1,1,10**6)
4*sum(x**2+y**2<1)/10**6


# Note that area of an circle is $\pi r^2$. We want to find the roots of $6x^5-5x^4-4x^3+3x^2=x^2(x-1)(6x^2+x-3)$. For the last quadratic, the root is $x=\frac{-1\pm\sqrt{1+4\cdot6\cdot3}}{12}$. First a slow method. 

# In[ ]:


def f(x):
    return 6*x**5-5*x**4-4*x**3+3*x**2
def df(x):
    return 30*x**4-20*x**3-12*x**2+6*x
def ivt(lend,rend):
    for i in range(100):
        temp=(lend+rend)/2
        if f(temp)==0:
            break
        elif (f(temp)>0 and f(rend)>0) or (f(temp)<0 and f(rend)<0):
            rend=temp
        else:
            lend=temp
    return (lend+rend)/2
ivt(-0.5,0.5)


# In[ ]:


ivt(0.5,1.5)


# In[ ]:


sol=ivt(0.01,0.99)
sol


# In[ ]:


f(sol)


# In[ ]:


sol=ivt(-0.99,-0.01)
sol


# In[ ]:


f(sol)


# Now the Newton method. For a starting value, obtain the tangent at that point. Then draw the tangent until it intersects the $x$ axis. The point of intersection will be your new $x$. Repeat. $x-\frac{f(x)}{f'(x)}=x_{new}$.

# In[ ]:


def newtons(f, df, x0, e):
    delta=100
    while delta > e:
        x1 = x0 - f(x0)/df(x0)
        delta = abs(f(x1)-f(x0))
        x0=x1
    print('Root is at: ', x0)
    print('f(x) at root is: ', f(x0))
newtons(f,df,0.3,1e-16)


# In[ ]:


newtons(f,df,0.5,1e-16)


# In[ ]:


newtons(f,df,0.9,1e-16)


# In[ ]:


newtons(f,df,-0.6,1e-16)


# In[ ]:


x=np.linspace(-1.1,1.1,num=10)
x


# In[ ]:


def newtons2(f, df, x0, e):
    delta=1
    while delta > e:
        x1 = x0 - f(x0)/df(x0)
        delta = abs(f(x1)-f(x0))
        x0=x1
    return x0
solution=[]
for t in np.nditer(x):
    solution.append(newtons2(f,df,t,1e-14))
sol=set(solution)
sol


# In[ ]:


set([ round(elem, 3) for elem in list(sol)])

