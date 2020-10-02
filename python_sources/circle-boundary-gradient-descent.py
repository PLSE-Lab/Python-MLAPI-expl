#!/usr/bin/env python
# coding: utf-8

# <h3 style="text-align:center">Localize a set(cluster) of points with gradient descent procedure</h3>
# 
# <p>Having a circle in a random position and a cluster of blue points, we are trying to move  the circle in order to include the blue points into the circle, avoidig the red points, which must stay outside the circle.</p>
# <p>We will use the trigonometric definition of a circle with radius = 1 and origin in (0,0): ${ x }^{ 2 }+{ y }^{ 2 }=1$</p>
# <p>So:</p>
# <p>points inside the circle are described by the formula: ${ x }^{ 2 }+{ y }^{ 2 }-1<0$</p>
# <p>and</p>
# <p>points outside the circle are described by the formula: ${ x }^{ 2 }+{ y }^{ 2 }-1>0$</p>
# <p>A circle with radius b and origin (-x1, -x2) is defined by: ${ ({ w }_{ 1 }+{ x }_{ 1 }) }^{ 2 }+{ ({ w }_{ 2 }+{ x }_{ 2 }) }^{ 2 }-b=0$</p>
# <p>For convenience, in order to have positive score for a positive prediction (points inside the circle), we will change the formula sign of the ${ Score }_{ point }={ -({ w }_{ 1 }+{ x }_{ 1 }) }^{ 2 }-{ ({ w }_{ 2 }+{ x }_{ 2 }) }^{ 2 }+b=0$</p>
# <p>Continuous prediction function is: $\hat { y } = sigmoid({ Score }_{ point })$</p>
# <p>Weights will be updated according to method: ${ w }_{ i }={ w }_{ i }+{ w }_{ i }\cdot \frac { d }{ d{ w } } (Error)$</p>
# <p>As a consequence, weights of score function will be updated with a <i>high rates at the beginning</i>, and <i>lower rates</i> as approaching to minimum error point.</p>

# - Sigmoid activation function
# 
# $$\sigma(x) = \frac{1}{1+e^{-x}}$$
# 
# - Output (prediction) formula
# 
# $$\hat{y} = \sigma({ -({ w }_{ 1 }+{ x }_{ 1 }) }^{ 2 }-{ ({ w }_{ 2 }+{ x }_{ 2 }) }^{ 2 }+b)$$
# 
# - Error function
# 
# $$Error(y, \hat{y}) = - y \log(\hat{y}) - (1-y) \log(1-\hat{y})$$
# 
# - The function that updates the weights
# 
# $$ w_i \longrightarrow { w }_{ i }+(-2)\cdot (\widehat { y } -y)({ w }_{ i }+{ x }_{ i })\cdot \alpha$$
# 
# $$ \alpha \quad -\quad being\quad the\quad learn\quad rate\ $$

# <p>Proof of error gradient:</p>
# <br>
# <p>$\sigma '(x)=\frac { d }{ dx } \frac { 1 }{ 1+{ e }^{ -x } } =\frac { d }{ dx } { (1+{ e }^{ -x }) }^{ -1 }=-{ (1+{ e }^{ -x }) }^{ -2 }\cdot ({ -e }^{ -x })=\frac { { e }^{ -x } }{ { (1+{ e }^{ -x }) }^{ 2 } } =\frac { 1 }{ 1+{ e }^{ -x } } \cdot \frac { { e }^{ -x } }{ 1+{ e }^{ -x } } \Rightarrow \\ \sigma '(x)=\sigma (x)(1-\sigma (x))\\ $</p>
# <p>$Gradient\quad of\quad E\quad for\quad a\quad point\quad ({ x }_{ 1 }...{ x }_{ n }):\quad \Delta E=(\frac { d }{ d{ w }_{ 1 } } { E },\quad ...\quad ,\frac { d }{ d{ w }_{ n } } { E },\quad \frac { d }{ { d }_{ b } } E)$</p>
# <br>
# <p>$\frac { d }{ d{ w }_{ j } } \widehat { y } = \frac { d }{ d{ w }_{ j } } \sigma (-{ (W+x) }^{ 2 }+b)=\\ =\sigma (-{ (W+x) }^{ 2 }+b)(1-\sigma (-{ (W+x) }^{ 2 }+b)\cdot \frac { d }{ d{ w }_{ j } } (-{ (W+x) }^{ 2 }+b)=\\ =\widehat { y } (1-\widehat { y } )\cdot \frac { d }{ d{ w }_{ j } } (-{ (W+x) }^{ 2 }+b)=\\ =\widehat { y } (1-\widehat { y } )\cdot \frac { d }{ d{ w }_{ j } } (-{ ({ w }_{ 1 }+{ x }_{ 1 }) }^{ 2 }-...-{ ({ w }_{ j }+{ x }_{ j }) }^{ 2 }+...+-{ ({ w }_{ n }+{ x }_{ n }) }^{ 2 }+b)=\\ =\widehat { y } (1-\widehat { y } )\cdot (-2)\cdot ({ w }_{ j }+{ x }_{ j })$</p>
# <br>
# <p>$\frac { d }{ d{ w }_{ j } } E=\frac { d }{ d{ w }_{ j } } (-y\cdot ln(\widehat { y } )-(1-y)ln(1-\widehat { y } ))=\\ =-y\frac { d }{ d{ w }_{ j } } ln(\widehat { y } )-(1-y)\frac { d }{ d{ w }_{ j } } ln(1-\widehat { y } )=\\ =-y\cdot \frac { 1 }{ \widehat { y }  } \cdot \frac { d }{ d{ w }_{ j } } \widehat { y } -(1-y)\cdot \frac { 1 }{ 1-\widehat { y }  } \frac { d }{ d{ w }_{ j } } (1-\widehat { y } )=\\ =-y\cdot \frac { 1 }{ \widehat { y }  } \cdot \widehat { y } (1-\widehat { y } )\cdot (-2)\cdot ({ w }_{ j }+{ x }_{ j })-(1-y)\cdot \frac { 1 }{ 1-\widehat { y }  } \cdot (-1)\widehat { y } (1-\widehat { y } )\cdot (-2)\cdot ({ w }_{ j }+{ x }_{ j })=\\ =-y\cdot (1-\widehat { y } )\cdot { (-2)\cdot ({ w }_{ j }+{ x }_{ j }) }+(1-y)\cdot \widehat { y } \cdot (-2)\cdot ({ w }_{ j }+{ x }_{ j })=(-2)\cdot ({ w }_{ j }+{ x }_{ j })\cdot (-y+y\widehat { y } +\widehat { y } -y\widehat { y } )=\\ =(-2)\cdot ({ w }_{ j }+{ x }_{ j })\cdot (\widehat { y } -y)$</p>
# <br>
# <p>$Erors\quad of\quad point\quad ({ x }_{ 1 },\quad ...\quad ,{ x }_{ n })\quad is:\quad \Delta E=[(-2)\cdot (\widehat { y } -y)\cdot ({ w }_{ 1 }+{ x }_{ 1 }),\quad ...\quad ,\quad (-2)\cdot (\widehat { y } -y)\cdot ({ w }_{ 1 }+{ x }_{ 1 })]$</p>

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#data = pd.read_csv('test.csv', header=None)
#X = np.array(data[[0,1]])
#pointType = np.array(data[2])

#array data points: x1, x2
data = np.array([
    [5,8,1],
    [4,7,1],
    [4,8,1],
    [7,7,0],
    [6.5,7.2,0],
    [5,7,0],
    [5.5,7.5,0],
    [7,8,0],

])
X = data[:, [0,1]]
pointType = data[:, [2]]
pointType = pointType.flatten()

datamin = np.min(X, axis=0)
datamax = np.max(X, axis=0)
xmin, ymin = datamin
xmax, ymax = datamax

def plot_points(X, y):
    admitted = X[np.argwhere(y==1)]
    rejected = X[np.argwhere(y==0)]
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s = 25, color = 'blue', edgecolor = 'k',zorder=2)
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s = 25, color = 'red', edgecolor = 'k',zorder=2)
    
phi = np.linspace(0, 2*np.pi, 200)

def circle(w1=0,w2=0,b=1,color='green'):
    x = b*np.cos(phi)
    y = b*np.sin(phi)
    plt.axis("equal")
    plt.plot(x-w1,y-w2,color,zorder=1)
  


# In[ ]:


# Activation (sigmoid) function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def score(points,weights,bias):
    squares = (points+weights)**2
    if (squares.shape == (2,)):
        s = np.sum(squares)
    else:
        s = np.sum(squares, axis=1)
    s = -1*(s-bias)
    return s

#likelihood function
#yhat:continuous prediction function which returns [0,1] prob. instead of {0,1} discrete values
def probability(score):
    return sigmoid(score)

#0 class points have 1-p probability to be correct classified
def likelihood(y,p):
    return y*(p) + (1 - y) * (1-p)

#error function
#log_loss = log_likelihood = -1 *log(likelihood)
def log_loss(likelihood):
    return -1*np.log(likelihood)

def update(x, y, weights, bias, learnrate):
    s = score(x,weights,bias)
    p = probability(s)
    
    weights += learnrate * (-2) * (y-p) * (weights+x)
    #bias += learnrate * (y-p)
    return weights, bias       


# In[ ]:


bias = 1.2
weights = [-8,-8]

s = score(X,weights,bias)
p = probability(s)
l = likelihood(pointType,p) 
err = log_loss(l)
#print(X,weights)
#print(s)
#print(p)
#print(l)
#print(err)

epochs = 12
learnrate = 0.1
errors = []
last_loss = None


circle(weights[0],weights[1],bias,'yellow')


for e in range(epochs):
    #circle(weights[0],weights[1],bias)
    
    for x, y in zip(X, pointType):
 
        s = score(x,weights,bias)
        p = probability(s)
        l = likelihood(y,p)        
        err = log_loss(l)
        #print(s,p,l,err)
        
        #print("1",weights,bias)
        weights,bias = update(x, y, weights, bias, learnrate)
        #print("2",weights,bias)
        #print(x,weights,x.shape)
        #s = score(x,weights,bias)
        
    circle(weights[0],weights[1],bias)
    
    s = score(X,weights,bias)
    p = probability(s)
    l = likelihood(pointType,p)        
    err = log_loss(l)
    loss = np.mean(err)
    errors.append(loss) 
    
    #print(loss)
    
    #if e % (epochs / 10) == 0:
    if True:
        print("\n========== Epoch", e,"==========")
        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
            break
        else:
            print("Train loss: ", loss) 
        last_loss = loss    
        
circle(weights[0],weights[1],bias,'black')

more = 2
plt.xlim(xmin-more,xmax+more)
plt.ylim(ymin-more,ymax+more)
plot_points(X, pointType)


# In[ ]:





# In[ ]:




