#!/usr/bin/env python
# coding: utf-8

# # Risk and Bias-Variance Trade-Off
# We consider a univariate regression problem where we want to predict the variable $y$ knowing one variable $x$.
# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.preprocessing
import sklearn.pipeline
import sklearn.linear_model

# read .csv file and return a pandas DataFrame object
s = pd.read_csv("../input/lab1.csv")
s.describe()


# In[ ]:


x, y= s.x, s.y
plt.scatter(x, y)


# ## 2 Ridge Regression
# In this section, we want to fit a polynomial model to the data. This can be done using a linear model $y=\beta_0+\sideset{}{_{i=1}^d}\sum  \beta_i x^i$. The coefficients ${(\beta_i)}_{0 \leq i\leq d}$ of such a model can be obtained by a ridge regression.
# 
# The function `fitridge(s,alpha)` returns a $10^{th}$ degree polynomial model fitted using a Ridge procedure on the data set `s` and the regularizing parameter `alpha`.

# In[ ]:


def fitridge(s, alpha):
    poly =  sklearn.preprocessing.PolynomialFeatures(10, include_bias=False)
    scale = sklearn.preprocessing.StandardScaler()
    ridge = sklearn.linear_model.Ridge(alpha=alpha)
    model = sklearn.pipeline.make_pipeline(poly, scale, ridge)
    model.fit(s[["x"]], s.y)
    return model


# Using `fitridge`, let's fit a 10 degree polynomial models to `s` using differents values `alpha`. Then, using `plotmodel` with the additionnal argument `color="red"` or `color="blue"` for instance, plot different polynomial models (_i.e._ different alpha) with different colors.
# 
# Please note that, if `h` is the value returned by `fitridge`, then `plotmodel(h.predict,plotrange)` will plot the curve associated with the fitted model `h`.

# In[ ]:


def plotmodel(modelpredict, plotrange, **kwargs):
    x = np.linspace(plotrange[0], plotrange[1], 200)
    plt.plot(x,modelpredict(np.expand_dims(x, -1)), **kwargs)
    plt.ylim((-35,30))

plotrange = (0, 15)


# In[ ]:


model_10 = fitridge(s, 0.01)
plotmodel(model_10.predict, [0,15])
plt.scatter(x, y, color='red')


# In[ ]:


model_under = fitridge(s, 0)
plotmodel(model_under.predict, [0,15])

model_hover = fitridge(s, 1000)
plotmodel(model_hover.predict, [0,15])

plotmodel(model_10.predict, [0,15])

plt.scatter(x, y, color='red')


# Using these plots, we can find 3 values of alpha to obtain 3 models: $h_{over}$ that seems to  over-fit, $h_{well}$ that seems to fit well and $h_{under}$ that seems to under-fit. To help you decide, re-run the cell with several different values.
# 
# # 3 Risk and Empirical Risk
# 
# 
# 
# In general, the visual method used in the previous question cannot be used to choose alpha because several explanatory variables are involved. Furthermore, we might want to quantify and automate this process.
# 
# 
# The goal in the regression problem is to minimize the expected loss when $h$ is used to predict $y$ when $(x,y)$ are drawn from $(X,Y)$ _i.e._ minimize the **risk** $R$: $$R(h)=\mathbb{E}_{X,Y}\left[l\left(h(X),Y\right)\right]$$
# 
# Thus we just have to found an alpha minimizing the **risk**. However, the **risk** cannot be computed without knowing the jointly distributed random variables $(X,Y)$.
# 
# 
# For want of anything better, using a set $s$ of $n$ draws from $(X,Y)$, one can compute the **empirical risk**:
# 
# $$R_{\text{emp}}(h,s)=\frac{1}{\lvert s \rvert} \underset{(x,y) \in s}{\sum} l\left(h(x),y\right)$$
# 
# 
# Thus, a suitable method to select an alpha is to split the set $s$ in two parts, the first part is used as a _training set_ $\mathrm{ts}$ and the second part as a _validation set_ $\mathrm{vs}$. The _training set_ is used to fit the model $h_{\mathrm{ts}}$ and the _validation set_ is used to estimate the performance of the model. The model with the best performance will be the selected one.
# 
# 
# Let us consider a quadratic loss $l(\hat{y},y)=(\hat{y}-y)^2$.
# 
# 
# Let's write the function `risk_emp(h,s)` that returns $R_{\text{emp}}(h,s)$. To do so, you can assume that `h` has a method `predict` that returns an array of the values predicted by `h` for the set `s`. The method `h.predict` takes a 2-D matrix as input. Thus, its input will be `s[["x"]]`, not `s.x`.
# 
# 
# 

# In[ ]:


def risk_emp(h,s):
    y = s.y
    y_ = h.predict(s[["x"]])
    return sum((y_ - y)**2)/len(y_)


# Now we are going to split `s` evenly into  a _training set_ $\mathrm{ts}$ and a _validation set_ $\mathrm{vs}$ with $40$ examples each. We fit the model with $\mathrm{ts}$.

# In[ ]:


ts, vs = s[:40], s[40:]
re_ts = []
re_vs = []
alphas = np.array([0.00001, 0.01, 0.1, 1, 1000])
for alpha in alphas:
    model_ts = fitridge(ts, alpha)
    re_ts.append(risk_emp(model_ts, ts))
    re_vs.append(risk_emp(model_ts, vs))

plt.plot(np.log10(alphas), re_ts)
plt.plot(np.log10(alphas), re_vs)


# # 4 Bias-Variance Trade-Off
# 
# 
# In this section, we look at the bias-variance trade-off. For a given $X=x_0$, the expected loss using a given model $h_s$ is $\mathbb{E}_{Y_0}\left[\left(Y_0-h_s\left(x_0\right)\right)^2\right]$ where $Y_0=(Y|X=x_0)$. It tells us how precise a fixed model $h_s$ is. If we want to know if a learning algorithm is "efficient" using sets of $n$ examples (not a specific set!), we have to look at $\mathbb{E}_{s,Y_0}\left[\left(Y_0-h_s\left(x_0\right)\right)^2\right]$. It gives the average expected loss obtained using a given learning algorithm on sets of $n$ examples.
# 
# To better understand this average expected loss, we can decompose this term in two terms called the **_bias_** and the **_variance_**:
# $$\mathbb{E}_{s,Y_0}\left[\left(Y_0-h_s\left(x_0\right)\right)^2\right]=\underset{\text{average squared bias}}{\underbrace{\mathbb{E}_{Y_0}\left[\left(Y_0-\mathbb{E}_{h_s}\left[h_s\left(x_0\right)\right]\right)^2\right]}}+\underset{\text{variance}}{\underbrace{\mathbb{E}_{s}\left[\left(\mathbb{E}_{s}\left[h_s\left(x_0\right)\right]-h_s\left(x_0\right)\right)^2\right]}}$$
# 
# 
# <font color='red'>**The cell code below have a function `generate(n)` that generates a new data set containing $n$ observations of the variables $(X,Y)$ . Please note that, in most problems, all the functions in the cell code below are not available. Only one set of $n$ observations of $(X,Y)$ is available. In our case, it would be the data set stored in the file "lab1.csv".**
# </font>

# In[ ]:


def truemodel(x):
    p = [1, 2, -1.3, 0.1, -0.001]
    res = 0
    # Horner's method to compute efficiently values of a polynom: a0 + x (a1 +x ( a2 + x ...)))
    for pi in reversed(p):
        res = pi + x * res
    return res


# generatey(x) is a function drawing random values from the unknown distribution Y|X=x.
# It generates an examples set of Y for a given X=x.
# This kind of examples set is not available in practical application.
def generatey(x):
    e = 10
    return truemodel(x) + e * np.random.randn(*x.shape)


# generate(n) is a function drawing random values from the unknown distribution (X,Y).
# It generates a set of n examples.
# This kind of examples set is what you have in practical application.
def generate(n):
  a = 0
  b = 15
  x = a + (b - a) * np.random.rand(n) # np.random.rand(d0, d1, ...) generates d0 * d1 * ... values drawn in U(0,1).
  y = generatey(x)
  return pd.DataFrame({"x":x, "y":y})


# The function `compute_hx0s(x0,fit,k,n)` returns an array of `k` values of $h_s(x_0)$. These `k` values are different from each other because, at each iteration, we compute a new $h_s$ using a new set $s$ of `n` examples. These $h_s$ are computed using the function call `fit(s)`.

# In[ ]:


def compute_hx0s(x0,fit,k,n):
    result = []
    for i in range(k):
        ts = generate(n)
        hs = fit(ts)
        result.append(hs.predict(np.array([[x0]]))[0])
    return np.array(result)


# ## 4.1 Computing the Variance in the Bias-Variance Decomposition
# For the following, we consider $x_0=7.5$,  $\alpha=1e-10$ and $n=40$. Using `compute_hx0s` and `np.var`, let's compute an estimation of the **_variance_** term.

# In[ ]:


x0, k, n = 7.5, 300, 40
print(np.var(compute_hx0s(x0,lambda s: fitridge(s, 1e-10),k,n)))


# ## 4.2 Computing the Bias in the Bias-Variance Decomposition
# 

# In[ ]:


y = generatey(np.repeat(x0, k))
y_ = compute_hx0s(x0,lambda s: fitridge(s, 1e-10),k,n)
print(np.mean((y - y_)**2))


# ## 4.3 Plotting the Bias and Variance as a Function of the Hyperparameter

# In[ ]:


biases = []
variances = []
alphas = np.array([0.00001, 0.01,0.05, 0.1, 0.5, 1, 5, 10, 50, 1000])
for alpha in alphas:
    y = generatey(np.repeat(x0, k))
    y_ = compute_hx0s(x0,lambda s: fitridge(s, alpha),k,n)
    
    biases.append(np.mean((y - y_)**2))
    variances.append(np.var(y_))

plt.plot(np.log10(alphas), biases)
plt.plot(np.log10(alphas), variances)

