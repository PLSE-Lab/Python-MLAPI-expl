#!/usr/bin/env python
# coding: utf-8

# # Support Vector Machines (SVM)
# ### Contents
# * Introduction
# * Margins
#     * Functional Margin
#     * Geometric Margin
# * Optimal Margin Classifier
# * Kernels
# * ~~Non-separable case~~
# * ~~Regularization~~
#     * ~~L1 Regularization~~
#     * ~~L2 Regularization~~
# * ~~Sequential Minimal Optimization (SMO) algorithm~~

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

from sklearn.svm import SVC
from sklearn.datasets.samples_generator import make_blobs, make_circles
from mpl_toolkits import mplot3d
from ipywidgets import interact, fixed
# Any results you write to the current directory are saved as output.
iris = datasets.load_iris()


# ## Introduction
# 
# Before starting with SVM, Let me talk about what we want to achieve from Machine Learning algorithms in general.
# 
# ### Geometric Interpretation
# Below in the plot we see that from machine learning we need get a **line** inbetween the data which can differentiate between two categories in a good manner. For 3-D features it will be a **plane** and for higher dimensions it will be a **hyper-plane**.
# For now think of these two categories of data as boxes and triangles. It doesn't matter what the data is about.

# In[ ]:


X = iris.data[:, [2, 0]]
X = X[:100]
y = iris.target
y = y[:100]
svm = SVC(C=0.5, kernel='linear')
svm.fit(X, y)

fig = plt.figure(figsize=(16, 10))
# Plotting decision regions
plot_decision_regions(X, y, clf=svm, legend=2, zoom_factor=4.0)

# Adding axes annotations
plt.xlabel('petal length [cm]')
plt.ylabel('sepal length [cm]')
plt.title('SVM')
plt.show()


# ### Mathematical Interpretation
# In machine learning all we want to do is learn a function $f(x)$ which can give us correct predictions $y$ for a given input $X$.
# * The function $f(x)$ is known as **hypothesis function** which is mostly denoted by $h(x)$.
# ![image](https://i.imgur.com/L3SHSc6.png)
# 
# * Let us take an example of a linear function $ h_\theta(x) = \theta{_0} + \theta{_1}{x_1}  + \theta{_2}{x_2} + \theta{_3}{x_3} + \theta{_n}{x_n}$. Where $n$ in the dimension for the feature vector $X$.
# * Here we want to learn the parameters $\theta_i : for\ i\ in \ \{0,1,2,3,...,n\}$

# ## Margins
# As described above in the geometric interpretation we want to draw a line between the data such that we can differentiate between the two type of data. But which line is a best line for this case. we can clearly see that we can draw many lines which can differentiate the above data correctly. But which one is the best one. To answer this question let's take an example of logistic regression where we predict output based on the probability.
# So we will predict $1$ if we get $h_{\theta}(x) \geq 0.5$ , $0$ otherwise. The larger the $h_{\theta}(x)$ we can say with more confidence that our prediction is going to be correct and vice-versa.
# 
# 
# 

# In[ ]:


fig = plt.figure(figsize=(16, 10))
plt.scatter(x=[1,2,3,4,4,3.5,3.1,2.5,2.2, 4.9], y=[1,2,3,4,5,2,3,1,2,4], marker='o', s=100)
plt.scatter(x=[5.5, 6, 7, 8, 7,6.4,7.7,7.6, 5.4,6], y=[4,3,5,6,7,6,7,5,6,7.7], marker='+', s=100)
plt.plot([5.3,5], [0,8], 'ro-')


# ### Notation
# * We will be considering a binary classification problem where $y \in \{-1,1\}$.
# * Rather than using $\theta$ we will be using $w$ and $b$ in its place. where $b$ is just $\theta_0$ and the rest is represented by $w$.
# $$
# h_{w,b}(x) = g(w^Tx + b)
# $$
# 
# $$g(z) = \big\{ 1\ if\ z \geq\ 0,\ else\ -1\big\}$$

# ### Functional margin
# * For a **training example** $(x^{(i)}, y^{(i)})$, we define the **functional margin** of $(w,b)$ with respect to the **training example**
# $$
# \large{\hat{\gamma}^{(i)} = y^{(i)}(w^Tx^{(i)}+b)}
# $$
# 
# * For a given **training set** $S = \{ (x^{(i)}, y^{(i)}); i = 1,...,m\}$ function margin of $(w,b)$ with respect to $S$ is the smallest functional margins of the individual training examples.
# 
# $$
# \large{\hat{\gamma} = \mathop{min}_{\textbf{i=1,...,m}}\hat{\gamma}^{(i)}}
# $$
# 
# On problem why functional margin is not a great extimator of confidence of predictions is because we can scale it by any constant and increase to any arbitary number, but still the predictions will be the same ${g(w^Tx+b)\ =\ g(2w^Tx+2b)}$.
# 
# So, Intuitively, we want some kind of normalization condition, such that we don't have this problem.

# # Geometric Margin
# 

# In[ ]:


fig = plt.figure(figsize=(16, 10))
plt.scatter(x=[1,2,3,4,4,3.5,3.1,2.5,2.2, 4.9], y=[1,2,3,4,5,2,3,1,2,4], marker='o', s=100)
plt.scatter(x=[5.5, 6, 7, 8, 7,6.4,7.7,7.6, 5.4,6, 7.25], y=[4,3,5,6,7,6,7,5,6,7.7, 4.4], marker='+', s=100)
plt.plot([5.3,5], [0,8], 'ro-')
plt.plot([5.15, 7.2], [4,4.4], '>-g', color='green')
plt.text(4.95, 4, 'B', fontsize=22)
plt.text(7.3, 4.4, 'A', fontsize=22)
plt.plot([5.1, 7.2], [4+2,4.4+2], '>-g', color='blue')
plt.text(6, 5.6, '$W$', fontsize=22)
plt.text(6, 3.5, '$\gamma^{(i)}$', fontsize=22)


# Geometric margin $\gamma^{(i)}$ is defined as the perpendicular distance between the point A and B in the above figure. Now the question is:
# 
# How to find $\gamma^{(i)}$?
# 
# $\frac{w}{||w||}$ is a unit-length vector pointing in the same direction as $w$.
# 
# Since point A represents $x^{(i)}$, B is given $x^{(i)}-\gamma^{(i)}.(w/||w||)$. As $w/||w||$ is a unit vector perpendicular from B to A. So using the negative of this as a multiplier we go to the opposite direction and scaling it with $\gamma^{(i)}$ makes us reach B. 
# 
# We know that point B lies on the line(hyperplane), so it must satisfy the equation of the hyperplane.
# $$
# w^T\Big(x^{(i)} - \gamma^{(i)}\frac{w}{||w||}\Big) + b = 0
# $$
# 
# Solving the above equation for $\gamma^{(i)}$ gives us 
# $$
# \large{\gamma^{(i)} = \Big(\big( \frac{w}{||w||}  \big)^T x^{(i)} + \frac{b}{||w||} \Big)}
# $$
# 
# The above calculation was done for the case of a single positive training example $x{(i)}$. In general case
# 
# * **geometric margin** of $(w,b)$ with respect to **training example** $(x^{(i)}, y^{(i)})$ is defined as
# $$
# \large{\gamma^{(i)} = y^{(i)}\Big(\big( \frac{w}{||w||}  \big)^T x^{(i)} + \frac{b}{||w||} \Big)}
# $$
# 
# * **geometric margin** of $(w,b)$ with respect to training set $S = \{ (x^{(i)}, y^{(i)}); i = 1,...,m\}$ is the smallest of the geometric margin for indivisual training examples
# $$
# \large{\gamma = \mathop{min}_{\textbf{i=1,...,m}}\gamma^{(i)}}
# $$
# 

# # Optimal margin classifier
# After understanding the equations in the section before, we can easily say all we want from our decision boundary is to **maximize** the **geometric margin**.
# 
# This would mean we will have a classifier which seperates the positive and negative training examples with a "gap".
# 
# So, how do we propose that as a optimization problem:
# $$
# \begin{aligned}
# & max_{\gamma, w, b}
# & & \gamma \\
# & \text{subject to}
# & & y^{(i)}(w^Tx^{(i)} + b) \geq \gamma, i = 1,...,m \\
# & & & ||w|| = 1
# \end{aligned}
# $$
# 
# What that means is that I want to maximize $\gamma$, subject to each training example having **functional margin** at least $\gamma$. And the constraint $||w|| = 1$ ensures that the **functional margin** equals to the **geometric margin**. So, we are also guaranteed that all the **geometric margins** are at least $\gamma$.
# 
# The problem with the above optimization equation is that the constraint $||w|| = 1$ is non-convex set. To see this let's see what this will give us in two dimensions. The constraint represents a set of points on the boundary of a circle (in n dimensions it will be a n-ball). 

# In[ ]:


c = plt.Circle((0.5, 0.5), 0.2, fill=False)
fig, ax = plt.subplots(figsize=(16, 10))
ax.add_artist(c)
plt.text(0.45, 0.5, '$||w|| = 1$', fontsize=22)


# Let's get rid of the non-convex constraint $||w|| = 1$. And redefine an equivalent optimization problem as:
# $$
# \begin{aligned}
# & max_{\hat{\gamma}, w, b}
# & & \frac{\hat{\gamma}}{||w||} \\
# & \text{subject to}
# & & y^{(i)}(w^Tx^{(i)} + b) \geq \hat{\gamma}, i = 1,...,m \\
# \end{aligned}
# $$
# 
# Here, we're going to maximize $\frac{\hat{\gamma}}{||w||}$, subject to **functional margins** all being at least $\hat{\gamma}$. Since, geometric margin and functional margin are related with $\gamma = \frac{\hat{\gamma}}{||w||}$, this gives us the answer we want. We are essentially normalizing the functional margin so that scaling won't affect it. The only problem we have with this is that $\frac{\hat{\gamma}}{||w||}$ is non-convex.

# In[ ]:


y = [1 for i in range(2450)] + [-1 for i in range(2550)]
np.random.shuffle(y)
x = np.random.randint(1, 1000 + 1, size=5000)
w = -1
b = 1
gamma_hat = (y * (w * x + b))
fig, ax = plt.subplots(figsize=(16, 10))
plt.scatter(x, gamma_hat)


# We know that we can scale **functional margin** with any scaling constraint $w$ and $b$ without changing anything. Now, we will introduce a scaling constraint $\hat{\gamma} = 1$. i.e. the functional margin of $w,b$ with respect to the training set must be $1$.
# 
# This is a scaling constraint and can be satisfied by rescaling $w,b$. 
# 
# Notice, that if $\hat{\gamma} = 1$ we will change the above objective function to $1/||w||$. This is same as minimizing $||w||^2$.
# 
# So, we get a convex optimization problem:
# $$
# \large{\begin{aligned}
# & min_{\gamma, w, b}
# & & \frac{1}{2}{||w||^2} \\
# & \text{subject to}
# & & y^{(i)}(w^Tx^{(i)} + b) \geq 1, i = 1,...,m \\
# \end{aligned}}
# $$
# 
# This optimization problem can be solved using any commercial quadratic programming (QP) code.

# # Using Langrange Duality
# Consider the **primal** problem of the following form:
# $$
# \begin{aligned}
# & min_{w}
# & & f(w) \\
# & \text{subject to}
# & & g_i(w) \leq 0, i = 1,...,k \\
# & & & h_i(w) = 0, i = 1,...,l \\
# \end{aligned}
# $$
# 
# To, solve this we start by defining the **generalized Lagrangian**
# $$
# \mathcal{L}(w, \alpha, \beta) = f(w) + \sum_{i=1}^{k}\alpha_ig_i(w) + \sum_{i=1}^{l}\beta_ih_i(w).
# $$
# 
# Here $\alpha_i's$ and $\beta_i$'s are called the **Lagrange multipliers**. We would then find and set $\mathcal{L}$'s partial derivatives to zero
# 
# $$\frac{\partial \mathcal{L}}{\partial w_i} = 0 ; \frac{\partial \mathcal{L}}{\partial \alpha_i} = 0; \frac{\partial \mathcal{L}}{\partial \beta_i} = 0, $$
# 
# and solve for $w, \alpha, \beta$.
# 
# Consider the quantity
# $$ \theta_{\mathcal{P}}(w) =  \mathop{max}_{\alpha,\beta:\alpha_i\geq0} \mathcal{L}(w, \alpha, \beta).$$
# 
# Here $\mathcal{P}$ subscript stands for **primal**.
# 
# Let some w be given, If w violates any of the primal constraints ($g_i(w) > 0\ or\ h_i(w) \neq 0\ for\ some\ i$). Then $\theta_{\mathcal{P}}(w) = \infty$.
# 
# Conversely, if the constrainta are indeed satisfied for a particular value of $w$, then $\theta_{\mathcal{P}} = f(w)$.
# 
# Hence,
# $$
# \theta_{\mathcal{P}}(w) = 
# \begin{cases}
#       f(w) &if\ w\ satisfies\ primal\ constraints,\\
#       \infty & otherwise\\
# \end{cases} 
# $$
# 
# Thus, $theta_{\mathcal{P}}$ takes the same value as the objective function in our problem for all values of w that satisfies the primal constraints, and is positive infinity if the constraints are violated. Hence, if we consider the minimization problem
# 
# $$ \mathop{min}_{w} \theta_{\mathcal{P}}(w) = \mathop{min}_{w} \mathop{max}_{\alpha,\beta:\alpha_i\geq0} \mathcal{L}(w, \alpha, \beta).$$
# 
# We see that it is the same problem (i.e. has the same solutions as) original, primal problem.
# 
# We also define the **value of primal problem** , 
# $$
# p^* = \mathop{min}_{w} \theta_{\mathcal{P}}(w) 
# $$
# 
# Now, let's look at a different problem, We define
# $$
# \theta_{\mathop{D}}(\alpha, \beta) = \mathop{min}_{w} \mathcal{L}(w, \alpha, \beta).
# $$
# 
# Here $\mathop{D}$ subscript stands for **dual**. Note also that whereas in the definition of $\theta_{\mathcal{P}}$ we were optimining with respect to $\alpha$, $\beta$, here we are minimizing with respect to $w$.
# 
# We can now propose the **dual** optimization problem.
# $$ \mathop{max}_{\alpha,\beta:\alpha_i\geq0} \theta_{\mathcal{D}}(\alpha, \beta) = \mathop{max}_{\alpha,\beta:\alpha_i\geq0} \mathop{min}_{w} \mathcal{L}(w, \alpha, \beta).
# $$
# 
# This is exactly the same as our primal problem shown above, except that the order of the **max** and **min** are now exchanged.
# 
# We also define
# $$d^* = \mathop{max}_{\alpha,\beta:\alpha_i\geq0} \theta_{\mathcal{D}}(\alpha, \beta)$$
# as the optimal **value** of the dual problem objective.
# 
# It can easily be shown that
# $$
# d^* \leq p^*
# $$
# As "max min" of a function is always less than or equal to the "min max".
# 
# However, under some conditions $d^* = p^*$, Lets see what these conditions are.
# 
# * Suppose $f$ and the $g_i$'s are convex, and the $h_i$'s are affine.
# * Suppose that the constraints $g_i$ are (strictly) feasible; this means that there exists some $w$ so that $g_i(w) < 0 \text{ for all i}$.
# 
# Under the above assumptions, there must exist $w^*$, $\alpha^*$, $\beta^*$ so that $w^*$ is the solution to the primal problem, $\alpha^*$, $\beta^*$ are the solution to the dual problem, and moreover $p^*$ = $d^*$ = $\mathcal{L}(w^*, \alpha^*, \beta^*)$.
# 
# Moreover, $w^*$, $\alpha^*$, $\beta^*$ satisfy the **Karush-Kuhn-Tucker (KKT) conditions**, which are as follows:
# 
# * Condition 1:
# $$
# \frac{\partial}{\partial w_i}\mathcal{L}(w^*, \alpha^*, \beta^*) = 0,\ \  i = 1,...,n
# $$
# 
# * Condition 2:
# $$
# \frac{\partial}{\partial \beta_i}\mathcal{L}(w^*, \alpha^*, \beta^*) = 0,\ \  i = 1,...,l
# $$
# 
# * **Condition 3**:
# $$
# \large{\alpha_{i}^{*}g_i(w^*) = 0, i = 1,...,k}
# $$
# 
# * Condition 4:
# $$
# g_i(w^*) \leq 0, i = 1,...,k
# $$
# 
# * Condition 5:
# $$
# \alpha^* \geq 0, i = 1,...,k
# $$
# 
# Moreover, if some $w^*$, $\alpha^*$, $\beta^*$ satisfy the KKT conditions, then it is also a solution to the primal and dual problems.
# 
# The third condition is called the KKT **dual complementarity** condition. Specifically, it implies that if $\alpha_i^* > 0$, then $g_i(w^*) = 0$. (i.e. the "$g_i(w) \leq 0$" constraint is active, meaning it holds with equality rathere than with inequality.)
# 

# # Optimal margin classifiers
# We previously, proposed the following (primal) optimization problem for finding the optimal margin classifier:
# $$
# \large{\begin{aligned}
# & min_{\gamma, w, b}
# & & \frac{1}{2}{||w||^2} \\
# & \text{subject to}
# & & y^{(i)}(w^Tx^{(i)} + b) \geq 1, i = 1,...,m \\
# \end{aligned}}
# $$
# 
# We can write the constraints as
# $$
# g_i(w) = -y^{(i)}(w^Tx^{(i)} + b) + 1 \leq 0, i = 1,...,m
# $$
# 
# We have one such constraint for each training example. Note that from the KKT dual complementarity condition, we will have $\alpha_i > 0$ only for the training examples that have functional margin exactly equal to one (i.e. the ones corresponding to constraints that hold with equality, g_i(w) = 0). Consider the figure below, in which a maximum margin separating hyperplane is shown by the solid line.
# 

# In[ ]:


X, y = make_blobs(n_samples=50, centers=2,
                  random_state=0, cluster_std=0.60)

model = SVC(kernel='linear', C=1E10)
model.fit(X, y)

fig, ax = plt.subplots(figsize=(16, 10))
plt.scatter(X[:, 0], X[:, 1], c=y, s=50)

xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
x = np.linspace(xlim[0], xlim[1], 30)
y = np.linspace(ylim[0], ylim[1], 30)
Y, X = np.meshgrid(y, x)
xy = np.vstack([X.ravel(), Y.ravel()]).T
P = model.decision_function(xy).reshape(X.shape)

# plot decision boundary and margins
ax.contour(X, Y, P, colors='k',
           levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])

# plot support vectors
ax.scatter(model.support_vectors_[:, 0],
           model.support_vectors_[:, 1],
           s=300, linewidth=1, facecolors='none');
ax.set_xlim(xlim)
ax.set_ylim(ylim)


# The points with the smallest margins are esactly the ones closese toe the decision boundary; here, these are the three points (one negative and two positive examples) that lie on the dashed lines parallel to the decision boundary.
# 
# Thus only three of the $\alpha_i$'s will be non-zero at the optimal solution to our optimization problem. These three points are called the **support vectors** in this problem.

# Let's construct the Lagrangian for our optimization problem:
# $$
# \mathcal{L}(w, b, \alpha) = \frac{1}{2} ||w||^2 - \sum_{i=1}^{m}\alpha_i [y^{(i)}(w^Tx^{(i)} + b) - 1]
# $$
# 
# We don't have $\beta_i$'s as we don't have any equality constraints.
# 
# Lets find the dual form of the problem. To do so, we need to first minimize $\mathcal{L}(w, b, \alpha)$ with respect to $w$ and $b$(for fixed $\alpha$), to get $\theta_\mathcal{D}$, which we'll do by setting the derivatives of $\mathcal{L}$ with respect to $w$ and $b$ to zero. We have:
# $$
# \nabla_w \mathcal{L}(w, b, \alpha) = w - \sum_{i=1}^{m}\alpha_iy^{(i)}x^{(i)} = 0
# $$
# This implies that
# $$
# w = \sum_{i=1}^{m}\alpha_iy^{(i)}x^{(i)} \hspace{50pt}(1)
# $$
# As for the derivative with respect to b, we obtain
# $$
# \frac{\partial}{\partial b_i} \mathcal{L}(w, b, \alpha) = \sum_{i=1}^{m}\alpha_iy^{(i)} = 0 \hspace{50pt}(2)
# $$
# If we take the definition of $w$ from equation (1) and plug that back into the Lagrangian, and simplify, we get
# $$
# \mathcal{L}(w, b, \alpha) = \sum_{i=1}^{m}\alpha_i - \frac{1}{2} \sum_{i,j=1}^{m}y^{(i)}y^{(j)}\alpha_i\alpha_j\langle x^{(i)}, y^{(j)}\rangle.
# $$
# 
# 
# $$
# \begin{aligned}
# & max_{\alpha}
# & & W(\alpha) =  \mathcal{L}(w, b, \alpha) = \sum_{i=1}^{m}\alpha_i - \frac{1}{2} \sum_{i,j=1}^{m}y^{(i)}y^{(j)}\alpha_i\alpha_j\langle x^{(i)}, y^{(j)}\rangle\\
# & \text{subject to}
# & & \alpha_i \geq 0, i = 1,...,m \\
# & & & \sum_{i=1}^{m}\alpha_iy^{(i)} = 0 \\
# \end{aligned}
# $$
# 
# Using some maths we can prove that conditions required for $p^* = d^*$ and the KKT conditions to hold are indeed satisfied in our optimization problem. Hence, we can **solve the dual** in lieu of solving the primal problem.
# 
# Specifically, in the dual problem above, we have a maximization problem in which the parameters are the $\alpha_i$'s.
# 

# Let us suppose that we use some algorithm (SMO) to get optimal values of $\alpha$'s i.e. find the $\alpha$'s that maximize $W(\alpha)$ subject to the constraints), the we can use Equation (1) to go back and find the optimal value of $w$'s as a function of the $\alpha$'s. Having found $w^*$, by considering the primal problem, it is also straightforward to find the optimal value for the intercept term b as
# $$
# b^* = -\frac{max_{i:y^{(i)}=-1}w^{*T}x^{(i)} + min_{i:y^{(i)}=1}w^{*T}x^{(i)}}{2}
# $$
# 
# Suppose we've fit our model's parameters to a training set, and now wish to make a prediction at a new point input x. We would then calculate $w^Tx + b$, and predict $y = 1$ iff this quantity is bigger than zero. But using equation (1) this quantity can also be written as:
# $$
# w^Tx + b = \Bigg(\sum^{m}_{i=1} \alpha_iy^{(i)}x^{(i)}\Bigg)^T x + b \\
# \large= \sum^{m}_{i=1} \alpha_iy^{(i)}\langle x^{(i)}, x\rangle + b
# $$
# 
# We know that **$\alpha_i$'s are non-zero only for the support vectors** and we were also able to write the entire algorithm in terms of only inner products between input feature vectors.

# ## Kernels
# Let's say instead of using just $x$ we use $x$, $x^2$ and $x^3$ as the features to obtain a cubic function. To distinguish between the two sets of variables, we'll call the original input value the input **attributes** of the problem. When that is mapped to some new set of quantities the input features.
# 
# We will also let $\phi$ denote the **feature mapping**, which maps from the attributes to the features. For instance in our example above, we had
# $$
# \phi(x) = \begin{bmatrix}
#            x \\
#            x^{2} \\
#            x^3
#           \end{bmatrix}
# $$
# 
# Rather than applying SVMs using the original input attributes x, we may insted want to learn using some feature $\phi(x)$. To do so, we simply need to go over our previous algorithm, and replace $x$ everywhere in it with $\phi(x)$.
# 
# Since the algorithm can be written entirely in terms of the inner products $\langle x,z\rangle$. Specifically, given a feature mapping $\phi$, we define the corresponding **Kernel** to be
# $$
# K(x,z) = \phi(x)^T\phi(z).
# $$
# 
# Then, everywhere we previously had $\langle x,z\rangle$ in our algorithm, we could simply replace it with $K(x,z)$, and our algorithm would now be learning using the features $\phi$.
# 
# Now, given $\phi$, we could easily compute $K(x,z)$ by finding $\phi(x)$ and $\phi(z)$ and taking their inner product. But what's more interesting is that often, $K(x,z)$ may be very inexpensive to calculate, even though $\phi(x)$ itself may be very expensive to calculate. In such settings, by using in our algorithm an efficient way to calculate $K(x,z)$, we can get SVMs to learn in the high dimensional feature space given by $\phi$, but without ever having to explicitly find or represent vectors $\phi(x)$.
# 
# Lets see an example. Suppose $x$, $z$ $\in R^n$, and consider
# $$
# K(x,z) = (x^Tz)^2.
# $$
# 
# We can also write this as 
# $$
# K(x,z) = \Bigg( \sum^{n}_{i=1}x_iz_i \Bigg) \Bigg(\sum^{n}_{j=1}x_jz_j\Bigg)\\
# = \sum^{n}_{i=1}\sum^{n}_{j=1} x_ix_jz_iz_j \\
# = \sum^{n}_{i,j=1}(x_ix_j)(z_iz_j)
# $$
# 
# Thus we see that $K(x,z) = \phi(x)^T\phi(z)$, where as mapping $\phi$ is given(shown here for the case of n=3) by
# $$
# \phi(x) = \begin{bmatrix}
#            x_1 x_1 \\
#            x_1 x_2 \\
#            x_1 x_3 \\
#            x_2 x_1 \\
#            x_2 x_2 \\
#            x_2 x_3 \\
#            x_3 x_1 \\
#            x_3 x_2 \\
#            x_3 x_3
#           \end{bmatrix}
# $$
# 
# Note that whereas calculating the high dimensional $\phi(x)$ requires $O(n^2)$ time, finding $K(x,z)$ takes only $O(n)$ time. Linear in the dimension of the input attributes.

# ### Gaussian kernel
# Let's talk about a slightly different view of kernels. Intutively, if $\phi(x)$ and $\phi(z)$ are close together, we might expect $K(x,z)$ to be large. Conversely, if $\phi(x)$ and $\phi(z)$ are far apart say nearly orthogonal to each other then $K(x,z) = \phi(x)^T\phi(x)$.
# 
# So we can think of $K(x,z)$ as some measurement of how similar $\phi(x)$ and $\phi(z)$ are, or of how similar are $x$ and $z$.
# 
# Given this intution, suppose that for some learning problem that you're working on, you've come up with some function $K(x,z)$ that you might think might be a resonable measure of how similar $x$ and $z$ are. For instance, you chose
# $$
# K(x,z) = exp\big(-\frac{||x-z||^2}{2\sigma^2}\big)
# $$
# This is a reasonable measure of x and z's similarity, and is close to 1 when x and z are close, and near 0 when x and z are far apart.
# This kernel corresponds to an **infinite dimensional feature mapping** $\phi$. If x is 1-D we have,
# $$
# \phi_{RBF}(x) = e^{-\gamma x^2}\big[1,\sqrt{\frac{2\gamma}{1!}}x, \sqrt{\frac{(2\gamma)^2}{2!}}x^2, \sqrt{\frac{(2\gamma)^3}{3!}}x^3,\ldots\big]^T,
# $$
# where, $\gamma = \frac{1}{2\sigma^2}$
# This is known as **Radial Basis Function (RBF)**.
# 
# Let's see what we can do with it:

# In[ ]:


fig, ax = plt.subplots(figsize=(16, 10))
def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

X, y = make_circles(100, factor=.1, noise=.1)

clf = SVC(kernel='linear').fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(clf, plot_support=False)


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 10))
r = np.exp(-(X ** 2).sum(1))

def plot_3D(elev=30, azim=30, X=X, y=y):
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap='autumn')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')

plot_3D(X=X, y=y)


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 10))
clf = SVC(kernel='rbf', C=1E6, gamma='auto')
clf.fit(X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(clf)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=300, lw=1, facecolors='none')


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 10))
clf = SVC(kernel='poly', degree = 2, C=1E6, gamma='auto')
clf.fit(X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(clf)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=300, lw=1, facecolors='none')


# In[ ]:


fig, axs = plt.subplots(figsize=(16, 50), nrows = 8)

for i in range(8):
    clf = SVC(kernel='poly', degree = i+3, C=1E6, gamma='auto')
    clf.fit(X, y)
    axs[i].set_title("Degree: " + str(i+3))
    axs[i].scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plot_svc_decision_function(clf, ax=axs[i])
    axs[i].scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                s=300, lw=1, facecolors='none')


# ### Theorem (Mercer):
# Let $K: R^n X R^n \rightarrow R$ be given. Then $K$ to be a valid(Mercer) kernel, it is neccessary and sufficient that for any $\{x^{(i)}, ..., x^{(m)}\}, (m < \infty)$, the corresponding kernel matrix is **symmetric positive semi-definite**.

# # Sources
# * http://cs229.stanford.edu/notes/cs229-notes3.pdf
# * https://www.csie.ntu.edu.tw/~cjlin/talks/kuleuven_svm.pdf
# * https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html
# * https://stats.stackexchange.com/questions/122631/how-does-the-phix-i-function-look-for-gaussian-rbf-kernel
# * https://stats.stackexchange.com/questions/80398/how-can-svm-find-an-infinite-feature-space-where-linear-separation-is-always-p/168309#168309
# * https://math.stackexchange.com/questions/1301585/why-is-the-constraint-w-1-non-convex/1301592
# * https://datascience.stackexchange.com/questions/6054/in-svm-algorithm-why-vector-w-is-orthogonal-to-the-separating-hyperplane
# 
