#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import seaborn as sns
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


n = 100
A = np.random.normal(0,1,(n,n))
B = A.copy()
# Dot product implemented in pure Python
def dot_py(A,B):
    N,D = A.shape
    D,M = B.shape
    out = np.zeros((N,D))
    for n in range(N):
        for m in range(M):
            for d in range(D):
                out[n,m] += A[n,d]*B[d,m]
    return out


# In[ ]:


### Univariate normal (Gaussian) distribution


X = rnd.normal(loc=0, scale=4, size=550)

#or using scipy package
from scipy.stats import norm 

X = norm.rvs(loc=0,scale=4,size=550)

plt.figure(figsize=(8,4))
# Histogram plot 
plt.hist(X,bins=30, rwidth=0.9, alpha=0.5, density=True)
plt.grid()
_=sns.kdeplot(X, color='r')


# In[ ]:


### Numpy Usage Example: Univariate normal (Gaussian) PDF


# <center>
# $\boxed{\small{\mathcal{N}(x | \mu, \sigma)} = 
# \frac{1}{\sqrt{2\pi\sigma^2}}\exp{\Big(-\frac{(x - \mu)^2}{2\sigma^2}\Big)}}$

# In[ ]:


def normal_pdf(X, mean=0, sigma=1):
    return 1./np.sqrt(2*np.pi*sigma**2)*np.exp(-(X - mean)**2/(2*sigma**2))


# In[ ]:


X = np.linspace(-5,8,100)
plt.figure(figsize=(6,3))
plt.grid()
plt.axvline(1,c='r',ls='--', label='$\\mu$')
_=plt.plot(X,normal_pdf(X,1,2),label='pdf')
_=plt.legend()


# In[ ]:


X = np.linspace(-5,8,100)
plt.figure(figsize=(6,3))
plt.grid()
plt.axvline(1,c='r',ls='--', label='$\\mu$')
_=plt.plot(X,normal_pdf(X,1,2),label='pdf')
_=plt.legend()


# In[ ]:


m = 1
sigma = 4
print(normal_pdf(X,m,sigma)[:3], norm.pdf(X,m,sigma)[:3]) # print the values just for first three elements


# In[ ]:


assert np.allclose(normal_pdf(X,m,sigma), norm.pdf(X,m,sigma)) # make sure all of the results are the same


# ### Analyse for different parameter values ( i.e. means and variances )

# In[ ]:


# Define some means and standart deviations
mu = [-5,0,5]
K = len(mu)
sigma2 = np.power(K*[2], np.arange(1,K+1))
print(mu,sigma2)


# In[ ]:


import matplotlib as mpl
# set some global plot settings
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['figure.figsize'] = (10,8)

# define the grid of the canvas (plot) 
fig,axs = plt.subplots(K,K,sharey=True,figsize=(8,8))

X = np.linspace(-10,10,100) # define some testing values to calculate Gaussian PDF
K = len(mu)
for i in range(K):
    for j in range(K):
        ax = axs[i,j] 
        m = mu[i]
        s2 = sigma2[j]
        pdf = normal_pdf(X,mean=m,sigma=s2)
        ax.plot(X,pdf, label=f'$\mu$ = {m} \n$\sigma^2$ = {s2}')
        ax.axvline(m,ls='--',c='g')
        ax.grid()
        ax.legend(loc=1)
axs[2,1].set_xlabel('$x$',fontsize=15)
axs[1,0].set_ylabel('$pdf(x)$', fontsize=15)
fig.tight_layout()


# ## Coin flipping example 
# 
# 

# #### Model distribution (likelihood)
# <br>
# <center>
# $
# \large{p( h,N | \theta) = \text{Binomial}(h,N |\theta) = \binom{N}{h} \theta^{h} (1 - \theta)^{N-h}}
# $
# </center>
# <hr>

# In[ ]:


theta_real = 0.35 # true theta value for the Binomial distribution\


# In[ ]:


# Observed data

trials = [0, 1, 2, 3, 4, 8, 16, 32, 50, 150, 210, 270, 330] # number of trials
data = [0, 1, 1, 1, 1, 4, 6, 9, 13, 48, 78, 96, 118]  # number of the observed heads


# #### Prior distribution
# <center>
# $
# \large{p(\theta | \alpha, \beta) = \text{Beta}(\alpha,\beta)= \frac{1}{B(\alpha,\beta)}}\theta^{\alpha-1}(1 - \theta)^{\beta-1}, \quad B(\alpha,\beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}
# $
# <br>

# ### Beta distribution

# In[ ]:


from scipy.stats import beta

alphas = [1, .1, 5]
betas = [1, .1, 5]

x = np.linspace(0, 1, 200)
K = len(alphas)
fig, axs = plt.subplots(K, K, sharex=True, sharey=True, figsize = (10,10))
for i in range(K):
    for j in range(K):
        alpha = alphas[i]
        bet = betas[j]
        rv = beta(alpha, bet)
        ax = axs[i,j]
        ax.plot(x, rv.pdf(x))
        ax.plot(0, 0, label="$\\alpha$ = {:3.2f}\n$\\beta$ = {:3.2f}".format(alpha, bet), alpha=0)
        ax.axvline(alpha/(alpha + bet),ls='--',c='g')
        ax.legend()
        ax.grid()
axs[2,1].set_xlabel('$x$',fontsize=15)
axs[1,0].set_ylabel('$pdf(x)$', fontsize=15)
fig.tight_layout()


# #### Posterior distribution
# <center>
#     $\large{\overbrace{p(\theta | h, \alpha, \beta)}^{\text{posterior}} \propto  \overbrace{\theta^{h} (1 - \theta)^{N-h}}^{\text{likelihood}} \overbrace{\theta^{\alpha-1}(1 - \theta)^{\beta-1}}^{\text{prior}}
#     }$
# <br><br>

# <center>
# $
# \large{p(\theta | h, \alpha, \beta) \propto \theta^{ h + \alpha -1 }(1 - \theta)^{N - h + \beta-1} }
# $
# </center>
#     <hr>
# <center>
# $
# \large{p(\theta | h, \alpha, \beta) = \text{Beta}(\hat{\alpha},\hat{\beta}), \text{where} \quad \hat{\alpha} = h+\alpha, \quad \hat{\beta} = N - h + \beta }
# $

# In[ ]:


beta_params = [(1, 1), (0.1, 0.1), (5, 5)] # Beta distiribution parameters used as priors
x = np.linspace(0, 1, 100)


# In[ ]:


fig = plt.figure(figsize=(16,12))
for idx, N in enumerate(trials):
    if idx == 0:
        plt.subplot(5, 3, 2)
    else:
        plt.subplot(5, 3, idx+3)
    heads = data[idx]
    for (alpha_prior, beta_prior), c in zip(beta_params, ('b', 'r', 'g')):
        alpha_hat = heads + alpha_prior
        beta_hat = N - heads + beta_prior
        p_post = beta.pdf(x, alpha_hat, beta_hat)
        plt.plot(x, p_post, c)
        plt.grid(axis='x')
        plt.fill_between(x, 0, p_post, color=c, alpha=0.1)
    plt.axvline(alpha_hat/(alpha_hat + beta_hat),ymax=1, c='r',linestyle='--')
    plt.axvline(theta_real, ymax=1, color='g',linestyle='--')
    plt.plot(0, 0, label="{:d} experiments\n{:d} heads".format(N,heads), alpha=0)
    plt.xlim(0,1)
    plt.xlabel(r'$\theta$', fontsize=15)
    plt.legend()
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.tight_layout()


# # Mulivariate Distributions

# * ## Numpy Usage Example: Gaussian (multivariate) PDF 
# <br><br>
# <center>
# $
# \boxed{
#     {\mathcal{N}(x | \mu, \Sigma)} = 
#     \frac{1}{\sqrt{(2\pi)^d |\Sigma|}}
#     \exp\Big(-\frac{1}{2} (x - \mu)^{\top}\Sigma^{-1}(x-\mu)\Big)
#     , \quad \mu \in \mathbb{R}^{(d)}, \quad \Sigma \in \mathbb{R}^{(d,d)}
# }
# $

# In[ ]:


import numpy.linalg as la
def multivariate_normal_pdf(X, mean=np.zeros(2), cov = 1*np.eye(2)):
    N,d = X.shape
    Xm  = X - mean[None] # (N,d) - (1,d)
    C = 1./np.sqrt(2*np.pi**d*np.linalg.det(cov))
    lst = [C*np.exp(-0.5*(Xm[i]).T @ la.inv(cov) @ (Xm[i])) for i in range(N)]
    return np.array(lst)


# ### Multivariate normal (Gaussian) distribution (d=2)   
# 

# In[ ]:


mu = np.zeros(2)
cov = np.eye(2)
X = rnd.multivariate_normal(mu,cov,size=100)
print('Shape:',X.shape)
print(X[:5])


# In[ ]:


from matplotlib import cm
def draw_plot_gaussian(mu,cov,n=500):
    X = rnd.multivariate_normal(mu,cov,size=n)
    cs = multivariate_normal_pdf(X,mu,cov)
    plt.figure()
    plt.scatter(*X.T,marker='*',c=cs, cmap=cm.rainbow,
                label=f'$\mu$ = {np.round(X.mean(0))}\n$\sigma$={np.round(X.std(0))}')
    plt.scatter(*X.mean(0),color='k',linewidth=5)
    plt.margins(0.2)
    plt.grid()
    _=plt.legend()


# In[ ]:


draw_plot_gaussian(mu,cov)


# In[ ]:


s1 = 1
s2 = 4
ps = [0.2,-0.5,0.8]
for p in ps:
    cov = np.array(
        [
            [s1*s1,p*s1*s2],
            [p*s2*s1,s2*s2]
        ])
    draw_plot_gaussian(mu,cov)


# ### Dirichlet distribution ( distribution of distributions ) Multivariate Beta

# <center>
# $
# \boxed{
#     {
#         K \geq 2, \quad \alpha = (\alpha_1,...,\alpha_K), \quad \alpha_i > 0 \\
#         x_i \in (0,1) \quad \text{and} \quad \sum_{i=1}^K x_i=1, \quad \hat{\alpha} = \sum_{i=1}^K \alpha_i \\
#         \mathcal{Dir}(x|\alpha) = \mathcal{C}(\alpha)\prod_{i=1}^K x_i^{\alpha-1}, 
#         \quad \mathcal{C}(\alpha) = \frac{\Gamma(\hat{\alpha})}{\prod_{i=1}^K \Gamma(\alpha_i)}, \quad
#         \Gamma(z) =  \int_0^{\infty} x^{z-1}e^{-x}dx
#     }
# }
# $

# In[ ]:


from scipy.special import gamma

def dirichlet_pdf(X, alpha):
    assert np.all(X > 0) & np.all(X < 1)
    assert np.all(alpha > 0)
    assert len(alpha.shape) == 1
    assert len(alpha) >= 2
    assert X.shape[0] == len(alpha)

    B = np.prod(gamma(alpha))/gamma(np.sum(alpha))

    # K,N ** K
    return np.prod(X ** (alpha[:,None]-1), 0) / B


# In[ ]:


K = 5
X = np.abs(rnd.normal(0,1,size=(K,100)))
X = X / X.sum(0)[None]
X[:,:5],np.sum(X[:,:5],0)


# In[ ]:


alpha = np.ones(K)
dirichlet_pdf(X[:,:2], alpha)


# In[ ]:


# There is a PDF implementation in SciPy package 
from scipy.stats import dirichlet
alpha = np.ones(K)
dirichlet.pdf(X[:,:1],alpha)


# In[ ]:


# Sample from the Dirichlet distribution
dirichlet.rvs(alpha,size=3)


# In[ ]:


assert np.allclose( dirichlet_pdf(X,alpha), dirichlet.pdf(X,alpha))

