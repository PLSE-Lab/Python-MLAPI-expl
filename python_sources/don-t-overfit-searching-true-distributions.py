#!/usr/bin/env python
# coding: utf-8

# # Our Mission
# 
# We have neither won the competition nor yielded a high score on the leaderboard. Even though we want to share our idea how to model the true distributions that may have been used to generate the data. We assumed that the competition sponsors used 2 gaussian distributions to draw random samples - a small part given as train and a big part given as test. To solve the task we tried to learn how these gaussians look like in sense of their mean and variance values. We setup a semi-supervised gaussian mixture model that starts with intial values for means and variances using the train data. Our idea was then to adjust these values by learning how the gaussians fit the test data best. Hence instead of leaderboard probing we tried to use the information that is already provided by the test data...
# 
# Unfortunately our model does mad things and learning was not as expected. We are now on our journey to find out what has happened and we like to share our ideas with you. :-) Please feel free to comment, let us know what has negatively influenced the learning process in your opinion. 
# 
# Let's share learning experiences! Happy kaggling ;-)
# 
# **Caution: Still work in progress**
# 
# ## Table of contents
# 
# 1. Preparation
# 2. Exploring the data structure
# 3. Recursive feature elimination with logistic regression
# 4. Trying to catch the true distributions
# 5. Fit on Chris Deottes Top Useful Features
# 6. What have we learnt?

# ## Preparation

# ### Loading packages

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import StratifiedKFold


import seaborn as sns
sns.set()

import os
print(os.listdir("../input"))

# ignore warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ### Loading data

# In[ ]:


train = pd.read_csv("../input/train.csv", index_col=0)
train.shape


# In[ ]:


test = pd.read_csv("../input/test.csv", index_col=0)
test.shape


# In[ ]:


test.shape[0]/train.shape[0]


# Wuaahh! 79 times more samples in test than in train! :-)

# In[ ]:


submission = pd.read_csv("../input/sample_submission.csv")
submission.head()


# ## Exploring the data structure

# In[ ]:


combined = train.drop("target", axis=1).append(test)
corr = combined.corr().values.flatten()
corr = corr[corr !=1]


fig, ax = plt.subplots(3,2, figsize=(20,20))
sns.countplot(train.target, ax=ax[0,0], palette="Paired")
ax[0,0].set_title("Target distribution")
ax[0,0].set_xlabel("")

sns.distplot(corr, ax=ax[0,1], color="Darkorange")
ax[0,1].set_title("Feature correlation values in combined data")

sns.distplot(test.mean(), ax=ax[1,0], color="mediumseagreen")
ax[1,0].set_title("Distribution of feature means in test")

sns.distplot(test.std(), ax=ax[1,1], color="mediumseagreen");
ax[1,1].set_title("Distribution of feature stds in test")

sns.distplot(train.drop("target", axis=1).mean(), ax=ax[2,0], color="orangered")
ax[2,0].set_title("Distribution of feature means in train")

sns.distplot(train.drop("target", axis=1).std(), ax=ax[2,1]);
ax[2,1].set_title("Distribution of feature stds in train")


# ### Insights
# 
# * We have to deal with imbalanced classes!
# * The features of the combined data (train and test) look very decorrelated. (No feature interactions)
# * The test data leads to the assumption that most of the feature distributions have zero mean and unit variance. 
# * This is confirmed for the train data as well even though we find a higher spread. 

# In[ ]:


train[train.target==0].shape[0] / train.shape[0]


# ### Recursive feature elimination with logistic regression
# 
# As we can see no relevant feature correlations, let's assume that the data can be separated linearily into both classes. In this case logistic regression can do a very good job and we can use it as well to find most important features.  

# In[ ]:


X = train.drop("target", axis=1).values
y = train.target.values


# #### Settings
# 
# I'm going to use Chris Deottes settings on logistic regression with l1-regularisation because I hope that this will reveal better insights than we obtained with our hyperparameters. 

# In[ ]:


n_splits = 25
solver = "liblinear"
penalty = "l1"
C=0.1
class_weight="balanced"


# #### Fit with all features

# In[ ]:


skf= StratifiedKFold(n_splits=n_splits, random_state=0)
lr = LogisticRegression(solver=solver, penalty=penalty, C=C, class_weight=class_weight)
scores = []

for train_idx, test_idx in skf.split(X, y):
    x_train, x_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    lr.fit(x_train, y_train)
    proba = lr.predict_proba(x_test)
    scores.append(roc_auc_score(y_test, proba[:,1]))


# In[ ]:


np.mean(scores)


# In[ ]:


np.std(scores)


# What does logistic regression yield on public and private LB with all 300 features? 

# #### Recursive feature elimination
# 
# In our team we recursively dropped features by starting with the least important feature logistic regression had found and we can extract by looking at the model weights:

# In[ ]:


base_importance = np.abs(lr.coef_)[0]
importance = np.argsort(base_importance)


# In[ ]:


to_try = train.drop("target", axis=1)

mean_drop_scores = []
std_drop_scores = []
for feat in importance[0:-1]:
    to_try = to_try.drop(str(feat), axis=1)
    X = to_try.values
    
    lr = LogisticRegression(solver=solver, C=C, penalty=penalty, class_weight=class_weight)
    stk = StratifiedKFold(n_splits=n_splits, random_state=0)
    scores = []
    for train_idx, test_idx in stk.split(X, y):
        x_train, x_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    
        lr.fit(x_train, y_train)
        proba = lr.predict_proba(x_test)
        scores.append(roc_auc_score(y_test, proba[:,1]))
    
        
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    mean_drop_scores.append(mean_score)
    std_drop_scores.append(std_score)


# In[ ]:


plt.figure(figsize=(20,5))
plt.plot(np.arange(0, len(importance[0:-1])), mean_drop_scores, 'o-', color="red")
plt.fill_between(np.arange(0, len(importance[0:-1])),
                 mean_drop_scores-np.array(std_drop_scores),
                 mean_drop_scores+np.array(std_drop_scores), color="tomato", alpha=0.2)


# We can clearly see that there are less than 50 features that were selected as important by logistic regression with Chris hyperparameters (In our original solution we obtained almost 50). What are the names of these vars and how many do overlap with Chris final useful vars?

# In[ ]:


important_features = importance[250::]
important_features = important_features.astype(np.str)
important_features


# In[ ]:


chris_important_features = ["239", "217", "209", "199", "189",
                            "164", "117", "108", "91", "73",
                            "65", "63", "45", "33", "16"]


# In[ ]:


set(chris_important_features).difference(important_features)


# That's interesting! Even though we used only 50 most important features of log reg from which only ~ 30 show an impact on the solution, there are still 4 features Chris has found with LB probing that we haved missed! ;-)

# What can logistic regression yield on the LB with our top 30 important features and Chris ones?

# ## Trying to catch the true distributions
# 
# Let's take a look at some scatter and kde-plots to highlight what we are going to assume in the next section:

# In[ ]:


important_features = important_features[::-1]


# In[ ]:


fig, ax = plt.subplots(3,4,figsize=(20,15))
for m in range(1,5):
    ax[0,m-1].scatter(train.loc[train.target==1, important_features[0]].values,
                    train.loc[train.target==1, important_features[m]].values,
                    c="red")
    ax[0,m-1].scatter(train.loc[train.target==0, important_features[0]].values,
                    train.loc[train.target==0, important_features[m]].values,
                    c="blue")
    ax[0,m-1].set_xlabel(important_features[0])
    ax[0,m-1].set_ylabel(important_features[m])
    
    sns.kdeplot(train.loc[train.target==1, important_features[0]],
                train.loc[train.target==1, important_features[m]],
                ax=ax[1,m-1], shade=True, shade_lowest=False, cmap="Reds", alpha=0.3)
    sns.kdeplot(train.loc[train.target==0, important_features[0]],
                train.loc[train.target==0, important_features[m]],
                ax=ax[1,m-1], shade=True, shade_lowest=False, cmap="Blues", alpha=0.3)
    
    ax[2,m-1].scatter(test.loc[:, important_features[0]].values,
                      test.loc[:, important_features[m]].values,
                      c="mediumseagreen",s=0.05, alpha=0.5)
    ax[2,m-1].set_xlabel(important_features[0])
    ax[2,m-1].set_ylabel(important_features[m])


# #### Insights
# 
# * We can see that the 5 selected most important features by our logistic regression model show slight deviations from the 0 mean we have observed.
# * We can't see a clear bimodality in the densities of the test data with out eyes. 

# Let's see how the mean values and standard deviations spread from 0 mean and unit variance with respect to the importance given by logistic regression:

# In[ ]:


hot_means = train[train.target==1].drop("target", axis=1).mean()
cold_means = train[train.target==0].drop("target", axis=1).mean()

hot_stds = train[train.target==1].drop("target", axis=1).std()
cold_stds = train[train.target==0].drop("target", axis=1).std()
feature_importance_value = base_importance

fig, ax = plt.subplots(2,2,figsize=(20,20))
ax[0,0].scatter(hot_means, cold_means, c=feature_importance_value, s=50, cmap="Greens")
ax[0,1].scatter(hot_stds, cold_stds, c=feature_importance_value, s=50, cmap="Greens");
ax[0,0].set_xlim([-0.7, 0.7])
ax[0,0].set_ylim([-0.7, 0.7])
ax[0,0].set_xlabel("feature means of hot samples")
ax[0,0].set_ylabel("feature means of cold samples")
ax[0,1].set_xlabel("feature stds of hot samples")
ax[0,1].set_ylabel("feature stds of cold samples")
ax[0,1].set_xlim([0.7, 1.3])
ax[0,1].set_ylim([0.7, 1.3])

ax[1,0].scatter(hot_means, hot_stds, c=feature_importance_value, s=50, cmap="Reds")
ax[1,0].set_xlabel("Hot mean values")
ax[1,0].set_ylabel("Hot std values")
ax[1,1].scatter(cold_means, cold_stds, c=feature_importance_value, s=50, cmap="Blues");
ax[1,1].set_xlabel("Cold mean values")
ax[1,1].set_ylabel("Cold std values")


# ### Semi-supervised gaussians
# 
# Imagine the case where no leaderboard is given to probe... **What would you do in this case?** You have only a very small number of samples in your data with class labels (like in train) and much more unlabeled data (like test) but you need to build up a classifier. What can you do instead of logistic regression? What can you try out to cover the gaussian nature of this data?
# 
# We built up a few assumptions about the data structure to find a model that does not only try to make nice predictions but also explains how the data was generated:
# 
# * The data per class was drawn using a multivariate gaussian distribution.
# * As features are almost decorrelated we expect the covariance matrix to be diagonal but with own standard deviations for each single feature. This way the multivariate gaussian turns into a factorization of single feature gaussians. 
# * We guess that the mean values of unimportant features is zero whereas those of important ones slightly differ from this value.  
# * Using these assumptions we can built up a gaussian mixture model that tries to fit the test data best during learning. 
# * We don't want to waste our training data and like to use class labels as guidance to fit the two gaussians.
# 
# 
# Before this competiton we had no knowledge how to setup a semi-supervised mixture model but with help of this [nice paper](https://www.cs.ubc.ca/~schmidtm/Courses/540-W16/EM.pdf) we were able to derive the learning equations for the means and standard deviations for each feature of the gaussians per target class:
# 
# #### The model
# 
# We like to describe our observed data (train features and targets, test features) by a latent variable model that uses two latent variables as representatives for our hidden test targets:
# 
# $$p(X,\hat{X}, Y) = \pi_{0} \cdot N(X,\hat{X}, Y|\mu_{0}, \sigma_{0}) + \pi_{1} \cdot N(X,\hat{X}, Y|\mu_{1}, \sigma_{1})$$
# 
# To find optimal parameters for the two gaussians that can describe our data best we need to maximize the equation with respect to the parameters. We can't do this directly and need the EM-algorithm to solve this task and for that we need the complete data log likelihood. Fortunately the paper is of great help to set it up:
# 
# $$ \mathbb{E}_{\hat{y}|X,Y,\hat{X},\theta^{old}} \ln p(X, Y, \hat{X}, \hat{Y}|\theta) = \sum_{n} p(x_{n}, y_{n}|\theta) + \sum_{m} \sum_{k} \gamma_{km} \ln p(\hat{x}_{m}, \hat{y}_{m}|\theta)$$
# 
# The first part with $n$ describes the contribution of the training data to a gaussian distribution $k$ whereas the second part with $m$ describes the contribution of the unlabeled test data. 
# 
# #### E-Step
# 
# The E-Step tries to estimate how probable it is that a data point "m" was drawn by the current gaussian "k". The associated value is called responsibility. We update these responsibilities during the E-Step by using only test data spots $\hat{x}_{m}$:
# 
# $$ \gamma_{mk} = \frac{\pi_{k} N(\hat{x}_{m}|\mu_{k}, \sigma_{k})}{\pi_{0} \cdot N(\hat{x}_{m}|\mu_{0}, \sigma_{0}) + \pi_{1} \cdot N(\hat{x}_{m}|\mu_{1}, \sigma_{1})}$$
# 
# #### M-Step
# 
# We recompute the mean of each component and the effective number using the data spots of train and test as well as the responsibilities:
# 
# $$M_{eff, k} = \sum_{m} \gamma_{mk}$$
# 
# $$\mu_{k} = \frac{\sum_{n} x_{n} + \sum_{m} \gamma_{mk} \hat{x}_{m}}{N_{k}+M_{k}} $$
# 
# $$\sigma_{k} = \frac{1}{N + M_{k}} \cdot \left( \sum_{n} (x_{n}-\mu_{k})^{2} + \sum_{m} \gamma_{km} (\hat{x}_{m} - \mu_{k})^{2} \right)$$
# 
# $$\pi_{k} = \frac{N+M_{eff,k}}{N+M_{total}} $$
# 
# 

# In[ ]:


class MySemiGMM:
    
    def __init__(self, mu, sigma):
        
        self.sigma = sigma
        self.mu = mu
        self.old_mus = []
        self.old_sigmas = []
        self.old_pis = []
        self.train_roc_convergence = []
        self.pi = [0.36, 0.64]
    
    def set_data(self, x_test, x_train, y_train):
        self.x_test = x_test
        self.y_train = y_train
        self.x_train = x_train
        self.hot_data = x_train[np.where(y_train==1)]
        self.cold_data = x_train[np.where(y_train==0)]
        self.n_hot = len(self.hot_data)
        self.n_cold = len(self.cold_data)
    
    def update_Meff(self):
        self.Meff = np.sum(self.gamma, axis=0)
        
    def update_pars(self):
        denominator_cold = self.n_cold + self.Meff[0]
        counter_cold = np.sum(self.cold_data, axis=0)
        counter_cold += np.einsum('id,i -> d', self.x_test, self.gamma[:,0])
        mu_cold = counter_cold/denominator_cold
        
        denominator_hot = self.n_hot + self.Meff[1]
        counter_hot = np.sum(self.hot_data, axis=0)
        counter_hot += np.einsum('id,i -> d', self.x_test, self.gamma[:,1])
        mu_hot = counter_hot/denominator_hot 
        
        self.mu[0] = mu_cold
        self.mu[1] = mu_hot
        
        cold_var = np.sum(np.power(self.x_train - self.mu[0], 2), axis=0) 
        cold_var += np.einsum('id,i -> d', np.power(self.x_test-self.mu[0], 2), self.gamma[:,0])
        sigma_cold = np.sqrt(1/(self.n_cold + self.Meff[0]) * cold_var)
        
        hot_var = np.sum(np.power(self.x_train - self.mu[1], 2), axis=0) 
        hot_var += np.einsum('id,i -> d', np.power(self.x_test-self.mu[1], 2), self.gamma[:,1])
        sigma_hot = np.sqrt(1/(self.n_hot + self.Meff[1]) * hot_var)
        
        self.sigma[0] = sigma_cold
        self.sigma[1] = sigma_hot
        
        self.pi[0] = (self.n_cold + self.Meff[0]) / (self.x_train.shape[0] + self.x_test.shape[0])
        self.pi[1] = (self.n_hot + self.Meff[1]) / (self.x_train.shape[0] + self.x_test.shape[0])
    
    def learn(self, n_iter=10):
        # E-Step 
        self.old_mus.append(self.mu.copy())
        self.old_sigmas.append(self.sigma.copy())
        self.old_pis.append(self.pi.copy())
        for n in range(n_iter):
            self.gamma = self.get_gamma(self.x_test)
            # M-Step
            self.update_Meff()
            self.update_pars()
            self.old_mus.append(self.mu.copy())
            self.old_sigmas.append(self.sigma.copy())
            self.old_pis.append(self.pi.copy())
            score = roc_auc_score(self.y_train, self.predict(self.x_train)[:,1])
            if n % 1 == 0:
                print(n, score)
            self.train_roc_convergence.append(score)
        return self.mu
    
    def get_single_normal_proba(self, x, my_mu, my_sigma, my_pi):
        sigma2 = np.power(my_sigma,2)
        return my_pi * np.prod(1/np.sqrt(2*np.pi*sigma2) * np.exp(-0.5 * (1/sigma2) * np.power((x-my_mu),2)))
    
    def get_gamma(self, X):
        probabilities = np.zeros(shape=(X.shape[0], 2))
        for n in range(X.shape[0]):
            norm_value = self.get_single_normal_proba(X[n,:], self.mu[0], self.sigma[0], self.pi[0]) 
            norm_value += self.get_single_normal_proba(X[n,:], self.mu[1], self.sigma[1], self.pi[1])
            for m in range(len(self.mu)):
                probabilities[n,m] = self.get_single_normal_proba(X[n,:], self.mu[m], self.sigma[m], self.pi[m])
                probabilities[n,m] /= norm_value
        return probabilities
    
    def predict(self, X):
        return self.get_gamma(X)


# ### Semi-Sup with all features 

# In[ ]:


important_features = train.drop("target", axis=1).columns.values


# In[ ]:


m1 = train.loc[train.target==0, important_features].mean().values
m2 = train.loc[train.target==1, important_features].mean().values
mu = np.array([m1, m2])
old_mu = mu.copy()


s1 = train.loc[train.target==0, important_features].std().values
s2 = train.loc[train.target==1, important_features].std().values
sigma = np.array([s1,s2])
old_sigma = sigma.copy()

x_train = train.loc[:, important_features].values
y_train = train.target.values
x_test = test.loc[:, important_features].values

model = MySemiGMM(mu, sigma)
model.set_data(x_test, x_train, y_train)
new_mu = model.learn()

proba_test = model.predict(x_test)
proba_train = model.predict(x_train);


# In[ ]:


c1 = 33
c2 = 65 

selection = important_features

fig, ax = plt.subplots(2,3,figsize=(20,10))

for l in range(len(model.old_mus)-1):
    old_mu = model.old_mus[l]
    new_mu = model.old_mus[l+1]

    ax[0,0].scatter(old_mu[0,c1], old_mu[0,c2], color="blue", marker='+')
    ax[0,0].scatter(new_mu[0,c1], new_mu[0,c2], color="blue", marker='+')
    ax[0,0].scatter(old_mu[1,c1], old_mu[1,c2], color="red", marker='+')
    ax[0,0].scatter(new_mu[1,c1], new_mu[1,c2], color="red", marker='+')
    ax[0,0].set_xlabel(selection[c1])
    ax[0,0].set_ylabel(selection[c2])
    #ax[0].set_xlim([-0.4,0.4])
    #ax[0].set_ylim([-0.4,0.4])
    ax[0,0].set_title("How the means move")
    
    old_sigma = model.old_sigmas[l]
    new_sigma = model.old_sigmas[l+1]

    ax[0,1].scatter(old_sigma[0,c1], old_sigma[0,c2], color="blue")
    ax[0,1].scatter(new_sigma[0,c1], new_sigma[0,c2], color="blue")
    ax[0,1].scatter(old_sigma[1,c1], old_sigma[1,c2], color="red")
    ax[0,1].scatter(new_sigma[1,c1], new_sigma[1,c2], color="red")
    ax[0,1].set_xlabel(selection[c1])
    ax[0,1].set_ylabel(selection[c2])
    ax[0,1].set_title("How the stds move")
    
    old_pi = model.old_pis[l]
    new_pi = model.old_pis[l+1]
    
    ax[0,2].scatter(old_pi[0], old_pi[0], color="blue")
    ax[0,2].scatter(new_pi[0], new_pi[0], color="blue")
    ax[0,2].scatter(old_pi[1], old_pi[1], color="red")
    ax[0,2].scatter(new_pi[1], new_pi[1], color="red")
    ax[0,2].set_xlabel(selection[c1])
    ax[0,2].set_ylabel(selection[c2])
    ax[0,2].set_title("How the priors pi move")
    
ax[1,0].scatter(x_test[:,c1], x_test[:,c2], c=proba_test[:,1], cmap="coolwarm", s=2)
ax[1,0].set_xlabel(selection[c1])
ax[1,0].set_ylabel(selection[c2])
ax[1,0].set_title("Test data")

ax[1,1].scatter(x_train[:,c1], x_train[:,c2], c=proba_train[:,1], cmap="coolwarm")
ax[1,1].set_xlabel(selection[c1])
ax[1,1].set_ylabel(selection[c2])
ax[1,1].set_title("Predicted proba train")

ax[1,2].scatter(x_train[:,c1], x_train[:,c2], c=y_train, cmap="coolwarm")
ax[1,2].set_title("True target values in train");
ax[1,2].set_xlabel(selection[c1])
ax[1,2].set_ylabel(selection[c2]);


# ### Insights
# 
# * The roc-score on the train data decreases. This is not really what we want as we like to obtain a good model for both: test and train data.  
# * Our **model tries to explain the data with only one major gaussian**. This is of course not what we wanted as we prefer a model that can describes both: the density of the data and the targets in train. 
# 
# 
# ### Trying to understand what happened
# 
# * Our start values are given by the means $\mu$ and variances $\sigma^{2}$ per feature in the train data with respect to targets and the priors $\pi$ by the fraction of classes in train. They define our initial gaussian distributions.
# * The first step our model takes is to compute how responsible each gaussian for a sample point in test. As the distributions are very close to each other and the prior for the gaussian that we like to explain hot targets is higher, we obtain higher probabilities for the $1$-gaussian for the majority of samples than for the other gaussian. 
# * These high $\gamma_{m,1}$ values cause high shifts in the learnt parameters $\mu$, $\sigma$, $\pi$ and $M_{eff}$. This way we yield an even higher prior for $\pi_{1}$ than before and the situation becomes more dramatic than before. A cycle of **"the winner takes it all" starts**. 
# * But even in the case when we start with a high prior for the zero-component gaussian the one-component gaussian takes it all. 
# * Let's take a look at the mean value: In the counter it consists of a sum computed on the train data plus a $\gamma$ weighted sum computed on the test data. As training data is small, the first, **supervised part has only a small contribution to the learning process whereas the second, unsupervised part has a huge impact on the new mean value**. The same holds for the variance. How to solve that? I don't know if this is ok in sense of probabilities (have to check), but what about weighting the parts differently, like $0.9 * supervised + 0.1 * unsupervised$? But how to change the effective number $M_{eff}$ of samples in the current gaussian? Perhaps it's better to derive the equations using probabilities. 

# In[ ]:


fig, ax = plt.subplots(1,1,figsize=(20,5))
ax.plot(model.mu[1], 'ro')
ax.plot(model.mu[0], 'bo');


# ## Trying to fix the learning failure
# 
# Let's try to balance out the constributions of the super- and unsupervised part of the semi-learning procedure. For this purpose I would like to setup a new likelihood function to derive the adjusted E- and M-Step for expectation maximization. I'm curious if this can solve the problem:
# 

# ## Fit on Chris Deottes Top features

# In[ ]:


chris_important_features


# In[ ]:


m1 = train.loc[train.target==0, chris_important_features].mean().values
m2 = train.loc[train.target==1, chris_important_features].mean().values
mu = np.array([m1, m2])
old_mu = mu.copy()


s1 = train.loc[train.target==0, chris_important_features].std().values
s2 = train.loc[train.target==1, chris_important_features].std().values
sigma = np.array([s1,s2])
old_sigma = sigma.copy()

x_train = train.loc[:, chris_important_features].values
y_train = train.target.values
x_test = test.loc[:, chris_important_features].values

model = MySemiGMM(mu, sigma)
model.set_data(x_test, x_train, y_train)
new_mu = model.learn()

proba_test = model.predict(x_test)
proba_train = model.predict(x_train);


# In[ ]:


fig, ax = plt.subplots(1,2,figsize=(20,5))
ax[0].plot(model.mu[0], 'ro')
ax[0].plot(model.mu[1], 'bo')
ax[1].plot(model.sigma[0], 'ro')
ax[1].plot(model.sigma[1], 'bo')


# ## What have we learnt?
