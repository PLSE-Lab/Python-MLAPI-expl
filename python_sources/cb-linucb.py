#!/usr/bin/env python
# coding: utf-8

# # Contextual Bandit
# Learned from: https://github.com/etiennekintzler/bandits_algorithm/blob/master/linUCB.ipynb

# Learned from: A Contextual-Bandit Approach to Personalized News Article Recommendation (paper from Li et al.)
# ## Problem formulation
# 
# A contextual bandit A proceeds in discrete trials t = 1, 2, 3, ... In trial t
# 1. The algorithm observes current user $u_t$ and a set $A_t$ of arms (actions). These two made up $x_{t, a}$ for every $a \in A_t$, and is called the $context$.
# 2. Based on the observed payoffs in previous trials, $A$ chooses an arm $a_t \in A_t$, and receives a payoff $r_{t,a_t}$ whose expectation depends on both the user $u_t$ and the arm $a_t$.
# 3. $A$ improves arm-selection strategy with the new observation $(x_{t, a_t}, a_t, r_{t, a_t})$.
# 4. The total T-trial payoff of $A$ is defined as $\sum_{t=1}^{T}r_{t, a_t}$
# 5. Optimal expected T-trial payoff is defined as $E[\sum_{t=1}^Tr_{t, a_t^*}]$, where $a_t^*$ is the arm with maximum expected payoff at trial $t$.
# 6. The goal for $A$ is to maximize payoff or minimize the regret, which is defined as:
# \begin{equation}
#     R_A(T) = E[\sum_{t=1}^Tr_{t, a_t^*}] - E[\sum_{t=1}^Tr_{t, a_t}]
# \end{equation}
# 
# ## The algorithm
# Assume that the payoff of an arm $a$ is linear in its d-dimensional feature x_{t, a} with some unknown coefficient vector $\theta_a^*$:
# \begin{equation}
# E[r_{t, a} | x_{t, a}] = x_{t, a}^T\theta_a^*
# \end{equation}
# 
# Let $D_a$ be a design matrix of dimension $m\times d$ at trial $t$, $m$ is the training inputs (e.g., m contexts that are observed previously for article a), and $c_a \in R^m$ be the corresponding response vector (e.g., the corresponding $m$ click/no-click user feedback). Applying ridge regression to the training data ($D_a$, $c_a$) gives an estimate of the coefficients:
# 
# \begin{equation}
# \hat\theta_a = (D_a^TD_a + I_d)^{-1}D_a^Tc_a
# \end{equation}
# 
# It can be proved that, with probability at least $1-\delta$:
# \begin{equation}
# |x_{t, a}^T\hat\theta_a - E[r_{t, a}|x_{t, a}]| \leq \alpha \sqrt{x_{t, a}^T(D_a^TDa + I_d)^{-1}x_{t, a}}
# \end{equation}
# for any $\delta>0$ and $x_{t, a} \in R^d$, where $\alpha = \sqrt{ln(2/\delta)/2}$ is a constant.
# 
# In other words, the inequality above gives a reasonably tight UCB for the expected payoff of arm a, from which a UCB-type arm-selection strategy can be derived: at each trial t, choose:
# \begin{equation}
# a_t = arg\,\underset{a\in A_t}{max}(x_{t, a}^T + \alpha\sqrt{x_{t, a}^T A_a^{-1}x_{t, a}})
# \end{equation}
# where $A_a = D_a^TD_a + I_d$
# 
# Also, the expected payoff $x_{t, a}^T\theta_a^*$ can be calculated as $x_{t, a}^TA_a^{-1}$, and then $\sqrt{x_{t, a}^TA_a^{-1}x_{t, a}}$ becomes the standard deviation.
# 
# Therefore the upper bound should be:
# \begin{equation}
#     \hat\theta_{t, a} = A_a^{-1}b_a \\
#     p_{t, a} = \hat\theta_{t, a}x_{t, a} + \alpha*\sqrt{x_{t, a}A^{-1}x_{t, a}}
# \end{equation}

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


np.random.seed(123)


# ### Defining constants

# In[ ]:


N_TRIAL = 2000
N_ARMS = 16
N_FEATURE  = 5
BEST_ARMS = [3, 7, 9, 15]


# ## 1. Problem setting
# ### Setting problems parameters

# In[ ]:


def make_design_matrix(n_trial, n_arms, n_feature):
    available_arms = np.arange(n_arms)
    X = np.array([[np.random.uniform(low=0, high=1, size=n_feature) for _ in available_arms] for _ in np.arange(n_trial)])
    return X


# In[ ]:


D = make_design_matrix(N_TRIAL, N_ARMS, N_FEATURE)


# In[ ]:


D.shape


# In[ ]:


def make_theta(n_arms, n_feature, best_arms, bias = 1):
    true_theta = np.array([np.random.normal(size=n_feature, scale=1/4) for _ in np.arange(n_arms)])
    true_theta[best_arms] = true_theta[best_arms] + bias
    return true_theta


# In[ ]:


theta = make_theta(N_ARMS, N_FEATURE, BEST_ARMS, bias=1)


# In[ ]:


def generate_reward(arm, x, theta, scale_noise = 1/10):
    signal = theta[arm].dot(x)
    noise = np.random.normal(scale=scale_noise)
    return (signal + noise)


# In[ ]:


def make_regret(payoff, oracle):
    return np.cumsum(oracle - payoff)


# For each arm, feature vector is i.i.d and simulated according to U(\[0, 1\]).
# 
# Theta vector is simulated according to $N(0_d, cI_d)$ with $c=1/4$.
# 
# Highly profitable arms are created adding positive bias to $\theta$. These arms are defined by the constant BEST_ARMS.
# 
# By hypothesis, the expected payoff is linear in its feature vector $x_{t, a}$ with some unknown coefficient vector $\theta^*_a$:
# 
# \begin{equation*}
# E[r_{t, a}|x_{t, a}] = x_{t, a}^T\theta^*_a
# \end{equation*}
# 
# Regret is defined as $R_A(T) = E[\sum_{t=1}^T r_{t, a_t^*}] - E[\sum_{t=1}^T r_{t, a_t}]$
# 

# ### Simulation of design matrix and weight vector (theta)

# In[ ]:


X = make_design_matrix(n_trial=N_TRIAL, n_arms= N_ARMS, n_feature=N_FEATURE)
true_theta = make_theta(n_arms = N_ARMS, n_feature=N_FEATURE, best_arms=BEST_ARMS)


# ### Graphical representation of average payoff per arm

# In[ ]:


# Average reward
ave_reward = np.mean([[generate_reward(arm=arm, x=X[t, arm], theta= true_theta) for arm in np.arange(N_ARMS)] for t in np.arange(N_TRIAL)], axis=0)


# In[ ]:


ave_reward


# In[ ]:


f, (left, right) = plt.subplots(1, 2, figsize=(15, 10))
f.suptitle(t="Visualizing of simulated parameters: true theta and average reward", fontsize=20)
# True theta
left.matshow(true_theta)
f.colorbar(left.imshow(true_theta), ax = left)
left.set_xlabel("feature number")
left.set_ylabel("arm number")
left.set_yticks(np.arange(N_ARMS))
left.set_title("True theta matrix")
# Average reward
right.bar(np.arange(N_ARMS), ave_reward)
right.set_title("Average reward per arm")
right.set_xlabel("arm number")
right.set_ylabel("average reward")
plt.show()


# On average, the reward is much higher for the best arms. In this example, the arm 9 is the most profitable.
# 
# Should note on whether the features are positive? Because if there are negative features high positive theta will lead to low rewards.

# ## 2. Algorithm (disjoint version)

# #### Algorithm definition

# In[ ]:


A = np.array([np.diag(np.ones(shape=6)) for _ in np.arange(16)])


# In[ ]:


A


# In[ ]:


b = np.array([np.zeros(shape=5) for _ in np.arange(16)])


# In[ ]:


b


# In[ ]:


def linUCB_disjoint(alpha, X, generate_reward, true_theta):
    print("linUCB disjoint with exploration parameter alpha: ", alpha)
    n_trial, n_arms, n_feature = X.shape
    # 1. Initialize object
    # 1.1. Output object
    arm_choice = np.empty(n_trial) # store arm choice (integer) for each trial
    r_payoff = np.empty(n_trial) # store payoff (float) for each trial
    theta = np.empty(shape=(n_trial, n_arms, n_feature)) # record theta over each trial (n_arms, n_feature) per trial
    p = np.empty(shape = (n_trial, n_arms)) # predictions for reward of each arm for each trial
    # 1.2 Intermediate object
    A = np.array([np.diag(np.ones(shape=n_feature)) for _ in np.arange(n_arms)])
    b = np.array([np.zeros(shape=n_feature) for _ in np.arange(n_arms)])
    # 2. Algo
    for t in np.arange(n_trial):
        # Compute estimates (theta) and prediction (p) for all arms
        for a in np.arange(n_arms):
            inv_A = np.linalg.inv(A[a])
            theta[t, a] = inv_A.dot(b[a])
            p[t, a] = theta[t, a].dot(X[t, a]) + alpha * np.sqrt(X[t, a].dot(inv_A).dot(X[t, a]))
        # Choosing best arms
        chosen_arm = np.argmax(p[t])
        x_chosen_arm = X[t, chosen_arm]
        r_payoff[t] = generate_reward(arm=chosen_arm, x=x_chosen_arm, theta=true_theta)

        arm_choice[t] = chosen_arm
        
        # update intermediate objects (A and b)
        A[chosen_arm] += np.outer(x_chosen_arm, x_chosen_arm.T)
        b[chosen_arm] += r_payoff[t]*x_chosen_arm
    return dict(theta=theta, p=p, arm_choice=arm_choice, r_payoff=r_payoff)


# ### Defining oracle and random payoff

# In[ ]:


oracle = np.array([np.max([generate_reward(arm=arm, x=X[t, arm], theta = true_theta) for arm in np.arange(N_ARMS)]) for t in np.arange(N_TRIAL)])


# In[ ]:


len(oracle)


# In[ ]:


oracle[0]


# In[ ]:


payoff_random = np.array([generate_reward(arm=np.random.choice(N_ARMS), x= X[t, np.random.choice(N_ARMS)], theta = true_theta) for t in np.arange(X.shape[0])])


# In[ ]:


payoff_random


# In[ ]:


regret_random = make_regret(payoff=payoff_random, oracle=oracle)


# ### Algorithm testing for various alpha

# In[ ]:


alpha_to_test = [0, 1, 2.5, 5, 10, 20]
results_dict = {alpha: linUCB_disjoint(alpha=alpha, X=X, generate_reward=generate_reward, true_theta = true_theta) for alpha in alpha_to_test}


# ## 3. Analyzing regrets, coefficients estimates and chosen arm

# ### Function definition

# In[ ]:


def plot_regrets(results, oracle):
    [plt.plot(make_regret(payoff=x['r_payoff'], oracle=oracle), label="alpha: "+str(alpha)) for (alpha, x) in results.items()]


# In[ ]:


def plot_estimates(x, alpha, true_theta=None, abs_ylim = None, ncol = 4):
    print("Estimates plot for alpha: ", alpha)
    if true_theta is not None:
        print("Parameter true_theta has been supplied. Plotting convergence")
    for i, arm in enumerate(np.arange(N_ARMS)):
        plt.subplot(np.ceil(N_ARMS/ncol), ncol, 1+i)
        if true_theta is not None:
            data_to_plot = pd.DataFrame(x[alpha]["theta"][:, arm, :]) - true_theta[arm]
        else:
            data_to_plot = pd.DataFrame(x[alpha]["theta"][:, arm, ])
        plt.plot(data_to_plot)
        
        if (arm in BEST_ARMS):
            title = 'Arm: ' + str(arm) + " (best)"
        else:
            title = "Arm: " + str(arm)
        plt.title(title)
        
        if abs_ylim is not None:
            plt.ylim([-abs_ylim, abs_ylim])
    plt.legend(["c"+str(feature) for feature in np.arange(N_FEATURE)])


# In[ ]:


def plot_selected_arms(x, bar_width=0.15):
    for (i, alpha) in enumerate(x):
        xi, yi = np.unique(x[alpha]["arm_choice"], return_counts=True)
        plt.bar(xi + i*bar_width, yi, label="alpha: " + str(alpha), width=bar_width)
    
    plt.xticks(np.arange(N_ARMS) + round(len(x)/2)*bar_width, np.arange(N_ARMS))
    plt.legend()


# ## 3.1. Representing regret according to various level of exploration value

# In[ ]:


plt.figure(figsize=(12.5, 7.5))
plot_regrets(results_dict, oracle)
plt.plot(make_regret(payoff=payoff_random, oracle=oracle), label = "random", linestyle='--')
plt.legend()
plt.title("Regrets for various levels of alpha")
plt.show()


# In[ ]:


plt.figure(figsize=(12.5, 17.5))
plot_estimates(results_dict, alpha=2.5, true_theta = true_theta, abs_ylim=3/4)


# In[ ]:


plt.figure(figsize=(12.5, 17.5))
plot_estimates(results_dict, alpha=20, true_theta=true_theta, abs_ylim=3/4)


# In[ ]:


plt.figure(figsize=(15, 5))
plot_selected_arms(results_dict)


# Low exploration parameter (alpha = 1) prevents the algorithm from visiting highly rentable arm (the arm 15 for instnace). No exploration causes the algorithm to get stuck in the first highly rentable arm found.
