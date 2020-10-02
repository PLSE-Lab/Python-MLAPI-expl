#!/usr/bin/env python
# coding: utf-8

# #  Multi-armed Bandits
# 
# In probability theory, the multi-armed bandit problem (sometimes called the K-[1] or N-armed bandit problem[2]) is a problem in which a fixed limited set of resources must be allocated between competing (alternative) choices in a way that maximizes their expected gain, when each choice's properties are only partially known at the time of allocation, and may become better understood as time passes or by allocating resources to the choice.[3][4] This is a classic reinforcement learning problem that exemplifies the exploration-exploitation tradeoff dilemma. The name comes from imagining a gambler at a row of slot machines (sometimes known as "one-armed bandits"), who has to decide which machines to play, how many times to play each machine and in which order to play them, and whether to continue with the current machine or try a different machine.[5] The multi-armed bandit problem also falls into the broad category of stochastic scheduling. - https://en.wikipedia.org/wiki/Multi-armed_bandit
# 
# * Problem Definition
# * Methods
#  * e-greedy
#  * Softmax Exploration
#  * UCB
#  * Thompson Sampling
# * Results
# 
# Ref: https://github.com/marlesson/MaB-Experiments

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import random
from numpy.random.mtrand import RandomState

seed = 12
random.seed(seed)
rng = RandomState(seed)
np.random.seed(seed)


# ## Problem Definition
# 
# We are a digital ad-sales company and our main goal is to make a profit from selling the ads. We must choose the best ad, among several in our database to be displayed. Each click we earn R\\$ 1.00 and for each display ads we spend a total of R\\$ 0.20.  Therefore, we must create an algorithm that maximizes the company's financial return.
# 
# * In this case, the available actions are clicked or no click
# * The reward will be boolean 1 if click or 0 not.

# In[ ]:


# The reward function depends on the likelihood of clicking

def reward(arm_prob, step = 0):
    '''
    Reward function. 
    '''

    return rng.choice([1, 0], p=[arm_prob, 1.0 - arm_prob])


# We will create 100 ads with different probabilities of being clicked. The MaB algorithms must choose what should be presented without knowing this probability distribution

# In[ ]:


# Total ads
total_arms  = 50

# probs of ads
arms        = [random.betavariate(1.4, 5.4) for i in range(total_arms)]

# Rouns test
rounds      = 2000


# The click probability distribution of all ads

# In[ ]:


fig = px.histogram(x=arms, labels={'x': 'Prob(click)', 'count': 'Anuncios'}, histnorm='probability')
fig.update_layout(template="plotly_white")
fig.show()


# Simulation of ground-truth probabilities rewards of each arm

# In[ ]:


data = []
for i, a in enumerate(arms):
    i_rounds = np.array([i+1 for i in range(rounds)])
    values   = np.random.binomial(1, arms[i], rounds) #+ decay
    values   = np.cumsum(values)/i_rounds# if cum else row.values
    
    data.append(go.Scatter(name="Arm "+str(i), x=[i for i in range(rounds)], y=values))

fig = go.Figure(data=data)
fig.update_layout(template="plotly_white", 
                  xaxis_title_text='Time Step', 
                  yaxis_title_text="Cummulative Mean clicke",
                  title="Cumulative mean click received over time")
fig


# The best arm is the "Arm 32". It has the highest click probability of all, which is 68%.  The MaB algorithms have to find it.

# In[ ]:


def build_plot_rewards(policy = "", rewards = [], cum=False):
    x = [i+1 for i in range(len(rewards))]
    y = np.cumsum(rewards) if cum else np.cumsum(rewards)/x
    return go.Scatter(name=policy, x=x, y=y)


def plot_cum_mean_reward(experiments, rounds, arms, cum=False):

    data = []
    for name, obj in experiments.items():
        _arms, rewards = run(rounds, arms, reward, obj)
        data.append(build_plot_rewards(name, rewards, cum=cum))

    fig = go.Figure(data=data)
    fig.update_layout(template="plotly_white", 
                      xaxis_title_text='Time Step', 
                      yaxis_title_text="Cummulative Mean Reward",
                      title="Cumulative mean reward received over time")
    return fig   

def plot_exploration_arm(rounds, arms, arms_rewards):
    count_per_arms = {}
    for a in range(len(arms)):
        count_per_arms[a] = np.zeros(rounds)

    for r in range(rounds):
        count_per_arms[arms_rewards[r]][r] = 1

    fig = go.Figure()
    x   = (np.array(range(rounds)) + 1)

    for arm, values in count_per_arms.items():    
        fig.add_trace(go.Scatter(
            name="Arm "+str(arm),
            x=x, y=np.cumsum(values),
            hoverinfo='x+y',
            mode='lines',
            line=dict(width=0.5),
            stackgroup='one',
            groupnorm='percent' # define stack group
        ))

    fig.update_layout(template="plotly_white", 
                  xaxis_title_text='Time Step', 
                  yaxis_title_text="Cummulative Exploration Arm",
                  title="Cumulative Exploration Arms over time",
                  yaxis_range=(0, 100))

    return fig              


# In[ ]:


class Bandit(object):
    # Base Badit Class
    #
    def __init__(self, total_arms, seed=42):
        self._total_arms   = total_arms
        self._rng          = RandomState(seed)
        self._arms_rewards =  {}
        
    def act(self):
        pass
    
    def update(self, arm, reward):
        if arm in self._arms_rewards:
            self._arms_rewards[arm].append(reward)
        else:
            self._arms_rewards[arm] = [reward]
    
    def reduction_rewards(self, func= np.mean):
        return np.array([func(self._arms_rewards[i]) for i in range(self._total_arms)])


# In[ ]:


def run(iteraction, arms, func_reward, policy, verbose=False):
    
    arms_rewards  = {}
    rewards       = []
    arms_selected = []
    
    # init rewards
    for arm, prob in enumerate(arms):
        policy.update(arm, 0)

    # run env
    for step in range(iteraction):
        arm    = policy.act()
        
        reward = func_reward(arms[arm], step=step)
        
        policy.update(arm, reward)
        
        arms_selected.append(arm)
        rewards.append(reward)
        
        if verbose:
            print("{}: arm {} with reward {}".format(step, arm, reward))
    
    if verbose:
        print("Reward Cum: {}".format(np.sum(rewards)))
    
    return arms_selected, rewards


# ## Random

# In[ ]:


class RandomPolicy(Bandit):
    # Random Select ARM
    #
    def __init__(self, total_arms, seed=42):
        super().__init__(total_arms, seed=42)

    def act(self):
        return self._rng.choice(list(self._arms_rewards.keys()))
    
arms_selected, rewards = run(10, arms, reward, RandomPolicy(total_arms), verbose=True)    


# Show cumulative mean reward received over time

# In[ ]:


experiments = {"Random": RandomPolicy(total_arms)}

plot_cum_mean_reward(experiments,rounds, arms)


# Using a random algorithm the average reward is ~0.19. That is to say that for each iteration it would have a probability of ~0.19 for the ad to be clicked. 

# In[ ]:


arms_rewards, random_rewards = run(iteraction = rounds, 
                                   arms= arms, 
                                   func_reward = reward, 
                                   policy = RandomPolicy(total_arms))
plot_exploration_arm(rounds, arms, arms_rewards)


# The exploitation of each ad is carried out in the same way, regardless of the reward given

# ## Epsilon Greedy

# In[ ]:


class EpsilonGreedy(Bandit):
    def __init__(self, total_arms, epsilon=0.1, seed = 42):
        super().__init__(total_arms, seed=seed)
        self._total_arms = total_arms
        self._epsilon    = epsilon

    def act(self):
        '''
        Choice an arm
        '''

        if self._rng.choice([True, False], p=[self._epsilon, 1.0 - self._epsilon]):
            action = self._rng.randint(0, self._total_arms)
        else:
            action = np.argmax(self.reduction_rewards())

        return action


# In[ ]:


# Simulate

experiments = {
    "Random": RandomPolicy(total_arms),
    "Epsilon e=0.0": EpsilonGreedy(total_arms, epsilon=0.0),
    "Epsilon e=0.1": EpsilonGreedy(total_arms, epsilon=0.1),
    "Epsilon e=0.2": EpsilonGreedy(total_arms, epsilon=0.2),
    "Epsilon e=0.5": EpsilonGreedy(total_arms, epsilon=0.5),
}
 

plot_cum_mean_reward(experiments, rounds, arms)


# In[ ]:


arms_rewards, random_rewards = run(iteraction = rounds, 
                                   arms= arms, 
                                   func_reward = reward, 
                                   policy = EpsilonGreedy(total_arms, epsilon=0.1))
plot_exploration_arm(rounds, arms, arms_rewards)


# ## Softmax Exploration

# In[ ]:


from scipy.special import softmax, expit


# In[ ]:


class SoftmaxExplorer(Bandit):
    def __init__(self, total_arms, logit_multiplier = 1.0, seed = 42):
        super().__init__(total_arms, seed=seed)
        self._logit_multiplier = logit_multiplier

    def act(self):
        reward_mean  = self.reduction_rewards(np.mean)
        
        reward_logit = expit(reward_mean)

        arms_probs   = softmax(self._logit_multiplier * reward_logit)

        action       = self._rng.choice(list(range(self._total_arms)), p = arms_probs)
        
        return action  


# In[ ]:


# Simulate
experiments = {
    "Random": RandomPolicy(total_arms),
    "Softmax Exploration x1": SoftmaxExplorer(total_arms, logit_multiplier=1),
    "Softmax Exploration x10": SoftmaxExplorer(total_arms, logit_multiplier=10),
    "Softmax Exploration x100": SoftmaxExplorer(total_arms, logit_multiplier=100),
    "Softmax Exploration x500": SoftmaxExplorer(total_arms, logit_multiplier=500),
}

plot_cum_mean_reward(experiments, rounds, arms)


# In[ ]:


arms_rewards, random_rewards = run(iteraction = rounds, 
                                   arms= arms, 
                                   func_reward = reward, 
                                   policy = SoftmaxExplorer(total_arms, logit_multiplier=500))
plot_exploration_arm(rounds, arms, arms_rewards)


# ## UCB - Upper Confidence Bound

# In[ ]:


class UCB(Bandit):
    def __init__(self, total_arms, c = 2, seed = 42):
        super().__init__(total_arms, seed=seed)
        self._c            = c
        self._times        = 1
        self._action_times = np.zeros(total_arms)
    

    def act(self):
        reward_mean      = self.reduction_rewards()

        confidence_bound = reward_mean +                             self._c * np.sqrt(                                  np.log(self._times) / (self._action_times + 0.1))  # c=2
        
        action       = np.argmax(confidence_bound)

        self._times += 1
        self._action_times[action] += 1
        
        return action


# In[ ]:



# Simulate
experiments = {
    "Random": RandomPolicy(total_arms),
    "UCB c=0.01": UCB(total_arms, c=0.01),
    "UCB c=0.5": UCB(total_arms, c=0.5),
    "UCB c=1": UCB(total_arms, c=1),
    "UCB c=2": UCB(total_arms, c=2),
}

plot_cum_mean_reward(experiments, rounds, arms)


# In[ ]:


arms_rewards, random_rewards = run(iteraction = rounds, 
                                   arms= arms, 
                                   func_reward = reward, 
                                   policy = UCB(total_arms, c=0.5))
plot_exploration_arm(rounds, arms, arms_rewards)


# ## Thompson Sampling

# In[ ]:


class ThompsonSampling(Bandit):
    def __init__(self, total_arms, seed = 42):
        super().__init__(total_arms, seed=seed)        
        self._alpha        = np.ones(total_arms)
        self._beta         = np.ones(total_arms)
    
    def act(self):
        reward_prior = [random.betavariate(self._alpha[i], self._beta[i]) 
                            for i in range(self._total_arms)]
        
        return np.argmax(reward_prior)
    
    def update(self, arm, reward):
        super().update(arm, reward)        
        
        self._alpha[arm] += reward
        self._beta[arm]  += 1 - reward


# In[ ]:





# In[ ]:


# Simulate
experiments = {
    "Random": RandomPolicy(total_arms),
    "Tompson Sampling": ThompsonSampling(total_arms),
}

plot_cum_mean_reward(experiments, rounds, arms)


# In[ ]:


arms_rewards, random_rewards = run(iteraction = rounds, 
                                   arms= arms, 
                                   func_reward = reward, 
                                   policy = ThompsonSampling(total_arms))
plot_exploration_arm(rounds, arms, arms_rewards)


# In[ ]:





# ## Compare Results

# In[ ]:


# Simulate
experiments = {
    "UCB c=0.5":    UCB(total_arms, c=0.5),
    "Tompson Sampling": ThompsonSampling(total_arms),
    "Softmax Exploration x100": SoftmaxExplorer(total_arms, logit_multiplier=100),
    "e-greedy e=0.1": EpsilonGreedy(total_arms, epsilon=0.1),
    "Random": RandomPolicy(total_arms),
}

plot_cum_mean_reward(experiments, rounds, arms)


# Of the methods tested, the best for the 2000 steps was UCB. Tompson Sampling's method was still adjusting and it would probably look good too. All methods were above the random in all steps. It was possible to observe that depending on the parameter of the exploration method, the results vary widely.
# 
# By converting the click probabilities for each method, it is possible to arrive at an estimated gain value when using the method in question. 
# 
# ```$(model, step) = [Prob(click|model)*valClick + (1-Prob(Click|model))*ValNoClick]*step```

# Model                | Prob(click)   | Click \\$ | NoClick \\$ | Total \\$
# -------------------- | ------------- | --------|-----------|--------
# UCB                  | 0.58          | 0,464   |  -0,084   |  0,38  
# Tompson Sampling     | 0.57          | 0,456   |  -0,086   |  0,37 
# Softmax Exploration  | 0.54          | 0,432   |   -0,092  |  0,34 
# e-Greedy             | 0.50          | 0,4     |   -0,1    |  0,3 
# Random               | 0.22          | 0,176   |   -0,156  |  0,02 

# **Thus, the UCB collects R\\$ 0.38 per step. Over the 2000 steps the total amount collected is R\\$ 760, if we used Random it would be R\\$ 40,00**

# code: https://github.com/marlesson/MaB-Experiments
