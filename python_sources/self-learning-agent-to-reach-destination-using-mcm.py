#!/usr/bin/env python
# coding: utf-8

# There are 7 states - 0,1,2,3,4,5,6.
# 
# The agent start's at state 3. In each time step, the agent can choose an action left or right.
# The agent performs random walk to reach the terminal states 0 or 6.
# 
# All transitions have a reward zero except transition from 5 to 6 which has a reward 1.
# 
# Initial policy is uniform random policy.

# ## Problem Statement
# The agent has to reach terminal state (6) in optimal way.

# At each time step t, the agent recieves some representation of the environment's state, S(t) and on that basis selects an action A(t). One time step later, at t+1, the agent recieves a numerical reward R(t+1) and finds itself in a new state S(t+1).
# 
# We use R(t+1) instead of R(t) to denote the reward due to A(t) because it emphasizes that the next reward and next state, R(t+1) and S(t+1), are jointly determined

# ## Solution
# This is a **reinforcement learning** problem which can be solved using **Monte Carlo Method** to find an optimal path.
# 
# In a MCM, the agent performs many random walks to learn about the value of every state.

# In[ ]:


import random


# In[ ]:


class RandomWalk:   #Environment
    def __init__(self):
        self.state_count=7  #Total 7 states
        self.reward=[0 for i in range(0,self.state_count)]
        self.reward[6]=1  #When it goes to state 6, agent will get a reward 1 otherwise 0.
        
    def getReward(self,state):   #Terminal state
        if state==6:
            return (1)
        else:
            return (0)
        
    def nextState(self,state,action):
        return (state+action)


# In[ ]:


class Agent:
    def __init__(self):
        self.state_count=7
        self.action=None
        self.env=RandomWalk()
        self.N=[0 for i in range(self.state_count)]    #Number of first visits to state
        self.value=[0 for i in range(self.state_count)]
        self.gamma=0.9
        
    def initialize(self,state):
        self.state=state
        self.moves=[self.state]
        self.reward=[0]
        
    def makeAction(self):
        self.action=random.choice([-1,1])
        self.state=self.env.nextState(self.state,self.action)
        self.moves.append(self.state)
        self.reward.append(self.env.getReward(self.state))
    
    def isEnd(self):
        if self.state == 6 or self.state == 0:
            return (True)
        else:
            return (False)
        
    def updateValuesMCM(self):   #Recieves agent object as parameter
        for state in range(self.state_count):
            if state in self.moves:
                self.N[state]+=1
                returns=0
                for i in range(self.moves.index(state),len(self.moves)):
                    returns+=self.reward[i]*(self.gamma**(i-self.moves.index(state)))
                self.value[state]=self.value[state]+((returns-self.value[state])/self.N[state])
                
    def getValue(self):
        return (A.value)


# In[ ]:


def train(A):
    epoch_value={}
    temp=[]
    for epoch in range(0,201):
        A.initialize(3)
        while not A.isEnd():
            A.makeAction()      
        A.updateValuesMCM()
        if (epoch<101 and epoch%50==0) or (epoch%200==0):
            epoch_value[epoch]=[round(i,4) for i in A.getValue()]
    return (A,epoch_value)


# In[ ]:


A=Agent()
A,epoch_value=train(A)


# In[ ]:


epoch_value


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.figure(figsize=(20,8))
plt.title('Optimal value of state change with epoch')
plt.xlabel('State')
plt.ylabel('Value')
for i in epoch_value.keys():
    plt.plot(epoch_value[i],label=i)
    
plt.legend(loc='upper left')
plt.show()


# ## Reference:
# 
# [Reinforcement Learning: An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)

# In[ ]:




