#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# In[ ]:


class MarkovChain(object):
    def __init__(self, transition_prob):
        # create the transition matrix
        self.transition_prob = transition_prob
        # Create all possible states form the transition matrix
        self.states = list(transition_prob.keys())
        
    def next_state(self, current_state):
        return np.random.choice(self.states,
                               p=[self.transition_prob[current_state][next_state] 
                                  for next_state in self.states])
    def generate_states(self, current_state, no=10):
        future_states = []
        for i in range(no):
            next_state = self.next_state(current_state)
            future_states.append(next_state)
            current_state = next_state
        return future_states


# In[ ]:


transition_prob = {'Sunny': {'Sunny': 0.8, 'Rainy': 0.19, 'Snowy': 0.01},
                   'Rainy': {'Sunny': 0.2, 'Rainy': 0.7,'Snowy': 0.1},
                   'Snowy': {'Sunny': 0.1, 'Rainy': 0.2, 'Snowy': 0.7}}


# In[ ]:


weather_chain = MarkovChain(transition_prob=transition_prob)
weather_chain.generate_states(current_state='Snowy', no=10)


# In[ ]:


class MarkovChain(object):
    def __init__(self, transition_matrix, states):

        self.transition_matrix = np.atleast_2d(transition_matrix)
        self.states = states
        self.index_dict = {self.states[index]: index for index in 
                           range(len(self.states))}
        self.state_dict = {index: self.states[index] for index in
                           range(len(self.states))}
 
    def next_state(self, current_state):

        return np.random.choice(
         self.states, 
         p=self.transition_matrix[self.index_dict[current_state], :]
        )
 
    def generate_states(self, current_state, no=10):

        future_states = []
        for i in range(no):
            next_state = self.next_state(current_state)
            future_states.append(next_state)
            current_state = next_state
        return future_states


# In[ ]:


transition_matrix = [[0.8, 0.19, 0.01],
                     [0.2,  0.7,  0.1],
                     [0.1,  0.2,  0.7]]


# In[ ]:


weather_chain = MarkovChain(transition_matrix=transition_matrix,
                            states=['Sunny', 'Rainy', 'Snowy'])
weather_chain.generate_states(current_state='Snowy', no=10)

