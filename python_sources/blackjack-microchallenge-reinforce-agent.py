#!/usr/bin/env python
# coding: utf-8

# # Intro
# 
# Ready for a quick test of your logic and programming skills?
# 
# In today's micro-challenge, you will write the logic for a blackjack playing program.  Our dealer will test your program by playing 50,000 hands of blackjack. You'll see how frequently your program won, and you can discuss how your approach stacks up against others in the challenge.
# 
# ![Blackjack](http://www.hightechgambling.com/sites/default/files/styles/large/public/casino/table_games/blackjack.jpg)

# # Blackjack Rules
# 
# We'll use a slightly simplified version of blackjack (aka twenty-one). In this version, there is one player (who you'll control) and a dealer. Play proceeds as follows:
# 
# - The player is dealt two face-up cards. The dealer is dealt one face-up card.
# - The player may ask to be dealt another card ('hit') as many times as they wish. If the sum of their cards exceeds 21, they lose the round immediately.
# - The dealer then deals additional cards to himself until either:
#     - The sum of the dealer's cards exceeds 21, in which case the player wins the round, or
#     - The sum of the dealer's cards is greater than or equal to 17. If the player's total is greater than the dealer's, the player wins. Otherwise, the dealer wins (even in case of a tie).
# 
# When calculating the sum of cards, Jack, Queen, and King count for 10. Aces can count as 1 or 11. (When referring to a player's "total" above, we mean the largest total that can be made without exceeding 21. So A+8 = 19, A+8+8 = 17.)
# 
# # The Blackjack Player
# You'll write a function representing the player's decision-making strategy. Here is a simple (though unintelligent) example.
# 
# **Run this code cell** so you can see simulation results below using the logic of never taking a new card.

# In[ ]:


def should_hit(player_total, dealer_card_val, player_aces):
    """Return True if the player should hit (request another card) given the current game
    state, or False if the player should stay. player_aces is the number of aces the player has.
    """
    return False


# We'll simulate games between your player agent and our own dealer agent by calling your function. So it must use the name `should_hit`.

# # The Blackjack Simulator
# 
# Run the cell below to set up our simulator environment:

# In[ ]:


# SETUP. You don't need to worry for now about what this code does or how it works. 
# If you're curious about the code, it's available under an open source license at https://github.com/Kaggle/learntools/
from learntools.core import binder; binder.bind(globals())
from learntools.python.ex3 import q7 as blackjack
# Returns a message "Sorry, no auto-checking available for this question." (You can ignore.)
blackjack.check()
print('Setup complete.')


# Once you have run the set-up code, you can see the action for a single game of blackjack with the following line:

# In[ ]:


blackjack.simulate_one_game()


# You can see how your player does in a sample of 50,000 games with the following command:

# In[ ]:


blackjack.simulate(n_games=50000)


# # Your Turn
# 
# Write your own `should_hit` function in the cell below. Then run the cell and see how your agent did in repeated play.

# Ok Kaggle, lets see what we can do!
# 
# Lets first implement an Agent interface and a full blackjack learning environment for our agent to learn.

# In[ ]:


class Agent(object):
    # Our policy that maps state to action parameterized by w
    def policy(self, state):     
        raise NotImplementedError('You need to overwrite the policy method.')
        
    def predict(self, *state):
        return self.policy(state)
    
    def train(self, *state):
        return self.policy(state)
    
    def store_reward(self, reward):
        pass

    def update(self):
        pass

    # Vectorized softmax Jacobian
    @staticmethod
    def softmax_grad(softmax):
        s = softmax.reshape(-1,1)
        return np.diagflat(s) - np.dot(s, s.T)


# We will need some cards to play with:

# In[ ]:


from random import shuffle

# gfenerator for card drawing without repetition
def get_deck():
    cards = (list(range(1, 9)) + [10] * 4 + [11]) * 4
    while cards:
        shuffle(cards)
        yield cards.pop()

deck = get_deck()
for i in range(2):
    print(next(deck))


# The total calculation functions looks moreless like this:

# In[ ]:


def calculate_total(aces, partial):
    total = partial
    for i in range(aces):
        if partial + 11 > 21:
            total += 1          # if score > 21, aces have value 1
        else:
            total += 11         # atherwise, aces have value 11
    return total

def make_environment(dealer_partial, dealer_aces, player_partial, player_aces):
    return (calculate_total(player_aces, player_partial),
        calculate_total(dealer_aces, dealer_partial),
        player_aces)


# Now we can simulate one full game!

# In[ ]:


def simulate_game(agent, train=False):
    # init scores and deck
    dealer_partial = 0
    dealer_aces = 0
    player_partial = 0
    player_aces = 0
    deck = get_deck()
    
    # initial draw for the player
    for _ in range(2):
        card = next(deck)
        if card == 11:
            player_aces += 1
        else:
            player_partial += card
            
    # then for the dealer
    card = next(deck)
    if card == 11:
        dealer_aces += 1
    else:
        dealer_partial += card
    
    # player's turn
    # draw cards according to the provided policy
    while getattr(agent, 'train' if train else 'predict')(*make_environment(dealer_partial, dealer_aces, player_partial, player_aces)):
        card = next(deck)
        if card == 11:
            player_aces += 1
            
        else:
            player_partial += card
            
        if calculate_total(player_aces, player_partial) > 21:
            agent.store_reward(-1)  
            return 0 # return 0 indicating house's victory
        
        agent.store_reward(0)
        
    # dealer's turn
    while calculate_total(dealer_aces, dealer_partial) < 17:
        card = next(deck)
        if card == 11:
            dealer_aces += 1
        else:
            dealer_partial += card
            
    # calculate totals
    player_total = calculate_total(player_aces, player_partial)
    dealer_total = calculate_total(dealer_aces, dealer_partial)
    
    # return 1 for player's victory, 0 otherwise
    if dealer_total > 21 or player_total > dealer_total:
        agent.store_reward(1)
        return 1
    
    agent.store_reward(-1)
    return 0


# Lets test our code for validity.

# In[ ]:


class Looser(Agent):
    def predict(self, *state):
        """This should never win, for sanity check"""
        return True

for i in range(1000):
    assert not simulate_game(Looser())
    
print('All good till here.')


# Lets simulate a bunch of games for our dummy agent.

# In[ ]:


def simulate(agent, n_games=5, train=False):
    player_victories = 0
    for i in range(n_games):
        player_victories += simulate_game(agent, train)
        agent.update()
        
        if i % 1000 == 0:
            print(f'{i}/{n_games} - Player won {player_victories} out of {i} (win rate = {player_victories / (i + 1) * 100}%)', end='\r')
        
    print(f'\nPlayer won {player_victories} out of {n_games} (win rate = {player_victories / n_games * 100}%)')

simulate(Looser(), 1000)


# Now we need some better policy than 'always lose'! Lets give it a try.
# 
# We are going to use some reinforcement learning technics, REINFORCE Policy Gradients to be more specific, because why not?

# In[ ]:


import numpy as np
np.random.seed(1)

class REINFORCE(Agent):
    '''
    REINFORCE Policy Gradients agent with linear shallow model
    https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf
    https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ
    https://medium.com/samkirkiles/reinforce-policy-gradients-from-scratch-in-numpy-6a09ae0dfe12
    '''
    def __init__(self, state_dim, n_actions, learning_rate, gamma):
        # Init weight
        self.w = np.random.rand(state_dim, n_actions) * 0.1
        self.n_actions = n_actions
        self.lr = learning_rate
        self.g = gamma
        self.grads = []
        self.rewards = []
                   
    @staticmethod
    def preprocess_state(state):
#         return np.array([state]).reshape((1, -1))
        return np.array([state[0] / 21 - 0.5, state[1] / 21 - 0.5, state[2] / 4 - 0.5]).reshape((1, -1))
        
    # Our policy that maps state to action parameterized by w
    def policy(self, state):
        exp = np.exp(state.dot(self.w))
        probs = exp / np.sum(exp)
        action = np.random.choice(self.n_actions, p=probs[0])
        return action, probs
    
    def predict(self, *state):
        state = self.preprocess_state(state)
        return np.argmax(self.policy(state)[1][0])
        
    def train(self, *state):
        state = self.preprocess_state(state)
        action, probs = self.policy(state)
        dsoftmax = self.softmax_grad(probs)[action,:]
        dlog = dsoftmax / probs[0, action]
        grad = state.T.dot(dlog[None,:])
        self.grads.append(grad)
        return action
    
    def store_reward(self, reward):
        # Compute gradient and save with reward in memory for our weight update
        self.rewards.append(reward)

    def update(self):
        for i in range(len(self.grads)):
            # Loop through everything that happend in the episode and update towards the log policy gradient times **FUTURE** reward
            self.w += self.lr * self.grads[i] * sum([r * (self.g ** r) for t, r in enumerate(self.rewards[i:])])
        self.grads = []
        self.rewards = []


# In[ ]:


smart = REINFORCE(3, 2, 0.01, 0.9999)
simulate(smart, 200000, train=True)


# In[ ]:


simulate(smart, 50000, train=False)


# In[ ]:


def should_hit(player_total, dealer_card_val, player_aces):
    """Return True if the player should hit (request another card) given the current game
    state, or False if the player should stay. player_aces is the number of aces the player has.
    """
    return smart.predict(player_total, dealer_card_val, player_aces)

blackjack.simulate(n_games=50000)


# # Discuss Your Results
# 
# How high can you get your win rate? We have a [discussion thread](https://www.kaggle.com/learn-forum/58735#latest-348767) to discuss your results.

# ---
# This exercise is from the **[Python Course](https://www.kaggle.com/Learn/python)** on Kaggle Learn.
# 
# Check out **[Kaggle Learn](https://www.kaggle.com/Learn)**  for more instruction and fun exercises.
