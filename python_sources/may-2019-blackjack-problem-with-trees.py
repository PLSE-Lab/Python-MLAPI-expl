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
# When calculating the sum of cards, Jack, Queen, and King count for 10. Aces can count as 1 or 11 (when referring to a player's "total" above, we mean the largest total that can be made without exceeding 21. So e.g. A+8 = 19, A+8+8 = 17)
# 
# # The Blackjack Player
# You'll write a function representing the player's decision-making strategy. Here is a simple (though unintelligent) example.
# 
# **Run this code cell** so you can see simulation results below using the logic of never taking a new card.

# In[1]:


def should_hit(player_total, dealer_card_val, player_aces):
    """Return True if the player should hit (request another card) given the current game
    state, or False if the player should stay. player_aces is the number of aces the player has.
    """
    return False


# We'll simulate games between your player agent and our own dealer agent by calling your function. So it must use the name `should_hit`

# # The Blackjack Simulator
# 
# Run the cell below to set up our simulator environment:

# In[2]:


# SETUP. You don't need to worry for now about what this code does or how it works. 
# If you're curious about the code, it's available under an open source license at https://github.com/Kaggle/learntools/
from learntools.core import binder; binder.bind(globals())
from learntools.python.ex3 import q7 as blackjack
print('Setup complete.')


# Once you have run the set-up code. You can see the action for a single game of blackjack with the following line:

# In[4]:


blackjack.simulate_one_game()


# You can see how your player does in a sample of 50,000 games with the following command:

# In[5]:


blackjack.simulate(n_games=50000)


# # My Turn
# 
# Write your own `should_hit` function in the cell below. Then run the cell and see how your agent did in repeated play.

# In[6]:


import numpy as np

def should_hit(player_total, dealer_card_val, player_aces):
    """Return True if the player should hit (request another card) given the current game
    state, or False if the player should stay. player_aces is the number of aces the player has.
    """
    return np.random.choice([True, False])

blackjack.simulate(n_games=50000)


# This result is interesting.  Coming to this blackjack table and doing nothing except placing down a bet each turn provides 38.5% probability of winning each hand.  However, if the player randomly decides to hit this significantly reduces the probability to 27.5%.  Bottomline, this supports the idea that you need to have a plan when you sit down to the table and if in doubt then stand!

# In[7]:


from contextlib import redirect_stdout
from io import StringIO

def simulate_game(verbose=False):
    out_buffer = StringIO()
    with redirect_stdout(out_buffer):
        blackjack.simulate_one_game()
    out_str = out_buffer.getvalue()
    if verbose: 
        print(out_str)
    return any(['Player wins' in x for x in out_str.split('\n')]) # if any lines say player wins then we won
simulate_game(True)

global val_list
val_list = []
def should_hit(player_total, dealer_card_val, player_aces):
    global val_list
    cur_move = np.random.choice([True, False])
    val_list+=[(player_total, dealer_card_val, player_aces, cur_move)]
    return cur_move


# In[8]:


import pandas as pd
out_rows=[]
for i in range(100000):
    val_list=[]
    c_score = simulate_game(False)
    for i, (player_total, dealer_card_val, player_aces, cur_move) in enumerate(reversed(val_list)):
        score = 1.0*c_score if i==0 else 0.4+0.1*c_score
        out_rows+=[{'result': score,
                    'player_total': player_total, 
                    'dealer_card_val': dealer_card_val, 
                    'player_aces': player_aces, 
                    'cur_move': cur_move
                   }]
move_df = pd.DataFrame(out_rows)
print("Win Percentages: %2.2f" % (move_df[move_df['result'].isin([0.0, 1.0])]['result'].mean()*100))
move_df.head(10)


# In[19]:


#MAKE A MODEL
from graphviz import Source
from IPython.display import SVG
from sklearn.tree import export_graphviz
x_vars=['player_total', 'dealer_card_val', 'player_aces']
show_tree = lambda x: SVG(Source(export_graphviz(x, out_file=None, feature_names=x_vars)).pipe(format='svg'))


from sklearn.tree import DecisionTreeRegressor
hit_df = move_df.query('cur_move').groupby(x_vars).agg('mean').reset_index()
hit_tree = DecisionTreeRegressor(max_leaf_nodes = 5)
hit_tree.fit(hit_df[x_vars], hit_df['result'])
show_tree(hit_tree)


# In[20]:


stand_df = move_df.query('not cur_move').groupby(x_vars).agg('mean').reset_index()
stand_tree = DecisionTreeRegressor(max_leaf_nodes = 5)
stand_tree.fit(stand_df[x_vars], stand_df['result'])
show_tree(stand_tree)


# In[21]:


def should_hit(player_total, dealer_card_val, player_aces):
    hit_result = hit_tree.predict(
        np.reshape([player_total, dealer_card_val, player_aces], 
                   (1, -1)))[0]
    stand_result = stand_tree.predict(
        np.reshape([player_total, dealer_card_val, player_aces], 
                   (1, -1)))[0]
    return hit_result>stand_result # a slightly more conservative strategy
blackjack.simulate(n_games=50000)


# **SO, HOW MANY LEAF NODES DO WE ACTUALLY NEED?**

# In[18]:


for i in [4,5,6,7,8,9,10]:
    hit_tree = DecisionTreeRegressor(max_leaf_nodes = i)
    hit_tree.fit(hit_df[x_vars], hit_df['result'])
    stand_tree = DecisionTreeRegressor(max_leaf_nodes = i)
    stand_tree.fit(stand_df[x_vars], stand_df['result'])
    print("Running model with max_leaf_nodes=%s" % (i))
    blackjack.simulate(n_games=500000)


# In[ ]:


hit_tree = DecisionTreeRegressor(max_leaf_nodes = i)
hit_tree.fit(hit_df[x_vars], hit_df['result'])
stand_tree = DecisionTreeRegressor(max_leaf_nodes = i)
stand_tree.fit(stand_df[x_vars], stand_df['result'])


# Based upon these results it appears that either 5 or 6 leaf nodes would be acceptable.  But, since it does not help to make it more complicated then 5 leaf nodes would be the best solution.  No since in making things more complicated if it results in zero improvement on the results!

# In[22]:


#LET'S SEE IF WE CAN FIGURE OUT WHAT THE TREE IS LOOKING AT
#https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree
#SOME NICE CODE TO TRY TO DECIPHER A TREE
from sklearn.tree import _tree

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)

    
tree_to_code(hit_tree, x_vars)


# In[23]:


tree_to_code(stand_tree, x_vars)


# # Discuss Your Results
# 
# How high can you get your win rate? We have a [discussion thread](https://www.kaggle.com/learn-forum/58735#latest-348767) to discuss your results. Or if you think you've done well, reply to our [Challenge tweet](https://twitter.com/kaggle) to let us know.

# ---
# This exercise is from the **[Python Course](https://www.kaggle.com/Learn/python)** on Kaggle Learn.
# 
# Check out **[Kaggle Learn](https://www.kaggle.com/Learn)**  for more instruction and fun exercises.
