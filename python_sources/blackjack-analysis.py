#!/usr/bin/env python
# coding: utf-8

# # Black Jack simulator
# 
# In this kernel, I am going to explore the way to encounter the winning strategy of BlackJack and how to arrive at the winning program.
# For now , I will be considering an infinite set of decks that can be dealed and work from the simplest  variation of black jack to more complex versions.

# In[ ]:


# First, I am importing some packages that I think I may need later on
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


# Now, I will create the deck of cards and their corresponding probabilities.
# We have cards 2,3,4,5,6,7,8,9,10 and Jack, Queen, King and Ace cards in the deck, making for a total of 13 cards that can be dealt with the same probability on next draw.
# 
# I will create a dictionary consisting of the value of cards and count of cards. 
# For simplicity, I will consider the Jack, Queen and King cards to be no different from a number 10 card( which makes no difference)
# and the ace card to hold a default value of 1. I am not considering the high value of 11 Ace card for now. So, we can consider the deck to have cards 1 to 9 and 4 10 cards. 
# 
# Forgive my jargon for black jack, as I am not familiar with it.

# In[ ]:


table = list(range(1,10)) + [10]*4
table


# So as we can see above, the table contains the list of cards that can occur equally on each draw, just like the numbers on a 13 faced dice, except that 10 is repeated 4 times.
# Now, I will construct a probability dictionary so that I can get the probability of each card occuring on each draw.

# In[ ]:


draw_count = {str(card_value):table.count(card_value) for card_value in set(table)}
print('Card count in each draw',draw_count)
total_cards = sum(card_count for card_count in draw_count.values())
print('Net cards', total_cards)
draw = {k:(v/total_cards) for k,v in draw_count.items()}
print('Probability of each card:', draw)


# As seen above, each card can be drawn with probability 1/13 = 7.7%, with exception of the no 10 card, which can be drawn with 4/13 = 30.8% probability.
# 
# Now, I will start to construct a skeletal function of the black jack probability measure.
# Here, I am just constructing a function which will get the probability of a player win based on the total count of cards that a player has and the count of cards that a dealer has. I will compute the stay probability and the hit probability of the player recursively so we can make a decision at any stage whether the player needs to hit or stay.

# In[ ]:


def bj_probability(player_count, dealer_count):
    '''Returns a list as follows: [probability player wins if he opts to stay, probability player wins if he opts to hit]'''
    if player_count > 21: return [0,0] # Player busts
    if dealer_count > 21: return [1,1] # Dealer busts
    if dealer_count >=17: 
        prob_win = 1*(player_count > dealer_count) # Player only wins if his count is higher than dealer's 
                                                   # once dealer hits 17 or more
        return [prob_win, prob_win]
    # Here is for other undecided scenarios
    # Stay prob = probability of drawing each card * winning after dealer draws each card and player opts to stay again
    stay_prob = sum(draw[card]  * bj_probability(player_count, dealer_count + int(card))[0]  for card in draw)
    # Hit prob = probability of drawing each card * winning after dealer draws each card 
    # and player decides to play hit or stay depending on max prob of winning 
    hit_prob = sum(draw[card] * max(bj_probability(player_count + int(card), dealer_count)) for card in draw)
    return [stay_prob, hit_prob]


# ### Converting the above into a DP problem
# As an exercise, you can time the below queries.
# ```
# %%time
# print(bj_probability(16,16))
# ```
# 
# ```
# %%time
# print(bj_probability(14,14))
# ```
# 
# ```
# %%time
# print(bj_probability(12,12))
# ```
# 
# ```
# %%time
# print(bj_probability(11,11))
# ```
# -------------------------------------------------------------------------
# Try running the below query and estimate the time it will take as an exercise(Wait till the black death of universe)
# 
# ```
# %%time
# print(bj_probability(0,0))
# ```
# Considering it took about a minute to get probability of (11,11), I will use DP version, which is surprisingly easy.
# 

# In[ ]:


arr = [[None for i in range(23)] for j in range(23)]

def bj_probability_with_dp(player_count, dealer_count):
    '''Returns a list as follows: [probability player wins if he opts to stay, probability player wins if he opts to hit]'''
    if player_count > 21: return [0,0] # Player busts
    if dealer_count > 21: return [1,1] # Dealer busts
    if arr[player_count][dealer_count] is None:
        if dealer_count >=17: 
            prob_win = 1*(player_count > dealer_count) # Player only wins if his count is higher than dealer's 
                                                       # once dealer hits 17 or more
            result =  [prob_win, prob_win]
        else:
            # Here is for other undecided scenarios
            # Stay prob = probability of drawing each card * winning after dealer draws each card and player opts to stay again
            stay_prob = sum(draw[card]  * bj_probability_with_dp(player_count, dealer_count + int(card))[0]  for card in draw)
            # Hit prob = probability of drawing each card * winning after dealer draws each card 
            # and player decides to play hit or stay depending on max prob of winning 
            hit_prob = sum(draw[card] * max(bj_probability_with_dp(player_count + int(card), dealer_count)) for card in draw)
            result = [stay_prob, hit_prob]
        arr[player_count][dealer_count] = result
    return arr[player_count][dealer_count]


# Now, computing the entire probability matrix for all scenarios is easy peasy

# In[ ]:


get_ipython().run_cell_magic('time', '', 'print(bj_probability_with_dp(0,0))')


# Easy solution within 17 ms..

# In[ ]:


df = pd.DataFrame(data=arr, columns=list(range(23)))
df.head(10)


# Now that we represented the probabilities, we can move on to next steps`
# 
# We can compare the results from both approaches to check if they are same

# In[ ]:


for i in range(12,18):
    print(i, 'normal prob', bj_probability(i,i), 'dp prob', bj_probability_with_dp(i,i))


# In[ ]:


df


# In[ ]:


df_1 =df.rename_axis('Player_count', axis='rows').rename_axis('Dealer_count', axis='columns').iloc[:-1,:-1]
df_1


# In[ ]:


def stay_prob(x): return x[0]
df_stay = df_1.applymap(lambda x:stay_prob(x))
df_stay


# In[ ]:


df_hit = df_1.applymap(lambda x:x[1])
df_hit


# In[ ]:


plt.figure(figsize=(7,7))
plt.imshow(df_stay.values)
plt.xlabel('Dealer count')
plt.ylabel('Player count')
plt.title('Stay probabilities colormap')
plt.colorbar()
plt.show()


# In[ ]:


plt.figure(figsize=(7,7))
plt.imshow(df_hit.values)
plt.xlabel('Dealer count')
plt.ylabel('Player count')
plt.title('Hit probabilities colormap')
plt.colorbar()
plt.show()


# As we observe, the hit probabilities get lower and lower as player count gets higher, but not always. As we see, near player count 11, the hit probability is little higher.
# The stay probability follows a pattern too and peaks near 7-10, where the dealer may be more likely to end up below the player

# In[ ]:




