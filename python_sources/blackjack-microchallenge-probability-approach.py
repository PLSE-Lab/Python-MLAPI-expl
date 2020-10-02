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

# In[3]:


blackjack.simulate_one_game()


# You can see how your player does in a sample of 50,000 games with the following command:

# In[4]:


blackjack.simulate(n_games=50000)


# # Your Turn
# 
# Write your own `should_hit` function in the cell below. Then run the cell and see how your agent did in repeated play.

# ## Introduction
# 
# 
# 
# After checking the game, there is two possibility to win :  
#     1) The player does not busts and the dealer busts  
#     2) The player does not busts and got a HIGHER (draw is considered as loosing for the player) score than the dealer who does not busts either. That means the player got to have at least 18 if the dealer does not busts.  
#     
# Now, here is what I thought : independent of the player, there is a certain probability that the dealer will have any result from 17 to 21 and bust. And regarding the player total score and the number of aces, there is a given probability that the player busts by taking another card. 
# There is another important point : the game is not simulataneous. If the player have 50% of chance to bust and so does the dealer, the player will loose more than 50% of the time as he played first.
# It is therefore important to take that into account and evaluate the corrected probability to bust considering the fact that the player plays first.
# After that, it seems logical to hit if the corrected probability to busts is below the probability of the dealer to have a higher score than the player.
# 
# 
#   
#   
# - First of all, I simulated 1000000 games for the dealer to catch the probability to have any results depending on the first card. 
# 
# - Then, I created two function, one to see the probability of the dealer to be at least at a given score depending on the first card (e.g : probability to have 19,20 or 21 with a 9 as a first card), and one to see the probability of the player to bust by taking a new card.
# 
# - Then, to do the correction in the probability, i created a simple game where 2 players have a various probability to lose at each turn (and does nothing else), but the player 1 plays first. By playing this game 100000 for different initial probability for each player, I obtained the actual probability of player 1 loosing. I did it for 20 values (from 5% to 100% of initial probability to bust) for the player, and the 10 values obtained before that showed the probability of the dealer to bust regarding it's first card.
# 
# - Finally, I did another function (separate in 3 for clarity) that took the probability of the player and dealer to bust (define previously) and give the corrected probability of the the player to loose, using what was obtained in the previous step.
# 
# - The decision to hit is then simple : if the corrected probability to bust by taking another card is lower than the probability of the dealer to have an higher score than the one player have right now, then the player should hit. If not, he should stay.
# 
# 

# ## 1) Simulate the dealer outcomes depending on the first card

# In[5]:


#Initialization of the functions

import numpy as np
import random as rand
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


numbers_final={2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[]} #Actual number at the end (17,18,19,20,21 or 22 for all numbers above 21)
boom={2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[]} #Wether it busts or not. If bust, result is 1. The smaller the mean, the better for the dealer

#Draw the next card, all card above 9 (10, J, Q, K) or a 10, the 1 is an 11 (A)
def next_card(): 
    num=rand.randint(1,13)
    if num>9:
        num=10
    if num==1:
        num=11
    return num

#One game for the dealer depending on the first card, as if he played alone (draw until 17, busts after 21, ace is 1 if there is a bust)
def game(first_card): 
    ace=0 #Number of ace
    tot=first_card #Total score
    if tot==11: #It means there is an ace
        ace=ace+1
    while True: #Taking card until the score is between 17 or 21, if the score is above 21, if there is an ace, substract 10 point, if there is not, count it as 22 : bust.
        new_card=next_card()
        if new_card==11:
            ace=ace+1
        tot=tot+new_card
        if tot>16:
            if tot>21:
                if ace>0:
                    tot=tot-10
                    ace=ace-1
                else:
                    return 22
            else:
                return tot
                


# In[6]:


#Repeat game and save first card and result
num_game=int(1.e6)
k=0.1
print("Begin")
for i in range(num_game):
    if num_game*k<i:
        print(round(k*100),"%")
        k=k+0.1
    first_card=next_card()
    res=game(first_card)
    numbers_final[first_card].append(res)
    if res==22:
        boom[first_card].append(1)
    else:
        boom[first_card].append(0)
print("Simulation of the",num_game, "games over")


# In[7]:


#Watch the probability to bust depending on the first card
num_list=[]
val_list=[]
for number in boom:
    num_list.append(number)
    val_list.append(np.mean(boom[number]))
    
plt.figure(figsize=(16,12))
plt.plot(num_list,val_list,'ro')
plt.show()    


# In[8]:


#Watch an histogram (in percent) of the probability to have each outcome depending on the first card (22 = bust)
from matplotlib.ticker import PercentFormatter
plt.figure(figsize=(30,18))
k=1
for i in numbers_final:
    plt.subplot(2,5,k)
    k=k+1
    title="Result distribution begining with card "+str(i)
    plt.title(title)
    plt.hist(numbers_final[i],weights=np.ones(len(numbers_final[i])) / len(numbers_final[i]))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.show()


# In[9]:


#create a dictionary with the first number as each key associated with a list that goes this way : first, probability to have 17, then 18, 19, 20, 21 and bust (6 values)
#proba_numbers_final[first_card][k], with k=0 --> 17,  k=1 --> 18,  k=2 --> 19,  k=3 --> 20,  k=4 --> 21,  k=5 --> bust
proba_numbers_final={2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[]}
for num_ini in numbers_final:
    for i in [17,18,19,20,21,22]:
        proba_numbers_final[num_ini].append(numbers_final[num_ini].count(i)/len(numbers_final[num_ini]))
    


# ## 2) Correction as the player play first

# In[10]:


def game(p1,p2): #small game where each player have a probability to loose at each turn, depending on the inputs, and where p1 always play first.
    while True:
        if rand.random()<p1:
            return 1
        elif rand.random()<p2:
            return 0


# In[68]:


#p2 will always be the proba to bust for the dealer, so it will be values of val_list (10 values)
n_max=100000
k=0
proba_eq_ref=np.zeros((20,len(val_list)))
print("Begin")
for p1 in np.arange(1,21):
    n2=-1
    print(k*5.,"%")    
    k=k+1
    for p2 in val_list:
        n2+=1
        for i in range(n_max):
            proba_eq_ref[p1-3,n2]+=game(p1/20.,p2)/n_max
print("End")


# In[69]:


import seaborn as sns
plt.figure(figsize=(18,16))
sns.heatmap(proba_eq_ref, annot=True)


# ## 3) Create the functions that give the different probabilities to win/bust

# In[70]:


import math


def proba_score_dealer(dealer_card_val, score): #proba that the dealer get "score" OR HIGHER depending on his first card
    if score < 18:
        return 1- proba_numbers_final[dealer_card_val][5]
    elif score == 18:
        return 1- (proba_numbers_final[dealer_card_val][5]+proba_numbers_final[dealer_card_val][0])
    elif score == 19:
        return 1- (proba_numbers_final[dealer_card_val][5] + proba_numbers_final[dealer_card_val][0] + proba_numbers_final[dealer_card_val][1])
    elif score == 20:
        return proba_numbers_final[dealer_card_val][3]+proba_numbers_final[dealer_card_val][4]
    elif score == 21:
        return proba_numbers_final[dealer_card_val][4]
    else:
        return 0

    
def proba_bust_player(player_total,player_aces): #proba that the player busts depending on it's curent total
    if player_total<12:
        return 0
    else:
        if player_aces>0:
            val=21-player_total
            num_bad_cards=13-val-3
            for i in range(num_bad_cards):
                if i==0:
                    val+=(21-player_total)*4/13
                else:
                    val+=(21-player_total+i)/13
        else:
            val=21.-player_total
        return 1-val/13.


######################
######################

#Then, there is the correction due to the player playing first

def round_5(num): #give the number in percent rounded at the nearest 5 (e.g 0.33 --> 35(%)) with a minimum of 5%
    if num<0.05:
        return 5
    if num>0.95:
        return 100
    num_100=num*100
    a=math.floor(num_100)
    if num_100-a<0.25:
        num_100=a
    elif num_100-a<0.75:
        num_100=a+0.5
    else:
        num_100=a+1
    
    return num_100


def proba_equivalence(p1,dealer_card_val): #give the actual probability for p1 to lose a game given the probability (rounded at 5%)
    p1_5=int(round_5(p1)/5)
    p2=dealer_card_val-2
    
    return proba_eq_ref[p1_5-1,p2]

def actual_proba_loose(player_total, player_aces, dealer_card_val): #actual proba of p1 to "loose" the game when taking the correction "he plays first" into account (loose here means in many game of this situation he would bust before the dealer)
    if proba_bust_player(player_total,player_aces)==0:
        return 0
    elif proba_bust_player(player_total,player_aces)==1:
        return 1
    else:
        return proba_equivalence(proba_bust_player(player_total,player_aces),dealer_card_val)




# ## 4) Define the should_hit function and simulate the games

# In[73]:


#Function should_hit and simulation
count_hit12=0

def should_hit(player_total, dealer_card_val, player_aces):
    global count_hit12
    if actual_proba_loose(player_total, player_aces, dealer_card_val)<proba_score_dealer(dealer_card_val,player_total):
        if player_total>11:
            count_hit12+=1
        return True
    else:
        return False


# In[74]:


#Simulation of num_game
num_game=1000000
blackjack.simulate(n_games=num_game)
print("Average number of hit when the player have more than 11 points : ",count_hit12/num_game)


# # Discuss Your Results
# 
# How high can you get your win rate? We have a [discussion thread](https://www.kaggle.com/learn-forum/58735#latest-348767) to discuss your results. Or if you think you've done well, reply to our [Challenge tweet](https://twitter.com/kaggle) to let us know.

# ---
# This exercise is from the **[Python Course](https://www.kaggle.com/Learn/python)** on Kaggle Learn.
# 
# Check out **[Kaggle Learn](https://www.kaggle.com/Learn)**  for more instruction and fun exercises.
