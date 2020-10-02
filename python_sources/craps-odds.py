#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Let's play Craps! #
# 
# A traditional casino dice game, craps is a betting game wherein bets are resolved by the throw of two six-sided dice.
# 
# # To Pass or Don't Pass #
# 
# The game of craps is broken into two segments, the come out roll and the roll for the point. There are two bets one can make on the come out rollm, the Pass bet or Don't Pass bet. 
# 
# Here's what can happen, depending on the results of the come out roll:
# * 7 or 11 : the Pass bet wins the Don't Pass bet loses, and the game restarts with another come out roll
# * 2, 3: the Pass bet loses, the Don't Pass bet wins, and the game restarts with another come out roll
# * 12: the Pass bet loses, the Don't Pass bet is ignored (you neither win nor lose), and the game restarts with another come out roll
# * 4, 5, 6, 8, 9, 10: neither bet wins, a marker (called the point) is placed on this number and the game moves to the second stage where the player rolls for the point
# 
# If you make it to the second stage of the game the Pass and Don't Pass bets from the first round stay put. At this point, dice are rolled until the game is resolved as follows:
# * 7: the Pass bet loses and the Don't Pass bet wins
# * the dice roll matches the point number: the Pass bet wins and the Don't Pass bet loses
# 
# Generally, a Pass bet is seen as betting *with* the person rolling the dice (that is, you are betting they will hit a 7, 11, or make the point) while a Don't Pass bet is seen as betting *against* the shooter.
# 
# Let's run some games and see which bet wins more often. 
# 
# Specifically, let's run the following simulation 10,000 times: start with a chip count of 100 and play 100 dice throws through to completion (these numbers can be adjusted in the code by changing the value of the 'number_of_iterations' and 'number_of_throws' variables, respectively).

# In[ ]:


np.random.seed(187)

# create function that rolls dice
def dice_roll():
    """Returns the cumulative result of two d6 dice"""
    die1 = np.random.randint(1,7)
    die2 = np.random.randint(1,7)
    rollresult = die1 + die2
    
    return rollresult

# create function that runs game
def simple_craps_game(bettype, betamount, chipcount):
    """plays a simple game of craps, takes type of bet (pass / don't pass), bet amount, and current chip count as args, returns updated chip count based on game results"""
    # get come out dice roll
    comeoutroll = dice_roll()
    
    # determine results from come out roll
    if (comeoutroll in [7, 11] and bettype == "Pass") :
         return betamount + chipcount    
    elif (comeoutroll in [2, 3, 12] and bettype == "Pass") :
        return chipcount - betamount    
    elif (comeoutroll in [7, 11] and bettype == "Don't Pass") :
        return chipcount - betamount   
    elif (comeoutroll == 12 and bettype == "Don't Pass") :
        return chipcount
    
    point = comeoutroll
    
    # roll for point    
    pointroll = dice_roll()
    
    # determine results if rolling for the point    
    if bettype == "Pass" :
        while (pointroll != point or pointroll != 7) :
            pointroll = dice_roll()           
            if pointroll == point :
                return betamount + chipcount            
            elif pointroll == 7 :
                return chipcount - betamount
        
    if bettype == "Don't Pass" :
        while (pointroll != point or pointroll != 7) :
            pointroll = dice_roll()            
            if pointroll == point :
                return chipcount - betamount            
            elif pointroll == 7 :
                return chipcount + betamount


# In[ ]:


# set up random walk through X number of games for Pass and Don't Pass bet types
all_pass = []
all_dontpass = []

# play with the random walks
number_of_throws = 100
number_of_iterations = 10000

# PASS BET
for y in range(number_of_iterations) :
    chip_count_result = 100
    random_walk_pass = []
    for x in range(number_of_throws) :
        chip_count_result = simple_craps_game("Pass",5,chip_count_result)
        random_walk_pass.append(chip_count_result)     
    all_pass.append(random_walk_pass)

all_pass_t = np.transpose(np.array(all_pass))   
all_pass_final_chip_count = all_pass_t[-1,:]

print("Average Chip Count =", np.average(all_pass_final_chip_count),"\nMedian Chip Count =", np.median(all_pass_final_chip_count))
plt.hist(all_pass_final_chip_count,20,edgecolor='black')
plt.title("Pass Bet Histogram")
plt.xlabel("Final Chip Count")
plt.show()


# DON'T PASS BET
for y in range(number_of_iterations) :
    chip_count_result = 100
    random_walk_dontpass = []
    for x in range(number_of_throws) :
        chip_count_result = simple_craps_game("Don't Pass",5,chip_count_result)
        random_walk_dontpass.append(chip_count_result)
    all_dontpass.append(random_walk_dontpass)
    
all_dontpass_t = np.transpose(np.array(all_dontpass))
all_dontpass_final_chip_count = all_dontpass_t[-1,:]

print("Average Chip Count =", np.average(all_dontpass_final_chip_count),"\nMedian Chip Count =", np.median(all_dontpass_final_chip_count))
plt.hist(all_dontpass_final_chip_count,20,edgecolor='black')
plt.title("Don't Pass Bet Histogram")
plt.xlabel("Final Chip Count")
plt.show()


# # Playing the Field #
# 
# At any point in time you can place a bet on the Field, which returns winnings after each roll. The results are as follows:
# * 3, 4, 9, 10, 11: the bet wins
# * 5, 6, 7, 8: the bet loses
# * 2, 12: the bet wins and pays out double
# 
# Let's run some games and see if it makes sense to play the Field.

# In[ ]:


all_field = []

number_of_throws = 100
number_of_iterations = 10000

for y in range(number_of_iterations):

    field_walk = []
    chip_count = 100
    bet_amount = 5
    
    for x in range(number_of_throws) :
        roll = dice_roll()
        if roll in [3, 4, 9, 10, 11] :
            chip_count = chip_count + bet_amount
        elif roll in [5, 6, 7, 8] :
            chip_count = chip_count - bet_amount
        elif roll in [2, 12] :
            chip_count = chip_count + (bet_amount * 2)

        field_walk.append(chip_count)
    all_field.append(field_walk)
    
all_field_t = np.transpose(np.array(all_field))
field_final_chip_count = all_field_t[-1,:]

print("Average Chip Count =", np.average(field_final_chip_count),"\nMedian Chip Count =", np.median(field_final_chip_count))
plt.hist(field_final_chip_count,20,edgecolor='black')
plt.title("Field Bet Histogram")
plt.xlabel("Final Chip Count")
plt.show()

    
    
    
    
    
    

