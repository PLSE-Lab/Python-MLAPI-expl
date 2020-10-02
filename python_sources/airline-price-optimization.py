#!/usr/bin/env python
# coding: utf-8

# # Intro
# 
# Data scientists tend to focus on **prediction** because that's where conventional machine learning excels. But real world decision-making involves both prediction and **optimization**.  After predicting what will happen, you decide what to do about it.
# 
# Optimization gets less attention than it deserves. So this micro-challenge will test your optimization skills as you write a function to improve how airlines set prices.
# 
# ![Imgur](https://i.imgur.com/AKrbLMR.jpg)
# 
# 
# # The Problem
# 
# You recently started Aviato.com, a startup that helps airlines set ticket prices. 
# 
# Aviato's success will depend on a function called `pricing_function`.  This notebook already includes a very simple version of `pricing_function`.  You will modify `pricing_function` to maximize the total revenue collected for all flights in our simulated environment.
# 
# For each flight, `pricing_function` will be run once per (simulated) day to set that day's ticket price. The seats you don't sell today will be available to sell tomorrow, unless the flight leaves that day.
# 
# Your `pricing_function` is run for one flight at a time, and it takes following inputs:
# - **Number of days until the flight**
# - **Number of seats they have left to sell**
# - **A variable called `demand_level` that determines how many tickets you can sell at any given price. **
# 
# The quantity you sell at any price is:
# > quantity_sold = demand_level - price
# 
# Ticket quantities are capped at the number of seats available.
# 
# Your function will output the ticket price.
# 
# You learn the `demand_level` for each day at the time you need to make predictions for that day. For all days in the future, you only know `demand_level` will be drawn from the uniform distribution between 100 and 200.  So, for any day in the future, it is equally likely to be each value between 100 and 200.
# 
# In case this is still unclear, some relevant implementation code is shown below.
# 
# # The Simulator
# We will run your pricing function in a simulator to test how well it performs on a range of flight situations.  **Run the following code cell to set up your simulation environment:**

# In[ ]:


import sys
sys.path.append('../input')
from flight_revenue_simulator import simulate_revenue, score_me


# In case you want to check your understanding of the simulator logic, here is a simplified version of some of the key logic (leaving out the code that prints your progress). If you feel you understand the description above, you can skip reading this code.
# 
# ```
# def _tickets_sold(p, demand_level, max_qty):
#         quantity_demanded = floor(max(0, p - demand_level))
#         return min(quantity_demanded, max_qty)
# 
# def simulate_revenue(days_left, tickets_left, pricing_function, rev_to_date=0, demand_level_min=100, demand_level_max=200):
#     if (days_left == 0) or (tickets_left == 0):
#         return rev_to_date
#     else:
#         demand_level = uniform(demand_level_min, demand_level_max)
#         p = pricing_function(days_left, tickets_left, demand_level)
#         q = _tickets_sold(demand_level, p, tickets_left)
#         return _total_revenue(days_left = days_left-1, 
#                               tickets_left = tickets_left-q, 
#                               pricing_function = pricing_function, 
#                               rev_to_date = rev_to_date + p * q,
#                               demand_level_min = demand_level_min,
#                               demand_level_max = demand_level_max
#                              )
# ```
# 
# # Your Code
# 
# Here is starter code for the pricing function.  If you use this function, you will sell 10 tickets each day (until you run out of tickets).

# In[ ]:


def pricing_function(days_left, tickets_left, demand_level):
    """Sample pricing function"""
    price = demand_level - 10
    return price


# To see a small example of how your code works, test it with the following function:

# In[ ]:


simulate_revenue(days_left=7, tickets_left=50, pricing_function=pricing_function, verbose=True)


# You can try simulations for a variety of values.
# 
# Once you feel good about your pricing function, run it with the following cell to to see how it performs on a wider range of flights.

# In[ ]:


score_me(pricing_function)


# # Discuss
# Want to discuss your solution or hear what others have done?  There is a [discussion thread](https://www.kaggle.com/general/62469) just for you.

# ---
# *This micro-challenge is from an exercise in an upcoming Optimization course on **[Kaggle Learn](https://www.kaggle.com/Learn?utm_medium=website&utm_source=kaggle.com&utm_campaign=micro+challenge+2018)**.  If you enjoyed this challenge and want to beef up your data science skills, you might enjoy our other courses.*
