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


class BasePricePolicy(object):
    def __init__(self):
        pass

    def __call__(self, days_left, tickets_left, demand_level):
        # raise NotImplementedError()
        return demand_level - 10


# A simple price policy.

# In[ ]:


class SimplePricePolicy(BasePricePolicy):
    def __init__(self):
        super(SimplePricePolicy, self).__init__()
    
    def __call__(self, days_left, tickets_left, demand_level):
        if days_left > 1:
            qty = 10
        else:
            qty = min(tickets_left, demand_level / 2)

        price = demand_level - qty
        return price


# 

# In[ ]:


import numpy as np


# \begin{equation}
# \begin{aligned}
#     & \underset{\lambda \geq 0}{\text{minimize}}
#       & & \lambda + \tfrac12 \sum_{i=1}^m (y_i - \lambda)_+^2
#           \,, \\
#     & & & \sum_{i=1}^m (y_i - \lambda)_+ \leq 1
#         \,.
# \end{aligned}    
# \end{equation}
# 
# However for all $\lambda$ satisfying $\sum_{i=1}^m (y_i - \lambda)_+ \leq 1$,
# the objective is non-decreasing (check the derivative), whence the problem
# is equivalent to:
# \begin{equation}
#     \inf_{\lambda \geq 0} \{\lambda
#         \colon \lambda \geq 0\,, \sum_{i=1}^m (y_i - \lambda)_+ \leq 1\}
#         \,.
# \end{equation}

# In[ ]:


from scipy.optimize import brentq

def opt_value(q_bar, A):
    if q_bar <= 0:
        return np.zeros_like(A[:, :1])

    # for the y-vector
    y = 0.5 * A / q_bar
    if y.ndim < 2:
        y = y.reshape(1, -1)

    # solve the opt problem
    gfun = lambda C, z: 1 - np.sum(np.maximum(z - C, 0))
    C_opt = np.array([[brentq(gfun, 0, max(max(x), 0), args=(x,))
                       if gfun(0, x) < 0 else 0 for x in y]]).T

    # get the solution and the value
    x_opt = np.maximum(y - C_opt, 0)
    V_opt = C_opt + 0.5 * np.sum(x_opt**2, axis=-1, keepdims=True)

    return 2 * q_bar * q_bar * V_opt


# \begin{align}
#     V_t(\bar{q}, A_{t:},\, \ldots,\, A_T)
#         & = \max_{q_{t:}\geq 0\,, \sum_{s=t}^T q_s \leq \bar{q}}
#                 \sum_{s=t}^T \tfrac{A_s^2}4 - \tfrac{4 \bar{q}^2}4
#                 \sum_{s=t}^T \bigl( \tfrac{A_s}{2\bar{q}} - \tfrac{q_s}{\bar{q}} \bigr)^2 \\
#         & = \bigl\{ y_s = \tfrac{A_s}{2\bar{q}} \bigr\}
#           = 2 \bar{q}^2 \bigl( \tfrac12 \|y\|^2 - \min_{x \geq 0,\, \|x \|\leq 1} \tfrac12 \| y - x \|^2 \bigr) \\
#         & = 2 \bar{q}^2 \inf \bigl\{\lambda + \tfrac12 \sum_{i=1}^m (y_i - \lambda)_+^2
#             \colon \sum_{i=1}^m (y_i - \lambda)_+ \leq 1,\, \lambda \geq 0 \bigr\}
#         \,,
# \end{align}

# In[ ]:


from scipy.optimize import fmin_cobyla

class SimulatedPricePolicy(BasePricePolicy):
    def __init__(self, n_simulations=100):
        super(SimulatedPricePolicy, self).__init__()
        self.n_simulations = n_simulations
    
    def __call__(self, days_left, tickets_left, demand_level):
        if days_left < 2:
            qty = min(tickets_left, demand_level / 2)
        else:
            A_future = np.random.uniform(100, 200, size=(self.n_simulations, days_left - 1))

            f_fun = lambda x: - ((demand_level - x) * x + opt_value(tickets_left - x, A_future).mean())
            gfun1 = lambda x: x
            gfun2 = lambda x: tickets_left - x
            qty_opt = fmin_cobyla(f_fun, tickets_left / 2, (gfun1, gfun2))

            qty = qty_opt.item()

        # end if

        price = demand_level - qty
        return price


# The dynamic problem is:
# \begin{align}
#     V_t(\bar{q}, A)
#         &= \max_{q \in [0, \bar{q}]} (A - q) q + \mathbb{E}_z V_{t+1}(\bar{q}-q, z)
#         \,, \\
#     V_T(\bar{q}, A)
#         &= \max_{q \in [0, \bar{q}]} (A - q) q
#          = \bigl(A - q_T^* \bigr) q_T^* \Big\vert_{q_T^* = \min\{\tfrac12 A, \bar{q}\}}
#          = \tfrac14 \bigl(A^2 - (A - 2 \bar{q})_+^2 \bigr)
#         \,.
# \end{align}

# In[ ]:


class ApproxPricePolicy(BasePricePolicy):
    def __init__(self, demand_bins):
        super(ApproxPricePolicy, self).__init__()
        self.demand_bins = demand_bins

    def __call__(self, days_left, tickets_left, demand_level):
        tickets_left, days_left = int(tickets_left), int(days_left)
        self.compute_dp(days_left, tickets_left)

        # get the bin of the current demand level
        demand_level_bin = np.digitize(demand_level, self.demand_bins) - 1

        # the optimal quantity is just the argmax (by construction)
        qty = np.argmax(self.current_[demand_level_bin, :tickets_left + 1]
                        + self.value_[days_left - 1, tickets_left::-1])

        price = demand_level - qty
        return price

    def compute_dp(self, days_left, tickets_left):
        dp_computed_, tickets_left = hasattr(self, "value_"), int(tickets_left)
        if dp_computed_:
            n_days, n_tickets_p1 = self.value_.shape
            dp_computed_ = (n_days >= days_left) and (n_tickets_p1 > tickets_left)

        if dp_computed_:
            return

        # It is necessary to recompute
        self.value_, self.current_ = self._compute_dp(days_left, tickets_left)
    
    def _compute_dp(self, n_days, n_tickets):
        # compute (A - q) * q, q=0..n_tickets
        current = np.zeros((len(self.demand_bins), 1 + n_tickets), dtype=float)
        for q in range(1 + n_tickets):
            current[:, q] = (self.demand_bins - q) * q

        # Compute \mathbb{E}_A \max_{q\in [0, x]} V_{t+1}(x, q; A)
        #  for all x=0..n_tickets, t=1..n_days
        V_tilde = np.zeros((n_days, 1 + n_tickets), dtype=float)
        for t in range(1, n_days):
            # V_t(x, q; A) = (A - q) * q + \tilde{V}_{t+1}(x - q), q=0..x
            # V_t(x; A) = \max_{q=0}^x V_t(x, q, A)
            # \tilde{V}_t(x) = \mathbb{E}_A V_t(x; A)
            for x in range(1 + n_tickets):
                V_txq = current[:, :x + 1] + V_tilde[t - 1, np.newaxis, x::-1]
                V_tilde[t, x] = np.mean(np.max(V_txq, axis=-1), axis=0)
            # end for
        # end for
        return V_tilde, current


# In[ ]:


# pricing_function = SimplePricePolicy()


# Simulated policy is slow, because it fails to utizie the recursive structure of the Bellman equation.
# It also uses `brentq` root finder (bisection), and COBYLA solver for optimizing quantity, which slow it down further.

# In[ ]:


# pricing_function = SimulatedPricePolicy(100)


# This pricing policy is a direct implementation of the Bellman recursion. By making the grid for $A$ coarser it is possible to trade off accuracy for speed.

# In[ ]:


pricing_function = ApproxPricePolicy(np.linspace(100, 200, num=2001))


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

# In[ ]:




