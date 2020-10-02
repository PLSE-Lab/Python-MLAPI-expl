#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
from scipy import sparse

import matplotlib.pylab as plt

from functools import partial
from pathlib import Path


# As we all know, Santa faced the hard problem. I'll show you technique, called Local Search, that might help Santa. It's fun.
# 
# The current idea is basic:
# 
# 1. Initialize any solution (schedule) for all families.
# 2. For n iterations do:
#     1. Find family and day, where to move it
#     2. Move selected family from `dec_day` (decreasing occupancy day) to `inc_day` (increasing occupancy day).
#     3. If cost is better, then the solution is the best one, save it.
#     
# Sounds like a plan! I know couple of hard workers (described as classes below), who will help us, I'll introduce them in the notebook. All described classes are tested, and work as expected (hopefully).
# 
# Let's look at the data again and create several matrices and constants:

# In[ ]:


DATA_DIR = Path("../input/santa-workshop-tour-2019/")
family_data = pd.read_csv(DATA_DIR/"family_data.csv", index_col="family_id")
submission = pd.read_csv(DATA_DIR/"sample_submission.csv", index_col="family_id")

df = family_data.join(submission)
print("shape: %d x %d" % df.shape)
df.head(2)


# In[ ]:


choices = df.loc[:, "choice_0":"choice_9"].values
fam_sizes = df["n_people"].values
n_fams = fam_sizes.shape[0]
n_days = 100


# # The crew
# ## 0. Cost calculation
# 
# The functions below describe preference cost and accounting penalty calculations.
# 
# The `cost_and_occupancy` function was inspired by Faster Cost Function kernel.
# 
# References:
#  - Faster Cost Function: https://www.kaggle.com/xhlulu/santa-s-2019-faster-cost-function-24-s

# In[ ]:


MIN_OCCUPANCY, MAX_OCCUPANCY = 125, 300
OCCUPANCY_PENALTY = 1e4 # Auxiliary cost that will help satisfy occupancy constraint

GIFT_CARD = [0, 50, 50, 100, 200, 200, 300, 300, 400, 500, 500]
BUFFET_OFF = [0, 0, .25, .25, .25, .5, .5, 1., 1., 1., 1.]
BUFFET = 36
RIDE = 398

gift_cost = lambda x: GIFT_CARD[x]
buffet_cost = lambda x: int(BUFFET_OFF[x] * BUFFET)
ride_cost = lambda x: int(max(0, .5 * (x % 11 - 8)) * RIDE)

def preference_cost(choice, n_people):
    """ Function to compute preference cost provided by Santa """
    return gift_cost(choice) +         (buffet_cost(choice) + ride_cost(choice)) * n_people

def make_penalties_matrix(min_fam_size, max_fam_size, n_choices=11):
    """ Create matrix of all possible penalties """
    return np.asarray([[preference_cost(i, n) for i in range(n_choices)]
                        for n in range(min_fam_size, max_fam_size + 1)])

def make_choice_matrix(choices, n_days):
    """ Create auxiliary matrix (n_families, n_days+1)
        with choice numbers as values """
    n_families = choices.shape[0]
    choice_matrix = sparse.lil_matrix((n_families, n_days+1), dtype=np.int8)
    for i, row in enumerate(choices):
        for j in range(len(row)):
            choice_matrix[i, row[j]] = j + 1
    return choice_matrix

def costs_and_occupancy(submissions, fam_sizes, n_days, choice_matrix,
        penalties_matrix, min_fam_size):
    """ Compute array of actual penalties """
    l = submissions.shape[0]
    costs = np.zeros(shape=(l,), dtype=np.float64)
    occupancy = np.zeros(shape=(n_days+1, ), dtype=np.float64)
    for i in range(l):
        day = submissions[i]
        choice = choice_matrix[i, day] - 1
        fam_size = fam_sizes[i]

        costs[i] = penalties_matrix[fam_size - min_fam_size, choice]
        occupancy[day] += fam_size
    return costs, occupancy

def accounting_penalty(n_cur, n_prev):
    """ Function to calculate one day accounting penalty """
    p = .5 + np.abs(n_cur - n_prev) / 50.
    p = np.power(n_cur, p)
    res = (n_cur - 125) / 400
    return res * p


# ## 1. Santa's Accountant
# 
# That's lots of simple maths. It's boring and I don't want to calculate it every time on my own.
# 
# We need the guy, who understands those formulas above. Let's take Santa's Accountant.
# 
# > SantasAccountant works just fine for her purpose. I believe, she won't get changed during the journey.

# In[ ]:


class SantasAccountant(object):

    """ Accountant, who fairly calculates preference costs for each family
        and provides up-to-date each day accountant penalties """

    def __init__(self, choices, fam_sizes, days=None, n_days=100, n_fams=None,
            min_occupancy=MIN_OCCUPANCY, max_occupancy=MAX_OCCUPANCY):

        # Static
        self.choices, self.fam_sizes = choices, fam_sizes
        self.n_days = n_days
        self.n_fams = n_fams or len(fam_sizes)
        self.min_occupancy, self.max_occupancy = min_occupancy, max_occupancy
        self.min_fam_size, self.max_fam_size = tuple(map(int,
            (self.fam_sizes.min(), self.fam_sizes.max())))

        # Solution array
        self.days = self.init_days() if isinstance(days, type(None)) else days

        # Costs calculation
        self.choice_matrix = make_choice_matrix(choices, n_days)
        self.penalties_matrix = make_penalties_matrix(
            self.min_fam_size, self.max_fam_size)
        self.costs_and_occupancy = partial(
            costs_and_occupancy,
            n_days=self.n_days,
            choice_matrix=self.choice_matrix,
            penalties_matrix=self.penalties_matrix,
            min_fam_size=self.min_fam_size)
        self.reset_occupancy()

    def init_days(self):
        """ Random days initialization """
        return np.random.randint(1, self.n_days+1, size=(self.n_fams, ))

    def reset_occupancy(self):
        """ Calculate preference cost, occupancy and accounting
            penalties based on self.days """
        self.preference_costs, self.occupancy =             self.costs_and_occupancy(self.days, self.fam_sizes)
        self.make_acc_penalties()

    def make_acc_penalties(self):
        """ Calculate accounting penalties for all days """
        self.acc_penalties =             np.zeros(shape=(self.n_days+1,), dtype=np.float64)

        for i in range(1, self.n_days):
            self.acc_penalties[i] = self.day_acc_penalty(i)

    def day_acc_penalty(self, day):
        """ Calculate accounting penalties for one day """
        # Following line requires 2 excessive zero values in acc_penalty
        # matrix, but makes life easier
        if day in (0, 100): return 0

        occ = self.occupancy[day]

        # That helps find solutions satisfying occupancy constraint
        if occ > self.max_occupancy:
            return (occ - self.max_occupancy) * OCCUPANCY_PENALTY
        elif occ < self.min_occupancy:
            return (self.min_occupancy - occ) * OCCUPANCY_PENALTY

        return accounting_penalty(occ, self.occupancy[day+1])

    def update_acc_penalty(self, day):
        """ Update account penalties for selected day """
        for d in range(day-1, day+1):
            self.acc_penalties[d] = self.day_acc_penalty(d)

    def get_preference_cost(self, fam_id, day):
        """ Get preference cost for family in selected day """
        choice = self.choice_matrix[fam_id, day] - 1
        fam_size = self.fam_sizes[fam_id] -  self.min_fam_size
        return self.penalties_matrix[fam_size, choice]

    def cost(self):
        """ Resulting cost """
        cost = np.sum(self.preference_costs) + np.sum(self.acc_penalties)
        return cost


# Here's an example of testing SantasAccountant skills, if you want to examine her:

# In[ ]:


accountant = SantasAccountant(choices, fam_sizes)

true_acc_penalties = accountant.acc_penalties
for i in 1, 50, 99:
    accountant.update_acc_penalty(i)
    test_acc_penalties = accountant.acc_penalties
    assert np.all(np.equal(true_acc_penalties, test_acc_penalties)), f"Day {i} problem"


# ## 2. Sleigh
# 
# Santa's Sleigh is a very versatile vehicle. 
# 
# One interesting thing about it is it can move a family from one day to another day between the schedules. SantasAccountant is a good guy, but she can only use formulas. So Sleigh is quite useful in our journey. Let's grab it too!
# 
# Moreover, Sleigh can not only move families between days, it also can forecast cost improvement before moves! Wow... That might be helpful.
# 
# Who knows what else it can do?!
# 
# > That class is stable for single moves.

# In[ ]:


class Sleigh (SantasAccountant):

    """ Sleight can move family from one day to another day throught schedules
        and forecast any move's cost improvement """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_acc_penalty(self, day):
        """ Get acc penalty between day - 1 and day + 1 """
        acc_penalty = [self.acc_penalties[day + i - 1] for i in range(2)]
        return acc_penalty

    def try_acc_penalty(self, inc_day, fam_size, dec_day=None):
        """ Calculate acc penalty between day - 1 and day + 1 if
            day occupancy is increased by fam_size """
        day_occs = list()
        for i in range(3):
            d = inc_day + i - 1
            if d in (0, 101): continue

            day_occs.append(self.occupancy[d])
            if i == 1:
                day_occs[-1] += fam_size

        dec_day = dec_day or 0
        if isinstance(dec_day, type(None)):
            pass
        elif dec_day == inc_day - 1 or dec_day == inc_day + 1:
            add_if_not_day_1 = 1 * (min(dec_day, inc_day) > 1)
            dec_day_id = dec_day - inc_day + add_if_not_day_1
            day_occs[dec_day_id] -= fam_size

        acc_penalty = [accounting_penalty(i, j)
            for i, j in zip(day_occs[:-1], day_occs[1:])]
        return acc_penalty

    def acc_penalty_imp(self, fam_id, dec_day, inc_day):
        """ Calculate change in accounting penalty
            if family moved from dec_day to inc_day """
        fam_size = self.fam_sizes[fam_id]

        # Sort days
        day_0, day_1 = sorted([dec_day, inc_day])
        decrease_0 = np.power(-1, int(day_0==dec_day))

        penalties_0 = self.get_acc_penalty(day_0)
        penalties_1 = self.try_acc_penalty(day_0, decrease_0*fam_size, day_1)

        if day_1 == day_0 + 1:
            penalties_0.pop()
            penalties_1.pop()

        penalties_0 += self.get_acc_penalty(day_1)
        penalties_1 += self.try_acc_penalty(day_1,-decrease_0*fam_size, day_0)

        imp = sum(penalties_1) - sum(penalties_0)

        return imp

    def pref_cost_imp(self, fam_id, dec_day, inc_day):
        """ Calculate change in accounting penalty
            if family moved from dec_day to inc_day """
        dec_day = self.days[fam_id]
        pref_cost_1 = self.get_preference_cost(fam_id, inc_day)
        pref_cost_0 = self.preference_costs[fam_id]
        imp = pref_cost_1 - pref_cost_0
        return imp

    def improvement(self, fam_id, inc_day):
        """ Get improvement if family moved to inc_day """
        dec_day = self.days[fam_id]
        return self.acc_penalty_imp(fam_id, dec_day, inc_day) +             self.pref_cost_imp(fam_id, dec_day, inc_day)

    def move(self, fam_id, inc_day):
        """ Move family to inc_day """
        fam_size = self.fam_sizes[fam_id]
        dec_day = self.days[fam_id]
        self.days[fam_id] = inc_day

        # Update reference cost
        self.preference_costs[fam_id] +=             self.pref_cost_imp(fam_id, dec_day, inc_day)

        # Update occupations
        self.occupancy[dec_day] -= fam_size
        self.occupancy[inc_day] += fam_size

        # Update acc_penalties
        self.update_acc_penalty(inc_day)
        self.update_acc_penalty(dec_day)


# ## 3. Reindeer
# 
# Turned out, that Reindeers can travel through days schedule finding moves for families. They're good at just that, but they should be taught. Let's ride them and let them drive the search.
# 
# Actually, we've taken them because they memorize well, as any deer, duh... Otherwise we could simply lose our precious best schedule.
# 
# > The current Reindeer is very inexperienced and random, it needs to be improved for better search.

# In[ ]:


class Reindeer (Sleigh):

    """ Reindeer drives search. It finds moves and travels with Sleight """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.reset_costs()
        self.best_cost = self.cost()
        self.best_days = self.days.copy()

    def reset_costs(self):
        """ Reset costs list (e.g. to remove early enormous costs) """
        self.costs = list()

    def feasible(self):
        """ Check if solution satisfies occupancy constraint """
        return np.all([
            np.all(self.occupancy[1:] <= self.max_occupancy),
            np.all(self.occupancy[1:] >= self.min_occupancy),])

    def select_dec_day(self, thresh_occ=None):
        """ Select any day with occupancy above threshold """
        thresh_occ = thresh_occ or self.min_occupancy

        possible_days = np.argwhere(np.where(self.occupancy > thresh_occ,
                                     self.acc_penalties, 0)).flatten()
        if len(possible_days) == 0: return 0

        dec_day = np.random.choice(possible_days)
        return dec_day

    def select_family(self, day):
        """ Select random family among families with assigned day """
        fam_id = np.random.choice(
            np.argwhere(self.days == day).flatten())
        return fam_id

    def select_inc_day(self, fam_id, dec_day):
        """ Find the day that minimizes cost function
            for the selected family """
        improvements = np.zeros(shape=(self.choices.shape[1], ))

        # Filter days with low occupancy in separate list
        little_occ_days, big_occ_days = list(), list()
        for i, inc_day in enumerate(self.choices[fam_id]):
            improvements[i] = 9e9 if inc_day == dec_day                 else self.improvement(fam_id, inc_day)
            if self.occupancy[inc_day] < self.min_occupancy:
                little_occ_days.append(i)
            else:
                big_occ_days.append(i)

        # Select day among days with low occupancy first
        got_little = False
        if len(little_occ_days):
            got_little = True
            ix_min_little = np.argmin(improvements[little_occ_days])
            ix_min = little_occ_days[ix_min_little]

        # Select day among others if choice from previous block is bad
        if not got_little or improvements[ix_min_little] > 0:
            ix_min_big = np.argmin(improvements[big_occ_days])
            ix_min_big = big_occ_days[ix_min_big]
            if not got_little or ix_min_big < ix_min:
                ix_min = ix_min_big
        return self.choices[fam_id, ix_min]

    def find_move(self):
        """ Find next family to move and its destination day """
        dec_day = self.select_dec_day(self.max_occupancy)
        if dec_day == 0:
            dec_day = self.select_dec_day(self.min_occupancy)
        fam_id = self.select_family(dec_day)
        inc_day = self.select_inc_day(fam_id, dec_day)

        return fam_id, inc_day

    def travel(self, fam_id, inc_day):
        """ Move family to selected day, check if it improves cost function
            and save if best """
        self.move(fam_id, inc_day)
        cost = self.cost()
        self.costs.append(cost)
        if self.feasible() and cost < self.best_cost:
            self.best_cost = cost
            self.best_days = self.days.copy()

    def search(self, n_steps=100):
        """ Random (with little priorities) local search """
        for _ in range(n_steps):
            fam_id, inc_day = self.find_move()
            self.travel(fam_id, inc_day)


# # Run
# 
# We're ready to travel towards the best schedule. Let's first initialize Reindeer:

# In[ ]:


Prancer = Reindeer(choices, fam_sizes)
Prancer.days


# Now, just sit, relax and let Prancer the Reindeer search the minimal cost.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'c_prev = Prancer.cost()\nPrancer.search(n_steps=50000)\n\nprint(f"Best cost: {Prancer.best_cost:.0f}")\nprint(f"Cost improvement: {c_prev - Prancer.best_cost:.0f}")\nplt.plot(range(len(Prancer.costs)), Prancer.costs)\nplt.show()')


# The current search algorithm gets stuck near $500k, but I haven't thought about search optimization yet.
# 
# I've implemented templates, so anyone can enjoy the ride with the Local Search squad including SantasAccountant, Sleigh and Reindeer, you know... I want to help Santa to save money, so he can buy more socks, more candies, more GPUs, more pianos, more board games and more other Christmas presents for little kids!
# 
# Let's make Christmas merry again!

# In[ ]:


submission['assigned_day'] = Prancer.best_days
final_score = Prancer.best_cost
submission.to_csv(f'submission_{final_score:.0f}.csv')

