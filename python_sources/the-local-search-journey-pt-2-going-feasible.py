#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pylab as plt

import pandas as pd
import numpy as np
from scipy import sparse

from copy import deepcopy
from functools import partial, reduce
from collections import namedtuple


# In[ ]:


plt.rcParams.update({"font.size": 13, "figure.figsize": (12, 4)})


# In[ ]:


df = pd.read_csv("../input/santa-workshop-tour-2019/family_data.csv", index_col="family_id")

choices = df.loc[:, "choice_0":"choice_9"].values
fam_sizes = df["n_people"].values
n_fams = fam_sizes.shape[0]
n_days = 100


# [Part I](https://www.kaggle.com/kopytok/santa-2019-the-local-search-journey)
# 
# ---
# 
# I've searched with Local Search team for a while and noticed several findings in Reindeers behaviour I want to tell you about.
# 
# You've already met SantasAccountant and Sleigh in the following code block.

# In[ ]:


MIN_OCCUPANCY, MAX_OCCUPANCY = 125, 300
OCCUPANCY_PENALTY = 1e4

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
    pow = .5 + np.abs(n_cur - n_prev) / 50.
    pow = np.power(n_cur, pow)
    res = (n_cur - 125) / 400
    return res * pow


class SantasAccountant(object):

    """ Accountant, who fairly calculates preference costs for each family
        and provides up-to-date each day accountant penalties """

    def __init__(self, choices, fam_sizes, days=None, n_days=100, n_fams=None,
            min_occupancy=MIN_OCCUPANCY, max_occupancy=MAX_OCCUPANCY,
            feasible=False):

        # Trigger
        self.feasible = feasible

        # Static
        self.choices, self.fam_sizes = choices, fam_sizes
        self.n_choices, self.n_days = self.choices.shape[1], n_days
        self.n_fams = n_fams or len(fam_sizes)
        self.min_occupancy, self.max_occupancy = min_occupancy, max_occupancy
        self.min_fam_size, self.max_fam_size = tuple(map(int,
            (self.fam_sizes.min(), self.fam_sizes.max())))

        # Solution array
        self.days = self.init_days() if isinstance(days, type(None)) else days
        self.make_choice_nums()

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

    def get_choice_num(self, fam_id):
        """ Get choice number for one family """
        choice_num = np.argwhere(self.choices[fam_id]==self.days[fam_id])
        if len(choice_num):
            return choice_num.flatten()[0]
        return 10

    def get_worst_choice_num(self):
        """ Get current worst choice number  """
        return int(np.max(self.choice_nums))

    def get_day_priciest_fams(self, day, choice_num=None):
        """ Get list of priciest families in selected day """
        mask = np.argwhere(self.days == day).flatten()
        day_choice_nums = self.choice_nums[mask]
        choice_num = choice_num or np.max(day_choice_nums)
        return np.argwhere(day_choice_nums == choice_num).flatten()

    def make_choice_nums(self):
        """ Make array of choice numbers for each family """
        self.choice_nums = -np.ones(shape=(self.n_fams, ))
        for fam_id in range(self.n_fams):
            self.choice_nums[fam_id] = self.get_choice_num(fam_id)

    def reset_occupancy(self):
        """ Calculate preference cost, occupancy and accounting
            penalties based on self.days """
        self.preference_costs, self.occupancy =             self.costs_and_occupancy(self.days, self.fam_sizes)
        self.make_acc_penalties()
        self.calculate_cost()

    def make_acc_penalties(self):
        """ Calculate accounting penalties for all days """
        self.acc_penalties =             np.zeros(shape=(self.n_days+1,), dtype=np.float64)

        for i in range(1, self.n_days):
            self.acc_penalties[i] = self.day_acc_penalty(i)

    def day_acc_penalty(self, day):
        """ Calculate accounting penalties for one day """
        # Following line requires 2 excessive zero columns in acc_penalty
        # matrix, but makes life easier
        if day in (0, 100): return 0

        occ = self.occupancy[day]

        if occ > self.max_occupancy:
            # I believe that helps to maintain occupancy constraint
            return (occ - self.max_occupancy) * OCCUPANCY_PENALTY                 if not self.feasible else 9e9
        elif occ < self.min_occupancy:
            return (self.min_occupancy - occ) * OCCUPANCY_PENALTY                 if not self.feasible else 9e9

        return accounting_penalty(occ, self.occupancy[day+1])

    def update_acc_penalty(self, day):
        """ Update account penalties for selected day """
        imp = 0
        for d in range(day-1, day+1):
            acc_penalty = self.day_acc_penalty(d)
            imp += acc_penalty - self.acc_penalties[d]
            self.acc_penalties[d] = acc_penalty
        return imp

    def get_preference_cost(self, fam_id, day):
        """ Get preference cost for family in selected day """
        choice = self.choice_matrix[fam_id, day] - 1
        fam_size = self.fam_sizes[fam_id] -  self.min_fam_size
        return self.penalties_matrix[fam_size, choice]

    def calculate_cost(self):
        """ Calculate cost from prepared arrays """
        self.cost = np.sum(self.preference_costs) + np.sum(self.acc_penalties)

    def update_cost(self, imp):
        """ Update cost with imp """
        self.cost += imp


class Sleigh (SantasAccountant):

    """ Sleight can move family from one day to another day throught schedules
        and forecast any move's cost improvement """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    Move = namedtuple("Move", ("fam_id", "fam_size", "dec_day", "inc_day"))

    def copy(self):
        """ Copy sleigh as a separate instance """
        return deepcopy(self)

    def select_fams(self, constraints, sample=True):
        """ Select families """
        ix = reduce(lambda x, y: x & y, constraints)
        candidate_fams = np.argwhere(ix).flatten()
        if sample and len(candidate_fams) > self.fam_batch_size:
            candidate_fams = np.random.choice(candidate_fams,
                size=self.fam_batch_size, replace=False)
        return candidate_fams

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

    def acc_penalty_imp(self, fam_size, dec_day, inc_day):
        """ Calculate change in accounting penalty
            if family moved from dec_day to inc_day """
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
        pref_cost_1 = self.get_preference_cost(fam_id, inc_day)
        pref_cost_0 = self.preference_costs[fam_id]
        imp = pref_cost_1 - pref_cost_0
        return imp

    def improvement(self, mv):
        """ Get improvement if move mv is made """
        if mv.dec_day == mv.inc_day: return 6e6
        return self.acc_penalty_imp(mv.fam_size, mv.dec_day, mv.inc_day) +             self.pref_cost_imp(mv.fam_id, mv.dec_day, mv.inc_day)

    def move(self, mv):
        """ Make move mv """
        if mv.dec_day == mv.inc_day:
            return
        self.days[mv.fam_id] = mv.inc_day
        self.choice_nums[mv.fam_id] = self.get_choice_num(mv.fam_id)

        # Update reference cost
        imp = self.pref_cost_imp(mv.fam_id, mv.dec_day, mv.inc_day)
        self.preference_costs[mv.fam_id] += imp

        # Update occupations
        self.occupancy[mv.dec_day] -= mv.fam_size
        self.occupancy[mv.inc_day] += mv.fam_size

        # Update acc_penalties and cost
        imp += self.update_acc_penalty(mv.inc_day)
        imp += self.update_acc_penalty(mv.dec_day)
        self.update_cost(imp)


# # Solution space
# 
# First, let me describe you, how Reindeers see the solutions space. They see the solution space as a valley with the best precious solution hidden deep down between the hills (or even inside a cave). We enter the valley from any side in the hills and then we descend.
# 
# We're looking only for feasible solutions and those are connected with feasible tunnels. Each tunnel leads from one feasible place to another. Most of them are dark caves, thus it's hard to see where they lead. 
# 
# All Reindeers can see if the place is inside any feasible tunnel, but few can see where those tunnels lead. For example, Prancer the Reindeer can only examine feasibility on the spot.
# 
# <img src="https://i.ibb.co/Lg68w77/2019-12-11-00-05-11.jpg" alt="drawing" width="600"/>

# # Fun annealing 
# 
# It was noticed, that every Reindeer needs to have fun sometimes, otherwise they get bored fast. I understand that. Everybody needs to have a rest, but the job should be done.
# 
# I've found out, that it's possible to control Reindeer's Fun level. I've studied physics and the mechanism of Reindeer Fun reminds me of temperature annealing. You can have a closer look at it in the following block. 

# In[ ]:


def proba(imp, fun):
    """ Probability of not being greedy """
    return np.exp(imp / fun)

def get_fun(x, max_fun, fun_mult, n_reset=None):
    """ At every step fun decreases with fun_mult rate,
        starting with max_fun """
    return max_fun * np.power(fun_mult, x % n_reset if n_reset else x)
    
def generate_and_plot_fun(max_imp, max_fun, fun_mult, n, n_reset):
    """ Generate and plot fun """
    x = np.linspace(0, n)
    fig, axs = plt.subplots(ncols=2, sharey=True)
    
    fun = get_fun(x, max_fun, fun_mult)
    xx, yy = np.meshgrid(fun, np.arange(0, max_imp))
    z = proba(-yy.ravel(), xx.ravel()).reshape(xx.shape)
    axs[0].contourf(xx, yy, z, 50, alpha=.8, cmap="RdYlGn")
    axs[0].set_ylabel("Bad improvement")
    axs[0].set_xlabel("Fun")
    axs[0].grid(axis="y")
    
    fun = get_fun(x, max_fun, fun_mult, n_reset)
    ff, yy = np.meshgrid(fun, np.arange(0, max_imp))
    xx, _  = np.meshgrid(x, np.arange(0, max_imp))
    z = proba(-yy.ravel(), ff.ravel()).reshape(xx.shape)
    cntf = axs[1].contourf(xx, yy, z, 50, alpha=.8, cmap="RdYlGn")
    plt.colorbar(cntf, ax=axs[1])
    axs[1].grid(axis="y")
    axs[1].set_xlabel("Step")
    
    plt.suptitle("Probability of maybe")
    plt.show()


# In[ ]:


feed_dict = dict(
    max_imp  = int(4e2),
    max_fun  = 100,
    fun_mult = .98,
    n_reset  = 300,
    n        = 1000,
)
generate_and_plot_fun(**feed_dict)


# Sometimes Reindeers get stuck in a hole. In case they don't feel Fun, they dig deeper without realizing they're stuck. On the contrary, when they feel Fun, they may follow any random squirrel and get unstack easily (but often I want them to go deeper).

# # Reindeers
# 
# All Reindeers have lots in common. Their usual behavior is described in Reindeer class.

# In[ ]:


def maybe(p):
    """ True with probability p """
    return np.random.rand() < p


class Reindeer (Sleigh):

    """ Reindeer drives search. It finds moves and rides with Sleight """

    def __init__(self, min_fun, max_fun, fun_mult, patience=30,
            days_batch_size=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Parameters, that control Reindeer fun. It works the same
        # as temperature annealing.
        self.min_fun, self.max_fun = min_fun, max_fun
        self.fun_mult = fun_mult
        self.fun = self.max_fun

        self.check_imp = False
        self.patience = patience # set best_days as days after that
                                 # number of fun resets without improvement
        self.got_best = False
        self.no_best_resets = 0

        # Number of days, sampled for candidates
        self.days_batch_size = days_batch_size or 6

        self.step = 0

        self.reset_reindeer()
        self.best_cost = 9e9
        self.best_days = self.days.copy()

    def is_feasible(self, days=None):
        """ Check if solution satisfies occupancy constraint """
        if not isinstance(days, type(None)):
            _, occupancy = self.costs_and_occupancy(days, self.fam_sizes)
        else:
            occupancy = self.occupancy
        return np.all([
            np.all(occupancy[1:] <= self.max_occupancy),
            np.all(occupancy[1:] >= self.min_occupancy),])

    def reset_reindeer(self):
        """ Reset costs list (e.g. to remove early enormous costs) """
        self.costs = list()

    def maybe(self, imp):
        """ Probability, based on fun.
            Bad improvement values are positive. We need maybe
            parameter between 0 and 1, thus -imp """
        return maybe(np.exp(-imp / self.fun))

    def update_fun(self, reset=False):
        """ Update fun """
        if reset or self.fun < self.min_fun:
            self.fun = self.max_fun
            if not self.got_best:
                self.no_best_resets += 1
            else:
                self.no_best_resets = 0
                self.got_best = False
            if self.no_best_resets > self.patience and self.best_cost < 6e5:
                self.days = self.best_days.copy()
                self.reset_occupancy()
        else:
            self.fun *= self.fun_mult

    def imps_and_moves(self, candidate_fams, candidate_days=None):
        """ Collect imps and moves for selected
            candidate families between candidate days """
        imps_cnt, imps, candidate_moves = 0, list(), list()
        for i, fam_id in enumerate(candidate_fams):
            fam_choices = self.choices[fam_id]
            feed_dict = dict(
                fam_id=fam_id,
                fam_size=self.fam_sizes[fam_id],
                dec_day=self.days[fam_id]
            )
            if isinstance(candidate_days, type(None)):
                candidate_days = fam_choices
            for day in candidate_days:
                feed_dict["inc_day"] = day
                mv = self.Move(**feed_dict)
                imp = self.improvement(mv)
                imps_cnt -=- 1
                imps.append(imp)
                candidate_moves.append(mv)
                if imps_cnt > self.max_imps:
                    break
            if imps_cnt > self.max_imps:
                break
        return imps, candidate_moves

    def select(self, imps, random=False):
        """ Select moves by imps """
        if random:
            ix = np.random.randint(len(imps))
            imp = imps[ix]
            if imp < 0 or self.maybe(imp):
                return ix
        ix = np.argmin(imps)
        return ix

    def select_fam(self, candidate_fams, candidate_days=None):
        """ Select family to move """
        imps, candidate_moves =             self.imps_and_moves(candidate_fams, candidate_days)
        ix = self.select(imps)
        mv = candidate_moves[ix]
        return mv

    def select_from_candidate_days(self, day, candidate_days, increase):
        """ Select family with provided assigned day
            and candidate days """
        if increase:
            constraints = (
                np.isin(self.days, candidate_days),
                np.any(self.choices == day, axis=1),
            )
        else:
            constraints = (
                self.days == day,
                np.any(np.isin(self.choices, candidate_days), axis=1),
            )
        candidate_fams = self.select_fams(constraints)
        if not len(candidate_fams):
            return None

        if increase:
            mv = self.select_fam(candidate_fams, [day, ])
        else:
            mv = self.select_fam(candidate_fams, candidate_days)
        return mv

    def select_inc_day(self, day):
        """ Select family at assigned `day`,
            that will be moved to other day """
        constraints = [self.days == day, ]
        candidate_fams = self.select_fams(constraints)
        mv = self.select_fam(candidate_fams)
        return mv

    def select_dec_day(self, day):
        """ Select family, that can be moved to `day` """
        constraints = [np.any(self.choices == day, axis=1), ]
        candidate_fams = self.select_fams(constraints)
        mv = self.select_fam(candidate_fams, [day,])
        return mv

    def find_moves(self):
        """ Find next moves and output it in same format as
            self.ride input parameters """
        pass

    def ride(self, moves):
        """ Move family to selected day, check if it improves cost function
            and save if best """
        for mv in moves:
            if self.check_imp:
                imp = self.improvement(mv)
                if imp >= 0 and not self.maybe(imp): return

            self.move(mv)
            if self.is_feasible() and self.cost < self.best_cost:
                self.got_best = True
                self.best_cost = self.cost
                self.best_days = self.days.copy()
        self.costs.append(self.cost)

    def search(self, n_iter=100):
        """ Random (with little priorities) local search """
        for _ in range(n_iter):
            self.step += 1
            self.update_fun()
            moves = self.find_moves()
            if not moves: continue
            self.ride(moves)


# I've observed Prancer the Reindeer's search and realized, that it's a RandomInfeasibleReindeer. Mostly, it goes where it pleases and it doesn't really care about feasibility. It can only distinguish between feasible and infeasible places. Once it loses feasibility, it rides in any abyss of infeasible solutions and explores thousands of irrelevant places.
# 
# RandomInfeasibleReindeer's code is below.

# In[ ]:


class RandomInfeasibleReindeer(Reindeer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.feasible = False
        self.check_imp = True

    def select_dec_day(self, thresh_occ=None):
        """ Select any day with occupancy above threshold """
        thresh_occ = thresh_occ or self.min_occupancy

        possible_days = np.argwhere(np.where(self.occupancy > thresh_occ,
                                     self.acc_penalties, 0)).flatten()
        if len(possible_days) == 0: return 0

        dec_day = np.random.choice(possible_days)
        return dec_day

    def select_fam(self, dec_day):
        """ Select random family among families with assigned day """
        fam_id = np.random.choice(np.argwhere(self.days == dec_day).flatten())
        return fam_id, self.fam_sizes[fam_id]

    def select_inc_day(self, fam_id, fam_size, dec_day):
        """ Find the day that minimizes cost function
            for the selected family """
        # Filter days with low occupancy in separate list
        feed_dict = dict(
            fam_id=fam_id,
            fam_size=fam_size,
            dec_day=dec_day,
        )
        little_occ_days, big_occ_days = list(), list()

        imps = np.zeros(shape=(self.n_choices, ))
        for c, inc_day in enumerate(self.choices[fam_id]):
            feed_dict["inc_day"] = inc_day
            imps[c] = self.improvement(self.Move(**feed_dict))
            if self.occupancy[inc_day] < self.min_occupancy:
                little_occ_days.append(c)
            else:
                big_occ_days.append(c)

        # Select day among days with low occupancy first
        got_solution = False
        if len(little_occ_days):
            ix_min_little = np.argmin(imps[little_occ_days])
            if not imps[ix_min_little] == 6e6:
                got_solution = True
                ix_min = little_occ_days[ix_min_little]

        # Select day among others if choice from previous block is bad
        if (not got_solution or imps[ix_min_little] > 0) and len(big_occ_days):
            ix_min_big = np.argmin(imps[big_occ_days])
            ix_min_big = big_occ_days[ix_min_big]
            if imps[ix_min_big] == 6e6:
                return None
            if not got_solution or ix_min_big < ix_min:
                ix_min = ix_min_big
            got_solution = True

        if not got_solution:
            return None

        return self.choices[fam_id, ix_min]

    def find_moves(self):
        """ Find next family to move and its destination day """
        dec_day = self.select_dec_day(self.max_occupancy)
        if dec_day == 0:
            dec_day = self.select_dec_day(self.min_occupancy)
        fam_id, fam_size = self.select_fam(dec_day)
        inc_day = self.select_inc_day(fam_id, fam_size, dec_day)
        if inc_day:
            mv = self.Move(fam_id, fam_size, dec_day, inc_day)
            return mv,
        else:
            return None


# Luckily, there are FeasibleReindeers, who can ride only through feasible tunnels, and explore only feasible places. 
# 
# Here's the plan: 
#  1. I let random Prancer jump for a while and find decent place (I think $600k is good enough for feasible starting point).
#  2. Ask Donner to continue search from that place.
# 
# In the beginning we can appear anywhere in the valley. We don't know where feasible tunnels are, so I'll let Prancer to have lots of Fun and let it jump everywhere it wishes.
# 
# 100k steps with 3 restarts and Fun level between 20k-500k will likely find any decent feasible place.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'params = dict(\n    choices   = choices,\n    fam_sizes = fam_sizes,\n    \n    min_fun   = 2e4,\n    max_fun   = 5e5,\n    fun_mult  = .98,\n    patience  = 30,\n)\n\n# Run Prancer 3 times and get any feasible solution\n# lower than $600k, if all costs are higher, the take the last solution\nfor i in range(3):\n    Prancer = RandomInfeasibleReindeer(**params)\n    Prancer.search(n_iter=int(1e5))\n    if Prancer.best_cost < 6e5:\n        best_prancer_place = Prancer.best_days\n        break\n\nif Prancer.best_cost >= 6e5:\n    best_prancer_place = Prancer.best_days\n    \nassert Prancer.is_feasible(best_prancer_place), "Not feasible solution"\nprint(f"Cost at solution, found by Prancer: {Prancer.best_cost:.0f}")\n\nplt.plot(range(len(Prancer.costs)), Prancer.costs)\nplt.show()')


# Once we have any feasible place, it is given to FeasibleReindeer as a feasible starting point. The problem with FeasibleReindeers is that they can only explore feasible places, but it's not always obvious, how to reach the next feasible place. Most of the caves are shattered and it's hard to see where they lead.
# 
# As I wrote earlier, Sleigh is good at predicting moves, but right now the feasible slide module is broken. Good news is that Santa's Helpers are already fixing that.
# 
# Fortunately, we've found Donner the Reindeer, who can explore feasible places on it's own. However, it consumes a lot of energy, thus it's slow. At every feasible location it creates multiple copies of itself and observes, which feasible places copies find. Then, it greedily selects the best move or sometimes for Fun. Maximum number of such copies can be set with `fam_batch_size` and `days_batch_size` hyperparameters.
# 
# FeasibleReindeer description is in the following block.

# In[ ]:


def feasible_tunnel(reindeer, mv):
    """ Move family from dec_day to inc_day. If not feasible,
        continue moving until solution gets feasible """
    low_occ, high_occ = list(), list()

    cost = reindeer.cost
    moves = list()
    while True:
        reindeer.move(mv)
        moves.append(mv)

        if reindeer.occupancy[mv.dec_day] < reindeer.min_occupancy:
            low_occ.append(mv.dec_day)
        if reindeer.occupancy[mv.inc_day] > reindeer.max_occupancy:
            high_occ.append(mv.inc_day)

        mv =  None
        low_cnt, high_cnt = len(low_occ), len(high_occ)
        if low_cnt > high_cnt:
            day = np.random.choice(low_occ)
            increase = True
            if len(high_occ):
                mv = reindeer.select_from_candidate_days(
                    day, high_occ, increase=increase)
        elif high_occ and low_cnt <= high_cnt:
            day = np.random.choice(high_occ)
            increase = False
            if len(low_occ):
                mv = reindeer.select_from_candidate_days(
                    day, low_occ, increase=increase)
        elif not low_cnt and not high_cnt:
            # feasible output
            imp = reindeer.cost - cost
            return imp, moves
        else:
            print("WRONG logic in feasible_tunnel")

        if isinstance(mv, type(None)):
            mv = reindeer.select_dec_day(day) if increase                 else reindeer.select_inc_day(day)

        if mv.dec_day in high_occ:
            high_occ.remove(mv.dec_day)
        if mv.inc_day in low_occ:
            low_occ.remove(mv.inc_day)


class FeasibleReindeer (Reindeer):

    def __init__(self, fam_batch_size, max_imps, drop_prob, *args, **kwargs):
        super().__init__(feasible=True, *args, **kwargs)

        self.fam_batch_size = fam_batch_size
        self.max_imps = max_imps
        self.drop_prob = drop_prob

    def families_batch(self):
        """ Select batch of families """
        return np.random.choice(self.n_fams,
            size=self.fam_batch_size, replace=False)

    def worst_days_list(self):
        """ Select list of days """
        acc_penalties =             np.where(np.random.rand(self.n_days+1) < self.drop_prob,
                     self.acc_penalties, 0)

        return np.argsort(-acc_penalties)[:self.days_batch_size]

    def find_moves(self):
        """ Find moves only in feasible space
            1. Get random batch of families
            2. Find their best moves, that doesn't break feasibility
            3. Output the best one """
        cost = self.cost
        candidate_fams = self.families_batch()

        imps, batch_moves = list(), list()
        for fam_id in candidate_fams:
            feed_dict = dict(
                fam_id=fam_id,
                fam_size=self.fam_sizes[fam_id],
                dec_day=self.days[fam_id],
            )
            fam_imps = np.zeros(shape=(self.n_choices, ))
            fam_moves = list()
            for i, inc_day in enumerate(self.choices[fam_id]):
                feed_dict["inc_day"] = inc_day
                reindeer = self.copy() # sorry for that
                imp, moves = feasible_tunnel(reindeer, self.Move(**feed_dict))
                fam_imps[i] = imp
                fam_moves.append(moves)
            ix = self.select(fam_imps)
            imps.append(fam_imps[ix])
            batch_moves.append(fam_moves[ix])

        ix = self.select(imps)
        return batch_moves[ix]


# Since Donner is truly heavy, I want it to find places not worse than the current one. That's why I don't let it having Fun during the search and set max Fun level to 1.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'donner_params = dict(\n        choices=choices,\n        fam_sizes=fam_sizes,\n        days = best_prancer_place,\n\n        min_fun=.005,\n        max_fun=1,\n        fun_mult=.98,\n\n        drop_prob=.5,\n        fam_batch_size=5,\n        days_batch_size=3,\n        max_imps = 10,\n        patience = 30,\n    )\nDonner = FeasibleReindeer(**donner_params)\nprint(f"Initial Donner cost: {Donner.cost:.0f}")\n\n# That\'s truly slow\nDonner.search(20)\nprint(f"Current best cost: {Donner.best_cost:.0f}")\nassert Donner.is_feasible(), "Not feasible"\n\nplt.plot(range(len(Donner.costs)), Donner.costs)\nplt.show()')


# I've let Donner to jump all over the valley during my workday, and Donner has found the cost of about $130k. It could go deeper, but gosh it's slow. I don't want to exploit it like that, I'd better wait for the fixed Sleigh's feasible slide module.

# ---
