#!/usr/bin/env python
# coding: utf-8

# The goal of this notebook is to identify the optimal strategy for playing the simplified blackjack game described in [booleans and conditionals tutorial](https://www.kaggle.com/colinmorris/booleans-and-conditionals) from the [introductory python course](https://www.kaggle.com/learn/python).
# 
# To achieve the result we'll accomplish the following steps:
# 1. Generate all possible hands in the game.
# 2. Compute probabilities for the hands.
# 3. Use the probabilities computed above to identify the odds of winning for the player and dealer in specific scenarios.
# Validate the formulas by selecting simple game strategies and comparing the theoretical results computed for the strategies with actual game simulations.
# 4. Identify the best strategy to play the game.

# **0. Game initialization**
# 
# It would be hard to test our ideas if we try to simulate the original game from the start. For example it would be practically impossible to test that the function that generates card hands works correctly on the original game. Therefore we need to be able to test the hand generation on simpler games (with fewer cards). The global variables below and the initialization function is used to make the games customizable.

# In[ ]:


import pytest

from decimal import Decimal, getcontext
from collections import defaultdict, OrderedDict

PRECISION = Decimal(1 ** - (getcontext().prec - 2))

ACE = None
ACE_LOW = None
ACE_DIFF = None
CARDS = None
P_ACE_LOW = None
SCORE_MAX = None
SCORE_DEALER_STOP = None


def INIT_GAME(cards, score_max=None, ace=None, ace_low=None, score_dealer_stop=None):
    assert len(cards) > 1, 'Can\'t play game with 1 card only'
    global ACE, ACE_LOW, ACE_DIFF, CARDS, P_ACE_LOW, SCORE_MAX, SCORE_DEALER_STOP
    CARDS = cards
    c = sorted(cards.keys())
    ACE = ace or c[-1]
    ACE_LOW = ace_low or 1
    ACE_DIFF = ACE - ACE_LOW
    P_ACE_LOW = cards[ACE]
    SCORE_MAX = score_max or c[-1] + c[-2]
    SCORE_DEALER_STOP = score_dealer_stop or round(ACE * 1.55)


# **1. Generate all possible hands**
# 
# A valid game hand consists of cards that sum up to at most `SCORE_MAX`. We should also take into account the conversion of `ACE` card into `ACE_LOW` if the score exceeds `SCORE_MAX`. To test the validity of the function we will check whether we are able to generate all hands for simpler games.

# In[ ]:


def generate_hands(score_max, hand=None, hand_sum=0):
    hand = hand or []
    for c in CARDS:
        hsum = hand_sum + c
        if hsum > score_max:
            if hsum - ACE_DIFF <= score_max:
                if c == ACE:
                    c = ACE_LOW
                    hsum -= ACE_DIFF
                    h = list(hand)
                else:
                    try:
                        i_ace = hand.index(ACE)
                        h = list(hand)
                        h[i_ace] = ACE_LOW
                        hsum -= ACE_DIFF
                    except ValueError:
                        continue
            else:
                continue
        else:
            h = list(hand)
        h.append(c)
        yield h, hsum
        if hsum < score_max:
            for h, hsum in generate_hands(score_max, h, hsum):
                yield h, hsum


def test_generate_hands(hands):
    n_hands = 0
    for h, hsum in generate_hands(SCORE_MAX):
        th = tuple(h)
        assert hsum <= SCORE_MAX
        assert th in hands, th
        hands.remove(th)
        n_hands += 1
    assert not hands, 'Some valid hands where not generated: %s' % hands
    return n_hands

# Test case 1: card deck with 2 cards only, maximal_score = 5
p = Decimal(1) / Decimal(2)
cards = {2: p, 3: p}
INIT_GAME(cards, score_max=5)
all_hands = {
    (2,), (3,), 
    (2,2), (2,3), (3,1), (3,2), 
    (2,2,1), (3,1,1), (1,1,2), 
    (1,1,2,1)
}

n_hands = test_generate_hands(all_hands)

print(
    'Total possible hands for configuration (cards=%s, blackjack_score=%d, ace_card=%d): %d' % (
        list(CARDS), SCORE_MAX, ACE, n_hands
    )
)

# Test case 2: card deck with 3 cards, maximal_score = 7

p = Decimal(1) / Decimal(3)
cards = {2: p, 3: p, 4: p}
INIT_GAME(cards, score_max=7)
all_hands = {
    (2,), (3,), (4,), 
    (2,2), (2,3), (2,4), (3,2), (3,3), (3,4), (4,2), (4,3), (4,1),
    (1,1,3), (1,2,2), (1,2,3), 
    (2,1,2), (2,1,3), (2,2,1), (2,2,2), (2,2,3), (2,3,1), (2,3,2), (2,4,1), 
    (3,2,1), (3,2,2), (3,3,1), 
    (4,1,1), (4,1,2), (4,2,1),
    (1,1,1,2), (1,1,1,3), (1,1,3,1), (1,1,3,2), (1,2,2,1), (1,2,2,2), (1,2,3,1), 
    (2,1,2,1), (2,1,2,2), (2,1,3,1), (2,2,1,1), (2,2,1,2), (2,2,2,1), (2,3,1,1), 
    (3,2,1,1), 
    (4,1,1,1),
    (1,1,1,2,1), (1,1,1,2,2), (1,1,1,3,1), (1,1,3,1,1), (1,2,2,1,1), 
    (2,1,2,1,1), (2,2,1,1,1),
    (1,1,1,2,1,1)
}

INIT_GAME(cards)
n_hands = test_generate_hands(all_hands)
print(
    'Total possible hands for configuration (cards=%s, blackjack_score=%d, ace_card=%d): %d' % (
        list(CARDS), SCORE_MAX, ACE, n_hands
    )
)


# Now that we've successfully tested the hand generation logic, let's compute the actual hands possible in the original blackjack game.

# In[ ]:


p = Decimal(1)/Decimal(13)
p10 = Decimal(4)/Decimal(13)
cards = {2:p, 3:p, 4:p, 5:p, 6:p, 7:p, 8:p, 9:p, 10:p10, 11:p}
INIT_GAME(cards)

assert 21 == SCORE_MAX
assert 17 == SCORE_DEALER_STOP
assert 11 == ACE
assert 1 == ACE_LOW

GAME_HANDS = tuple((tuple(h), hsum) for h, hsum in generate_hands(SCORE_MAX))

print(
    'Total possible hands for configuration (cards=%s, blackjack_score=%d, ace_card=%d): %d' % (
        list(CARDS), SCORE_MAX, ACE, len(GAME_HANDS)
    )
)


# In[ ]:


# Some tests to check if hands were generated correctly

assert ((2, 2), 4) in GAME_HANDS
assert ((11, 10), 21) in GAME_HANDS
assert ((11, 8, 1), 20) in GAME_HANDS
assert ((2, 2, 2, 2, 2, 2, 2, 2, 2, 2), 20) in GAME_HANDS
assert ((2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1), 21) in GAME_HANDS
assert ((1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2), 21) not in GAME_HANDS
assert ((11, 2, 2, 2, 2, 2), 21) in GAME_HANDS
assert ((7, 7, 7), 21) in GAME_HANDS
assert ((11, 1), 12) in GAME_HANDS
assert ((1, 5, 3, 3, 2), 14) in GAME_HANDS
assert ((10, 10, 1), 21) in GAME_HANDS
assert ((8, 10, 1, 2), 21) in GAME_HANDS

print('Tests passed: generate_hands()')


# **2. Compute probabilities for the hands**
# 

# In[ ]:


def prob_hand(hand):
    result = Decimal(1)
    for c in hand:
        result *= CARDS.get(c, P_ACE_LOW)
    return result

assert Decimal(4) / Decimal(13**2) == pytest.approx(prob_hand([10, 11]), abs=PRECISION)
assert Decimal(4) / Decimal(13**3) == pytest.approx(prob_hand([9, 10, 2]), abs=PRECISION)
assert Decimal(1) / Decimal(13**3) == pytest.approx(prob_hand([1, 6, 9]), abs=PRECISION)
assert Decimal(1) / Decimal(13**10) == pytest.approx(prob_hand([2] * 10), abs=PRECISION)

tp = defaultdict(Decimal)
hand_count = defaultdict(int)

for h, hsum in GAME_HANDS:
    lh = len(h)
    tp[lh] += prob_hand(h)
    hand_count[lh] += 1

for n_cards in tp:
    if n_cards <= 2:
        assert tp[n_cards] == pytest.approx(1, abs=PRECISION)
    else:
        assert tp[n_cards] < 1
    print(
        'Total hands with %d card(s) = %d with total probability = %s' % (
            n_cards, hand_count[n_cards], tp[n_cards]
        )
    )

print('Tests passed: prob_hand()')


# **3. Player Odds of Winning**
# 
# The player's choice at any point in the game is whether he should stay with the current hand or should get another card with the goal of improving the total without busting. Therefore we need to compute the odds of winning when the player stops collecting cards vs the odds of winning when he is dealt another card. In all cases the probability is affected by the dealer's score/card.

# We need to compute 2 kinds of probabilities of winning:
# 1. Player stays with his current hand. In this case the player wins only if his score is greater than the dealer's score. Therefore the probability of winning can be expressed by the following formula:
# 
# $$ p\_win\_stay(player\_score, dealer\_card) = 1 - \sum_{dealer\_score=\max\{17, player\_score\}}^{21}{prob\_dealer(dealer\_card, dealer\_score)}$$
# 
# Dealer score can never be smaller than 17 (`SCORE_DEALER_STOP`)
# 2. Player asks for another card. In this case the probability can be expressed as the sum of probabilities of winning with the new hand total times the probability of receiving each possible card. Note that the new total score could be adjusted if the the player has a high ace and the new total score is greater than `SCORE_MAX`.
# 
# $$ p\_win\_hit(player\_score, dealer\_card, has\_ace) = \sum_{card=2}^{11}{P[card] * p\_win\_stay(player\_new\_score, dealer\_card)}$$
# 
# $$ 
# player\_new\_score = 
# \begin{cases} 
# player\_score + card,& \text{if } player\_score + card \leq 21 \\
# player\_score - 10 + card,& \text{if } player\_score + card \gt 21 \land has\_ace
# \end{cases}
# $$
# 
# $$
# p\_win\_stay(player\_new\_score, dealer\_card) = 0,\text{ if } player\_new\_score \gt 21 \text{ (player busts) }
# $$
# 
# The action plan therefore is:
# 1. Compute `prob_dealer()`
# 2. Compute `p_win_stay()`
# 3. Compute `p_hit_stay()`

# ** 3.1. Dealer odds of obtaining a score given initial card **
# 
# To figure out the odds of the dealer obtaining a score given an initial card we need to be able to separate the list of valid dealer hands from the set of total hands we generated and stored in `GAME_HANDS`. Then we could filter the list of hands and add the hand probabilities for each `(dealer_card, dealer_total)` combination.

# In[ ]:


def is_dealer_hand(hand, hsum=None):
    if not hsum:
        hsum = sum(hand)
    if hsum < SCORE_DEALER_STOP or hsum > SCORE_MAX:
        return False
    s = 0
    low_ace_count = 0
    for c in hand[:-1]:
        if c == ACE_LOW:
            low_ace_count += 1
            c = ACE
        s += c
        if SCORE_DEALER_STOP <= s <= SCORE_MAX:
            return False
    return s - ACE_DIFF * low_ace_count < SCORE_DEALER_STOP

assert not is_dealer_hand([3, 4]), 'Sum is too low'
assert not is_dealer_hand([3, 10, 2]), 'Sum is too low'
assert is_dealer_hand([3, 4, 10])
assert not is_dealer_hand([3, 4, 10, 2]), 'Should have stopped collecting cards after 10'
assert not is_dealer_hand([3] * 7), 'Should have stopped collecting cards after first 6 threes'
assert is_dealer_hand([3] * 6)
assert is_dealer_hand([11, 5, 5])
assert not is_dealer_hand([1, 5, 5, 5, 5]), 'Should have stopped collecting cards after second 5'
assert is_dealer_hand([1, 5, 10, 2]), 'Initial ace should have been converted to low'
assert not is_dealer_hand([1, 1, 10, 3, 3, 2]), 'Should have stopped after second 3'
assert not is_dealer_hand([1, 1, 4, 10]), 'Sum is too low'
assert is_dealer_hand([1, 1, 4, 10, 1])
assert is_dealer_hand([1, 1, 4, 10, 2])
assert not is_dealer_hand([1, 1, 4, 10, 2, 3]), 'Should have stopped after 2'
assert not is_dealer_hand([11, 1, 1, 1, 4, 2]), 'Should have stopped after 4'
assert not is_dealer_hand([11, 1, 6, 2]), 'Should have stopped after 6'
assert is_dealer_hand([1, 1, 6, 10])

print('Tests passed: is_dealer_hand()')


# In[ ]:


# P_INIT is the probability of obtaining a 2-card hand with a given score 
# (probability of the player's initial hand score)

P_INIT = defaultdict(Decimal)
P = defaultdict(Decimal)
PROB_DEALER = defaultdict(Decimal)

for hand, hsum in GAME_HANDS:
    p = prob_hand(hand)
    if len(hand) >= 2:
        P[hsum] += p
        if len(hand) == 2:
            P_INIT[hsum] += p
    if is_dealer_hand(hand):
        first_card = hand[0]
        if first_card == ACE_LOW:
            first_card = ACE
        PROB_DEALER[first_card, hsum] += p / CARDS[first_card]

for k in sorted(PROB_DEALER):
    print(k, PROB_DEALER[k])


# ** 3.2. Player odds of winning if stays with current hand **

# In[ ]:


def prob_win_stay(player_score, dealer_card):
    return 1 - sum(
        PROB_DEALER[dealer_card, score]  
        for score in range(max(SCORE_DEALER_STOP, player_score), SCORE_MAX + 1)
    )

P_WIN_STAY = defaultdict(Decimal)

possible_player_scores = list(range(4, SCORE_MAX + 1))

for player_score in possible_player_scores:
    for dealer_card in CARDS:
        P_WIN_STAY[player_score, dealer_card] = prob_win_stay(player_score, dealer_card)
        assert 0 < P_WIN_STAY[player_score, dealer_card] < 1

# Tests
for dealer_card in CARDS:
    p_prev = 0
    for player_score in possible_player_scores:
        assert P_WIN_STAY[player_score, dealer_card] >= p_prev,             'Odds of winning should not decrease with increasing player score'
        p_prev = P_WIN_STAY[player_score, dealer_card]

# Test against actual game data
def p_stay(player_score, prob=P):
    return prob[player_score] * sum(
        dealer_card_prob * P_WIN_STAY[player_score, dealer_card]
        for dealer_card, dealer_card_prob in CARDS.items()
    )

# TODO: use 2 stddev instead of 5 percentage range
pct = Decimal(5) / Decimal(100)
# actual_win_rate taken from actual game simulations, iter=1e6
for ps, actual_win_rate in [
    (4,  Decimal(0.001724)),
    (5,  Decimal(0.003379)),
    (6,  Decimal(0.005087)),
    (7,  Decimal(0.006960)),
    (8,  Decimal(0.009181)),
    (9,  Decimal(0.011334)),
    (10, Decimal(0.013465)),
    (11, Decimal(0.016264)),
    (12, Decimal(0.036160)),
    (13, Decimal(0.038669)),
    (14, Decimal(0.039808)),
    (15, Decimal(0.041533)),
    (16, Decimal(0.043268)),
    (17, Decimal(0.044566)),
    (18, Decimal(0.070093)),
    (19, Decimal(0.096658)),
    (20, Decimal(0.159679)),
    (21, Decimal(0.156376)),
]:
    expected_win_rate = pytest.approx(p_stay(ps), rel=pct)
    assert expected_win_rate == actual_win_rate, (ps, expected_win_rate, actual_win_rate)

# Test simple strategy: no hits

stay_strategy_score = sum(p_stay(ps, prob=P_INIT) for ps in possible_player_scores)

assert pytest.approx(stay_strategy_score, rel=Decimal(1)/Decimal(100)) == Decimal(0.38)

print('Win rate if stay with initial hand: %s' % stay_strategy_score)
    
print('Tests passed: p_stay()')
        
for k in sorted(P_WIN_STAY):
    print(k, P_WIN_STAY[k])
        


# ** 3.3. Player odds of winning if hits with current hand **

# In[ ]:


def prob_win_hit(player_score, dealer_card, has_ace):
    result = Decimal(0)
    for card in CARDS:
        total_score = player_score + card
        if total_score <= SCORE_MAX:
            result += CARDS[card] * P_WIN_STAY[total_score, dealer_card]
        elif has_ace:
            total_score -= ACE - ACE_LOW
            if total_score <= SCORE_MAX:
                result += CARDS[card] * P_WIN_STAY[total_score, dealer_card]
    return result  

P_WIN_HIT_NO_ACE = defaultdict(Decimal)
P_WIN_HIT_ACE = defaultdict(Decimal)

for player_score in possible_player_scores:
    for dealer_card in CARDS:
        P_WIN_HIT_NO_ACE[player_score, dealer_card] = prob_win_hit(player_score, dealer_card, False)
        P_WIN_HIT_ACE[player_score, dealer_card] = prob_win_hit(player_score, dealer_card, True)
        assert 0 <= P_WIN_HIT_NO_ACE[player_score, dealer_card] <= P_WIN_HIT_ACE[player_score, dealer_card] < 1

assert 0 == P_WIN_HIT_NO_ACE[SCORE_MAX, 2]
assert P_WIN_STAY[SCORE_MAX, 2] > P_WIN_HIT_ACE[SCORE_MAX, 2]
assert 0 == P_WIN_HIT_NO_ACE[SCORE_MAX - 1, 2]
assert P_WIN_STAY[SCORE_MAX - 1, 2] > P_WIN_HIT_ACE[SCORE_MAX - 1, 2]
assert pytest.approx(CARDS[2] * P_WIN_STAY[SCORE_MAX, 2]) == P_WIN_HIT_NO_ACE[SCORE_MAX - 2, 2]
assert pytest.approx(
    CARDS[2] * P_WIN_STAY[SCORE_MAX - 1, 2] + CARDS[3] * P_WIN_STAY[SCORE_MAX, 2]
) == P_WIN_HIT_NO_ACE[SCORE_MAX - 3, 2]

assert P_WIN_HIT_NO_ACE[2, 2] == P_WIN_HIT_ACE[2, 2] == P_WIN_STAY[2, 2]
assert P_WIN_HIT_NO_ACE[10, 2] > P_WIN_STAY[10, 2]
assert P_WIN_HIT_NO_ACE[10, 2] == P_WIN_HIT_ACE[10, 2]

print('Tests passed: prob_win_hit()')


# **4. Best strategy**
# 
# The idea is that for any given (player_score, dealer_card) combination we compute the max(P_WIN_STAY, P_WIN_HIT) where P_WIN_HIT is either P_WIN_HIT_NO_ACE or P_WIN_HIT_ACE.

# In[ ]:


import numpy as np
import pandas as pd

result_no_ace = np.empty((18, 10), dtype=str)
result_ace = np.empty((18, 10), dtype=str)

for player_score in possible_player_scores:
    for dealer_card in CARDS:
        p_stay = P_WIN_STAY[player_score, dealer_card]
        p_hit_no_ace = P_WIN_HIT_NO_ACE[player_score, dealer_card]
        if round(p_hit_no_ace - p_stay, ndigits=12) >= 0:
            result_no_ace[player_score - 4, dealer_card - 2] = 'H'
        else:
            result_no_ace[player_score - 4, dealer_card - 2] = 'S'
            
        p_hit_ace = P_WIN_HIT_ACE[player_score, dealer_card]
        if round(p_hit_ace - p_stay, ndigits=12) >= 0:
            result_ace[player_score - 4, dealer_card - 2] = 'H'
        else:
            result_ace[player_score - 4, dealer_card - 2] = 'S'

def print_decision_table(result):
    d = pd.DataFrame(
        result, 
        columns=['D-%d' % d for d in CARDS],
        index=['P-%d' % p for p in possible_player_scores]
    )
    d = d.style.applymap(lambda val: 'background-color: red' if val == 'S' else 'background-color: green')
    display(d)
    
d1 = print_decision_table(result_no_ace)
d2 = print_decision_table(result_ace)

