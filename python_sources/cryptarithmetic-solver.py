#!/usr/bin/env python
# coding: utf-8

# # Cryptarithmetic Solver
# 
# > Verbal arithmetic, also known as alphametics, cryptarithmetic, cryptarithm or word addition, is a type of mathematical game consisting of a mathematical equation among unknown numbers, whose digits are represented by letters. The goal is to identify the value of each letter. The name can be extended to puzzles that use non-alphabetic symbols instead of letters.
# >
# > <img src='https://wikimedia.org/api/rest_v1/media/math/render/svg/60eeaf958fa73a6a989f00725cf7d4c3f516e929' style='text-align:left' alt="SEND + MORE = MONEY"/>
# > - https://en.wikipedia.org/wiki/Verbal_arithmetic
# 
# This is a general purpose solver that can handle addition, subtraction, multiplication, integer division and rasing to powers.
# 
# This implementation uses the z3 constraint satisfaction solver 
# - https://github.com/Z3Prover/z3/wiki

# In[ ]:


# TODO: get z3 installed in the default kaggle-docker image
get_ipython().system('pip3 install -q z3-solver')


# In[ ]:


import re
import time

from z3 import *


def cryptarithmetic(input: str, limit=None, unique=True):
    start_time  = time.perf_counter()
    solver      = Solver()
    token_words = re.findall(r'\b[a-zA-Z]\w*\b', input)  # words must start with a letter

    letters = { l: Int(l) for l in list("".join(token_words)) }
    words   = { w: Int(w) for w in list(token_words)          }

    # Constraint: convert letters to numbers
    for l,s in letters.items(): solver.add(0 <= s, s <= 9)

    # Constraint: letters must be unique (optional)
    if unique and len(letters) <= 10:
        solver.add(Distinct(*letters.values()))

    # Constraint: words must be unique
    solver.add(Distinct(*words.values()))

    # Constraint: first letter of words must not be zero
    for word in words.keys():
        solver.add( letters[word[0]] != 0 )

    # Constraint: convert words to decimal values
    for word, word_symbol in words.items():
        solver.add(word_symbol == Sum(*[
            letter_symbol * 10**index
            for index,letter_symbol in enumerate(reversed([
                letters[l] for l in list(word)
                ]))
            ]))

    # Constraint: problem definition as defined by input
    solver.add(eval(input, None, words))

    solutions = []
    print(input)
    while str(solver.check()) == 'sat':
        solutions.append({ str(s): solver.model()[s] for w,s in words.items() })
        print(solutions[-1])
        solver.add(Or(*[ s != solver.model()[s] for w,s in words.items() ]))
        if limit and len(solutions) >= limit: break

    run_time = round(time.perf_counter() - start_time, 1)
    print(f'== {len(solutions)} solutions found in {run_time}s ==\n')
    return solutions


# # Cryptarithmetic Addition

# In[ ]:


cryptarithmetic('XY - X == YX')
cryptarithmetic('TWO + TWO == FOUR')
cryptarithmetic('EIGHT - FOUR == FOUR', limit=4)
pass


# # Cryptarithmetic Multiplication

# In[ ]:


cryptarithmetic('X / Y == 2')  # Division by 2 is rounded
cryptarithmetic('ONE * TWO == THREE', limit=1)
cryptarithmetic("Y == A * X + B")
cryptarithmetic('( FOUR - TWO ) * ( NINE - ONE ) + TWO == EIGHTEEN', limit=1)
pass


# # Cryptarithmetic Powers

# In[ ]:


cryptarithmetic("A**2 + B**2 == C**2",  unique=False)
cryptarithmetic("A**2 - B**2 == C**2",  unique=False)
cryptarithmetic("A**2 * B**2 == CD**2", unique=False)
pass


# # Cryptarithmetic Challenges
# - https://www.reddit.com/r/dailyprogrammer/comments/7p5p2o/20180108_challenge_346_easy_cryptarithmetic_solver/

# In[ ]:


challenges = [
    "WHAT + WAS + THY == CAUSE",
    "HIS + HORSE + IS == SLAIN",
    "HERE + SHE == COMES",
    "FOR + LACK + OF == TREAD",
    "I + WILL + PAY + THE == THEFT",
]
for challenge in challenges:
    cryptarithmetic(challenge, limit=1)
pass


# In[ ]:



challenges = [
    " ".join([
        "TEN + HERONS + REST + NEAR + NORTH + SEA + SHORE + AS + TAN + TERNS + SOAR + TO + ENTER + THERE + AS + ",
        "HERONS + NEST + ON + STONES + AT + SHORE + THREE + STARS + ARE + SEEN + TERN + SNORES + ARE + NEAR == SEVVOTH",        
    ]),
    " ".join([
        "SO + MANY + MORE + MEN + SEEM + TO + SAY + THAT + THEY + MAY + SOON + TRY + TO + STAY + AT + HOME + ",
        "SO + AS + TO + SEE + OR + HEAR + THE + SAME + ONE + MAN + TRY + TO + MEET + THE + TEAM + ON + THE + ",
        "MOON + AS + HE + HAS + AT + THE + OTHER + TEN == TESTS",
    ]),
]
for challenge in challenges:
    cryptarithmetic(challenge, limit=1)
pass


# This last one takes too long to run on Kaggle, but runs in 7h on localhost

# In[ ]:


longest = " ".join([
    "THIS + A + FIRE + THEREFORE + FOR + ALL + HISTORIES + I + TELL + A + TALE + THAT + FALSIFIES + ",
    "ITS + TITLE + TIS + A + LIE + THE + TALE + OF + THE + LAST + FIRE + HORSES + LATE + AFTER + ",
    "THE + FIRST + FATHERS + FORESEE + THE + HORRORS + THE + LAST + FREE + TROLL + TERRIFIES + THE + ",
    "HORSES + OF + FIRE + THE + TROLL + RESTS + AT + THE + HOLE + OF + LOSSES + IT + IS + THERE + ",
    "THAT + SHE + STORES + ROLES + OF + LEATHERS + AFTER + SHE + SATISFIES + HER + HATE + OFF + ",
    "THOSE + FEARS + A + TASTE + RISES + AS + SHE + HEARS + THE + LEAST + FAR + HORSE + THOSE + ",
    "FAST + HORSES + THAT + FIRST + HEAR + THE + TROLL + FLEE + OFF + TO + THE + FOREST + THE + ",
    "HORSES + THAT + ALERTS + RAISE + THE + STARES + OF + THE + OTHERS + AS + THE + TROLL + ASSAILS + ",
    "AT + THE + TOTAL + SHIFT + HER + TEETH + TEAR + HOOF + OFF + TORSO + AS + THE + LAST + HORSE + ",
    "FORFEITS + ITS + LIFE + THE + FIRST + FATHERS + HEAR + OF + THE + HORRORS + THEIR + FEARS + ",
    "THAT + THE + FIRES + FOR + THEIR + FEASTS + ARREST + AS + THE + FIRST + FATHERS + RESETTLE + ",
    "THE + LAST + OF + THE + FIRE + HORSES + THE + LAST + TROLL + HARASSES + THE + FOREST + HEART + ",
    "FREE + AT + LAST + OF + THE + LAST + TROLL + ALL + OFFER + THEIR + FIRE + HEAT + TO + THE + ",
    "ASSISTERS + FAR + OFF + THE + TROLL + FASTS + ITS + LIFE + SHORTER + AS + STARS + RISE + THE + ", 
    "HORSES + REST + SAFE + AFTER + ALL + SHARE + HOT + FISH + AS + THEIR + AFFILIATES + TAILOR + ",
    "A + ROOFS + FOR + THEIR + SAFE == FORTRESSES"    
])
# cryptarithmetic(longest, limit=0)


# ```
# {
#   'THIS': 9874,
#   'A': 1,
#   'FIRE': 5730,
#   'THEREFORE': 980305630,
#   'FOR': 563,
#   'ALL': 122,
#   'HISTORIES': 874963704,
#   'I': 7,
#   'TELL': 9022,
#   'TALE': 9120,
#   'THAT': 9819,
#   'FALSIFIES': 512475704,
#   'ITS': 794,
#   'TITLE': 97920,
#   'TIS': 974,
#   'LIE': 270,
#   'THE': 980,
#   'OF': 65,
#   'LAST': 2149,
#   'HORSES': 863404,
#   'LATE': 2190,
#   'AFTER': 15903,
#   'FIRST': 57349,
#   'FATHERS': 5198034,
#   'FORESEE': 5630400,
#   'HORRORS': 8633634,
#   'FREE': 5300,
#   'TROLL': 93622,
#   'TERRIFIES': 903375704,
#   'RESTS': 30494,
#   'AT': 19,
#   'HOLE': 8620,
#   'LOSSES': 264404,
#   'IT': 79,
#   'IS': 74,
#   'THERE': 98030,
#   'SHE': 480,
#   'STORES': 496304,
#   'ROLES': 36204,
#   'LEATHERS': 20198034,
#   'SATISFIES': 419745704,
#   'HER': 803,
#   'HATE': 8190,
#   'OFF': 655,
#   'THOSE': 98640,
#   'FEARS': 50134,
#   'TASTE': 91490,
#   'RISES': 37404,
#   'AS': 14,
#   'HEARS': 80134,
#   'LEAST': 20149,
#   'FAR': 513,
#   'HORSE': 86340,
#   'FAST': 5149,
#   'HEAR': 8013,
#   'FLEE': 5200,
#   'TO': 96,
#   'FOREST': 563049,
#   'ALERTS': 120394,
#   'RAISE': 31740,
#   'STARES': 491304,
#   'OTHERS': 698034,
#   'ASSAILS': 1441724,
#   'TOTAL': 96912,
#   'SHIFT': 48759,
#   'TEETH': 90098,
#   'TEAR': 9013,
#   'HOOF': 8665,
#   'TORSO': 96346,
#   'FORFEITS': 56350794,
#   'LIFE': 2750,
#   'THEIR': 98073,
#   'FIRES': 57304,
#   'FEASTS': 501494,
#   'ARREST': 133049,
#   'RESETTLE': 30409920,
#   'HARASSES': 81314404,
#   'HEART': 80139,
#   'OFFER': 65503,
#   'HEAT': 8019,
#   'ASSISTERS': 144749034,
#   'FASTS': 51494,
#   'SHORTER': 4863903,
#   'STARS': 49134,
#   'RISE': 3740,
#   'REST': 3049,
#   'SAFE': 4150,
#   'SHARE': 48130,
#   'HOT': 869,
#   'FISH': 5748,
#   'AFFILIATES': 1557271904,
#   'TAILOR': 917263,
#   'ROOFS': 36654,
#   'FORTRESSES': 5639304404
# }
# == 1 solutions found in 26153.0s ==
# ```

# In[ ]:





# In[ ]:




