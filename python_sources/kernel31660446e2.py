import pandas as pd
import numpy as np
import sys

# FOR KAGGLE:
sys.argv = ["program.py", "1099", "../input/train.csv", "solution.csv"]


# DAYS - total number of days
# WEEKS - total number of WEEKS (= DAYS / 7)

# the algorithm uses day numeration from 0,
# the input/output data - from 1


# takes list of day-numbers,
# returns binary matrix of visits:
# v[i][j] shows whether there was visit
# on day j of week i
def visit_matrix(days):
    v = np.zeros(shape=(WEEKS,7))
    for day in days:
        w = day // 7
        d = day % 7
        v[w][d] = 1
    return v

# calculates and returns array of weights,
# used for calculating probabilities
def weights():
    w = np.empty(shape=(WEEKS,))
    for i in range(WEEKS):
        w[i] = (i + 1) / WEEKS
    s = sum(w)
    for i in range(WEEKS):
        w[i] /= s
    return w

# takes matrix of visits,
# returns array of 7 probabilities
# for each day
def probabilities(v):
    w = weights()
    p = np.zeros(shape=(7,))
    for j in range(7):
        for i in range(WEEKS):
            p[j] += w[i] * v[i][j]
    return p

# probability that the first visit
# will be on day j 
# (using day probabilities p)
def day_pro(j, p):
    pp = p[j]
    for r in range(j):
        pp *= 1 - p[r]
    return pp

# takes list of day-numbers for a user,
# returns the most probable day of the next visit
def predict(days):
    v = visit_matrix(days)
    p = probabilities(v)
    return max(range(7), key = lambda j: day_pro(j, p))




# ------------------------------------------------------------

if len(sys.argv) != 4:
    print("Usage: python3 program.py 1099 train.csv solution.csv")
    sys.exit(1)

DAYS = int(sys.argv[1])
WEEKS = DAYS // 7

data = pd.read_csv(sys.argv[2])
answers = []
for v in data.values:
    days = [int(w)-1 for w in v[1].split()]
    answers.append(predict(days)+1)
pd.DataFrame({'id': range(1, len(answers)+1), 'nextvisit': answers}).to_csv(sys.argv[3], index=False)

