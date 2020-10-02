import pandas as pd

events = pd.read_csv("../input/CAERS_ASCII_2004_2017Q2.csv")
outcomes = []

for _, reactions in events['AEC_One Row Outcomes'].str.split(",").iteritems():
    outcomes += [l.strip().title() for l in reactions]


import matplotlib.pyplot as plt

pd.Series(outcomes).value_counts().plot.bar(fontsize=8)
fig = plt.gcf()
fig.subplots_adjust(bottom=0.55)
plt.savefig("output.png")
