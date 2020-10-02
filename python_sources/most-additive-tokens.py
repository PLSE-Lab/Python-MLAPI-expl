import pandas as pd
import re
import itertools

additives = pd.read_csv("../input/indirect-additives.csv", 
                        encoding='latin1')

# Tokenize.
def tokenize(s):
    tokens = re.split(' |, |-|\n', s)
    return tokens

tokens = list(itertools.chain(*additives['Substance'].map(tokenize).values))
tokens = list(filter(lambda t: len(t) > 0, tokens))
token_counts = pd.Series(tokens).value_counts()

# Filter some things out.
token_counts.index = token_counts.index.map(lambda s: s.replace(")", " ").replace("(", " ").strip())
token_counts = token_counts[token_counts.index.map(lambda s: not s.isdigit()).values]
token_counts = token_counts[token_counts.index.map(lambda s: not s in ['N', 'AND']).values]

# Plot.
import matplotlib.pyplot as plt
token_counts.head(20).plot.bar()
fig = plt.gcf()
fig.subplots_adjust(bottom=0.3)
fig.savefig("output.png")