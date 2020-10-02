import pandas as pd

words = pd.read_csv("../input/unigram_freq.csv").set_index("word")

import matplotlib.pyplot as plt

words.loc[['he', 'she', 'they']].plot.bar()
fig = plt.gcf()
fig.subplots_adjust(bottom=0.1)
plt.savefig("output.png")