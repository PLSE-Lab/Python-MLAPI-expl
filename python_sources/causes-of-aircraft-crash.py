import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('../input/3-Airplane_Crashes_Since_1908.txt')
comm = 0
carg = 0
mil = 0
priv = 0

for o, s in zip(data.Operator, data.Summary):
    text = (str(o) + ' | ' + str(s)).lower()
    if 'military' in text:
        mil += 1
    elif 'private' in text:
        priv += 1
    elif ('cargo' in text) or ('mail' in text):
        carg += 1
    else:
        comm += 1


sns.barplot(x=['commerical', 'cargo', 'military', 'private'], y=[comm, carg, mil, priv])
plt.show()
