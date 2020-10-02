#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# The [*Benford's law*](https://en.wikipedia.org/wiki/Benford%27s_law) of first digits states (simplified) that in real life datasets, the first digits of the numbers contained in this dataset follow the frequency $f(d) = log_{10}(1 + \frac{1}{d}), d=1..9$. This is used for detecting fraud, especially electoral fraud. This law is admitted in US and Swiss (canton Geneva) jurisdictions.
# 
# # Result
# 
# We consider here the numbers in the columns 'Remain' and 'Leave', and see how they follow the Benford's law:

# In[ ]:


import pandas as pd
import numpy as np
from math import log10
import matplotlib.pyplot as plt

referendum = pd.read_csv('../input/referendum.csv')

THRESHOLD = 1800000

# Benford law
def first_digit_occurencies(col, threshold=0):
    filtered = referendum[(referendum['Electorate'] < threshold )]
    digits = np.unique(filtered[col].astype(str).apply(lambda x: int(x[0])), return_counts=True)
    values = []
    idx = 0
    for i in range(10):
        if not i in digits[0]:
            values.append(0)
        else:
            values.append(digits[1][idx])
            idx+=1
    print(values)    
    return np.array(values[1:])

cols = ['Leave', 'Remain']
fdf = [0] * 9
for col in cols:
    fdf_col = first_digit_occurencies(col, THRESHOLD)
    fdf = [fdf[i] + fdf_col[i] for i in range(9)]

fdf = [fdf[i] / sum(fdf) for i in range(len(fdf))]
theoric_fdf = [log10(1 + 1 / (i + 1)) for i in range(9)]
signs = np.arange(1, 10)
ind = np.arange(9) #auxiliary variable so that the 'theoretical repartition' line doesn't get shifted
comp = {'sign': signs, 'Actual frequency': fdf, 'Benford frequency': theoric_fdf, 'ind': ind}
comp = pd.DataFrame(comp)

#plot
_, ax = plt.subplots()
comp[['ind', 'Benford frequency']].plot(x='ind', linestyle='-', marker='o', ax=ax)
comp[['sign', 'Actual frequency']].plot(x='sign', kind='bar', ax=ax)
plt.title("Actual vs. Benford's First Digit Frequency")
plt.show()


# # Conclusion
# 
# The actual first digit frequency doesn't really fit the Benford's law. I personally trust the British electoral organisation, but:
# 
# * Here we applied this method on only 764 digits, and the Benford' distribution is an asymptotic value
# 
# * The Benford's law is not always reliable, especially if there are constraints on the numbers

# In[ ]:




