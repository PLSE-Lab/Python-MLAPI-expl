import pandas as pd
import numpy as np
from scipy import stats
columns = ['ST', 'PUMA']
pd.read_csv('../input/pums/ss13husa.csv', chunksize=1000, usecols=columns)
print('Testing')