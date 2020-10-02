# this script was inspired by other scoring scripts already posted

import numpy as np
import pandas as pd
from sympy import sieve

# prep
cities = pd.read_csv('../input/cities.csv', index_col=['CityId'])
pnums = list(sieve.primerange(0, cities.shape[0]))

# function
def score_it(path):
    path_df = cities.reindex(path).reset_index()
    path_df['step'] = np.sqrt((path_df.X - path_df.X.shift())**2 + (path_df.Y - path_df.Y.shift())**2)
    path_df['step_adj'] = np.where((path_df.index) % 10 != 0, path_df.step, path_df.step + 
                            path_df.step*0.1*(~path_df.CityId.shift().isin(pnums)))
    return path_df.step_adj.sum()

# usage: path is array_like
sub = pd.read_csv('../input/sample_submission.csv')
print(score_it(sub.Path.values))