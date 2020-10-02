import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pl


###loading and initial shaping from redmelonnette
# read data
pop_a = '../input/pums//ss13husa.csv'
pop_b = '../input/pums//ss13husb.csv'

popa = pd.read_csv(pop_a, usecols = ['SERIALNO', 'REGION', 'FS'])
popb = pd.read_csv(pop_b, usecols = ['SERIALNO', 'REGION', 'FS'])

pop = pd.DataFrame(pd.concat([popa, popb], axis = 0))
pop= pop.dropna()

