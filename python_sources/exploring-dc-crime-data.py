# This Python 3 environment comes with many helpful analytics libraries installed

import numpy as np 
import pandas as pd


df = pd.read_csv('../input/crime_homicide_subset.csv', encoding='latin1', sep=',')
df.head(5)