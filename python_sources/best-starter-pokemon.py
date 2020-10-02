# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Any results you write to the current directory are saved as output.

df = pd.read_csv('../input/Pokemon.csv')
starters = []
types = pd.Series(['Grass', 'Fire', 'Water'])

for i in range(0, max(df['Generation'])):
    current_gen = df[df['Generation'] == i].head(15)
    current_gen = current_gen[current_gen['Type 1'].isin(types)]
    current_gen.drop(['Type 2', 'Generation', '#', 'Legendary', 'Total'], axis=1, inplace=True)
    current_gen = current_gen[~current_gen['Name'].str.contains(" ")]
    starters.append(current_gen)
    
for frame in starters:
    print(frame[~frame['Name'].str.contains(' ')])
    