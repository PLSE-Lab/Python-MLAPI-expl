# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 



# Any results you write to the current directory are saved as output.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pkmn = pd.read_csv('../input/Pokemon.csv')

pkmn.head()

