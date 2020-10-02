# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.

#==============Top 50 Powerful Superheros==============
## Which superhero has most of the powers that is the powerful superhero

powers = pd.read_csv('../input/super_hero_powers.csv',index_col =0)

no_of_powers = (powers == True).astype(int).sum(axis=1)
no_of_powers_1 = no_of_powers.sort_values(ascending=False).head(n=50)

import matplotlib.pyplot as plt
import seaborn as sns

no_of_powers_1.plot.bar(
    figsize=(20, 10),
    color='mediumvioletred',
    fontsize=12,
    title='Top 50 Powerful Superheros', 
)
sns.despine(bottom=True,left=True)

#================Top 20 Rare Powers==================
##Thoes Powers which has less no. of Superhero

rare_powers = (powers==True).astype(int).sum(axis=0)
rare_powers_1 = rare_powers.sort_values(ascending =True).head(n=20)

rare_powers_1.plot.bar(
    figsize=(12, 6),
    color='mediumvioletred',
    fontsize=16,
    title='Top 20 Rare Powers', 
)
sns.despine(bottom=True,left=True)

#==============Top 20 ommon Power==================
##Thoes Power which has most of the Superhero

most_powers = rare_powers.sort_values(ascending=False).head(n=20)

most_powers.plot.bar(
    figsize=(12, 6),
    color='mediumvioletred',
    fontsize=16,
    title='Top 20 Common Powers', 
)
sns.despine(bottom=True,left=True)

#==================Top 10 Comics======================
##which comic has number of superheros character

info = pd.read_csv('../input/heroes_information.csv',index_col=0)

bes_com = info['Publisher'].value_counts().head(n=10)
bes_com.plot.bar(
    figsize=(8, 6),
    color='mediumvioletred',
    fontsize=16,
    title='Top 10 comics', 
)
sns.despine(bottom=True,left=True)

#=================Gender Info=====================

info['Gender'].replace(
    to_replace = ['-'],
    value = 'other',
    inplace = True,
    )
gen = info['Gender'].value_counts()

gen.plot.bar(
    figsize=(4, 6),
    color='mediumvioletred',
    fontsize= 12,
    title='Gender Information', 
)
sns.despine(bottom=True,left=True)