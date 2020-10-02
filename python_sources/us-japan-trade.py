# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

year_data_path = '../input/year_latest.csv'
country_data_path = '../input/country_eng.csv'
categories_data_path = '../input/hs2_eng.csv'

year_df = pd.read_csv(year_data_path)
country_df = pd.read_csv(country_data_path)
comb_df = pd.merge(year_df, country_df, on=['Country'])
usa_df = comb_df[comb_df['Country_name'] == 'United_States_of_America']
cat = pd.read_csv(categories_data_path)
usa_hs2 = usa_df[['Year', 'VY', 'hs2']].groupby(['hs2','Year'], as_index=False)
df = usa_hs2.aggregate(np.sum)
df = pd.merge(df, cat, on=['hs2'])
goods = df[df['VY'] >  0.15 * 1e10]
names = np.unique(goods['hs2_name'].values)

for name in names:
    d = df[df['hs2_name'] == name]
    plt.plot(d['Year'], d['VY'], label=name)

plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.show()

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.