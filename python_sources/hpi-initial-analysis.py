# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df_us = pd.read_csv("../input/HPI_master.csv",
                 parse_dates={'date_idx': [6,7]},
                 nrows=3080)
                 
df_us.shape
df_us.info()

df_us.isnull().any()

df_us.set_index("date_idx", inplace=True, drop=False)

print('Earliest date is:', df_us.date_idx.min())
print('Latest date is:', df_us.date_idx.max())

df_us.describe(include=['O'])

df_us.place_name.unique()

df_us.set_index('place_name', inplace=True, drop=False)
df_us_plot = df_us.loc['United States']

import bokeh.charts
import bokeh.charts.utils
import bokeh.io
import bokeh.models
import bokeh.palettes
import bokeh.plotting

bokeh.io.output_notebook()

p = bokeh.charts.Line(df_us_plot, x='date_idx', y='index_nsa', color='firebrick', 
                      title="Monthly Aggregate Home Price Values in the U.S.")

# Display it
#bokeh.io.show(p)
