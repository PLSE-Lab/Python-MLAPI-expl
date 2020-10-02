# -*- coding: utf-8 -*-
"""
Based on : Richard
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from pylab import *
import seaborn as sns
# Any results you write to the current directory are saved as output.
muni_data = pd.read_csv('../input/municipality_indicators.csv')
fire_data = pd.read_csv('../input/school_fire_cases_1998_2014.csv')

print(muni_data['period'].describe())

print(muni_data[['municipality_name','period']])
