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

k1 = pd.read_csv("../input/kpis_1998_2003.csv")
print(k1.shape)
k2 = pd.read_csv("../input/kpis_2004_2008.csv")
print(k2.shape)
k3 = pd.read_csv("../input/kpis_2009_2011.csv")
print(k3.shape)
k4 = pd.read_csv("../input/kpis_2012_2013.csv")
print(k4.shape)
k5 = pd.read_csv("../input/municipality_indicators.csv")
print(k5.shape)
