# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# Any results you write to the current directory are saved as output


crashes = pd.read_csv("../input/Airplane_Crashes_and_Fatalities_Since_1908.csv")
#print(crashes.loc[:,'Summary'])

op = crashes.loc[:, "Operator"]
ab = crashes.loc[:, "Aboard"]
ft = crashes.loc[:, "Fatalities"]
reldict = {"operator":op, "aboard":ab, "fatalities":ft}

rel = pd.DataFrame(reldict)

cld = rel.groupby(rel.operator).sum()

ct = rel[cld[operator]].value_counts()

rto = cld.fatalities/cld.aboard
cld["percentage dead"] = rto
print(ct)

#hist = cld.hist(