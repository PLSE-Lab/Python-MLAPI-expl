# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

# Any results you write to the current directory are saved as output.

data = pd.read_csv('../input/titanic_data.csv')
data_c = data[data.Age >= 0]
data_c['AgeBracket']=0
data_c.AgeBracket=np.array([min(int(age/10),3) for age in data_c.Age])
lookup=(data_c.groupby(['Sex', 'Pclass', 'Age']).Survived.mean())
print(lookup)