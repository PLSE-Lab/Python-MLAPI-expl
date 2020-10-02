# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.
idades = [22,65,45,55,21,22,34,42,41,4,99,101,120,122,130,111,115,80,75,54,44,64,13,18,48]
 
ids = [x for x in range(len(idades))]
 
plt.bar(ids,idades)
plt.show()
 
  

#df.iloc[df.groupby('Product ID')['Sales'].agg(pd.Series.idxmax)]