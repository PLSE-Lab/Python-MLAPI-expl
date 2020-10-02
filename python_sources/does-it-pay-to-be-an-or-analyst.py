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

cols = ['OCCP', 'AGEP']

dfa = pd.DataFrame(columns=cols)
chunka = pd.read_csv('../input/pums//ss13pusa.csv', chunksize=1000, usecols=cols)

while True:
    try:
        sub_df = chunka.get_chunk()
        #sub_df = sub_df.dropna()
        dfa = pd.concat([sub_df, dfa])
    except:
        break
    
dfa = dfa[dfa['OCCP']==1220]
print(dfa)
# dfb = pd.DataFrame(columns=cols)
# chunkb = pd.read_csv('../input/pums//ss13pusb.csv', chunksize=1000, usecols=cols)
# while True:
#     try:
#         sub_df = chunkb.get_chunk()
#         #sub_df = sub_df.dropna()
#         dfb = pd.concat([sub_df, dfb])
#     except:
#         break
    
# data = pd.concat([dfa, dfb])
