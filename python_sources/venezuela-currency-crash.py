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

import matplotlib.pyplot as plt

erates = pd.read_csv("../input/exchange_rates.csv")
print(erates['SPOT EXCHANGE RATE - VENEZUELA '])
#vmean=erates['SPOT EXCHANGE RATE - VENEZUELA '].astype(np.float32).mean()
erates['SPOT EXCHANGE RATE - VENEZUELA '] = erates['SPOT EXCHANGE RATE - VENEZUELA '].fillna(0)
print(erates.columns)
erates_rates = erates[['Series Description', 'SPOT EXCHANGE RATE - VENEZUELA ']][5:]
#erates_rates['SPOT EXCHANGE RATE - VENEZUELA '].replace(to_replace="\$([0-9,\.]+).*",value=0,regex=True,inplace=True)
erates_rates['SPOT EXCHANGE RATE - VENEZUELA ']=pd.to_numeric(erates_rates['SPOT EXCHANGE RATE - VENEZUELA '].astype(str).str.replace(',',''),errors='coerce')
erates_mean =erates_rates['SPOT EXCHANGE RATE - VENEZUELA '].mean()
erates_rates.loc[erates_rates['SPOT EXCHANGE RATE - VENEZUELA '] == 0, 'SPOT EXCHANGE RATE - VENEZUELA ' ] = erates_mean
erates_rates['Series Description'] = pd.to_datetime(erates_rates['Series Description'],format='%Y-%m-%d')

plt.plot(erates_rates['Series Description'],erates_rates['SPOT EXCHANGE RATE - VENEZUELA '])
plt.show()