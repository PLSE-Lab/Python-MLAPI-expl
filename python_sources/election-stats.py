# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

county_facts = pd.read_csv('../input/county_facts.csv')
# the first line of the file
#print(county_facts['INC110213'][1])
# Any results you write to the current directory are saved as output.

coef = []
for i in range(len(county_facts['INC910213'])):
    coef.append(float(county_facts['INC910213'][i] / county_facts['HSG495213'][i]))
state_coef = dict(zip(range(len(coef)), coef))
CA_CO_coef = {}
for i in range(len(state_coef)):
    if(county_facts['state_abbreviation'][i] == 'CA'):
        CA_CO_coef[i] = state_coef[i]
#del state_coef['United States']
sorted = sorted(CA_CO_coef, key=CA_CO_coef.__getitem__, reverse = True)
for i in range(10):
    print(county_facts['area_name'][sorted[i]] , ', ', county_facts['state_abbreviation'][sorted[i]], ' ', state_coef[sorted[i]])
    