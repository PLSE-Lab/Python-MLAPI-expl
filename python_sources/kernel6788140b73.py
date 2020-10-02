# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

x=[1,2,51,34,72,90,7,9]
print(x)
y={1,2,51,34,72,90,7,9}
print(y)
z=(1,2,51,34,72,90,7,9)
print(z)
k={"Poedeljak":"1. radni dan",
  "Utorak":"2. radni dan",
  "Sreda":"3. radni dan",
  "Cetvrtak":"4. radni dan",
  "Petak":"5. radni dan",
  "Subota":"1. dan vikenda",
  "Nedelja":"2. dan vikenda"}
print(k)