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
def myFunction(num1, num2, num3):
    result = num1 + num2 + num3
    print(result)
myFunction(4, 4, 4)

import this
    
print(2 + 2 - 1) # My comment rn
print("yes")

soda = ["Fanta", "Coca-Cola", "Sprite"]
soda.append("Bang's Root Beer")
energy_drinks = ["100 Plus", "Monster Energy", "Redbull"]
beverages = soda + energy_drinks
print(beverages)

num = 4
if num == 4:
    print("yes")
else:
    print("no")