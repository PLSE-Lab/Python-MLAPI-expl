# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # visulaization libraries

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
digimon_list_data = pd.read_csv("../input/DigiDB_digimonlist.csv")
# Any results you write to the current directory are saved as output.
# Describe the data digimon_list_data.describe()

digimon_list_data_columns = digimon_list_data['Lv 50 HP']
digimon_list_data_columns.hist()
plt.title("Level 50 HP")
plt.ylabel('Levels')
plt.xlabel('Health')
