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
df= pd.read_csv("../input/FoodFacts.csv", low_memory=True)
# Creating a copy of the original dataset as sub. All experiments will be done on sub

df = df[['countries_en', 'fat_100g', 'carbohydrates_100g', 'proteins_100g']]
df = df[np.isfinite(df['fat_100g']) & np.isfinite(df['carbohydrates_100g']) & np.isfinite(df['proteins_100g'])]
df['Total_Calories'] = (df['fat_100g']*9)+(df['carbohydrates_100g']*4)+(df['proteins_100g']*3.9)
df['Fat_Calories'] = (df['fat_100g']*9)
df['Carb_Calories'] = (df['carbohydrates_100g']*4)
df['Protein_Calories'] = (df['proteins_100g']*3.9)
grp = df.groupby(['countries_en'])['Fat_Calories', 'Carb_Calories', 'Protein_Calories'].mean()
print(grp)#['Fat_Calories', 'Carb_Calories', 'Protein_Calories'].mean())
