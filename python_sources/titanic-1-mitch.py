import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Graphing

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
#Loading dataset in to a data frame
df = pd.read_csv("../input/train.csv") 
for dataset in df:
    dataset['FamilySize'] = df['SibSp'] + df['Parch'] + 1
print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())

fig = plt.figure(figsize=(18,6))

df.Survived.value_counts(normalize=True).plot(kind="bar", alpha=1)

plt.show()

#Histogram of...

#Summary stats and first 10 rows
df.describe()
df.head(5)