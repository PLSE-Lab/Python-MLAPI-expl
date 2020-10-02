# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# read the data (San Francisco building permits)
sfPermit = pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv")

# %% [code]
sfPermit.sample(5)

# %% [code]
# Calculate total number of cells in dataframe

totalCells = np.product(sfPermit.shape)

# Count number of missing values per column

missingCount = sfPermit.isnull().sum()

# Calculate total number of missing values

totalMissing = missingCount.sum()

print (totalCells)
print( missingCount)
print( totalMissing)

# %% [code]
missingCount[['Street Number Suffix', 'Zipcode']]

# %% [code]
sfPermit.head()

# %% [code]
sfPermitCleanCols = sfPermit.dropna(axis=1)
sfPermitCleanCols.head()

# %% [code]
print('columns in original datasets: %d \n' % sfPermit.shape[1])
print('columns with na dropped: %d' % sfPermitCleanCols.shape[1])

# %% [code]
imputeSfPermit = sfPermit.fillna(method='ffill', axis=0).fillna("0")

imputeSfPermit.head()

# %% [code]
