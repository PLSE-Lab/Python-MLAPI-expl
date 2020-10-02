import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv("../input/train_ver2.csv", usecols=['fecha_dato', 'ind_nuevo'])
df = df[df['fecha_dato'] == "2016-05-28"].drop("fecha_dato", axis = 1)
unique_ids = pd.Series(df['ind_nuevo'].unique())
print(unique_ids)
print(pd.value_counts(df.values.ravel()))