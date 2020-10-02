import pandas as pd

pdx_pumas = [1314, 1301, 1302, 1303, 1305]

husb = pd.read_csv("../input/pums/ss13husb.csv")
# print(husb.head())
# print(husb[['PUMA']].head())

pdx_housing = husb.loc[husb['PUMA'].isin(pdx_pumas)]

print(pdx_housing.head())


