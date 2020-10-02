import pandas as pd
import numpy as np

types = {'Semana': np.uint8,'Cliente_ID': np.uint16, 'Producto_ID': np.uint16,
'Demanda_uni_equil': np.uint8}
types2 = {'Cliente_ID': np.uint16, 'Producto_ID': np.uint16}

dftrain = pd.read_csv('../input/train.csv', usecols=types.keys(), dtype=types)
dftest = pd.read_csv('../input/test.csv',skiprows=0)

print('Loading finished')
demand_prod_cus = (dftrain.groupby(['Cliente_ID', 'Producto_ID']))['Demanda_uni_equil'].median().to_dict()
print('creating dictionary finished')
dftest['predict']=""

numrows =0

for index, row in dftest.iterrows():
    if(numrows%100==0):
        print(numrows)
    numrows+=1
    week = int(row[1])
    ClientId = int(row[5])
    ProductID = int(row[6])
    n=0
    if (ClientId,ProductID) in demand_prod_cus:
        n = demand_prod_cus[(ClientId,ProductID)]
    
    dftest.loc[index, 'predict'] = n
        
dftest.to_csv("1.csv",index = False)  