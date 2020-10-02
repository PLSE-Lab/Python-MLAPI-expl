# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy  as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
data = pd.read_csv('../input/transactions.csv')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#data.info
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~../input/transactions.csv~~~~~~~~~~~~~~~~~~~~~~~')
print('Data Frame of transaction data     : \n', data)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n')
print('--------------------Meta Information of Data---------------------------------')
print('Number of Rows in transactions.csv : ', data.shape[0])
print('Number of Cols in transactions.csv : ', data.shape[1])
print('Columns/Header in transactions.csv : ', data.columns.values)
data.rename(columns={'ProductID': 'PID', 'UserID':'UID', 'Quantity':'QTY', 'TransactionID':'TID', 'TransactionDate':'TDT'}, inplace='TRUE')
print('Renamed header names as            : ', data.columns.values)
print('-----------------------------------------------------------------------------\n\n')


print('--------------------Ordering Based on the columns--------------------------------')
print('data.sort_values(''TID'', ascending=False)\n ',data.sort_values('TID', ascending=False))
print('data.sort_values([''QTY'', ''TDT''], ascending=[True, False])\n', data.sort_values(['QTY', 'TDT'], ascending=[True, False]))
print('---------------------------------------------------------------------------------\n\n')

print('--------------------Ordering Based on the rows-----------------------------------')
print('data[[''PID'', ''QTY'', ''TDT'', ''TID'', ''UID'']]\n', data[['PID', 'QTY', 'TDT', 'TID', 'UID']])
print('data[pd.unique([''UID''] + transactions.columns.values.tolist()).tolist()]\n', data[pd.unique(['UID'] + data.columns.values.tolist()).tolist()])
print('---------------------------------------------------------------------------------\n\n')

print('--------------------DataFrame to Arrays------------------------------------------')
print('data.PID.values \n', data.PID.values)
print('data[[''PID'']].values[:, 0] \n', data[['PID']].values[:, 0])
print('---------------------------------------------------------------------------------\n\n')

print('-------------------------Accessing Rows------------------------------------------')
print('data.iloc[[0,2,5]] \n',data.iloc[[0,2,5]])
print('data.drop([0,2,5], axis=0) \n',data.drop([0,2,5], axis=0))
print('data[:3]\n', data[:3])
print('data.head(3)\n',data.head(3))
print('data[3:]\n',data[3:])
print('data.tail(-3)\n',data.tail(-3))
print('data.tail(2)\n',data.tail(2))
print('data.tail(-2)\n',data.tail(-2))
print('data[data.QTY > 1]\n',data[data.QTY > 1])
print('data[data.UID == 2]\n',data[data.UID == 2])
print('data[(data.QTY > 1) & (data.UID == 2)]\n',data[(data.QTY > 1) & (data.UID == 2)])
print('data[data.QTY + data.UID > 3]\n',data[data.QTY + data.UID > 3])
print('---------------------------------------------------------------------------------\n\n')


print('-------------------------Accessing Rows Based on other data -------------------------')
a = np.array([True, True, True, False, True, True, False, False, True, False])
print('a = \n', a)
print('data[a] = \n', data[a])
b = np.array([1, -3, 2, 2, 0, -4, -4, 0, 0, 2])
print('b = ', b)
print('data[b > 0] = \n', data[b > 0])
print('data[a | (b < 0)] \n', data[a | (b < 0)])
print('data[~a & (b >= 0)] \n', data[~a & (b >= 0)])
print('-------------------------------------------------------------------------------------\n\n')

print('-------------------------Accessing Columns ------------------------------------------')
print('data.iloc[:, [0, 2]] \n', data.iloc[:, [0, 2]])
print('data[[''TID'', ''TDT'']] \n', data[['TID', 'TDT']])
print('data.loc[data.TID > 5, [''TID'', ''TDT'']] \n',data.loc[data.TID > 5, ['TID', 'TDT']])
print('data.drop(["TID", "UID", "QTY"], axis=1) \n',data.drop(["TID", "UID", "QTY"], axis=1))
print('-------------------------------------------------------------------------------------\n\n')


print('-------------------------Manipulation of columns--------------------------------------------')
data['TDT'] = pd.to_datetime(data.TDT)
print('data after TDT Convert \n', data)
data['NC'] = data.UID + data.PID
print('data after adding a new column NC \n', data)
#print('data after  data.loc[data.TID % 2 == 0, ''NC''] = np.nan \n', data.loc[data.TID % 2 == 0 & data.NC == np.nan])
data['RowIdx'] = np.arange(data.shape[0])
print('data[''RowIdx''] = np.arange(data.shape[0]) \n', data)
data['QTYRNK'] = data.QTY.rank(method='average')
print('data[''QTYRNK''] = data.QTY.rank(method=''average'') \n', data)
data['QTYMIN'] = data.QTY.min()
data['QTYMAX'] = data.QTY.max()
print('data[''QTYMIN''] = data.QTY.min() and  data[''QTYMIN''] = data.QTY.max() \n', data)
data.drop('NC', axis=1, inplace=True)
print('data.drop(''NC'', axis=1, inplace=True), axis=1, inplace=True) \n', data)
data.drop(['QTYRNK', 'QTYMIN', 'QTYMAX'], axis=1, inplace=True)
print('transactions.drop([''QTYRNK'', ''QTYMIN'', ''QTYMAX''], axis=1, inplace=True) \n', data)
print('--------------------------------------------------------------------------------------------')

#print('------------------------COMMANDS FOR LEARNING -------------------')
#print('Accessing PID --> data.PID.values  : ', data.PID.values)
#print('Accessing PID --> data[["PID"]]    : \n', data[["PID"]])
#print('Accessing PID --> data[["PID","UID"]]    : \n', data[["PID","UID"]])
#print('Accessing PID --> data.iloc[1]    : \n', data.iloc[1])
#print('Accessing PID --> data.iloc[[1]]    : \n', data.iloc[[1]])
#print('-----------------------------------------------------------------')

print('-------------------------Aggregation/Grouping-----------------------------------------------')
print(data.groupby('UID').apply(lambda x: pd.Series(dict(
    Transactions=x.shape[0]
))).reset_index())

print(data.groupby('UID').apply(lambda x: pd.Series(dict(
    Transactions=x.shape[0],
    QuantityAvg=x.QTY.mean()
))).reset_index())
print('--------------------------------------------------------------------------------------------')

sessions     = pd.read_csv('../input/sessions.csv')
products     = pd.read_csv('../input/products.csv')
users = pd.read_csv('../input/users.csv')

# Convert date columns to Date type
users['Registered'] = pd.to_datetime(users.Registered)
users['Cancelled'] = pd.to_datetime(users.Cancelled)
print('--------------------------------------JOINS---------------------------------------------------')
users.rename(columns={'UserID':'UID'}, inplace='TRUE')
print('data.merge(users, how=''left'', on=''UID'') \n', data.merge(users, how='left', on='UID'))
print('data[~data[''UID''].isin(users[''UID''])] \n', data[~data['UID'].isin(users['UID'])])
print('data.merge(users, how=''inner'', on=''UID'')', data.merge(users, how='inner', on='UID'))
print('data.merge(users, how=''outer'', on=''UID'') \n', data.merge(users, how='outer', on='UID'))

print('----------------------------------------------------------------------------------------------')