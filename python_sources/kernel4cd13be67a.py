# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
Orders_or=pd.read_csv('/kaggle/input/ungrd-rd2-auo/orders.csv',dtype='unicode')
Devices_or=pd.read_csv('/kaggle/input/ungrd-rd2-auo/devices.csv',dtype='unicode')
CreditCard_or=pd.read_csv('/kaggle/input/ungrd-rd2-auo/credit_cards.csv',dtype='unicode')
BankAcc_or=pd.read_csv('/kaggle/input/ungrd-rd2-auo/bank_accounts.csv',dtype='unicode')

from collections import defaultdict
Devices=defaultdict(list)
CreditCard=defaultdict(list)
BankAcc=defaultdict(list)
Buyer_ID=defaultdict(list)
Seller_ID=defaultdict(list)

for userid, device in zip(list(Devices_or['userid']),list(Devices_or['device'])):
    Devices[userid].append(device)
for userid, credit_card in zip(list(CreditCard_or['userid']),list(CreditCard_or['credit_card'])):
    CreditCard[userid].append(credit_card)
for userid, bank_account in zip(list(BankAcc_or['userid']),list(BankAcc_or['bank_account'])):
    BankAcc[userid].append(bank_account)
for orderid, buyer_userid in zip(list(Orders_or['orderid']),list(Orders_or['buyer_userid'])):
    Buyer_ID[orderid].append(buyer_userid)
for orderid, seller_userid in zip(list(Orders_or['orderid']),list(Orders_or['seller_userid'])):
    Seller_ID[orderid].append(seller_userid)

def invert_dict(d):
    inv_temp = defaultdict(list)
    values = []
    for key, values in d.items():
        for value in values:
            inv_temp[value].append(key)
    return inv_temp
inv_Devices=invert_dict(Devices)
inv_CreditCard=invert_dict(CreditCard)
inv_BankAcc=invert_dict(BankAcc)


result=[]
for order in Buyer_ID.keys():
    a = 0
    Buyer_related_id = []
    Seller_related_id = []
    for temp in Devices[Buyer_ID[order][0]]:
        Buyer_related_id.extend(inv_Devices[temp])
    for temp in CreditCard[Buyer_ID[order][0]]:
        Buyer_related_id.extend(inv_CreditCard[temp])
    for temp in BankAcc[Buyer_ID[order][0]]:
        Buyer_related_id.extend(inv_BankAcc[temp])
    for temp in Devices[Seller_ID[order][0]]:
        Seller_related_id.extend(inv_Devices[temp])
    for temp in CreditCard[Seller_ID[order][0]]:
        Seller_related_id.extend(inv_CreditCard[temp])
    for temp in BankAcc[Seller_ID[order][0]]:
        Seller_related_id.extend(inv_BankAcc[temp])
    for temp in Buyer_related_id:
        if temp in Seller_related_id:
            a+=1
    if a>0:
        result.append(int(1))
    else:
        result.append(int(0))

result_csv = pd.DataFrame({'orderid':list(Orders_or['orderid']),'is_fraud':result})
result_csv.to_csv('Submission_2.csv',index=False)
print(sum(result))