

import csv
import Pandas as pd
import datetime
from random import *
import itertools


// how to count 

Customers_LowCategory = 10
Customers_MediumCategory = 10
Customers_HighCategory = 10

numCoupon_lowCategory = 2
numCoupon_mediumCategory = 2 
numCoupon_highCategory = 2 



#random binarystring of size n and k '1's
def kbits(n, k):
    
    for bits in itertools.combinations(range(n), k):
        s = ['0'] * n
        for bit in bits:
            s[bit] = '1'
        
    return s


transactions_data = pd.read_csv('Desktop/transaction_df.csv', engine ='python')

transactions_data.user_id


Low = kbits(Customers_LowCategory, numCoupon_lowCategory) 
for i in range(1,Customers_LowCategory):
 	CouponAllocation[i] = Low[i] 
# max coupon = numCoupon_lowCategory 


Medium = kbits(Customers_MediumCategory, numCoupon_MediumCategory) 
for i in range(1,Customers_MediumCategory): 
 	CouponAllocation[i] = Medium[i]



Medium = kbits(Customers_HighCategory, numCoupon_HighCategory) 
for i in range(1,Customers_HighCategory): 
 	CouponAllocation[i] = High[i]


transactions_data['BoolCoupon_Alloted'] = CouponAllocation 







































