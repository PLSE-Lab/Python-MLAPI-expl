"""Create a data set of 70 T-shirts of randomized brands, designs, colour, size and prices """

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import numpy.random as randn

n=70

tsh=pd.DataFrame({'Date ordered':pd.date_range('2016-05-05', periods=n), 'Order number':['SDH-YU%d' % r for r in range(n)],
                  'Brand': np.random.choice(['Kent','Yolo','Wang','BUCK','BLOT'],n),'Size':np.random.choice(['S','XS','M','L','XL','XXL'],n),
                  'colour':np.random.choice(['red','white','green','indigo','purple','black'],n),
                  'Design':np.random.choice(['Stripes','plain','spiral','diamond','checked'],n),
                  'Price':15+np.random.randn(n).round(2),
                  'Delivery status':np.random.choice(['order received','shipping','delivered','delayed','damaged'],n)})

# Print the T-shirt dataset

print(tsh)

#select brand

print(tsh['Brand'])

#select data frame with Kent 

print(tsh[tsh['Brand']=='Kent'])

#select data frame with Kent and size M

print(tsh[(tsh.Brand=='Kent')&(tsh.Size=='M')])

#Select head of data frame

print(tsh.head())

#select of data frame

print(tsh.tail())

#Descriptive statistics of the data frame

print(tsh.describe().round(2))

# aggregate sum and mean of the pirce in the data frame

print(tsh.Price.agg(['sum','mean']).round(2))

#Create a copy of the data frame

tsh1=tsh.copy()

#Fill in the 1st 24 rows of the new dataframe with not a number

tsh1.iloc[:24]=np.nan

#Print the new data frame

print(tsh1)

#Fill missing strings with 'missing' and missing prices with the mean price

print(tsh1[['Brand', 'Date ordered', 'Delivery status','Design','Order number','Size','colour']].fillna('missing'),
tsh1['Price'].fillna(tsh1['Price'].mean()).round(2))

#Group the first data frame by brand sorted by the sum of the prices

print(tsh.groupby(['Brand']).sum())

#Group the first data frame by size and colour sorted by the mean of the prices

print(tsh.groupby(['Size','colour']).mean())

#Group the first data frame by Brand, colour and delivery status sorted by the mean of the prices

print(tsh.groupby(['Brand','colour','Delivery status']).mean())

#Group by date ordered and design aggregate by the sum of the prices

grtsh=tsh.groupby(['Date ordered','Design'])

print(grtsh.aggregate(np.sum))

#Describe the statistics of the data frame grouped by design size and brand

grtsh=tsh.groupby(['Design','Size','Brand'])

print(grtsh.describe())

#Set delivery status as index

print(tsh.set_index('Delivery status'))

#Set Date ordered and order number as index

print(tsh.set_index(['Date ordered','Order number']))

#Create another dataframe and concatenate the first data frame with the third one

tsh2=pd.DataFrame({'Date ordered':pd.date_range('2017-05-05', periods=n), 'Order number':['SDH-YU%d' % r for r in range(n)],
                  'Brand': np.random.choice(['Kullihun','Molo','Wang','BUCK','BLOT'],n),'Size':np.random.choice(['S','XS','M','L','XL','XXL'],n),
                  'colour':np.random.choice(['red','white','green','yellow','purple','black'],n),
                  'Design':np.random.choice(['Stripes','plain','spiral','diamond','tartan'],n),
                  'Price':15+np.random.randn(n).round(2),
                  'Delivery status':np.random.choice(['order received','shipping','delivered','delayed','cancelled'],n)})


print(pd.concat([tsh,tsh2]))

#Merge 2 dataframes by brand

print(pd.merge(tsh,tsh2,on='Brand'))

#Merge 2 dataframes by price and size

print(pd.merge(tsh,tsh2,on=['Price','Size']))