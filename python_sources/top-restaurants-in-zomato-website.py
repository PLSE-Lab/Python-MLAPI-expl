# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import sys
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
zomato = pd.read_csv('../input/zomato.csv')
# Renaming column name of cost for two to a smaller one #
zomato.rename(columns={'approx_cost(for two people)':'cost_for_two'},inplace=True)
zomato.info()
# Any results you write to the current directory are saved as output.
##Data Cleaning##
zomato.cost_for_two.fillna('0',inplace=True)
zomato.cost_for_two = zomato.cost_for_two.str.replace(',', '')
zomato.cost_for_two = zomato.cost_for_two.astype('int64')
zomato['rate'] = [''.join(c.split()) for c in zomato['rate'].astype(str)] ###Removes unnecessary spaces###
zomato['rate'] = zomato.rate.str.replace('/5','')
zomato['rate'] = zomato['rate'].str.replace('-','0')
zomato['rate'] = zomato['rate'].str.replace('nan','0')
zomato['rate'] = zomato['rate'].str.replace('NEW','0')

zomato['listed_in(city)'].value_counts() ###provides area wise restaurants list###
#BTM have highest number of restaurants. Koramangala 7th block have second highest number of restaurants##
#Least number of restaurants are in New BEL Road, Banashankari, Rajaji Nagar#
zomato.groupby('listed_in(type)')['name'].count() ###Provides count of type of restaurants###
#Most of the restaurants are of type Dineout and Delivery#

zomato.groupby('rate')['cost_for_two'].value_counts().tail(60)
zomato.rate.max()
#4.9 is the maximum rating given by customers#
zomato.groupby([zomato.rate=='4.9'])['name'].tail()
#Gives list of restaurants with high rating#
zomato.groupby([zomato.rate=='4.9'])['location','name'].tail()
#From the above data, it looks like most of high rated restaurants are costly ones#

