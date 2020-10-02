# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 21:46:13 2019

@author: Rajesh
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir('../input'))
n = 1 
fl='../input/Reviews.csv'
reviews=pd.read_csv(fl,skiprows=lambda i: i % n != 0,usecols=['ProductId','Score','Time'])
reviews['Time'] = pd.to_datetime(reviews['Time'], unit='s')
reviews['special_day']= reviews.Time.astype(object).astype(str)
reviews.dtypes

# Any results you write to the current directory are saved as output.
# 1. Which are the top 10 Product IDs that get reviewed the most?
ratings=reviews[['ProductId','Score']]
ans1=ratings.groupby('ProductId').count().sort_values(by='Score',ascending=False).reset_index().head(10)
ans1.plot (x='ProductId',y='Score',kind='bar',title='Top 10 Reviewed by count ')

# 2. Which are the top 10 most favorably reviewed Product IDs?
multi_agg=ratings.groupby('ProductId').agg(['mean','count']).add_suffix('_total').reset_index()
multi_agg.columns=['product','avg_review','no_of_review']
multi_agg
multi_agg['weight']=multi_agg.avg_review*1000+multi_agg.avg_review*multi_agg.no_of_review # if rating is same use count as sub key
ans2=multi_agg.sort_values(by='weight',ascending=False).head(10)

# 3. How may product reviews are generated on a daily, monthly and yearly basis? 
daily_avg_review=reviews['Time'].size/reviews['Time'].unique().size
yyyymm=reviews.special_day.str.slice(0,7)
yyyy=reviews.special_day.str.slice(0,4)
monthly_avg_review=yyyymm.size/yyyymm.unique().size
yearly_avg_review=yyyy.size/yyyy.unique().size
data = [['Daily',daily_avg_review],['Monthly',monthly_avg_review],['Yearly',yearly_avg_review]]
ans3 = pd.DataFrame(data,columns=['Frequency','Average'])
ans3.plot(x='Frequency',y='Average',kind='bar',title='Periodic Average Reviews')


# 4. Product sales- festival mapping. By festival mapping we are trying to analyze the top 3 Product IDs that get reviewed the most for each of the major festivals in the USA.
CHRISTMAS='-12-25'
VALENTINE='-02-14'
g=reviews[((reviews.special_day.str.contains(CHRISTMAS) )| (reviews.special_day.str.contains(VALENTINE)))].groupby('ProductId')
ans4=g.count().add_suffix('_total').reset_index().sort_values(by='special_day_total',ascending=False).head(9)
ans4.plot(x='ProductId',y='special_day_total',kind='bar',title='Top 9 Reviews count on special US holidays')

# 5. Visualization of results observed. in the below case question2 ##
ans2['avg']=ans2['avg_review']*1000  # scaling up  for better visibility 
ans2['cnt']=ans2['no_of_review']*10   # scaling up  for better visibility 
ans2.plot (x='product',y=['avg','cnt','weight'],kind='bar',title='Top 10 Weighted Reviews')




