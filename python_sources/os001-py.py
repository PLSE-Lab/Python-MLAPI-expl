#Obama salary plot

#Faiz ul haque

#Version 1





#importing numpy,pandas and matplotlib



import numpy as np

import pandas as pd

import matplotlib.pyplot as pp





#function to remove $ sign

def rd(a):

    if(a[0]=='$'):

       a=a[1:]

    return a

    

#obama salary o.csv change the path

a=pd.read_csv('../input/obama_staff_salaries.csv')



#count the values

b=a['salary'].value_counts()

print(b)



#get the salary

sal=a['salary']



#apply rd to sal elements removing $

sal1=map(rd,sal)

#sort and remove duplicates

l=list(set(sal1))

#convert to int for plotting

ll=list(map(int,map(float,l)))

pp.bar(ll,b,1500)

pp.show()