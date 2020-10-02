#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import os
print(os.listdir("../input"))


# In[ ]:


# Read Data file
Department_Information=pd.read_csv('../input/Department_Information.csv')
Student_Performance_Data=pd.read_csv('../input/Student_Performance_Data.csv')
Student_Counceling_Information=pd.read_csv('../input/Student_Counceling_Information.csv')


# In[ ]:


Department_Information


# In[ ]:


# User defined function:
# 1) Lambda Fuction:
from datetime import datetime,timedelta
Department_Information['DOE_Year']=Department_Information.apply(lambda x : pd.to_datetime(x['DOE']).strftime('%Y') , axis=1)
Department_Information.head()


# In[ ]:


# 2) Def Function :
def DOE_Month(data):
    Month=pd.to_datetime(data['DOE']).strftime('%B')
    return Month


# In[ ]:


Department_Information['DOE_Month']=Department_Information.apply(DOE_Month , axis=1)
Department_Information.head()


# In[ ]:


# If else loop :
a=200
b=33
if b > a:
    print("b is greater than a")
else:
    print("b is not greater than a")


# In[ ]:


Student_Performance_Data.head()


# In[ ]:


def result(data):
    if data['Marks']>=30:
        return 'Pass'
    else:
        return 'Fail'


# In[ ]:


Student_Performance_Data['Results']=Student_Performance_Data.apply(result , axis=1)
Student_Performance_Data.head()


# In[ ]:


Student_Performance_Data.Results.value_counts()


# In[ ]:


# Nested If else loop:
def Grade(data):
    if data['Marks'] < 30:
        return 'Fail'
    elif data['Marks'] >= 30 and data['Marks'] < 45:
        return 'III'
    elif data['Marks'] >= 45 and data['Marks'] < 60:
        return 'II'
    elif data['Marks'] >= 60 and data['Marks'] < 75:
        return 'I'
    elif data['Marks'] >= 75 and data['Marks'] <=100:
        return 'A+'
    else:
        return 'Error'


# In[ ]:


Student_Performance_Data['Grade']=Student_Performance_Data.apply(Grade , axis=1)
Student_Performance_Data.head()


# In[ ]:


Student_Performance_Data.Grade.value_counts()


# In[ ]:


#For Loop :

for i in range(10):
    print(i)


# In[ ]:


Names=['Umesh' , 'Tabrez' , 'Paras' , 'Dhananjay' , 'Ritesh' , 'Raghu' ]
for x in Names:
    if x == 'Dhananjay':
        break
    print(x)


# In[ ]:


# While loop:
i=1
while i < 5:
    print(i)
    i += 1


# In[ ]:


# While loop:
i=1
while i < 10:
    print(i)
    if i == 6:
        break
    i += 2   


# In[ ]:


def fun(data):
    while int(data['DOE_Year']) < 1950:
        return 'Before_1950'
    if int(data['DOE_Year']) >= 1950 and int(data['DOE_Year']) < 2000:
        return 'Between_1950_to_1999'
    else:
        return 'After_1999'
    


# In[ ]:


Department_Information['How_old']=Department_Information.apply(fun , axis=1)
Department_Information.head()


# In[ ]:


# Data Sorting:
string = ' Today is Sunday'
sorted_string = sorted(string)
sorted_string


# In[ ]:


Department_Information['DOE_Year']=sorted(Department_Information.DOE_Year)        # Ascending Order - (Smallest to largest)
Department_Information


# In[ ]:


Department_Information['DOE_Year']=sorted(Department_Information.DOE_Year , reverse=True)  # Descending Order - (Largest to smallest)
Department_Information.head()


# In[ ]:


# Add row and column to data:
# Adding Row:
New_row=pd.DataFrame({'Department_ID':'IDEPT4560' , 'Department_Name':'Statistics' , 'DOE':'1966-08-19T00:00:00Z' ,
                     'DOE_Year':'1966' , 'DOE_Month':'August' , 'How_old':'Between_1950_to_1999'},index=[0])
Department_Information=pd.concat([New_row,Department_Information]).reset_index(drop=True)
Department_Information.head()


# In[ ]:


# Adding Column :
Department_Information['Concat']=Department_Information.apply(lambda x:x['DOE_Year']+'|'+x['DOE_Month'] , axis=1)
Department_Information


# In[ ]:


print('{}\n{}\n'.format(Student_Counceling_Information.head() , Student_Performance_Data.head()))


# In[ ]:


df1={'Customer_ID':pd.Series([1,2,3,4,5,6]) ,
     'Product':pd.Series(['Oven','Oven','Oven','TV','TV','TV'])}
df1=pd.DataFrame(df1)
df2={'Customer_ID':pd.Series([2,4,6]) , 
     'State':pd.Series(['California','California','Texas'])}
df2=pd.DataFrame(df2)
print('{}\n{}\n'.format(df1,df2))


# In[ ]:


# Data Merge :
#1) Left Join 
#2) Right join
#3) Inner join
#4) Outer join
#5) Anti join(Left outer join)


#1) Left Join:

Left_join=pd.merge(df1 , df2 , on='Customer_ID' , how = 'left')
Left_join.head()


# In[ ]:


#2) Right Join :
Right_Join=pd.merge(df1 , df2 , on='Customer_ID' , how = 'right')
Right_Join


# In[ ]:


#3) Inner Join
Inner_join=pd.merge(df1,df2,on='Customer_ID',how='inner')
Inner_join


# In[ ]:


#4) Outer Join :
Outer_Join=pd.merge(df1,df2,on ='Customer_ID',how='outer')
Outer_Join


# In[ ]:


#5) Anti join:
merge=pd.merge(df1,df2,on='Customer_ID',how='outer',indicator=True)
Anti_Join=merge[merge['_merge'] == 'left_only']
Anti_Join

