#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#hello every one again !!  
#i will try to do  datascience toolbox samples.

import numpy as np # we import numpy library as np
import pandas as pd # we import pandas library as pd


# In[ ]:


my_data = pd.read_csv('../input/pokemon.csv') # importing poke data


# In[ ]:


my_data.head(7)


# In[ ]:


dropped_data=my_data.drop(columns='#') # i want to drop '#'columns 
dropped_data.head(7)


# In[ ]:


data=dropped_data
#as you see on top this blog, cloumn names have space char ' ',dot char '.' and upper char. 
# i want to replace all and lower type these 
data.columns=[i.replace('.','') for i in data.columns] 
data.columns=[i.replace(' ','_') for i in data.columns]
data.columns=[i.lower() for i in data.columns]

data.columns #checking column names


# In[ ]:


data.head()


# In[ ]:


#as we see, poke data have many 'Nan' values. we will change 'nan' name values to "invalid_name"
data.name=[ "invalid_name"  if str(i)=='nan' else str(i) for i in data.name]


# In[ ]:


#also type column have many 'Nan' values. we will change this types to "invalid_type"
data.type_1=[ "invalid_type"  if str(i)=='NaN' else str(i) for i in data.type_1]


# In[ ]:


#again we'll change 
data.type_2=[ "invalid_type"  if str(i)=='nan' else str(i) for i in data.type_2]


# In[ ]:


#we can do this changing lambda function
#note: apply() function in to pandas library function. it applys every item data.name list in parantheses function.

data.name = data.name.apply(lambda x: str(x).replace(" ", "_") if  " " in str(x) else x) #replacing space char to "_" char.
data.name[:15] #checking


# In[ ]:


#we do same thing for type_1 column
data.type_1 = data.type_1.apply(lambda x: str(x).replace(" ", "_") if  " " in str(x) else x)

data.type_1[:25]


# In[ ]:


#we do same thing for type_2 column. bu we dont do it  lambda function we ll do list comprehension method.
data.type_2=[i.replace("invalid_type_2", "nan")  for i in data.type_2]

data.type_2[:25]


# In[ ]:


#our data column names have upper character. we want to change this.
#we do same thing lambda function

for i in data:
    data.name=data.name.apply(lambda x: str(x).lower()) #changing name columns value lower character
    data.type_1=data.type_1.apply(lambda x: str(x).lower()) ##changing type1 columns value lower character
    data.type_2=data.type_2.apply(lambda x: str(x).lower()) #changing type2 columns value lower character
    data.legendary=data.legendary.apply(lambda x: str(x).lower()) ##changing legendary columns value lower character
    
data.head()

