#!/usr/bin/env python
# coding: utf-8

# I hope you find this kernel helpful and some<font color="red"><b> UPVOTES</b></font>  would be very much appreciated
# 
# 

# In[ ]:


import warnings                       # to hide warnings if any
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))


# ### **Importing Required Libraries**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# ### **Reading the Datasets**

# In[ ]:


marvel = pd.read_csv('../input/marvel-wikia-data.csv')
dc = pd.read_csv('../input/dc-wikia-data.csv')


# In[ ]:


marvel.head(3)


# In[ ]:


dc.head(3)


# ### **Cleaning the dataset**

# Both the datasets contains some unnecessary columns like
# 1. page_id
# 2. urlslug
# 3. FIRST APPEARANCE
# Removing unnecessary columns

# In[ ]:


#Function to remove unnecessary columns

def remove_col(df, *col):                    # Here df = name of dataframe, *col = names of unnecessary columns
    df.drop([*col], axis = 1, inplace = True)


# In[ ]:


remove_col(dc,'page_id','urlslug','FIRST APPEARANCE')        # removing columns from dc's dataset


# In[ ]:


remove_col(marvel,'page_id','urlslug','FIRST APPEARANCE')    # removing column's from marvel's dataset


# I am also removing the name given in '( )' in the names columns and only using the superhero names of characters

# In[ ]:


marvel['name'] = marvel['name'].apply(lambda x: x.split('(')[0])
dc['name'] = dc['name'].apply(lambda x: x.split('(')[0])


# Correcting the **'Year'** column in Marvel's dataset and DC's dataset as they  have a mismatch. 

# In[ ]:


marvel.rename(columns = {'Year':'YEAR'}, inplace = True)


# ### **Cleaned Dataset**

# #### **1. Marvel's Dataset**

# In[ ]:


marvel.head(3)


# ####** 2. DC's Dataset**

# In[ ]:


dc.head(3)


# ### **Describing the Dataset**

# #### ** 1. Total number of rows and columns in dataset**

# In[ ]:


row, col = dc.shape[0], dc.shape[1]
print("-------- DC'S DATASET --------")
print('Number of rows: ',row)
print('Number of columns: ',col)

print()

row,col = marvel.shape[0], marvel.shape[1]
print("-------- MARVEL'S DATASET --------")
print('Number of rows: ',row)
print('Number of columns: ',col)


# #### **2. Each columns datatype and number of entries**

# In[ ]:


print("-------- DC'S DATASET --------")
print('')
dc.info()


# In[ ]:


print("-------- MARVEL'S DATASET --------")
print('')
marvel.info()


# #### **3. Colums in the dataset**

# Both datasets contains same columns, so using anyone of them to describe the colums

# #### **1. Name of columns in the datasets is:**

# In[ ]:


count = 0
for i in dc.columns:
    count = count + 1
    print(count,'. ',i)


# #### **2. Describing each column**

# In[ ]:


# function to print unique values in a given column.
# Here col = column name

def print_unique(col):
    print('Unique values in the ', col, 'Column are: ')
    print()
    count = 0
    for i in dc[col].unique():
        count = count + 1
        print(count,'. ',i)


# #### **i. ID Column**

# In[ ]:


print_unique('ID')


# #### ** ii. ALIGN Column**

# In[ ]:


print_unique('ALIGN')


# #### **iii. Eye Column**

# Eye Column gives the color of eyes of the respective character.

# In[ ]:


print_unique('EYE')


# #### iv. **HAIR Column**

# HAIR Column tells about a character's hair color.

# In[ ]:


print_unique('HAIR')


# #### **v. SEX Column**

# SEX Column tells about a character's gender.

# In[ ]:


print_unique('SEX')


# #### **vi. GSM Column**

# **GSM** stands for **G**ender and **S**exual **M**inorities and tells about a character's sexual orientation

# In[ ]:


print_unique('GSM')


# #### **vii. ALIVE Column**

# ALIVE Column tells whether a character is currently alive or dead

# In[ ]:


print_unique('ALIVE')


# #### **viii. APPEARANCES Column**

# The APPEARANCES Column contain information about when the characters first appeared in the comics

# ### **Exploratory Data Analysis and Visualization**

# In[ ]:


# function to draw count plots for both marvel and dc comics side by side
#Here col = Name of column
#color = color palette's name
#xtic = x-axis label's rotation

sns.set_style('darkgrid')

def plot_countplot(col,hue = None, color = 'magma',xtic = 0,ylim = 13000):
    plt.figure(figsize=(12,6))
    
    plt.subplot(1,2,1)
    sns.countplot(x = col, data = dc, hue = hue, palette = color)
    plt.xticks(rotation = xtic)
    plt.ylim(0,ylim)
    plt.title('DC Comics')
    
    plt.subplot(1,2,2)
    sns.countplot(x = col, data = marvel, hue = hue, palette = color)
    plt.xticks(rotation = xtic)
    plt.ylim(0,ylim)
    plt.title('Marvel Comics')
    
    plt.tight_layout()
    


# In[ ]:


# function to print the count of different values present in each column
# df = name of dataset
# col = column name

def stats(df, col):
    for i in df[col].unique():
        print(i,': ',len(df[df[col] == i]))


# ### **1. Number of Characters on the basis of Identity**

# In[ ]:


plot_countplot('ID',xtic=30)


# **Plot Summary**

# #### **i. DC Universe**

# In[ ]:


stats(dc,'ID')


# #### **ii. Marvel's Universe**

# In[ ]:


stats(marvel,'ID')


# In DC Comics more superheroes are known by the public whereas in Marvel Comics superheros having a secret identity is great.

# #### **2. Characters on the basis of their alignment**

# In[ ]:


plot_countplot('ALIGN',xtic=45)


# #### **Plot summary**

# #### ** i. DC Comics**

# In[ ]:


stats(dc,'ALIGN')


# #### **ii. Marvel Comics**

# In[ ]:


stats(marvel,'ALIGN')


# DC Comics have an almost equal number of superheros and villans whereas in Marvel Comics the number of villans is far greater than superheroes.

# #### **3. Number of Characters on the basis of gender**

# In[ ]:


plot_countplot('SEX',xtic=40)


# #### **Plot summary**

# #### **i. DC Comics**

# In[ ]:


stats(dc,'SEX')


# #### **ii. Marvel Comics**

# In[ ]:


stats(marvel,'SEX')


# The number of male characters in both the comics are greater than female characters.

# #### **4. Number of characters Living or Dead**

# In[ ]:


plot_countplot('ALIVE')


# #### **Plot Summary**

# #### **i. DC Comics**

# In[ ]:


stats(dc,'ALIVE')


# #### **ii. Marvel Comics**

# In[ ]:


stats(marvel,'ALIVE')


# Most of the characters in both the comics are currently alive

# #### **5. Number of characters on the basis of their eye color**

# In[ ]:


plot_countplot('EYE',ylim = 2000,xtic=60)


# **Plot Summary**

# **i. DC Comics**

# In[ ]:


stats(dc,'EYE')


# **ii. Marvel Comics**

# In[ ]:


stats(marvel,'EYE')


# Most of the characters in both the comics have blue or brown eyes

# #### **6. Grouping alive and dead characters according to their alignment**

# In[ ]:


plot_countplot('ALIVE',hue='ALIGN',ylim=5000)


# **Plot Summary**

# #### ** i. DC Comics**

# In[ ]:


alive_align = dc.groupby(['ALIVE','ALIGN']).aggregate('count')
alive_align['name']


# #### **ii. Marvel Comics**

# In[ ]:


alive_align = marvel.groupby(['ALIVE','ALIGN']).aggregate('count')
alive_align['name']


# The number of bad characters who are alive is more in Marvel's universe whereas there are more good characters currently alive in the DC universe.

# #### **7. Plotting Appearance of Characters according to Years**

# **i. DC Comics**

# In[ ]:


plt.figure(figsize=(20,6))
sns.lineplot(x = 'YEAR',y = 'APPEARANCES',hue = 'SEX',data= dc,markers= True,dashes=False,lw=2)
plt.ylim(0,1000)
plt.show()


# Most of the Male and Female characters appeared before the 1940's in DC Comics

# **ii. Marvel Comics**

# In[ ]:


plt.figure(figsize=(20,6))
sns.lineplot(x = 'YEAR',y = 'APPEARANCES',hue = 'SEX',data= marvel,markers= True,dashes=False,lw=2)
plt.ylim(0,600)
plt.show()


# Most of the male and female characters appeared during 1960 - 1970.Also, many agender characters appeared during 1980-85.

# **The EDA is currently incomplete and other features will be added shortly.**<br>
# **Suggestions are welcome.**

# In[ ]:




