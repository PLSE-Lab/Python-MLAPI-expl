#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Pandas has most of our functionalities..including loading as a
# dataframe object, as well as removing na, finding description of data
import pandas as pd


# In[ ]:


#Reads the loan file and stores everything loaded into a variable named data
# Henceforth, we can perform all operations on the variable named data
#Thus you should always put the loaded data into a variable

#It isnt necessary to name the variable data, I have used that name
#because it is the most sensible name for this variable

data = pd.read_csv("../input/train.csv")


# In[ ]:


#This file was with the data, it has description of all the columns
#In normal hackathons, you might or might not get a meta data file

#Metadata is simply explaination about the given data
metadata = pd.read_excel("../input/LCDataDictionary.xlsx")


# In[ ]:


#We visually inspect the first and last 10 rows
data.head(10)


# In[ ]:


data.tail(10)


# In[ ]:


#Visual inspections show last few columns had null values..how about
#we check the entire table for nulls? We do that using the function
#info()
#to get info of the table, the syntax is variablename.info()
#In our case that will be data.info()


# In[ ]:


data.info()


# In[ ]:


#We see that there are several null values! We also see the several data
#types that are present in the data
"""
This is in pg 119 of the python book I forwarded
"""
##Keep in mind that we need to eliminate the nulls and handle the
##categorical values before we build a model/train it


# In[ ]:


#Whats the number of rows and columns??
data.shape


# In[ ]:


#Do not forget, shape is an attribute of data..
"""
Explanation of Class and Object for everyone that missed it
"""

#Consider the IPL example:

#IPL wants to keep data of all the players. To do that, they 
#create a template of how the data will be stored..

# If it was for Sachin
##IPL necessarily needs to store data of each player's matchwise
##performance too! That can be a table with columns like
##date of match, location, time, team in which sachin played, against whom,
## etc.
## These will have several rows storing each match's data

##few values for Sachin like (age, height, weight, year of starting, DOB)
##are constant for sachin. 
##These do not need to be stored in the table.
##These are the attributes and can be stored seperately in this template
##Thus the template has a table and several values like dob, age, etc
##to get attributes, you just type variablename.NameOfAttribute
##thus sachin.age can store his age, sachin.height can store his height etc
##A sensible template would also handle operations like, if IPL wants
##sachin's strike rate; I would expect the code to PROCESS the entire 
##match data and give me a result (note that I want a process to be run
##on the data). To get that, the template should also have few inbuilt
##function for the same. Thus sachin.strikeRate() should give me the
##strikerate (notice that process to be done on data is a function
## and is associated with round bracket -> ())

#IPL can use this template for every other player they have! 

#The template is called a class. A class has attributes and functions 
#associated with it. When you use the template for a specific
#instance, it is called an object. 
#Thus as the above example is of variable named sachin, sachin is the
#object of the IPL template







# In[ ]:


#Statistical data of our table
data.describe()


# In[ ]:


##when someone is filling out your survey, they arent always so sweet to
##fill all the fields. They will skip optional fields.
##In python, all blank fields will be filled by Null/nan/na


# In[ ]:


#Trying out different configurations for dropna
#dropna is useful to drop null values
#This drops all rows which have even 1 occurance of null
new_data = data.dropna()


# In[ ]:


#We see how many rows are left!!
new_data.shape


# In[ ]:


#NO ROWS LEFT!! Cant dropna on rows..but how about columns?


# In[ ]:


new_data_2 = data.dropna(axis = 1)


# In[ ]:


#This is good..we get 34 columns
new_data_2.shape


# In[ ]:


#How do you get number of null values for each column??

#One way is using the function isnull(), 
# it returns a table with true false. true value measn the cell was null

#when we write isnull().sum(), it gives us columnwise sum of the true-false table
#derived from isnull function
new_data_2.isnull().sum()


# In[ ]:


#Percentage of null values
pctg = (data.isnull().sum()/data.shape[0])*100


# In[ ]:


#Sorting values!
pctg.sort_values(ascending =  False)


# In[ ]:


92*887379/100


# In[ ]:


#Dropping columns with less than 92% data absent by using thresh parameter
new_data = data.dropna(axis = 1,thresh= data.shape[0]*92/100)


# In[ ]:


#These for and if statements eleminate the null values
#in Categorical columns nulls are replaced by mode
#in numerical columns, nulls are replaced by mean


# In[ ]:


for col_name in new_data.columns:
    if new_data[col_name].dtype == "object":
        mode_var = new_data[col_name].mode()[0]
        new_data[col_name].fillna(mode_var,inplace = True)
    else:
        mean_var = new_data[col_name].mean()
        new_data[col_name].fillna(mean_var,inplace = True)


# In[ ]:


#Confirming if our nulls are deleted
new_data.isnull().sum()


# In[ ]:


#We havent converted all categorical to numerical!!

#Wha are the different ways of doing it??


"""
0. Replace
1. One hot encoding
2. LabelEncoder
    2.1 categorical to codes
    2.2 sklearn's encoder
"""


# In[ ]:


#What is our X and Y?
#Y is loan_status!

new_data["loan_status"].unique()

#How to find the count of each class?? Is it imbalenced??


# In[ ]:


new_data["loan_status"].value_counts()


# In[ ]:


#Imbalenced!! And I see repeatred words charged off and fully paid.. 
#lets merge them with 'does not meet.....""


# In[ ]:


new_data = new_data[new_data["loan_status"] != "Current"]


# In[ ]:


new_data = new_data[new_data["loan_status"] != "Issued"]


# In[ ]:


"""Fully Paid
Charged Off
Late (31-120 days)
In Grace Period
Late (16-30 days)
Does not meet the credit policy. Status:Fully Paid
Default
Does not meet the credit policy. Status:Charged Off""".split("\n")


# In[ ]:


def converts(x):
    values = """Fully Paid
Charged Off
Late (31-120 days)
In Grace Period
Late (16-30 days)
Does not meet the credit policy. Status:Fully Paid
Default
Does not meet the credit policy. Status:Charged Off""".split("\n")
    is_it_positive = [1,0,0,0,0,1,0,0]
    for this_string,ret_value in zip(values,is_it_positive):
        if x == this_string:
            return ret_value

new_data["loan_status"] = new_data["loan_status"].apply(converts)


# In[ ]:


#Google what is charged off, current and fully paid


# In[ ]:


new_data["term"]


# In[ ]:


from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

enc.fit(new_data["term"])


# In[ ]:


new_data["term"] = enc.transform(new_data["term"])


# In[ ]:


#count by function, count by indexing, count by aggregation/groupby


# In[ ]:


new_data["loan_status"]


# In[ ]:





# In[ ]:





# In[ ]:


class_marks = [1,2,3,4,5,6,7,8,9,10,1,1,1,2,34,4,5,6,78,32,64,3,4,6,7,11]


for mark in class_marks:
    if mark < 30:
        print("Fail")
    else:
        print("Pass")
    


# In[ ]:


co = "addr_state"
data[co].dtype == "object"


# In[ ]:


from sklearn.preprocessing import LabelEncoder

for co_name in new_data.columns:
    if new_data[co_name].dtype == "object":
        template = LabelEncoder()
        template.fit(new_data[co_name])
        new_data[co_name] = template.transform(new_data[co_name])

new_data.head(10)
        


# In[ ]:


from sklearn.model_selection import train_test_split

y = new_data["loan_status"]
x = new_data.drop(labels = "loan_status", axis = 1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(x,y, random_state=23)


# In[ ]:


X_train


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rftemplate = RandomForestClassifier()
rftemplate.fit(x,y)


# In[ ]:


rftemplate.score(x,y)


# #https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_Matplotlib_Cheat_Sheet.pdf
# #https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_Seaborn_Cheat_Sheet.pdf

# In[ ]:





# In[ ]:





# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt

import seaborn as sns


plt.figure()
sns.distplot(new_data["loan_amnt"])

plt.figure()
plt.hist(new_data["loan_amnt"],bins = 100)


# In[ ]:


new_data.columns


# In[ ]:


plt.figure()
new_data["installment"].plot(kind = "hist")


# In[ ]:


sns.countplot(data["title"])


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd



# In[ ]:


#create a new canvas
plt.figure()

#plot scatterplot..x axis is loan amount, y  is annual income
canv_scatter = sns.scatterplot(x = "loan_amnt",y = "annual_inc", data = new_data)
canv_scatter.set_title("Loan Amount vs Income")


# In[ ]:





# In[ ]:


sns.scatterplot(x = "loan_amnt",y = "annual_inc", data = new_data, hue = "loan_status")


# In[ ]:


new_data["loan_status"].value_counts()


# In[ ]:





# In[ ]:


canv1 = sns.FacetGrid(data = new_data,row = "term", col= "loan_status")
canv1.map(plt.hist,"loan_amnt")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




