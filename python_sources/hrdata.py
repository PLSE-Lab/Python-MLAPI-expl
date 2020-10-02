#!/usr/bin/env python
# coding: utf-8

# ## Work In Progress
# **Problem Statement -** 
# 
#  **Given the data of the employees of a company, predict the attrition, so that HR can take preventive measure controls to minimze it.**
#  
# **Analysis Required**
# - Does the data actually helps us to predict  attrition rate?
# - What are the features that may contribute for attrition?
# - How can we decrease the attrition?**
# **

# In[ ]:


# import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st
sns.set(color_codes=True)

#Visualize plots within the notebook

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Importing the dataset
data = pd.read_csv('../input/hr-data/HR_Employee_Attrition_Data.csv')


# In[ ]:


#Check if the data is imported correctly
data.sample(5)           


# In[ ]:


#Finding the description of data

data.describe()


# ### Data Exploration

# In[ ]:


#Check the datatypes , shape to know more about data
print(data.shape)
print(data.dtypes)


# In[ ]:


#Check any duplicated data

data[data.duplicated()]


# In[ ]:


#Finding the missing values

data.isnull().sum()


# ### Cleaning the data

# In[ ]:


data.sample(5)


# In[ ]:


#We will seperate


# In[ ]:


## Finding the outliers first

Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
## We will only consider 99.3 percentile useful data for our analysis and remove remaining 0.7 percentile data below or above it
mini = Q1 - 1.5 * IQR
maxi = Q3 + 1.5 * IQR
data = data[~((data < mini) | (data > maxi)).any(axis = 1)]
mini


# In[ ]:


##Lets map attrition rate Yes - 1 , No : 0
at = {'Yes' : 1 , 'No' : 0}

data['Attrition'] = list(map(lambda x : at[x] , data.Attrition))


# In[ ]:


##Drop unnecessary columns that do not effect the attrition

data.drop(columns = ['EmployeeCount', 'EmployeeNumber', 'Over18' , 'StandardHours'] , inplace = True)


# In[ ]:


data.shape


# ### What is the age group of the employees where attrition occurs maximum? 

# In[ ]:


plt.figure(figsize = (16,6))
sns.distplot(data.Age, kde = False , color = 'blue',  hist_kws={"alpha": 1})
sns.distplot(data[data.Attrition == 1].Age , color = 'orange', kde = False , hist_kws={"alpha": 1})


# ### We can conclude that maximum employees age between 28 - 38 years and  attrition is maximum at 25 -35 years.

# In[ ]:


### Lets visualize the percentage of males wrt females 

sns.countplot(x = 'Gender' , hue = 'Attrition' , data = data)


# ### We can conclude attrition of males are nearly as twice as females.

# In[ ]:


### Lets draw pairplot to have better understanding for the relationships between data

##https://towardsdatascience.com/exploratory-data-analysis-in-python-c9a77dfa39ce

##https://medium.com/@monikapdb/employee-attrition-analysis-using-machine-learning-methods-73564358e87f


# In[ ]:


plt.subplots(figsize =(10,6))
c = data.corr()
sns.heatmap(c, cmap = 'coolwarm', annot = False)


# In[ ]:



##Corelation shows that Age, Joblevel, Monthly Income, TotalWorkingYears, YearsAtCompany , YearsInCurrentRole ,         
##YearsSinceLastPromotion , YearsWithCurrManager   are important features for analysis.        


# In[ ]:


(data.PerformanceRating == 3)


# ## The average performance rating is 3 for all employees where attrition is Yes.

# In[ ]:


data.drop('PerformanceRating' , axis = 1 , inplace = True)


# In[ ]:


##Visualize the monthly income
##Let us bin the monthly salary first

bins = [1000, 1500 , ]
df['binned'] = pd.cut(df['percentage'], bins)
print (df)


# In[ ]:


df[df.MonthlyIncome > 12000]


# ### We can conclude that there are only avg of 4 people where monthlyincome > 12000, where Attrition is True

# In[ ]:


##whats the avg no of people where salary > 12000 and attrition == 0
data[(data.MonthlyIncome > 12000) & (data.Attrition == 0)].shape


# #MonthlyIncome also depends on JobLevel, TotalWorkingYears (See Correlation map)
# ### What is the joblevel and workingyears of the employees that determine their monthly income?

# In[ ]:


sns.barplot(x = 'MonthlyIncome' , y = 'JobLevel', data = data)


# In[ ]:


#https://towardsdatascience.com/why-feature-correlation-matters-a-lot-847e8ba439c4

# list(map(lambda x : ( x , x + 1499) , range(1000, 17000, 1499)))

l = []
def f(start):
    nextval = 0
    while(nextval <= 17000):
        x = start
        nextval = start + 1499
        l.append((x, nextval))
        start = nextval + 1
    return l
bins = f(1000)
bins

MI = pd.DataFrame(bins)


# In[ ]:





# In[ ]:


MI


# In[ ]:


data[['MonthlyIncome', 'Attrition']]


# In[ ]:


data['Yes'] = pd.cut(data[data['Attrition'] == 1].MonthlyIncome , 10)
# pd.cut(data['MonthlyIncome'], 10)


# In[ ]:


data['MonthlyIncome']


# In[ ]:


data.drop('Yes' , axis = 1, inplace = True)


# In[ ]:


data.shape


# In[ ]:



df = pd.DataFrame({
    'AT': data[data['Attrition'] == 1].MonthlyIncome,
    'NAT': data[data['Attrition'] == 0].MonthlyIncome
})


# In[ ]:


df


# In[ ]:


custom_bucket_array = np.linspace(1000, 20000, 9, dtype = 'int')
custom_bucket_array


# In[ ]:


df['AT'] = pd.cut(df['AT'], custom_bucket_array)
df['NAT'] = pd.cut(df['NAT'], custom_bucket_array)
df.head()


# In[ ]:


df


# In[ ]:


data[['MonthlyIncome', 'Attrition']]


# In[ ]:



a = df.groupby('AT').size()
b = df.groupby('NAT').size()
plt.figure(figsize = (16,6))
categories = df['NAT'].cat.categories
ind = np.array([x for x, _ in enumerate(a.index)])
width = 0.35    
plt.bar(x = ind, height = a, width = 0.35, label='Attrition')
plt.bar(ind + width, b, width,
    label='No Attrition')

plt.xticks( ind + width / 2  , labels = a.index, fontweight = 'bold')
plt.legend()
plt.xticks(rotation = 90)
plt.yticks(fontweight = 'bold')
plt.show()




# In[ ]:


np.random.normal(10, 3, (3,3))


# In[ ]:


import random
random.randrange(1,20, size = (2,1))


# In[ ]:


type(np.random.randint(10, size=(3,4,5)))


# In[ ]:


np.array(np.random.randint(1,20,(2,2)))


# In[ ]:


np.array([[1,2,3],[4,5,6], [9,0, 2]])


# In[ ]:


x = np.array([3,4,5])
grid = np.array([17,18,19])
np.concatenate([x,grid])


# In[ ]:


np.array([[9],[9]])


# In[ ]:


income = np.random.randint(1000, 9000, 9000).reshape(9000,1)
attrition = np.random.randint(0,2,9000).reshape(9000,1)
test = np.random.randint(3,6,9000).reshape(9000,1)
df = pd.DataFrame(np.hstack([income, attrition, test]), columns = ['income', 'attrition', 'test'])
df.head()


# In[ ]:


r = pd.cut(df.income, np.arange(1000, 9001, 500))


# In[ ]:


print(r.index)


# In[ ]:


dat = df.groupby([r, 'attrition']).count().unstack()
dat


# In[ ]:





# In[ ]:


s = pd.Series(['x',2,7,1], index = list('ABCD'))
s


# In[ ]:


pd.DataFrame(s).reset_index()


# In[ ]:



dat.columns = dat.columns.droplevel()
dat.head()


# In[ ]:


dat


# In[ ]:


row_sum = dat.sum(axis = 1)
row_sum.head()


# In[ ]:


dat.iloc[:,0] = dat.iloc[:,0]/ row_sum * 100
dat.iloc[:,1] = dat.iloc[:,1]/ row_sum * 100


# In[ ]:


dat


# In[ ]:


df.plot(kind = 'bar', stacked = True)


# In[ ]:


df1 = pd.DataFrame({'id':[1,1,1,2,2,2,2,3,4],'value':[1,2,3,1,2,3,4,1,1]})
df1


# In[ ]:


df1.loc[:3, ['id', 'value']]


# In[ ]:


df1.groupby('id').apply(lambda x: x['value'].reset_index()).reset_index()


# In[ ]:


stocks = pd.read_csv('http://bit.ly/smallstocks')


# In[ ]:


data


# In[ ]:


ser = data.groupby(['Attrition', 'BusinessTravel']).Age.mean()
ser


# In[ ]:


ser.unstack()


# In[ ]:


dta = data.pivot_table(values = 'Age', index = 'Attrition', columns = 'BusinessTravel' , aggfunc = 'mean')
dta


# In[ ]:


ser


# In[ ]:


ser.loc[, 'Travel_Rarely']


# In[ ]:


ser


# In[ ]:


data.head(50)


# In[ ]:


##dataframe with multiindex

data.set_index(['Attrition' , 'BusinessTravel'], inplace = True)


# In[ ]:


data.sort_index(inplace = True)


# In[ ]:


data.loc['No', 'Travel_Rarely']


# In[ ]:


data.loc['No']


# In[ ]:


data['Age'][(data.Age > 32) & (data.Attrition == 'Yes')]


# In[ ]:


def addition(n): 
    return n + n 
  
# We double all numbers using map() 
numbers = (1, 2, 3, 4) 
result = map(addition, numbers) 
print(list(result)) 


# In[ ]:


data['title'] = [0] * dataset[]

