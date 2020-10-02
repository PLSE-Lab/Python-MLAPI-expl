#!/usr/bin/env python
# coding: utf-8

# <p>Objective : To explore group by and aggregation methods on data using python (library: Pandas).</p>
# <p style="color:#666666">Last updated: 06th Feb 2017<br>Akshay Sehgal, www.asehgal.com</p>

# <h2>1. Introduction</h2>
# 
# <p>SQL groupby is probably the most popular feature for data transformation and it helps to be able to replicate the same form of data manipulation techniques using python for designing more advance data science systems. As a result, its important to understand the basic components of a groupby clause.</p>
# <ul>
# <li><b>Select</b> - Is the list of aggregated features that the analyst is interested in</li>
# <li><b>From</b> - Source of the data</li>
# <li><b>Group By</b> - Feature(s) whose distinct values will be the basis of the grouping of selected aggregate features</li>
# <li><b>Where</b> - Any additional conditions that need to be checked on the raw data, before grouping up the data</li>
# <li><b>Having</b> - Any additional conditions that need to be checked on OUTPUT of the group by query, before displaying it</li>
# </ul>
# 
# <p>Keeping these concepts in mind, the Pandas groupby method will be explored in detail below.</p>

# In[ ]:


#Import Titanic Data
import pandas as pd
df = pd.read_csv('../input/titanic_data.csv')


# <h2>2. Syntax</h2>
# 
# <p>The core syntax can be broken down similar to the Select-From-Groupby-Where clause. Sample code is given below :</p>
# 
# <b><code>Table_name.groupby(['Group'])['Feature'].aggregation()</code></b>
# 
# <ul>
# <li>Table_name to specify the FROM</li>
# <li>'Group' is the list of GROUP BY variables</li>
# <li>'Feature' is the list of SELECT variables (with or without WHERE condition)</li>
# <li>Aggregate() is to specify the aggregation</li>
# </ul>

# In[ ]:


#Two step query to find sum of survived people, grouped by their passenger class (1 > 2 > 3)
group_survived = df.groupby(['Pclass'])
out_survived = group_survived['Survived'].sum()
print(out_survived)


# In[ ]:


#Above snippet can be implemented in a single command as follows
out_survived = df.groupby(['Pclass'])['Survived'].sum()
print(out_survived)


# <h3>2.1 Adding more groups/levels</h3>
# <p>We can pass a list of features in the groupby() to increase the levels of divisions in data as below :</p>

# In[ ]:


#Three level groupby to find mean of age
df.groupby(['Survived','Pclass','Sex'])['Age'].mean()


# <h3>2.2 Adding more variables/features</h3>
# <p>Similarly, we can also pass a list of features after the groupby to increase the variables we want to aggregate, as below :</p>

# In[ ]:


#Three level groupby to find mean of age and fares
#reset_index() just arranges the column names properly like a data frame
df.groupby(['Survived','Pclass','Sex'])['Age','Fare'].mean().reset_index()


# <h3>2.3 WHERE Clause</h3>
# <p>Adding a Where clause is quite intuitive as you can specify this as conditions before the groupby() method. This first applies the where condition on the dataframe, then groups it and aggregates given variables to throw results.</p>

# In[ ]:


#Fare and Age average for only those who survived
df[df['Survived']==1].groupby(['Pclass','Sex'])['Age','Fare'].mean().reset_index()


# In[ ]:


#The same query above can be broken down into 3 steps for better understanding
df1 = df[df['Survived']==1]
grouped_data = df1.groupby(['Pclass','Sex'])
output = grouped_data['Age','Fare'].mean()
print(output.reset_index())


# <h3>2.4 Multiple Aggregations - Stepwise</h3>
# <p>Till now only one aggregation is being applied on variables in all the examples above. Next is how to create multiple types of aggregations on data. This task can be performed step by step with first grouping the table, next creating 1 aggregate variable at a time, then finally combining them into a single dataframe using pd.DataFrame()</p>

# In[ ]:


##Step 1: Group by Gender
groupby1 = df.groupby(['Sex'])

##Step 2: Calculate different aggregations on 'Fare' variable
meanfare = groupby1['Fare'].mean()
maxfare = groupby1['Fare'].max()
minfare = groupby1['Fare'].min()
stdfare = groupby1['Fare'].std()
rangefare = maxfare-minfare  #Can also create custom aggregations

##Step 3: Combine into a single DataFrame
#Min, Mean, Max
farestats1 = pd.DataFrame({'meanfare':meanfare,'maxfare':maxfare,'minfare':minfare})
#Mean, Range, Standard deviation
farestats2 = pd.DataFrame({'meanfare':meanfare,'stdfare':stdfare,'rangefare':rangefare})

print(farestats1.reset_index())
print(farestats2.reset_index())


# <h3>2.5 Multiple Aggregations - using agg()</h3>
# <p>This is an advanced way of using multiple aggregations on different variables by use of AGG() and DICTIONARIES.</p>
# <p>The difference between [ ] and { } parenthesis is that square brackets represent a list where each element is unique, while curly brackets represent a set(), where we have the ability to create dictionaries for later use. One such use of dictionaries is agg() method a.k.a aggregate method.</p>

# In[ ]:


##First define the functions that need to be performed

#Dictionary 'f' uses 3 aggregations on same variable 'fare' 
f = {'Fare':['mean','max','min']}

#The dictionary is then passed into the aggregate() method
df.groupby(['Sex']).agg(f).reset_index()


# In[ ]:


#Dictionary 'g' contains 2 separate aggregations on 2 different  variables 'fare' and 'age' respectively
g = {'Fare':['mean','std'],'Age':['mean','std']}
df.groupby(['Sex']).agg(g).reset_index()


# <h3>2.6 Renaming aggregated variables</h3>
# <p>Initialy while creating dictionaries, we used { } to define the first level of the dictionary, but the sub-levels were inputted still in [ ]. Here the only difference is that instead of passing a list [ ] into a dictionary element, we pass another dictionary to it, since we can associate labels to dictionary elements easily. For example :</p>
# <li> <b>{ 'element1' : ['a','b'] }</b> is a dictionary with list 'a','b' passed to element1.</li>
# <li> <b>{ 'element1' : {'a','b'} }</b> is a dictionary with dictionary 'a','b' passed to element1.</li>
# 
# <p> This allows adding labels to the dictionary inside as follows :</p>
# <li> <b>{ 'element1' : { 'label1':'a'  ,  'label2':'b' }}</b> </li>

# In[ ]:


#Dictionary h contains mean() as average, max() as maximas and min() as minimas, associated with variable 'fare'
h = {'Fare':{'average':'mean','maximas':'max','minimas':'min'}}
print(df.groupby(['Sex']).agg(h).reset_index())


# <h3>2.7 Custom Aggregations</h3>
# <p>There are 2 ways of creating custom aggregations. One is using the step by step method above to create the custom aggregation (as shown previously with 'Rangefare' aggregation). The other method is using LAMBDA X to create the aggregation, as shown below</p>

# In[ ]:


#Lambda function can be associated with a calculation as well as a label to create custom aggregations
i = {'Fare':{'average':'mean','deviation':'std','range': lambda x : max(x)-min(x)}}
print(df.groupby(['Sex']).agg(i).reset_index())


# <h2>3. Pandas predefined methods</h2>
# 
# <p>This is a work in progress list of aggregate methods that can be used with groupby(). </p>

# In[ ]:


#Different methods can be called during pandas groupby and aggregate

non_null_count = df.groupby(['Sex'])['Age'].count()
summation = df.groupby(['Sex'])['Age'].sum()
average = df.groupby(['Sex'])['Age'].mean()
mean_absolute_dev = df.groupby(['Sex'])['Age'].mad()
arithmetic_median = df.groupby(['Sex'])['Age'].median()
maximum = df.groupby(['Sex'])['Age'].max()
minimum = df.groupby(['Sex'])['Age'].min()
product = df.groupby(['Sex'])['Age'].prod()
unbiased_std_dev = df.groupby(['Sex'])['Age'].std()
unbiased_variance = df.groupby(['Sex'])['Age'].var()
unbiased_std_err_of_mean = df.groupby(['Sex'])['Age'].sem()
unbiased_skewness_3rdmoment = df.groupby(['Sex'])['Age'].skew()

print(arithmetic_median)


# <h2>4. References</h2>
# <ul>
# <li>http://www.scipy-lectures.org/packages/statistics/index.html#hypothesis-testing-comparing-two-groups</li>
# <li>https://www.simple-talk.com/sql/t-sql-programming/sql-group-by-basics/</li>
# <li>http://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/</li>
# <li>http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html</li>
# </ul>
