#!/usr/bin/env python
# coding: utf-8

# # Panda Group by:  all about group by

# You cant ever work on data without coming across the need to aggregate. If you are working with pandas, it means you will have to work with GroupBy Objects. 
# 
# Pandas groupby is interesting..by that i mean it can be deceptively simple. You think you know everything ..and BAM!! something new comes along
# 
# I promise , once you read this, you will know all about group by operations .(except multi index ..coz i hate hateeee hierarchical data :) )
# 
# 

# ## Things you will learn:
# 
# * **how to group by a single column**
# * **how to group by multiple columns**
# * ** what do i get when i run a groupby operation**
# * **what kind of operations can i execute on the returned groupby object**
# * **how can i aggregate my data**
# * **how can i aggregate and use custom names for the resultant columns**
# * **how can i aggregate based on my own custom function**
# * **how can i change the multi indexed structure to simple table when i aggregate on multi column group by**
# * **how can i apply a transformation**
# * **how can i group by using my own custom function**

# ***Jumping right into code...***

# In[ ]:


import pandas as panda
import numpy as np

## lets create a very simple data frame

animals = ['horse','cat','dog','tiger','whale','dog','tiger','horse','cat','rat','cat']
weight = [245,12,15,200,780,17,176,234,12,9,15]
age = [14,15,4,17,34,2,21,21,2,5,15]
a = panda.DataFrame(
    {
        'animal': animals,
        'weight': weight,
        'age' : age,
        # 'color' : ['']

    }
)

a


# ### **1. I would like to group  by the animal breed**
# 
# This is a simple .groupby(by=['list of columns'] operation. What we will learn?
# * the returned type
# * how the returned type changes
# * how to display the group names
# * how to display the group itself

# In[ ]:


group_by_animal = a.groupby(by = ['animal'])

## groupby returns pandas.core.groupby.DataFrameGroupBy or SeriesGroupBy
## when resultant is multi columns - DataFrameGroupBy
## when resultant is single column - SeriesGroupBy
## try this and see a = panda.DataFrame({'id':np.random.randint(1,10,(6))})
##a.groupby(by=['id'])

print(type(group_by_animal)) ## 


# In[ ]:



## what if i want to know the group names
print('\n The group names are :',group_by_animal.groups.keys())


# In[ ]:



##what if i want to know how the grouping has been done
## essentially which are the groups..which indices form those groups
print('\n My different groups and the indices they belong to :',group_by_animal.groups)


# In[ ]:



## how do i view the groupby.. one very simple method is list
print('\n i would like to see the group : \n' , list(group_by_animal))


# In[ ]:



## view by iterating

for name, group in group_by_animal:
    print('\n Group name: ', name , ' \n group is :', group)


# ***Useful Trick* **: groupby returns data by default sorted according to order of appearance. If you do not need a sorted appearance send sorted = False (eg a.groupby(by =['animal'],sorted=False) . This will improve performance

# ### **2. I would like to group  by the animal breed and age **

# In[ ]:


group_by_animal_and_age = a.groupby(['animal','age'])

for name, group in group_by_animal_and_age:
    print('\n name is :', name)
    print('\n group is : \n', group)


# ### **3. I would like to display group where i know the group name/key **
# 

# In[ ]:


i_know_group_key = 'cat'
i_know_another_group_key = ('whale', 34)
print(group_by_animal.get_group(i_know_group_key),'\n\n', group_by_animal_and_age.get_group(i_know_another_group_key))


# ### **4. I would like to display max age by animal : First Technique**
# 
# This is a clear call to aggregation. When you hear words like max,count, mean,sum,etc in a query immediately think of grouping by and then running an aggregation function

# In[ ]:




## group_by_animal['age'] is simple syntactic sugar.

group_by_animal['age'].max()


# ### **4. I would like to display max age by animal : Second Technique**
# 

# In[ ]:



# or you may you aggregate
group_by_animal['age'].agg(np.max)


# In[ ]:


## example using aggregate
group_by_animal['age'].aggregate(np.max)


# Just like max, you can use many other aggregation functions such as sum/count/mean/std deviation, etc. 

# ### **5. I would like to display max age by animal and the mean age as well**
# 
# This is a clear call to two aggregation on same data set. Lets see how we can use agg() function to achieve the result

# In[ ]:


## pass in a list of aggregation function in the add parameter
group_by_animal['age'].agg([np.max, np.mean])


# In[ ]:


## you will notice in the above example that our group keys becomes our index.
## If you would like to change that, simply use reset_index
group_by_animal['age']    .agg([np.max, np.mean])        .reset_index()


# **Quick Tip: ****what you see above is an example of method chaining

# In[ ]:


# i do not like the column names of the output .
# i would like to give it my own column names

group_by_animal['age'].agg({'i_am_the_max_age':np.max,'i_am_the_mean_age':np.mean})


# In[ ]:


# another way to rename the columns using method chaining

group_by_animal['age']    .agg([np.max, np.mean])        .rename(columns = {'amax':'i am the max age', 'mean': 'i am the mean age'})            .reset_index()


# Total List of Aggregate Functions
# 
# 
# **Function	 : Description** <br>
# mean()	     :Compute mean of groups <br>
# sum()	      :Compute sum of group values<br>
# size()	       :Compute group sizes<br>
# count()	      :Compute count of group<br>
# std()	        :Standard deviation of groups<br>
# var()	        :Compute variance of groups<br>
# sem()	      :Standard error of the mean of groups<br>
# describe()	:Generates descriptive statistics<br>
# first()	         :Compute first of group values<br>
# last()	        :Compute last of group values<br>
# nth()	       :Take nth value, or a subset if n is a list<br>
# min()	      :Compute min of group values<br>
# max()	     :Compute max of group values<br>

# ### **6. I would like to display max age by animal and my own custom function**
# 
# 

# In[ ]:


my_own_custom = lambda x : np.mean(x)-10
group_by_animal['age']        .agg({'max':np.max,'mean':np.mean,'mean_minus_10':my_own_custom})            .reset_index()


# ### **7. I would like to display max weight of  data grouped by animal and age **

# In[ ]:


group_by_animal_and_age['weight'].agg(np.max)


# You can see above that the data is presented in hierarchical format, since we have grouped by two columns.
# How can we fix that? easy.. use reset index

# In[ ]:


group_by_animal_and_age['weight'].agg(np.max).reset_index()


# ### **8. I would like to display how the age differs from the mean age of animals**
# 
# This is a tricky part. If you look at the query a second time, there are two things:
# 
# 1. It requires a group by. since we need to find mean
# 2. The resultant dataset needs to be same as original data set , since we need to show diff between age and mean age
# 3. NOTE: groupby will mostly reduce the len of datasets
# 
# We will see two different ways of achieving the same result

# In[ ]:


mean_age = group_by_animal['age']        .agg({'mean_age':np.mean})                .reset_index()
print(mean_age.shape, a.shape) ## you will see very clearly how the row counts differ

print(mean_age)


# In[ ]:


## we merge the mean back with the original data set
b = a.merge(mean_age, how='inner', on='animal')
b['diff_in_age_with_mean'] = b.age -b.mean_age
b


# While there is absolutely nothing wrong with the above approach, we have to be careful while using merge. Merge is computationally expensive and for larger datasets can lead to out of memory error
# 
# Lets look at a builtin panda technique

# In[ ]:


a["diff_in_age_from_mean"] = group_by_animal["age"].transform(lambda x: x -x.mean())
a


# *What is happening in the above syntax?? How are we achieving everything in one single line of code???*
# 
# transform function : achieves a combination effect on the grouped by splits. Specifically, it achieves a transformation on the whole data set  ensuring the output shape is same as the input shape. The way to think of transformation is :
# 
# 1. groupby divides data in chunks
# 2. transform combines the chunks while applying some transformation. in our case it subtracts from the mean age

# ### **9. How can we group using our own custom function**
# 
# Pandas support grouping by a callable. One big caveat though: callable function runs on the index of the dataframe
# 
# To look at this example, we will change our animal data set such that indices are no longer RangeIndex but infact the animal name

# In[ ]:


a.index = a.animal
a


# In[ ]:


def is_animal_ending_with_at(animal_name):
    if animal_name.endswith('at'):
        return 'animals_ending_with_at'
    else:
        return 'other animals'

list(a.groupby(by = is_animal_ending_with_at))
    

