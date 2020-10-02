#!/usr/bin/env python
# coding: utf-8

# # Easy Peasy Pandas

# ## Pandas with jokes
# ## Content
# 
# * **Reading the data - *First thing first***
# * **Selecting specific columns and slicing - *getting hands on the beast***
# * **loc, iloc - *My Favourite***
# * **Conditioning - *The coolest feature***
# * **Statistics - *The Queen***
# * **Saving the data - *Final Task***

# ### Before Getting Started
# #### If you are running on local machine and doesn't have Pandas installed, it can be installed with the command:
# * Enter following command in terminal:
# * *pip install pandas*
# * It should install the required pandas library.
# * Note - If pip is by default pointed to python2, use pip3 instead.
# 
# Do as I say, I promise the fun ahead! :)

# ### If you find it useful please hit the upvote button and comment if you want to suggest anything
# ### For Numpy Tutorial Notebook follow the link: [Numpy with jokes](https://www.kaggle.com/mohtashimnawaz/numpy-with-jokes-and-funs/comments)

# In[ ]:


# importing the library
import pandas as pd   
# importing as 'pd' makes it easier to refer as we can use 'pd' instead of whole 'pandas' 


# #### For the purpose of illustration we will use the most basic Titanic dataset on kaggle.
# #### We shall see almost all basic functions of pandas on the dataset. However, Since Pandas is large, not everything is covered in the notebook.
# #### You are encouraged to visit the documentation, it has everything about pandas.
# #### Pandas documentation : [Pandas Documentation](https://pandas.pydata.org/docs/)
# 
# #### Titanic - 'Even Dead I Am The Hero' :) :(

# ### Reading the data
# ##### As they say - reading is key to success. (They = Me in here xD)

# In[ ]:


# Pandas read_csv(filepath) function is used to read the csv file
# Excel files can be read using read_excel() function.
# I have already added the data to this notebook (See top right corner)

# Titanic dataset contains two data files, train.csv, test.csv. We shall import them seperately.
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
print("Pandas - We have read the whole data hooman, within fraction of seconds, you slow snail. Haha..")


# In[ ]:


# We can print the first few and last few rows of the dataset using pandas head(number_of_rows), tail(number_of_rows) function.
# By default, number_of_rows = 5
# Lets print first 3 rows of train and last 5 rows of test dataset 


# In[ ]:


print("Me - I'll go for you head Pandas")
train.head(3)


# In[ ]:


print("Me - ...and then for you tail... , I am the mighty Hooman")
test.tail()


# In[ ]:


# We can get the shape of each dataset as follows:
print("Shape of training data",train.shape)
print("Shape of test data", test.shape)


# In[ ]:


# Its clear from above that test data has one column less which is the ground truth column.
# To get a list of column names, we do as follows:
print("Train set columns:", train.columns)
print("Test set columns:", test.columns)


# ### Selecting specific columns and slicing
# #### They say, Playing with data makes you expert in data exploration... (As usual, They=Me xD)

# In[ ]:


# selecting a particular column by its name
train['Name']


# In[ ]:


# Due to space limitations, only a few columns are shown
# This can also be done in a different way as given below:
train.Name


# In[ ]:


# Slicing is one of the most important concept
# To select rows from 5-10, we need to do the following:
print("Me - I summon the rows from 5 to 10, bring them to me.")
print("Pandas - At your service, Hooman")
train[5:11]
# Pandas uses last index as exclusive index so if we want rows till 10, we need to write 10+1 i.e. 11.


# In[ ]:


# This printed all the columns, but if we want a specific column, we can do the following:
train['Name'][0:5]


# In[ ]:


# To select multiple columns:
train[['Name', 'Age', 'Sex']]


# ##### There is a subtle difference in operations involving a single column and the last operation involving multiple columns. Last operation returns a DataFrame while other operations return a Series. If you want the single column operation to return DataFrame, you can do like as follows:
# * train[['Name']]
# 
# ##### Series and DataFrame are pandas specific Data Structures. For more details refer to documentation. 
# ##### Oh Pandas, you neccessary evil!!!

# In[ ]:


# Illustration
print(type(train[['Name']]))
print(type(train['Name']))


# ### loc, iloc
# #### Most powerful pandas features
# #### They say - With great power comes great responsibility (They = Spiderman....Surprised? Haha..)

# In[ ]:


# Pandas provides different ways to deal with columns and rows. loc, iloc are two such very powerful ways.
# loc - used with index names
# iloc - used with index numbers

print("Me - Pandas, show me the magic...")
print("Le Pandas...")
print()
print(train.iloc[5]) # Returns specific row
print(type(train.iloc[5]))


# In[ ]:


# To get multiple rows
print("Type: ",type(train.iloc[1:5])) # 5 is exclusive i.e. rows from 1-4 will be displayed
train.iloc[1:5]


# In[ ]:


# if we want a specific row and a specific column
train.iloc[2,1]


# In[ ]:


# Although most of the time we don't need an iterartor, pandas provides a method, df.iterrow()
# This can be used to iterate the dataframe


# In[ ]:


# Counterpart of iloc is loc, which allows using index names
train.loc[2]
# Note that 2 is being interpreted as the index name not as index number


# In[ ]:


# There is a pandas method which provides deatiled analysis of DataFrame.
print("Me - Pandas...Is it everything?")
print("Pandas - Don't under estimate the power of Pandas, you hooman...")
train.describe()


# In[ ]:


# We can sort the values using df.sort_values() method
train.sort_values('Age')
# We can provide extra arguement ascending = False if we want to sort in descending order. By default ascending =True


# In[ ]:


# Or if we want to sort using multiple columns, we can do as follows:
train.sort_values(['Age', 'Name'], ascending=[1,0])
# 1- ascending for Age, 0 - Descending for Name


# ### Conditioning - The Coolest Feature
# #### They say - ....... (They=Me....I am not a chatterbox man, xD!!!)

# In[ ]:


# Let's say we want to select Pclass = 1 passengers...We wanna see how many rich kids
# But before that, let's combine the test and train data into one...
total = train.append(test) # There are other ways too, nbut I find it simple


# In[ ]:


# Let's see the shape of total
print(total.shape)
print("Yes! We have appended the data successfully")


# In[ ]:


# Now let' find out the rich kids...:)
total[total['Pclass']==1]


# In[ ]:


# Ye! We have all the rich kids!!!
# Pandas - But wait, kids?, 58 years old kid? On what planet Hooman?
# So let's find out real kids, lets say of age less than 16


# In[ ]:


print("Rich kids:")
print()
total[(total['Pclass']==1) & (total['Age']<16)]  # Don't miss the paranthesis, mighty Pandas warns you Hooman


# In[ ]:


# Before we move further, I, as a responsible creater of this notebook, wanna give you a bonus...:)
print("Finding count of null values in all columns....")
total.isnull().sum()


# In[ ]:


# One more bonus - We can find unique values and count of each for each column...
total['Pclass'].value_counts()


# In[ ]:


# Now that we have got a lot of things... I wanna put these rich kids to a seprate class of Pclass=0
# Note that no such class actually exists... We shall make a new class for these rich kids..:(
# To do this, we will summon the mighty loc method...I told you its powerful...
print(type(total))
total.loc[(total['Pclass']==1) & (total['Age']<16), 'Pclass']=0
print("I hope it does the task :(")


# In[ ]:


# Let's check if we did what we wanted...
print("New Pclass should be zero")
total[(total['Pclass']==0) & (total['Age']<16)]


# In[ ]:


print("See we added the new Pclass that was not before")
print("To confirm: I summon the bonus I gave to you:::)))")
total['Pclass'].value_counts()


# In[ ]:


# At last, I wanna combine Parch and SibSp to Family columns as Siblings and Parent/Children are a part of family
# Creaters of Dataset - Yes, we know but we want you to work more :(((
total['Family'] = total['Parch']+total['SibSp']
total.head()


# In[ ]:


# As you can see at the end there is a column named 'Family'...Hurray.. We united the family...


# ### Statictics - Helping hands for a Data Scientist
# #### They say - Mathemetics is queen of all sciences (They = A famous mathemetician...)

# In[ ]:


# Let's find the mean and median of Age
print("Mean of Age:", total['Age'].mean())
print("Median of Age:",total['Age'].median())


# In[ ]:


# Let's find out max, min of age -- :p...I wanna know the oldest grandpa out there in titanic and youngest seet child
print("Minimum Age:",total['Age'].min())
print("Maximum Age:",total['Age'].max())


# In[ ]:


# At last, a few more...
print("Sum of all age (IDK why I am finding it): ", total['Age'].sum())


# In[ ]:


# Aggregation is yet another powerful weapon of Pandas...
print("It gives mean of ages grouped by Pclass:")
total.groupby(['Pclass']).mean()


# ### Saving the data - *The final task*

# In[ ]:


# Now at last, we want to save the data... But wait another bonus... We want to drop some columns first
# Lets drop SibSp and Parch as they are combined into family already...
total.drop(['Parch','SibSp'],inplace = True, axis=1)
# inplace = True modofies the 'total' inplace. axis = 1 specifies the operation is column-wise
total.head()


# In[ ]:


# As you can see there is no SibSp and Parch...
# Now let's save this to a new file...
total.to_csv('modified.csv')


# #### We have seen a lot about Pandas and it is enough to get started and start working with pandas...
# #### However, as alrady said pandas is huge, so refer to the documentation if you want to explore more
# #### I hope you enjoyed learning pandas. Please Hit the upvote button and share and comment.
# # Thank you
