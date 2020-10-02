#!/usr/bin/env python
# coding: utf-8

# ![](https://storage.googleapis.com/kagglesdsdata/datasets/479883/897149/NumPy-CheatSheet1.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1579544260&Signature=NpZx%2B3iZoTAxgpZC7xsQP1ptIVTSspqDFq9aooQY9Xlm%2F%2BEe5iz9346B5uSs1nidLwG5acu6yTO8y%2BKxruVjmeMDQITMsiiihrcPOPlyRVhYFBJJIh8QRSoPnS7gbg%2Bpr5ZsroNzegDwc43fxYttuCyqvEUke1otgNCOPfxS%2BCzkfxXs2jSPZS57xfRH%2BtpHwtgJqfRtyViCK31Vh8y8Xvw80Bw0ysr%2Bg6HHgGlshLRkYFAGf5AmTedJuM9dRtLiOo%2B3cmHJIAo48wfeaaMySh2%2BYKuSfVBy2Cp0zUJ7EoZSVTzcHWftnyBSSmUsJBe6tnmB6VKeKDKvCfva%2FQC8SQ%3D%3D)
# 
# # NUMPY
# A good understanding of NUMPY and PANDAS is must to manipulate and get the understandable features from the data

# In[ ]:


import numpy as np
import pandas as pd


# ## 1. How to create a 1D array?
# Create a 1D array starting from 0 to 9

# In[ ]:


arr = np.arange(10)
arr


# ## 2. How to replace items that satisfy a condition with another value in numpy array?
# Replace all even number from the array with -1
# 

# In[ ]:


arr[arr % 2 == 1] = -1
arr


# ## 3.  How to reshape an array?
# Convert arry from 1D (1,10) to (2,5) a 2d array

# In[ ]:


arr = np.arange(10)
arr.reshape(2, -1)  # Setting to -1 automatically decides the number of cols


# ## 4. How to generate custom sequences in numpy without hardcoding?
# To generate a pattern in array only using numpy functions

# In[ ]:


a =  np.array([2,4,6])
np.r_[np.repeat(a, 3), np.tile(a, 3)]


# ## 5. How to get the common items between two python numpy arrays?

# In[ ]:


a = np.array([1,2,3,9,3,4,3,4,5,6])
b = np.array([10,6,5,7,11,12,4,5,9,66])
np.intersect1d(a,b)


# ## 6. How to remove from one array those items that exist in another?

# In[ ]:


a = np.array([1,2,3,9,3,4,3,4,5,6])
b = np.array([10,6,5,7,11,12,4,5,9,66])
np.setdiff1d(a,b)


# ## 7. How to swap two columns in a 2d numpy array?
# create a 2D array with 3 column replace them in the order of 1,0,2

# In[ ]:


arr = np.arange(9).reshape(3,3)
arr = arr[[1,0,2], :]
arr


# ## 8. How to load Dataset from URL to NumPy

# In[ ]:


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_1d = np.genfromtxt(url, delimiter=',', dtype=None)
iris_1d


# ## 9. How to find medium mean and standard Deviation form an array?
# Find Medium,mean and SD of petal_length

# In[ ]:


petal_length = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[2]) #usecols is number of coloumn
mu, med, sd = np.mean(petal_length), np.median(petal_length), np.std(petal_length)
print("\nMean:",mu,"\nMedian", med,"\nStandard Deviation", sd)


# ## 10.How to normalize an array so the values range exactly between 0 and 1?
# Normalize the value array of petal_length

# In[ ]:


petal_length = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[2]) #usecols is number of coloumn
pmax, pmin = petal_length.max(), petal_length.min()

p = (petal_length - pmin)/(pmax - pmin)

p


# ## 11. How to find if a given array has any null values?
# 

# In[ ]:


iris = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
np.isnan(iris).any()


# ## 12.How to replace all missing values with 0 in a numpy array?

# In[ ]:


iris = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
iris[np.isnan(iris)] = 0
iris


# ## 12. How to convert and array to a list

# In[ ]:


iris = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
iris_list = iris.tolist()
iris_list


# ## 13. How to create dataframe from array 

# In[ ]:


data = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
iris_df = pd.DataFrame(data=data[1:,1:],    # values
            index=data[1:,0],    # 1st column as index
           columns=data[0,1:])  # 1st row as the column names
iris_df


# ## 14. How to create Image using numpy array

# In[ ]:


from PIL import Image
import numpy as np

w, h = 512, 512
data = np.zeros((h, w, 3), dtype=np.uint8)
data[0:256, 0:256] = [255, 0, 0] # red patch in upper left
img = Image.fromarray(data, 'RGB')

img


# ## 15.How to sort values of arrays according to another array?

# In[ ]:


score = np.array([70, 60, 50, 10, 90, 40, 80])
name = np.array(['Ada', 'Ben', 'Charlie', 'Danny', 'Eden', 'Fanny', 'George'])
sorted_name = name[np.argsort(score)] # an array of names in ascending order of their scores
sorted_name


# 
# ![](https://storage.googleapis.com/kagglesdsdata/datasets/479883/897149/Python-Pandas-CheatSheet.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1579544291&Signature=NMUZkcPIfOzbSLxhVI4ukjtfRDhOndqDgDiJlQsueCpS8ZtRAuzEHVi%2Fx5sMwbWULYO5quvddbH1%2Bkuo93546BGixzaIj1PnmLakZMLAbMbGE9oM8VWMT7men%2BqRT8XwchcgFadQOJc9f5rWHj56nLmaGQYdRFxuryLNqIu6ys4LBXNex0iteyMxw5p0GlAMa4Vqbgh8hERJ9FvopTDJWTvHut%2F7ErhBnk0k%2BGMKPMm1IxshBztoc7Tv5JUBpu0%2Fzj3EQ1U1S9B3OF2p5x4qlNo2BFozpUsDol76jpl%2BNlxJGVhXhAC7e8yEA3AbKHVgKvYz1WiBBrFz9IMtvJENXQ%3D%3D)
# 
# # PANDAS

# ## 1.How to create a Dataframe

# In[ ]:


df = pd.DataFrame({"col1": [1, 2, 3,2,3,4,5,6,8,7,9]})
df


# ## 2.How to add coloumn to dataframe

# In[ ]:



df["Col2"]= [10, 20, 30,20,30,40,50,60,80,70,90]
df


# ## 2.How to display only top 10 elements in dataframe

# In[ ]:


df.head(10)


# ## 2.How to display only bottom 10 elements in dataframe

# In[ ]:


df.tail(10)


# ## 3.How to load CSV?

# In[ ]:


iris_df = pd.read_csv("/kaggle/input/iris-flower-dataset/IRIS.csv")
iris_df


# ## 4.How to profile the whole Dataset?

# In[ ]:


import pandas_profiling 
titanic_df = pd.read_csv("/kaggle/input/titanic/train.csv")
titanic_df_p = titanic_df.profile_report()
titanic_df_p


# ## 5. How to change column name?

# In[ ]:


df = pd.DataFrame({"col1": [1, 2, 3,2,3,4,5,6,8,7,9] , "col2":[10, 20, 30,20,30,40,50,60,80,70,90]})
df.rename(columns={"col1": "COL1", "col2": "COL2"})


# ## 6. How to add Coloumn to Dataframe with no change in index?

# In[ ]:


df = pd.DataFrame({"col1": [1, 2, 3,2,3,4,5,6,8,7,9] , "col2":[10, 20, 30,20,30,40,50,60,80,70,90]})
df2 = pd.DataFrame({"col1":[10] , "col2":[100]})
df.append(df2)


# ## 6. How to add Coloumn to Dataframe with change in index

# In[ ]:


df = pd.DataFrame({"col1": [1, 2, 3,2,3,4,5,6,8,7,9] , "col2":[10, 20, 30,20,30,40,50,60,80,70,90]})
df2 = pd.DataFrame({"col1":[10] , "col2":[100]})
df.append(df2, ignore_index=True)


# ## 7. Create a list of dates from given start date to end date?

# In[ ]:


date_from = "2019-01-01"
date_to = "2019-01-12"
date_range = pd.date_range(date_from, date_to, freq="D")
date_range


# ## 8. Merge two dataframe

# In[ ]:


left = pd.DataFrame({"key": ["key1", "key2", "key3", "key4"], "value_l": [1, 2, 3, 4]})
left


# In[ ]:


right = pd.DataFrame({"key": ["key3", "key2", "key1", "key6"], "value_r": [3, 2, 1, 6]})
right


# In[ ]:


df_merge = left.merge(right, on='key', how='left', indicator=True)
df_merge


# ## 9. How to Filter Data by multiple categories?

# In[ ]:


iris_df = pd.read_csv("/kaggle/input/iris-flower-dataset/IRIS.csv")
filtered_iris_df = iris_df[iris_df.species.isin(['Iris-setosa','Iris-virginica'])] 
filtered_iris_df


# ## 10.Split column value to multiple columns?

# In[ ]:


df1 = pd.DataFrame({'city':['new york','mumbai','paris'] , 'temp_windspeed': [[21,4],[34,3],[26,5]]})
df1


# In[ ]:


df2 = df1.temp_windspeed.apply(pd.Series)
df2.rename(columns= {'0':'temperature','1':'windspeed'})
df2

