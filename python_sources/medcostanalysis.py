#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pyforest')


# Let's begin our project by importing the libraries

# In[ ]:


from pyforest import *
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')


# Now, let's upload the downloaded csv file 'insurance.csv'

# In[ ]:


data = pd.read_csv('../input/insurance.csv')
data.head()


# So, let's have a look at the summary of our dataset.

# In[ ]:


data.describe()


# Now, we shall have a look at the attributes data types, memory usage, null and non-null features,etc

# In[ ]:


data.info()


# Check if there is any null data or not.

# In[ ]:


data.isnull().sum()


# After observing, that there are no null values we should have a look at our dataset's shape

# In[ ]:


data.shape


# We should have a look at the duplicates

# In[ ]:


data.duplicated().sum() 


# Sorting the values of age attribute to see the min and max age. You can also try data.age.min() and data.age.max()

# In[ ]:


data.sort_values(['age'])


# Let's see how many rows has southeast region. Accordingly, you can try with all regions.

# In[ ]:


data[data.region == 'southeast']


# Let's have a look at the mean() of the data

# In[ ]:


data.mean()


# Now, we should have a look at the total memory usage by all features  

# In[ ]:


data.info(memory_usage='deep')


# Accordingly, we can also have a look at which attribute uses how much bytes 

# In[ ]:


data.memory_usage(deep=True)


# As we can see that the usage of memory is very high, we need to convert all the categorical columns into numeric by LabelEncoding.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(data.sex.drop_duplicates())
data.sex = le.transform(data.sex)
le.fit(data.smoker.drop_duplicates())
data.smoker = le.transform(data.smoker)
le.fit(data.region.drop_duplicates())
data.region = le.transform(data.region)


# Now, we can have a look if the usage of memory has been decreased or not

# In[ ]:


data.memory_usage(deep=True)


# It can be clearly observed that there has been some decline in the amount of memory usage. Now, let's generate heatmap to see the correlation between the features.

# In[ ]:


f, ax = plt.subplots(figsize=(10,8))
corr = data.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(240, 10, as_cmap=True), square=True, ax=ax)


# Now, let's perform EDA by generating the plot of charges for smokers and non-smokers.

# In[ ]:


f = plt.figure(figsize=(12,5))
ax = f.add_subplot(121)
sns.distplot(data[(data.smoker == 1)]['charges'], color='c', ax=ax)
ax.set_title('Distribution of charges for smokers')

ax = f.add_subplot(122)
sns.distplot(data[(data.smoker == 0)]['charges'], color='b', ax=ax)
ax.set_title('Distribution of charges for non-smokers')


# We can also have a look at the total number of smokers and non-smokers.

# In[ ]:


data.smoker.value_counts() #0 means non-smokers and 1 means smokers


# We will generate the countplot of smokers vs sex to see the number of smokers and non-smokers.

# In[ ]:


sns.countplot(x='smoker', hue='sex', palette='pink', data=data)


# Similarly, we can also generate the plot to see the charges of smokers and non-smokers.

# In[ ]:


sns.catplot(x='sex', y='charges', kind='violin', hue='smoker', palette='magma', data=data)


# We will generate the boxplot of smokers and non-smokers vs the charges of women.

# In[ ]:


plt.figure(figsize=(12, 5))
plt.title('Box plot for charges of women')
sns.boxplot(x='smoker', y='charges', data=data[(data.sex == 1)])


# We will generate the boxplot of smokers and non-smokers vs the charges of men.

# In[ ]:


plt.figure(figsize=(12, 5))
plt.title('Box plot for charges of men')
sns.boxplot(x='smoker', y='charges', data=data[(data.sex == 0)])


# Generating the plot of distribution of age.

# In[ ]:


plt.figure(figsize=(12, 5))
plt.title('Distribution of Age')
ax = sns.distplot(data['age'], color='b')


# Suppose, we want to observe whole dataset, then we need to write the below code.

# In[ ]:


pd.set_option('display.max_rows', None)


# By writing the above code, we will get the data of all the rows and columns.

# In[ ]:


data


# Let's see the children who are of 18 years and who smoke.

# In[ ]:


data.loc[(data.age == 18) & (data.smoker == 1)]


# Now, we can see that all the rows are in numeric.

# In[ ]:


data.dtypes


# Let's look at the mean, minimum and maximum of the charges.

# In[ ]:


data.charges.mean()


# In[ ]:


data.charges.min()


# In[ ]:


data.charges.max()


# We can see the data of charges > than mean and one who smokes. 

# In[ ]:


data.loc[(data.charges >= 13270) & (data.smoker == 1)]


# We can also observe who has the maximum charge.

# In[ ]:


data.loc[(data.charges == 63770.42801), :]


# We can also get the data about the sex who smokes. (male and female)

# In[ ]:


data.loc[(data.sex == 0) & (data.smoker == 1)]


# In addition, we can also get the data of smokers according to the particular region. 

# In[ ]:


data.loc[(data.smoker == 1) & (data.region == 0)]


# By generating joint plot we can see the charges distribution and age for smokers and non-smokers

# In[ ]:


g = sns.jointplot(x = 'age', y = 'charges', data=data[(data.smoker == 0)], kind = 'kde', color='b')
g.plot_joint(plt.scatter, c='w', s=30, linewidth=1, marker='+')
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels('$X$', '$Y$')
ax.set_title('Distribution of charges and age for non-smokers')


# Linear plot for smokers and non-smokers.

# In[ ]:


sns.lmplot(x='age',y='charges', hue='smoker', data=data, palette='inferno_r', size=7)
plt.title('Smokers and Non-Smokers')


# Distribution of BMI.

# In[ ]:


plt.figure(figsize=(12,5))
plt.title('Distribution of BMI')
ax = sns.distplot(data['bmi'], color = 'y')


# Calculating the BMI of underweight persons.

# In[ ]:


data.loc[(data.bmi <=18.5)]   #Here, <=18.5 means the person is under weight.


# In[ ]:


data[data.bmi >= 30]   #If the BMI exceeds the value of 30 means, person has obesity.


# Let's check which person of age 18 has obesity.

# In[ ]:


data.loc[(data.bmi >= 30) & (data.age == 18)] 


# Similarly, we can also get the output of person of age 18 who are underweight.

# In[ ]:


data.loc[(data.bmi <= 18.5) & (data.age == 18)] 


# Applying machine learning algorithms to see the results.

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = data.iloc[:,:6].values
y = data.iloc[:,6].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(regressor.score(X_test, y_test))


# In[ ]:


X1 = data.iloc[:,[0,1,2,3,4]].values
y1 = data.iloc[:,6].values
X_train, X_test,y_train, y_test = train_test_split(X1, y1, test_size=0.25)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(regressor.score(X_test, y_test))


# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.25)
lin_reg = LinearRegression()
lin_reg = lin_reg.fit(X_train, y_train)
print(lin_reg.score(X_test, y_test))

