#!/usr/bin/env python
# coding: utf-8

# ## Exploratory Data Analysis of Iris Flower Dataset:
# ### Objective
# * classify the species of iris flower it is based on the available features.

# In[ ]:


from IPython.display import Image
Image(url= 'https://bit.ly/2HV74du')


# ## Mean Value Observation of Different Features Based on the Available Species:

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

train = pd.read_csv('../input/iris/Iris.csv',encoding = 'utf-8')
train.drop(['Id'],axis = 1,inplace = True)
train.head()


train.info()

train_categoury = train.select_dtypes(include = ['object'])
print('Number of Object Type Columns:',train_categoury.shape)
train_categoury.head(3)

train_Numeric = train.select_dtypes(include = ['float64'])
train_Numeric.head(3)


train.describe()

train_Avg = train.groupby('Species').SepalLengthCm.mean()
train_Avg1=pd.DataFrame(train_Avg)
train_Avg1

Y = train['Species'].unique()
y=pd.DataFrame(Y)
Y

plt.plot(Y,train_Avg,marker = 's')
plt.xlabel('Species')
plt.ylabel('Average_Sepal_Length_(Cm)')
plt.title('Species_Vs_Average_Sepal_Width_(Cm)',size = 15)


# In[ ]:


list(train_Numeric)


# In[ ]:


train_avg2 = train.groupby('Species').SepalWidthCm.mean()
train_avg2.head()


# In[ ]:


plt.plot(Y,train_avg2,marker = 's')
plt.xlabel('Species')
plt.ylabel('Average_Sepal_Width_(Cm)')
plt.title('Species_Vs_Average_Sepal_Width_(Cm)',size = 15)


# In[ ]:


train_avg3 = train.groupby('Species').PetalLengthCm.mean()


# In[ ]:


plt.plot(Y,train_avg3,marker = 's')
plt.xlabel('Species')
plt.ylabel('Average_Petal_Length_(Cm)')
plt.title('Species_Vs_Average_Petal_Length_(Cm)',size = 15)


# In[ ]:


train_avg4 = train.groupby('Species').PetalWidthCm.mean()


# In[ ]:


plt.plot(Y,train_avg4,marker = 's')
plt.xlabel('Species')
plt.ylabel('Average_Petal_Width_(Cm)')
plt.title('Species_Vs_Average_Petal_Width_(Cm)',size = 15)


# In[ ]:


plt.figure(figsize = (12,8))
sns.set_style('whitegrid')
plt.plot(Y,train_Avg,marker = 's')
plt.plot(Y,train_avg2,marker = 's')
plt.plot(Y,train_avg3,marker = 's')
plt.plot(Y,train_avg4,marker = 's')
plt.xlabel('Species',size = 15)
plt.ylabel('Ave_of_numeric_variables',size = 15)
plt.title(' Species Vs Numeric_variables',size = 15)
plt.legend()


# ## Conclutions:

# ### 1. Petal Width:
# ##### a) The average value for the petal width was lowest for all numeric variables.
# ##### b) Setosa has the lowest mean in all species and Verginica has the highest mean value in Petal width
# ##### c) It looks like petal width and sepal length is equally and sepal length is equally increasing both looks parallel to each other
# ### 2. Sepal Length:
# ##### a) The value of the mean of Sepal Length for all Species is highest. Do, you can separate the sepals from petals by looking at the length of sepals.
# ##### b)The mean value of sepal length for setosa is the lowest among all species
# ### 3. Sepal Width:
# ##### a)Mean of sepal width for setosa is the highest among the others, then comes virginica then versicolor.
# ### 4. Petal Length:
# ##### a) Mean of petal length: Setosa < versicolor < Virginica
# 
# * At the end we can conclude that in most of the cases like sepal length, petal length, and petal width, the mean value for those three is the lowest for setosa, so you can assume that it is smallest among them as virginica is the highest, so it is the biggest among the other two.

# In[ ]:


train.head()


# # Pair-Plot:

# In[ ]:


plt.close()
sns.set_style('whitegrid')
sns.pairplot(train,hue = 'Species',size = 3)
plt.show()


# ## Conclution: 
# - As we can't visualize 4D data, so we need a medium to visualize 4 features like SL(Sepal Length), SW(Sepal Width), PL(Petal Length) and PW (Petal Width) altogether and it is done by using pair-plots. As the name suggests, we make a pair of plots. Since we have four variables and we are picking two of them to generate a plot. So, we can have 4C2 = 6 combinations which we can observe from the upper triangle or the lower triangle of the plot.
#     
# **Itis a kind of matrix of plots**
# ### Lets Observe:  
# >Ignare Diagonal elements and look at non-diagonal element
# 
# 1.  PL and PW are able to separate flowers much better than my SL and SW.(element no:4x4)
# 2. So, we can write a simple if-else statement to classify any flower. like- if PL<= 2 & PW <=1, then we can classify that flower as Setosa.
# ### Limitation of Pair-plots:
#  - Pair-plot is easy to understand when the dimensionality of the data is small and here it is 4.

# 

# In[ ]:


plt.rcParams['figure.figsize']=(12,9)


# # Ploting PDFs:

# In[ ]:





for i in list(train_Numeric):
    sns.distplot(train[i],bins = 23,hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 2}, color = 'darkblue',kde=True,hist=True)
    plt.xlabel(i,size = 15)
    plt.ylabel('percentage of distribution',size = 15)
    plt.title('histogram of '+ i,size = 15)
    plt.xticks(np.arange(0,10,1))
    plt.figure(figsize=(20,18))
    plt.show()


# ## Conclutions:
# We have plotted the PDFs of different continuous random variables like SL, SW, PL, and PW.
# #### PDF(Probablity Density Function):
#  In probability theory, a probability density function, or density of a continuous random variable, is a function whose value at any given sample in the sample space can be interpreted as providing a relative likelihood that the value of the random variable would equal that sample. It is the measure of the percentage of distribution for a certain range of values.
# 
# 1. percentage of the distribution having SL OF 5 - 6.5 value is the highest which is approximately 0.4. So, most of the points are around 5 - 6.5.
# 2. From the 2nd plot, we can observe that the distribution has taken the shape of a bell, which implies that the distribution is a Normal Distribution where the mean value is almost near to 3. From 2nd PDF we can conclude that almost 90 percent of the distribution is near about 3.
# 3. As PL of Setosa is low and for verginica and Versicolor is high so the distribution is divided into two. We can easily observe the difference between the values of PL of setosa and other classes of Iris flower.
# 4. For PW everything goes the same as PL.

# In[ ]:


train.SepalWidthCm.std()


# # Box-Plots:

# In[ ]:


#plt.figure(figsize= (8,6))
for i in train_Numeric:
    plt.figure(figsize= (8,6))
    a = sns.boxplot(x='Species',y = i ,data = train)
    plt.title('Finding number of outliers in ' + i,size = 15)
    a.set_xlabel('Species',fontsize = 15)
    a.set_ylabel(i,fontsize=15)
    plt.show()


# ## Conclutions:
# #### Advantages of Box-Plots:
# By visualizing any data using box-plots we can understand how much outlier is there, where the 25th,50th and 75th percentile values of the variable lies, what is the min and max values of the data and also if we put a threshold then how much the error will be in that prediction, what we can't get from plotting PDFs.
# - Iris-Virginica is having one outlier with respect to the SL, two outliers with respect to PW. With respect to PL, setosa is having 3 outliers and Versicolor is having one outlier. From the 4th plot, we can say Setosa in having two outliers w.r.t PW.

# In[ ]:





# # Heatmap and Corelation:

# In[ ]:


from string import ascii_letters

corr = train.corr()

mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Fig:1',size=15)


# 1. PL and PW are positively correlated 
# 2. SL and PW are positively correlated
# 3. PL and SL are positively correlated
# 4. SW and PL are negatively correlated
# 5. SL and PW are negatively correlated
# 6. SW and SL are almost not correlated because that color value is near to zero. 
# 

# In[ ]:


sns.heatmap(corr,cmap = 'Blues',square=True)
plt.title('Fig:2',size=20)


# ## conclution:
# Fig:1 and Fig:2 are the same,but the Fig:1 showes the essential parts of the pot.We know in pearson corelation the value of covariance gets converted between 1 to -1.If the value of pearson corelation is near to 1 then the corelation is +ve and if it is near to -1 then it is -ve corelation.
# 
# 1. PL and PW are positively corelated 
# 2. SL and PW are positively corelated
# 3. PL and SL are positively corelated
# 4. SW and PL are negatively corelater
# 5. SL and PW are negatively corelater
# 6. SW and SL are almost not corelated because that color value is near to zero. 
# 

# In[ ]:





# # Ploting CDFs:

# In[ ]:


for i in train_Numeric:
    
    x = np.sort(train[i])
    y = np.arange(1 , len(x)+1) / len(x)
    plt.plot(x,y,marker='.',linestyle='none')
    #plt.margins(0.02)
    plt.legend()
    plt.show()


# In[ ]:


list(train_Numeric)


# In[ ]:


plt.figure(figsize=(8,6))

x_SepalLengthCm = np.sort(train['SepalLengthCm'])
x_SepalWidthCm = np.sort(train['SepalWidthCm'])
x_PetalLengthCm = np.sort(train['PetalLengthCm'])
x_PetalWidthCm = np.sort(train['PetalWidthCm'])

y1 = np.arange(1 , len(x_SepalLengthCm)+1) / len(x_SepalLengthCm)
y2 = np.arange(1 , len(x_SepalWidthCm)+1) / len(x_SepalWidthCm)
y3 = np.arange(1 , len(x_PetalLengthCm)+1) / len(x_PetalLengthCm)
y4 = np.arange(1 , len(x_PetalWidthCm)+1) / len(x_PetalWidthCm)

plt.plot(x_SepalLengthCm,y1,marker='.',linestyle='none',label='SepalLengthCm')
plt.plot(x_SepalWidthCm,y2,marker='.',linestyle='none',label='SepalWidthCm')
plt.plot(x_PetalLengthCm,y3,marker='.',linestyle='none',label='PetalLengthCm')
plt.plot(x_PetalWidthCm,y4,marker='.',linestyle='none',label='PetalWidthCm')

plt.margins(0.02)
plt.legend(loc=4,fontsize='large')
plt.show()


# ## Conclutions:
# In probability theory and statistics, the cumulative distribution function of a real-valued random variable, or just distribution function of, evaluated at, is the probability that will take a value less than or equal to.
# Or, we can say for value 'x' of random variable X probability of that variable having less than x or equal to x.
# ![CDF](https://bit.ly/2Tbc9DN)
# - From the plot, we can easily find out what is the 50th,25th or 75th percentile.
# - As the value of PL and PW for setosa is very small and as there is no intersection between the value of PL AND PW for virginica and Versicolor so we are having the discrete CDF plots for PL and PW. 
# 
# 

# ## End Conclution:
# Based on the plots we can understand that the four random variables are unique and their values define every Species directly. 
# By performing this EDA we are able to conclude that without using any ML algorithm we can classify any flower based on their corresponding values except that we can understand where the central data points are lying, we can understand how the distribution is, what is the 25th,50th and 75th percentile values of each features, how much outlier is present and how much correlated the features are.
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




