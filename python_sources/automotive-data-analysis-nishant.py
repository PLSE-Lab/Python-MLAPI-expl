#!/usr/bin/env python
# coding: utf-8

# # Automotive Data Analysis
# 
# #### Nishant Madu
# 

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
auto=pd.read_csv('../input/Automobile_data.txt')
auto


# ## Cleaning the Garbage Values-
# ####                                                 -Replace all the garbage values to NaN by using the 'replace' method.
# ####                                                -Once the method is applied all the garbage values will be replaced by NaN

# In[3]:


automobile=auto.replace('?', np.nan)
automobile


# ## 'isnull' method-
# #### 'isnull' method helps in displayiing all the NaN values to true.The methos name is self explanatory.

# In[4]:


automobile.isnull()


# In[5]:


automobile


# #### 'dtypes' helps us to understand the datatypes present in the dataframe 

# In[6]:


automobile.dtypes


# #### Below is where we are manipulating the datatypes.Here the datatype is changed from one type to other.In this case the 'Object' datatype is convereted to numeric.

# In[7]:


automobile['bore']=pd.to_numeric(automobile['bore'],errors='coerce')
automobile['stroke']=pd.to_numeric(automobile['stroke'],errors='coerce')
automobile['horsepower']=pd.to_numeric(automobile['horsepower'],errors='coerce')
automobile['peak-rpm']=pd.to_numeric(automobile['peak-rpm'],errors='coerce')
automobile['price']=pd.to_numeric(automobile['price'],errors='coerce')


# In[8]:


automobile.dtypes


# ## Univariate Analysis
# 
# ### Univariate Analysis on 'Make'

# In[9]:


automobile.make.value_counts().plot(figsize=(15,8),kind='bar',stacked=True,colormap='bone')
plt.xlabel('Make')
plt.ylabel('Count')


# ### Univariate Analysis on 'no-of-cylinders'

# In[10]:


automobile['num-of-cylinders'].value_counts().plot(figsize=(15,8),kind='bar',stacked=True,colormap='bone')
plt.xlabel('num-of-cylinders')
plt.ylabel('Count')


# ### Univariate Analysis on 'Fuel System'

# In[11]:


automobile['fuel-system'].value_counts().plot(figsize=(15,8),kind='barh',stacked=True,colormap='bone')
plt.xlabel('Count')
plt.ylabel('fuel-system')


# ### Univaraite Analysis on 'Drive Wheels'

# In[12]:


automobile['drive-wheels'].value_counts().plot(figsize=(15,8),kind='bar',stacked=True,colormap='bone')
plt.xlabel('drive-wheels')
plt.ylabel('Count')


# #### Plotting 'Histogram' and also calculating the mean from the data pertaining to 'Price of Cars'. 

# In[13]:


import numpy as n
import matplotlib.pyplot as plt


# This is  the colormap I'd like to use.
cm = plt.cm.get_cmap('RdYlBu_r')

# Get the histogramp
Y,X = n.histogram(automobile['price'], range=(5000,23000),bins=5)
plt.xlabel("Price of Cars")
plt.ylabel("Frequency")
plt.title("Average Price of Cars")
plt.axvline(x=automobile['price'].mean(),linewidth=3,color='r')
x_span = X.max()-X.min()
C = [cm(((x-X.min())/x_span)) for x in X]

plt.bar(X[:-1],Y,color=C,width=X[1]-X[0])
plt.show()


# ### The Average 'MPG in City' is calculated using the Histogram

# In[14]:


import numpy as n
import matplotlib.pyplot as plt

# This is  the colormap I'd like to use.
cm = plt.cm.get_cmap('YlGnBu')

# Get the histogramp
Y,X = n.histogram(automobile['city-mpg'], range=(15,48),bins=5)
plt.xlabel("Number Of Miles in City")
plt.ylabel("Frequency")
plt.title("Average Miles Per Gallon in City")
plt.axvline(x=automobile['city-mpg'].mean(),linewidth=3,color='r')
x_span = X.max()-X.min()
C = [cm(((x-X.min())/x_span)) for x in X]

plt.bar(X[:-1],Y,color=C,width=X[1]-X[0])
plt.show()


# ### The Average 'MPG on Highway' is calculated using the Histogram

# In[15]:


import numpy as n
import matplotlib.pyplot as plt

# This is  the colormap I'd like to use.
cm = plt.cm.get_cmap('RdYlGn')

# Get the histogramp
Y,X = n.histogram(automobile['highway-mpg'], range=(19,48),bins=5)
plt.xlabel("Number Of Miles on Highway")
plt.ylabel("Frequency")
plt.title("Average Miles Per Gallon on Highway")
plt.axvline(x=automobile['highway-mpg'].mean(),linewidth=3,color='r')
x_span = X.max()-X.min()
C = [cm(((x-X.min())/x_span)) for x in X]

plt.bar(X[:-1],Y,color=C,width=X[1]-X[0])
plt.show()


# # Performance Criteria 
# ### *A few variables add up to the Performance of a car and also, how it compares with the actual meaning of how the performance variables are taken.Below are few which are taken into Performance Criteria Variables/factors.
# #### 1. Bore Vs Stroke Ratio
# #### 2. Bore/Stroke Ratio Vs Horsepower
# #### 3. Oversquared Vs Undersquared Engines
# #### 4. RPM Vs Torque

# ## 1. In the below dataset we are creating a new column named 'bore_stroke_ratio' for the ratio between Bore and Stroke.

# In[16]:


automobile['bore_stroke_ratio'] = automobile['bore']/automobile['stroke']
automobile


# # Note - The Bore/Stroke gives us how much Power/Torque a car has.
# ## - If the Ratio is more than 1 it is said to be having more 'RPM'(Powerful).
# 
# ## - If the Ratio is less than 1 it is said to be having more Torque.

# # 2. But not Bore/Stroke alone can be taken to judge a car's performance. It should also match with BHP(Break Horse Power) and Vice-Versa. 
# 
# ### Below is where we plot a graph between Bore/Stroke Ratio Vs Horsepower to see at what interval the cars in the dataset has an equally good Bore/Stroke and higher BHP.

# In[17]:


import numpy as np
import matplotlib.pyplot as plt

plt.scatter(automobile['bore_stroke_ratio'],automobile['horsepower'],color='blue')
plt.title('Permormance Criteria')
plt.xlabel('Bore : Stroke Ratio')
plt.ylabel('Horsepower')
plt.show()


# 
# 

# In[18]:


sns.regplot(x='bore_stroke_ratio', y='horsepower', data=automobile)


# # 3. The concept of 'Oversquared' and 'Undersquared' Engines are nothing but the engines which have greater than 1 Ratio is called 'Oversquared' and engines which have less than 1 Ratio is called 'Undersquared'.
# 
# ### Below is where we see the number of 'Oversquared' and 'Undersquared' engines we have from the provided dataset and the average of it.

# #### *  Below is where we plot and see how many Oversquared Engines are present and the average of it.

# In[19]:


oversquared = automobile[automobile['bore_stroke_ratio']>1]


# In[20]:


import numpy as n
import matplotlib.pyplot as plt

# This is  the colormap I'd like to use.
cm = plt.cm.get_cmap('PuRd')

# Get the histogramp
Y,X = n.histogram(oversquared['bore_stroke_ratio'], range=(1,1.6),bins=5)
plt.xlabel("Oversquared Engines Range")
plt.ylabel("Frequency")
plt.title("Average of Oversquared Engines/Engines which are more Powerful")
plt.axvline(x=oversquared['bore_stroke_ratio'].mean(),linewidth=3,color='r')
x_span = X.max()-X.min()
C = [cm(((x-X.min())/x_span)) for x in X]

plt.bar(X[:-1],Y,color=C,width=X[1]-X[0])
plt.show()


# #### The number of  'Oversquared' Engines are 90.

# In[21]:


oversquared.shape


# #### *  Below is where we plot and see how many Underaquared Engines are present and the average of it.

# In[22]:


undersquared = automobile[automobile['bore_stroke_ratio']<1]


# In[23]:


import numpy as n
import matplotlib.pyplot as plt

# This is  the colormap I'd like to use.
cm = plt.cm.get_cmap('GnBu')

# Get the histogramp
Y,X = n.histogram(undersquared['bore_stroke_ratio'], range=(0.85,0.99),bins=5)
plt.xlabel("Undersquared Engines Range")
plt.ylabel("Frequency")
plt.title("Average of Undersquared Engines/Engines which have more Torque")
plt.axvline(x=undersquared['bore_stroke_ratio'].mean(),linewidth=3,color='r')
x_span = X.max()-X.min()
C = [cm(((x-X.min())/x_span)) for x in X]

plt.bar(X[:-1],Y,color=C,width=X[1]-X[0])
plt.show()


# #### The number of  'Undersquared' Engines are 105.

# In[24]:


undersquared.shape


# In[25]:


automobile['oversquared']=automobile['bore_stroke_ratio']>1


# In[26]:


automobile['undersquared']=automobile['bore_stroke_ratio']<1


# In[ ]:





# In[ ]:




