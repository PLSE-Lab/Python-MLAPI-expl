#!/usr/bin/env python
# coding: utf-8

# # 1.1Getting the data set

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
data=pd.read_csv("haberman.csv")
print(data.shape)
print(data.columns)


# In[ ]:


# as in our data set column names are not given therfore we give nmaes for the columns
#Attribute Information:
#Age of patient at time of operation (numerical)
#Patient's year of operation (year - 1900, numerical)
#Number of positive axillary nodes detected (numerical)
#Survival status (class attribute) 1 = the patient survived 5 years or longer 2 = the patient died within 5 year
#we are taking Number of positive auxillary nodes detected as pan
colum=['age','year','pan','survs']
#loading the data set again
data=pd.read_csv("haberman.csv",names=colum)
print(data.columns)
data["survs"].value_counts()


# # Plotting 2D  Scatter Plot

# In[ ]:


# This is an unbalanced dataset 
# No. of data points for survival is 81 and for non survival is 225 
data.plot(kind ='scatter',x='survs',y='pan');
plt.show()
#as we have only 2 points 1(survived for 5 or longer years) and 2(died within 5 years)
#we cannot make much sense from th graph
#remember pan is positive auxillary nodes detected


# In[ ]:


#we use seaborn library for color coding graph
#always add legand to graph 
sns.set_style("whitegrid");
sns.FacetGrid(data, hue="survs", size=4)    .map(plt.scatter, "pan", "age")    .add_legend();
plt.show();
#its also a complex graph so we cannot find some near by point also as these are just overlapping each other


# Observation:
# 1.we plot a simple 2d scatter plot but its hard to distinguish between who survived and who doesnt.Since it is imbalanced dataset
# 2.applying color coding also doesn't gives us satisfactory results  

# # 3D Pair Plots 

# In[ ]:


#These are just pair wise sactter plots that we have seen earlier
sns.set_style("whitegrid");
sns.pairplot(data,hue="survs",size=3);
plt.show()
#The Diagnol elements are PDF(probabilty density function) we not considering them for now 


# In[ ]:





# OBERVATION:
# we still cannot make any aprroximation from 3d pair plots also as the plots is extremly overlapping

# # Histogram &PDF

# In[ ]:


#we use seaborn library for histogram and pdf
sns.FacetGrid(data,hue="survs",size=4)   .map(sns.distplot,"pan")    .add_legend();
plt.show();
#the orange and blue lines are called PDF


# In[ ]:


sns.FacetGrid(data,hue="survs",size=4)   .map(sns.distplot,"year")    .add_legend();
plt.show();


# In[ ]:


sns.FacetGrid(data,hue="survs",size=4)   .map(sns.distplot,"age")    .add_legend();
plt.show();
    


# OBSERVATION:
# 1.we observe that in the above graphs pan(positive axillary nodes detected) is the best way to get some outcomes
# 2.Other plots are highly overlapped

# # CDF

# In[ ]:


#cdf are cummulative density frequency 
#for plotting the graph we use surv as 1(more than 5 year years of age) & nsurv as 2(less than 5 years) 
surv = data[data['survs'] == 1]
nsurv = data[data['survs'] == 2]
counts, bin_edges = np.histogram(surv['pan'], bins=5,density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.show();


# In[ ]:


counts, bin_edges = np.histogram(surv['pan'], bins=5,density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)


counts, bin_edges=np.histogram(nsurv['pan'],bins=5,density=True)
pdf=counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
sns.FacerGrid
    .add_legend();
plt.show();


# OBSERVATION:
# we can observe that these are (line seperable) at point 
# aprroximately our 90% answers will be correct at point 10

# # Box Plot & Whiskers

# In[ ]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
colum=['age','year','pan','survs']
#loading the data set again
data=pd.read_csv("haberman.csv",names=colum)
sns.boxplot(x='survs',y='pan',data=data)
plt.show()


# # Violin Plots

# In[ ]:


import seaborn as sns
sns.violinplot(x='survs',y='pan',data=data,size=4)
plt.show()


# OBSERVATION:
# 1.with the help of box whiskers and violin plots we can seperate the persons who will live for more than fiver years and less than that with approximate 75% accuracy 
# 2. From all the data analysis we see that violin plot and box plot & whiskers are more helpful for analysing data

# In[ ]:





# In[ ]:





# In[ ]:




