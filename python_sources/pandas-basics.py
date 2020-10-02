#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

haberman = pd.read_csv("../input/haberman12/haberman12.csv")
haberman


# In[ ]:


print(haberman.shape)#trying to do from start


# In[ ]:


haberman['status'].value_counts() 


# In[ ]:


# first i read the all comment below the assignment
#replacing the status with yes or no
#it really takes too much time 
#haberman.status.map(dict(1:yes, 2: no))     
#haberman['status'] = haberman['status'].map({1:"yes", 2:"no"})
haberman['status'] = haberman['status'].map({1:"yes", 2:"no"})
haberman.head(20)


# In[ ]:


haberman.columns


# In[ ]:


haberman['status'].unique()


# In[ ]:


haberman.info() #checking updated info


# #Observation
# 1. There are 4 columns of haberman dataset which has integer datatype
# 2. the class label "status" is interger datatype but now labelled as a 1 as "yes" & 2 as "no" survived and unsurvied
# 3. sometimes jupyter notebook not work properly on unbalanced dataset 

# In[ ]:


#univariate analysis
sns.FacetGrid(haberman, hue="status", size=5)    .map(sns.distplot, "age")    .add_legend();
plt.show();


# #obervation
# on X-axis 40-65 range patients are survived most 
# and seems majority of survived patients are higher

# In[ ]:


sns.FacetGrid(haberman, hue = "status" , size=5)     .map(sns.distplot, "year")     .add_legend();
plt.show();


# #obervation
# on x axis year having range 60-66 had highest rate

# In[ ]:


#PDF, CDF, BOXPLOT, VOILIN PLOT
#PDF
import numpy as np
haberman_yes = haberman.loc[haberman["status"] == "yes"];
haberman_no = haberman.loc[haberman["status"] == "no"];


plt.figure(figsize=(20,5))
plt.subplot(141)#1=no.of row , 4=no.of columns 1=fig. number
counts,bin_edges=np.histogram(haberman_yes["age"],bins=10,density=True)
pdf = counts/(sum(counts)) #formulae
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)
plt.ylabel("COUNT")
plt.xlabel('AGE')
plt.legend(['PDF-age', 'CDF-age'])
plt.title('PDF-CDF of AGE Status = YES')

plt.subplot(142)#row 1 fig no 2
counts,bin_edges=np.histogram(haberman_yes["year"],bins=10,density=True)
pdf = counts/(sum(counts)) #formulae
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)
plt.ylabel("COUNT")
plt.xlabel('year')
plt.legend(['PDF-year', 'CDF-year'])
plt.title('PDF-CDF of year Status = YES')


plt.subplot(143)#row 1 fig 3
counts,bin_edges=np.histogram(haberman_no["age"],bins=10,density=True)
pdf = counts/(sum(counts)) #formulae
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)
plt.ylabel("COUNT")
plt.xlabel('AGE')
plt.legend(['PDF-age', 'CDF-age'])
plt.title('PDF-CDF of AGE Status = NO')


plt.subplot(144)#row 1 fig 4
counts,bin_edges=np.histogram(haberman_no["year"],bins=10,density=True)
pdf = counts/(sum(counts)) #formulae
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)
plt.ylabel("COUNT")
plt.xlabel('YEAR')
plt.legend(['PDF-year', 'CDF-year'])
plt.title('PDF-CDF of YEAR Status = NO')
plt.show();


# In[ ]:


plt.figure(figsize=(20,5))
plt.subplot(131)
sns.boxplot(x='status', y='nodes', data=haberman)
plt.title("box plot of NODES(y-axis) and STATUS(x-axis)")

plt.subplot(132)
sns.boxplot(x='status', y='age', data=haberman)
plt.title("box plot of AGE(y-axis) and STATUS(x-axis)")

plt.subplot(133)
sns.boxplot(x='status',y='year',data=haberman)
plt.title("box plot of YEAR(y-axis) and STATUS(x-axis)")
plt.show()


# #Observation
# from nodes and status it seems big change of their death
# Q1-25 percentile
# Q2-50 percentile or median
# Q3-75 percentile
# and Q3 + 1.5*IQR maximum
# Q1 - 1.5*IQR minimum

# In[ ]:


plt.figure(figsize=(20,5))
plt.subplot(131)
sns.violinplot(x='status', y='nodes', data=haberman ,size=8)
plt.title("violinplot of NODES(y-axis) and STATUS(x-axis)")

plt.subplot(132)
sns.violinplot(x='status', y='age', data=haberman, size=8)
plt.title("violinplot of AGE(y-axis) and STATUS(x-axis)")

plt.subplot(133)
sns.violinplot(x='status',y='year',data=haberman, size=8)
plt.title("violinplot of YEAR(y-axis) and STATUS(x-axis)")
plt.show()


# #obervation
# it is combines the benifits of boxplot and histogram plot

# In[ ]:


#BI-variate analysis
#scatter plot
haberman.plot(kind='scatter', x='age', y='nodes') ;
plt.show()


# In[ ]:


sns.set_style("whitegrid");
sns.FacetGrid(haberman, hue="status", size=5)    .map(plt.scatter, "age", "nodes")    .add_legend();
plt.show();


# #observation
# age<40 and nodes<30 have higher change to live
# and
# age>40 and node<20 maybe to die

# In[ ]:


#pair plot
plt.close();
sns.set_style("whitegrid");
sns.pairplot(haberman, hue="status", size=4);
plt.show()


# obervation
# 1.it simply works on 4 feature not on 6 or n number
# 2.not lineraly sepearable
# 3.it look like diagnoal matrix

# ALL OBSERVATIONS

# 1.we are worked on haberman which is very important dataset from kaggle and haberman is unbalanced dataset.
# 2.but appropriate dataset 
# 3.There are 4 columns of haberman dataset which has integer datatype
# 4.the class label "status" is interger datatype but now labelled as a 1 as "yes" & 2 as "no" survived and unsurvied
# 5.sometimes jupyter notebook not work properly on unbalanced dataset.
# 6.from box plot 45% patient are below age of 52.
# 7.on X-axis 40-65 range patients are survived most 
# and seems majority of survived patients are higher
# 8.on x axis year having range 60-66 had highest rate
# 9.it is combines the benifits of boxplot and histogram plot
# 10.In scatter plot age<40 and nodes<30 have higher change to live
# and
# age>40 and node<20 maybe to die
# 11.it simply works on 4 feature not on 6 or n number
# 12.not lineraly sepearable
# 13.it look like diagnoal matrix
# 
