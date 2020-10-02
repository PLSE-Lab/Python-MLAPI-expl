#!/usr/bin/env python
# coding: utf-8

# # Exercise:

# 1. Download Haberman Cancer Survival dataset from Kaggle. You may have to create a Kaggle account to donwload data. (https://www.kaggle.com/gilsousa/habermans-survival-data-set)
# 2. Perform a similar alanlaysis as above on this dataset with the following sections:
# 3. High level statistics of the dataset: number of points, numer of features, number of classes, data-points per class.
# 4. Explain our objective.
# 5. Perform Univaraite analysis(PDF, CDF, Boxplot, Voilin plots) to understand which features are useful towards classification.
# 6. Perform Bi-variate analysis (scatter plots, pair-plots) to see if combinations of features are useful in classfication.
# 7. Write your observations in english as crisply and unambigously as possible. Always quantify your results.

# In[ ]:


# The dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's 
# Billings Hospital on the survival of patients who had undergone surgery for breast cancer.
# Attribute Information:
    # Age of patient at time of operation (numerical)
    # Patient's year of operation (year - 1900, numerical)
    # Number of positive axillary nodes detected (numerical)
    # Survival status (class attribute) 1 = the patient survived 5 years or longer 2 = the patient died within 5 year

import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
habermans = pd.read_csv("../input/haberman.csv",header=None,names=['age','year of operation','positive axillary nodes','Survival status'])
# As the dataset is large will be displaying top records from top and last using head and tail 
habermans.head()
warnings.filterwarnings("ignore")


# In[ ]:


# Displaying dataset from the bottom
habermans.tail()


# # 2.Perform a similar alanlaysis as above on this dataset with the following sections:

# In[ ]:


#2
print(habermans.describe())
print("50th percentile of Survival Status " ,np.percentile(habermans['Survival status'],50))
print("75th percentile of Survival Status ",np.percentile(habermans['Survival status'],75))
print("70th percentile of Survival Status ",np.percentile(habermans['Survival status'],70))
print("73rd percentile of Survival Status ",np.percentile(habermans['Survival status'],73))
print("74th percentile of Survival Status ",np.percentile(habermans['Survival status'],74))


# # Observations from #2:
#     > Age : Patients who undergone operations ranges from 30 to 83
#     > Year Of Operation : Operation year range between  1958 to 1969 where 75% of the treatment completed by 1965 
#     > Though maximum of 'Number of Positive axillary nodes' is 52.0 ,75% observations has less than equal to just 4.0
#     > Survival status data set is imbalanced with 73% status as '1'
#     

# # 3.High level statistics of the dataset: number of points, numer of features, 
# # number of classes, data-points per class.

# In[ ]:


#3
print ("Number of Data Points & Features {0}". format(habermans.shape))
print("Name of coulmns or data features {0}". format(habermans.columns))
print("Number of classes, data-points per class : \n{0}" .format(habermans['Survival status'].value_counts()))


# # Observations from #3:
#     > Habermans data set has 306 records with 4 coulumns or data features 
#     > Four coulumns are 'age', 'year of operation', 'positive axillary nodes',
#        'Survival status'
#     > There are two category in 'Survival status'class , 1 and 2  where 225 observations fall under survival status '1' and 81 under '2'
#     
#     

# # 4. Explain our objective.
The objective here is to find  given the data features 'age','year of operation' & 'number of positive axillary nodes'  of a patient which category it will fall .i.e whether he/she will survive more than 5 yrs or die within 5 yrs.
# # 5. Perform Univaraite analysis(PDF, CDF, Boxplot, Voilin plots) to understand which features are useful towards classification.

# In[ ]:


# 5.1 Lets plot Histogram with PDF of 'age', 'year of operations', 'positive axillary nodes'


# In[ ]:


sns.FacetGrid(habermans, hue="Survival status", size=5)    .map(sns.distplot, "age")    .add_legend();
plt.title("Histogram with PDF of age");
sns.FacetGrid(habermans, hue="Survival status", size=5)    .map(sns.distplot, "year of operation")    .add_legend();
plt.title("Histogram with PDF of year of operation");
sns.FacetGrid(habermans, hue="Survival status", size=5)    .map(sns.distplot, "positive axillary nodes")    .add_legend();
plt.title("Histogram with PDF of positive axillary nodes");
plt.show();


# # Observations from #5.1: 
#     > From PDF we can say that chances for patients to survive greater than 5 years  is high when 'number of positive axillary nodes' <= 4
#     > Patients whose age is <= 34 years will surely survive for 5 years or longer and who are >=78 will die within 5 years
#   
#     
#     

# In[ ]:


# 5.2 Lets plot CDF of 'age', 'year of operations', 'positive axillary nodes' using KDE


# In[ ]:


habermans_above5=habermans.loc[habermans["Survival status"]==1]
habermans_below5=habermans.loc[habermans['Survival status']==2]
# PDF & CDF on positive axillary nodes
counts, bin_edges = np.histogram(habermans_above5['positive axillary nodes'], bins=10, density = True,)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf,color='green',label='pdf for survival status \'1')
plt.plot(bin_edges[1:], cdf,color='blue',label='cdf for survival status \'1')
counts, bin_edges = np.histogram(habermans_below5['positive axillary nodes'], bins=10, density = True,)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf,color='red',label='pdf for Survival status \'2')
plt.plot(bin_edges[1:], cdf,color='black',label='cdf for Survival status \'2')
pylab.legend(loc='center')
plt.xlabel('positive axillary nodes')
plt.title('PDF & CDF on positive axillary nodes')
plt.show()

>   Almost 82% of the patients with status '1' who survived 5yrs or longer has 'number of postives axillary nodes' less than or equal to 4 
# In[ ]:


# PDF & CDF on age
counts, bin_edges = np.histogram(habermans_above5['age'], bins=10, density = True,)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf,color='green',label='pdf for survival status \'1')
plt.plot(bin_edges[1:], cdf,color='blue',label='cdf for survival status \'1')
counts, bin_edges = np.histogram(habermans_below5['age'], bins=10, density = True,)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf,color='red',label='pdf for survival status \'2')
plt.plot(bin_edges[1:], cdf,color='black',label='cdf for survival status \'2')
pylab.legend(loc='upper left')
plt.xlabel('age')
plt.title('PDF & CDF on age')
plt.show()

> 8% of the patients with status as '1' has age less than equal to 34 yrs
> Patients age above 78 yrs will surely die within 5 yrs

# In[ ]:


# PDF & CDF on year of operations
counts, bin_edges = np.histogram(habermans_above5['year of operation'], bins=10, density = True,)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf,color='green',label='pdf for survival status \'1')
plt.plot(bin_edges[1:], cdf,color='blue',label='cdf for survival status \'1')
counts, bin_edges = np.histogram(habermans_below5['year of operation'], bins=10, density = True,)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf,color='red',label='pdf for survival status \'2')
plt.plot(bin_edges[1:], cdf,color='black',label='cdf for survival status \'2')
pylab.legend(loc='upper left')
plt.xlabel('year of operation')
plt.title('PDF & CDF on year of operation')
plt.show()

>  Before 1960-1961 approx 30% of the treatments happened where survival status with '2' is higher
>  8%(46-38) of the Patients undergone operation between  mid 1960- 1963 where survival status with '1' is higher
>  33%(48-70) Operation happened between year 1963-1965 and had almost equal chances of having status as either '1' or     '2'
>  21% (70-90) operations occured between 1965-1967 with more chances of survival status as '2' .
>  Post 1967 rest of the treatment happend with high chances of getting status as '1'
# In[ ]:


# 5.3 Lets plot BoxPlot of 'age', 'year of operations', 'positive axillary nodes' 


# In[ ]:


sns.boxplot(x='Survival status',y='positive axillary nodes', data=habermans)
plt.title('BoxPlot of positive axillary nodes')
plt.show()
sns.boxplot(x='Survival status',y='year of operation', data=habermans)
plt.title('BoxPlot of year of operation')
plt.show()
sns.boxplot(x='Survival status',y='age', data=habermans)
plt.title('BoxPlot of age')
plt.show()


# # Observations from #5.3 :
#     > Patients with 'number of positive axillary nodes' <=4 have more chance of survival status '1' and above 8 will surely fall under survival status '2'
#     > Operation happened before 1960 have less chances of survival status '1'
#     > Operation happened after 1965 have high chances of survival status '1'
#     > Patients less than equal to 34 yrs has more chance of survival status '1'
#     > Patients age above 78 yrs will surely die within 5 yrs
#     

# In[ ]:


# 5.4 Lets plot ViolinPlot of 'age', 'year of operations', 'positive axillary nodes' 


# In[ ]:


sns.violinplot(x="Survival status", y="positive axillary nodes", data=habermans, size=8)
plt.title('ViolinPlot of positive axillary nodes')
plt.show()
sns.violinplot(x='Survival status',y='year of operation', data=habermans)
plt.title('ViolinPlot of year of operation')
plt.show()
sns.violinplot(x='Survival status',y='age', data=habermans)
plt.title('ViolinPlot of age')
plt.show()


# 
# 
# # Observations from #5.4 :
#     > Survivors >5 yrs has 'number of positive axillary nodes' densed at 0 to 4 and for above 8 status is '2'
#     > Operation happened before 1960 have high chances of survival status less than 5 years
#     > Operation happened after 1965 have high chances of survival status more than 5 years
#     > Patients less than equal to 34 yrs has more chance of survival status '1'
#     > Patients age above 78 yrs have very high chances of dying within 5 yrs

# # 6. Perform Bi-variate analysis (scatter plots, pair-plots) to see if combinations of features are useful in classfication.

# In[ ]:


plt.close();
sns.set_style("whitegrid");
sns.pairplot(habermans, hue="Survival status", vars=['age', 'year of operation', 'positive axillary nodes'],size=4);
plt.show()


# # Observation from 6
#    1) From PairPlot we can deduce that 'postive axillary nodes' & 'year of operation' are the two attributes which can help us classify the survival status better among all attribute combinations

# In[ ]:


sns.set_style("whitegrid");
sns.FacetGrid(habermans, hue="Survival status", size=4)    .map(plt.scatter, "positive axillary nodes", "year of operation")    .add_legend();
plt.show();


# # Final Conclusion
1) Patients whose age<=34yrs & 'number of positive axillary nodes' <=4 & year of operation after 1965 has highest chance of surviving status as '1' i.e they will surely survive for 5 yrs or longer
2) Patients whose age >78 yrs & 'number of positive axillary nodes' > 4 & year of operation before 1960 has least chance of surviving more than 5 years
3) From PairPlot we can identify that 'postive axillary nodes' & 'year of operation' are the two attributes which can help us classify the survival status better among all attribute combinations
4) Patients having 'number of positive axillary nodes' <=4 & year of operation after 1965 has more chances of survival status as '1'
5) Patients having 'number of positive axillary nodes' > 4 & year of operation before 1960 has more chances of survival status as '2'
6) Patients having 'number of positive axillary nodes' > 8-10 has very high chances of survival status as '2'
7) Operation happened b/w 1961-1963 and post 1967 has higher chances of getting status as '1'

# In[ ]:




