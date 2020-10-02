#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing Libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
# assigning Dir to read files
import os
print(os.listdir("../input"))
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


# reading Haberman data
Haberman_Data = pd.read_csv('../input/haberman.csv')

#Total Datapoints
print("Data Shape")
print(Haberman_Data.shape)
print('*************************')
# 306 rows and 4 columns

print("priting first few observatios")
print(Haberman_Data.head())
print('*************************')

#Print Columns
print("Columns")
print(Haberman_Data.columns)
print('*************************')
#three independent columns and one dependent column

Haberman_Data['Age']=Haberman_Data['30']
Haberman_Data['Op_year']=Haberman_Data['64']
Haberman_Data['axil_nodes']=Haberman_Data['1']
Haberman_Data['Surv_Status']=Haberman_Data['1.1']

#Datapoint per class
print("Data Points per class")
print(Haberman_Data['Surv_Status'].value_counts())
print('*************************')


# In[ ]:


#Class variable value 1 indicates 'the patient survived 5 years or longer' and 2 indicates 'the patient died within 5 year'
#Creating dictionary for the calss variable
Surv_stat = {1:'Long Expectancy', 2:'Short Expectancy'}
Haberman_Data['Surv_Status'].replace(Surv_stat,inplace = True)
print(Haberman_Data.tail())


# In[ ]:


#Creating two Dataset Based on class
Haberman_L = Haberman_Data[Haberman_Data['Surv_Status'] == 'Long Expectancy']
Haberman_S = Haberman_Data[Haberman_Data['Surv_Status'] == 'Short Expectancy']

# Undestanding Data
print('Mean of Age',np.mean(Haberman_Data['Age']))
print('Mean of nodes',np.mean(Haberman_Data['axil_nodes']))
print('Min of Age',np.min(Haberman_Data['Age']))
print('Min of nodes',np.min(Haberman_Data['axil_nodes']))
print('Max of Age',np.max(Haberman_Data['Age']))
print('Max of nodes',np.max(Haberman_Data['axil_nodes']))
print('Median of Age', np.median(Haberman_Data['Age']))
print('Median of nodes', np.median(Haberman_Data['axil_nodes']))
print('*******************************************')


# 1. Surgery was performed for patient of age ranging from 30yrs to 83yrs with mean of 52.
# 2. on an average patient with 4 lymph nodes were operated.
# 3. Lymph nodes show a huge range from 0 being minimum to 52 being maximum.

# __Objective : To predict if surgery can increase patient life expectancy.__

# In[ ]:


sb.set_style("whitegrid");
sb.FacetGrid(Haberman_Data, hue="Surv_Status", height=4)    .map(plt.scatter, "Age", "axil_nodes")    .add_legend();
plt.title('2D Scatter Plot of Haberman Data')
plt.show();


# In[ ]:


#2D Scatter Plot
sb.set_style('whitegrid')
sb.pairplot(Haberman_Data, hue = 'Surv_Status', height = 3)
plt.suptitle('Pair Plot Haberman Data')
plt.show()


# Any set of the feature shows huge overlap with no significant linear separation.

# In[ ]:


# Histograms
sb.set_style('dark')
sb.FacetGrid(Haberman_Data, hue = 'Surv_Status', height = 4)            .map(sb.distplot, 'Age')            .add_legend()
plt.title('Age Histogram')
plt.ylabel('Density')
plt.show()

sb.set_style('dark')
sb.FacetGrid(Haberman_Data, hue = 'Surv_Status', height = 4)            .map(sb.distplot, 'axil_nodes')            .add_legend()
plt.title('Axilary Nodes Histogram')
plt.ylabel('Density')
plt.xlabel('Axilary Nodes')
plt.show()

sb.set_style('dark')
sb.FacetGrid(Haberman_Data, hue = 'Surv_Status', height = 4)            .map(sb.distplot, 'Op_year')            .add_legend()
plt.title('Histogram for year operation')
plt.ylabel('Density')
plt.xlabel('Operation Year')
plt.show()


# Although number axilary nodes shows overlap but still we can see a little impact on life expectancy. 

# In[ ]:


label=['PDF of long expectancy', 'CDF of long expectancy', 'PDF of short expectancy', 'CDF of short expectancy']
count, bin_edges = np.histogram(Haberman_L['axil_nodes'], bins = 10, density = True )
pdf  = count/(sum(count))
cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:], pdf);
plt.plot(bin_edges[1:], cdf)

count, bin_edges = np.histogram(Haberman_S['axil_nodes'], bins = 10, density = True )
pdf  = count/(sum(count))
cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)
plt.xlabel('Axilary nodes')
plt.ylabel('Patient %')
plt.title('Axilary nodes PDF & CDF')
plt.legend(label)
plt.show()

#Age
count, bin_edges = np.histogram(Haberman_L['Age'], bins = 10, density = True )
pdf  = count/(sum(count))
cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:], pdf);
plt.plot(bin_edges[1:], cdf)

count, bin_edges = np.histogram(Haberman_S['Age'], bins = 10, density = True )
pdf  = count/(sum(count))
cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)
plt.xlabel('Age')
plt.ylabel('Patient %')
plt.title('Age PDF & CDF')
plt.legend(label)
plt.show()


# Although it doesn't show a significant understanding of life expectancy But
# 1. 80% of patient with 0 lymph nodes survived more than 5 years while 60% died before 5 years as well.

# In[ ]:


#Percentile
print('Quantile of axil node for long expectancy',np.percentile(Haberman_L['axil_nodes'], np.arange(25,125,25)))
print('Quantile of axil node for short expectancy',np.percentile(Haberman_S['axil_nodes'], np.arange(25,125,25)))

print('---------------------------------------------------------------------------------------------------------')
print('Quantile of Age for long expectancy',np.percentile(Haberman_L['Age'], np.arange(0,125,25)))
print('Quantile of Age for short expectancy',np.percentile(Haberman_S['Age'], np.arange(0,125,25)))

print('---------------------------------------------------------------------------------------------------------')
print('90th of axil node for short expectancy',np.percentile(Haberman_L['axil_nodes'], 90))
print('90th of axil node for short expectancy',np.percentile(Haberman_S['axil_nodes'], 90))


# In[ ]:


#boxplot
sb.boxplot(x='Surv_Status',y='axil_nodes',data=Haberman_Data)
plt.title('Axil Nodes Boxplot')
plt.show()

sb.boxplot(x='Surv_Status',y='Age',data=Haberman_Data)
plt.title('Age Boxplot')
plt.show()

sb.boxplot(x='Surv_Status',y='Op_year',data=Haberman_Data)
plt.title('Boxplot for Operation year')
plt.show()


# 1. Axil nodes box plot shows outlier for both class.
# 2. Higher the number of nodes lesser is the life expectancy but doesn't help us to predict life expactancy.
# 3. Age doesn't show any significant effect.
# 4. We can see more survival after 1965 and less survival before 1960.

# In[ ]:


#violinplot
sb.violinplot(x = 'Surv_Status', y = 'axil_nodes', data = Haberman_Data)
plt.title('Axil Nodes Violinplot')
plt.show()

sb.violinplot(x = 'Surv_Status', y = 'Age', data = Haberman_Data)
plt.title('Age Violinplot')
plt.show()

sb.violinplot(x = 'Surv_Status', y = 'Op_year', data = Haberman_Data)
plt.title('Violinplot for Operation Year')
plt.show()


# __Conclusion__
# 1. Given Data is an imbalance data.
# 2. Dataset has shape of 306*4 with one class varaible.
# 3. None of the feature has signicant effect to help us to predict life expecatancy of pateints after surgery.
# 4. Number of lymph node shows a mild significant on survival as ~75% of ptients who survived have 0 nodes.
# 5. Patient undergone surgry after 1965 have higher chances to survive. 
