#!/usr/bin/env python
# coding: utf-8

# In this notebook, the mall customers dataset is analyzed. 
# 
# The dataset belongs to 200 mall visitors. It contains features as gender, age, annual income and spending score.
# 
# The primary purpose of this analyse is seperating visitors in different groups which are based on their demographics, incomes and spending behaviors. Those groups are profiled and defined by the differentiates from the others. If there were a marketing department, those groups would be evaluated with them in order to determinate the marketing activities.
# 
# The other purpose of the analyse is to build a predictive model for spending score. This model may be used to calculate a new customer's spending score on the first day of the lifecycle, which gives the advantage of changing it with marketing activities.
# 
# With those purposes, the steps below are executed:
# 
# 1. Explanatory Data Analysis-EDA
# 
# 2. Data Transformation
# 
# 3. Customer Segmentation
# 
# 4. Predictive Model Building

# Import the neccessary libraries and reading the data

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_completeness_v_measure, silhouette_score
from scipy.spatial.distance import cdist

#reading data
data=pd.read_csv('../input/Mall_Customers.csv')


# **Explanatory Data Analysis-EDA**
# 
# 1. Check the data table
# 
# * Check the shape of data
# 
# * Check the data types
# 
# * Calculate the descriptive statistics and make graphs
# 
# * Check the missing values
# 
# * Check the extreme values
# 

# 1-Check the data table

# In[ ]:


data.head()


# In[ ]:


#You can use CustomerID as index
data.set_index('CustomerID',inplace=True)
data.head()


# 2-Check the shape of the data

# In[ ]:


data.shape


# 3-Check the data types

# In[ ]:


data.dtypes


# 4-Calculate the descriptive statistics and make graphs

# In[ ]:


#print descriptive statistics
data.describe(include='all')


# In[ ]:


#plot the data
features=data.columns.values
fig=plt.figure(figsize=(18,12))
temp_plot_numb=1
for feature in features:
    fig.add_subplot(2,2,
                    temp_plot_numb)
    temp_plot_numb+=1
    title='{} Distribution'.format(feature)
    sns.countplot(data[feature])
    plt.title(title)
plt.tight_layout()


# In[ ]:


#Create a pair plot the visualize the relationship between features
sns.pairplot(data,
             hue='Gender',
             height=4);


# 5- Check the missing values

# In[ ]:


#is there any missing value ?
data.isnull().sum().any()   


# 6-Check the extreme values

# In[ ]:


def check_extreme_values_and_visualize(data_frame,
                                       lower_limit_perc,
                                       upper_limit_perc):
    '''this user made function takes 3 inputs and returns a plot with extreme values marked differently'''
    lower_limit=np.percentile(data_frame, lower_limit_perc, axis=0)
    upper_limit=np.percentile(data_frame, upper_limit_perc, axis=0)
    select_extreme_values= np.logical_or(data_frame>=upper_limit,data_frame<=lower_limit)
    plt.plot(data_frame[select_extreme_values],'ro')
    plt.plot(data_frame[~select_extreme_values],'bo')
    plt.ylabel(data_frame.name)
    plt.title('{} Distribution and Extreme Values'.format(data_frame.name))

#visualize the extreme values    
fig=plt.figure(figsize=(25,12))
ax1=fig.add_subplot(1,3,1)
check_extreme_values_and_visualize(data_frame=data['Age'],
                                       lower_limit_perc=5,upper_limit_perc=95)
ax1=fig.add_subplot(1,3,2)
check_extreme_values_and_visualize(data_frame=data['Annual Income (k$)'],
                                       lower_limit_perc=5,upper_limit_perc=95)
ax1=fig.add_subplot(1,3,3)
check_extreme_values_and_visualize(data_frame=data['Spending Score (1-100)'],
                                       lower_limit_perc=5,upper_limit_perc=95)


# *What have I done and learnt from the explanatory data analysis-EDA steps ?*
# 
# 1. The first 5 row of the dataset is examined and 'customer id' column is defined as index. 
#     * Customer id is a numeric id feature and it hasn't any additional effect on the analysis.
#     * Gender column is categoric. I need to transform this feature to numeric on further steps.
# 2. After examination of the dataset and features, number of observations are displayed.
#     * There are 200 observations. Considiration of this size of data, I should be careful about dealing with missing values. 
# 3. After viewing the first 5 rows, I assure myself about the datatypes.
# 4. Calculating the descriptive statistics and making graphs give great information about the data characteristics.visualisation
#     * Nearly %60 percent of the data belongs to female mall visitors.
#     * Age feature changes between 18 and 70, which means all visitors are adults. The mean of age is 38 and accourding to the median, half of the visitors are younger than 36.
#     * Annual income changes between 15 and 137, which looks like a wide range. This feature need to be focused on during the extreme values step.
#     * Spending score changes between 1 and 99. Half of the visitors have spending score less than 50.
#     * With pair plot, the relationships between the features are visualized. The customer groups are obviously seen in the graphs.
# 5. The missing values are checked and none found.
# 6. The features distribution is visualized with differentiating the highest and lowest values. 
#     * There is no serious evidence of extreme values, except spending score feature. No action is taken until the first result of the analyse completed. 
# 

# **Data Transformation**
# 
# 1. Impute the missing and extreme values
# 2. Standardize the variables
# 3. Convert all features to numeric

# 1-Impute the missing and extreme values
# 
# I skip this step for now. There is no null values in our dataset. On the other hand spending score feature might contains extreme values. But I don't do any transformation before building the predictive model.

# 2-Standardize the variables

# In[ ]:


MM_Scaler=MinMaxScaler() #define a scaler
data_standardized=data.copy() # keep the original data for further usage
#standardize Age
data_standardized['Age']=MM_Scaler.fit_transform(data['Age'].values.reshape(-1,1))
#standardize annual income
data_standardized['Annual Income (k$)']=MM_Scaler.fit_transform(data['Annual Income (k$)'].values.reshape(-1,1))
#standardize spending score
data_standardized['Spending Score (1-100)']=MM_Scaler.fit_transform(data['Spending Score (1-100)'].values.reshape(-1,1))


#  3- Convert all features to numeric

# In[ ]:


data_standardized=pd.get_dummies(data=data_standardized,columns=['Gender'])


# *What have I done in the data transformation step ?*
# 1. I didn't do any imputation in this step. Beside there is no missing values, I want to decide whether I deal with extreme values of the spending score feature or not.
# 2. I used min-max scaler for standardization. Z-score standardization is an another option but I prefer max-min standardization. I might change it according to the first result of the analyse.
# 3. I convert the gender feature to numeric values. Only one column is enough to present the gender column but I choose to keep both in order to help on further steps.

# **Segmentation**
# 
# 1. Run K-means algorithm
#     * Find the best K parameter for best seperation
#     * Profile the outputs with different K parameters in order to select the most useful customer grouping for marketing purposes.
# 2. Identify the customer groups with most growth potential.
#     * Assign the customers to the groups which are different from their original groups.

# 1-Run K-means algorithm

# In[ ]:


#Find the best K parameter for best seperation

distortions = [] # create an empty list for distortion values
silhouette_scrs=[] #create an empty list for silhouette values

X=data_standardized[['Age','Annual Income (k$)','Spending Score (1-100)','Gender_Female','Gender_Male']]

#a look is created to try and compare different results of k parameters.
for cluster in np.arange(3,15):
    km=KMeans(n_clusters=cluster,random_state=2)
    col_name='Cluster_{}'.format(cluster)
    data_standardized[col_name]=km.fit_predict(X)
    distortions.append(sum(np.min(cdist(X,
                                        km.cluster_centers_,
                                        'euclidean'), 
                                  axis=1)) / X.shape[0])
    silhouette_scrs.append(silhouette_score(X,data_standardized[col_name]))

#Graphs are created to visualize the distortion and silhouette score distrubution after each clustering step with different k parameters
fig=plt.figure(figsize=(28,18))
ax1=fig.add_subplot(2,1,1)
plt.xticks(np.arange(0,12),np.arange(3,15))
ax1=plt.plot(distortions)
ax1=plt.plot(distortions,'ro')
plt.title('Distortion Distribution')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')

ax2=fig.add_subplot(2,1,2)
plt.xticks(np.arange(0,12),np.arange(3,15))
ax2=plt.plot(silhouette_scrs)
ax2=plt.plot(silhouette_scrs,'ro')
plt.title('Silhouette Score Distribution')


# In[ ]:


#After examining the plots,best possible K Parameters is 10.
#Before profiling, I add the cluster and female gender columns to the original data.
data['Gender_Female']=data_standardized['Gender_Female']
data['Cluster_10']=data_standardized['Cluster_10']


# In[ ]:


#In this step, The 10 different customer segments/groups are profiled in order to identify them by their differentiates with the help of visualization

feature_names=['Age','Annual Income (k$)','Spending Score (1-100)','Gender']
#make plots of the cluster 10 and entire population
fig=plt.figure(figsize=(15,40))
plot_numb=1  # this parameter is used to place the plots on the screen.
plot_numb2=3 # this parameter is used to place the plots on the screen.
for feature_name in feature_names:
    ax1=fig.add_subplot(5,2,plot_numb)
    plot_numb +=2 
    ax2=fig.add_subplot(5,3,plot_numb2)
    plot_numb2 +=3
    
    if feature_name=='Gender':        # Historgram is used to visualize the gender feature.
        sns.countplot(hue='Gender',
                      data=data,
                      x='Cluster_10',
                      ax=ax1)
        sns.countplot(x='Gender',
                      data=data,ax=ax2)       
    else:
        sns.boxplot(x='Cluster_10',   # Boxplot is used to visualize the features except gender.
                    y=feature_name,
                    data=data,
                    ax=ax1)
        sns.boxplot(y=feature_name,
                    data=data,
                    ax=ax2)        
        
    plt1_title='{}--{}'.format(feature_name,'Cluster-10')
    ax1.set_title(plt1_title)
    ax1.set_xlabel('Cluster Number')
    plt2_title='{}--{}'.format(feature_name,'Population')
    ax2.set_title(plt2_title)
    ax2.set_xlabel('Cluster Number')


# In[ ]:


Cluster_Profile_df=pd.DataFrame({'Age':['Young','Old','Average','Average','Young','Old','Average','Average','Old','Young'],
                                 'Annual Income (k$)':['Average','Average','High','Low','Low','Average','High','High','Low','Low'],
                                  'Spending Score (1-100)':['Average','Average','High','Low','High','Average','Low','High','Low','Average'],
                                   'Gender':['Female Dominated','Male Dominated','Male Dominated','Female Dominated','Female Dominated','Female Dominated','Male Dominated','Female Dominated','Female Dominated','Male Dominated'], 
                                'Group Size':data['Cluster_10'].value_counts()} )
Cluster_Profile_df.sort_values(by=['Age','Annual Income (k$)','Gender','Spending Score (1-100)'])


# **Customer Groups/Cluster Profiles:**
# 
# I make profiling of customer groups to decide whether the k-means clustering algorithm made the seperation successfully or not. If the cluster profiles make sense and the customer groups are actionable then the k-means clustering algorithm made satisfactory results.
# 
# After running K-means clustering algorithm and create 10 customer segments, I profile the customer segments below:
# 
# 
# *Young Customer Segments*
# 
#     Cluster-0: Average income -- Average spending score -- Woman dominated
#     Cluster-4: Low income  -- High spending score -- Woman dominated
#     Cluster-9: Low income  -- Average spending score -- Man dominated
# 
# *Middle Aged Customer Segments*
# 
#     Cluster-2:* High income	-- High spending score	-- 	Man dominated
#     Cluster-6:* High income	-- Low spending score	-- 	Man dominated
#     Cluster-7:* High income	-- High spending score	-- 	Woman dominated
# 
# *Older Adult Customer Segments*
# 
#     Cluster-1: Average income -- Average spending score	-- 	Man dominated
#     Cluster-3: High income -- Low spending score	-- 	Woman dominated
#     Cluster-5: Average income -- Average spending score -- Woman dominated
#     Cluster-8: High income	-- Low spending score	-- 	Woman dominated

# **Identify the customer groups with most growth potential**
# 
# The main purpose of this grocery store is to sell products to customers as much as possible. The amount of the purchase that a customer made is presented as the spending score. With this perspective, potential customer segments are defined as segments without high spending score. If the customers in those segments fulfill the potential, their segment will change to another segment with higher spending score. 
# 
# After profiling and labeling 10 customer segments, potential customer segments are identified as 0, 9, 6, 1, 3, 5, 8.   The store owner needs to understand the customers in those segments better. Marketing strategies should be designed according to those potential customer segments.
