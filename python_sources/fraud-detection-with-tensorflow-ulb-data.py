#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Before beginning the analysis, we import the necessary libraries/packages that will be required.

# In[ ]:


import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sbn
from scipy.stats import ks_2samp,wasserstein_distance,energy_distance
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import random
from imblearn.over_sampling import RandomOverSampler,SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

from sklearn.metrics import precision_recall_curve,auc
from sklearn.model_selection import cross_val_score,StratifiedKFold,train_test_split,GridSearchCV,ParameterGrid


# Show plot output in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

sbn.set_style("darkgrid")


# # Exploratory Data Analysis

# First, we load the data:

# In[ ]:


credit_card_data=pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')


# We check the first few lines of the dataframe, and get some descriptive statistics for each column:

# In[ ]:


credit_card_data.head(30)


# In[ ]:


descriptive_stats=credit_card_data.describe()
descriptive_stats


# In[ ]:


df_info=credit_card_data.info()
df_info


# As can be seen, we have 284,807 entries (rows) and 31 features (columns). All of the data is numerical, either of type ```float64``` or ```int64```.
# 
# As can be seen from ```df_info```, there is no missing data, which can be more formally checked as following:

# In[ ]:


# check more formally if there is any missing data
# isnull() checks each entry, and the following any() check across rows and columns
credit_card_data.isnull().any().any()


# It would be useful to create separate dataframes for the features and for the target variable. The target variable is the class (no fraud, i.e. ```Class=0``` or fraud, i.e. ```Class=1```). The rest of the columns correspond to the features of each transaction.

# In[ ]:


# Create the dataframe for the target
labels=credit_card_data[['Class']]

# Create the dataframe for the features
features=credit_card_data.drop(['Class'],axis=1)


# To get to know our dataset better, let us first check the amount of examples corresponding to each class:

# In[ ]:


labels['Class'].value_counts()


# The majority corresponds to non-fraud cases (```Class=0```). The corresponding visualization helps to understand the magnitude of each class type:

# In[ ]:


number_of_frauds=labels['Class'].value_counts()[1]
number_of_nonfrauds=len(labels['Class'])-number_of_frauds

fig,ax=plt.subplots(1,1,figsize=(10,10))
sbn.countplot(labels['Class'],ax=ax)
ax.set_title('Number of examples in each class (logarithmic scale)',fontsize=16)
ax.set_yscale('log')
ax.text(0-0.4,number_of_nonfrauds+10000,'Number of non-frauds: '+str(number_of_nonfrauds),fontsize=14)
ax.text(1-0.4,number_of_frauds+150,'Number of frauds: '+str(number_of_frauds),fontsize=14)


plt.show()


# In[ ]:


fraud_ratio=number_of_frauds/(len(labels['Class']))
print('Fraudulent examples comprise the {0:.4f} % of the total examples'.format(100*fraud_ratio))


# The fact that the number of examples in ```Class=0``` is much higher (3 orders of magnitude) suggests that we are dealing with a highly imbalanced dataset. Therefore, care should be taken in order to incorporate this information in our subsequent models.
# 
# For the rest of the features, let us deal first with those for which a description is provided. 
# These are ```Time```, and ```Amount```. 

# According to the [description](https://www.kaggle.com/mlg-ulb/creditcardfraud) of the dataset, the ```Time``` feature corresponds to the seconds elapsed from the first transaction in the dataset (row 0), while the ```Amount``` feature corresponds to the amount of the transaction.

# ## Time of the transaction
# 
# Let us first look into the ```Time``` feature. From the descriptive stats we get some useful information, which we repeat here for convenience:

# In[ ]:


descriptive_stats[['Time']]


# Working in seconds might be a bit difficult to interpret and visualize, so it would be useful to transform the seconds to hours for our analysis. Of course, for modeling the fraud detector/classifier the values in seconds carry much more information (for example, fraudulent transactions might take place a few seconds apart and this is not captured by the hour-resolution we set for the moment).

# In[ ]:


time_in_hours=(features[['Time']]/(60.0*60.0)).astype(int)
time_in_hours.describe()


# The amount of transactions is more or less balanced between the two days (median of the time distribution is 23 hours, which corresponds to the last hour of the first day, since the counting starts from 0).
# 
# We can view the number of transactions by taking the counts for each hour:

# In[ ]:


# Take the value counts from the time_in_hours and convert to dataframe.
# The index of the new dataframe now corresponds to the hour
transactions_per_hour=time_in_hours['Time'].value_counts(sort=False).to_frame()

# Reset the index to the dataframe so that the previous index becomes column
transactions_per_hour.reset_index(inplace=True)

# Change the name of the columns for better comprehension
transactions_per_hour.columns=['Hour','Transactions']

transactions_per_hour.head(10)


# Consequently, we have information now regarding the amount of transactions per hour and the distribution of the hourly number of transactions

# In[ ]:


transactions_info=transactions_per_hour[['Transactions']].describe()
transactions_info


# More intuition can be gained by visualizing the aforementioned information:

# In[ ]:


fig,axes=plt.subplots(1,2,figsize=(10,5))

ax=axes[0]

ax.hist(features['Time']/(60*60),bins=range(48))
ax.set_xticks([0,8,16,24,32,40,47])
ax.set_xlim([-1,49])
ax.set_title('Number of transactions per hour',fontsize=16)
ax.set_xlabel('Time [hours]',fontsize=14)
ax.set_ylabel('Counts' ,fontsize=14)

ax=axes[1]
ax.hist(transactions_per_hour['Transactions'],bins=10)
ax.set_title('Counts of the hourly transaction number',fontsize=16)
ax.set_xlabel('Hourly transactions',fontsize=14)


plt.subplots_adjust(wspace = 0.5)
plt.show()


# As can be seen from the two plots, the majority of transactions occur from hour 8 to hour 22 and from hour 32 (hour 8 of the second day) to hour 46 (hour 23 of the second day). This maximum number of hourly transactions is around 8000 as is suggested by the right plot. There is still however a significant amount of transactions that occur outside these hours. In order to see if there is any correlation between the number of hourly transactions and the nature of these kind of transactions (fraud / not fraud), we can input additional information in the histograms.

# In[ ]:


fig = plt.figure(figsize=(15, 10))
grid = plt.GridSpec(3, 3, wspace=0.4, hspace=0.3)

ax1=fig.add_subplot(grid[0, 0])
ax2=fig.add_subplot(grid[1, 0])
ax3=fig.add_subplot(grid[2, 0])
ax4=fig.add_subplot(grid[:,1:])


ax=ax1

ax.hist(features['Time']/(60*60),bins=range(48))
ax.set_xticks([0,8,16,24,32,40,47])
ax.set_xlim([-1,49])
ax.set_title('Number of transactions per hour',fontsize=14)
ax.set_ylabel('Counts' ,fontsize=14)

ax=ax2

ax.hist(features[labels['Class']==0]['Time']/(60*60),bins=range(48))
ax.set_xticks([0,8,16,24,32,40,47])
ax.set_xlim([-1,49])
ax.set_title('Number of non-fraudulent transactions per hour',fontsize=14)
ax.set_ylabel('Counts' ,fontsize=14)

ax=ax3
ax.hist(features[labels['Class']==1]['Time']/(60*60),bins=range(48))
ax.set_xticks([0,8,16,24,32,40,47])
ax.set_xlim([-1,49])
ax.set_title('Number of fraudulent transactions per hour',fontsize=14)
ax.set_ylabel('Counts' ,fontsize=14)
ax.set_xlabel('Time [hours]',fontsize=14)

ax=ax4
sbn.distplot(features[labels['Class']==0]['Time']/(60*60),
             kde=True,hist=True,
             ax=ax,label='No Fraud',
             color='green',
             norm_hist=True
             )
sbn.distplot(features[labels['Class']==1]['Time']/(60*60),
             kde=True,
             hist=True,
             norm_hist=True,
             ax=ax,label='Fraud',color='red')

# sbn.distplot(features['Time']/(60*60),kde=True,hist=False,ax=ax,label='Total')
ax.set_xlabel('Time [hours]',fontsize=14)
ax.set_title('Distribution of frauds and non-frauds vs hours',fontsize=15)
# Use log scale to emphasize big (relative) differences
ax.set_yscale('log')
plt.show()


# It is clear that during both days fraudulent transactions occur. What is more, the peak of the fraudulent transactions occurs for a much smaller timespan than for normal transactions. Also, we see that we have the maximum number of frauds around hour 10 of the first day and around hour 2 of the second day. Overall, with respect to the ```Time``` feature, we could say that fraudulent transactions follow a more uniform distribution relative to the normal transactions, meaning that change in magnitude of the normal transactions is much stronger, while the number of fraudulent transactions acquires values in a much smaller range most of time.
# 
# As a result, during off-peak times (for example at night or very early in the morning), when a normal transaction seldom occurs, there is a higher probability that a fradulent transaction takes place.
# 

# Another piece of information we can collect is to check for the time elapsed between fraudulent transactions. For this, we use the original dataset, with the time in seconds.

# In[ ]:


# Get the frauds
time_of_frauds=credit_card_data[credit_card_data['Class']==1][['Time']]
# Calculate the difference with previous row and add it as an additional column
time_of_frauds['Time difference']=time_of_frauds['Time'].diff()
time_of_frauds.head()


# In[ ]:


fig,axes=plt.subplots(1,2,figsize=(10,5))

# Set the bin edges to correspond to every 10 seconds
bins=[10*i for i in range(int(len(time_of_frauds)/10))]

# Select time difference up to which we zoom in
seconds_to_zoom=100
# Set the corresponding bins every 10 seconds
bins_zoom=[10*i for i in range(int(seconds_to_zoom/10)+1)]

ax=axes[0]
sbn.distplot(time_of_frauds['Time difference'].dropna(),ax=ax,norm_hist=False,kde=False,bins=bins)
ax.set_xlabel('Time difference [seconds]',fontsize=13)
ax.set_ylabel('Counts of fraudulent transactions',fontsize=13)
ax.set_title('Time difference distribution between\n consequtive frauds',fontsize=16)
ax.set_xticks(bins[::5])

ax=axes[1]
sbn.distplot(time_of_frauds['Time difference'].dropna(),ax=ax,norm_hist=False,kde=False,bins=bins_zoom)
ax.set_xlabel('Time difference [seconds]',fontsize=13)
ax.set_ylabel('Counts of fraudulent transactions',fontsize=13)
ax.set_title('Time difference distribution between\n consequtive frauds (zoomed in)',fontsize=16)
ax.set_xlim([0,100])
ax.set_xticks(bins_zoom)


plt.subplots_adjust(wspace = 0.5)

plt.show()


# It is evident that the majority of fraudulent transactions occur in a very short time period, that is in the range of 10 seconds between each other. Fewer fraudulent transactions occur over a wider time span. Does the time difference distribution for normal transactions exhibit a specific behavior itself?

# In[ ]:


# Get the non - frauds
time_of_nonfrauds=credit_card_data[credit_card_data['Class']==0][['Time']]
# Calculate the difference with previous row and add it as an additional column
time_of_nonfrauds['Time difference']=time_of_nonfrauds['Time'].diff()
time_of_nonfrauds.head()


# In[ ]:


fig,axes=plt.subplots(1,2,figsize=(10,5))

# Set the bin edges to correspond to every 10 seconds
bins=[10*i for i in range(int(len(time_of_frauds)/10))]

# Select time difference up to which we zoom in
seconds_to_zoom=100
# Set the corresponding bins every 10 seconds
bins_zoom=[10*i for i in range(int(seconds_to_zoom/10)+1)]

ax=axes[0]
sbn.distplot(time_of_nonfrauds['Time difference'].dropna(),ax=ax,norm_hist=False,kde=False,bins=bins)
ax.set_xlabel('Time difference [seconds]',fontsize=13)
ax.set_ylabel('Counts of non fraudulent transactions',fontsize=13)
ax.set_title('Time difference distribution between\n consequtive non frauds',fontsize=16)
ax.set_xticks(bins[::5])

ax=axes[1]
sbn.distplot(time_of_nonfrauds['Time difference'].dropna(),ax=ax,norm_hist=False,kde=False,bins=bins_zoom)
ax.set_xlabel('Time difference [seconds]',fontsize=13)
ax.set_ylabel('Counts of non fraudulent transactions',fontsize=13)
ax.set_title('Time difference distribution between\n consequtive non frauds (zoomed in)',fontsize=16)
ax.set_xlim([0,100])
ax.set_xticks(bins_zoom)


plt.subplots_adjust(wspace = 0.5)

plt.show()


# The majority of normal transactions occur within a very limited time span, which is approximately 10 seconds. However, the last two plots do not suggest a correlation between the time difference of two random transactions and their class. A better interpretation is that given a specific class of transaction (fraud/no fraud), it is most probable (suggested by the current data) that a similar class of transaction will occur within the first 10 seconds. Moreover, for fraudulent transactions, there is also a finite probability that another fraudulent transaction will take place in a broader timespan. Perhaps a more informative feature would combine additional data for the transaction, such as the place, encrypted id of the user, etc. which unfortunately is not available in this dataset.

# What about the time difference between transactions in general?

# In[ ]:


# Get the non - frauds
time_diff=credit_card_data[['Time','Class']]
# Calculate the difference with previous row and add it as an additional column
time_diff['Time difference']=time_diff['Time'].diff()
time_diff[['Time difference']].describe()


# In[ ]:


fig,ax=plt.subplots(1,1)
sbn.distplot(time_diff[time_diff['Class']==0]['Time difference'].dropna(),
             kde=False,bins=[5*i for i in range(7)],
             label='Non-fraud')
sbn.distplot(time_diff[time_diff['Class']==1]['Time difference'].dropna(),
             kde=False,color='green',
             bins=[5*i for i in range(7)],
             label='Fraud')
ax.set_yscale('log')
ax.set_xlabel('Time difference [seconds]',fontsize=13)
ax.set_ylabel('Counts of transactions',fontsize=13)
ax.set_title('Type of transactions vs. time difference \nbetween consequtive transactions',fontsize=16)
ax.legend()
plt.show()


# We see that for the given credit card system, for consequtive transactions that have a time difference larger than 20 seconds, the latter transaction cannot be fraudulent. This is highly dependent on these two consequtive days and cannot suggest for sure that a transaction occuring >20 seconds from the last one cannot be a fraudulent one on any given day. There is a trend however where the ratio of fraudulent to non-fraudulent transactions is higher as the time interval between transactions get shorter. Fraudulent and non-fraudulent transactions have similar distributions, but the one of normal transactions contains more outlier information. Should we decide to use this engineered feature, we need to bear in mind that shuffling the dataset ** before ** creating it is prohibited, since the results depend on the time difference.

# ## Transaction amount
# 
# The second feature for which a clear description exists is ```Amount```. Let us begin by getting some descriptive statistics for this column.

# In[ ]:


features[['Amount']].describe()


# The majority of transactions (75%) include amounts up to 77.165. There are however some outliers which "push" the mean to higher values. Let us look into the distribution of the ```Amount``` feature for each class separately:

# In[ ]:


fig,axes=plt.subplots(1,2,figsize=(10,6))

ax=axes[0]

sbn.distplot(credit_card_data[credit_card_data['Class']==0]['Amount'],
             ax=ax,
             kde=False,
             norm_hist=False,
             bins=20,
             color='green',
             label='No Fraud')

sbn.distplot(credit_card_data[credit_card_data['Class']==1]['Amount'],
             ax=ax,
             kde=False,
             norm_hist=False,
             bins=20,
             color='red',
             label='Fraud')

ax.set_yscale('log')
ax.set_title('Histogram of "Amount" for each class',fontsize=16)
ax.set_xlabel('Amount',fontsize=13)
ax.set_ylabel('Counts',fontsize=13)

ax=axes[1]

sbn.distplot(credit_card_data[credit_card_data['Class']==0]['Amount'],
             ax=ax,
             kde=False,
             norm_hist=True,
             bins=20,
             color='green',
             label='No Fraud')

sbn.distplot(credit_card_data[credit_card_data['Class']==1]['Amount'],
             ax=ax,
             kde=False,
             norm_hist=True,
             bins=20,
             color='red',
             label='Fraud')

ax.set_yscale('log')
ax.set_title('Histogram of "Amount" for each class',fontsize=16)
ax.set_xlabel('Amount',fontsize=13)
ax.set_ylabel('Density',fontsize=13)


plt.subplots_adjust(wspace = 0.5)

plt.show()


# It is evident that all fraudulent transactions (```Class=1```) regard smaller amounts than non-fraudulent transactions. Some more descriptive statistics can be obtained for the ```Amount``` in each class:

# In[ ]:


print('No Fraud')
features[labels['Class']==0][['Amount']].describe()


# In[ ]:


print('Fraud')
features[labels['Class']==1][['Amount']].describe()


# The maximum value of fraudulent transactions is 2,125.87, while for normal transactions is 25,961.16. Moreover, most fraudulent transactions (75%) occur at similar values like normal ones (105.89 for ```Class=1``` and 77.05 for ```Class=0```). The main conclusion from the ```Amount``` variable is that for higher values, it is less probable for a transaction to be categorized as fraud.

# It would be intersting to combine ```Amount```, ```Time``` and ```Class```. We can plot ```Amount``` as a function of ```Time``` for each ```Class```. In this case, scatter plots are most appropriate:

# In[ ]:


fig = plt.figure(figsize=(15, 10))
grid = plt.GridSpec(2, 2, wspace=0.4, hspace=0.3)


ax1=fig.add_subplot(grid[0, :])
ax2=fig.add_subplot(grid[1, 0])
ax3=fig.add_subplot(grid[1, 1])


sbn.scatterplot(x='Time',y='Amount',data=credit_card_data,ax=ax1,alpha=0.35)
ax1.set_title('Amount vs time for all transactions',fontsize=16)
ax1.set_xlabel('Time [seconds]',fontsize=13)
ax1.set_ylabel('Amount',fontsize=13)



sbn.scatterplot(x='Time',y='Amount',data=credit_card_data[credit_card_data['Class']==0],ax=ax2,color='green',alpha=0.35)
ax2.set_title('Amount vs time for normal transactions',fontsize=16)
ax2.set_xlabel('Time [seconds]',fontsize=13)
ax2.set_ylabel('Amount',fontsize=13)

sbn.scatterplot(x='Time',y='Amount',data=credit_card_data[credit_card_data['Class']==1],ax=ax3,color='red',alpha=0.35)
ax3.set_title('Amount vs time for fraudulent transactions',fontsize=16)
ax3.set_xlabel('Time [seconds]',fontsize=13)
ax3.set_ylabel('Amount',fontsize=13)

plt.show()


# Unfortunately, there is no additional information gain regarding ```Class```, when we "combine" ```Amount``` and ```Time```. Small-amount transactions (which are also the majority in both cases) for each class happen along the full timespan while the higher-amount transactions for both types, do not have a specific relation with respect to time.

# As a final notice, we point out that ```Time``` and ```Amount``` occur at different scales. More specifically, ```Time``` (in seconds) ranges from 0 to 175,000 while ```Amount``` ranges from 0 to 25,000, that is we have an order of magnitude difference. Consequently, both of these features, need to be scaled appropriately if they are to be used as input to a learning algorithm.

# # Anonymised features V1-V28

# After investigation of the ```Time``` and ```Amount``` features, we continue with an analysis of the anonymised features (```V1-V28```). As is mentioned in the description, these features have resulted from PCA in a higher-dimensional feature space that contained personalized information about each transaction. Unfortunately, such additional information is not provided and we are only given the PCA version of the features.

# We can have a look at the ```min, max, std``` and ```mean``` values of these features, which we isolate in a separate dataframe called ```anonymised_features```.

# In[ ]:


anonymised_features=features.drop(['Time','Amount'],axis=1)


# In[ ]:


anonymised_features.describe()


# As can be seen, features ```V1-V28``` have ```mean=0```, ```std``` can range from 1.95 to 0.33, with the ```min``` and ```max``` taking values in a larger range.

# We can plot a histogram along with a fitted distribution for each of these features, and for each class separately.

# In[ ]:


anonymised_features=features.drop(['Time','Amount'],axis=1)

plt.figure(figsize=(12,28*4))
grid = plt.GridSpec(28, 1)

ks_distances=[]
emd_distances=[]
ed_distances=[]

for idx,feature in enumerate(anonymised_features.columns):
    
    plt.subplot(grid[idx])
        
    sbn.distplot(anonymised_features[labels['Class']==0][feature],
                 kde=True,
                 color='green',
                 label='No Fraud',bins=30)
    
    sbn.distplot(anonymised_features[labels['Class']==1][feature],
                 kde=True,
                 color='red',
                 label='Fraud',bins=30)
    
    
    ks=ks_2samp(anonymised_features[labels['Class']==1][feature].values,
                anonymised_features[labels['Class']==0][feature].values)
        
    emd=wasserstein_distance(anonymised_features[labels['Class']==1][feature].values,
                             anonymised_features[labels['Class']==0][feature].values)
    
    ed=energy_distance(anonymised_features[labels['Class']==1][feature].values,
                       anonymised_features[labels['Class']==0][feature].values)
    
    ks_distances.append(ks[0])
    emd_distances.append(emd)
    ed_distances.append(ed)
        
    plt.title(feature+': KS: {0:.2f}, EMD: {1:.2f}, ED: {2:.2f}'.format(ks[0],emd,ed),fontsize=20)
    plt.xlabel(feature,fontsize=18)
    plt.legend()
    
plt.subplots_adjust(hspace = 0.5)

plt.show()


# By inspection of the plots, it is evident that there are certain features, for which the distributions for each ```Class``` are very close to each other. For example, the distributions of ```V23``` are much more similar than the corresponding ones of ```V18```. Another way to compare two distributions visually would be Q-Q plots.
# More formally, this dissimilarity can be quantified by calculating the distance between the distributions. The calculated statistical distances using Kolmogorov-Smirnov statistic (KS), Wasserstein Distance (or Earth's mover distance, EMD) and Energy Distance (ED,Cramer-von Mises) are annotated on top of each plot. Specifically, since the KS is a statistical test, the p-value is also important. For all cases, the p value is very small and thus, we only need to focus on statistic value to say whethere two samples come from the same distribution. 
# 
# We should also mention that we are mostly interested in the relative difference between the distributions of the features. Using the values for just one feature we cannot say with certainty if this feature contributes well to specifying the target variable. However, by comparing the distance metrics for one feature with those of another feature, we can conclude whether one of the two features contributes more or less to predicting the output. As an example, we consider feature ```V28``` with ```KS=0.37, EMD=0.3, ED=0.33```. Although by visual inspection it seems that the two distributions are similar, we cannot say if this is a small or large distance. By looking however at ```V16```, with ```KS=0.69, EMD=4.18, ED=1.94```, we can conclude that ```V16``` is a "stronger" predictor than ```V28``` regarding ```Class```.
# 
# Below, we plot the distances for each feature

# In[ ]:


fig,axes=plt.subplots(3,1,figsize=(15,20),sharex=False)

ax=axes[0]

sbn.barplot(x=np.arange(28),y=ks_distances,ax=ax)
ax.set_title('Kolmogorov-Smirnov Statistic',fontsize=16)
ax.set_xticklabels(anonymised_features.columns.values)


ax=axes[1]

sbn.barplot(x=np.arange(28),y=emd_distances,ax=ax)
ax.set_title('Wasserstein distance',fontsize=16)
ax.set_xticklabels(anonymised_features.columns.values)

ax=axes[2]

sbn.barplot(x=np.arange(28),y=ks_distances,ax=ax)
ax.set_title('Energy distance',fontsize=16)

ax.set_xticklabels(anonymised_features.columns.values)


plt.show()


# Based on these distances, we can decide to keep or discard certain features from predicting the target variable. The reason is because the corresponding feature values do not differentiate between ```Class=1``` and ```Class=0```: for each value of one feature if the calculated distances are small, the probability of obtaining either of the classes is similar, which means that the outcome for the ```Class``` variable is more stronly affected by other features than this one. Deciding to discard the feature in question, we are less likely to teach the "noise" to the model and thus less likely to overfit the dataset.

# # Modeling of Fraud/No Fraud classifier

# Based on the analysis above, we summarize the main points so far:
# 
# * Highly imbalanced dataset (minority ```Class=1```, approx. 0.17% of total examples)
# * ```Time``` and ```Amount``` feature columns can play a role in predicting the ```Class``` but need to be appropriately scaled
# * Some of the PCA features could be discarded when predicting the target variable. Scaling can also be applied to these features
# 
# 
# In the following, we are going to consider all the given features and decide through testing if some ought to be discarded when predicting the ```Class```. Moreover, we assume, since no further information is given, that the transactions are independent, i.e. we do not know whether they come from the same user, location, credit card, etc.
# In this way, we can treat the ```Time``` attribute merely as a numerical feature, without interpreting its temporal meaning. More specifically, the temporal order of the examples will not be treated further. As a result, we will also not use any additional engineered features obtained by using the ```Time``` values.

# To tackle the problem of class imbalance, we are going to use techniques such as oversampling, undersampling or a combination of those. 
# 
# With oversampling, we sample more data from the minority class (here ```Class=1```) either randomly with replacement, or by generating new examples using a nearest neighbors algorithm with the SMOTE technique. With undersampling, we take a small sample from the majority class (here ```Class=0```) in order to match the scale of the number of examples in the minority class. The drawback here is that we train the classifier with a smaller training set, where some critical information might have been removed. A combination of both techniques can be applied as follows: we apply SMOTE and then we clean-up some of the noisy-generated examples (connecting outliers to inliers for example).
# 
# We should also mention that is important to ** first ** split the dataset into train and validation sets and perform resampling ** afterwards **. The reason is that we would like to train on the artificially augmented dataset, but we want to optimize with respect to data that the algorithm has not seen. Moreover, the kind of data that we optimize our algorithm on, should be of the same distribution as data on which the algorithm will eventually be used. As a result, we cannot use the augmented dataset for validation since:
# 
# 1) Any artificially created data during resampling will be seen again during validation
# 
# 2) We will artificially change the ratio of frauds/non-frauds and make it bigger. However, this ratio does not (hopefully) resemble real-world situations.

# In the rest of this notebook, there is a detailed implementation of a Neural Net (NN) algorithm using the low-level ```Tensorflow``` API. 
# 
# We create custom estimators, that are wrapped by a ```Scikit-learn```- respective custom estimator, that is used to implement a ```Pipeline```. This ```Pipeline``` is "fed" data and:
# 
# 1) Preprocess it, by selecting columns and applying some scaling
# 
# 2) Trains/tests the model on the given data
# 
# In order to scan a representative range of some hyperparameters, we implement a grid-search procedure. This procedure does the following:
# 
# 1) Splits the data into train and validation examples defined by ```folds```. We are going to use ```StratifiedKFold``` splitting of the data, so that we keep the ratio of frauds/non-frauds the same in train and validation sets. If we used random splitting, then in some training splits, 0 cases of fraud would appear. In that case, the algorithm would not learn anything useful.
# 
# 2) The train part of the data is passed to the ```fit``` method of our custom ```Scikit-learn``` estimator, where it is first resampled, using one of the afore-mentioned techniques. The ```fit``` method, implements the NN using the low-level or high-level ```Tensorflow``` API and trains the model by fetching batches of the * resampled * data. After the training is done, the ```fit``` method saves the model for later usage.
# 
# 3) The validation part of the dataset (which does not contribute to resampling), is passed to the ```score``` method of the ```Scikit-learn``` custom estimator. This method restores the trained model, and calculates the metric on the validation data. For the specific case, the metric that we have chosen is the Area Under the Precision-Recall Curve (AUCPRC). Another suitable metric could be the F1-score. As a reminder, precision measures how good the model is at predicting the positive class (here, representing frauds). At the same time, recall is a measure of how much we can trust our model when it predicts the positive class, i.e. how well it performs when we actually have a positive class. Both these measures need to be high. However, as our model will output probabilities (using ```softmax```), whether an example is positive (```Class=1```) or negative (```Class=0```) will also depend on the probability threshold that we choose to decide whether an example belongs to either class. By continuously modifying this threshold we can generate couples of operating points of precision and recall. These points form a curve, and we want its area to be as close to 1.0 as possible, meaning high precision and high recall.
# 
# 4) The process is repeated for every split scheme of the data and for every configuration of hyperparameters that we have chosen. The output is then comprised of AUCPRC scores for each split and each parameter configuration. 
# 

# ---

# For the ```Pipeline``` implementation, we create the following custom classes, which inherit from suitable parent classes. All the classes that implement ```fit``` and ```transform``` methods are transformers and are used in the first stages of the ```Pipeline```. Classes that implement ```fit```, ```predict``` and ```score``` methods are the estimators which occur at the end of the ```Pipeline```.

# In[ ]:


class ColumnSelector(BaseEstimator, TransformerMixin):
    
    """
    (Transformer)
    Class that implements selection of specific columns.
    The desired column or columns are passed as an argument to the constructor


    """

    def __init__(self, cols):

        """
        :param cols: desired columns to keep
        :return: the transformed dataframe

        """

        self.cols = cols

    def fit(self, X, y=None):

        """
        :param X: dataframe
        :param y: none
        :return: self

        """

        return self

    def transform(self, X):

        """

        :param X: dataframe
        :return: the dataframe with only the selected cols

        """

        # First check if X is a pandas DataFrame

        assert isinstance(X, pd.DataFrame)

        try:

            # Return the desired columns if all of them exist in the dataframe X
            return X[self.cols]

        except KeyError:

            # Find which are the missing columns, i.e. desired cols to keep that do not exist in the dataframe
            missing_cols = list(set(self.cols) - set(X.columns))

            raise KeyError("The columns: %s do not exist in the data" % missing_cols)
            
            
class Scaler(BaseEstimator, TransformerMixin):
    
    """
    (Transformer)
    Class that implements scaling.
    
    method: string, either 'normalize' or 'standardize'
        
    """

    def __init__(self, method):

        self.method = method

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        if self.method == 'normalize':

            return (X - X.min()) / (X.max() - X.min())

        elif self.method == 'standardize':

            return (X - X.mean()) / X.std()
        


# In[ ]:


class DataPrep(object):
    
    """
    (Transformer)
    Prepare data and implement pipeline


    columns_to_keep: list of which columns of the dataframe to keep
    normalization_method:string, 'normalize' or 'standardize' denoting the scaling type
    
    """

    ### Pass the desired arguments to the constructor

    def __init__(self, columns_to_keep,normalization_method):
        self.columns_to_keep = columns_to_keep
        self.normalization_method=normalization_method

    def pipeline_creator(self):
        
        """
        
        The pipelines are "trivial", but could be extended by adding more functionalities (e.g. select which cols to 
        normalize, deal with categorical cols...)
        
        """
        
        #Data Selection

        data_select_pipeline = Pipeline([

            ('Columns to keep', ColumnSelector(self.columns_to_keep)),

        ])

    
        norm_pipeline = Pipeline([
            
            ('Custom Scaling', Scaler(self.normalization_method))
        ])
        
        
        preprocess_pipeline=Pipeline([
            
            ('Data selection',data_select_pipeline),
            ('Data scaling',norm_pipeline)

        ])
        
        
        return preprocess_pipeline


# In[ ]:


class MyEstimator(BaseEstimator):
    
    """
    (Estimator)
    
    Class that implements our custom estimator
    
    """

    def __init__(self,sampling_method,learning_rate,batch_size,num_epochs,keep_prob,model_dir,model_name):
        
        """
        sampling_method:string, 'over' ,'under', 'over_SMOTE' ,'both'
        learning_rate: float, the learning rate of the algorithm
        batch_size:the batch size to feed each step of optimization
        num_epochs: int, iteration over the dataset
        keep_prob: probability to keep nodes at the dropout layer (if any)
        model_dir: where to save output for tensorboard visualization
        
        
        """
        
        self.sampling_method = sampling_method        
        self.learning_rate = learning_rate
        self.batch_size=batch_size
        self.num_epochs=num_epochs
        self.model_dir=model_dir
        self.keep_prob=keep_prob
        self.model_name=model_name
    
#     def make_tf_dataset(self,X,y,num_epochs,batch_size):
        
#         dataset=tf.data.Dataset.from_tensor_slices({'features':X,'labels':y})
#         dataset.shuffle(X.shape[1])
#         dataset.batch(batch_size)
#         dataset.repeat(num_epochs)
        
#         return dataset
    
    def resample_dataset(self,X,y):
        
        """
        Method that implements the resampling. X and y correspond to data after train/test split
        sampling_strategy=1 denotes that the ratio of classes will be equal in the resampled datasets
        
        Beware, in oversampling, huge datasets could result
        
        """
        
        if self.sampling_method=='over':
            
            sampler=RandomOverSampler(sampling_strategy=0.03,random_state=0)
            X_resampled,y_resampled=sampler.fit_resample(X,y)
            
        elif self.sampling_method=='under':
            
            sampler=RandomUnderSampler(sampling_strategy=0.03,random_state=0)
            X_resampled,y_resampled=sampler.fit_resample(X,y)
            
        elif self.sampling_method=='over_SMOTE':
            
            sampler=SMOTE(sampling_strategy=1,random_state=0)
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            
        elif self.sampling_method=='both':
            
            sampler=SMOTEENN(sampling_strategy=0.03,random_state=0)
            X_resampled, y_resampled = sampler.fit_resample(X, y['Class'].values)
        
        else:
            
            print('No resampling is used!')
            
            X_resampled=X
            y_resampled=y
        
        
        return X_resampled, y_resampled
        
        
    
        
    def fit(self,X,y):
        
        """
        
        The fit method should implement training.
        We resample the data, create the graph, run the training for num_epochs and for each step we feed in
        batch_size of examples. We finally save the model for usage by predict and score methods
        
        
        """
        
        #for debugging
        print('\n Number of input examples for fit: '+str(X.shape[0]))
        
        # undersampling/oversampling
        
        X_resampled, y_resampled=self.resample_dataset(X,y)
        
        y_resampled=np.asarray(y_resampled)
        y_resampled=np.reshape(y_resampled,(-1,))
        
        # Had to use float16 in order not to run out of memory; maybe it is not an issue if you run python script not in Jupyter
        X_resampled=X_resampled.astype(np.float16)
        y_resampled=y_resampled.astype(np.int16)
        
#         print(X_resampled.shape)
#         print(X_resampled.dtype)

#         print(y_resampled.shape)
        
        print('\n Number of examples after resampling: '+str(X_resampled.shape[0]))
        
        
        # Create the graph (placeholders, variables, ops...) and then train the model for num_epochs
        
        # Reset graph so no overlapping names occur accross multiple runs
        tf.reset_default_graph()

        """
        
        Create placeholders for the input. Every row of features correspond to the same row at output
        The dimensions can be transposed, and this will affect how the matrix multiplication is done, so a good practice
        would be to keep track of the dimensions, which we will append as comments in the code
        
        As a note, the values of the placeholders are not saved with the tf.train.Saver().save() method
        
        """
        # Features placeholder will have shape (batch_size,number_of_features)
        
        features=tf.placeholder(tf.float16,[None,X_resampled.shape[1]],name='features')
        
        # Labels placeholder will have shape(number_of_features,1)
        
        labels=tf.placeholder(tf.int32,[None,],name='labels')
        
        # We create a placeholder for the keep probability of the dropout layer.
        # !!! SET TO 1 DURING DEV/TEST
        
        prob_to_keep=tf.placeholder(tf.float16,name='keep_prob')
        
        '''
        As an example algorithm, we use a neural net (NN) with multiple layers. 
        Logistic regression can be modeled with a NN of 1 (hidden) layer with 1 node
        
        By using more hidden layers, we let the network learn more complex functions of the input 
        than just a linear combination of it
        
        '''
        
        # Define the number of hidden layers and nodes. 
        # As a rule, we keep the ratio of nodes between consequtive hidden layers fixed
        
        ratio=1.5
        
        # The input nodes regard the input layer and correspond to the number of features
        
        input_nodes=X_resampled.shape[1]
        
        # Nodes in the first hidden layer
        
        hidden_nodes_1=8
        
        # Nodes in the second hidden layer
        
        hidden_nodes_2=round(hidden_nodes_1*ratio)
        
        # Nodes in the third hidden layer
        
        hidden_nodes_3=round(hidden_nodes_2*ratio)
        
        # Output nodes: we have class 0 and class 1
        # With one output node, the output of the network will yield the probability of class=1, i.e. the probability to have fraud.
        # Since the classes are mutually exclusive, the probability of no fraud will be 1-P(fraud)
        
        output_nodes=2
        
        # Construct hidden layer 1
        
        with tf.name_scope('Hidden_Layer_1'):
        
            W1 = tf.get_variable('W1',[input_nodes,hidden_nodes_1],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float16)
            b1 = tf.get_variable('B1',[hidden_nodes_1],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float16)
            
            # out_1 will be matrix multiplication (batch_size,number_of_features)*(input_nodes,hidden_nodes_1)
            # out_1 will be of shape (batch_size,hidden_nodes_1)
            
            # We use ReLU non-linearities to speed-up learning
            out_1=tf.nn.relu(tf.matmul(features,W1)+b1,name='out_1')
            
            
            tf.summary.histogram('Weights',W1)
            tf.summary.histogram('Biases',b1)
            tf.summary.histogram('Activations',out_1)
            
        
        # Construct hidden layer 2
         
        with tf.name_scope('Hidden_Layer_2'):
            
            W2=tf.get_variable('W2',[hidden_nodes_1,hidden_nodes_2],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float16)
            
            b2 = tf.get_variable('B2',[hidden_nodes_2],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float16)

            
            # out_2 will be matrix multiplication (batch_size,hidden_nodes_1)*(hidden_nodes_1,hidden_nodes_2)
            # out_2 will be of shape (batch_size,hidden_nodes_2)

            out_2=tf.nn.relu(tf.matmul(out_1,W2)+b2,name='out_2')
            
            out_2=tf.nn.dropout(out_2,prob_to_keep,name='out_2_dropout')
        
            tf.summary.histogram('Weights',W2)
            tf.summary.histogram('Biases',b2)
            tf.summary.histogram('Activations',out_2)
            
        # Construct hidden layer 3 (comment out if needed)
        '''
        with tf.name_scope('Hidden_Layer_3'):
            
            W3=tf.get_variable('W3',[hidden_nodes_2,hidden_nodes_3],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float16)
            
            b3 = tf.get_variable('B3',[hidden_nodes_3],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float16)

            # out_3 will be matrix multiplication (batch_size,hidden_nodes_2)*(hidden_nodes_2,hidden_nodes_3)
            # out_3 will be of shape (batch_size,hidden_nodes_3)
            
            out_3=tf.nn.relu(tf.matmul(out_2,W3)+b3,name='out_3')
            
            out_3=tf.nn.dropout(out_3,prob_to_keep,name='out_3_dropout')
            
            
            tf.summary.histogram('Weights',W3)
            tf.summary.histogram('Biases',b3)
            tf.summary.histogram('Activations',out_3)
        '''    
        
        # construct hidden layer 4 (modify the dimensions accordingly)
        
        with tf.name_scope('Output_Layer'):
            
            W4 = tf.get_variable('W4',[hidden_nodes_2,output_nodes],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float16)
            b4 = tf.get_variable('B4',[output_nodes],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float16)

            
            # out_4 will be matrix multiplication (batch_size,hidden_nodes_3)*(hidden_nodes_3,2)
            # out_4 will be of shape (batch_size,2)
            
            # We do not apply any non-linearity to the output, as this will be taken care by the loss operation

            out_4=tf.add(tf.matmul(out_2,W4),b4,name='out_4')
            
            
            tf.summary.histogram('Weights',W4)
            tf.summary.histogram('Biases',b4)
            tf.summary.histogram('Activations',out_4)
               
            
        with tf.name_scope('Loss'):
            
            loss=tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=out_4),name='loss')
            
            # We decide to calculate and keep only the loss. 
            # Accuracy is not so good a metric when we have imbalanced datasets
            
            tf.summary.scalar('loss',loss)
        
       
        optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate,epsilon=0.05,name='optimizer')
        train_op=optimizer.minimize(loss,name='training_objective')
                      
        # Define an operation to initialize global variables    
        init=tf.global_variables_initializer()
        
        merged_summary_op=tf.summary.merge_all()
        
        with tf.Session() as sess:
            
            # Initialize the global variables
            
            sess.run(init)
            
            print('\n Training has started...')
            
            
            # Define operation to write to tensorboard
            
            summary_writer = tf.summary.FileWriter(self.model_dir, graph=tf.get_default_graph())
            
            
            # Train the network for num_epochs (iterations over dataset)
            
            for epoch in range(self.num_epochs):
                
                # For each epoch, reset the training cost
                
                batch_cost=0
                
                num_batches=int(X_resampled.shape[0]/self.batch_size)
                
                # Each optimization step will be done using a batch of the data of size batch_size
                
                for batch in range(num_batches):                    
                    
                    batch_x=X_resampled[batch*self.batch_size:(batch+1)*self.batch_size,:]
                    batch_y=y_resampled[batch*self.batch_size:(batch+1)*self.batch_size]
                    
                                    
                    _,temp_cost,summary=sess.run([train_op,loss,merged_summary_op],
                                                 feed_dict={features:batch_x,labels:batch_y,prob_to_keep:self.keep_prob})
                    
                    # Calculate an average cost over the number of batches
                    
                    batch_cost+=temp_cost/num_batches
                                        
                    # Write all the selected variables for every iteration 
                    summary_writer.add_summary(summary=summary, global_step=epoch * num_batches + batch )
                
                    to_string='Minibatch:'+str(batch)+'/'+str(num_batches)
                
                    sys.stdout.write('\r'+to_string)
                
                # Print cost (training loss) at regular intervals
                if epoch % 10 ==0:
                    
                    print('\nTraining cost at epoch {} : {}'.format(epoch,batch_cost))
                    
            
            print('\n Optimization finished! \n')
        
        
        
        # Save the model. We have to do that inside the session.
        # We decide to keep the last iteration
        
            saver=tf.train.Saver()
            print('\n Saving trained model...\n')
            saver.save(sess,'./'+self.model_name)
        
        
    def predict(self,X):
        
        """
        Predict the output probabilities from a trained model. First we restore the model and load any desired tensors
        and nodes. We add on top the softmax layer, create the corresponding op and run the session
        
        """
        
        with tf.Session() as sess:
            
            
        # this loads the graph
            saver = tf.train.import_meta_graph(self.model_name+'.meta')
        
        # this gets all the saved variables
            saver.restore(sess,tf.train.latest_checkpoint('./'))
            
        # get the graph
        
            graph=tf.get_default_graph()
            
            # get placeholder tensors to feed new values
            features=graph.get_tensor_by_name('features:0')
#             labels=graph.get_tensor_by_name('labels:0')
            keep_prob=graph.get_tensor_by_name('keep_prob:0')
            
            # get the desired operation to restore. this will be the output of the last layer
            op_to_restore=graph.get_tensor_by_name('Output_Layer/out_4:0')
            
            # For prediction the keep_prob of the dropout layer will be equal to 1, i.e. no dropout
            logits=sess.run(op_to_restore,feed_dict={features:X,keep_prob:1.0})
            
            # The output of this operation needs to be passed to a softmax layer in order to get as output probabilities
            # Define the necessary op
            softmax=tf.nn.softmax(logits)
            
            # Run the op
            probabilities=sess.run(softmax)
            
            # The output will be of shape (num_examples,2)
            # The first column corresponds to P(Class=0) , that is no fraud
            # The second column corresponds to P(Class=1), that is fraud
            
            return probabilities
            
    
    
    
    '''

    Our custom estimator needs to implement a score method that will be used 
    to select the best custom score using grid search
    
    For such imbalanced dataset that we have, a good metric, 
    as alredy discussed, will be the area under the precision-recall curve
        
    '''
    
    def score(self,X,y):
        
        """
        
        Scorer function to be used for selection of the best parameters.
        Our scorer function will calculate the area under the precison recall curve
        
        
        """
        
        print('\n Scoring using '+str(X.shape[0])+' examples')
        
        # Get the probabilities for each class
        probs=self.predict(X)
        
        # Get the probabilities for fraud for each example
        fraud_probs=probs[:,1]
        
        print('\n Got probabilities for '+str(X.shape[0])+' examples')
        
        # Define the operation to get the area under the precision-recall curve
        
        with tf.name_scope('Scoring'):
        
            area=tf.metrics.auc(labels=y,
                            predictions=fraud_probs,
                            curve='PR',
                            summation_method='careful_interpolation',
                            name='AUCPRC')
        
        # Define an initialization operation on the global AND local variables
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        
        # Create session and run
        
        with tf.Session() as sess:
            
            # Initialize the variables
            sess.run(init)
            
            # Run the area node to calculate AUCPRC
            out=sess.run(area)
        
        return out[1]
    
                


# In[ ]:


# use 'under' for resampling to get results quicker, as here the dataset is smaller

params={'sampling_method':['under'],'learning_rate':[0.003,0.01],'batch_size':[128],
        'num_epochs':[100],'keep_prob':[0.35,0.7],'cols':['full','discard'],'cv_splits':[3]}

params=ParameterGrid(params)
scores_runs=[]

for run,item in enumerate(params):
    
    if item['cols']=='full':
        
        cols=features.columns
        pipeline_data=DataPrep(cols,'standardize').pipeline_creator()
        
    elif item['cols']=='discard':
        # drop the anonymised features with the smallest wasserstein distance.
        cols=features.drop(['V28','V27','V26','V25','V24','V23','V22','V20','V21','V19','V15','V13','V6'],axis=1).columns
        pipeline_data=DataPrep(cols,'standardize').pipeline_creator()

    
    folds=StratifiedKFold(n_splits=item['cv_splits'],shuffle=True,random_state=42)
        
    sampling_method=item['sampling_method']
    learning_rate=item['learning_rate']
    batch_size=item['batch_size']
    num_epochs=item['num_epochs']
    keep_prob=item['keep_prob']
    
    full_pipeline=Pipeline([
    
    ('data prep',pipeline_data),
    ('estimator',MyEstimator(sampling_method,learning_rate,batch_size,num_epochs,keep_prob,model_dir='./tmp_over'+str(run),
    model_name='over-model-run'+str(run)))

    ])
        
    scores=cross_val_score(full_pipeline,features,labels,cv=folds.split(features,labels))
    
    scores_runs.append(scores)
    
    np.save('scores'+str(run),scores)
    


# Let us collect the results from the saved ```.npy``` files and create a dataframe:

# In[ ]:


# The parameters used when undersampling
params_under={'sampling_method':['under'],'learning_rate':[0.003,0.01],'batch_size':[128],
        'num_epochs':[100],'keep_prob':[0.35,0.7],'cols':['full','discard'],'cv_splits':[3]}


params=ParameterGrid(params_under)

# make a dataframe containing as entries the parameters dictionary for each run
results=pd.DataFrame([item for item in params])


# Create an ```Numpy``` array containing all the scores of every run for easy access

# In[ ]:


scores=[]

for run,item in enumerate(params):
    run_scores=np.load('scores'+str(run)+'.npy')
    scores.append(run_scores)

    
scores_array=np.array(scores)
scores_array


# Create new columns (3 for each cross-validation split) and assign to each of them the corresponding AUCPRC score

# In[ ]:


results['cv_1']=scores_array[:,0]
results['cv_2']=scores_array[:,1]
results['cv_3']=scores_array[:,2]


# In[ ]:


results['mean_score']=results[['cv_1','cv_2','cv_3']].mean(axis=1)
results


# # Some final thoughts...

# After an analysis of the data, we have concluded the following:
# 
# 1) Highly imbalanced dataset
# 
# 2) Given features need to be scaled. Some of them could be discarded
# 
# The high-imbalanced nature comes from the fact that fraudulent transactions are much more rare than normal transactions. This high imbalance is the most important attribute of this dataset, that determines the classification methodology/procedure. More specifically, the metric on which we evaluate our classifiers should no longer be accuracy, as we could achieve a very high accuracy, simply by predicting every time ```no fraud```. We are now interested in predicting correctly the fraudulent cases, and when we predict fraud, we need to be as sure as possible that this is fraud. We are less interested in cases where we predict fraud but it is not a case of fraud in reality (false positive). The metric on which we evaluate therefore, should capture these facts. We need high 
# 
# * precision = True Positives / (True Positives + False Positives) (how good we detect positive class, i.e. fraud)
# * recall = True Positives / (True Positives + False Negatives) (how good quality is our positive class prediction)
# 
# We have chosen to use as a metric the area under the Precision-Recall curve. High precision and recall for positive class should yield high area under the curve. One important notice here is that we evaluate on unseen data, that is data that have splitted * before * doing any resampling of the dataset. This is done to ensure that we optimize on data that come from distributions we would expect to see in reality. 
# 
# It would not make sense therefore, to use this metric on the training set and compare it with the corresponding one on the validation set. For the training set, we measure during training the loss. The lower the loss, the better we have learnt the training set. However, this does not mean that it would perform well on the validation set too. We have two separate jobs: do well on the training set and optimize w.r.t. the validation set (orthogonal training). Early stopping (stopping the training in order to achieve high area under the curve) does not fall into this category. Low loss and bad metric would mean that the model does well on the training set but is too specific to it; it does not generalize well to new examples. One solution would be to make the model simpler, introduce regularization (here done with dropout layers), get more examples (change from undersampling to over-sampling or over-sampling with SMOTE or a combination of both - too computationally expensive).
# 
# 
# Finally regarding the importance of the features, the method followed here is by no means the only one. Alternatives would include:
# 
# * We could explore the correlation coefficients between the output and the values of the features (show the linearity of the relations)
# * We could start building a model including all of the features and at each step we could decide which feature to remove in order to get a better performance
# * Same as before, but starting from one feature and deciding which one to add in order to boost performance
# * Ridge regression would take some of the feature coefficients to 0. Based on the remaining coefficients, we could build more complex models.
# * Decision tree algorithms have inherently the notion of feature importance

# In[ ]:




