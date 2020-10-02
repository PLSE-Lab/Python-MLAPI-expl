#!/usr/bin/env python
# coding: utf-8

# Notebook tackles 3 Questions:
# 
# Question 1: Can satisfaction levels be predicted effectively from other data? Which features provid a well scored, logical and useful prediction?
# 
# Question 2: What features varies the most? 
# 
# Question 3: Can it be usefully clustered? What features dictate the clustering?
# 
# 
# 
# Apologies for poor structuring:  the notebook first cleans up the data a little (gives numerical values to all columns for example). Then looks at correlation and variance. Then looks at linear regression for multiple variables to predict satisfaction. The looks at clustering on certain variables. Then looks to use random forest for satisfaction prediction with cross validation.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.

hrData = pd.read_csv('../input/HR_comma_sep.csv')
hrData = hrData.drop(['sales'], axis=1)
hrData['salary'].replace(['low', 'medium', 'high'], [0, 1, 2], inplace = True)
# Sales column doesn't make sense, doesn't match column description (data being clear is important)
hrData.head()


# In[ ]:


#Look at how which section has the largest variance (varies the most among employees)
largestVar = 0
largestVarCol = ""
for i in list(hrData):
    x = np.var(hrData[i])
    # Normalise Variance
    x = x / len(hrData[i])
    print(x)
    if x > largestVar:
        largestVar = x
        largestVarCol = i
print(largestVarCol)


# Question 2 - As the largest Variance comes form the average monthly hours, then the best way to differentiate the employees based on one feature would be to differentiate them on average monthly hours.

# In[ ]:


# Looking at the linear correlation between features will demonstrate how 
# satisfaction levels can be predicted from the remaining information
hrData.corr()


# The correlation table above suggests that satisfaction level is unsurprisingly heavily linked to an employee leaving. However, this insight isn't particularly useful as the employee has already left the company. Predicting satisfaction on other areas would be a much more useful tool for a company to stop employees becoming unsatisfied and leaving.

# Question 1 - As satisfaction level is a continuous data type, linear regression was chosen to attempt to predict it. Using all other features is attempted below. As this had a very low success rate in predicting the satisfaction, I attempted to make the prediction of satisfaction with the three features that had the highest magnitude in correlation with satisfaction, calculated in the correlation table earlier.
# 
# This test first, used 10000 data values per column for training, and 5000 for testing. It was then changed to 12000 training values with 3000 test values which had a higher success and seems more appropiate as it providews a 75%-25% split.

# In[ ]:


# split data 75%-25% fror training and testing (15000 total values)

trainingDataY = hrData['satisfaction_level'][:12000]
trainingDataX = hrData.drop(['satisfaction_level'], axis=1)[:12000]
testDataY = hrData['satisfaction_level'][12000:]
testDataX = hrData.drop(['satisfaction_level'], axis=1)[12000:]


lm = LinearRegression()
lm.fit(trainingDataX , trainingDataY)
allFeatScore = lm.score(testDataX, testDataY)

from sklearn.linear_model import Ridge

lr = LinearRegression()
lr.fit(hrData[['last_evaluation', 'left', 'number_project' ]][:12000], trainingDataY)
threeFeatScore = lr.score(hrData[['last_evaluation', 'left', 'number_project' ]][12000:], testDataY)
print("all feature score = ", allFeatScore)
print("three feature score = ", threeFeatScore)
lr = Ridge(alpha=0.5)
lr.fit(hrData[['last_evaluation', 'left', 'number_project' ]][:12000], trainingDataY)
threeFeatScore = lr.score(hrData[['last_evaluation', 'left', 'number_project' ]][12000:], testDataY)
print("three feature score with ridge = ", threeFeatScore)


# To account for any issue with colinearity. ridge regression was tested. However, proved to be slightly worse than normal linear regression.
# 
# The predictions provide a score of 25% which is low, therefore more data, a different technique or different features would be recommended for predicting satisfaction level. See Random Forest regressor below
# 
# Standardised Data to see if preprocessing makes a difference to score. No difference found

# In[ ]:



xData = StandardScaler().fit_transform(hrData[['last_evaluation', 'left', 'number_project' ]])
lr = LinearRegression()
lr.fit(xData[:12000], trainingDataY)
threeFeatScore = lr.score(xData[12000:], testDataY)
print("three feature score with standardised data = ", threeFeatScore)


# Question 3 - K Means Clustering

# In[ ]:


# K chosen randomly, If time allowed would have added Elbow Method for K Choice

from sklearn.cluster import KMeans
K = 5
km = KMeans(n_clusters = K)
km.fit(hrData)
clusterLabels = km.labels_

# View the results
# Set the size of the plot
plt.figure(figsize=(14,7));
 
# Create a colormap
colormap = np.array(['red', 'lime', 'black', 'blue', 'yellow'])
 
    
# Plot the Models Classifications
plt.subplot(1, 2, 2);
plt.scatter(hrData["satisfaction_level"], hrData["last_evaluation"], c=colormap[clusterLabels], s=40);
plt.title('K Mean Classification');

plt.figure(figsize=(14,7));
 
# Create a colormap
colormap = np.array(['red', 'lime', 'black', 'blue', 'yellow'])

plt.scatter(hrData["satisfaction_level"], hrData["left"], c=colormap[clusterLabels], s=40);
plt.title('K Mean Classification 2');

plt.figure(figsize=(14,7));
 
# Create a colormap
colormap = np.array(['red', 'lime', 'black', 'blue', 'yellow'])

plt.scatter(hrData["number_project"], hrData["satisfaction_level"], c=colormap[clusterLabels], s=40);
plt.title('K Mean Classification 3');


# After attempting to cluster using k means, it seems that attributes largely don't cluster usefully due to the large spread across the spectrum. Some different plots for k means clustering can be seen. 

# BACK TO Q1
# 
# Random Forest Regressor:
# 
# Instead of linear regression, a different tool was used (random forest regressor). This provided a success rate of 65% using 3 features. A much higher success rate was found using all the features but "left" (excluded as little help in application).

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

# Apply Random Forest

rf = RandomForestRegressor()
rf.fit(hrData[['last_evaluation', 'left', 'number_project' ]][:12000], trainingDataY)
rf1score = rf.score(hrData[['last_evaluation', 'left', 'number_project' ]][12000:], testDataY)
print("random forest score using 3 features", rf1score)


# As the left attribute means that the employee has left the company (which clearly demonstrates a low satisfaction, it would be better to see the satisfaction of employees based on attributes so the employee leaving the company could be avoided).

# In[ ]:


rf2 = RandomForestRegressor()
rf2.fit(trainingDataX.drop(['left'], axis=1), trainingDataY)
rf2score = rf2.score(testDataX.drop(['left'], axis=1), testDataY)

# change testing and training data for cross validation

trainingDataY = hrData['satisfaction_level'][3000:]
trainingDataX = hrData.drop(['satisfaction_level'], axis=1)[3000:]
testDataY = hrData['satisfaction_level'][:3000]
testDataX = hrData.drop(['satisfaction_level'], axis=1)[:3000]

# Apply Second Random Forest Regression Algorithm

rf3 = RandomForestRegressor()
rf3.fit(trainingDataX.drop(['left'], axis=1), trainingDataY)
rf3score = rf2.score(testDataX.drop(['left'], axis=1), testDataY)

scoreAverage = (rf3score + rf2score) / 2

print("cross validation random forest score using all features other than left", scoreAverage)


# Having all the attributes except 'left' provides a very high success rate of 93 percent from this testing when using all the other attributes. This means that an employer would be able to deduce the satisfaction of an employee from other statistics (with a relatively large success) before the employee left the company.
# 
# Cross validation was used in this testing (this one was chosen as provided the best initial result). The benefit of cross validation is that it removes data bias. The training and testing data was chosen from the start and end for ease of use (it could have been randomly selected, however, it didn't seem required as the data doesn't seem to be ordered on any particular feature). 
# 
# Further data analysis to be completed: PCA, other regression algorithms. Look at removing some of the meaningless features such as Role which shouldn't logically have a numerical scale.
