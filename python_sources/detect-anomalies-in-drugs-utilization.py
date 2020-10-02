#!/usr/bin/env python
# coding: utf-8

# # GOAL
# * Using 13,260 samples with 11 features identify 3 anomalies
# * 3 / 13260 = **0.02%** of samples
# * Unsupervised learning, anomaly detection challenge 
# * Advantage of unsupervised learning: it can detect anomalies it has never before seen (supervised learning can't do that)
# * There are three main parts to this notebook: Data, Model, Results 

# In[ ]:


# Import modules

import numpy as np
from numpy import ma
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib import ticker, cm
import matplotlib.colors as colors
get_ipython().run_line_magic('matplotlib', 'inline')

from scipy.stats import multivariate_normal
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import os
print(os.listdir("../input"))


# # DATA

# In[ ]:


# LOAD DATA

dfRaw = pd.read_csv('../input/DrugsUtilizationAnomalyDetection.csv')
print(dfRaw.shape)
print(dfRaw.columns)
dfRaw.drop('Unnamed: 0', axis=1, inplace=True)
print(dfRaw.columns)
dfRaw.head()


# In[ ]:


dfRaw.describe()


# In[ ]:


# Check for MISSING data

print('Any missing value ?',dfRaw.isnull().values.any())


# In[ ]:


# Histograms of all cols to view distribution of raw data

for colName in dfRaw.columns:
    dfRaw.groupby(colName).size().plot.bar()
    plt.show()


# # One Hot Encoding (OHE)
# * Without OHE the model will find a relationship between doc_id 545 and 546 
# * This is incorrect representation of the doc_id as there is NO relationship between doc 545 and 546 just because of their numbers
# * The above is true for ALL cols with CARDINAL data (menus, items arranged in a list, etc)
# 

# In[ ]:


# Make the doc id and event id strings so they will be One Hot Encoded (integers are not OHE)

dfRaw['doctor_id'] = dfRaw['doctor_id'].apply(str)
dfRaw['event_id'] = dfRaw['event_id'].apply(str)


# One Hot Encoder

print('Before One Hot Encoder',dfRaw.shape)
dfDum = pd.get_dummies(dfRaw)
print('After One Hot Encoder',dfDum.shape)
dfDum.head()


# # Normalization
# * NORMALIZE the data in order to remove any false importance of large numbers
# * Otherwise the model will give 20 times more importance to 1 mg of morphine vs 0.05 mg of fentanyl

# In[ ]:


# NORMALIZE data

x = dfDum.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
dfNorm = pd.DataFrame(x_scaled)

print(dfNorm.shape)
dfNorm.head()


# # MODEL

# # PCA - Primary Component Analysis 
# * PCA is not to be confused with Patient Controlled Analgesia...  ;-)
# * A sparse data matrix may prevent a model (supervised or unsupervised) from learning meaningful wisdom
# * PCA reduces a very sparse matrix 13,260 X 6,550 (most of it being zeros) to 2 dimensions 
# * WHILE MINIMIZING the VARIANCE loss in the process 
# * PCA is also used for visualization purposes as humans can't grasp more than 4-5 dims
# * See below the data projected on 2 dims

# In[ ]:


# PCA  

pca = PCA(n_components = 2) 
dataPostPCA = pca.fit_transform(dfNorm)

print('Before PCA', dfNorm.shape)
print('Afer PCA',dataPostPCA.shape)


# # Multi-variate Gaussian Prob Distribution 
# * The prob learned is for each coordinate X and Y in the 2 dims post PCA dataset
# * The prob of each event occuring is derived from this Multi-variate Gaussian Prob Distribution
# * min prob of normal data = the occuring prob of the set of X and Y which is the RAREST
# * See below how to use this min prob in order find a specific number of anomalies

# In[ ]:


# Calculate the Multi-variate Gaussian Prob Distribution and the min/max prob of normal data

p = multivariate_normal(mean=np.mean(dataPostPCA,axis=0), cov=np.cov(dataPostPCA.T))

x = p.pdf(dataPostPCA)

print("min prob of x dataPostPCA", min(x))
print("max prob of x dataPostPCA", max(x))


# # Set the level of Anomaly Detection
# * 0.01 = 1% probability of occurence  
# * PLAY with this parameter to detect a specific number of anomalies
# * Start with a level around the min prob found above
# * Function below displays the normal data  with prob > Anomaly Level on the Gaussian 2 dims prob distribution as purple points
# * And depending on the Anomaly Level it display as red points the anomalies with prob < AL

# In[ ]:


# Set the level of Anomaly Detection

AnomaLevel = 0.0035 

NumRows = dataPostPCA.shape[0]

# Display the normal data on the Gaussian 2 dims prob distribution for SampleNormal
x, y = np.mgrid[-2:2:0.1, -2:2:0.1]  # See the whole prob distribution

pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
fig, ax = plt.subplots()
cs = ax.contourf(x, y, p.pdf(pos), 20)
cbar = fig.colorbar(cs)

# Location on chart of the anomaly points in red = BELOW AnomaLevel (purple = above the AnomaLevel)
anomalPostPCA = np.zeros((NumRows,2))
probsPostPCA = np.zeros(NumRows)
ya = np.zeros(NumRows)
#ya = []
anomCounter = 0
anomList = []
anomProbList = []

print('Below the anomaly level of ', AnomaLevel*100, '% probability of occurence')
for j in range(0,NumRows):
    x = p.pdf(dataPostPCA[j])
    if x >= AnomaLevel:
        color="purple"
        plt.scatter(dataPostPCA[j][0],dataPostPCA[j][1], color=color)
        #pass # To see the dots OVER the Anomaly Level (ALL of them) - uncomment the above and comment me ... it takes a couple of mins
    else:
        color="red"
        plt.scatter(dataPostPCA[j][0],dataPostPCA[j][1], color=color)
        anomalPostPCA[j] = dataPostPCA[j]
        probsPostPCA[j] = round(x*100,5)
        ya[anomCounter] = round(x*100,5)
        anomList.append(anomalPostPCA[j])
        anomProbList.append(probsPostPCA[j])
        print(anomalPostPCA[j],' has ',probsPostPCA[j],'% of occurence')
        anomCounter = anomCounter + 1
print(anomCounter, ' anomalies') 

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(12,12)
plt.show()


# # RESULTS
# * The above identified 3 anomalies, defined as having a prob of occurence LESS than the Anomaly Level (set at 0.35%)
# * We have these anomalies' 2 dims post PCA and their index, ordered row number
# * Using this index we identify the anomalies in their original format

# In[ ]:


# Find the ORIGINAL anomalies rows using the 2 dims values post PCA (as NO shuffle of the data was involved)

PostPCAdf = pd.DataFrame(dataPostPCA)
#print('PostPCAdf', PostPCAdf.shape)
print('row num and its % of occurence')      

AnomRowList=[]

for n in range(PostPCAdf.shape[0]):
    for k in range(len(anomList)):
        if math.isclose(PostPCAdf[0][n], anomList[k][0], rel_tol=1e-5) and math.isclose(PostPCAdf[1][n], anomList[k][1], rel_tol=1e-5):
            AnomRowList.append(n)
            print(n, anomProbList[k] , '% occurence')
      
dfRaw.iloc[AnomRowList]


# * I've tried below to clarify the details of the doc 686 
# * Other similar anomalies
# * Stumbled upon the fourth anomaly ... also doc 686
# * Modifying Anomaly Level from 0.35% to 0.36% uncovers this 4th anomaly ...also doc 686

# In[ ]:


# All instances of this doc
dfRaw[dfRaw['doctor_id']=='686']


# In[ ]:


# Similar instances to the anomalies

dfRaw[(dfRaw['amount_result']=='too_much_dispense') & 
      (dfRaw['event_resolution']=='user_reconciled')& 
      (dfRaw['med_name']=='HYDROmorphone')]


# In[ ]:


# IF you add another condition below for dfRaw['doctor_id']=='686' ...
# Doc 686 has another suspicious instance 
# The chance for this FOURTH anomaly, which is with the same doc (id = 686) is 0.35478 %

dfRaw[(dfRaw['amount_result']=='too_much_dispense') & 
      (dfRaw['event_resolution']=='user_reconciled')& 
      (dfRaw['med_name']=='HYDROmorphone') &
      (dfRaw['doctor_id']=='686')
     ]


# * Modifying Anomaly Level to 0.4% adds two unrelated docs to the anomalies
