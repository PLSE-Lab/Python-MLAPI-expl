#!/usr/bin/env python
# coding: utf-8

# ![https://storage.googleapis.com/kaggle-datasets-images/25855/32949/ed5875d8aff7555fb428720964cc173f/dataset-cover.jpg?t=2018-05-09-14-31-28](http://storage.googleapis.com/kaggle-datasets-images/25855/32949/ed5875d8aff7555fb428720964cc173f/dataset-cover.jpg?t=2018-05-09-14-31-28)
# 
# PREDICTING A PULSAR STAR
# Dr Robert Lyon
# 
# HTRU2 is a data set which describes a sample of pulsar candidates collected during the High Time Resolution Universe Survey .
# 
# Pulsars are a rare type of Neutron star that produce radio emission detectable here on Earth. They are of considerable scientific interest as probes of space-time, the inter-stellar medium, and states of matter .
# 
# As pulsars rotate, their emission beam sweeps across the sky, and when this crosses our line of sight, produces a detectable pattern of broadband radio emission. As pulsars
# rotate rapidly, this pattern repeats periodically. Thus pulsar search involves looking for periodic radio signals with large radio telescopes.
# 
# Each pulsar produces a slightly different emission pattern, which varies slightly with each rotation . Thus a potential signal detection known as a 'candidate', is averaged over many rotations of the pulsar, as determined by the length of an observation. In the absence of additional info, each candidate could potentially describe a real pulsar. However in practice almost all detections are caused by radio frequency interference (RFI) and noise, making legitimate signals hard to find.
# 
# Machine learning tools are now being used to automatically label pulsar candidates to facilitate rapid analysis. Classification systems in particular are being widely adopted,
# which treat the candidate data sets as binary classification problems. Here the legitimate pulsar examples are a minority positive class, and spurious examples the majority negative class.
# 
# The data set shared here contains 16,259 spurious examples caused by RFI/noise, and 1,639 real pulsar examples. These examples have all been checked by human annotators.
# 
# Each row lists the variables first, and the class label is the final entry. The class labels used are 0 (negative) and 1 (positive).
# 
# Attribute Information:
# Each candidate is described by 8 continuous variables, and a single class variable. The first four are simple statistics obtained from the integrated pulse profile (folded profile). This is an array of continuous variables that describe a longitude-resolved version of the signal that has been averaged in both time and frequency . The remaining four variables are similarly obtained from the DM-SNR curve . These are summarised below:
# 
# Mean of the integrated profile.
# Standard deviation of the integrated profile.
# Excess kurtosis of the integrated profile.
# Skewness of the integrated profile.
# Mean of the DM-SNR curve.
# Standard deviation of the DM-SNR curve.
# Excess kurtosis of the DM-SNR curve.
# Skewness of the DM-SNR curve.
# Class
# HTRU 2 Summary
# 17,898 total examples.
# 1,639 positive examples.
# 16,259 negative examples.
# 
# Source: https://archive.ics.uci.edu/ml/datasets/HTRU2
# 
# Dr Robert Lyon
# University of Manchester
# School of Physics and Astronomy
# Alan Turing Building
# Manchester M13 9PL
# United Kingdom
# robert.lyon '@' manchester.ac.uk

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


path = "/kaggle/input/predicting-a-pulsar-star/pulsar_stars.csv"
stars = pd.read_csv(path)
results = []
print("size of our data : ", len(stars))
stars.head()


# Obs : targe_class = 0 means that the star is not a "Pulsar Star", whereas 1 is a "Pulsar Star"

# In[ ]:


print("Checking if there is missing values :\n",stars.isnull().sum())


# Lets see the correlations between the features!

# In[ ]:


stars.columns
stars.rename(columns={" Excess kurtosis of the DM-SNR curve": " Excess kurtosis of the DM SNR curve"," Skewness of the DM-SNR curve":"Skewness of the DM SNR curve", " Mean of the DM-SNR curve" : " Mean of the DM SNR curve", " Standard deviation of the DM-SNR curve" : " Standard deviation of the DM SNR curve" },inplace=True)
stars.columns


# In[ ]:


cor = stars.corr()


# This correlation matrix is visualized for instance with a heatmap plot:

# In[ ]:


import seaborn as sns; sns.set()
plt.figure(figsize=(18,10))


ax = sns.heatmap(
    cor, 
    center=0,
    vmin = -1, vmax = 1.0,
    linewidth=.9,
    cmap =  sns.color_palette("RdBu_r", 7),#cmap="YlGnBu",
    annot=True,
    square=True,

)


# As a symetrical matrix, we can just visualize its half:

# In[ ]:


mask = np.zeros_like(cor)
mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(18, 10))
    ax = sns.heatmap(cor, center = 0, linewidth = 0.9, vmin = -1, vmax = 1, 
    cmap =  sns.color_palette("RdBu_r", 7),annot = True, mask=mask, square=True)
    


# In[ ]:


corr_m = cor.abs()
sol = (corr_m.where(np.triu(np.ones(corr_m.shape), k=1).astype(np.bool))
                 .stack()
                 .sort_values(ascending=False))
sol[:3]


# > Getting the 3 strongest correlations with Pearsonr: (we can notice that from the previous correlation matrix, the first 2 are almost the equals)

# In[ ]:


from heapq import nlargest
from operator import itemgetter
import itertools
from itertools import combinations
from scipy import stats
all_cor = []
all_cord = {}
for one,two in itertools.combinations(stars.columns,2):

    v,_ = stats.pearsonr(stars[one],stars[two])
    all_cor.append(f"Correlation btw {one} and {two} is: {v:.4f}")
    all_cord[one+"-"+two] = round(v,4)

all_cor

m = dict(sorted(all_cord.items(), key = itemgetter(1), reverse = True)[:3])
print("The 3 strongest correlations are : ",m)


# For better visualization, we can plot some scatter graphs to see the correlations between the features. We will not display all possible combinations. But you can do that uncommenting the the first foor loop!

# In[ ]:



import matplotlib.pyplot as plt

i = 0
plt.figure()
f, ax = plt.subplots(1, 3, figsize=(21, 7), sharex=True)
#for first,sec in itertools.combinations(stars.columns.drop(stars.columns[len(stars.columns)-1]),2):
for key,val in m.items():
    #print(first,sec)
    pair = key.split("-")
    first = pair[0]
    sec = pair[1]
    
    sns.scatterplot(x=first, y=sec, data=stars, hue="target_class", ax = ax[i])
    if i == 2:
        break
    i += 1
    


# 1. From the above graphs, we can easily observe the positive correlation between the features as the linear relation between them has a positive coefficient (an increase in one of the variables result in an increase of the other one).
# 2. Mean of the DM SNR curve doesn't tell us much about the nature of the star (we have Pulsar and not-Pulsar stars from all the possible range of this variable)
# 3. In the other hand, the Standard Deviation of the DM SNR says us that stars with this variable with values < 40 have a great probability to be a Pulsar
# 4. The same for the Skewness of the Integrated Profile: stars with values > 10 has a great probability to be a Pulsar

# In[ ]:


fr = 0.1
vsize = int(len(stars)*fr)

train = stars[:-2*vsize]
valid = stars[-2*vsize:-vsize]
test = stars[:-vsize]

for each in [train,valid,test]:
    print(f"Percentage of target values : {stars.target_class.mean():.4f}")


# Defining our model:

# In[ ]:


import lightgbm as lgb
from sklearn import metrics
val_pred = []
ground = []
def training(feat_cols):
    plt.figure()
    global val_pred, ground
    evals_result = {}
    
    dtrain = lgb.Dataset(data=train[feat_cols], label=train["target_class"])
    dvalid = lgb.Dataset(data=valid[feat_cols], label=valid["target_class"])
    dtest = lgb.Dataset(data=test[feat_cols], label=test["target_class"])

    param = {"num_leaves" : 64, "objectives":"binary"}
    param["metric"] = "auc"

    num_round = 500
    bst = lgb.train(param,dtrain,num_round,valid_sets=[dvalid],evals_result = evals_result, early_stopping_rounds = 10)
    
    #lgb.plot_metric(evals_result, metric="auc", figsize=(7,7))
    lgb.plot_importance(bst, max_num_features=10,figsize=(10,10))
    
    ypred = bst.predict(test[feat_cols])
    score = metrics.roc_auc_score(test["target_class"], ypred)

    val_pred = ypred
    ground = test["target_class"]
    
    print(f"our score is: {score:.4f}")
    return score, dvalid


# Lets see how performant will be our model with less features. We will notice that the difference is not big. Actually, as we are not dealing with a huge big dataset and lightGBM is a fast model, we should not concern about reducing the data as we did (even if just only 2 columns reduction...)

# Choosing the feature columns:

# In[ ]:



features = []
for key,val in m.items():
    feat = key.split("-")
    for each2 in feat:
        if each2 not in features:
            features.append(each2)
#features
res = {"baseline":"","selected features":""}

res["selected features"],_ = (training(features))

## With all columns:
feat_cols = stars.columns.drop("target_class")
res["baseline"],_ = (training(feat_cols))

res


# Now lets take a look at what the model has as output from its predictions:

# In[ ]:


diferr = pd.DataFrame(columns=["Prediction", "Ground_Truth"])
diferr["Ground_Truth"] = ground
diferr["Prediction"] = val_pred
diferr
print("Predictions for label = 0. Not pulsar stars\n",diferr.loc[diferr["Ground_Truth"]==0])
print("Predictions for label = 1. Pulsar stars\n",diferr.loc[diferr["Ground_Truth"]==1])


# These numbers may not tell too much if we don't know exactly what a binary regression classifier is doing. 
# 
# Accordling to https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc:
# 
# > AUC provides an aggregate measure of performance across all possible classification thresholds. One way of interpreting AUC is as the probability that the model ranks a random positive example more highly than a random negative example. For example, given the following examples, which are arranged from left to right in ascending order of logistic regression predictions:
# > 
# > Positive and negative examples ranked in ascending order of logistic regression score
# ![graph](http://developers.google.com/machine-learning/crash-course/images/AUCPredictionsRanked.svg)> 
# > Figure 6. Predictions ranked in ascending order of logistic regression score.
# > 
# > AUC represents the probability that a random positive (green) example is positioned to the right of a random negative (red) example.
# 
# Now it make sense that our model is predicting a probability that is bigger for targeting 1 than for targeting 0. This is because, for classifying a star as a Pulsar Star (label = 1), the model should predict a probability that is closer to 1 than the probability of a not Pulsar Star (label = 0). The last should predict probabilities that are closer to 0.
# 
# Below we can visualize the ROC curve, that is the area below the graph formed by the True Positive Rate (the model is predicting well) and the False Positive Rate (the model is saying that a star is a Pulsar Star when actually is not)
# 

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, auc, accuracy_score)
from sklearn.model_selection import GridSearchCV

plt.figure(figsize=(10,10))
valAcc = accuracy_score(ground, np.round(val_pred))
fprVal, tprVal, thresholdsVal = roc_curve(ground, val_pred)
valAUC =  auc(fprVal, tprVal)
print("Our threscholds : {}. Type : {}. Lenght : {}".format(thresholdsVal, type(thresholdsVal), len(thresholdsVal)))
print("valAUC : {} and valAcc : {}".format(valAUC, valAcc))
#Plot ROC curve from tpr and fpr.
plt.plot(fprVal, tprVal, label="Validation")
plt.legend()
plt.ylabel('True positive rate.')
plt.xlabel('False positive rate')
plt.title("ROC curve for validation")
plt.show()


# We can see that our model has a much higher prediction of TPR than of FPT! This is way our score is close to the value 1 (0.9897)

# Score for all columns features : **0.9897150578428906 with 17 iterations**
# Score for first 4 columns as features : **0.9862164089395752 with 19 iterations**
# Score for last 4 columns as features : 0.9355167561834904 with 16 iterations
# Score for 1st and 7th columns as features: **0.9742193387833641 with 17 iterations**
# Score for 1st and 3th columns as features (*worst pearson'r correlation): 0.971777916403* with 26 iterations
# Score for first 2 columns as features *(best pearson'r correlation) :0.9605646569128446* with 12 iterations
results =["base_TEST_all_columns0.9897150578428906",
          "base_TEST_first_4_columns0.9862164089395752",
          "base_TEST_last_4_columns0.9355167561834904",
          "base_TEST_1st_and_7th_columns0.9742193387833641",
          "base_TEST_1st_and_3th_columns0.971777916403",
          "base_TEST_first_2_columns0.9605646569128446"]
# In[ ]:


import csv
csvfile = "/kaggle/working/results.csv"

with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in results:
        writer.writerow([val])  

