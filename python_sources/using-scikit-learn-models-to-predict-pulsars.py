#!/usr/bin/env python
# coding: utf-8

# # Kaggle description
# 
# ### PREDICTING A PULSAR STAR
# Dr Robert Lyon
# 
# HTRU2 is a data set which describes a sample of pulsar candidates collected during the High Time Resolution Universe Survey .
# 
# Pulsars are a rare type of Neutron star that produce radio emission detectable here on Earth. They are of considerable scientific interest as probes of space-time, the inter-stellar medium, and states of matter .
# 
# As pulsars rotate, their emission beam sweeps across the sky, and when this crosses our line of sight, produces a detectable pattern of broadband radio emission. As pulsars rotate rapidly, this pattern repeats periodically. Thus pulsar search involves looking for periodic radio signals with large radio telescopes.
# 
# <img src="./NRAOPulsar.jpg" width=300/>
# 
# Each pulsar produces a slightly different emission pattern, which varies slightly with each rotation . Thus a potential signal detection known as a 'candidate', is averaged over many rotations of the pulsar, as determined by the length of an observation. In the absence of additional info, each candidate could potentially describe a real pulsar. However in practice almost all detections are caused by radio frequency interference (RFI) and noise, making legitimate signals hard to find.
# 
# Machine learning tools are now being used to automatically label pulsar candidates to facilitate rapid analysis. Classification systems in particular are being widely adopted, which treat the candidate data sets as binary classification problems. Here the legitimate pulsar examples are a minority positive class, and spurious examples the majority negative class.
# 
# The data set shared here contains 16,259 spurious examples caused by RFI/noise, and 1,639 real pulsar examples. These examples have all been checked by human annotators.
# 
# Each row lists the variables first, and the class label is the final entry. The class labels used are 0 (negative) and 1 (positive).
# 
# Attribute Information:
# Each candidate is described by 8 continuous variables, and a single class variable. The first four are simple statistics obtained from the integrated pulse profile (folded profile). This is an array of continuous variables that describe a longitude-resolved version of the signal that has been averaged in both time and frequency . The remaining four variables are similarly obtained from the DM-SNR curve . These are summarised below:
# 
# - Mean of the integrated profile.
# - Standard deviation of the integrated profile.
# - Excess kurtosis of the integrated profile.
# - Skewness of the integrated profile.
# - Mean of the DM-SNR curve.
# - Standard deviation of the DM-SNR curve.
# - Excess kurtosis of the DM-SNR curve.
# - Skewness of the DM-SNR curve.
# - Class
# 
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

# **The goal of this project is to build different classification models to predict whether a star is a pulsar or not, while applying different techniques of the Data Science process.**

# Importing usefull packages

# In[ ]:


import numpy as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# Data ingestion

# In[ ]:


data = pd.read_csv('../input/predicting-a-pulsar-star/pulsar_stars.csv')


# ## Data exploration

# Looking for null values

# In[ ]:


data.isnull().any()


# In[ ]:


data.describe()


# The data set is clean and ready to use

# #### Distribution of the data

# In[ ]:


data.iloc[:, :-1].boxplot(whis = "range", vert = False, figsize = (10,7))
plt.title("Range and distribution of the data in each column")


# ### Comparison of the distribution of positive and negative cases for each variable.

# In[ ]:


pos = data[data.iloc[:,8] == 1]
neg = data[data.iloc[:,8] == 0]


# In[ ]:


pos.describe()


# In[ ]:


neg.describe()


# In[ ]:


for i in range(8):
    fig = plt.figure(figsize = (6,8))
    fig = sns.violinplot(data = data, x = data.columns[-1], y = data.columns[i], scale = 'area', palette = {0: 'tab:orange', 1: 'tab:blue'})
    fig.set_title('Distribution of positive and negative cases')
    plt.show()
    


# #### We can see that the distributions for positive and negative cases are different but mostly overlaped

# ## Plotting the positive and negative cases by pair of attributes
# 

# The next cell makes the pairplots using seaborn. You can zoom in

# In[ ]:


data1 = data.copy()
data1.loc[data1['target_class'] == 1, 'target_class'] = "Positive"
data1.loc[data1['target_class'] == 0, 'target_class'] = "Negative"

sns.pairplot(data1, hue = 'target_class', height = 4, markers = '.', vars = data1.columns[:-1], palette = 'bright', hue_order = ['Negative', 'Positive'])


# Now let's do something similar with matplotlib that shows the paterns more clearly while showing normalized histograms in the diagonal. Again, you can zoom in.

# In[ ]:


fig, axes = plt.subplots(8, 8, figsize = (60,40))

labels = data.columns

for a in range(8):
    for b in range(8):
        
        if a == b:
            axes[a,b].hist(neg.iloc[:,a], bins = 40, color = 'r', alpha = 0.5, label = 'Negative', density = True)
            axes[a,b].hist(pos.iloc[:,a], bins = 40, color = 'b', alpha = 0.5, label = 'Positive', density = True)
            axes[a,b].set_xlabel(labels[a])
            axes[a,b].legend(markerscale = 20)
            
        else:
            axes[a,b].scatter(neg.iloc[:,a], neg.iloc[:,b], s = 0.1, c = 'red', alpha = 1, label = 'Negative')
            axes[a,b].scatter(pos.iloc[:,a], pos.iloc[:,b], s = 0.1, c = 'b', alpha = 1, label = 'Positive')
            axes[a,b].set_xlabel(labels[a])
            axes[a,b].set_ylabel(labels[b])
            axes[a,b].legend(markerscale = 20)


# For some pairs, positive cases extend in different regions than negative ones:

# In[ ]:


plt.figure(figsize = (8,6))
plt.scatter(neg.iloc[:,2], neg.iloc[:,0], s = 0.1, c = 'red', alpha = 1, label = 'Negative')
plt.scatter(pos.iloc[:,2], pos.iloc[:,0], s = 0.1, c = 'b', alpha = 1, label = 'Positive')
plt.xlabel(labels[2])
plt.ylabel(labels[0])
plt.legend(markerscale = 20)


# But for other pairs, positive and negative cases extend in the same regions:

# In[ ]:


plt.figure(figsize = (8,6))
plt.scatter(neg.iloc[:,5], neg.iloc[:,4], s = 0.1, c = 'red', alpha = 1, label = 'Negative')
plt.scatter(pos.iloc[:,5], pos.iloc[:,4], s = 0.1, c = 'b', alpha = 1, label = 'Positive')
plt.xlabel(labels[5])
plt.ylabel(labels[4])
plt.legend(markerscale = 20)


# ### We can see that for some attributes the positive cases are mostly separated from the negative ones while in other cases they are mostly overlapping

# # Models

# In[ ]:



from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# Splitting the data

# In[ ]:


xtrain, xtest, ytrain, ytest = train_test_split(data.iloc[:,0:8], data.iloc[:,8])


# We define a function to calculate and print the number of true positives, true negatives, false positives, false negatives and the recall, along with the accuracy.
# The recall is the proportion of true cases labeled correctly.
# Since the data set has many more negative than positive cases, we also want to have a good recall. In a data set with 90% negative cases (like this one), a model that labels all samples as "negative" will have 90% accuracy. But it is not useful at all.
# Therefore, we define a metric "score" that is the average between accuracy and recall. We will use this to choose the best model
# 

# In[ ]:


def metrics(test, pred):
    # tp: true positive, fp: false positive, tn: true negative, fnfalse negative
    tp, fp, tn, fn, score = 0, 0, 0, 0, 0
    for a, b in zip(test, pred):
        if b ==1:
            if a ==1: tp += 1
            else: fp +=1
        else:
            if a == 0: tn +=1
            else: fn +=1
    
    recall = tp / (tp + fn)
    accu = accuracy_score(test, pred)
    score = (recall + accu)/2
    
    return accu, tp,fp, tn, fn, recall, score


def print_metrics(test, pred):
    accu, tp,fp, tn, fn, recall, score = metrics(test,pred)
    print(f"Accuracy:       {accu: .3f} \nTrue positive:   {tp} \nFalse positive:  {fp} \nTrue negative:   {tn} \nFalse negative:  {fn} \nRecall:         {recall: .3f} \nScore:          {score: .3f}")


# The returned values of the "metrics" function will be saved in the "results" dataframe for each model.

# In[ ]:


results = pd.DataFrame(index = ['Accuracy', 'True positives', 'False positives', 'True negatives', 'False negatives', 'Recall', 'Score'])


# In[ ]:


LR = LogisticRegression()

LR.fit(xtrain, ytrain)
LR_prediction = LR.predict(xtest)
results['Logistic regression'] = metrics(ytest, LR_prediction)
print_metrics(ytest, LR_prediction)


# In[ ]:


results


# In[ ]:


GNB = GaussianNB()

GNB.fit(xtrain, ytrain)
GNB_prediction = GNB.predict(xtest)
results['Gaussian Naive Bayes'] = metrics(ytest, GNB_prediction)


# In[ ]:


KNC = KNeighborsClassifier(n_neighbors = 5)

KNC.fit(xtrain, ytrain)
KNC_prediction = KNC.predict(xtest)
results['K Neaghbors'] = metrics(ytest, KNC_prediction)


# In[ ]:


DTC = DecisionTreeClassifier(criterion = "entropy")

DTC.fit(xtrain, ytrain)
DTC_prediction = DTC.predict(xtest)
results['Decision tree'] = metrics(ytest, DTC_prediction)


# In[ ]:


RFC = RandomForestClassifier(n_estimators = 100)

RFC.fit(xtrain, ytrain)
RFC_prediction = RFC.predict(xtest)
results['Random forest'] = metrics(ytest, RFC_prediction)


# In[ ]:


MLPC = MLPClassifier(hidden_layer_sizes = (4), activation = "tanh", max_iter = 400)

MLPC.fit(xtrain, ytrain)
MLPC_prediction = MLPC.predict(xtest)
results['Neural Network'] = metrics(ytest, MLPC_prediction)


# # Results

# In[ ]:


results


# In[ ]:


#plotting accuracy and recall in %


fig, ax = plt.subplots(figsize = (12,5))
w = 0.3
x = np.arange(len(results.columns))
labels = results.columns

rects1 = ax.bar(x - w/2, results.loc['Accuracy'] * 100, width = w, label = 'Accuracy')
rects2 = ax.bar(x + w/2, results.loc['Recall'] * 100, width = w, label = 'Recall')
ax.grid(axis = 'y')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation = 45)
ax.set_ylabel('Ratio (%)')
ax.legend(loc = 4)
ax.set_ylim(0,105)
ax.set_title('Model performance')

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:4.1f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)




plt.show()


# ### Some of these models depend on the initial random state. We will build 200 of them and we will keep the best of each.

# In[ ]:


names = ['DTC', 'RFC', 'MLPC']
d = {}                                   #the fitted models will be saved in this dictionary
results2 = pd.DataFrame(index = ['Accuracy', 'True positives', 'False positives', 'True negatives', 'False negatives', 'Recall', 'Score'])
num = 200

for i in range(num):
    models = [DecisionTreeClassifier(), RandomForestClassifier(n_estimators = 100), MLPClassifier(hidden_layer_sizes = (4), activation = "tanh", max_iter = 400)] #definido dentro del bucle for para que en cada iteraci'on use un estado inicial pseudoaleatorio ditinto
    for name, model in zip(names, models):
        name += str(i)
        model.fit(xtrain, ytrain)
        prediction= model.predict(xtest)
        results2[name] = metrics(ytest, prediction)
        d[name] = model


# In[ ]:



for a in range(3):
    c = []
    for i in range(num):
        c.append(a + i*3)

    
    best = results2.iloc[-1,c].idxmax()
    results[best] = results2[best]


# In[ ]:


results


# In[ ]:


#plotting accuracy and recall in %


fig, ax = plt.subplots(figsize = (14,5))
w = 0.3
x = np.arange(len(results.columns))
labels = results.columns

rects1 = ax.bar(x - w/2, results.loc['Accuracy'] * 100, width = w, label = 'Accuracy')
rects2 = ax.bar(x + w/2, results.loc['Recall'] * 100, width = w, label = 'Recall')
ax.grid(axis = 'y')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation = 45)
ax.set_ylabel('Ratio (%)')
ax.legend(loc = 4)
ax.set_ylim(0,105)
ax.set_title('Model performance')

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:4.1f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)




plt.show()


# Let's see the distribution of scores of all the models through some histograms

# In[ ]:


allDTC = pd.DataFrame(index = ['Accuracy', 'True positives', 'False positives', 'True negatives', 'False negatives', 'Recall', 'Score'])
allRFC = pd.DataFrame(index = ['Accuracy', 'True positives', 'False positives', 'True negatives', 'False negatives', 'Recall', 'Score'])
allMLPC = pd.DataFrame(index = ['Accuracy', 'True positives', 'False positives', 'True negatives', 'False negatives', 'Recall', 'Score'])

for name in results2.columns:
    if name.startswith('DTC'):
        allDTC[name] = results2[name]
        
    elif name.startswith('RFC'):
        allRFC[name] = results2[name]
        
    else:
        allMLPC[name] = results2[name]

histfig, histaxes = plt.subplots(1,3, figsize = (16,4))
histaxes[0].hist(allDTC.loc['Score'], bins = 30)
histaxes[0].set_title('Histogram of Decision Tree \nClassifiers scores')
histaxes[1].hist(allRFC.loc['Score'], bins = 30)
histaxes[1].set_title('Histogram of Random Forest \nClassifiers scores')
histaxes[2].hist(allMLPC.loc['Score'], bins = 30)
histaxes[2].set_title('Histogram of Multi Layer Perceptron \nClassifiers scores')

plt.show()


# (Disclaimer: in some runs of the script this case doesn't accur)
# We can see that some neural networks models have a very low score. If we see the metrics for this model we can see that all the samples has been classified as non pulsars. This is the best to what it has been able to converge given the initial random state, an stil has an accuracy of 90%. This shows why it is important to choose a good metric and to initialize many models when they depend on the initial random state. 

# In[ ]:


worst_idx = allMLPC.loc['Score'].idxmin()

worst = d[worst_idx]
worst_prediction = worst.predict(xtest)
print_metrics(ytest, worst_prediction)

