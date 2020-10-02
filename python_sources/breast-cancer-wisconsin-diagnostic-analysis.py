#!/usr/bin/env python
# coding: utf-8

#     The objective of this exercise is to use machine learning techniques to perform supervised learning on Breast Cancer Wisconsin data set provided here. 

# In[ ]:


#Lets import required libraries before we proceed further
import numpy as np 
import pandas as pd
import matplotlib
import seaborn as sns
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm


# In[ ]:


#Load data into dataframe
dfBCancer = pd.read_csv("../input/data.csv")
dfBCancer.info()


# In[ ]:


dfBCancer.describe()


# In[ ]:


dfBCancer.head(5)


# Few observations after initial scan of the dataset:
# * Dataset contains: 33 columns: 32 of which are numericals where as one is object.
# * Column diagnosis represents the label vector.
# * Column Unnamed: 32 has all the records as Nan ( count for this column is showing zero when called describe function).
# * Column ID has unique value for all the rows and a simple scan of each value of this column suggests that its an unique identifier for each row.
# * Rest 30 column can be divided into blocks of 10 features each representing Mean, Worst and Standard Error values.
# * Its also evident that scale of features representing Mean and Worst values is significantly different from respective "se" features.

# In[ ]:


#Drop Unnamed: 32 as it only contains Nan
# axis=1: represents column
#inplace = True : represents whether we want to delete column from this dataframe instance, in inplace=False is specified it will return a new
#dataframe having column "Unnamed: 32" deleted but will not change the original datafram
dfBCancer.drop(["Unnamed: 32"], axis=1, inplace=True)


# In[ ]:


fig, axs = plt.subplots(6, 5, figsize=(16,20))
df=dfBCancer.drop(["id"], axis=1)
g = sns.FacetGrid(df)
k=1
for i in range(6):
    for j in range(5):
        axs[i][j].set_title (df.columns[k])
        g.map(sns.boxplot, "diagnosis",  df.columns[k],  ax=axs[i][j])
        k=k+1
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9,
                    hspace=0.4, wspace=0.3)
plt.show()


# In[ ]:


# As we mentioned earlier we can see that above data frame contains block of features representing mean, worst and se
# In order to analyse these features separately we shall be dividing dfBCancer dataframe into 3 dataframes. 
# This will help us in performing further analysis on this dataset

dfCancer_mean = dfBCancer.drop(["id", "diagnosis"], axis=1).iloc[:, 0:10]
dfCancer_se = dfBCancer.drop(["id", "diagnosis"], axis=1).iloc[:, 10:20]
dfCancer_worst = dfBCancer.drop(["id", "diagnosis"], axis=1).iloc[:, 20:30]

print(dfCancer_mean.columns)
print("----------------")
print(dfCancer_se.columns)
print("----------------")
print(dfCancer_worst.columns)


# I am interested in analysing if there is any correlation between columns in respective feature buckets. Dividing data into 3 sets makes it easier to perform analysis.
# Note that correlation does not necessary mean, that there is a causal relationship between these features. We shall delve deep into this later.

# In[ ]:


#Lets draw histogram
fig, axs = plt.subplots(2, 5, figsize=(12, 8))

g = sns.FacetGrid(dfCancer_mean)
k=0
for i in range(2):
    for j in range(5):
        g.map(sns.distplot, dfCancer_mean.columns[k], ax=axs[i][j])
        k=k+1
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9,
                    hspace=0.4, wspace=0.3)
plt.show()


# In[ ]:


# Compute the correlation matrix
corr = dfCancer_mean.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

 # Set up the matplotlib figure
f, ax = plt.subplots(figsize=(14, 14))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, annot= True, cbar_kws={"shrink": .5})
plt.show()


# Wow...we see that radius_mean, perimeter_mean and area_mean feature are highly positvely correlated (.99 & 1). Is there any causal relationship ? Well, if we go by the mathematical definition of radius, mean & area we know that mean and area are dependent on radius and therefore its expected that these features will be highly correlated.
# 
# Since these features are highly correlated and there is a causal relationship, we can safely drop 2 of these columns , hoping that these features will not add a lot of information in prediction.
# 
# Are there any other features which are correlated as well ?
# 
# We see that concavity_mean is highly correlated with concave points_mean. 0.92.
# while concave points_mean seem to be highly positively correlated to radius, area and perimeter features; concavity mean does not.
# Concavity_mean and concave points_mean are also highly positvely correlated to compactness_mean
# 
# Basis above analysis, shall be dropping following columns from the bCancer_mean dataframe:
# * perimeter_mean
# * area_mean
# * concave points_mean
# * compactness_mean
# 
# Lets check if similar relationship exists in other dataframes as well:

# In[ ]:


#Lets draw histogram
fig, axs = plt.subplots(2, 5, figsize=(12, 8))

g = sns.FacetGrid(dfCancer_se)
k=0
for i in range(2):
    for j in range(5):
        g.map(sns.distplot, dfCancer_se.columns[k], ax=axs[i][j])
        k=k+1
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9,
                    hspace=0.4, wspace=0.3)
plt.show()


# In[ ]:


# Compute the correlation matrix
corr = dfCancer_se.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

 # Set up the matplotlib figure
f, ax = plt.subplots(figsize=(14, 14))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(150, 275, s=80, l=55, n=9, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, annot= True, cbar_kws={"shrink": .5})
plt.show()


# In the dCancer_se dataframe we do see radius, perimeter and area highly positvely correlated >= 0.95 however in this dataset concavity_se and compactness_se are having correlation =0.8; though compactness_se is correlated to fractal_dimension_se with a correlation of 0.8 , fractal_dimension_se correlated to concavity_se with correlation of 0.73. Concave points_se's correlation to concavity_se and compctness_se is of the order of 0.74/0.77.
# 
# Basis above analysis, shall be dropping following columns from the bCancer_mean dataframe:
# * perimeter_se
# * area_se
# * concave points_se
# * compactness_se
# 

# In[ ]:


#Lets draw histogram
fig, axs = plt.subplots(2, 5, figsize=(12, 8))

g = sns.FacetGrid(dfCancer_worst)
k=0
for i in range(2):
    for j in range(5):
        g.map(sns.distplot, dfCancer_worst.columns[k], ax=axs[i][j])
        k=k+1
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9,
                    hspace=0.4, wspace=0.3)
plt.show()


# In[ ]:


# Compute the correlation matrix
corr = dfCancer_worst.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

 # Set up the matplotlib figure
f, ax = plt.subplots(figsize=(14, 14))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 50, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, annot= True, cbar_kws={"shrink": .5})
plt.show()


# Similar relationship exists in the dCancer_worst datafame therefore we shall be dropping following columns from the bCancer_mean dataframe:
# * perimeter_worst
# * area_worst
# * concave points_worst
# * compactness_worst

# In[ ]:


# I shall be combining all these dataframes back into one result dataframe after droppoing these columns, this time I shall not be using 
#inplace =True, as i dont want original dataframe's values to be lost lets concatenate it back
result = pd.concat([dfCancer_worst.drop(["area_worst", "perimeter_worst", "concave points_worst" , "compactness_worst"], axis=1),
                   dfCancer_se.drop(["area_se", "perimeter_se", "concave points_se" , "compactness_se"], axis=1),
                   dfCancer_mean.drop(["area_mean", "perimeter_mean", "concave points_mean" , "concavity_mean"], axis=1)],
                   axis=1)
# check if resulting dataframe as all the dataset combined except the ones which we didnt want
result.columns


# Ok good, we have 18 columns / features in the above dataframe. Lets start learning process now

# In[ ]:


# First convert categorical label into qunatitative values for prediction
factor = pd.factorize( dfBCancer.diagnosis)
diagnosis = factor[0]
definitions = factor[1]


# In[ ]:


# Split dataset into test and train data
trainX, testX, trainY, testY = train_test_split(result, diagnosis, test_size=0.35, random_state=42)


# I shall be first calling Logistic Regressor for prediction, however before that do you remember that our dataset had different scales for Mean, worst and se columns; therefore we should standardise data as well. 
# 
# Should you normalize or Standardize: http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-16.html
# Scikit Learn : http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler

# In[ ]:


from sklearn.preprocessing import StandardScaler
stdScalar= StandardScaler().fit(trainX)
trainX = stdScalar.transform(trainX)
testX= stdScalar.transform(testX)


# In[ ]:


print("Mean of trainX: ", trainX.mean(axis=0), " and standard deviation of trainX: ", trainX.std(axis=0))


# In[ ]:


print("Mean of testX: ", testX.mean(axis=0), " and standard deviation of testX: ", testX.std(axis=0))


# In[ ]:


regression = LogisticRegression()
regression.fit(trainX, trainY)

predtrainY = regression.predict(trainX)

print('Accuracy {:.2f}%'.format(accuracy_score(trainY, predtrainY) * 100))
print(classification_report(trainY, predtrainY))


# Great We have 98% Accuracy on training dataset. Lets check confusion matrix

# In[ ]:


#Create a Confusion matrix

#Reverse factorize
reversefactor = dict(zip(range(len(definitions)),definitions))
y_test = np.vectorize(reversefactor.get)(trainY)
y_pred = np.vectorize(reversefactor.get)(predtrainY)
cm = confusion_matrix(y_test, y_pred)

# plot
fig, ax = plt.subplots()
ax.matshow(cm, cmap=plt.cm.Greens, alpha=0.3)
ax.grid(False)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, s=cm[i,j], va='center', ha='center', fontsize=9)

plt.xlabel('True Predictions')
plt.ylabel('False Predictions')
plt.xticks(range(len(definitions)), definitions.values, rotation=90, fontsize=8)
plt.yticks(range(len(definitions)), definitions.values, fontsize=8)

plt.show()


# In[ ]:


# use this on the test dataset
predtestY = regression.predict(testX)
print('Accuracy {:.2f}%'.format(accuracy_score(testY, predtestY) * 100))
print(classification_report(testY, predtestY))


# Great we have 98% accuracy Test and Train both dataset. Lets print confusion matrix for test dataset as well

# In[ ]:


#Create a Confusion matrix

#Reverse factorize
reversefactor = dict(zip(range(len(definitions)),definitions))
y_test = np.vectorize(reversefactor.get)(testY)
y_pred = np.vectorize(reversefactor.get)(predtestY)
cm = confusion_matrix(y_test, y_pred)

# plot
fig, ax = plt.subplots()
ax.matshow(cm, cmap=plt.cm.Greens, alpha=0.3)
ax.grid(False)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, s=cm[i,j], va='center', ha='center', fontsize=9)

plt.xlabel('True Predictions')
plt.ylabel('False Predictions')
plt.xticks(range(len(definitions)), definitions.values, rotation=90, fontsize=8)
plt.yticks(range(len(definitions)), definitions.values, fontsize=8)

plt.show()


# Prediction using Random Forest, Since Random Forest is a Decision Tree based estimator; we don't need to standardize features:

# In[ ]:


randclassifier = RandomForestClassifier(max_depth=13,max_features ='sqrt', n_estimators=50,class_weight="balanced", random_state=42)

randclassifier.fit(trainX,trainY)
predtrainY = randclassifier.predict(trainX)

print('Accuracy {:.2f}%'.format(accuracy_score(trainY, predtrainY) * 100))
print(classification_report(trainY, predtrainY))


# Great, We have 100% accuracy on train data , lets check accuracy on the test dataset

# In[ ]:


#use this on the test dataset

predtestY = randclassifier.predict(testX)

print('Accuracy {:.2f}%'.format(accuracy_score(testY, predtestY) * 100))
print(classification_report(testY, predtestY))


# In[ ]:


#Lets draw validation curve now for Random Forest Model

plt.figure(figsize=(10,8))

param_range = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

train_scores, test_scores = validation_curve(
    randclassifier, result, diagnosis, param_name="n_estimators", param_range=param_range,
    cv=10, scoring="accuracy", n_jobs=1)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

lw = 2
# Plot mean accuracy scores for training and test sets
plt.plot(param_range, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(param_range, test_scores_mean, 'o-', color="g", label="Cross-validation score")

# Plot accurancy bands for training and test sets
plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,alpha=0.1,
                     color="r")
plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")

# Create plot
plt.xlabel("Number Of Trees")
plt.ylabel("Accuracy Score")

plt.tight_layout()
plt.legend(loc="best")
plt.ylim(0.8, 1.0)
plt.grid(True)
plt.show()   


# In[ ]:


svmClassifier = svm.SVC()
svmClassifier.fit(trainX,trainY)
predtrainY=svmClassifier.predict(trainX)

print('Accuracy {:.2f}%'.format(accuracy_score(trainY, predtrainY) * 100))
print(classification_report(trainY, predtrainY))


# In[ ]:


# use this on the test dataset
predtestY = svmClassifier.predict(testX)

print('Accuracy {:.2f}%'.format(accuracy_score(testY, predtestY) * 100))
print(classification_report(testY, predtestY))


# In[ ]:




