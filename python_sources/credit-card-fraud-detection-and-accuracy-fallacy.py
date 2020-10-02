#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing the basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.offline as py
from plotly import tools
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#Reading the dataset 
df = pd.read_csv('../input/creditcard.csv')
df.shape


# In[ ]:


#Let's check if there are any null values 
df.isnull().sum()


# In[ ]:


#Describe the dataset to get rough idea about the data 
df.describe()


# In[ ]:


#Well let's see all the columns in the dataset 
df.columns


# In[ ]:


#A brief look at the initial rows of the dataset 
df.head()


# In[ ]:


#It's good to shuffle the datset 
df = df.sample(frac=1, random_state=42).reset_index(drop=True)


# In[ ]:


#Let's take a look again 
df.head()


# In[ ]:


#Non fradulant cases
df.Class.value_counts()[0]


# In[ ]:


#Fradulant cases 
df.Class.value_counts()[1]


# In[ ]:


print('Percentage of correct transactions: {}'.format((df.Class.value_counts()[0]/df.shape[0])*100))


# In[ ]:


print('Percentage of fradulent transactions: {}'.format((df.Class.value_counts()[1]/df.shape[0])*100))


# In[ ]:


#Let's visualize the distribution of the classes (0 means safe and 1 means fraudulent)
import seaborn as sns
colors = ['green', 'red']

sns.countplot('Class', data=df, palette=colors)
plt.title('Normal v/s Fraudulent')


# In[ ]:


#Now let's map how much a feature affects our class 
cor = df.corr()
fig = plt.figure(figsize = (12, 9))

#Plotting the heatmap
sns.heatmap(cor, vmax = 0.7)
plt.show()


# In[ ]:


cor.shape


# In[ ]:


#This is how much a each feature affects the our class 
cor.iloc[-1,:]


# In[ ]:


#We need to delet the least and greatest values 
#From above analysis I've selected the following features 
#Note that I've included the class variable also because I intend to create a new dataframe using the new features
new_features=['V1','V3','V4','V7','V10','V11','V12','V14','V16','V17','V18','Class']


# In[ ]:


#Let's plot a heatmap again and see the relationship
cor = df[new_features].corr()
fig = plt.figure(figsize = (12, 9))

#Plotting the heatmap
sns.heatmap(cor, vmax = 0.7)
plt.show()


# In[ ]:


#We see that the class rows and columns are darker and brighter 
#This means that all the variables in our new dataset have a significant affect 


# In[ ]:


#Now splitting the dataset into the dependent variable(y) and independent variales(x)
x=df[new_features].iloc[:,:-1].values
y=df[new_features].iloc[:,-1].values


# In[ ]:


#Withoud reducing the features 
#x=df.iloc[:,:-1].values
#y=df.iloc[:,-1].values
#Feel free to try using all the features :)


# In[ ]:


x.shape


# In[ ]:


y.shape


# In[ ]:


x[:5]


# In[ ]:


y[:5]


# In[ ]:


#Spliting the data into train and test sets 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# In[ ]:


#Let's see how many safe and fraudulent cases are there in training set 
safe_train=(y_train==0).sum()
fraud_train=(y_train==1).sum()
print("Safe: {} \nFraud: {}".format(safe_train,fraud_train))


# In[ ]:


#Let's see how many safe and fraudulent cases are there in test set 
safe_test=(y_test==0).sum()
fraud_test=(y_test==1).sum()
print("Safe: {} \nFraud: {}".format(safe_test,fraud_test))


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


# In[ ]:


#Using Logistic Regression 
clf = LogisticRegression(random_state = 0)
clf.fit(x_train, y_train)


# In[ ]:


#Let's evaluate our model 
y_pred = clf.predict(x_test)
print("Training Accuracy: ",clf.score(x_train, y_train))
print("Testing Accuracy: ", clf.score(x_test, y_test))
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test,y_pred)) 


# # Looking at the Accuracy fallacy 
# 
# ## What it is ? 
# If we look at the accuracy on test set, it's nearly 0.99914 i.e. over 99.9% accuracy.
# Does that mean our model is phenomenal? No!
# Before that let me tell you what a confusion matrix for binary classification represents 
# We made the matrix between the actual values and predicted values, so the rows represent the actual values and columns respresent the predicted values. Now that you know what rows and colmns are let's understand what each cell represnt hence:
# 1. [0][0] -> True positives i.e. how many safe cases are there that our model predicted correctly. Here 71069 are the number of CORRECTLY PREDICTED safe cases. 
# 2. [0][1] -> False Negatives i.e. how many safe cases are there that our model predicted incorrectly. Here 4 are the number of the MISCLASSIFIED safe cases. Hence 4 safe cases were misclassified as fraud. This is potentially less dangerous as it's better to stop some safe transactions with slightest chance of fraud.
# 3. [1][0] -> False Positives i.e. how many fraud cases are there that our model predicted incorrectly. Here 57 are the number of the MISCLASSIFIED fraud cases. Here 57 fraud cases were misclassified as safe. This is very dangerous because we are letting the fraud cases pass through. This can cause huge loss to the organization. 
# 4. [1][1] -> True negatives i.e. how many fraud cases are there that our model predicted correctly. Here 72 are the number of CORRECTLY PREDICTED fraud cases.
# 
# ### We can see that despite having an accuracy over 99.9% our model predicted 57 fraud cases incorrectly. This is what I call accuracy fallacy.
# This usually happens when data is UNEVENLY DISTRIBUTED. From above code we realize that the number of fraud cases in trianing set is just 363 whereas the number of safe ones are 213242. This is very unevenly distributed and will give great accuracy but misclassify the dangerous classes.
# 
# To give you a perspective, let me misclassify all the fraudulent cases.
# Let's predict every case as safe, so our confusion matrix for the test set is as follows:
# 
# [[71073         0]
# 
#  [129           0]]
#  
# So new accuracy = 71073/(71073 +  129) = 0.9981
# which is nearly 99.8% accuracy.
# Again great accuracy, terrible performance.
# 
# ## So how we measure model performance?
# 
# ### The answer is precision, recall, f1 score and AUC-ROC curve.
# Lets look at each one of them one by one
# ![image.png](attachment:image.png)
# 
# Image source : Wikipedia 
# ### Precision:
# Precision refers to the percentage of your results which are relevant and is calculated as follows : 
# True Positives/(True Positives + Flase Positives)
# 
# ### Recall:
# Recall refers to the percentage of total relevant results correctly classified by your algorithm and is calculated as follows : True Positives/(True Positives + False Negatives)
# 
# ### F1 Score:
# It is the harmonic mean of precision and recall and is calculated as follows: 
# (2 x Precision x Recall)/(Precision + Recall)
# 
# ## Hence the greater the F1 score the better it is
# In the above example we see that f1 score of safe class is 1 but that of fraud is 0.70 and hence the macro average is 0.85 
# 
# ### AUC-ROC curve (Area Under Curve - Receiver Operating Characteristic Curve):
# The True Positive Rate (TPR) is plot against False Positive Rate (FPR) for the probabilities of the classifier predictions. Then, the area under the plot is calculated. The greater the area under the curve the better our model is.
# 
# TPR is also knowns as recall and hence = True Positives/(True Positives + False Negatives)
# 
# FPR is negation of specificity = 1 - Specificity = 1- True Negatives/(True Negatives + False Positives) 
# 
# Hence FPR = False Positives/(True Negatives + False Positives)
# 
# 

# In[ ]:


from sklearn.metrics import roc_curve, auc


# In[ ]:


#Calculating the FPR and TPR
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)


# In[ ]:


#Plotting the curves 
label = 'Logistic Regressoin Classifier AUC:' + ' {0:.2f}'.format(roc_auc)
label2 = 'Random Model' 
plt.figure(figsize = (20, 12))
plt.plot([0,1], [0,1], 'r--', label=label2)
plt.plot(fpr, tpr, c = 'g', label = label, linewidth = 4)
plt.xlabel('False Positive Rate', fontsize = 16)
plt.ylabel('True Positive Rate', fontsize = 16)
plt.title('Receiver Operating Characteristic', fontsize = 16)
plt.legend(loc = 'lower right', fontsize = 16)


# In[ ]:


#We see that AUC is 0.78 which is not bad, not great but fair enough


# ## Let's try a supervised anomaly detection algorithm KNN
# 
# We use anomaly detection algorithm to find unusual patterns in the data. Since the data is unbalanced greatly and there must be some unusual patterns in data, let's try K Nearest Neighbours Algorithm 

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
clf.fit(x_train, y_train)


# In[ ]:


y_pred = clf.predict(x_test)
print("Training Accuracy: ",clf.score(x_train, y_train))
print("Testing Accuracy: ", clf.score(x_test, y_test))
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test,y_pred)) 


# In[ ]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)


# In[ ]:


label = 'KNN Classifier AUC:' + ' {0:.2f}'.format(roc_auc)
label2 = 'Random Model' 
plt.figure(figsize = (20, 12))
plt.plot([0,1], [0,1], 'r--', label=label2)
plt.plot(fpr, tpr, c = 'g', label = label, linewidth = 4)
plt.xlabel('False Positive Rate', fontsize = 16)
plt.ylabel('True Positive Rate', fontsize = 16)
plt.title('Receiver Operating Characteristic', fontsize = 16)
plt.legend(loc = 'lower right', fontsize = 16)


# In[ ]:


#We see that AUC is 0.90 which is good, with different random state of data you may get different AUC.
#I once achieved 0.94
#Though I've kept a constant seed for random state wile shuffling the dataset, feel free to mess with that :D


# ## Now let's try an unsupervised algorithm namely Isolation forest 
# 
# Since it is unsupervised, we don't need the target feature that is our class variable. Also we can use entire dataset instead of training on one and hence testing on the other
# 
# Isolation forest tries to separate each point in the data. Here an anomalous point could be separated in a few steps while normal points which are closer could take significantly more steps to be segregated.

# In[ ]:


(df.Class.value_counts()[1]/df.Class.value_counts()[0])


# In[ ]:


#Importing and fitting the Isolation Forest Algorithm 
from sklearn.ensemble import IsolationForest
clf=IsolationForest(contamination=(df.Class.value_counts()[1]/df.shape[0]), random_state=123,max_features=x.shape[1])
clf.fit(x)


# In[ ]:


#Predicting the class
y_pred = clf.predict(x)


# In[ ]:


#Since the algorithm classifies one class as 1 and other as -1
#Let's see how many classes it predicted as fraudulent 
(y_pred==-1).sum()


# In[ ]:


#Since our class variables are either 0 or 1, so we need to replace the predicted classes as 0 and 1
y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1


# In[ ]:


#Let's see how good our model performed 
print("Training Accuracy: ",accuracy_score(y, y_pred))
cm = confusion_matrix(y, y_pred)
print(cm)
print(classification_report(y,y_pred))


# In[ ]:


fpr, tpr, thresholds = roc_curve(y, y_pred)
roc_auc = auc(fpr, tpr)


# In[ ]:


label = 'Isolation Forest Classifier AUC:' + ' {0:.2f}'.format(roc_auc)
label2 = 'Random Model' 
plt.figure(figsize = (20, 12))
plt.plot([0,1], [0,1], 'r--', label=label2)
plt.plot(fpr, tpr, c = 'g', label = label, linewidth = 4)
plt.xlabel('False Positive Rate', fontsize = 16)
plt.ylabel('True Positive Rate', fontsize = 16)
plt.title('Receiver Operating Characteristic', fontsize = 16)
plt.legend(loc = 'lower right', fontsize = 16)


# In[ ]:


#We get AUC score of 0.75, that's bad. Seems like this algorithm is not working very well on our dataset 


# In[ ]:


#Let's try another unsupervised algorithm 
from sklearn.neighbors import LocalOutlierFactor
clf=LocalOutlierFactor(n_neighbors=5,contamination=(df.Class.value_counts()[1]/df.shape[0]))


# In[ ]:


y_pred = clf.fit_predict(x)
y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1


# In[ ]:


print("Training Accuracy: ",accuracy_score(y, y_pred))
cm = confusion_matrix(y, y_pred)
print(cm)
print(classification_report(y,y_pred))


# In[ ]:


fpr, tpr, thresholds = roc_curve(y, y_pred)
roc_auc = auc(fpr, tpr)


# In[ ]:


label = 'KNN Classifier AUC:' + ' {0:.2f}'.format(roc_auc)
label2 = 'Random Model' 
plt.figure(figsize = (20, 12))
plt.plot([0,1], [0,1], 'r--', label=label2)
plt.plot(fpr, tpr, c = 'g', label = label, linewidth = 4)
plt.xlabel('False Positive Rate', fontsize = 16)
plt.ylabel('True Positive Rate', fontsize = 16)
plt.title('Receiver Operating Characteristic', fontsize = 16)
plt.legend(loc = 'lower right', fontsize = 16)


# In[ ]:


#WHAT!!!!? That's a garbage model with an AUC of 0.5.
#Moreover it made 0 true negatives and that's a teriible, terrible model


# For now it seems that *KNN* is performing quite well for the given data
