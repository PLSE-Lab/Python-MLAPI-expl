#!/usr/bin/env python
# coding: utf-8

# # **Credit card fraud detection**

# In this notebook i have tried to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase using algorithm - **Isolation Forest Algorithm** which performs much far better than algo like SVM etc.I have implemented Exploratory Data Analysis with feature scaling to enhance the model.

# If you liked my work, please upvote this kernel as it will  motivate me to perform more in-depth reserach towards this subject.

# In[ ]:


import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from pylab import rcParams
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Loading the dataset

# In[ ]:


data = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv',sep=',')
data.head()


# # Basic info on dataset

# In[ ]:


data.info()


# # **EDA:**

# # Getting target class insight

# In[ ]:


count_class=pd.value_counts(data['Class'])
count_class.plot(kind='bar',rot=0)
plt.title("Transaction Class Distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency") 


# In[ ]:


fraud = data[data['Class']==1]
normal = data[data['Class']==0]
print(fraud.shape,normal.shape)


# In[ ]:


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by class')
bins = 50
ax1.hist(fraud.Amount, bins = bins)
ax1.set_title('Fraud')
ax2.hist(normal.Amount, bins = bins)
ax2.set_title('Normal')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.show();


# so we get that the fraud transactions are generally of far less amount as compared to the non-fraud ones.

# # Correlation insight

# In[ ]:


import seaborn as sns
#get correlations   of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# # Scaling Time and Amount cloumns

# In[ ]:


from sklearn.preprocessing import RobustScaler

rob_scaler = RobustScaler()
data['scaled_amount'] = rob_scaler.fit_transform(data['Amount'].values.reshape(-1,1))
data['scaled_time'] = rob_scaler.fit_transform(data['Time'].values.reshape(-1,1))
data.drop(['Time','Amount'], axis=1, inplace=True)
scaled_amount = data['scaled_amount']
scaled_time = data['scaled_time']
data.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
data.insert(0, 'scaled_amount', scaled_amount)
data.insert(1, 'scaled_time', scaled_time)
data.head()


# # Calculating outlier fraction

# In[ ]:


Fraud = data[data['Class']==1]
Valid = data[data['Class']==0]
outlier_fraction = len(Fraud)/float(len(Valid))
print(outlier_fraction)
print("Fraud Cases : {}".format(len(Fraud)))
print("Valid Cases : {}".format(len(Valid)))


# # Creating independent and Dependent Features

# In[ ]:


columns = data.columns.tolist()
# Filter the columns to remove data we do not want 
columns = [c for c in columns if c not in ["Class"]]
# Store the variable we are predicting 
target = "Class"
# Define a random state 
state = np.random.RandomState(42)
X = data[columns]
Y = data[target]
X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))
# Print the shapes of X & Y
print(X.shape)
print(Y.shape)


# # Modelling:Isolation forest algorithm

# At the basis of the Isolation Forest algorithm there is the tendency of anomalous instances in a dataset to be easier to separate from the rest of the sample (isolate), compared to normal points. In order to isolate a data point the algorithm recursively generates partitions on the sample by randomly selecting an attribute and then randomly selecting a split value for the attribute, between the minimum and maximum values allowed for that attribute.
# 
# In simple language: The algorithm puts weighs on the leaf nodes according to the depth of the tree.So the points which are densely populated have higher weight as the depth is higher and on the other hand the outliers points as less in number weighs less thus seperating them in the algorithm.
# This is how the isolation forest algo does the screening process of outliers and help in anamoly detection.

# In[ ]:


classifier=IsolationForest(n_estimators=100, max_samples=len(X),contamination=outlier_fraction,random_state=state, verbose=0)
n_outliers = len(Fraud)
classifier.fit(X)
scores_prediction = classifier.decision_function(X)
y_pred = classifier.predict(X)
#Reshape the prediction values to 0 for Valid transactions , 1 for Fraud transactions
y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1
n_errors = (y_pred != Y).sum()
# Run Classification Metrics
print("{}: {}".format('ISOLATION FOREST',n_errors))
print("Accuracy Score :")
print(accuracy_score(Y,y_pred))
print("Classification Report :")
print(classification_report(Y,y_pred))

