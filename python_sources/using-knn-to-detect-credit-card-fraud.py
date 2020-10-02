#!/usr/bin/env python
# coding: utf-8

# As my first Machine Learning project, I was looking for a dataset that was already prepared for the application of supervised learning algorithms. In particular, I found this credit card fraud dataset which as written in its' description, is a PCA transformation of a very unbalanced dataset (248,807 transactions of which only 492 cases are fraudulent). Given that we have a mutli-dimensional feature set I hypothesised that if we comb through the 28 dimesnions in visualizable chunks of 2 or 3 dimensions and there is an obvious pattern in 'grouping' of transactions by class, then considering all dimensions at once, we should be able to identify the classification of a transaction based on a vote of the class from k-nearest previous cases in a complete high-dimensional space.

# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# To check that the positive cases of fraud are not located sporadically in the space created by two components of the 28 PCA transofrmed variables (causing a nearest neighbor vote to be insiginificant). We load the dataset into a dataframe, split it by class using a simple query and examine the scatter plots of some of the components. Given that I am relying only on the location of the transaction given by its features, I will not do time, or cost dependent analysis, so I will drop the "Time" and "Amount" columns of the dataframe; because, they were not included in the PCA transformation, and so the comparison to transformed variables would likely be counter productive for precision, recall and average precision scores.

# In[ ]:


import matplotlib.pyplot as plt #to plot the PCA variables in twos
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
plt.style.use('ggplot') 

df = pd.read_csv('../input/creditcard.csv',header=0) #load data into pandas DataFrame
df.drop(['Time'],1,inplace=True) #Drop the Time and Amount columns
df.drop(['Amount'],1,inplace=True)
#create two sub-DataFrames, based on the classification of the transaction
pos_df = df.query('Class==1') #sub-dataframe with only fraudulent transactions
neg_df = df.query('Class==0') #sub-dataframe with only valid transactions
'''we have 14 plots of 2 components each and since the components are ordered by variance, 
the first one accounts for the largest amount of variability of the data and so
we obtain the most general idea of the classification distribution with the first.
We also obtain a remainder of 12 plots which can be plotted in a more square subplot-grid'''
i = 1
fig1, two_axes = plt.subplots(1,2)
for ax in two_axes:
    y_axis = 'V'+str(i)
    x_axis = 'V'+str(i+1)
    ax.scatter(neg_df[x_axis],neg_df[y_axis],color='b',label='Valid',marker='.')
    ax.scatter(pos_df[x_axis],pos_df[y_axis],color='r',label='Fraudulent',marker='+')
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    i += 2
plt.legend()
plt.show()

fig2, axes_array = plt.subplots(3,4)

for plot_row in axes_array:
    for ax in plot_row:
        y_axis = 'V'+str(i)
        x_axis = 'V'+str(i+1)
        ax.scatter(neg_df[x_axis],neg_df[y_axis],color='b',label='Valid',marker='.')
        ax.scatter(pos_df[x_axis],pos_df[y_axis],color='r',label='Fraudulent',marker='+')
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        
        i += 2
#plt.legend()
plt.tight_layout(pad=0.1)
plt.show()


# Looking at the positioning and grouping of the fraudulent cases in all of the subplots we can see that there are only a handful of outlier cases, which with a k-nearest vote could be 'invisible' to the model attempting to identify them as fraudulent transactions (if these outliers are positioned in a cluster of positive cases within a different component dimension we may still be able to identify them). Therefore, it seems that a k-nearest algorithm would be able to correctly identify fraudulent cases.

# Now we will implement the KNN algorithm from the sklearn module:

# In[ ]:


import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, neighbors
from sklearn.metrics import confusion_matrix, average_precision_score

df = pd.read_csv('../input/creditcard.csv',header=0) #load data into pandas DataFrame
df.drop(['Time'],1,inplace=True) #Drop the Time and Amount columns
df.drop(['Amount'],1,inplace=True)

x_f = df.drop(['Class'],1) #create the feature set, removing the classification
y_c = df['Class'] #corresponding classification of the transactions

#randomly split dataframe into training and testing sub-sets, specify test set proportion
x_train, x_test, y_train, y_test = model_selection.train_test_split(x_f,y_c,test_size=0.2)
print('completed model selection')
classifier = neighbors.KNeighborsClassifier() #initialize the KNN classifier
classifier.fit(x_train,y_train) #fit the training data to the training classification
print('completed data fitting')

y_pred = classifier.predict(x_test) #generate predictions from the features of the test set
print('completed predictions')
TN,FP,FN,TP = confusion_matrix(y_test,y_pred).ravel() #obtain True/False; Positives/Negatives
P = round(TP/(TP+FP),5)*100 #precision percentage
R = round(TP/(TP+FN),5)*100 #recall percentage
avg_P = round(average_precision_score(y_test, y_pred),5)*100 #average percision score %
print('precision = TP / (TP + FP) : {} %'.format(P))
print('recall = TP / (TP + FN) : {} %'.format(R))
print('harmonic mean = 2(P*R)/(P+R) : {} % '.format(round(2*(P*R)/(P+R),3)))
print('average precision score is:',avg_P,'%')
print('accuracy = TP + TN / (ALL) : {} %'.format(round((TP+TN)/(TP+FP+TN+FN),5)*100) )


# As we can see in the accuracy scores, our model correctly classifies the vast majority of negative cases (valid transactions); however, given that there are only 492 positive cases corresponding to 1.7% of the entire dataset, given that it 'works' for negative cases we are mainly interested in precision and recall scores.
# 
# Running the model repeatedly, we obtain a minimum precision score of around 90%, depending on the split of data included in the testing set. The recall score is always lower than the precision score, indicating that we have more False Negatives than False Positives. Given that we are prediciting fraudulent cases of transactions, we *prefer*  a higher threshold for type I (FP) errors over type II (FN) errors. In an unbalanced dataset as this one, we are looking for 'needles in a haystack'. Hence, missing cases of fraudulent transactions is more detrimental than falsely identifying valid transactions as fraudulent, given that in practice the investigation into a transaction (once identified as fraudulent by the alogrithm) can occur with adidtional information; ie. we would have more 'needles' to inspect. However, if the the prediction was a False Negative, then our algorithm has identified it as a valid transaction and we cannot go into further detail to ultimately achieve the goal of being able to identify fraudulent credit card transactions.

# **Possible Next Steps **
# 
# Looking back into the visualization of the PCA components above, there are a few '+' cases that are on their 'own'; (do not have 3 or more positive cases surrounding them for a correct classification vote using 5 nearest neighbors). It is likely that these are the False Negative cases that are the most adverse to the performance of the algorithm. 
# 
# To further improve the classification we could create seperating hyperplanes (using some pre-defined decision boundary) to identify datapoints that can be classified as valid transactions based only on, *for example* the (V1, V2) coordinate. If we seperate the data in this way before applying the nearest neighbor algorithm, then there exists the possibility that we would remove 'wrong' votes and thus reduce the count of False Negatives. At the same time, removing valid cases before voting would increase the chance for False Positives as well (which would be O.K. given our error threshohld preferences). Ultimately, if the decrease in False Negatives is larger than the increase in False Positives, then the SVM layer would be beneficial to the algorithm. Furthermore, the high dimensionality of the data-set is not ideal for the algorithm in terms of prediction speed, so being able to eliminate certain data-points could prove useful. Equally, we can try to identify certain PCA components that do not significantly alter the performance of the predictor and remove them from the training and testing data as a part of the pre-processing.
