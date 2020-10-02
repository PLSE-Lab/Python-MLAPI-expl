#!/usr/bin/env python
# coding: utf-8

# ### Movie Profitability Classification
# 
# We want to predict whether a movie is profitable or not.
# 
# My definition of profitable is as follows:
# * A movie is profitable if it's revenue is larger than 2x it's budget.
# 
# I have already pre-processed a dataset, so we can get right to training our model!
# 
# Let's look at data briefly...
# * Class 0 = Unprofitable
# * Class 1 = Profitable

# ### Setup
# Imports reference:
# * [os](https://docs.python.org/3/library/os.html)
#     * Gives us access to file system to load in our data
# * [numpy](http://)
#     * Numerical python - Scientific computing with Python
# * [pandas](https://pandas.pydata.org/)
#     * Dataframes - An open source data analysis and manipulation tool
# * [seaborn](https://seaborn.pydata.org/)
#     * Open source statistical data visualization tool
# * [SKLearn](https://scikit-learn.org/stable/)
#     * Scikit-learn - An open source tool for machine learning and predictive data analysis

# In[ ]:


# Imports
import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn import preprocessing

# Load in the data
dataset = pd.read_csv("/kaggle/input/processed_movie_data.csv")
print("Dataset rows: %s, \nDataset columns: %s" %(dataset.shape))


# ### Visualizing the data
# 
# Let's look at an example of how we can identify whether a descriptive feature is informative about the target class.

# In[ ]:


# Example of undescriptive feature


# In[ ]:


# Example of descriptive feature


# These are the most informative features in the dataset:
# * **Director Ratio**
#     * The average profit-budget ratio of the director's other films
# * **Keywords Ratio**
#     * The average profit-budget ratio of the keywords in the film's summary
# * **Studios Ratio**
#     * The average profit-budget ratio of the studio's other films
# * **Lead Actor Ratio**
#     * The average profit-budget ratio of the lead actor's other films
# * **Budget**
#     * The budget of the film
#     
# So, lets create a dataset with only these features.

# In[ ]:


# Create a list of the most informative features' column names
informative_features = ['Director_Ratio', 'Keywords_Ratio', 'Studios_Ratio', 'Lead_Actor_Ratio', 'Budget']

# Isolate the target feature's values


# Create a new dataset, with only the most informative descriptive features


# Time to prepare our data and scikit-learn classifier!

# In[ ]:


# Store target feature values in a numpy array (Originally in a 2d list)


# Create a scaler


# Fit the data to the scaler


# Normalize our descriptive features to range [0,1]


# Classifier
clf = MLPClassifier(solver='adam', alpha=1e-5, random_state=1, max_iter=1000)


# Now, we set up our 5-fold cross validation for training and testing.
# * [5-fold cross validation example](https://miro.medium.com/max/2736/1*rgba1BIOUys7wQcXcL4U5A.png)

# In[ ]:


# Create a new 5-fold cross validator object

# Give it the data to split up


# In[ ]:


# Objects which will store performances among the 5 folds
# These will be used to compute average performance overall

# prediction_accuracies.
confusion_matrices = []
classification_accuracies = []

# predictions.
model_predictions = [] 
target_predictions = []


# Now, for each partition in the k-fold validation:
# * Split the descriptive features (X) and target feature (Y) into a test set and training set
# * Get the classification accuracy
# * Get the confusion matrix
#     

# In[ ]:


for train_index, test_index in kf.split(X):
    
    # X = Descriptive features for each partition

    
    # Y = Target feature values for each partition


    # Train the model using training sets

    
    # Get the predictions of the model

    
    # Get accuracy score of the model
    # (comparing model's predictions of profitability to expected values in the data)
    classification_accuracies.append(

    )

    # Get confusion matrix of the model
    confusion_matrices.append(

    )


# In[ ]:


print(classification_accuracies)


# In[ ]:


# Aggregate the 5 confusion matrices into a single one.
totals = np.array([[0, 0],[0, 0]])
for matrix in confusion_matrices:
    for i in range(0, 2):
        for j in range(0, 2):
            totals[i][j] += matrix[i][j]
tn, fp, fn, tp = totals.ravel()

# Print confusion matrix
print("Confusion Matrix: ")
for s1, s2 in [['tp: '+ str(tp), 'fp: '+ str(fp)],
               ['fn: '+ str(fn), 'tn: '+ str(tn)]]:
    print("[%-8s | %-8s]" %(s1, s2))


# ### Evaluating the model
# 
# Now we will evaluate the models performance by using some commonly used performance measures

# In[ ]:


# Classification accuracies
avg_classification_accuracy = sum(classification_accuracies)/len(classification_accuracies)
print("Accuracy: %.9f %%" %(avg_classification_accuracy*100))


# In[ ]:


# Calculate true positive rate


# Calculate true negative rate


# Calculate false positive rate


# Calculate false negative rate


# Print the outcomes
for key, value in [['True Positive Rate',tpr], 
                   ['True Negative Rate',tnr], 
                   ['False Positive Rate',fpr], 
                   ['False Negative Rate',fnr]]:
    print("%s: %.4f" %(key, value)) 


# ### Calculate some of the advanced metrics:
# * **Precision**
#     * % of positive predictions which are actually correct
# * **Recall**
#     * % of positive predictions which are found

# In[ ]:


# Calculate precision


# Calculate recall


for key, value in [['Precision', precision],
                   ['Recall', recall]]:
    print("%s: %.4f" %(key, value)) 


# In[ ]:


# Select Avatar dataset
avatar = dataset.loc[dataset['Movie_Title']=='Avatar'][informative_features]
# Predict whether it is successful or not
out = clf.predict(scaler.transform(avatar.values))
# Output
print("profitable" if out[0] == 1 else "unprofitable")


# In[ ]:


# Select Avatar dataset
iron_man_3 = dataset.loc[dataset['Movie_Title']=='Iron Man 3'][informative_features]
# Predict whether it is successful or not
out = clf.predict(scaler.transform(iron_man_3.values))
# Output
print("profitable" if out[0] == 1 else "unprofitable")


# In[ ]:


# Predict whether it is successful or not
home_for_the_holidays = dataset.loc[dataset['Movie_Title']=='Home for the Holidays'][informative_features]
out = clf.predict(scaler.transform(home_for_the_holidays.values))
# Output
print("profitable" if out[0] == 1 else "unprofitable")

