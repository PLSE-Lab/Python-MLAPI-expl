#!/usr/bin/env python
# coding: utf-8

# # Credit card fraud detection
# Exploring the strategies for dealing with skewed datasets using SMOTE, Ensemble techniques, cross valdiation, kFold and Logistic regression.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# **Loading dataset**

# In[ ]:


data = pd.read_csv("../input/creditcard.csv")


# **Test for skewness**

# In[ ]:


import seaborn as sns
sns.countplot(x='Class', data=data)


# **Definition of unbalances is less than 10% of one class and 90% of another class. Therefore, our data is clearly skewed**

# **Normalize the amount column and drop any redundant columns i.e. time and amount**

# In[ ]:


from sklearn.preprocessing import StandardScaler
data['normAmount'] = StandardScaler().fit_transform(data['Amount'].reshape(-1, 1))
data = data.drop(['Time','Amount'],axis=1)


# #### 2. Assigning X and Y. No resampling.

# #### 3. Resampling.
# 
# - As we mentioned earlier, there are several ways to resample skewed data. Apart from under and over sampling, there is a very popular approach called SMOTE (Synthetic Minority Over-Sampling Technique), which is a combination of oversampling and undersampling, but the oversampling approach is not by replicating minority class but constructing new minority class data instance via an algorithm.
# 
# - In this notebook, we will use traditional UNDER-sampling. I will probably try to implement SMOTE in future versions of the code, but for now I will use traditional undersamplig.
# 
# - The way we will under sample the dataset will be by creating a 50/50 ratio. This will be done by randomly selecting "x" amount of sample from the majority class, being "x" the total number of records with the minority class.

# In[ ]:


X = data.ix[:, data.columns != 'Class']
y = data.ix[:, data.columns == 'Class']


# **Applying SMOTE with Tomek lines** 
# PRACTICE: Here we are using only one imblearn method for practice try different sampling techniques from the imblearn package

# In[ ]:


#Apply SMOTE to create synthetic observations from minority class
#Source:https://github.com/scikit-learn-contrib/imbalanced-learn
from imblearn.combine import  SMOTETomek #Over sampling
ee = SMOTETomek()
X_sampled,y_sampled  = ee.fit_sample(X, y.values.ravel())

#Number of records in original data
print('Total rows in original data:{}'.format(X.shape[0]))

#Percentage of fraudlent records in original data
Source_data_no_fraud_count = len(data[data.Class==0])
Source_data_fraud_count = len(data[data.Class==1])
print('Percentage of fraud counts in original dataset:{}%'.format((Source_data_fraud_count*100)/(Source_data_no_fraud_count+Source_data_fraud_count)))

#Number of records in original data
print('Total rows from ensembled data :{}'.format(X_sampled.shape[0]))

#Percentage of fraudlent records in sampled data
Sampled_data_no_fraud_count = len(y_sampled[y_sampled==0])
Sampled_data_fraud_count = len(y_sampled[y_sampled==1])
print('Percentage of fraud counts in the new data:{}%'.format((Sampled_data_fraud_count*100)/(Sampled_data_no_fraud_count+Sampled_data_fraud_count)))


# **Cross Validation** We need to split our training data into training set and test set. Our classifier will learn its parameters on the training set. 30% data will be excluded from training set and kept in test set.

# In[ ]:


from sklearn.cross_validation import train_test_split

# Whole dataset
X_train, X_test, y_train, y_test = train_test_split(X_sampled,y_sampled,test_size = 0.3, random_state = 0)

print("Number transactions train dataset: ", len(X_train))
print("Number transactions test dataset: ", len(X_test))
print("Total number of transactions: ", len(X_train)+len(X_test))

# Undersampled dataset
X_train_sampled, X_test_sampled, y_train_sampled, y_test_sampled = train_test_split(X_sampled,y_sampled,test_size = 0.3,random_state = 0)
print("")
print("Number transactions train in sampled dataset: ", len(X_train_sampled))
print("Number transactions test in sampled dataset: ", len(X_test_sampled))
print("Total number of transactions: ", len(X_train_sampled)+len(X_test_sampled))

X_train_sampled_df = pd.DataFrame(X_train_sampled)
y_train_sampled_df = pd.DataFrame(y_train_sampled)
X_test_sampled_df = pd.DataFrame(X_test_sampled)
y_test_sampled_df = pd.DataFrame(y_test_sampled)


# # Logistic regression classifier - Undersampled data
# 
# #### We are very interested in the recall score, because that is the metric that will help us try to capture the most fraudulent transactions. If you think how Accuracy, Precision and Recall work for a confusion matrix, recall would be the most interesting:
# 
# - Accuracy = (TP+TN)/total
# - Precision = TP/(TP+FP)
# - Recall = TP/(TP+FN)
# 
# #### As we know, due to the imbalacing of the data, many observations could be predicted as False Negatives, being, that we predict a normal transaction, but it is in fact a fraudulent one. Recall captures this.
# - Obviously, trying to increase recall, tends to come with a decrease of precision. However, in our case, if we predict that a transaction is fraudulent and turns out not to be, is not a massive problem compared to the opposite. 
# - We could even apply a cost function when having FN and FP with different weights for each type of error, but let's leave that aside for now.

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 


# **We will use KFold to identify the optiomal configuration for the LogisticClassifier.**
# 
#  1. Separate data into 5 folds. Define different parameter values to be 
#     used i.e. c_param_range For each value of parameter do an iteration.
#     In every iteration we will do 5 iterations with different data range
#     and calculate their score. For each iteration we will calculate the 
#     mean score. Once we know the score from each iteration we determine 
#     what was the maximum score out of the 5 iterations. We then use the 
#     corresponding configuration value as our best configuration value.  
#     This value will be used to setup our model in the next step.

# In[ ]:


def printing_Kfold_scores(x_train_data,y_train_data):
    fold = KFold(len(y_train_data),5,shuffle=False) 

    # Different C parameters
    c_param_range = [0.01,0.1,1,10,100]

    results_table = pd.DataFrame(index = range(len(c_param_range),2), columns = ['C_parameter','Mean recall score'])
    results_table['C_parameter'] = c_param_range

    # the k-fold will give 2 lists: train_indices = indices[0], test_indices = indices[1]
    j = 0
    for c_param in c_param_range:
        print('-------------------------------------------')
        print('C parameter: ', c_param)
        print('-------------------------------------------')
        print('')

        recall_accs = []
        for iteration, indices in enumerate(fold,start=1):

            # Call the logistic regression model with a certain C parameter
            lr = LogisticRegression(C = c_param, penalty = 'l1')

            # Use the training data to fit the model. In this case, we use the portion of the fold to train the model
            # with indices[0]. We then predict on the portion assigned as the 'test cross validation' with indices[1]
            lr.fit(x_train_data.iloc[indices[0],:],y_train_data.iloc[indices[0],:].values.ravel())

            # Predict values using the test indices in the training data
            y_pred_undersample = lr.predict(x_train_data.iloc[indices[1],:].values)

            # Calculate the recall score and append it to a list for recall scores representing the current c_parameter
            recall_acc = recall_score(y_train_data.iloc[indices[1],:].values,y_pred_undersample)
            recall_accs.append(recall_acc)
            print('Iteration ', iteration,': recall score = ', recall_acc)

        # The mean value of those recall scores is the metric we want to save and get hold of.
        results_table.ix[j,'Mean recall score'] = np.mean(recall_accs)
        j += 1
        print('')
        print('Mean recall score ', np.mean(recall_accs))
        print('')

    best_c = results_table.loc[results_table['Mean recall score'].idxmax()]['C_parameter']
    
    # Finally, we can check which C parameter is the best amongst the chosen.
    print('*********************************************************************************')
    print('Best model to choose from cross validation is with C parameter = ', best_c)
    print('*********************************************************************************')
    
    return best_c


# **Find the best score based using kfold technique**

# In[ ]:


best_c = printing_Kfold_scores(X_train_sampled_df,y_train_sampled_df)


# **Create a function to plot a fancy confusion matrix. Alternatively, on your local machine you can use the easier method of using python_ml package which has an API to readily create the confusion matrix.**

# In[ ]:


import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# ### Predictions on test set and plotting confusion matrix
# 
# #### We have been talking about using the recall metric as our proxy of how effective our predictive model is. Even though recall is still the recall we want to calculate, just bear mind in mind that the undersampled data hasn't got a skewness towards a certain class, which doesn't make recall metric as critical. 

# In[ ]:


# Use this C_parameter to build the final model with the whole training dataset and predict the classes in the test
# dataset
lr = LogisticRegression(C = best_c, penalty = 'l1')
lr.fit(X_train_sampled_df,y_train_sampled_df.values.ravel())
y_pred_sample = lr.predict(X_test_sampled_df.values)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test_sampled_df,y_pred_sample)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: {}%".format(100*cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])))

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix , classes=class_names, title='Confusion matrix')
plt.show()


# The model is offering an 91.65% recall accuracy on the generalised unseen data (test set). Not a bad percentage to be the first try. However, recall this is a 91.65% recall accuracy measure on a sampled test set.Now, let's apply the model we fitted and test it on the whole data.

# In[ ]:


# Use this C_parameter to build the final model with the whole training dataset and predict the classes in the test
# dataset
lr = LogisticRegression(C = best_c, penalty = 'l1')
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test,y_pred)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset:{}% ".format(100*(cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))))

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()


# ### Still a very decent recall accuracy when applying it to a much larger and skewed dataset! 
# 
# #### We can start to be happy with how initial approach is working.

# ### Plotting ROC curve and Precision-Recall curve.
# 
# - I find precision-recall curve much more convenient in this case as our problems relies on the "positive" class being more interesting than the negative class, but as we have calculated the recall precision, I am not going to plot the precision recall curves yet.
# 
# - AUC and ROC curve are also interesting to check if the model is also predicting as a whole correctly and not making many errors
# - The higer the AUC (Area Under Curve) score the better. It quantifies a measurement.

# In[ ]:


# ROC CURVE
lr = LogisticRegression(C = best_c, penalty = 'l1')
temp = lr.fit(X_train,y_train)

y_pred_sampled_score= temp.decision_function(X_test)

fpr, tpr, thresholds = roc_curve(y_test,y_pred_sampled_score)
roc_auc = auc(fpr,tpr)

# Plot ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# #### An additional comment that would be interesting to do is to initialise multiple undersampled datasets and repeat the process in loop. Remember that, to create an undersample data, we randomly got records from the majority class. Even though this is a valid technique, is doesn't represent the real population, so it would be interesting to repeat the process with different undersample configurations and check if the previous chosen parameters are still the most effective. In the end, the idea is to use a wider random representation of the whole dataset and rely on the averaged best parameters.

# # Logistic regression classifier - Skewed data
# 
# #### Having tested our previous approach, I find really interesting to test the same process on the skewed data. Our intuition is that skewness will introduce issues difficult to capture, and therefore, provide a less effective algorithm.
# - To be fair, taking into account the fact that the train and test datasets are substantially bigger than the undersampled ones, I believe a K-fold cross validation is necessary. I guess that by splitting the data with 60% in training set, 20% cross validation and 20% test should be enough... but let's take the same approach as before (no harm on this, it's just that K-fold is computationally more expensive)

# In[ ]:


best_c = printing_Kfold_scores(X_train,y_train)


# In[ ]:


# Use this C_parameter to build the final model with the whole training dataset and predict the classes in the test
# dataset
lr = LogisticRegression(C = best_c, penalty = 'l1')
lr.fit(X_train,y_train.values.ravel())
y_pred = lr.predict(X_test.values)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test,y_pred)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))
#Recall = TP/(TP+FN)
print("True positive: {}".format(cnf_matrix[1,1]))
print("True negatives: {}".format(cnf_matrix[1,0]))
# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()


# In[ ]:


from sklearn.metrics import recall_score
recall_score(y_test,y_pred)    


# # Before continuing... changing classification threshold.
# 
# #### We have seen that by undersampling the data, our algorithm does a much better job at detecting fraud. I wanted also to show how can we tweak our final classification by changing the thresold.
# - Initially, you build the classification model and then you predict unseen data using it.
# - We previously used the "predict()" method to decided whether a record should belong to "1" or "0". 
# - There is another method "predict_proba()".
#     - This method returns the probabilities for each class. The idea is that by changing the threshold to assign a record to class 1, we can control precision and recall.
#     
# #### Let's check this using the undersampled data (best C_param = 0.01)

# In[ ]:


lr = LogisticRegression(C = 0.01, penalty = 'l1')
lr.fit(X_train_undersample,y_train_undersample.values.ravel())
y_pred_undersample_proba = lr.predict_proba(X_test_undersample.values)

thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

plt.figure(figsize=(10,10))

j = 1
for i in thresholds:
    y_test_predictions_high_recall = y_pred_undersample_proba[:,1] > i
    
    plt.subplot(3,3,j)
    j += 1
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test_undersample,y_test_predictions_high_recall)
    np.set_printoptions(precision=2)

    print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

    # Plot non-normalized confusion matrix
    class_names = [0,1]
    plot_confusion_matrix(cnf_matrix
                          , classes=class_names
                          , title='Threshold >= %s'%i) 


# ### The pattern is very clear: the more you lower the required probability to put a certain in the class "1" category, more records will be put in that bucket.
# #### This implies an increase in recall (we want all the "1"s), but at the same time, a decrease in precision (we misclassify many of the other class).
# ### Therefore, even though recall is our goal metric (do not miss a fraud transaction), we also want to keep the model being accurate as a whole.
# - There is an option I think could be quite interesting to tackle this. We could assing cost to misclassifications, but being interested in classifying "1s" correctly, the cost for misclassifying "1s" should be bigger than "0" misclassifications. After that, the algorithm would select the threshold which minimises the total cost. A drawback I see is that we have to manually select the weight of each cost... therefore, I will leave this know as a thought.
# - Going back to the threshold changing, there is an option which is the Precisio-Recall curve. By visually seeing the performance of the model depending on the threshold we choose, we can investigate a sweet spot where recall is high enough whilst keeping a high precision value.
# 
# ### Investigate Precision-Recall curve and area under this curve.

# In[ ]:


from itertools import cycle

lr = LogisticRegression(C = 0.01, penalty = 'l1')
lr.fit(X_train_undersample,y_train_undersample.values.ravel())
y_pred_undersample_proba = lr.predict_proba(X_test_undersample.values)

thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'red', 'yellow', 'green', 'blue','black'])

plt.figure(figsize=(5,5))

j = 1
for i,color in zip(thresholds,colors):
    y_test_predictions_prob = y_pred_undersample_proba[:,1] > i
    
    precision, recall, thresholds = precision_recall_curve(y_test_undersample,y_test_predictions_prob)
    
    # Plot Precision-Recall curve
    plt.plot(recall, precision, color=color,
                 label='Threshold: %s'%i)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall example')
    plt.legend(loc="lower left")


# # Upcoming updates: 
# ## testing SVMs
# ## testing decision trees

# In[ ]:


#Introudcing SMOTE method for handling imbalanced data problem. 


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train_sampled_df)
rf_predictions = rf.predict(X_test_sampled_df)


# In[ ]:


sm = SMOTE(ratio='auto',kind='regular')    


# In[ ]:


X_resampled, y_resampled  = sm.fit_sample(X,y.values.ravel())


# In[ ]:


X_resampled.shape


# In[ ]:


y_resampled.shape


# In[ ]:


df =pd.DataFrame(X_resampled)


# In[ ]:


df.features


# In[ ]:




