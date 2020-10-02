#!/usr/bin/env python
# coding: utf-8

# # ** Mushroom Classification Test Classifiers Evaluation
# I created this classifiers evaluation mainly as a submission requirement for the class Data Mining MIT504 and also my personal training on learning the concepts of data mining for Future practices.
# 
# **Overview**
# 
# This dataset consists of 8124 records of the physical characteristics of gilled mushrooms in the Agaricus and Lepiota families, along with their edibility. The classification task is to determine the edibility given the physical characteristics of the mushrooms.
# 
# This dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family Mushroom drawn from The Audubon Society Field Guide to North American Mushrooms (1981). Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous one. The Guide clearly states that there is no simple rule for determining the edibility of a mushroom; no rule like "leaflets three, let it be'' for Poisonous Oak and Ivy.
# 
# Thanks to sir @Raven Duran for sharing his thoughs, it helps me to grasp the basic concept of data science workflow on supervised learning.

# ![](https://storage.googleapis.com/kaggle-datasets-images/478/974/557711140aeab7ca242d1e903c4e058e/dataset-cover.jpg)

# **I. Initialization:**
# Lets try to load our dataset csv file

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# **To Begin with** =
# Lets add some Libraries: 
# *import numpy as np* - for python,
# *import pandas as pd* - parsing our dataset,
# *import seabor as sns* - graphical visualization

# In[ ]:


# Import required libraries
import numpy as np
import pandas as pd
import seaborn as sns

# This is so we can check the files we have in the current system. Most Kaggle Notebook files are found under kaggle/input
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# **Step 1: ** Lets try to display the list of samples on our dataset using Data Frame Shape

# In[ ]:


# Load csv file that we found
# Take note that this data set is just 365kb, so we are confident in putting everything in memory. 
# Fortunately Kaggle gives a provision of 16gb of running memory for Free! 
# So if your data set gets to more than that, I suggest you either breakdown the data or you use a database instead
# as it will be more manageable for larger sets of data
df = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv') # df usually is used to abbreviate "Data Frame" from pandas library


# Remember we are using Python 3 in this Notebook, so the last statement will be evaluated as string for output if any.
# If you want to explicitly print, you can use the "print()" function

# Shape shows you the # of samples and the number of features / cols you have in your data set.
print(f'Data Frame Shape (rows, columns): {df.shape}') 

# This method is one of your basic tool to quickly view data. It shows first 5 rows of a frame
df.head() 


# # I. Quick Data Analysis and Exploration

# I try to balance myour dataset for me to see how many results on the dataset per class(edible or poisonous)

# In[ ]:


sns.countplot(data=df, x="class").set_title("Class Outcome - Edible-e/Poisonous-p")


# As you can see in the above chart ^, the dataset is not balance. This is very important as this is where we can see if our dataset needs balancing to clean up or correct biases or possibility of overfitting. But on my dataset case, I don't really need to balance since the dataset has a large set of values/data with minimal differences. So will skip on balancing the dataset!

# In[ ]:


sns.relplot(data=df, x="bruises", y="odor", hue="class", palette="bright", height=6)


# In corelation to their physical characteristics:
# 
# In here we just plotted the Mushrooms by their physical characteristics (odor and bruises). This is just to visualize the dataset we have in corelation to their characteristics. In here we can observe that the Poisonous(p) class has a higher counts rather that the edible(e)

# In[ ]:


sns.relplot(data=df, x="population", y="habitat", hue="class", palette="bright", height=6)


# In corelation to their Physical Characteristics:
# 
# In here we plotted the mushrooms by their environment. You can see on the plot above that it doesn't differ on that counts of which the mushroom grows to a specific location or nature.

# # II. Data Preparation, Balancing and Cleanup

# In[ ]:


# Check if there are any null values
df.isnull().values.any()


# In[ ]:


# Remove null values
df = df.dropna()


# In[ ]:


# Check if there are any null values
df.isnull().values.any()


# # III. Classifier Setups and Build Model
# 
# # **A. Import Necessary Libraries for Scoring and Evaluation**

# In[ ]:


# Import required libraries for performance metrics
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split


# Now this is the custom function (c.sir Raven) on doing matrix caculation to the Confusion Matrix to get the TP, FP, TN and FN respectively. This function returns an ordered list of the performance measures.
# 
# TP is True Positives FP is False Positives TN is True Negative FN is False Negative

# In[ ]:


def get_performance_measures(actual, prediction):
    matrix = confusion_matrix(actual, prediction)
    FP = matrix.sum(axis=0) - np.diag(matrix)  
    FN = matrix.sum(axis=1) - np.diag(matrix)
    TP = np.diag(matrix)
    TN = matrix.sum() - (FP + FN + TP)

    return(TP, FP, TN, FN)


# Below were custom scorers. A scorer is basically a benchmark of how well your model performs given an actual result and a predicted result.
# 
# There are actually already a comprehensive list of scorers built in Python but for the purpose of learning, we will be creating 2 custom scorers in this notebook: sensitivity and specificity. The rest we just rely on the library to do its work to save up space in this notebook.

# In[ ]:


#Custom Scorers

# # Sensitivity, hit rate, recall, or true positive rate
# TPR = TP/(TP+FN)
# # Specificity or true negative rate
# TNR = TN/(TN+FP) 
# # Precision or positive predictive value
# PPV = TP/(TP+FP)
# # Negative predictive value
# NPV = TN/(TN+FN)
# # Fall out or false positive rate
# FPR = FP/(FP+TN)
# # False negative rate
# FNR = FN/(TP+FN)
# # False discovery rate
# FDR = FP/(TP+FP)

# # Overall accuracy
# ACC = (TP+TN)/(TP+FP+FN+TN)

# Also remember:
# specificity = true negative rate
# sensitivity = true positive rate

def sensitivity_score(y_true, y_pred, mode="multiclass"):
    if mode == "multiclass":
        TP, FP, TN, FN = get_performance_measures(y_true, y_pred)
        TPR = (TP/(TP+FN)).mean()
    elif mode == "binary":
        TP, FP, TN, FN = get_performance_measures(y_true, y_pred)
        TPR = (TP/(TP+FN))[1] # Since the [0] part is the index
    else:
        raise Exception("Mode not recognized!")
    
    return TPR

def specificity_score(y_true, y_pred, mode="multiclass"):
    if mode == "multiclass":
        TP, FP, TN, FN = get_performance_measures(y_true, y_pred)
        TNR = (TN/(TN+FP)).mean()
    elif mode == "binary":
        TP, FP, TN, FN = get_performance_measures(y_true, y_pred)
        TNR = (TN/(TN+FP))[1]
    else:
        raise Exception("Mode not recognized!")
    
    return TNR


# # **B. Setup Scorers**

# In[ ]:


# To know what everaging to use: https://stats.stackexchange.com/questions/156923/should-i-make-decisions-based-on-micro-averaged-or-macro-averaged-evaluation-mea#:~:text=So%2C%20micro%2Daveraged%20measures%20add,is%20more%20like%20an%20average.


scoring = {
            'accuracy':make_scorer(accuracy_score), 
            'precision':make_scorer(precision_score, average='weighted'),
            'f1_score':make_scorer(f1_score, average='weighted'),
            'recall':make_scorer(recall_score, average='weighted'), 
            'sensitvity':make_scorer(sensitivity_score, mode="binary"), 
            'specificity':make_scorer(specificity_score, mode="binary"), 
           }


# # **C. Setting up our Classifiers**

# In[ ]:


# Import required libraries for machine learning classifiers
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.naive_bayes import GaussianNB #Naive Bayes
from sklearn.linear_model import LogisticRegression #Logistic Regression
from sklearn.svm import LinearSVC # Support Vector Machine
from sklearn.neighbors import KNeighborsClassifier #K-nearest Neighbors
from sklearn.cluster import KMeans #K-means

# Instantiate the machine learning classifiers
decisionTreeClassifier_model = DecisionTreeClassifier()
gaussianNB_model = GaussianNB()
logisticRegression_model = LogisticRegression(max_iter=10000)
linearSVC_model = LinearSVC(dual=False)
kNeighbors_model = KNeighborsClassifier()


# Let's make a custom evaluation function to easily aggregate our score results. You can actually do this manually, but by making a function; we can easily reuse it easily for future use.
# 
# The beauty of this function is that it is very versatile and can be use not only for this data set but also to other data sets as well as long as you format them properly.

# In[ ]:


# features = data frame set that contain your features that will be used as input to see if prediction is equal to actual result
# target = data frame set (1 column usually) that will contain your target or actual results.
# folds = this is added so we can easily change the number of folds we want to do with our data set.
# folding is a technique to minimise overfitting and therefore make our model more accurate.
def models_evaluation(features, target, folds):    
    # Perform cross-validation to each machine learning classifier
decisionTreeClassifier_result = cross_validate(decisionTreeClassifier_model, features, target, cv=folds, scoring=scoring)
gaussianNB_result = cross_validate(gaussianNB_model, features, target, cv=folds, scoring=scoring)
logisticRegression_result = cross_validate(logisticRegression_model, features, target, cv=folds, scoring=scoring)
linearSVC_result = cross_validate(linearSVC_model, features, target, cv=folds, scoring=scoring)
kNeighbors_result = cross_validate(kNeighbors_model, features, target, cv=folds, scoring=scoring)
    # kMeans_result = cross_validate(kMeans_model, features, target, cv=folds, scoring=scoring)

    # Create a data frame with the models perfoamnce metrics scores
    models_scores_table = pd.DataFrame({
      'Decision Tree':[
                        decisionTreeClassifier_result['test_accuracy'].mean(),
                        decisionTreeClassifier_result['test_precision'].mean(),
                        decisionTreeClassifier_result['test_recall'].mean(),
                        decisionTreeClassifier_result['test_sensitvity'].mean(),
                        decisionTreeClassifier_result['test_specificity'].mean(),
                        decisionTreeClassifier_result['test_f1_score'].mean()
                       ],

      'Gaussian Naive Bayes':[
                                gaussianNB_result['test_accuracy'].mean(),
                                gaussianNB_result['test_precision'].mean(),
                                gaussianNB_result['test_recall'].mean(),
                                gaussianNB_result['test_sensitvity'].mean(),
                                gaussianNB_result['test_specificity'].mean(),
                                gaussianNB_result['test_f1_score'].mean()
                              ],

      'Logistic Regression':[
                                logisticRegression_result['test_accuracy'].mean(),
                                logisticRegression_result['test_precision'].mean(),
                                logisticRegression_result['test_recall'].mean(),
                                logisticRegression_result['test_sensitvity'].mean(),
                                logisticRegression_result['test_specificity'].mean(),
                                logisticRegression_result['test_f1_score'].mean()
                            ],

      'Support Vector Classifier':[
                                    linearSVC_result['test_accuracy'].mean(),
                                    linearSVC_result['test_precision'].mean(),
                                    linearSVC_result['test_recall'].mean(),
                                    linearSVC_result['test_sensitvity'].mean(),
                                    linearSVC_result['test_specificity'].mean(),
                                    linearSVC_result['test_f1_score'].mean()
                                   ],

       'K-nearest Neighbors':[
                        kNeighbors_result['test_accuracy'].mean(),
                        kNeighbors_result['test_precision'].mean(),
                        kNeighbors_result['test_recall'].mean(),
                        kNeighbors_result['test_sensitvity'].mean(),
                        kNeighbors_result['test_specificity'].mean(),
                        kNeighbors_result['test_f1_score'].mean()
                       ],

      },

      index=['Accuracy', 'Precision', 'Recall', 'Sensitivity', 'Specificity', 'F1 Score', ])
    
    # Return models performance metrics scores data frame
    return(models_scores_table)


# In[ ]:





# In[ ]:


# Let's try to look at our data frame again one last time
df.head()


# # **D. Preparing Features and Targets**

# In[ ]:


# Actually what we are doing here is that we are just dropping the Species column since that is our class
# and the remaining columns will then be our features (eg. inputs to come up to a class)
# axis 0 basically means to drop all of that column
features = df.drop(columns="class", axis=0)

# Now let's see what features looks like
features

# Don't mind the left hand side, those are just index mainly used for viewing


# In[ ]:


#Specify target column
# Now we try to get the frame of only our target. Which is the "Class" column
target = df["class"]

# Do note that csv files are also zero-index, that means a row starts from zero.
target


# # IV. Running our Evaluation

# In[ ]:


evaluationResult = models_evaluation(features, target, 5)
view = evaluationResult
view = view.rename_axis('Test Type').reset_index() #Add the index names to the column. This will be used for our presentation

# https://pandas.pydata.org/docs/reference/api/pandas.melt.html
# Re-Organizing our dataframe to fit our view need
view = view.melt(var_name='Classifier', value_name='Value', id_vars='Test Type')
# result
sns.catplot(data=view, x="Test Type", y="Value", hue="Classifier", kind='bar', palette="bright", alpha=0.8, legend=True, height=5, margin_titles=True, aspect=2)

