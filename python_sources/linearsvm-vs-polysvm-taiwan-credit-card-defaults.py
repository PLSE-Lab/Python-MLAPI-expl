#!/usr/bin/env python
# coding: utf-8

# # Taiwan Credit Card Defaults
# This was originally an assignment for a Machine Learning class at National Taiwan Normal University. The goal of this assignment is to implement a machine learning system to analyze and make predictions about the Taiwan Credit Card defaults dataset. This dataset is one of the standard benchmark datasets in the UCI Machine Learning repository. The two goals of the assignment is to:
# 
# 1. predict a default on the next credit card payment using SVM
# 2. cluster the customers into k groups using kMeans clustering
# 
# I've wrote the code focusing more on readability than speed.
# 
# ### Importing modules

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
sns.set_style("whitegrid")


# ### Loading the Dataset
# The dataset contains the following features:
# * ID: ID of each client
# * LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit
# * SEX: Gender (1=male, 2=female)
# * EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
# * MARRIAGE: Marital status (1=married, 2=single, 3=others)
# * AGE: Age in years
# * PAY_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, ... 8=payment delay for eight months, 9=payment delay for nine months and above)
# * PAY_2: Repayment status in August, 2005 (scale same as above)
# * PAY_3: Repayment status in July, 2005 (scale same as above)
# * PAY_4: Repayment status in June, 2005 (scale same as above)
# * PAY_5: Repayment status in May, 2005 (scale same as above)
# * PAY_6: Repayment status in April, 2005 (scale same as above)
# * BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
# * BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
# * BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
# * BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
# * BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
# * BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
# * PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
# * PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
# * PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
# * PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
# * PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
# * PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)
# * **default.payment.next.month: Default payment (1=yes, 0=no)** -> This is the value we want to predict

# In[ ]:


dataset = pd.read_csv("../input/UCI_Credit_Card.csv")
print(dataset.head())
print(f"Amount of samples: {dataset.shape[0]}")


# ### Preliminar data analysis
# 
# Let's check the balance of the dataset for each of the nominal features. Here we print the number of samples for each value for the features SEX, EDUCATION, MARRIAGE and AGE. Also, let's plot the feature we want to predict (default.payment.next.month) to check the balance of the dataset.
# 
# We can observe that we have a much large number of females than males in the dataset. Also most of the people have an university level education, are single and are around 30 years old.
# Most importantly, we can notice that the dataset is clearly not balanced for the feature we want to predict. SVMs are sensitive to unbalanced data, so we have to assign different class weightings based on the number of samples of each class.

# In[ ]:


def plotDist(dataset):
    features_to_plot = ["SEX", "EDUCATION", "MARRIAGE", "default.payment.next.month"]
    # Define integer to string mappings for a pretty graph
    intToStr = {"SEX": {1: "Male", 2: "Female"},
                "EDUCATION": {1: "Graduate School", 2: "University", 3: "High School", 4: "Other", 5: "Unknown", 6: "Unknown", 0: "Unknown"},
                "MARRIAGE": {1: "Married", 2: "Single", 3: "Other", 0: "Unknown"},
                "default.payment.next.month": {0: "No", 1: "Yes"}
               }
    
    # Iterate the specified features
    for f in features_to_plot:
        count = {} # Use dictionary to count
        for i, s in enumerate(dataset[f]):
            # Manually replace the number with the nominal string for a pretty graph
            if f in intToStr.keys():
                s = intToStr[f][s]

            if s in count.keys():
                count[s] += 1
            else:
                count[s] = 1

        values = np.array(list(count.values()))
        keys = list(count.keys())
        
        # Plot graph
        fig, ax = plt.subplots(figsize=(13, 4))
        sns.barplot(keys, values)
        plt.title(f)
        plt.ylabel("Number of samples")
        plt.show()
        
plotDist(dataset)


# ### Data normalization
# SVMs are guaranteed to converge at some point. However, when data is unscaled this can take some time. In this step, some of the features are rescaled to be within the range of [-1, 1]. In this step the ID column is also dropped from the dataset since it doesn't provide any relevant information for this task.

# In[ ]:


# Normalize columns 12 to 23. BILL_AMT1 to BILL_AMT6 and PAY_AMT1 to PAY_AMT6
norm_columns = ["LIMIT_BAL", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
                "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "AGE"]
for c in norm_columns:
    max_val = np.max(dataset[c])
    min_val = np.min(dataset[c])
    
    dataset[c] = (dataset[c] - min_val) * 2 / (max_val - min_val) - 1

dataset.drop(columns=["ID"], inplace=True)
print(dataset.head())


# ### Splitting the dataset for training and validation

# In[ ]:


def trainTestSplit(dataset, valid_per=0.1):
    n_samples = dataset.shape[0] # Total number of samples
    n_val = int(valid_per * n_samples)
    
    indices = np.arange(0, n_samples) # Generate a big array with all indices
    np.random.shuffle(indices) # Shuffle the array, numpy shuffles inplace
    
    # Perform the splits
    x_train = dataset.iloc[indices[n_val:], :-1].values # Last column is the feature we want to predict
    y_train = dataset.iloc[indices[n_val:], -1].values
    x_test = dataset.iloc[indices[:n_val], :-1].values
    y_test = dataset.iloc[indices[:n_val], -1].values
    
    return x_train, y_train, x_test, y_test


# ## Goal 1 -- Predicting next month payment
# 
# ### Assuming same class weightings.
# 
# Now, let's train an SVM classifier to predict wether the credit default will be paid next month. In this step we assume the same weight for both classes, assuming that the dataset is balanced.
# 
# We can see that the accuracy seems reasonably good, however just looking at this metric masks an important issue that is happening. If take a look at recall values for class 1 we can see that it is VERY low. This means that the classifier is taking an "easy" route and just tends to classify most of the samples as class 0 (the dominant class), while underperforming on the other class. 
# 
# This shows that for unbalanced datasets just looking at accuracy isn't enough. We can think of an extreme example were 99% of the samples are of class 0 and just 1% of class 1. In this scenario if the classifier always predicted class 0 for any sample it would achieve an accuracy of 99%.

# In[ ]:


print("Getting new dataset split...")
# Get a dataset split for training and validation
x_train, y_train, x_test, y_test = trainTestSplit(dataset, 0.2)

print(f"Training on {x_train.shape[0]} samples...")
print("\n#### Linear SVM Results ####")
# Create Linear SVM model
lsvm = LinearSVC(max_iter=32000) # If we don't specify anything it assumed all classes have same weight
lsvm.fit(x_train, y_train)
y_pred = lsvm.predict(x_test)
linear_acc = lsvm.score(x_test, y_test)
print(f"Linear SVM Acc: {linear_acc*100} % - Validated on {y_test.shape[0]} samples")
print(classification_report(y_test, y_pred))

print("\n#### Polynomial SVM with Degree 3 Results ####")
# Create Polynomial SVM
svm = SVC(gamma='scale', kernel='poly', degree=3)
svm.fit(x_train, y_train)
poly_acc = svm.score(x_test, y_test)
y_pred = svm.predict(x_test)
print(f"Polynomial SVM Acc: {poly_acc*100} %")
print(classification_report(y_test, y_pred))


# ### Assigning class weightings relative to sample number
# 
# By initializing the SVM classifiers with the parameter class_weights="balanced" each class is assigned a weight based on its number of samples. The actual calculation is done by (n_samples / (n_classes * np.bincount(y)). Let's repeat the test with the same data.
# 
# We can see that the accuracy took a hit, however the recall for class 1 improved greatly. Nonetheless, the precision for predicting class 1 got much worse. I'm not sure why this is the case.

# In[ ]:


print(f"Training on {x_train.shape[0]} samples...")
print("\n#### Linear SVM Results ####")
# Create Linear SVM model
lsvm = LinearSVC(max_iter=32000, class_weight="balanced") # Compute weight based on sample count per class
lsvm.fit(x_train, y_train)
y_pred = lsvm.predict(x_test)
linear_acc = lsvm.score(x_test, y_test)
print(f"Linear SVM Acc: {linear_acc*100} % - Validated on {y_test.shape[0]} samples")
print(classification_report(y_test, y_pred))

print("\n#### Polynomial SVM with Degree 3 Results ####")
# Create Polynomial SVM
svm = SVC(gamma='scale', kernel='poly', degree=3, 
          class_weight="balanced") # Compute weight based on sample count per class
svm.fit(x_train, y_train)
poly_acc = svm.score(x_test, y_test)
y_pred = svm.predict(x_test)
print(f"Polynomial SVM Acc: {poly_acc*100} %")
print(classification_report(y_test, y_pred))


# ### Balancing the training dataset and using same class weightings
# 
# A third scenario we can try here is actually balancing the training dataset and using the same class weightings. This will be done by randomly dropping samples of the dominant class until we have the same number of samples per class.
# 
# We can observe that there's not much difference from using the class weightings from the previous run. In fact the results are so similar that both ways might be equivalent and the small changes might be just due differences in the dropped samples.

# In[ ]:


def balanceTrainSet(x_train, y_train):
    samples_per_class = np.bincount(y_train) # Count samples per class
    dom_class = np.argmax(samples_per_class) # Max class index
    min_class = np.argmin(samples_per_class) # Min class index
    n_min = samples_per_class[min_class] # Number of samples in min class
    
    # Get indices for the dominant and the minor class
    dom_indices = np.where(y_train == dom_class)[0]
    min_indices = np.where(y_train == min_class)[0]
    np.random.shuffle(dom_indices) # Shuffle dom_indices
    # Contatenate both indices, using only the same number of indices from dom_indices as in min_indices
    indices = np.concatenate([min_indices, dom_indices[:n_min]], axis=0)
    np.random.shuffle(indices)
    
    # Build the new training set
    new_x_train = x_train[indices]
    new_y_train = y_train[indices]
    
    return new_x_train, new_y_train
    
bal_x_train, bal_y_train = balanceTrainSet(x_train, y_train)

print(f"Training on {bal_x_train.shape[0]} samples...")
print("\n#### Linear SVM Results ####")
# Create Linear SVM model
lsvm = LinearSVC(max_iter=32000) # If we don't specify anything it assumed all classes have same weight
lsvm.fit(bal_x_train, bal_y_train)
y_pred = lsvm.predict(x_test)
linear_acc = lsvm.score(x_test, y_test)
print(f"Linear SVM Acc: {linear_acc*100} % - Validated on {y_test.shape[0]} samples")
print(classification_report(y_test, y_pred))

print("\n#### Polynomial SVM with Degree 3 Results ####")
# Create Polynomial SVM
svm = SVC(gamma='scale', kernel='poly', degree=3)
svm.fit(bal_x_train, bal_y_train)
poly_acc = svm.score(x_test, y_test)
y_pred = svm.predict(x_test)
print(f"Polynomial SVM Acc: {poly_acc*100} %")
print(classification_report(y_test, y_pred))


# ### Performing 10 Rounds of Cross Validation
# Next, let's perform 10 rounds of cross validation using the SVM method with class weightings relative to the number of samples per class. After 10 rounds the average accuracy and the mean average micro f1-score is presented for both SVM models. Micro F1-score is used because the classes are imbalanced.
# 
# Observing the results we can conclude that the SVM model with polynomial kernel of degree 3 performs better than the Linear SVM model on this data.

# In[ ]:


N_ROUNDS = 10
l_reports = [] # Linear SVM reports
p_reports = [] # Polynomial SVM reports
l_accs = [] # Linear SVM accuracy history
p_accs = [] # Polynomial SVM accuracy history
for i in range(N_ROUNDS):
    print(f"### Round {i+1} ###")
    print("Getting new dataset split...")
    x_train, y_train, x_test, y_test = trainTestSplit(dataset, 0.2)    
    print(f"Training on {x_train.shape[0]} samples...")
    
    # Create a new Linear SVM model
    lsvm = LinearSVC(max_iter=32000, class_weight="balanced") # Compute weight based on sample count per class
    lsvm.fit(x_train, y_train)
    y_pred = lsvm.predict(x_test)
    linear_acc = lsvm.score(x_test, y_test)
    print(f"Linear SVM Acc: {linear_acc*100} % - Validated on {y_test.shape[0]} samples")
    report = classification_report(y_test, y_pred, output_dict=True)
    l_reports.append(report)
    l_accs.append(linear_acc)

    # Create Polynomial SVM
    svm = SVC(gamma='scale', kernel='poly', degree=3, 
              class_weight="balanced") # Compute weight based on sample count per class
    svm.fit(x_train, y_train)
    poly_acc = svm.score(x_test, y_test)
    y_pred = svm.predict(x_test)
    print(f"Polynomial SVM Acc: {poly_acc*100} %")
    report = classification_report(y_test, y_pred, output_dict=True)
    p_reports.append(report)
    p_accs.append(poly_acc)
    
print("### Finished ###")
print("Linear SVM Results:")
print(l_reports[0])
mean_acc = np.mean(l_accs)
mean_f1 = np.mean([r["weighted avg"]["f1-score"] for r in l_reports])
print(f"\tMean Acc: {mean_acc*100}% -- Mean Weighted Avg F1-Score: {mean_f1*100}%")

print("Polynomial SVM with Degree 3 Results:")
mean_acc = np.mean(p_accs)
mean_f1 = np.mean([r["weighted avg"]["f1-score"] for r in p_reports])
print(f"\tMean Acc: {mean_acc*100}% -- Mean Weighted Avg F1-Score: {mean_f1*100}%")


# ## Feature Selection
# Here, we remove the feature with the least impact on generalization accuracy one-by-one, until there's no remaining features to be removed. The goal is to find whether some features are irrelevant and how much they impact the accuracy of the model. This task will be performed on the same data throughout and using the polynomial of degree 3 model.
# 
# This reveals a very interesting pattern. Using even one feature we can achieve a very similar performance (even slightly better).

# In[ ]:


FEATURE_NAMES = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2",
                 "PAY_3", "PAY_4", "PAY_5", "PAY_6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
                 "BILL_AMT4", "BILL_AMT5", "BILLT_AMT6", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
                 "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
# Due to computing time constraints use a much smaller training dataset here
x_train, y_train, x_test, y_test = trainTestSplit(dataset, 0.85)

def trainModel(x_train, y_train, x_test, y_test):
    svm = SVC(gamma='scale', kernel='poly', degree=3, 
              class_weight="balanced")
    svm.fit(x_train, y_train)
    return svm.score(x_test, y_test)

print(f"Will train on {x_train.shape[0]} samples and validate on {x_test.shape[0]} samples.")
# Train a baseline model for this data, including all the features
baseline_acc = trainModel(x_train, y_train, x_test, y_test)
print(f"Baseline Acc: {baseline_acc*100}%")
print("====================================")

remaining_features = np.arange(0, x_train.shape[1])
for i in range(x_train.shape[1]-1):
    feat_names = [FEATURE_NAMES[r] for r in remaining_features]
    print(f"Remaining features: ", end=' ')
    [print(feat, end=', ') for feat in feat_names]
    print()
    
    best_acc = 0.0
    least_impact_feature = 0
    # Find feature with least impact on performance
    for c in range(remaining_features.shape[0]):
        # Test by removing each of the columns
        curr_features = np.delete(remaining_features, c)
        part_x_train = x_train[:, curr_features]
        part_x_test = x_test[:, curr_features]
        
        acc = trainModel(part_x_train, y_train, part_x_test, y_test)
        if acc > best_acc:
            best_acc = acc
            least_impact_feature = c

    print(f"Removing feature {FEATURE_NAMES[remaining_features[least_impact_feature]]} -- Had the least impact on performance (Acc: {best_acc*100} %)")
    remaining_features = np.delete(remaining_features, least_impact_feature)
    print("====================================")
    
print(f"Last feature is {FEATURE_NAMES[remaining_features[0]]}")
part_x_train = x_train[:, remaining_features]
part_x_test = x_test[:, remaining_features]
acc = trainModel(part_x_train, y_train, part_x_test, y_test)
print(f"Accuracy: {acc*100} %")


# ## Clustering customers with K-Means
# Here, K-Means is used to cluster the customers based on the features, without taking in account if the default was paid next month. Then, a single cluster is chosen to be considered as the class where the default was paid and the overall accuracy is calculated. The test is repeated for different cluster sizes.
# 
# We can see that using only 2 clusters yielded the best (although not good) result. This happens probably because as we use a larger number of clusters the data becomes more partitioned, i.e. there's less samples per cluster, so the accuracy takes a drop even though the probability is "better" in-cluster. An interesting extension of this is exhaustively searching through all combinations of clusters to find the one that yields the best overall accuracy.

# In[ ]:


# Get a dataset split for clustering and validation
x_train, y_train, x_test, y_test = trainTestSplit(dataset, 0.2)    

n_clusters = [2, 4, 8, 16, 32]
for n in n_clusters:
    print(f"K-Means with {n} clusters:")
    # Create K-Means model
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(x_train) # Fit to training data
    # Get clusters from validation data
    clustered = kmeans.predict(x_test)
    # For each cluster, find the probability defaulting (that the client paid it next month)
    # by comparing the number of samples for each class
    highest_prob = 0.0
    highest_c = 0
    overall_acc = 0.0
    for c in range(n):
        # Retrieve samples that belong to the current cluster
        indices = np.where(clustered == c)[0]
        samples_in_cluster = [y_test[s] for s in indices]
        # How many of those samples are of customers with default paid next month?
        proportion = np.bincount(samples_in_cluster)
        prob = proportion[1] / np.sum(proportion)
        print(f"[Cluster {c}] - Probability of paid credit in this cluster: {prob*100} %")
        if prob > highest_prob:
            highest_c = c
            highest_prob = prob
            overall_acc = proportion[1] / y_test.shape[0]
    
    print(f"Choosing the single cluster {highest_c} as default_paid yields an overall Accuracy of {overall_acc*100} %")
    
    print()

