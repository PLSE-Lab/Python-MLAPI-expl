#!/usr/bin/env python
# coding: utf-8

# # Intuition on Model Evaluation
# We all have been there, we made changes to our model and now we would like to know if we actually improved upon our previous model. How do we approach this problem?
# 
# In this notebook we do a qualitative and quantitative exploration of effects that distort model evaluation with the goal to acquire a better intuition on that problem. The main focus here lies on sample bias and overfitting. Hopefully we are going to be less confused in the future about discrepancies between CV and LB scores :D.

# ## The Data and the Models
# For our analysis we will make use of a generated toy dataset which has several advantages:
# 1. It is simple and intuitive to understand.
# 2. We know it's underlying true distribution.
# 3. We can freely modify it's size.
# 
# With that in mind we are going to utilize a **balanced binary classification** dataset where all samples are generated from a **2D normal distribution** irrespective of their class. Because both classes are balanced and have the same distribution, **any model** is expected to achieve **50% accuracy** on unseen data. Below we see one such generated dataset with 1000 samples. We will refer to our target as *y* which can either be 1 (positive) or 0 (negative).
# 
# As our classification models we will be using a shallow random forest, logistic regression and SVM with high regularization. All of these are quick to train and match the low complexity of our toy dataset.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, log_loss
from time import clock_gettime_ns

np.random.seed(77)

modelTypes = ["randomForest", "logisticRegression", "SVM"]

def sample_data(size = 1000, stratified=True):
    X = np.random.multivariate_normal(mean = np.zeros(2), cov = np.diag(np.ones(2)), size = size)
    y = np.random.randint(0,2, size=size)
    if stratified:
        y = np.concatenate([np.zeros(size//2), np.ones(size - size//2)]).astype(np.int64)
    return X, y

def sample_model(modelType):
    model_dict = {"randomForest": RandomForestClassifier(n_estimators=10, max_depth=5),
                 "logisticRegression": LogisticRegression(solver="liblinear"),
                 "SVM": svm.SVC(gamma="scale", C=0.1)}
    return model_dict[modelType]
    
def run_train_validation_split(X,y, model, stratified = True, ratio_test = 0.2):
    if stratified:
        stratify = y
    else:
        stratify = None
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=stratify, test_size=ratio_test)
    model.fit(X_train, y_train)
    y_val_predicted = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_predicted)
    return val_acc

def run_KFold_CV(X,y, model, num_folds = 5, stratified = True):    
    folds = KFold(n_splits=num_folds, shuffle=True)
    if stratified:
        folds = StratifiedKFold(n_splits = num_folds, shuffle=True)
        
    y_predicted = np.zeros_like(y)
    for ind_train, ind_val in folds.split(X,y):
        X_train = X[ind_train]
        y_train = y[ind_train]
        X_val = X[ind_val]
        model.fit(X_train, y_train)
        y_predicted[ind_val] = model.predict(X_val)
    val_acc = accuracy_score(y, y_predicted)
    return val_acc

def plot_data(data):
    X, y = data
    df = pd.DataFrame(data=X, columns=["x1","x2"]) 
    df["y"] = y
    sns.scatterplot(data=df, x="x1", y="x2", hue="y")


# In[ ]:


# Generating and plotting the distribution of our toy dataset.
X, y = sample_data(stratified=False, size=1000)
plt.figure(figsize=(7,7))
plot_data((X,y))
plt.show()


# In[ ]:


print("Ratio of positive samples: {}".format(y.sum() / y.shape[0]))


# ## Sample Bias
# At this point we already encounter a first source of bias, namely sample bias. Even though we generate our dataset from a balanced distribution, we can observe a slight majority of positives. Intuitively we can think of the number of positive samples **k** as a sample from a binomial distribution with **N** being the size of the dataset and **p=0.5**. Therefore the deviation of the observed and true ratio of positive samples scales with **1/sqrt(N)**, meaning that increasingly large datasets diminish the effects of sample bias.
# 
# Suppose we have two models, one that only predicts positive and one that only predicts negative. Our biased dataset would favor the former even though both models perform equally on unseen data. Even without overfitting **sample bias misguides us in our evaluation of relative model strength.** 
# 
# Unfortunately sample bias is not just limited to the distribution of the classes but also present in the distribution of the features. It is also present in all conceivable interactions between features and target and therefore impossible to avoid. To showcase the effect of these interactions we sample multiple **stratified datasets** (without bias in the target) and for each dataset we train 1000 models. Each model is trained and validated based on a different stratified K-fold CV split in order to eliminate biases resulting from specific CV splits. We finally plot the validation accuracy distribution of our 1000 models for each generated dataset. **Differences between these distributions are attributed to sample bias of the individual datasets.**
# 
# We also observer that model performance depends on the dataset. For dataset 4, random forest is best performing whereas for dataset 2, it is logistic regression. Once again sample bias misguides us in relative model strength. In the context of kaggle competitions, sample bias is a big contributor to differences between CV, public LB and private LB. Intuitively we can expect the differences to decrease with increasing dataset size.

# In[ ]:


# Plotting the distribution of positive samples based on the dataset size.
plt.figure(figsize=(16,16))
plt.subplot(2,2,1)
for size in [200, 1000, 5000]:
    dist_positives = np.zeros(10000)
    for i in range(10000):
        _, y = sample_data(size=size, stratified=False)
        dist_positives[i] = y.sum() / size
    sns.kdeplot(dist_positives, label="size={}".format(size))
plt.legend()
plt.title("Distribution of positives based on dataset size")

# Plotting the distribution of validation accuracies for different stratified datasets and different model types.
num_datasets = 5
num_runs = 1000

datasets = [sample_data(size = 1000, stratified=True) for i in range(num_datasets)]
for i,modelType in enumerate(modelTypes):
    plt.subplot(2,2,i+2)
    val_accs_kfold = np.zeros(num_runs)
    for dataset_index in range(num_datasets):    
        X, y = datasets[dataset_index]
        for run in range(num_runs):
            model = sample_model(modelType)
            val_accs_kfold[run] = run_KFold_CV(X, y, model, num_folds=5, stratified=True)
        sns.kdeplot(val_accs_kfold, label="dataset {}".format(dataset_index))
    plt.title("{} validation accuracy distribution for each stratified dataset".format(modelType))

plt.show()


# To get a feeling for the kind of datasets a model performs better or worse we sample 100 small datasets with just 20 samples each. For every dataset we train multiple models with different K-fold CV splits and average out the validation accuracies. We plot the best and worst dataset according to the average validation accuracy. Results are presented separately for our used model types. 
# 
# We can see that there can be huge differences in the validation accuracies ranging from 0.25 up to 0.78. The best performing datasets show a strong separation between the classes whereas the worst datasets show strong homogeneity. These properties in data happen by chance and can not be avoided.

# In[ ]:


# Plotting the worst and best dataset (w.r.t validation accuracy) for each model type.
num_datasets = 100
num_runs = 10

plt.figure(figsize=(16,24))

datasets = [sample_data(size=20, stratified=True) for i in range(num_datasets)]

for i,modelType in enumerate(modelTypes):
    val_accs_kfold = np.zeros(num_datasets)
    for dataset_index in range(num_datasets):
        X, y = datasets[dataset_index]
        for run in range(num_runs):
            model = sample_model(modelType)
            val_accs_kfold[dataset_index] += run_KFold_CV(X, y, model, num_folds=5, stratified=True) / num_runs
    best_dataset_index = np.argmax(val_accs_kfold)
    plt.subplot(3,2,2*i+1)
    plot_data(datasets[best_dataset_index])
    plt.title("best dataset for {}, average validation acc: {:.2f}".format(modelType, val_accs_kfold[best_dataset_index]))
    
    worst_dataset_index = np.argmin(val_accs_kfold)
    plt.subplot(3,2,2*i+2)
    plot_data(datasets[worst_dataset_index])    
    plt.title("worst dataset for {}, average validation acc: {:.2f}".format(modelType, val_accs_kfold[worst_dataset_index]))

plt.show()


# ## Overfitting
# We know that the performance of a model on the training set is not representative of it's performance on unseen data. Because of that we usually evaluate models on a hold out set of our data in order to compare their relative strength. The most common approaches here are a train / validation split of the data or K-fold CV. At this point we should remind ourselves that those validation schemes are **noisy estimates** of the strength of a model, which can misguide us during model selection.  
# 
# In the following we are comparing a **stratified train / val split with 0.2 val ratio** and a **stratified 5 fold CV** and quantify their variance. To that end we generate a stratified dataset with 1000 samples and train models on different split seeds and plot the distribution of validation accuracies. We know for a fact that every such trained  model is equally strong given the properties of our toy dataset. Any variance in the accuracy distribution can therefore be attributed to the inaccuracy of our validation procedure (shifts in mean are a result of sample bias).

# In[ ]:


# Plotting validation accuracy distribution for K-fold CV and train/val split. 
num_runs = 1000

val_accs_train_val = np.zeros(num_runs)
val_accs_kfold = np.zeros(num_runs)

X, y = sample_data(size = 1000, stratified=True)

plt.figure(figsize=(15,10))

for modelType in modelTypes:
    for run in range(num_runs):
        model = sample_model(modelType=modelType)
        val_accs_train_val[run] = run_train_validation_split(X, y, model, ratio_test=0.2, stratified=True)
        val_accs_kfold[run] = run_KFold_CV(X, y, model, num_folds=5, stratified=True)


    sns.kdeplot(val_accs_train_val, label = "{} with train val split, 0.2 val ratio".format(modelType))
    sns.kdeplot(val_accs_kfold, label = "{} with 5-fold CV".format(modelType))
plt.title("Distribution of validation accuracy for different validation methods and model.")
plt.legend()


# We observe that K-fold CV has a much smaller variance in estimating model strength compared to a train test split. This makes intuitive sense because for the train / val split we only use 20% of our available data for validation whereas in K-fold CV we use 100%. Similar to the previously made argument we expect the standard deviation of our model strength estimates to decrease with **1/sqrt(N)**. We can also expect the variance in validation accuracy to decrease with increasing dataset size.
# 
# We also observe that the best respective models achieve accuracies way over 50%. To put this into context, when we finetune models and optimize features, we also go through to a high number of models and usually pick the best according to the validation results. Similar to a how a model overfits it's **parameters** to the **train data**, we are overfitting the **hyperparameters** to the **validation data** and overestimating our model strength. To draw the analogy even further, consider this: Just like complex models are more prone to overfitting during training we find that long hyperparameter searches are more prone to overfitting during validation.  
# 
# This highlights the importance of having a separate **test set** which we should only use very sparely to evaluate or model strength. We should be aware that **measuring a model against a dataset always comes with a cost!** 

# There is a lot that can go wrong during model evaluation and we should be aware of that. We observed that **sample bias** and **hyperparameter overfitting** to the validation set are two big distorting effects during model evaluation. We also observed that these distorting effects diminish with increasing dataset size. These ideas are pretty straigth forward, however we should always keep them in mind in order to avoid unpleasant surprises in data science projects or competitions :D.
# 
