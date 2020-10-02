#!/usr/bin/env python
# coding: utf-8

# # Big Data and Cloud Computing assignment 1
# 
# The features 
# before applying the feature selection algorithm are indicated as FDj , where Di is the ith-dataset (D1 or D2) where it is applied. The features after applying a feature selection algorithm to the 1st or the 2nd
# dataset are indicated as follows: FSi,Dj where Si the feature selection algorithm and Di the dataset where it is applied.
# 
# 
# - [A1 with FD1] vs. [A2 with FD1] 
# - [A1 with FD1] vs. [A1 with FS1,D1] 
# - [A1 with FD1] vs. [A1 with FS2,D1]
# - [A2 with FD1] vs. [A2 with FS1,D1]
# - [A2 with FD1] vs. [A2 with FS2,D1]
# 
# 

# ##  1. Import modules and data
# We combine all individual datasets into one, and further format the data so we can explore efficiently.

# In[ ]:


#Modules
import os
import time
import glob
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Grab all .dat files in input folder
path = '../input/' # use your path
all_files = glob.glob(os.path.join(path, "*.dat"))     # advisable to use os.path.join as this makes concatenation OS independent

# Remove batch 7 as there is an experimental error which caused mismatches in concentration levels. 
    # Refer to (https://www.sciencedirect.com/science/article/pii/S2352340915000050) for more info.

# Read each .dat file and concat all
df_from_each_file = (pd.read_csv(f, sep="\s+",index_col=0, header=None) for f in all_files)
df = pd.concat(df_from_each_file, sort=True)

# Seperate feature and value in each cell eg. 1:15596.16 --> 15596.16 
for col in df.columns.values:
    df[col] = df[col].apply(lambda x: float(str(x).split(':')[1]))

# Make Index(Gas type) a column and reset index to original
df = df.rename_axis('Gas').reset_index()

# Sort by Gas and reindex
df.sort_values(by=['Gas'],inplace=True)
df.reset_index(drop=True,inplace=True)


# ## 2. Exploratory data analysis
# We will check data for: 
# * Information on data types
# * Missing values
# * Correlations
# * Other observations (outliers etc.)
# 
# ### 2.1 Initial observartions

# In[ ]:


# Check total number of gases
df.Gas.nunique()


# In[ ]:


df.head()


# In[ ]:


# Check no values were lost in concatenation
df.shape


# We see 13910 total instances with data from 129 attributes. <br>
# One of the 129 attributes is our target that we want to predict. <br>
# Therefore there are 128 features that ay be used for feature engineering. <br>
# We can also see that we kept our data integrity after concatenation of all datasets.

# ### 2.2 Data types

# In[ ]:


pd.unique(df.dtypes),len(df.select_dtypes(exclude='object').columns) - 1


# We see only numerical data in the dataset (comprised of 64 bit float and int dtypes). <br> 
# Also we see 128 non-categorical columns which validated the above, with the following charecteristics: <br>

# In[ ]:


df.describe()


# ## 2.3 Exploring the numbers
# 

# ### 2.31 Gases

# Gases in the dataset by number:
# 
# 1. Ethanol
# 2. Ethylene
# 3. Ammonia
# 4. Acetaldehyde
# 5. Acetone 
# 6. Toluene

# In[ ]:


sns.countplot(df.Gas)
sns.set(style="darkgrid")
plt.title('Gas Count')
plt.show()


# In[ ]:


sns.distplot(df.Gas)
plt.xlim(1, 6)
plt.title('Distribution of Gas')
plt.show()


# ### 2.32 Concentrations

# In[ ]:


conc = df.iloc[:,1]


# In[ ]:


# Divide concentrations for readability in plot
conc_red = conc.apply(lambda x: x/10000)

fig = plt.figure(figsize=(22, 5))
fig.add_subplot(121)
sns.distplot(conc_red)
plt.title('Distribution of Concentrations')
plt.xlabel('Gas concentration Levels (x10000)')

fig.add_subplot(122)
sns.boxplot(conc_red)
plt.title('Concentration')
plt.xlabel('Gas concentration Levels (x10000)')

plt.show()


# #### Some observations on concentrations:
# 1. There is a positive skew in the concentration.
# 1. There are negative values

# In[ ]:


print("Skew of Gas concentration is: {}".format(conc.skew().round(decimals=2)))


# #### Handling skew
# The negative values in cocentration mean a log transformation of the concentration column is not possible, even when values are added to it. See below the graph after concentration is log transformed.

# In[ ]:


plt.figure(figsize=(22, 5))
sns.distplot(np.log(conc + 1 - min(conc)))
plt.title('Distribution of Log-transformed Concentration')
plt.xlabel('log(Concentration)')
plt.show()


# ### 2.33 Other attributes

# In[ ]:


attr = df.iloc[:,2:].copy()
attr.head()


# ### **Bivariate analysis - plotting gas concentration against attribute**

# In[ ]:


fig = plt.figure(figsize=(20,200))
for i in range(len(attr.columns)):
    fig.add_subplot(64,2,i+1)
    sns.scatterplot(attr.iloc[:,i],conc_red, hue="Gas", data=df, legend="full")
    plt.xlabel(attr.columns[i-1])
    plt.ylabel("Gas Concentration(x10000)")
    
fig.tight_layout()    
plt.show()


# ### **Correlation analysis**

# In[ ]:


correlation = df.corr()

f, ax = plt.subplots(figsize=(20,10))
plt.title('Correlations in dataset', size=20)
sns.heatmap(correlation)
plt.show()


# Despite the highly saturated graphic, we can clearly see which attributes have little to no correlation with any other attributes (notice dark red vertical and horizontal lines). <br><br>
# We can check which attributes have the highest correlation with the concentration level of the gas:

# In[ ]:


# Sort correletions of the concentration column
conc_corr = correlation.iloc[:,1].sort_values(ascending=False)

# Show all but with itself (correlation with self = 1)
conc_corr[1:].head(20)


# In[ ]:


# The bottom of the list 
conc_corr[1:].tail(20)


# ### **Bivariate analysis 2 - plotting gas concentration against only highly correlated attributes **

# In[ ]:


fig = plt.figure(figsize=(20,50))
for i in range(0,20):
    fig.add_subplot(10,2,i+1)
    sns.scatterplot(attr.iloc[:,conc_corr.index[i]],conc_red, hue="Gas", palette= "Set1", data=df, legend="full")
    plt.xlabel(conc_corr.index[i])
    plt.ylabel("Gas Concentration(x10000)")
    
fig.tight_layout()    
plt.show()


# 

# ## 3. Feature selection basic

# In[ ]:


# Make a copy of the data
df_copy = df.copy() 

# Assign features and target
X = df_copy.iloc[:,1:]
y = df_copy.iloc[:,0]


# In[ ]:


y.head()


# In[ ]:


X.head()


# ## 4. Principal Component Analysis

# ### 4.1 Scale data

# In[ ]:


from sklearn.preprocessing import StandardScaler
X_scaled = X.copy()
X_scaled = StandardScaler().fit(X_scaled).transform(X_scaled)


# ### 4.2 Eigenvectors and Eigenvalues

# #### 4.21 Covariance Matrix

# In[ ]:


cov_matrix = np.cov(X_scaled.T)


# #### 4.22 Eigenvector and eigenvalue calulations
# 

# In[ ]:


eig_val, eig_vec = np.linalg.eig(cov_matrix)
print('Eigenvectors \n%s' %eig_vec)
print('\nEigenvalues \n%s' %eig_val)


# In order to decide which eigenvectors can dropped without losing too much information for the construction of lower-dimensional subspace, we need to inspect the corresponding eigenvalues: The eigenvectors with the lowest eigenvalues bear the least information about the distribution of the data; those are the ones can be dropped.

# In[ ]:


# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])


# In[ ]:


tot = sum(eig_val)
var_exp = [(i / tot)*100 for i in sorted(eig_val, reverse=True)]


# The maximum variance of about 53%  can be explained by the first principal component.

# In[ ]:


plt.figure(figsize=(20, 4))
plt.bar(range(128), var_exp)
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.xlim(0,20)
plt.xticks(range(-1, 20))
plt.tight_layout()


# ### 4.3 Selecting Principal Components

# ### Scikit learn PCA

# In[ ]:


from sklearn.decomposition import PCA
pca = PCA()
X_scaled = pca.fit_transform(X_scaled)


# In[ ]:


plt.figure(figsize=(10, 4))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0,27)
plt.xticks(range(0,27))
plt.title('Cumulative variance of principle components')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.tight_layout()


# In[ ]:


print(pca.explained_variance_ratio_)


# We can see that the first 11 components consititute around 95% of the variance, therefore we can drop the remaining components. Literature varies when it comes to the maximum components to keep after PCA, some say keep each one until the variance ratio is 1, some say until 0.95.
# 
# Taking 0.95 as the threshhold, we can reduce the dimensionality of the input by a factor of 12  while compromising only 5% of potential information.

# In[ ]:


pca = PCA(n_components=2)
pca = pca.fit(X_scaled)
X_PCA = pca.transform(X_scaled)


# ##  4. Assessing Machine learning algorithms 1 and 2

# ### 4.1 Splitting data into training and test sets

# https://towardsdatascience.com/journey-to-the-center-of-multi-label-classification-384c40229bff <br>
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

# In[ ]:


from sklearn.preprocessing import label_binarize

# Binarize classes into one hot columns.
y_ohe = label_binarize(y, classes=[1,2,3,4,5,6])

# Store the amount of classes (We know there are 6 but this is good practice)
n_classes = y_ohe.shape[1]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=4)

y_train_nobinary = y_train.copy()
y_test_nobinary = y_test.copy()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = label_binarize(y_train, classes=[1,2,3,4,5,6])
y_test = label_binarize(y_test, classes=[1,2,3,4,5,6])


# ROC curve function

# In[ ]:


def plot_roc(y_test,y_pred,title):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes
    lw=2
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.3f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.3f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


# Permutation function

# In[ ]:


def permutation_test_between_clfs(y_test, pred_proba_1, pred_proba_2, nsamples=1000):
    auc_differences = []
    auc1 = roc_auc_score(y_test.ravel(), pred_proba_1.ravel())
    auc2 = roc_auc_score(y_test.ravel(), pred_proba_2.ravel())
    observed_difference = auc1 - auc2
    for _ in range(nsamples):
        mask = np.random.randint(2, size=len(pred_proba_1.ravel()))
        p1 = np.where(mask, pred_proba_1.ravel(), pred_proba_2.ravel())
        p2 = np.where(mask, pred_proba_2.ravel(), pred_proba_1.ravel())
        auc1 = roc_auc_score(y_test.ravel(), p1)
        auc2 = roc_auc_score(y_test.ravel(), p2)
        auc_differences.append(auc1 - auc2)
    return print("difference in roc curves: {0:.4f} \nprobability to observe a larger difference on a shuffled data set: {1}".format(observed_difference, np.mean(auc_differences >= observed_difference)))


# ### 4.2 Import and fit algorithm 1 - Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from scipy import interp
from itertools import cycle

start = time.time()

# Learn to predict each class against the other
classifier = OneVsRestClassifier(LogisticRegression(solver='sag',n_jobs=-1))
classifier.fit(X_train, y_train)
y_pred1 = classifier.predict_proba(X_test)

end = time.time()
print("\nTime taken: {:.2f} seconds".format(end-start))


# In[ ]:


confusion_matrix = metrics.confusion_matrix(y_test.argmax(axis=1),y_pred1.argmax(axis=1))
confusion_matrix


# In[ ]:


auc_roc = metrics.classification_report(np.argmax(y_test, axis=1),np.argmax(y_pred1, axis=1))
print('Logistic Regression Classification Report:\n {}'.format(auc_roc))


# In[ ]:


plot_roc(y_test,y_pred1,"ROC Logistic Regression")


# ### 4.3 Import and fit algorithm 2 - SVC

# In[ ]:


from sklearn.svm import SVC

start = time.time()

classifier = OneVsRestClassifier(SVC(kernel="linear",verbose=1, decision_function_shape='ovr', probability=True))
classifier.fit(X_train, y_train)
y_pred2 = classifier.predict_proba(X_test)

end = time.time()
print("\nTime taken: {:.2f} seconds".format(end-start))


# In[ ]:


confusion_matrix = metrics.confusion_matrix(y_test.argmax(axis=1),y_pred2.argmax(axis=1))
confusion_matrix


# In[ ]:


auc_roc = metrics.classification_report(np.argmax(y_test, axis=1),np.argmax(y_pred2, axis=1))
print('SVC Classification Report:\n {}'.format(auc_roc))


# In[ ]:


plot_roc(y_test,y_pred2,"ROC SVC")


# We will use the macro average method to evaluate the algorithm.

# # Log Reg vs SVC permutation test

# In[ ]:


permutation_test_between_clfs(y_test, y_pred1, y_pred2, nsamples=1000)


# ### 4.4 Apply feature selection technique to algorithm 1 - Logistic regression with Recursive Feature Elimination

# In[ ]:


from sklearn.feature_selection import RFE

start = time.time()

classifier = OneVsRestClassifier(LogisticRegression(solver='sag',n_jobs=-1))
rfe = RFE(classifier, n_features_to_select=64,verbose=1,step=1)
rfe = rfe.fit(X_train, y_train_nobinary)

end = time.time()
print("\nTime taken: {:.2f} seconds".format(end-start))


# In[ ]:


# List of best features ranked by RFE algorithm
features = X.columns[rfe.support_]
print(features)
X_train_rfe = pd.DataFrame(X_train)[features]
X_test_rfe = pd.DataFrame(X_test)[features]


# In[ ]:


classifier = OneVsRestClassifier(LogisticRegression(solver='sag',n_jobs=-1))
classifier.fit(X_train_rfe, y_train)
y_pred11 = classifier.predict_proba(X_test_rfe)


# In[ ]:


confusion_matrix = metrics.confusion_matrix(y_test.argmax(axis=1),y_pred11.argmax(axis=1))
confusion_matrix


# In[ ]:


auc_roc = metrics.classification_report(np.argmax(y_test, axis=1),np.argmax(y_pred11, axis=1))
print('Logistic regression with Recursive Feature Elimination:\n {}'.format(auc_roc))


# In[ ]:


plot_roc(y_test,y_pred11,'ROC for Logistic regression with Recursive Feature Elimination')


# # Log Reg vs Log Reg w/RFE permutation test

# In[ ]:


permutation_test_between_clfs(y_test, y_pred1, y_pred11, nsamples=1000)


# ### 4.5 Apply feature selection technique to algorithm 1 - Logistic regression with Chi Square test

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2 ,SelectKBest
norm = MinMaxScaler()
# Normalise training data 
X_train_norm = norm.fit_transform(X_train)


# In[ ]:


selector = SelectKBest(chi2, k=64)
selector.fit(X_train_norm, y_train)
X_train_kbest = selector.transform(X_train)
X_test_kbest = selector.transform(X_test)


# In[ ]:


classifier = OneVsRestClassifier(LogisticRegression(solver='sag',n_jobs=-1))
classifier.fit(X_train_kbest, y_train)
y_pred12 = classifier.predict_proba(X_test_kbest)


# In[ ]:


confusion_matrix = metrics.confusion_matrix(y_test.argmax(axis=1),y_pred12.argmax(axis=1))
confusion_matrix


# In[ ]:


auc_roc = metrics.classification_report(np.argmax(y_test, axis=1),np.argmax(y_pred12, axis=1))
print('Logistic regression with chi2 test feature selection:\n {}'.format(auc_roc))


# In[ ]:


plot_roc(y_test,y_pred12,'ROC Logistic regression with chi2 test')


# # Log Reg vs Log Reg w/chi2 permutation test

# In[ ]:


permutation_test_between_clfs(y_test, y_pred1, y_pred12, nsamples=1000)


# ### 4.5 Apply feature selection technique to algorithm 2 - SVC with Recursive Feature Elimination

# In[ ]:


classifier = OneVsRestClassifier(SVC(kernel="linear", decision_function_shape='ovr'))
rfe = RFE(classifier, n_features_to_select=64,verbose=1,step=1)
rfe = rfe.fit(X_train, y_train_nobinary)


# In[ ]:


# List of best features ranked by RFE algorithm
features = pd.DataFrame(X_train).columns[rfe.support_]
print(features)
X_train_rfe = pd.DataFrame(X_train)[features]
X_test_rfe = pd.DataFrame(X_test)[features]


# In[ ]:


classifier = OneVsRestClassifier(SVC(kernel="linear",probability=True, verbose=1, decision_function_shape='ovr'))
classifier.fit(X_train_rfe, y_train)
y_pred21 = classifier.predict_proba(X_test_rfe)


# In[ ]:


confusion_matrix = metrics.confusion_matrix(np.argmax(y_test, axis=1),np.argmax(y_pred21, axis=1))
confusion_matrix


# In[ ]:


auc_roc = metrics.classification_report(np.argmax(y_test, axis=1),np.argmax(y_pred21, axis=1))
print('SVC with Recursive Feature Elimination:\n {}'.format(auc_roc))


# In[ ]:


plot_roc(y_test,y_pred21,'ROC for SVC with Recursive Feature Elimination')


# # SVC vs SVC w/RFE permutation test

# In[ ]:


permutation_test_between_clfs(y_test, y_pred2, y_pred21, nsamples=1000)


# ### 4.5 Apply feature selection technique to algorithm 2 - SVC with chi2 test

# In[ ]:


classifier = OneVsRestClassifier(SVC(kernel="linear",probability=True , verbose=1, decision_function_shape='ovr'))
classifier.fit(X_train_kbest, y_train)
y_pred22 = classifier.predict_proba(X_test_kbest)


# In[ ]:


confusion_matrix = metrics.confusion_matrix(np.argmax(y_test, axis=1),np.argmax(y_pred22, axis=1))
confusion_matrix


# In[ ]:


auc_roc = metrics.classification_report(np.argmax(y_test, axis=1),np.argmax(y_pred22, axis=1))
print('SVC with chi2 test feature selection:\n {}'.format(auc_roc))


# In[ ]:


plot_roc(y_test,y_pred22,'ROC for SVC with feature selection based on chi2 test')


# # SVC vs SVC w/chi2 permutation test

# In[ ]:


permutation_test_between_clfs(y_test, y_pred2, y_pred22, nsamples=1000)

