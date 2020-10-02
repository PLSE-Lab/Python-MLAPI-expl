#!/usr/bin/env python
# coding: utf-8

# In this study, advanced machine learning methods will be utilized to build and test the performance of a selected algorithm for breast cancer diagnosis.
# 
# ## Dataset Description
# The Breast Cancer Wisconsin (Diagnostic) DataSet, obtained from Kaggle, contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass and describe	characteristics of the cell nuclei present in the image.
# 1. Number of instances: 569 
# 2. Number of attributes: 32 (ID, diagnosis, 30 real-valued input features)
# 3. Attribute information
#     * ID number
#     * Diagnosis (M = malignant, B = benign)
#     * Ten real-valued features are computed for each cell nucleus:
#         * radius (mean of distances from center to points on the perimeter)
#         * texture (standard deviation of gray-scale values)
#         * perimeter
#         * area
#         * smoothness (local variation in radius lengths)
#         * compactness (perimeter^2 / area - 1.0)
#         * concavity (severity of concave portions of the contour)
#         * concave points (number of concave portions of the contour)
#         * symmetry 
#         * fractal dimension ("coastline approximation" - 1)
#         
#     The mean, standard error, and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.
# 
# ## Data Loading and Cleaning

# In[ ]:


import pandas as pd
from IPython.display import HTML

df = pd.read_csv("../input/data.csv")
HTML(df.to_html())
print(df.shape)
print(df.dtypes)
df.head(n=3)


# The `id` and `Unnamed: 32` columns should be removed, since they are unnecessary. The dataset will be also examined for missing values, duplicated entries and unique values of 'diagnosis' column.

# In[ ]:


# Remove unnecessary columns
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)


# In[ ]:


# Find missing values
print('Missing values:\n{}'.format(df.isnull().sum()))

# Find duplicated records
print('\nNumber of duplicated records: {}'.format(df.duplicated().sum()))

# Find the unique values of 'diagnosis'.
print('\nUnique values of "diagnosis": {}'.format(df['diagnosis'].unique()))


# There are no missing values or duplicated records. Next the `diagnosis` distribution is checked.

# In[ ]:


total = df['diagnosis'].count()
malignant = df[df['diagnosis'] == "M"]['diagnosis'].count()
print("Malignant: ", malignant)
print("Benign: ", total - malignant)


# ## Data Exploration

# In[ ]:


# Generate statistics
df.describe()


# Plot pairwise relationships to check the correlations between the mean features.

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

g = sns.PairGrid(df.iloc[:, 0:11], hue="diagnosis", palette="Set2")
g = g.map_diag(plt.hist, edgecolor="w")
g = g.map_offdiag(plt.scatter, edgecolor="w", s=40)
plt.show()


# It seems that:
# 1. There are strong correlations between many variables. Next, a heatmap will be used to present the numerical correlations.
# 2. The univariate distributions on the diagonal show a separation of malignant and benign cells for several mean features. Malignant cells tend to have larger mean values of:
#     * radius
#     * perimeter
#     * area
#     * compactness
#     * concavity
#     * concave points

# In[ ]:


df_corr = df.iloc[:, 1:11].corr()
plt.figure(figsize=(8,8))
sns.heatmap(df_corr, cmap="Blues", annot=True)
plt.show()


# * As it was expected there are very strong correlations between radius, perimeter and area.
# * Compactness, concavity and and concave points are also highly correlated.
# 
# These highly correlated features result in redundant information. It is suggested to remove highly correlated features to avoid a predictive bias for the information contained in these features.

# ### Encode "diagnosis" to numerical values

# In[ ]:


df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})


# ## Machine Learning
# ### Split Data to Train/Test Sets
# Create train/test sets using the `train_test_split` function. The `test_size=0.3` inside the function indicates the percentage of the data that should be held over for testing.

# In[ ]:


from sklearn.model_selection import train_test_split

array = df.values

# Define the independent variables as features.
features = array[:,1:]

# Define the target (dependent) variable as labels.
labels = array[:,0]

# Create a train/test split using 30% test size.
features_train, features_test, labels_train, labels_test = train_test_split(features,
                                                                            labels,
                                                                            test_size=0.3,
                                                                            random_state=42)

# Check the split printing the shape of each set.
print(features_train.shape, labels_train.shape)
print(features_test.shape, labels_test.shape)


# ### K Nearest Neighbors (K-NN) Classifier
# K-NN was chosen amongst other algorithms (e.g. Support Vector Machines, Decision Trees and Naive Bayes), because it is quite fast and produces acceptable results. The speed of K-NN can be explained by the fact that this algorithm is a lazy learner and does not do much during training process unlike other classifiers that build the models. The performance of K-NN will be examined tuning the algorithm and applying various preprocessing steps.
# #### Evaluation of the algorithm
# Accuracy, i.e. the fraction of correct predictions is typically not enough information to evaluate a model. Although it is a starting point, it can lead to invalid decisions. Models with high accuracy may have inadequate precision or recall scores. For this reason the evaluation metrics that were also assessed are:
# * Precision or the ability of the classifier not to label as positive a sample that is negative. The best value is 1 and the worst value is 0. In our study case, precision is when the algorithm guesses that a cell is malignant and actually measures how certain we are that this cell is a true malignant. For example, a precision of 0.9 means that if the model predicts 100 malignant cells, the 90 of them are malignant and the rest 10 are benign (false).
# * Recall or the ability of the classifier to find all the positive samples. The best value is 1 and the worst value is 0. In context to the study, recall shows how well our identifier can find the malignant cells. For example, a low recall score of 0.8 indicates that our identifier finds only 80% of all the real malignant cells in the prediction. The rest 20% of real malignant cells will not be found by the diagnosis based on this algorithm, something that is unacceptable.
# * F1 score, a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal. The formula for the F1 score is: F1 = 2 x (precision x recall) / (precision + recall).

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from time import time

def print_ml_results():
    t0 = time()

    # Create classifier.
    clf = KNeighborsClassifier()

    # Fit the classifier on the training features and labels.
    t0 = time()
    clf.fit(features_train, labels_train)
    print("Training time:", round(time()-t0, 3), "s")

    # Make predictions.
    t1 = time()
    predictions = clf.predict(features_test)
    print("Prediction time:", round(time()-t1, 3), "s")

    # Evaluate the model.
    accuracy = clf.score(features_test, labels_test)
    report = classification_report(labels_test, predictions)

    # Print the reports.
    print("\nReport:\n")
    print("Accuracy: {}".format(accuracy))
    print("\n", report)
    print(confusion_matrix(labels_test, predictions))

print_ml_results()


# * The algorithm will be tuned to achieve an improved performance, especially a better recall score for the malignant class, since 90% can be considered a low recall score in this case.
# 
# #### Remove Highly Correlated Features and Run Again
# Investigate if removing manually features with a correlation higher than 0.8, can benefit the algorithm performance, although later this will be handled automatically by dimensionality reduction.

# In[ ]:


df_new = df[['diagnosis', 'radius_mean', 'texture_mean', 'smoothness_mean',
            'compactness_mean', 'symmetry_mean', 'fractal_dimension_mean',
            'radius_se', 'texture_se', 'smoothness_se',
            'compactness_se', 'concave points_se', 'symmetry_se',
            'fractal_dimension_se', 'concavity_worst', 'symmetry_worst',
            'fractal_dimension_worst']]

array = df_new.values

# Define the independent variables as features.
features_new = array[:,1:]

# Define the target (dependent) variable as labels.
labels_new = array[:,0]

# Create a train/test split using 30% test size.
features_train, features_test, labels_train, labels_test = train_test_split(features_new, labels_new,                                                                             test_size=0.3, random_state=42)

print_ml_results()


# There is a significant decrease in algorithm's accuracy and recall mostly for the malignant class. It's difficult to select manually the best features especially for datasets with many features correlated. Sometimes, ambiguity can occur when three or more variables are correlated. For example, if feature 1 is correlated with feature 2, while feature 2 is correlated with feature 3 but not feature 1, which one is better to remove? To resolve this automatically, dimensionality reduction methods are used such as Principal Component Analysis.

# ### Cross Validation
# Train/test split has a lurking danger if the split isn't random and when one subset of our data has only observations from one class, i.e. our data are imbalanced. This will result in overfitting. To avoid this, cross validation is applied. There are several cross validation methods such as K-Fold and Stratified K-Fold.
# 
# In K-Fold cross-validation, the original sample is randomly partitioned into k equal sized subsamples. Of the k subsamples, a single subsample is retained as the validation data for testing the model, and the remaining k-1 subsamples are used as training data. The cross-validation process is then repeated k times (the folds), with each of the k subsamples used exactly once as the validation data. The k results from the folds can then be averaged to produce a single estimation. The advantage of this method over repeated random sub-sampling is the increased accuracy because all observations are used for both training and validation, and each observation is used for validation exactly once.
# 
# If the original data comes in some sort of sorted shape, a shuffle of the order of the data points is necessary before splitting them up into folds. This can be done in `KFold()`, setting the `shuffle` parameter to `True`. If there are concerns about class imbalance, then the `StratifiedKFold()` class should be used instead. Where `KFold()` assigns points to folds without attention to output class, `StratifiedKFold()` assigns data points to folds so that each fold has approximately the same number of data points of each output class. This is most useful for when we have imbalanced numbers of data points in the outcome classes (e.g. one is rare compared to the others). For this class as well, it can be used `shuffle=True` to shuffle the data points' order before splitting into folds.

# ### Scale Features
# A common good practice in machine learning is feature scaling, normalization, standardization or binarization of the predictor variables. The main purposes of these methods are two:
# 1. Create comparable features in terms of units, e.g. if there are values in different units, then, the scaled data will be the same.
# 2. Create comparable features in terms of size, e.g. if two variables have vastly different ranges, the one with the larger range may dominate the predictive model, even though it may be less important to the target variable than the variable with the smaller range.
# 
# Feature scaling was applied here, since it is useful for algorithms that weigh inputs like regression and neural networks, as well as algorithms that use distance measures like K-NN.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
import numpy as np

np.set_printoptions(precision=2, suppress=True)

scaler = MinMaxScaler(feature_range=(0,1))
features_scaled = scaler.fit_transform(features)
print("Unscaled data\n", features_train)
print("\nScaled data\n", features_scaled)


# ### Principal Component Analysis (PCA)
# PCA is a preprocessing step, which decomposes a multivariate dataset in a set of successive orthogonal components that explain a maximum amount of the variance. It is used when we need to tackle datasets with a large number of features with different scales, some of which might be correlated. These correlations and the high dimension of the dataset bring a redudancy in the information. Applying PCA, the original features are transformed to linear combinations of new independent variables, which reduce the complexity of the dataset and thus, the computational cost.
# 
# Summarizing, the main purpose of principal component analysis is to:
# * identify hidden pattern in a data set,
# * reduce the dimensionnality of the data by removing the noise and redundancy in the data,
# * identify correlated variables

# In[ ]:


from sklearn.decomposition import PCA

pca = PCA(30)
projected = pca.fit_transform(features)
pca_inversed_data = pca.inverse_transform(np.eye(30))

plt.style.use('seaborn')

def plot_pca():
    plt.figure(figsize=(10, 4))
    plt.plot(pca_inversed_data.mean(axis=0), '--o', label = 'mean')
    plt.plot(np.square(pca_inversed_data.std(axis=0)), '--o', label = 'variance')
    plt.ylabel('Feature Contribution')
    plt.xlabel('Feature Index')
    plt.legend(loc='best')
    plt.xticks(np.arange(0, 30, 1.0))
    plt.show()

    plt.figure(figsize = (10, 4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), '--o')
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Explained Variance')
    plt.xticks(np.arange(0, 30, 1.0))
    plt.show()

plot_pca()


# Applying PCA on the unscaled dataset, it seems that more than 99% of the variance is explained by only one component, which is too good to be true. The feature contribution plot depicts that principal components 3 (`area_mean`) and 23 (`area_worst`) dominate the PCA. This is explained by the large variance of `area_mean` and `area_worst` (see std values of the Data Exploration section). To avoid this, feature scaling prior to PCA is highly recommended.

# In[ ]:


projected_scaled = pca.fit_transform(features_scaled)
pca_inversed_data = pca.inverse_transform(np.eye(30))

plot_pca()


# After applying scaling before PCA, 5 principal components are required to explain more than 90% of the variance. This shows a better handle on the variation within the dataset.
# 
# ### Univariate Feature Selection
# This preprocessing step is used to select the best features based on univariate statistical tests. Most common methods are:
# * `SelectKBest()`, which removes all but the k highest scoring features, and
# * `SelectPercentile()`, which removes all but a user-specified highest scoring percentage of features.
# 
# **Note:** First the dataset must be splitted into train and test sets, since performing feature selection on the whole dataset would lead to prediction bias.

# In[ ]:


from sklearn.feature_selection import SelectKBest

select = SelectKBest()
select.fit(features_train, labels_train)
scores = select.scores_
# Show the scores in a table
feature_scores = zip(df.columns.values.tolist(), scores)
ordered_feature_scores = sorted(feature_scores, key=lambda x: x[1], reverse=True)
for feature, score in ordered_feature_scores:
    print(feature, score)


# ### Tune the algorithm
# #### Putting it all together with GridSearchCV and Pipeline
# Algorithm tuning is a process in which we optimize the parameters that impact the model in order to enable the algorithm to perform with an improved performance. If we don't tune the algorithms well, performance will be poor with low accuracy, precision or recall. Most of the machine learning algorithms contain a set of parameters (hyperparameters), which should be set up adequately to perform the best. While all of the algorithms attempt to set reasonable default hyperparameters, they can often fail to provide optimal results for many real world datasets in practice. To find an optimized combination of hyperparameters, a metric is chosen to measure the algorithm's performance on an independent data set and hyperparameters that maximize this measure are adopted.
# 
# Tuning the models is a tedious, time-consuming process and there can sometimes be interactions between the choices we make in one step and the optimal value for a downstream step. Hopefully, there are two simple and easy tuning strategies, grid search and random search. Scikit-learn provides these two methods for algorithm parameter tuning. `GridSearchCV()` allows us to construct a grid of all the combinations of parameters passing one classifier to pipeline each time, tries each combination, and then reports back the best combination. So, instead of trying numerous values for each tuning parameter, `GridSearchCV()` will apply all the combinations of parameters - not just vary them independently - avoiding local optima.
# 
# The power of `GridSearchCV()` is that it multiplies out all the combinations of parameters and tries each one, making a k-fold cross-validated model for each combination. Then, we can ask for predictions and parameters from our `GridSearchCV()` object and it will automatically return to us the best set of predictions, as well as the best parameters.

# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV

# Create the scaler.
scaler = MinMaxScaler(feature_range=(0,1))

# Scale down all the features (both train and test dataset).
features = scaler.fit_transform(features)

# Create a train/test split using 30% test size.
features_train, features_test, labels_train, labels_test = train_test_split(features,
                                                                            labels,
                                                                            test_size=0.3,
                                                                            random_state=42)

# Create the classifier.
clf = KNeighborsClassifier()

# Create the pipeline.
pipeline = Pipeline([('reduce_dim', PCA()),
                     ('clf', clf)])

# Create the parameters.
n_features_options = [1, 3, 5, 7]
n_neighbors = [2, 4, 6]
algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']

parameters = [
    {
        'reduce_dim': [PCA(iterated_power=7)],
        'reduce_dim__n_components': n_features_options,
        'clf__n_neighbors': n_neighbors,
        'clf__algorithm': algorithm
    },
    {
        'reduce_dim': [SelectKBest()],
        'reduce_dim__k': n_features_options,
        'clf__n_neighbors': n_neighbors,
        'clf__algorithm': algorithm
    }]

# Create a function to find the best estimator.
def get_best_estimator(n_splits):

    t0 = time()

    # Create Stratified ShuffleSplit cross-validator.
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=3)

    # Create grid search.
    grid = GridSearchCV(pipeline, param_grid=parameters, scoring=('f1'), cv=sss, refit='f1')

    # Fit pipeline on features_train and labels_train.
    grid.fit(features_train, labels_train)

    # Make predictions.
    predictions = grid.predict(features_test)

    # Test predictions using sklearn.classification_report().
    report = classification_report(labels_test, predictions)

    # Find the best parameters and scores.
    best_parameters = grid.best_params_
    best_score = grid.best_score_

    # Print the reports.
    print("\nReport:\n")
    print(report)
    print("Best f1-score:")
    print(best_score)
    print("Best parameters:")
    print(best_parameters)
    print(confusion_matrix(labels_test, predictions))
    print("Time passed: ", round(time() - t0, 3), "s")
    
    return grid.best_estimator_

get_best_estimator(n_splits=20)


# #### Combine PCA and Feature Selection with FeatureUnion
# Often it is beneficial to combine several methods to obtain good performance. `FeatureUnion()` will be used to combine features obtained by PCA and univariate selection, `SelectKBest()`. Combining features using this transformer has the advantage that it allows cross validation and grid searches over the whole process. Datasets that benefit from this can often:
# * consist of heterogeneous data types (e.g. raster images and text captions),
# * are stored in a Pandas DataFrame and different columns require different processing pipelines.

# In[ ]:


# Build the estimator from PCA and univariate selection.
combined_features = FeatureUnion([('pca', PCA()), ('univ_select', SelectKBest())])

# Do grid search over k, n_components and K-NN parameters.
pipeline = Pipeline([('features', combined_features),
                     ('clf', clf)])

# Create the parameters.
n_features_options = [1, 3, 5, 7]
n_neighbors = [2, 4, 6]
algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']

parameters = [
    {
        'features__pca': [PCA(iterated_power=7)],
        'features__pca__n_components': n_features_options,
        'features__univ_select__k': n_features_options,
        'clf__n_neighbors': n_neighbors,
        'clf__algorithm': algorithm
    }]

get_best_estimator(20)


# ### Conclusions
# 
# In this study, K-NN algorithm was applied for the diagnosis of the Breast Cancer Wisconsin DataSet. It was found that precision and recall scores can be considerably improved applying the following steps:
# * Feature Scaling
# * Dimensionality Reduction
# * Cross Validation
# * Hyperparameter Optimization
# 
# For better results more data are required and other algorithms should be used.
# 
# ### References
# 1. http://scikit-learn.org/stable/index.html
# 2. https://jakevdp.github.io/PythonDataScienceHandbook/
# 3. http://jotterbach.github.io/2016/03/24/Principal_Component_Analysis/
# 3. https://medium.com/towards-data-science/train-test-split-and-cross-validation-in-python-80b61beca4b6
