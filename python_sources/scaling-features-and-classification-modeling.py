#!/usr/bin/env python
# coding: utf-8

# # Mobile Price Classification
# 
# 
# ### Meet Bob
# Bob has started his own mobile company. He wants to give tough fight to big companies like Apple,Samsung etc.
# 
# He does not know how to estimate price of mobiles his company creates. In this competitive mobile phone market you cannot simply assume things. To solve this problem he collects sales data of mobile phones of various companies.
# 
# Bob wants to find out some relation between features of a mobile phone(eg:- RAM,Internal Memory etc) and its selling price. But he is not so good at Machine Learning. So he needs your help to solve this problem.
# 
# > **Notice**: In this problem you do not have to predict actual price but a price range indicating how high the price is.
# 
# ---
# 
# ## Target
# In this Project,On the basis of the mobile Specification like Battery power, 3G enabled , wifi ,Bluetooth, Ram etc we are predicting Price range of the mobile.
# 
# ## About the Dataset
# - **battery_power**: Total energy a battery can store in one time measured in mAh
# - **blue**: Has bluetooth or not
# - **clock_speed**: Speed at which microprocessor executes instructions
# - **dual_sim**: Has dual sim support or not
# - **fc**: Front Camera mega pixels
# - **four_g**: Has 4G or not
# - **int_memory**: Internal Memory in Gigabytes
# - **m_dep**: Mobile Depth in cm
# - **mobile_wt**: Weight of mobile phone
# - **n_cores**: Number of cores of processor
# - **pc**: Primary Camera mega pixels
# - **px_height**: Pixel Resolution Height
# - **px_width**: Pixel Resolution Width
# - **ram**: Random Access Memory in Megabytes
# - **sc_h**: Screen Height of mobile in cm
# - **sc_w**: Screen Width of mobile in cm
# - **talk_time**: Longest time that a single battery charge will last when you are
# - **three_g**: Has 3G or not
# - **touch_screen**: Has touch screen or not
# - **wifi**: Has wifi or not

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data Preprocessing

# In[ ]:


data = pd.read_csv('../input/train.csv')


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


# Check if this data contains missing values
data.isnull().sum().max()


# In[ ]:


data['price_range'].value_counts()


# **To sum up**: The dataset is already well-balanced and does not contain any missing values.

# ## Exploring Data

# In[ ]:


data.describe().T


# In[ ]:


corr = data.corr()
plt.figure(figsize=(15,10))
sns.heatmap(corr, square=True, annot=True, annot_kws={'size':8})


# #### Low positive correlation: 
# - `pc` and `fc`
# - `three_g` and `four_g`
# - `px_width` and `px_height`
# - `sc_w` and `sc_height`
# 
# #### High positive correlation:
# - `ram` and `price_range`.
# 

# In[ ]:


sns.jointplot(data['ram'], data['price_range'],kind='kde')


# In[ ]:


sns.boxplot(data['price_range'], data['ram'])


# ## Preparing Data for Classification

# In[ ]:


X = data.drop(columns='price_range')
y = data['price_range']


# ### Feature Selection
# - Eliminate non-numerical
# - Eliminate non-ordinal
# - Eliminate features that have missing values
# - Eliminate colinearity (pc / fc, px_width / px_height, sc_h / sc_w)
# - Eliminate low variance features (since they do not carry much information)

# In[ ]:


X.var()


# In[ ]:


sns.distplot(X['m_dep'])


# In[ ]:


# Remove non-ordinal
X = X.drop(columns=['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi'])
# Remove colinearity
X = X.drop(columns=['fc', 'px_width', 'sc_w'])
# Remove low variance
X = X.drop(columns=['m_dep', 'clock_speed'])


# X = X[['ram']]
X.info()


# ### Train/Test Split

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# ### Scaling the Features
# Most of the times, your dataset will contain features highly varying in magnitudes, units and range. But since, most of the machine learning algorithms use Eucledian distance between two data points in their computations, this is a problem.
# 
# If left alone, these algorithms only take in the magnitude of features neglecting the units. The results would vary greatly between different units, 5kg and 5000gms. The features with high magnitudes will weigh in a lot more in the distance calculations than features with low magnitudes.
# 
# To supress this effect, we need to bring all features to the same level of magnitudes. This can be acheived by scaling.
# 
# Since both of KNN and SVM utitlize distance calculation behind the scene, we need to scale our features so that the large-valued features do not dominate the other features. More can be referenced [here](https://medium.com/greyatom/why-how-and-when-to-scale-your-features-4b30ab09db5e) and [here](http://sebastianraschka.com/Articles/2014_about_feature_scaling.html#about-standardization).
# 
# **Notice**: It is worth trying not scaling the features. So that we can see that for SVM,
# - The training time is very long (computationally expensive)
# - The model overfits
# 
# ### Multiple ways to scale
# - **Standard Scaler (Standardization)**: Scale the feature by shifting the mean back to 0 and variance to 1. By this way, we only shift the mean value to 0 and keep the distribution the same. Furthermore, this way can presereve the outliers in case they can contribute additional information to the problem.
# - **Min-max Scaler (Normalization)**: This one scales the range of values to between 0 and 1 and also eliminates the outliers.
# - Read more:
#     - Python machine Learning - SebastianRaschka
#     - [Medium - Why, How and When to scale your features ?](https://medium.com/greyatom/why-how-and-when-to-scale-your-features-4b30ab09db5e)
#     - [Quora](https://www.quora.com/When-should-you-perform-feature-scaling-and-mean-normalization-on-the-given-data-What-are-the-advantages-of-these-techniques)
#     - [Kaggle - Very good notebook](https://www.kaggle.com/rtatman/data-cleaning-challenge-scale-and-normalize-data?scriptVersionId=2945378)
#     - [statsStackExchange - Scaling ruins the result](https://stats.stackexchange.com/questions/172795/scaling-for-svm-destroys-my-results)
#     - [GeeksforGeeks - How and when to apply Scaling](https://www.geeksforgeeks.org/python-how-and-where-to-apply-feature-scaling/)
# 
# **Question here is:** 
# - Which one keeps the original distribution of the data ?
#     - Ans: Both
#     
# ### More about Standardization and Normalization
# - [StatisticsHowTo - Standardized Data](https://www.statisticshowto.datasciencecentral.com/standardized-values-examples/)
# - [StatisticsHowTo - Normalized Data](https://www.statisticshowto.datasciencecentral.com/normalized/)
# 
# ### Multiple ways to process the scaling
# - Fit the scaler on the whole dataset, scale the whole dataset and then train/test split
# - Train/test split and Fit the scaler (to obtain `mean` and `std`) to training set and then scale both training + testing set
# - Train/test split and Fit 2 different scalers on the training and testing set and scale them.
# - **The correct way is:** the 2nd answer.

# In[ ]:


from sklearn.preprocessing import StandardScaler

# Normalize Training Data 
scaler = StandardScaler().fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

#Converting numpy array to dataframe
X_train_std_df = pd.DataFrame(X_train_std, index=X_train.index, columns=X_train.columns)
X_test_std_df = pd.DataFrame(X_test_std, index=X_test.index, columns=X_test.columns) 


# In[ ]:


X_train_std_df.head()


# In[ ]:


train_std_data = pd.concat([X_train_std_df, y_train], axis=1)
train_std_data.var().sort_values()


# In[ ]:


plt.figure(figsize=(15,10))
sns.heatmap(train_std_data.corr(method='spearman'), annot=True)


# ## KNN

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()


# ### Tuning using K-fold Cross Validation
# 
# We can tune using only Train/Validation set, but there are still problems:
# - We use less training data
# - The model can potentially overfit to validation data, because:
#     - It is optimized based on validation data.
#     - Validation data can be only a small subset -> Cannot represent the population distribution.
#     
# **Solution**: A smart way is using `K-Fold CV`

# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


def plot_validation(param_grid, clf, X_train, y_train):
    val_error_rate = []

    for key in param_grid.keys():
        param_range = param_grid[key]
        for param in param_range:
            # https://stackoverflow.com/questions/337688/dynamic-keyword-arguments-in-python
            val_error = 1 - cross_val_score(clf.set_params(**{key: param}), X_train, y_train, cv=5).mean()
            val_error_rate.append(val_error)

        plt.figure(figsize=(15,7))
        plt.plot(param_range, val_error_rate, color='orange', linestyle='dashed', marker='o',
                 markerfacecolor='black', markersize=5, label='Validation Error')

        plt.xticks(np.arange(param_range.start, param_range.stop, param_range.step), rotation=60)
        plt.grid()
        plt.legend()
        plt.title('Validation Error vs. {}'.format(key))
        plt.xlabel(key)
        plt.ylabel('Validation Error')
        plt.show()
    

neighbors_range = range(1,200,5)
param_grid = {'n_neighbors': neighbors_range}
plot_validation(param_grid, knn, X_train_std_df, y_train)


# In[ ]:


best_k = 136

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_std_df, y_train)
1-knn.score(X_test_std_df, y_test)


# ### Overfitting
# - Seems like our model overfit slightly.
# - The most likely cause is the feature **ram**. Which has a high correlation with target variable and a high scale comparing to other features.
# 
# 
# 
# ### Additional refs for Overfitting
# - DataQuest
# - https://stackoverflow.com/questions/37776333/why-too-many-features-cause-over-fitting
# - https://elitedatascience.com/overfitting-in-machine-learning
# - https://elitedatascience.com/python-machine-learning-tutorial-scikit-learn
# - https://stats.stackexchange.com/questions/202318/how-much-is-too-much-overfitting
# - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.556.7571&rep=rep1&type=pdf
# - https://machinelearningcoban.com/2017/03/04/overfitting/
# - https://scikit-learn.org/stable/modules/cross_validation.html

# ## SVM

# In[ ]:


from sklearn.svm import SVC

svm = SVC(kernel='linear')


# **Recall**: The **C** parameter tells the SVM optimization how much you want to avoid misclassifying each training example. 
# - For large values of C, the optimization will choose a smaller-margin hyperplane if that hyperplane does a better job of getting all the training points classified correctly. 
# - Conversely, a very small value of C will cause the optimizer to look for a larger-margin separating hyperplane, even if that hyperplane misclassifies more points.
# 
# Thus for a very large value of C probably leads to overfitting of the model and for a very small value of C probably leads to underfitting. Thus the value of C must be chosen in such a way that it generalises the unseen data well.
# 

# In[ ]:


c_range =  range(1,200,20)
param_grid = {'C': c_range}
plot_validation(param_grid, svm, X_train_std_df, y_train)


# In[ ]:


best_c = 21
svm = SVC(kernel='linear',C=best_c)
svm.fit(X_train_std_df, y_train)
svm.score(X_test_std_df, y_test)


# ## Non-linear SVM
# 
# Technically, the **gamma** parameter is the inverse of the standard deviation of the RBF kernel (Gaussian function), which is used as similarity measure between two points.  Intuitive explaination can be found [here](https://chrisalbon.com/machine_learning/support_vector_machines/svc_parameters_using_rbf_kernel/).
# 
# With one hyperparameter, we can plot validation curve as above, but with more than one hyperparameter, we cannot. Therefore, we use GridSearchCV as a more proper and convinient way.

# ### GridSearchCV

# In[ ]:


# Using GridSearchCV to tune hyperparameters
from sklearn.model_selection import GridSearchCV

param_grid = {'C': c_range,
              'gamma': [.1, .5, .10, .25, .50, 1]}
gs = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
gs.fit(X_train_std_df,y_train)


# In[ ]:


print("The best hyperparameters {}.".format(gs.best_params_))
print("The Mean CV score of the best_estimator is {:.2f}.".format(gs.best_score_))


# In[ ]:


svm = SVC(kernel='rbf',C=1, gamma=0.1)
svm.fit(X_train_std_df, y_train)
svm.score(X_test_std_df, y_test)


# ## Inspecting Results

# ### KNN

# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_std_df, y_train)
pred = knn.predict(X_test_std_df)

print(knn.score(X_test_std_df,y_test))
print(classification_report(y_test,pred))

matrix=confusion_matrix(y_test,pred)
plt.figure(figsize = (10,7))
sns.heatmap(matrix,annot=True)


# ### SVM

# In[ ]:


svm = SVC(kernel='linear',C=best_c)
svm.fit(X_train_std_df, y_train)
pred = svm.predict(X_test_std_df)

print(svm.score(X_test_std_df,y_test))
print(classification_report(y_test,pred))

matrix=confusion_matrix(y_test,pred)
plt.figure(figsize = (10,7))
sns.heatmap(matrix,annot=True,fmt=".2f")


# ## Future works
# It is noticable that using only relevant features can significantly improve the performance of KNN and SVM. In other words, use all the features can worse the performance.
# 
# It's worth trying starting with `ram`, `battery_power`,`px_height`, and perform feature selection process. 
# 
# [Feature Selection techniques in Machine Learning - Towards Data Science](https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e)

# ### References
# - https://www.kaggle.com/azzion/svm-for-beginners-tutorial
# - https://www.kaggle.com/vikramb/mobile-price-prediction/notebook
# - https://www.kaggle.com/nirajvermafcb/support-vector-machine-detail-analysis
