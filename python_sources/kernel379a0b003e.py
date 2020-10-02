#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels.api as sm
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
# To ignore warnings
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


pc = pd.read_csv("C://Users/pchadha/Boosting_Kaggle_Practice/Prudential_Life_insurance/train.csv")


# In[ ]:


pc_test = pd.read_csv("C://Users/pchadha/Boosting_Kaggle_Practice/Prudential_Life_insurance/test.csv")


# In[ ]:


pd.set_option('display.max_row', 1000)
pd.set_option('display.max_columns', 300)
pd.set_option('display.max_colwidth',5000)


# In[ ]:


pc.head()


# In[ ]:


pc.Response.value_counts()


# In[ ]:


# The 'Response' shows that potential customers have been classified into 8 categories
# Most have been classified as level '8', followed by '6' and '7'


# Will first treat the columns (if any) with missing values 

# In[ ]:


pc.info(verbose = True)


# In[ ]:


# It can be seen that there are 18 float type, 109 int type and 1 object type variables in the dataset. 
#Total columns are 128, where 'Id' and 'Response' will not be part of model learning as they are customer 'id' and 'Target' 
# values respectively 


# In[ ]:


cols4 = pc.select_dtypes(include=['int64']).columns.values
cols5 = pc.select_dtypes(include = ['float64']).columns.values


# In[ ]:


round((pc.isnull().sum()/len(pc.index))*100,2)


# In[ ]:


# As we have lot of columns, will remove columns with above 50% missing values
col = []
for i in pc.columns:
    if round((pc[i].isnull().sum()/len(pc.index))*100,2) >= 50:
        col.append(i)
print(col)
print(len(col))
    


# In[ ]:


# Removing these columns
pc = pc.drop(col, axis = 1)


# In[ ]:


round((pc.isnull().sum()/len(pc.index))*100,2)


# In[ ]:


# Focussing only on columns that have missing values
col = []
for i in pc.columns:
    if round((pc[i].isnull().sum()/len(pc.index))*100,2) != 0:
        col.append(i)
print(col)
print(len(col))


# In[ ]:


# Analysing 'Employment_Info_1' feature
pc["Employment_Info_1"].describe()


# In[ ]:


pc["Employment_Info_1"].value_counts()


# In[ ]:


pc["Employment_Info_1"].max()


# In[ ]:


# These are normalized values related employment history and since there's insignificant number of missing rows in this case,
# will remove the missing rows rather than imputing them
pc = pc[~pd.isnull(pc["Employment_Info_1"])]


# In[ ]:


round((pc[col].isnull().sum()/len(pc.index))*100,2)


# In[ ]:


pc["Employment_Info_4"].describe()


# In[ ]:


pc["Employment_Info_4"].value_counts()


# In[ ]:


# We can see that '0' value dominates the distribution of values within this feature. Therefore, imputing value '0' 
# for missing values in this case
pc.loc[pd.isnull(pc["Employment_Info_4"]),"Employment_Info_4"] = 0 


# In[ ]:


round((pc[col].isnull().sum()/len(pc.index))*100,2)


# In[ ]:


pc.index = pd.RangeIndex(1, len(pc.index) + 1)


# In[ ]:


pc["Employment_Info_6"].describe()


# In[ ]:


pc["Employment_Info_6"].value_counts()


# This feature has no dominant value so will use iterative imputer to fill the null values here
# 

# In[ ]:


pc["Insurance_History_5"].describe()


# In[ ]:


pc["Insurance_History_5"].value_counts()


# Again, this feature too is normalized and has no dominant value. Therefore, here too will impute with use of Iterative imputer

# In[ ]:


pc["Family_Hist_2"].describe()


# In[ ]:


pc["Family_Hist_2"].value_counts()


# Again for similar reasons, will impute using Iterative Imputer

# In[ ]:


pc["Family_Hist_4"].describe()


# In[ ]:


pc["Family_Hist_4"].value_counts()


# Again for similar reasons, will impute using Iterative Imputer

# In[ ]:


pc["Medical_History_1"].describe()


# In[ ]:


pc["Medical_History_1"].value_counts()


# Again, no one particular dominant value and even 'mean' is not suitable here as there's significant diference between the median and the mean. Will impute using Iterative Imputer function here as well

# In[ ]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# In[ ]:


round((pc[col].isnull().sum()/len(pc.index))*100,2)


# In[ ]:


#Before imputing the values, will first analyze the 'object' feature 
pc["Product_Info_2"].value_counts()


# It can be seen that this is a categorical value that can be converted to numeric type using encoder. Will use 'labelencoder' to avoid creating multiple features using 'Dummy variables' as we already have lot of features.

# In[ ]:


lc = LabelEncoder()


# In[ ]:


pc["Product_Info_2"] = lc.fit_transform(pc["Product_Info_2"])


# In[ ]:


pc["Product_Info_2"].describe()


# In[ ]:


pc["Product_Info_2"].value_counts()


# In[ ]:


# Imputing values using Iterative Imputer
cols2 = pc.columns


# In[ ]:


pc_imp = pd.DataFrame(IterativeImputer().fit_transform(pc))


# In[ ]:


pc_imp.columns = cols2


# In[ ]:


pc_imp.head()


# In[ ]:


round((pc_imp[col].isnull().sum()/len(pc_imp.index))*100,2)


# It can be seen that all null values have been treated. However, side-effect of Iterative computer is that it converts all columns to float type while processing them. Will now convert the columns originally integer type to 'int' 

# In[ ]:


pc_imp[cols4] = pc_imp[cols4].astype(int)


# In[ ]:


pc_imp.info(verbose = True)


# In[ ]:


pc_imp["Product_Info_2"] = pc_imp["Product_Info_2"].astype(int)


# Visualizing features and their relations using univariate and bivariate analysis

# In[ ]:


# Removing 'Id' variable as it will not be used for model learning
pc_imp = pc_imp.drop('Id', axis =1)


# In[ ]:


pc_imp.head()


# Analysing few continuous features first
# Continuous variables are as follows:
# Product_Info_4, Ins_Age, Ht, Wt, BMI, Employment_Info_1, Employment_Info_4, Employment_Info_6, Insurance_History_5, Family_Hist_2, Family_Hist_3, Family_Hist_4, Family_Hist_5

# In[ ]:


# Box plots for outlier analysis
col = ["Product_Info_4", "Ins_Age", "Ht", "Wt", "BMI"]
coln = ["Employment_Info_1", "Employment_Info_4", "Employment_Info_6", "Insurance_History_5"]
col2 = ["Family_Hist_2", "Family_Hist_4"]


# In[ ]:


sns.boxplot(data = pc_imp[col], orient = 'v')


# In[ ]:


# We can see that there are few outliers in the case of features: 'Ht', 'Wt' and 'BMI'


# In[ ]:


plt.figure(figsize = (20,12))
sns.boxplot(data = pc_imp[coln], orient = 'v')


# In the above case, all features have outliers. Now all these variables are normalized so scaling is not required. 
# However, will remove outliers using IQR for variables 'BMI', 'Employment_Info_6' to avoid data getting skewed for these variables 

# In[ ]:


# Outlier removal for 'BMI' and 'Employment_Info_6'
Q1= pc_imp['BMI'].quantile(0.5)
Q3= pc_imp['BMI'].quantile(0.95)
Range=Q3-Q1
print(Range)
pc_imp= pc_imp[(pc_imp['BMI'] >= Q1-1.5*Range) & (pc_imp['BMI'] <= Q3+1.5*Range) ]


# In[ ]:


Q1= pc_imp['Employment_Info_6'].quantile(0.5)
Q3= pc_imp['Employment_Info_6'].quantile(0.95)
Range=Q3-Q1
pc_imp= pc_imp[(pc_imp['Employment_Info_6'] >= Q1-1.5*Range) & (pc_imp['Employment_Info_6'] <= Q3+1.5*Range) ]


# In[ ]:


plt.figure(figsize = (10,10))
sns.boxplot(data = pc_imp[col2], orient = 'v')


# Few outliers in the case of Family_Hist_2 and Family_Hist_4 features observed

# In[ ]:


# pairplot analysis to understand correlation between continous variables
colc = ["Product_Info_4", "Ins_Age", "Ht", "Wt", "BMI", "Employment_Info_1", "Employment_Info_4", "Employment_Info_6", "Insurance_History_5", "Family_Hist_2","Family_Hist_4"]


# In[ ]:


sns.pairplot(pc_imp[colc])


# From the pair plots, we can see that few variables such as "Family_Hist_2","Ins_Age" and "Family_Hist_4", "BMI", "Wt" and "Ht" are correlated to each other. 
# Will now plot heatmap to understand correlation in details between these variables

# In[ ]:


colc = ["Product_Info_4", "Ins_Age", "Ht", "Wt", "BMI", "Employment_Info_1", "Employment_Info_4", "Employment_Info_6", "Insurance_History_5", "Family_Hist_2","Family_Hist_4"]


# In[ ]:


plt.figure(figsize = (15, 15))
sns.heatmap(pc_imp[colc].corr(), annot = True, square = True, cmap="YlGnBu")
plt.show()


# As seen by the plots, correlation is observed between the same features. Since BMI is calculated using 'Wt' and 'Ht' details and so would be sufficient by itslef to help in identifying the risk level, will drop the 'Wt' and 'Ht' features. 
# 

# In[ ]:


pc_imp = pc_imp.drop(["Ht", "Wt"], axis = 1)


# Will now analyse 'Family_Hist_2', 'Ins_Age', 'Family_Hist_4' and 'BMI' w.r.t 'Response' variable

# In[ ]:


sns.boxplot(x = "Response",y = "Family_Hist_2", data = pc_imp)


# In[ ]:


sns.boxplot(x = "Response",y = "Family_Hist_4", data = pc_imp)


# In[ ]:


sns.boxplot(x = "Response",y = "Ins_Age", data = pc_imp)


# In[ ]:


sns.boxplot(x = "Response",y = "BMI", data = pc_imp)


# We can see from above plots that 'Family_Hist_2', 'Family_Hist_4' and 'Ins_Age' show similar distribution w.r.t 'Response' feature, with 'Family_Hist_2', 'Family_Hist_4' almost the same!. Therefore, can drop either of the features, will drop 'Family_Hist_4' for this iteration. 
# 'All' the features have least 'median' and 'max' (not considering outliers) values for 'Response' value -'8'
# The distribution of the above analysed features is similar for 'Response' values -'3', '4' and '8' 

# In[ ]:


pc_imp = pc_imp.drop("Family_Hist_4", axis =1)


# Analyse now few categorical variables

# In[ ]:


pc_imp["Product_Info_1"].value_counts()


# In[ ]:


pc_imp["Medical_History_8"].value_counts()


# In[ ]:


pc_imp["Medical_History_30"].value_counts()


# In[ ]:


pc_imp["Medical_History_1"].value_counts()


# In[ ]:


pc_imp["Medical_History_2"].value_counts()


# In[ ]:


pc_imp.shape


# In[ ]:


pc_imp.info(verbose = True)


# In[ ]:


#Preparing the data in the 'test' dataset as well now


# In[ ]:


# round((pc_test.isnull().sum()/len(pc_test.index))*100,2)


# In[ ]:


#First deleting the columns in 'test' dataset that have been deleted in the 'train' dataset
#colt = ['Family_Hist_3', 'Family_Hist_5', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32','Ht','Wt','Family_Hist_4','Id']


# In[ ]:


#pc_test_rev = pc_test.drop(colt, axis = 1)


# In[ ]:


#pc_test_rev.shape


# In[ ]:


# pc_test_rev.info(verbose = True)


# Model Building
# Will use Random Forest with default features and estimate performance parameters such as 'Precision', 'Accuracy', 'f-score', 'Recall' to judge the model performance
# Then will use GridSearch, along with Cross-validation, to tune few key parameters such as 'Max_depth', 'min_samples_leaf', 'n_estimators', 'max_features' and get the best possible result
# 

# In[ ]:


#Splitting data between test and train data sets
pc_imp_train, pc_imp_test = train_test_split(pc_imp, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[ ]:


y_pc_imp_train = pc_imp_train.pop("Response")


# In[ ]:


X_pc_imp_train = pc_imp_train


# In[ ]:


y_pc_imp_test = pc_imp_test.pop("Response")


# In[ ]:


X_pc_imp_test = pc_imp_test


# In[ ]:


X_pc_imp_train.shape


# In[ ]:


X_pc_imp_test.shape


# Will use PCA to see if dimensionality can be reduced to only the most important feature set

# In[ ]:


# Will now do PCA to reduce dimensionality
pca = PCA(svd_solver='randomized', random_state=42)


# In[ ]:


pca.fit(X_pc_imp_train)


# In[ ]:


pca.components_


# In[ ]:


colnames = list(X_pc_imp_train.columns)


# In[ ]:


#Dataframe with features and respective first two Prinicple components
pca_df = pd.DataFrame({'PC1':pca.components_[0], 'PC2':pca.components_[1], 'Features': colnames})


# In[ ]:


pca_df.head()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize = (15,15))
plt.scatter(pca_df.PC1, pca_df.PC2)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
for i, txt in enumerate(pca_df.Features):
    plt.annotate(txt, (pca_df.PC1[i],pca_df.PC2[i]))
plt.tight_layout()
plt.show()


# It can be seen that only two dimensions or features,  contrbute heavily in terms of variance along two principal components

# In[ ]:


pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]


# In[ ]:


# Plotting the cummulative variance and number of PCs graph to identify the correct number of PCs required to explain 95% of variance
get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize = (12,8))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of components")
plt.ylabel("Cummulative variance")
plt.show()


# It can be seen from the above plot too that PCA would reduce features to less than 10 as per the variance contribution. While this may be good in the ease of model calculation, we would loose certain important features that may not be important in terms of just considering variance, still would help in model being applicable over different set of data.
# Therefore, will not use PCA reduced feature/dimension set and instead, use the in-build feature filtering within RandomForest model to avoid multi-collinearity  

# In[ ]:


# Base Random Forest result
rfc=RandomForestClassifier()


# In[ ]:


rfc.fit(X_pc_imp_train, y_pc_imp_train)


# In[ ]:


Tr_predict = rfc.predict(X_pc_imp_train)


# In[ ]:


# Let's check the report of our default model
print(classification_report(y_pc_imp_train,Tr_predict))


# In[ ]:


# Printing confusion matrix
print(confusion_matrix(y_pc_imp_train,Tr_predict))


# The scores achieved are all quite high but also show strong signs of overfitting. Lets see the results on test data

# In[ ]:


test_predict = rfc.predict(X_pc_imp_test)


# In[ ]:


# Let's check the report of default model on test dataset
print(classification_report(y_pc_imp_test,test_predict))


# As per the apprehension, model is overfitted as the scores on test dataset are quite poor. 
# Randomforest usually do not overfit but we used default model without any tuninng so overfitting was not avoided.
# We will now tune the model for following hyper parameters:
# Max_depth
# n_estimators
# Max_features
# min_smaples_leaf
# 
# Will use grid search to estimate appropriate hyper parameters to get to best possible model

# In[ ]:


# Max depth tuning using CV an
n_folds = 5

# parameters to build the model on
parameters = {'max_depth': range(10, 80, 5)}

# instantiate the base model
rf_m = RandomForestClassifier()


# fit tree on training data
rf_m = GridSearchCV(rf_m, parameters, 
                    cv=n_folds, 
                   scoring="accuracy",
                    return_train_score=True)
rf_m.fit(X_pc_imp_train, y_pc_imp_train)


# In[ ]:


# scores of GridSearch CV
scores = rf_m.cv_results_
# plotting accuracies with max_depth
plt.figure()
plt.plot(scores["param_max_depth"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_max_depth"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# The best 'test' score is observed at 'max_depth' of 15. Lets now tune other hyper parameters

# In[ ]:


# n_estimators tuning using CV and keeping tuned 'max_depth' of 15
n_folds = 5

# parameters to build the model on
parameters = {'n_estimators': range(300, 2400, 300)}

# instantiate the base model
rf_e = RandomForestClassifier(max_depth=15)


# fit tree on training data
rf_e = GridSearchCV(rf_e, parameters, 
                    cv=n_folds, 
                   scoring="accuracy",
                    return_train_score=True)
rf_e.fit(X_pc_imp_train, y_pc_imp_train)


# In[ ]:


sc = pd.DataFrame(scores)


# In[ ]:


sc.head()


# In[ ]:


# scores of GridSearch CV
scores = rf_e.cv_results_
# plotting accuracies with max_depth
plt.figure()
plt.plot(scores["param_n_estimators"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_n_estimators"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("n_estimators")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# From the above graph, it can be observed that 'n_estimators' value of above '1250' seem to be most appropriate as the 'test' accuracy' is maximum at this value. 

# In[ ]:


# tuning Max_features
# specify number of folds for k-fold CV
n_folds = 5

# parameters to build the model on
parameters = {'max_features': [10,20,30,40,50]}

# instantiate the model
rf_mx = RandomForestClassifier(max_depth=15, n_estimators = 1250)


# fit tree on training data
rf_mx = GridSearchCV(rf_mx, parameters, 
                    cv=n_folds, 
                   scoring="accuracy",
                    return_train_score=True)
rf_mx.fit(X_pc_imp_train, y_pc_imp_train)


# In[ ]:


scores = rf_mx.cv_results_


# In[ ]:


sc = pd.DataFrame(scores)


# In[ ]:


sc.head()


# In[ ]:


# plotting accuracies with max_depth
plt.figure()
plt.plot(scores["param_max_features"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_max_features"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("max_features")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# It's observed that the 'target' accuracy becomes maximum at about 30 'max_features' and difference between 'training' and 'test' accuracy scores are minimum at this point too.  Therefore, will consider 'max_features' to be 30. 

# In[ ]:


# tuning min_samples_leaf
# specify number of folds for k-fold CV
n_folds = 5

# parameters to build the model on
parameters = {'min_samples_leaf': range(50, 1000, 50)}

# instantiate the model
rf_sl = RandomForestClassifier(max_depth=15, max_features = 30, n_estimators = 1250)


# fit tree on training data
rf_sl = GridSearchCV(rf_sl, parameters, 
                    cv=n_folds, 
                   scoring="accuracy",
                    return_train_score=True)
rf_sl.fit(X_pc_imp_train, y_pc_imp_train)


# In[ ]:


scores = rf_sl.cv_results_


# In[ ]:


sc = pd.DataFrame(scores)


# In[ ]:


sc.head()


# In[ ]:


# plotting accuracies with max_depth
plt.figure()
plt.plot(scores["param_min_samples_leaf"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_min_samples_leaf"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("min_samples_leaf")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# Will first consider min_samples_leaf value of 800 as both of the accuracy scores are first closest at this point. However, it can be seen that test and train accuracy scores are best with 'min_sample_leaf' values being around 50. So will consider value to be 50 or lower in case model does not give optimum results with following parameters:

# We have the following ideal values of the hyper parameters that were selected for tuning:
# 1. max_depth: 15
# 2. n_estimators: 1250
# 3. max_features: 30
# 4. min_sample_leaf: 800
# 
# Will now create the model with these parameters and analyse the result

# In[ ]:


rf_f = RandomForestClassifier(max_depth=15, n_estimators = 1250, max_features = 30, min_samples_leaf = 800)


# In[ ]:


rf_f.fit(X_pc_imp_train, y_pc_imp_train)


# In[ ]:


rf_pred = rf_f.predict(X_pc_imp_train)


# In[ ]:


print(classification_report(y_pc_imp_train,rf_pred))


# In[ ]:


print(metrics.accuracy_score(y_pc_imp_train, rf_pred))


# In[ ]:


print(metrics.precision_score(y_pc_imp_train, rf_pred, average = 'weighted'))


# In[ ]:


print(metrics.recall_score(y_pc_imp_train, rf_pred, average = 'weighted'))


# The model with parameters selected is not giving good results on train data itself. Lets try now with lower values of 'min_sample_leaf'

# In[ ]:


rf_f = RandomForestClassifier(max_depth=15, n_estimators = 1600, max_features = 40, min_samples_leaf = 40)


# In[ ]:


rf_f.fit(X_pc_imp_train, y_pc_imp_train)


# In[ ]:


rf_pred = rf_f.predict(X_pc_imp_train)


# In[ ]:


print(classification_report(y_pc_imp_train,rf_pred))


# In[ ]:


print(metrics.accuracy_score(y_pc_imp_train, rf_pred))


# In[ ]:


print(metrics.precision_score(y_pc_imp_train, rf_pred, average = 'weighted'))


# In[ ]:


print(metrics.recall_score(y_pc_imp_train, rf_pred, average = 'weighted'))


# Performance of this model is above 55% for all the key metrics such as 'Accuracy', 'Precision' and 'Recall'. Lets test this model on test data

# In[ ]:


rf_pred = rf_f.predict(X_pc_imp_test)


# In[ ]:


print(classification_report(y_pc_imp_test,rf_pred))


# In[ ]:


print(metrics.accuracy_score(y_pc_imp_test, rf_pred))


# In[ ]:


print(metrics.precision_score(y_pc_imp_test, rf_pred, average = 'weighted'))


# In[ ]:


print(metrics.recall_score(y_pc_imp_test, rf_pred, average = 'weighted'))


# Model performance for test data is quite better for this model as compared to original model. Still, will try few more models and see if the gap between test and train results can be reduced further

# In[ ]:


rf_f_1 = RandomForestClassifier(max_depth=15, n_estimators = 1600, max_features = 50, min_samples_leaf = 30)


# In[ ]:


rf_f_1.fit(X_pc_imp_train, y_pc_imp_train)


# In[ ]:


rf_pred = rf_f_1.predict(X_pc_imp_train)


# In[ ]:


print(classification_report(y_pc_imp_train,rf_pred))


# In[ ]:


print(metrics.accuracy_score(y_pc_imp_train, rf_pred))


# In[ ]:


print(metrics.precision_score(y_pc_imp_train, rf_pred, average = 'weighted'))


# In[ ]:


print(metrics.recall_score(y_pc_imp_train, rf_pred, average = 'weighted'))


# Performance of this model is about 57% for all the key metrics such as 'Accuracy', 'Precision' and 'Recall'. Lets test this model on test data

# In[ ]:


rf_pred_test = rf_f_1.predict(X_pc_imp_test)


# In[ ]:


print(classification_report(y_pc_imp_test,rf_pred_test))


# In[ ]:


print(metrics.accuracy_score(y_pc_imp_test, rf_pred_test))


# In[ ]:


print(metrics.precision_score(y_pc_imp_test, rf_pred_test, average = 'weighted'))


# In[ ]:


print(metrics.recall_score(y_pc_imp_test, rf_pred_test, average = 'weighted'))


# Results are bit closer for train and test data. However, the precision score is less than 50% which needs to be better. Lets try and optimize model bit more

# In[ ]:


rf_f_2 = RandomForestClassifier(max_depth=15, n_estimators = 1600, max_features = 50, min_samples_leaf = 20)


# In[ ]:


rf_f_2.fit(X_pc_imp_train, y_pc_imp_train)


# In[ ]:


rf_pred_train_2 = rf_f_2.predict(X_pc_imp_train)


# In[ ]:


print(classification_report(y_pc_imp_train,rf_pred_train_2))


# In[ ]:


print(metrics.accuracy_score(y_pc_imp_train, rf_pred_train_2))


# In[ ]:


print(metrics.precision_score(y_pc_imp_train, rf_pred_train_2, average = 'weighted'))


# In[ ]:


print(metrics.recall_score(y_pc_imp_train, rf_pred_train_2, average = 'weighted'))


# Performance of this model is near 60% for all the key metrics such as 'Accuracy', 'Precision' and 'Recall'. Lets test this model on test data

# In[ ]:


rf_pred_test_2 = rf_f_2.predict(X_pc_imp_test)


# In[ ]:


print(classification_report(y_pc_imp_test,rf_pred_test_2))


# In[ ]:


print(metrics.accuracy_score(y_pc_imp_test, rf_pred_test_2))


# In[ ]:


print(metrics.precision_score(y_pc_imp_test, rf_pred_test_2, average = 'weighted'))


# In[ ]:


print(metrics.recall_score(y_pc_imp_test, rf_pred_test_2, average = 'weighted'))


# Again, the scores of test and train are still not close enough for model to be considered an optimized one. Will try one last model with use of 'min_samples_leaf'. If model performance is not better, will not set the value of 'min_samples_leaf'

# In[ ]:


rf_f_3 = RandomForestClassifier(max_depth=15, n_estimators = 1800, max_features = 20, min_samples_leaf = 25)


# In[ ]:


rf_f_3.fit(X_pc_imp_train, y_pc_imp_train)


# In[ ]:


rf_pred_train_3 = rf_f_3.predict(X_pc_imp_train)


# In[ ]:


print(classification_report(y_pc_imp_train,rf_pred_train_3))


# In[ ]:


print(metrics.accuracy_score(y_pc_imp_train, rf_pred_train_3))


# In[ ]:


print(metrics.precision_score(y_pc_imp_train, rf_pred_train_3, average = 'weighted'))


# In[ ]:


print(metrics.recall_score(y_pc_imp_train, rf_pred_train_3, average = 'weighted'))


# Performance of this model is again near 60% for all the key metrics such as 'Accuracy', 'Precision' and 'Recall'. Lets test this model on test data

# In[ ]:


rf_pred_test_3 = rf_f_3.predict(X_pc_imp_test)


# In[ ]:


print(classification_report(y_pc_imp_test,rf_pred_test_3))


# In[ ]:


print(metrics.accuracy_score(y_pc_imp_test, rf_pred_test_3))


# In[ ]:


print(metrics.precision_score(y_pc_imp_test, rf_pred_test_3, average = 'weighted'))


# In[ ]:


print(metrics.recall_score(y_pc_imp_test, rf_pred_test_3, average = 'weighted'))


# Again, there's not much improvement in the model. Will now run the model without setting the parameter 'min_samples_leaf'

# In[ ]:


rf_f_4 = RandomForestClassifier(max_depth=15, n_estimators = 1800, max_features = 20)


# In[ ]:


rf_f_4.fit(X_pc_imp_train, y_pc_imp_train)


# In[ ]:


rf_pred_train_4 = rf_f_4.predict(X_pc_imp_train)


# In[ ]:


print(classification_report(y_pc_imp_train,rf_pred_train_4))


# In[ ]:


print(metrics.accuracy_score(y_pc_imp_train, rf_pred_train_4))


# In[ ]:


print(metrics.precision_score(y_pc_imp_train, rf_pred_train_4, average = 'weighted'))


# In[ ]:


print(metrics.recall_score(y_pc_imp_train, rf_pred_train_4, average = 'weighted'))


# Performance of this model is lot better with scores around 75% for all the key metrics such as 'Accuracy', 'Precision' and 'Recall'. Lets test this model on test data

# In[ ]:


rf_pred_test_4 = rf_f_4.predict(X_pc_imp_test)


# In[ ]:


print(classification_report(y_pc_imp_test,rf_pred_test_4))


# In[ ]:


print(metrics.accuracy_score(y_pc_imp_test, rf_pred_test_4))


# In[ ]:


print(metrics.precision_score(y_pc_imp_test, rf_pred_test_4, average = 'weighted'))


# In[ ]:


print(metrics.recall_score(y_pc_imp_test, rf_pred_test_4, average = 'weighted'))


# The test scores are best of any model so far but the difference between the train and test scores is quite huge. Will now select larger value of 'min_samples_leaf' than the default '1' to reduce overfitting

# In[ ]:


rf_f_5 = RandomForestClassifier(max_depth=15, n_estimators = 1800, max_features = 20, min_samples_leaf = 15)


# In[ ]:


rf_f_5.fit(X_pc_imp_train, y_pc_imp_train)


# In[ ]:


rf_pred_train_5 = rf_f_5.predict(X_pc_imp_train)


# In[ ]:


print(classification_report(y_pc_imp_train,rf_pred_train_5))


# In[ ]:


print(metrics.accuracy_score(y_pc_imp_train, rf_pred_train_5))


# In[ ]:


print(metrics.precision_score(y_pc_imp_train, rf_pred_train_5, average = 'weighted'))


# In[ ]:


print(metrics.recall_score(y_pc_imp_train, rf_pred_train_5, average = 'weighted'))


# Performance of this model is decent with scores around 60% for all the key metrics such as 'Accuracy', 'Precision' and 'Recall'. Lets test this model on test data and see if test scores are around the 'train' data scores

# In[ ]:


rf_pred_test_5 = rf_f_5.predict(X_pc_imp_test)


# In[ ]:


print(classification_report(y_pc_imp_test,rf_pred_test_5))


# In[ ]:


print(metrics.accuracy_score(y_pc_imp_test, rf_pred_test_5))


# In[ ]:


print(metrics.precision_score(y_pc_imp_test, rf_pred_test_5, average = 'weighted'))


# In[ ]:


print(metrics.recall_score(y_pc_imp_test, rf_pred_test_5, average = 'weighted'))


# The test and train scores are closer to each other, with test scores being above 50%. Lets try optimizing further

# In[ ]:


rf_f_6 = RandomForestClassifier(max_depth=15, n_estimators = 1800, max_features = 20, min_samples_leaf = 12)


# In[ ]:


rf_f_6.fit(X_pc_imp_train, y_pc_imp_train)


# In[ ]:


rf_pred_train_6 = rf_f_6.predict(X_pc_imp_train)


# In[ ]:


print(classification_report(y_pc_imp_train,rf_pred_train_6))


# In[ ]:


print(metrics.accuracy_score(y_pc_imp_train, rf_pred_train_6))


# In[ ]:


print(metrics.precision_score(y_pc_imp_train, rf_pred_train_6, average = 'weighted'))


# In[ ]:


print(metrics.recall_score(y_pc_imp_train, rf_pred_train_6, average = 'weighted'))


# Performance of this model is decent with scores around 60% for all the key metrics such as 'Accuracy', 'Precision' and 'Recall'. Lets test this model on test data and see if test scores are around the 'train' data scores

# In[ ]:


rf_pred_test_6 = rf_f_6.predict(X_pc_imp_test)


# In[ ]:


print(classification_report(y_pc_imp_test,rf_pred_test_6))


# In[ ]:


print(metrics.accuracy_score(y_pc_imp_test, rf_pred_test_6))


# In[ ]:


print(metrics.precision_score(y_pc_imp_test, rf_pred_test_6, average = 'weighted'))


# In[ ]:


print(metrics.recall_score(y_pc_imp_test, rf_pred_test_6, average = 'weighted'))


# Both train and test scores are closer with test scores above 50%. Will execute one more model with 'min_leaf_samples' less than 10, i.e., 8, and then select the best model out of the lot so far

# In[ ]:


rf_f_7 = RandomForestClassifier(max_depth=15, n_estimators = 1800, max_features = 20, min_samples_leaf = 8)


# In[ ]:


rf_f_7.fit(X_pc_imp_train, y_pc_imp_train)


# In[ ]:


rf_pred_train_7 = rf_f_7.predict(X_pc_imp_train)


# In[ ]:


print(classification_report(y_pc_imp_train,rf_pred_train_7))


# In[ ]:


print(metrics.accuracy_score(y_pc_imp_train, rf_pred_train_7))


# In[ ]:


print(metrics.precision_score(y_pc_imp_train, rf_pred_train_7, average = 'weighted'))


# In[ ]:


print(metrics.recall_score(y_pc_imp_train, rf_pred_train_7, average = 'weighted'))


# Performance of this model is decent with scores around 60% for all the key metrics such as 'Accuracy', 'Precision' and 'Recall'. Lets test this model on test data and see if test scores are around the 'train' data scores

# In[ ]:


rf_pred_test_7 = rf_f_7.predict(X_pc_imp_test)


# In[ ]:


print(classification_report(y_pc_imp_test,rf_pred_test_7))


# In[ ]:


print(metrics.accuracy_score(y_pc_imp_test, rf_pred_test_7))


# In[ ]:


print(metrics.precision_score(y_pc_imp_test, rf_pred_test_7, average = 'weighted'))


# In[ ]:


print(metrics.recall_score(y_pc_imp_test, rf_pred_test_7, average = 'weighted'))


# So far the last three models have given similar performances. Will now see if we can get better performance by optimizing 'Max_depth' now  

# In[ ]:


rf_f_8 = RandomForestClassifier(max_depth=20, n_estimators = 1800, max_features = 30, min_samples_leaf = 15)


# In[ ]:


rf_f_8.fit(X_pc_imp_train, y_pc_imp_train)


# In[ ]:


rf_pred_train_8 = rf_f_8.predict(X_pc_imp_train)


# In[ ]:


print(classification_report(y_pc_imp_train,rf_pred_train_8))


# In[ ]:


print(metrics.accuracy_score(y_pc_imp_train, rf_pred_train_8))


# In[ ]:


print(metrics.precision_score(y_pc_imp_train, rf_pred_train_8, average = 'weighted'))


# In[ ]:


print(metrics.recall_score(y_pc_imp_train, rf_pred_train_8, average = 'weighted'))


# Performance of this model is decent with scores around 60% for all the key metrics such as 'Accuracy', 'Precision' and 'Recall'. Lets test this model on test data and see if test scores are around the 'train' data scores

# In[ ]:


rf_pred_test_8 = rf_f_8.predict(X_pc_imp_test)


# In[ ]:


print(classification_report(y_pc_imp_test,rf_pred_test_8))


# In[ ]:


print(metrics.accuracy_score(y_pc_imp_test, rf_pred_test_8))


# In[ ]:


print(metrics.precision_score(y_pc_imp_test, rf_pred_test_8, average = 'weighted'))


# In[ ]:


print(metrics.recall_score(y_pc_imp_test, rf_pred_test_8, average = 'weighted'))


# In[ ]:


# Max depth tuning using CV an
n_folds = 5

# parameters to build the model on
parameters = {'max_depth': range(10, 120, 5)}

# instantiate the base model
rf_m = RandomForestClassifier(n_estimators = 1600, max_features = 30, min_samples_leaf = 15)


# fit tree on training data
rf_m = GridSearchCV(rf_m, parameters, 
                    cv=n_folds, 
                   scoring="accuracy",
                    return_train_score=True)
rf_m.fit(X_pc_imp_train, y_pc_imp_train)


# In[ ]:


# scores of GridSearch CV
scores = rf_m.cv_results_
# plotting accuracies with max_depth
plt.figure()
plt.plot(scores["param_max_depth"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_max_depth"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# We can see the max_depth of around 18 results in best test score and difference between train and test at this point is also minimum. Will keep the max_depth as '18'

# In[ ]:


rf_f_9 = RandomForestClassifier(max_depth=18, n_estimators = 1800, max_features = 25, min_samples_leaf = 15)


# In[ ]:


rf_f_9.fit(X_pc_imp_train, y_pc_imp_train)


# In[ ]:


rf_pred_train_9 = rf_f_9.predict(X_pc_imp_train)


# In[ ]:


print(classification_report(y_pc_imp_train,rf_pred_train_9))


# In[ ]:


print(metrics.accuracy_score(y_pc_imp_train, rf_pred_train_9))


# In[ ]:


print(metrics.precision_score(y_pc_imp_train, rf_pred_train_9, average = 'weighted'))


# In[ ]:


print(metrics.recall_score(y_pc_imp_train, rf_pred_train_9, average = 'weighted'))


# Performance of this model is decent with scores around 60% for all the key metrics such as 'Accuracy', 'Precision' and 'Recall'. Lets test this model on test data and see if test scores are around the 'train' data scores

# In[ ]:


rf_pred_test_9 = rf_f_9.predict(X_pc_imp_test)


# In[ ]:


print(classification_report(y_pc_imp_test,rf_pred_test_9))


# In[ ]:


print(metrics.accuracy_score(y_pc_imp_test, rf_pred_test_9))


# In[ ]:


print(metrics.precision_score(y_pc_imp_test, rf_pred_test_9, average = 'weighted'))


# In[ ]:


print(metrics.recall_score(y_pc_imp_test, rf_pred_test_9, average = 'weighted'))


# Even after performing multiple iterations, it can be seen that the test accuracy and recall scores are around 53%, while precision is at 50%. Will stick with this model then as this is giving best performance so far with train scores around 60% and test at 53%. 
# I am submitting this project with modelling performed by using Randomforest model though we could get better performance using models such as XGBOOST. However, for this submission I am sticking randomforest based model: 'rf_f_9'

# Preparing the data in the 'test' dataset as well now so that can predict the risk values using the selected model. 

# In[ ]:


round((pc_test.isnull().sum()/len(pc_test.index))*100,2)


# In[ ]:


#First deleting the columns in 'test' dataset that have been deleted in the 'train' dataset
colt = ['Family_Hist_3', 'Family_Hist_5', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32','Ht','Wt','Family_Hist_4','Id']


# In[ ]:


pc_test_rev = pc_test.drop(colt, axis = 1)


# In[ ]:


pc_test_rev.shape


# In[ ]:


round((pc_test_rev.isnull().sum()/len(pc_test_rev.index))*100,2)


# In[ ]:


# Focussing only on columns that have missing values
col = []
for i in pc_test_rev.columns:
    if round((pc_test_rev[i].isnull().sum()/len(pc_test_rev.index))*100,2) != 0:
        col.append(i)
print(col)
print(len(col))


# In[ ]:


# Employment_Info_1 and Employment_Info_4 have quite low percentage of missing values so will just remove the rows
pc_test_rev = pc_test_rev[~pd.isnull(pc_test_rev["Employment_Info_1"])]


# In[ ]:


pc_test_rev = pc_test_rev[~pd.isnull(pc_test_rev["Employment_Info_4"])]


# In[ ]:


round((pc_test_rev[col].isnull().sum()/len(pc_test_rev[col].index))*100,2)


# In[ ]:


pc_test_rev.index = pd.RangeIndex(1, len(pc_test_rev.index) + 1)


# In[ ]:


# Will treat missing values via iterative imputer for rest of columns. Will first encode the column 'Product_Info_2'


# In[ ]:


lc = LabelEncoder()


# In[ ]:


pc_test_rev["Product_Info_2"] = lc.fit_transform(pc_test_rev["Product_Info_2"])


# In[ ]:


# Imputing values using Iterative Imputer
colt2 = pc_test_rev.columns


# In[ ]:


pc_test_imp = pd.DataFrame(IterativeImputer().fit_transform(pc_test_rev))


# In[ ]:


pc_test_imp.columns = colt2


# In[ ]:


pc_test_imp.head()


# In[ ]:


round((pc_test_imp[col].isnull().sum()/len(pc_test_imp.index))*100,2)


# It can be seen that all null values have been treated. However, side-effect of Iterative computer is that it converts all columns to float type while processing them. Will now convert the columns originally integer type to 'int' 

# In[ ]:


colt4 = np.delete(cols4,0)


# In[ ]:


print(len(colt4))


# In[ ]:


colt4 = np.delete(colt4,107)


# In[ ]:


print(colt4)


# In[ ]:


pc_test_imp[colt4] = pc_test_imp[colt4].astype(int)


# In[ ]:


pc_test_imp["Product_Info_2"] = pc_test_imp["Product_Info_2"].astype(int)


# In[ ]:


pc_test_imp.head()


# Will now predict the risk classifiers using the model 'rf_f_9' for the above test data set for submission

# In[ ]:


rf_pred_sub = rf_f_9.predict(pc_test_imp)


# In[ ]:


rf_pred_sub


# In[ ]:


rf_pred_sub.reshape(-1)


# In[ ]:


pred_rk = pd.DataFrame({'Risk': rf_pred_sub})


# In[ ]:


pred_rk.head()


# In[ ]:


pred_rk.index = pd.RangeIndex(1, len(pred_rk.index) + 1)


# In[ ]:


len(pred_rk)


# Creating the required output file

# In[ ]:


pc_test.index = pd.RangeIndex(1, len(pc_test.index) + 1)


# In[ ]:


pc_test.head()


# In[ ]:


fn_dt = pd.DataFrame()


# In[ ]:


fn_dt['Id'] = pc_test['Id'] 


# In[ ]:


fn_dt['Response'] = pred_rk['Risk']


# In[ ]:


# Final dataframe to be submitted is:
fn_dt.head()


# In[ ]:


# Writing this to csv file for submission
fn_dt.to_csv("C://Users/pchadha/Boosting_Kaggle_Practice/Prudential_Life_insurance/submission_file.csv")


# In[ ]:




