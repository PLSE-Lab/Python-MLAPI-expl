#!/usr/bin/env python
# coding: utf-8

# # Introduction 
# 
# In this notebook, I took the UCI Heart Disease database to start a beginner guide to classification problems. I dealt with the data as if it was in a kaggle competition. Hence, I divided it into two sets: training and testing sets. The analysis of data and the adaptation of the processing phases were done considering only the training set. These modifications are then applied to the test set. This allows us to avoid having data leakage which is a common data science problem and a lot of notebooks in kaggle suffer from this problem. For those who do not know what the data leakage is. It's a problem when the test set data contribute to the processing and analysis phases. Thus, it will influence the model used and we will obtain great results. So what's the problem? We are trying to increase the performance of our models so what's the fuss? Well in real-life problems we won't have access to the testing data. We will obtain this data when the model is set in place to do its job. Therefore, including information that is available only in the test set will cause our model to be unrealistic.  
# 
# As I was saying I treated the data as if it was a kaggle competition. Therefore, I won't be using the cross-validation method for my built models. Of course, if you are interested to do a more precise work you can use the whole data and use cross-validation with some pipelines to process the data each time the cross-validation is done.
# 
# I hope my explanations are precise and clear. Please leave your comments and your remark so I can improve the notebook. 
# 
# 

# In[ ]:



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns


# # Data Analysis and Processing
# ## Data Acquisition and Preparation

# In[ ]:


Data = pd.read_csv("../input/heart-disease-uci/heart.csv")
Data.head()


# Now let's divide the data into train and test sets

# In[ ]:


from sklearn.model_selection import train_test_split
train, test =train_test_split(Data,test_size=0.2,random_state=0)
print("train : {} , test : {}".format(train.shape, test.shape))


# In[ ]:


y_train = train["target"]
X_train = train.drop("target",axis=1)
y_test = test["target"]
X_test = test.drop('target',axis=1)


# In[ ]:


X_train.isnull().sum()


# In[ ]:


X_test.isnull().sum()


# There is no missing values in the dataset.

# ## Data Analysis

# Let's start by visualizing the data:
# * Histograms:

# In[ ]:


plt.figure(figsize=(20,20))
i=1
for elt in X_train.columns:
    plt.subplot(4,4,i)
    X_train[elt].hist(bins=20)
    plt.xlabel(elt)
    i+=1
plt.show()


# We can notice that the age, trestbps, chol, thalach, and oldpeak are continuous values.

# * Bar Plots: 

# Now for the features that look like a categorical features let's draw the Bar plots. 

# In[ ]:


features=["sex","cp","fbs","restecg", "exang","slope", 'ca', 'thal']
i=1
plt.figure(figsize=(10,20))
for feature in features:
    plt.subplot(4,2,i)
    vals = np.sort(X_train[feature].unique()).tolist()
    for v in vals:
        s_b = X_train[X_train[feature]==v]
        s_b2 = y_train[s_b.index]
        a=s_b2[s_b2==1].shape[0]
        b=s_b2[s_b2==0].shape[0]
        plt.bar(v - 0.125, a, color = 'r', width = 0.25)
        plt.bar(v + 0.125, b, color = 'g', width = 0.25)
    plt.xlabel(feature)
    plt.legend(["Has Heart Disease","Healthy"])
    i+=1
plt.show()


# We can notice that some of these categorical features have no ordering relationship. Meaning that value in the middle can have more patients diagnosed with heart disease than other values for example thal. 
# It would be wise to transform these features into categorical ones and then replace them with dummy variables.
# 
# From these visualizations, we conclude that numeric categorical features with multiple values (i.e., cp, restecg, slope, ca, and thal) will be transformed into categorical features and will be replaced with dummy variables.

# * 2D Scatter Plots:

# Now for continuous variables let's see how they are related to each other. 

# In[ ]:


c_features=["age", "trestbps", "chol", "thalach", "oldpeak"]
plt.figure(figsize=(10,25))
i=1
for idx,elt in enumerate(c_features):
    f= idx + 1
    while f < len(c_features):
        plt.subplot(5,2,i)
        sns.scatterplot(X_train[c_features[f]],X_train[elt],hue = y_train,
                 palette=['green','red'],legend='full')
        plt.xlabel(c_features[f])
        plt.ylabel(elt)
        plt.legend()
        f+=1
        i+=1
plt.show()


# Let's check the skewness of the continuous variables!

# In[ ]:


from scipy.stats import skew


# In[ ]:


for elt in c_features: 
    print(elt, skew(X_train[elt]))


# So we note that two features are skewed to the right (i.e., chol and oldpeak). These two variables will be transformed with a log function so that we can reduce their skewness.

# ## Data Processing

# Now let's apply the decision we made. 

# ### Categorical Features

# * Transforming the data type to str.

# In[ ]:


features_to_str=["cp","restecg", "slope", 'ca', 'thal']
for elt in features_to_str:
    X_train[elt] = X_train[elt].apply(str)
    X_test[elt] = X_test[elt].apply(str)


# * Getting the dummy variables.

# In[ ]:


X_train = pd.get_dummies(data=X_train, prefix=features_to_str, 
                        prefix_sep='=', columns=features_to_str)
X_train


# In[ ]:


X_test = pd.get_dummies(data=X_test, prefix=features_to_str, 
                        prefix_sep='=', columns=features_to_str)
X_test


# We can notice that the X_train has now 27 features but the X_test has only 25 features. This is due to the transformation of categorical features to dummy variables. Some categorical features of the test data do not have all the values as the train data. So we will add the missing columns with zero values.

# In[ ]:


for elt in X_train.columns : 
    if elt not in X_test.columns: 
        X_test[elt]= np.zeros(len(X_test))
        print(elt)
print("X_train: {}, X_test: {}".format(X_train.shape, X_test.shape))


# Great now we have the same number of columns. One more thing before moving to the next step. Let's make sure that both X_train and X_test have the same order in the features.

# In[ ]:


X_test = X_test[X_train.columns]
X_test


# ### Numerical Features

# * Now let's reduce the skewness of the numeric variables

# In[ ]:


features_to_log = ["chol", "oldpeak"]
for elt in features_to_log:
    X_train[elt] = np.log(1+ X_train[elt])
    X_test[elt] = np.log(1+ X_test[elt])


# * Let's scale data
# 
# Some classification methods specially similarity-based like K-Nearest Neighbors are sensitive to the scale of features. for this, we will transform the numeric variables to the same scale using a standard transformation. 
# 
# It is important to make sure that we will only scale the continuous data meaning that categorical data and dummy variables will not be scaled as they are already in an [0,1] interval. It is also important to make sure that the scaler will be fitted to only the training data and then applied to both training and testing data to avoid data leakage.

# In[ ]:


from sklearn.preprocessing import StandardScaler
data_scaler = StandardScaler()
data_scaler.fit(X_train[c_features])
X_train[c_features] = data_scaler.transform(X_train[c_features])
X_test[c_features] = data_scaler.transform(X_test[c_features])
X_train.head()


# In[ ]:


X_test.head()


# Now our data is ready. Although we can still do some processing like feature engineering using Principal Component Analysis: PCA (for linear problems) and Kernel PCA (for nonlinear problems) and/or Linear Discriminant Analysis. To compare the efficiency of these methods, we will apply them to our problem. 

# ### Feature Engineering

# #### Principal Component Analysis (PCA)

# The principal component analysis is an unsupervised method for feature engineering. It is unsupervised because it won't consider the effects of the transformation on the target value. PCA creates new features that are a linear combination of the old once. It identifies the patterns in data, detects correlations and tries to reduce dimensionality while catching most of the variance. The number of returned new features that we technically call components is the most important parameter of the approach. At this point, we will retrieve all components (the same number of features). We will analyze this output then we will decide on the number of components.

# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


pca = PCA(n_components=None)
pca.fit(X_train)
pca_components = pd.DataFrame(pca.explained_variance_ratio_,columns=['Data Variance per Component'])
pca_components['Total Captured Variance'] = pca_components['Data Variance per Component'].cumsum()


# In[ ]:


pca_components


# This shows us the amount of variance we will catch with that number of component e.g., for one component we will have 0.257 of the data variance. Since our goal is to reduce the dimension of data we will choose only 10 components. so that we will be catching 89% of the data total variance.

# In[ ]:


pca= PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


# #### Principal Component Analysis (Kernel PCA)

# Kernel PCA works almost like the PCA but on nonlinear problems. At first, the algorithm will increase the dimension of data by applying a kernel transformation (e.g., RBF) to "linearize" the problem. Then it will apply the PCA. Kernel PCA has two important parameters, the kernel function (in this example we will use the Gaussian kernel RBF) and the number of components (similar to PCA we will use 10 components).

# In[ ]:


from sklearn.decomposition import KernelPCA
kpca = KernelPCA(kernel='rbf', n_components = 10)
X_train_kpca = kpca.fit_transform(X_train)
X_test_kpca = kpca.transform(X_test)


# #### Linear Discriminant Analysis (LDA)

# Linear discriminant analysis is a supervised feature engineering method. Unlike the PCA, LDA will create new features that maximize the separation between the different classes. Since we have only two classes (has heart disease and healthy) then we can create only one component using the LDA.

# In[ ]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit(X_train,y_train)
X_train_lda= lda.transform(X_train)
X_test_lda = lda.transform(X_test)


# # Model Building & Predictions:

# Now let's first start by defining the performance metrics. We have here a two-class classification problem. We can use several metrics like precision, accuracy, F1 score, etc...  
# 
# 1.     Accuracy Classification accuracy is our starting point. It is the number of correct predictions made divided by the total number of predictions made, multiplied by 100 to turn it into a percentage.
# 1.     Precision is the number of True Positives divided by the number of True Positives and False Positives. Put another way, it is the number of positive predictions divided by the total number of positive class values predicted. It is also called the Positive Predictive Value (PPV).
# 1.     Sensitivity is the number of True Positives divided by the number of True Positives and the number of False Negatives. Put another way it is the number of positive predictions divided by the number of positive class values in the test data. It is also called Recall or the True Positive Rate.
# 1.     Specificity
# 
# Unlike the regression problems, the classification problem is a multi-objective problem. In a two-class classification problem, we find two types of errors: 
# * False Positives: when the ground truth for the observation is class 0 or passive class (for example a healthy person is our case) and the model predicts this data point as class 1 or active class (in our case diagnosed with heart disease). Well, this type of error is important but not dangerous. In our case, for example, those who are diagnosed with heart disease will only require some small changes in their life, like a special food diet, should quit smoking, or prescribed some medications. I would like to think this kind of error is a warning. 
# * False Negatives: when the ground truth for the observation is class 1 (diagnosed with heart disease) and the model predicts the patient as healthy (class 0). This type of error is extremely dangerous as it causes that a person who is ill or presents a risk of having health complications will not get the medical attention he needs and may even cause some health complications. 
# 
# Therefore, in this application, we will focus mainly on reducing the False-negative errors and thus maximizing the recall_score for class 1.

# To see the predictions we will use the confusion matrix, and we will use the full report on classification. 

# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, recall_score, make_scorer


# The recall_score function can provides us with the recall of each of the classes when using the option "average=None". Otherwise, since we are dealing with a binary (two-classes) classification problem then using the recall score directly will return only the recall score for the active class (1). 

# Here we will code the function to produce the cumulative accuracy profile. The cumulative accuracy profile (CAP) is a great metric to measure the performance of the model. Well, let's think of it this way if we have a percentage of people who are diagnosed ill from a set a population we want our model to detect them immediately and avoid us from recheck if the people he selected to be ill are not ill. Well, this is the perfect performance that one can hope for. It sounds like having a crystal ball and we will directly predict the subjects that will be diagnosed with heart disease. The closer the performance of our model to this behavior the better it is. However, if we choose a completely random model we will always have the same percentage of detected illness and we will have to check the whole population to detect all the ill patients. Well, this is the baseline for a prediction model if any model is even lower than the random it is completely rubbish. Now the CAP is computed as the ratio of the area between the cumulative accuracy of our model and the random line over the area between the perfect model and the random line. Since this is quite an expensive computation we can approximate this result by estimating the CAP for 50% of the population. Once we have this value there is a rule of thumb to determine the performance of the model as follow: 
# * CAP(50%) < 60% very poor model 
# * 60% < CAP(50%) < 70% poor model 
# * 70% < CAP(50%) < 80% good model
# * 80% < CAP(50%) < 90% very good model 
# * 90% < CAP(50%) a too good to be true model (one should check for forward seeing predictors i.e., predictors that describe directly the outcome and that we can only obtain by knowing the outcome, or this may be caused by overfitting the model, or simply it's a great model).

# In[ ]:


def CAP_performance(y_true,y_pred,p=0.5,plot=False):
    df =pd.DataFrame()
    df["GT"] = y_true
    df["Predictions"]=y_pred
    positive_percentage = df["GT"].sum() / len(df)
    df.sort_values(by=["Predictions"],axis=0,ascending=False, inplace=True)
    df.reset_index(inplace=True,drop=True)
    df['CumPredictions'] = df["GT"].cumsum()
    df['CumAcc']=df['CumPredictions'] / len(df)
    idx = int(np.trunc(p*len(df)))
    CAP=(df['CumAcc'].values[idx]+df['CumAcc'].values[idx+1])/2
    if plot: 
        plt.figure()
        #random line : 
        plt.plot([0,len(df)],[0,100],color='black',label="Random Model")
        #perfect model : 
        plt.plot([0,df['GT'].sum(),len(df)],[0,100,100],color='green',label="Cristal Ball Model")
        #our model:
        x=list(range(len(df)+1))
        y= [0]+ (df['CumAcc']*100/positive_percentage).values.tolist()
        plt.plot(x,y,color='red',label="Model Performance")
        plt.plot([p*len(df),p*len(df),0],[0,CAP*100/positive_percentage,CAP*100/positive_percentage],
                 color='blue',label='{} %'.format(CAP*100/positive_percentage))
        plt.xlim(0,len(df)+1)
        plt.ylim(0,101)
        plt.legend()
        plt.show()
    return CAP/positive_percentage


# Let's create a data frame in which we will save all the performances of the models. For this we will create a dictionnary than we will convert it into a dataframe. 

# In[ ]:


performances = {}
performances["Method"]=[]
performances["Healthy_Recall"]=[]
performances["Disease_Recall"]=[]
performances["CAP"]=[]


# ## Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(random_state=0)
LR_pca = LogisticRegression(random_state=0)
LR_kpca = LogisticRegression(random_state=0)
LR_lda = LogisticRegression(random_state=0)


# In[ ]:


LR.fit(X_train,y_train)
y_pred = LR.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
cap = CAP_performance(y_test,y_pred,plot=True)


# In[ ]:


performances["Method"].append("Logistic Regression")
performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])
performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])
performances["CAP"].append(cap)


# In[ ]:


LR_pca.fit(X_train_pca,y_train)
y_pred = LR_pca.predict(X_test_pca)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
cap = CAP_performance(y_test,y_pred,plot=True)


# In[ ]:


performances["Method"].append("Logistic Regression PCA")
performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])
performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])
performances["CAP"].append(cap)


# In[ ]:


LR_kpca.fit(X_train_kpca,y_train)
y_pred = LR_kpca.predict(X_test_kpca)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
cap = CAP_performance(y_test,y_pred,plot=True)


# In[ ]:


performances["Method"].append("Logistic Regression K_PCA")
performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])
performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])
performances["CAP"].append(cap)


# In[ ]:


LR_lda.fit(X_train_lda,y_train)
y_pred = LR_lda.predict(X_test_lda)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
cap = CAP_performance(y_test,y_pred,plot=True)


# In[ ]:


performances["Method"].append("Logistic Regression LDA")
performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])
performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])
performances["CAP"].append(cap)


# ## Support Vector Machines

# In[ ]:


from sklearn.svm import SVC
SVM = SVC(C=0.5,gamma=0.1,random_state=0)
SVM_pca = SVC(C=0.5,gamma=0.1,random_state=0)
SVM_kpca = SVC(C=0.5,gamma=0.1,random_state=0)
SVM_lda = SVC(C=0.5,gamma=0.1,random_state=0)


# In[ ]:


SVM.fit(X_train,y_train)
y_pred = SVM.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
cap = CAP_performance(y_test,y_pred,plot=True)


# In[ ]:


performances["Method"].append("SVM")
performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])
performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])
performances["CAP"].append(cap)


# In[ ]:


SVM_pca.fit(X_train_pca,y_train)
y_pred = SVM_pca.predict(X_test_pca)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
cap = CAP_performance(y_test,y_pred,plot=True)


# In[ ]:


performances["Method"].append("SVM PCA")
performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])
performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])
performances["CAP"].append(cap)


# In[ ]:


SVM_kpca.fit(X_train_kpca,y_train)
y_pred = SVM_kpca.predict(X_test_kpca)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
cap = CAP_performance(y_test,y_pred,plot=True)


# In[ ]:


performances["Method"].append("SVM K_PCA")
performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])
performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])
performances["CAP"].append(cap)


# In[ ]:


SVM_lda.fit(X_train_lda,y_train)
y_pred = SVM_lda.predict(X_test_lda)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
cap = CAP_performance(y_test,y_pred,plot=True)


# In[ ]:


performances["Method"].append("SVM LDA")
performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])
performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])
performances["CAP"].append(cap)


# ## K-nearest neighbors

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
KNN= KNeighborsClassifier(n_neighbors=5)
KNN_pca= KNeighborsClassifier(n_neighbors=5)
KNN_kpca= KNeighborsClassifier(n_neighbors=5)
KNN_lda= KNeighborsClassifier(n_neighbors=5)


# In[ ]:


KNN.fit(X_train,y_train)
y_pred = KNN.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
cap = CAP_performance(y_test,y_pred,plot=True)


# In[ ]:


performances["Method"].append("KNN 5")
performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])
performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])
performances["CAP"].append(cap)


# In[ ]:


KNN_pca.fit(X_train_pca,y_train)
y_pred = KNN_pca.predict(X_test_pca)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
cap = CAP_performance(y_test,y_pred,plot=True)


# In[ ]:


performances["Method"].append("KNN 5 PCA")
performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])
performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])
performances["CAP"].append(cap)


# In[ ]:


KNN_kpca.fit(X_train_kpca,y_train)
y_pred = KNN_kpca.predict(X_test_kpca)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
cap = CAP_performance(y_test,y_pred,plot=True)


# In[ ]:


performances["Method"].append("KNN 5 K_PCA")
performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])
performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])
performances["CAP"].append(cap)


# In[ ]:


KNN_lda.fit(X_train_lda,y_train)
y_pred = KNN_lda.predict(X_test_lda)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
cap = CAP_performance(y_test,y_pred,plot=True)


# In[ ]:


performances["Method"].append("KNN 5 LDA")
performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])
performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])
performances["CAP"].append(cap)


# ## Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier(criterion='entropy',random_state=0)
DTC_pca = DecisionTreeClassifier(criterion='entropy',random_state=0)
DTC_kpca = DecisionTreeClassifier(criterion='entropy',random_state=0)
DTC_lda = DecisionTreeClassifier(criterion='entropy',random_state=0)


# In[ ]:


DTC.fit(X_train,y_train)
y_pred = DTC.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
cap = CAP_performance(y_test,y_pred,plot=True)


# In[ ]:


performances["Method"].append("DTC")
performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])
performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])
performances["CAP"].append(cap)


# In[ ]:


DTC_pca.fit(X_train_pca,y_train)
y_pred = DTC_pca.predict(X_test_pca)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
cap = CAP_performance(y_test,y_pred,plot=True)


# In[ ]:


performances["Method"].append("DTC PCA")
performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])
performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])
performances["CAP"].append(cap)


# In[ ]:


DTC_kpca.fit(X_train_kpca,y_train)
y_pred = DTC_kpca.predict(X_test_kpca)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
cap = CAP_performance(y_test,y_pred,plot=True)


# In[ ]:


performances["Method"].append("DTC K_PCA")
performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])
performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])
performances["CAP"].append(cap)


# In[ ]:


DTC_lda.fit(X_train_lda,y_train)
y_pred = DTC_lda.predict(X_test_lda)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
cap = CAP_performance(y_test,y_pred,plot=True)


# In[ ]:


performances["Method"].append("DTC LDA")
performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])
performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])
performances["CAP"].append(cap)


# ## Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators = 150 , criterion='entropy', random_state=0)
RFC_pca = RandomForestClassifier(n_estimators = 150 , criterion='entropy', random_state=0)
RFC_kpca = RandomForestClassifier(n_estimators = 150 , criterion='entropy', random_state=0)
RFC_lda = RandomForestClassifier(n_estimators = 150 , criterion='entropy', random_state=0)


# In[ ]:


RFC.fit(X_train,y_train)
y_pred = RFC.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
cap = CAP_performance(y_test,y_pred,plot=True)


# In[ ]:


performances["Method"].append("RF")
performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])
performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])
performances["CAP"].append(cap)


# In[ ]:


RFC_pca.fit(X_train_pca,y_train)
y_pred = RFC_pca.predict(X_test_pca)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
cap = CAP_performance(y_test,y_pred,plot=True)


# In[ ]:


performances["Method"].append("RF PCA")
performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])
performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])
performances["CAP"].append(cap)


# In[ ]:


RFC_kpca.fit(X_train_kpca,y_train)
y_pred = RFC_kpca.predict(X_test_kpca)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
cap = CAP_performance(y_test,y_pred,plot=True)


# In[ ]:


performances["Method"].append("RF K_PCA")
performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])
performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])
performances["CAP"].append(cap)


# In[ ]:


RFC_lda.fit(X_train_lda,y_train)
y_pred = RFC_lda.predict(X_test_lda)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
cap = CAP_performance(y_test,y_pred,plot=True)


# In[ ]:


performances["Method"].append("RF LDA")
performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])
performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])
performances["CAP"].append(cap)


# ## Extreme Gradient Boost

# In[ ]:


from xgboost import XGBClassifier
XGC = XGBClassifier(n_estimators = 150 , random_state=0)
XGC_pca = XGBClassifier(n_estimators = 150 , random_state=0)
XGC_kpca = XGBClassifier(n_estimators = 150 , random_state=0)
XGC_lda = XGBClassifier(n_estimators = 150 , random_state=0)


# In[ ]:


XGC.fit(X_train,y_train)
y_pred = XGC.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
cap = CAP_performance(y_test,y_pred,plot=True)


# In[ ]:


performances["Method"].append("XGB")
performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])
performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])
performances["CAP"].append(cap)


# In[ ]:


XGC_pca.fit(X_train_pca,y_train)
y_pred = XGC_pca.predict(X_test_pca)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
cap = CAP_performance(y_test,y_pred,plot=True)


# In[ ]:


performances["Method"].append("XGB PCA")
performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])
performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])
performances["CAP"].append(cap)


# In[ ]:


XGC_kpca.fit(X_train_kpca,y_train)
y_pred = XGC_kpca.predict(X_test_kpca)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
cap = CAP_performance(y_test,y_pred,plot=True)


# In[ ]:


performances["Method"].append("XGB K_PCA")
performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])
performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])
performances["CAP"].append(cap)


# In[ ]:


XGC_lda.fit(X_train_lda,y_train)
y_pred = XGC_lda.predict(X_test_lda)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
cap = CAP_performance(y_test,y_pred,plot=True)


# In[ ]:


performances["Method"].append("XGB LDA")
performances["Healthy_Recall"].append(recall_score(y_test,y_pred,average=None)[0])
performances["Disease_Recall"].append(recall_score(y_test,y_pred,average=None)[1])
performances["CAP"].append(cap)


# # Analyzing the Results

# In[ ]:


performances_df = pd.DataFrame(performances)
performances_df


# In[ ]:


plt.figure(figsize=(5,15))
plt.subplot(3,1,1)
plt.plot(performances_df["Healthy_Recall"])
plt.xticks(performances_df.index,performances_df["Method"],rotation='vertical')
plt.xlabel("Models")
plt.ylabel("Recall of Healthy Data Points")
plt.grid()
plt.subplot(3,1,2)
plt.plot(performances_df["Disease_Recall"])
plt.xticks(performances_df.index,performances_df["Method"],rotation='vertical')
plt.xlabel("Models")
plt.ylabel("Recall of Diseased Data Points")
plt.grid()
plt.subplot(3,1,3)
plt.plot(performances_df["CAP"])
plt.xticks(performances_df.index,performances_df["Method"],rotation='vertical')
plt.xlabel("Models")
plt.ylabel("Cumulative Accuracy Profile")
plt.grid()
plt.tight_layout(.5)
plt.show()

