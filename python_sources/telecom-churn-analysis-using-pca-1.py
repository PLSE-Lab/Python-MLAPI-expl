#!/usr/bin/env python
# coding: utf-8

# ## Telecom Churn: Logistic Regression with PCA
# 
# With 21 predictor variables, we need to predict whether a particular customer will switch to another telecom provider or not. In telecom terminology, customer attrition is referred to as 'churn'.

# ### Importing and Merging Data

# In[ ]:


# Importing Pandas and NumPy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Importing all datasets
churn_data = pd.read_csv("/kaggle/input/telecom-churn-analyis/customer_data.csv")
customer_data = pd.read_csv("/kaggle/input/telecom-churn-analyis/churn_data.csv")
internet_data = pd.read_csv("/kaggle/input/telecom-churn-analyis/internet_data.csv")


# In[ ]:


print(len(churn_data))
print(len(customer_data))
print(len(internet_data))


# In[ ]:


#Merging on 'customerID'
df_1 = pd.merge(churn_data, customer_data, how='inner', on='customerID')


# In[ ]:


#Final dataframe with all predictor variables
telecom = pd.merge(df_1, internet_data, how='inner', on='customerID')


# ### Let's understand the structure of our dataframe

# In[ ]:


# Let's see the head of our master dataset
telecom.head()


# In[ ]:


telecom.describe()


# ### Data Preparation

# In[ ]:


# Converting Yes to 1 and No to 0
telecom['PhoneService'] = telecom['PhoneService'].map({'Yes': 1, 'No': 0})
telecom['PaperlessBilling'] = telecom['PaperlessBilling'].map({'Yes': 1, 'No': 0})
telecom['Churn'] = telecom['Churn'].map({'Yes': 1, 'No': 0})
telecom['Partner'] = telecom['Partner'].map({'Yes': 1, 'No': 0})
telecom['Dependents'] = telecom['Dependents'].map({'Yes': 1, 'No': 0})


# ### Dummy Variable Creation

# In[ ]:


# Creating a dummy variable for the variable 'Contract' and dropping the first one.
cont = pd.get_dummies(telecom['Contract'],prefix='Contract',drop_first=True)
#Adding the results to the master dataframe
telecom = pd.concat([telecom,cont],axis=1)

# Creating a dummy variable for the variable 'PaymentMethod' and dropping the first one.
pm = pd.get_dummies(telecom['PaymentMethod'],prefix='PaymentMethod',drop_first=True)
#Adding the results to the master dataframe
telecom = pd.concat([telecom,pm],axis=1)

# Creating a dummy variable for the variable 'gender' and dropping the first one.
gen = pd.get_dummies(telecom['gender'],prefix='gender',drop_first=True)
#Adding the results to the master dataframe
telecom = pd.concat([telecom,gen],axis=1)

# Creating a dummy variable for the variable 'MultipleLines' and dropping the first one.
ml = pd.get_dummies(telecom['MultipleLines'],prefix='MultipleLines')
#  dropping MultipleLines_No phone service column
ml1 = ml.drop(['MultipleLines_No phone service'],1)
#Adding the results to the master dataframe
telecom = pd.concat([telecom,ml1],axis=1)

# Creating a dummy variable for the variable 'InternetService' and dropping the first one.
iser = pd.get_dummies(telecom['InternetService'],prefix='InternetService',drop_first=True)
#Adding the results to the master dataframe
telecom = pd.concat([telecom,iser],axis=1)

# Creating a dummy variable for the variable 'OnlineSecurity'.
os = pd.get_dummies(telecom['OnlineSecurity'],prefix='OnlineSecurity')
os1= os.drop(['OnlineSecurity_No internet service'],1)
#Adding the results to the master dataframe
telecom = pd.concat([telecom,os1],axis=1)

# Creating a dummy variable for the variable 'OnlineBackup'.
ob =pd.get_dummies(telecom['OnlineBackup'],prefix='OnlineBackup')
ob1 =ob.drop(['OnlineBackup_No internet service'],1)
#Adding the results to the master dataframe
telecom = pd.concat([telecom,ob1],axis=1)

# Creating a dummy variable for the variable 'DeviceProtection'. 
dp =pd.get_dummies(telecom['DeviceProtection'],prefix='DeviceProtection')
dp1 = dp.drop(['DeviceProtection_No internet service'],1)
#Adding the results to the master dataframe
telecom = pd.concat([telecom,dp1],axis=1)

# Creating a dummy variable for the variable 'TechSupport'. 
ts =pd.get_dummies(telecom['TechSupport'],prefix='TechSupport')
ts1 = ts.drop(['TechSupport_No internet service'],1)
#Adding the results to the master dataframe
telecom = pd.concat([telecom,ts1],axis=1)

# Creating a dummy variable for the variable 'StreamingTV'.
st =pd.get_dummies(telecom['StreamingTV'],prefix='StreamingTV')
st1 = st.drop(['StreamingTV_No internet service'],1)
#Adding the results to the master dataframe
telecom = pd.concat([telecom,st1],axis=1)

# Creating a dummy variable for the variable 'StreamingMovies'. 
sm =pd.get_dummies(telecom['StreamingMovies'],prefix='StreamingMovies')
sm1 = sm.drop(['StreamingMovies_No internet service'],1)
#Adding the results to the master dataframe
telecom = pd.concat([telecom,sm1],axis=1)


# ### Dropping the repeated variables

# In[ ]:


# We have created dummies for the below variables, so we can drop them
telecom = telecom.drop(['Contract','PaymentMethod','gender','MultipleLines','InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies'], 1)


# In[ ]:


#The varaible was imported as a string we need to convert it to float
telecom['TotalCharges'] =pd.to_numeric(telecom['TotalCharges'],errors='coerce')
#telecom['tenure'] = telecom['tenure'].astype(int).astype(float)


# In[ ]:


telecom.info()


# Now we can see we have all variables as integer.

# ### Checking for Outliers

# In[ ]:


# Checking for outliers in the continuous variables
num_telecom = telecom[['tenure','MonthlyCharges','SeniorCitizen','TotalCharges']]


# In[ ]:


# Checking outliers at 25%,50%,75%,90%,95% and 99%
num_telecom.describe(percentiles=[.25,.5,.75,.90,.95,.99])


# From the distribution shown above, you can see that there no outliner in your data. The numbers are gradually increasing.

# ### Checking for Missing Values and Inputing Them

# It means that 11/7043 = 0.001561834 i.e 0.1%, best is to remove these observations from the analysis

# In[ ]:


# Checking the percentage of missing values
round(100*(telecom.isnull().sum()/len(telecom.index)), 2)


# In[ ]:


# Removing NaN TotalCharges rows
telecom = telecom[~np.isnan(telecom['TotalCharges'])]


# In[ ]:


# Checking percentage of missing values after removing the missing values
round(100*(telecom.isnull().sum()/len(telecom.index)), 2)


# Now we don't have any missing values

# ### Feature Standardisation

# In[ ]:


# Normalising continuous features
df = telecom[['tenure','MonthlyCharges','TotalCharges']]


# In[ ]:


normalized_df=(df-df.mean())/df.std()
telecom = telecom.drop(['tenure','MonthlyCharges','TotalCharges'], 1)
telecom = pd.concat([telecom,normalized_df],axis=1)
telecom.head()


# ### Checking the Churn Rate

# In[ ]:


churn = (sum(telecom['Churn'])/len(telecom['Churn'].index))*100
churn


# We have almost 27% churn rate

# ## Model Building
# Let's start by splitting our data into a training set and a test set.

# ### Splitting Data into Training and Test Sets

# In[ ]:


from sklearn.model_selection import train_test_split

# Putting feature variable to X
X = telecom.drop(['Churn','customerID'],axis=1)

# Putting response variable to y
y = telecom['Churn']

y.head()


# In[ ]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7,test_size=0.3,random_state=100)


# ### Running Your First Training Model

# In[ ]:


import statsmodels.api as sm


# In[ ]:


# Logistic regression model
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# ### Correlation Matrix

# In[ ]:


# Importing matplotlib and seaborn
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Let's see the correlation matrix 
plt.figure(figsize = (20,10))        # Size of the figure
sns.heatmap(telecom.corr(),annot = True)


# ### Dropping highly correlated variables.

# In[ ]:


X_test2 = X_test.drop(['MultipleLines_No','OnlineSecurity_No','OnlineBackup_No','DeviceProtection_No','TechSupport_No','StreamingTV_No','StreamingMovies_No'],1)
X_train2 = X_train.drop(['MultipleLines_No','OnlineSecurity_No','OnlineBackup_No','DeviceProtection_No','TechSupport_No','StreamingTV_No','StreamingMovies_No'],1)


# ### Checking the Correlation Matrix

# After dropping highly correlated variables now let's check the correlation matrix again.

# In[ ]:


plt.figure(figsize = (20,10))
sns.heatmap(X_train2.corr(),annot = True)


# ### Re-Running the Model

# Now let's run our model again after dropping highly correlated variables

# In[ ]:


logm2 = sm.GLM(y_train,(sm.add_constant(X_train2)), family = sm.families.Binomial())
logm2.fit().summary()


# ### Feature Selection Using RFE

# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
from sklearn.feature_selection import RFE
rfe = RFE(logreg, 13)             # running RFE with 13 variables as output
rfe = rfe.fit(X,y)
print(rfe.support_)           # Printing the boolean results
print(rfe.ranking_)           # Printing the ranking


# In[ ]:


# Variables selected by RFE 
col = ['PhoneService', 'PaperlessBilling', 'Contract_One year', 'Contract_Two year',
       'PaymentMethod_Electronic check','MultipleLines_No','InternetService_Fiber optic', 'InternetService_No',
       'OnlineSecurity_Yes','TechSupport_Yes','StreamingMovies_No','tenure','TotalCharges']


# In[ ]:


# Let's run the model using the selected variables
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logsk = LogisticRegression(C=1e9)
#logsk.fit(X_train[col], y_train)
logsk.fit(X_train, y_train)


# In[ ]:


#Comparing the model with StatsModels
#logm4 = sm.GLM(y_train,(sm.add_constant(X_train[col])), family = sm.families.Binomial())
logm4 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
modres = logm4.fit()
logm4.fit().summary()


# In[ ]:


X_test[col].shape
#res = modres.predict(X_test[col])


# ### Making Predictions

# In[ ]:


# Predicted probabilities
y_pred = logsk.predict_proba(X_test)
# Converting y_pred to a dataframe which is an array
y_pred_df = pd.DataFrame(y_pred)
# Converting to column dataframe
y_pred_1 = y_pred_df.iloc[:,[1]]
# Let's see the head
y_pred_1.head()


# In[ ]:


# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)
y_test_df.head()


# In[ ]:


# Putting CustID to index
y_test_df['CustID'] = y_test_df.index
# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)
# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df,y_pred_1],axis=1)
# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 1 : 'Churn_Prob'})
# Rearranging the columns
y_pred_final = y_pred_final.reindex(['CustID','Churn','Churn_Prob'], axis=1)
# Let's see the head of y_pred_final
y_pred_final.head()


# In[ ]:


# Creating new column 'predicted' with 1 if Churn_Prob>0.5 else 0
y_pred_final['predicted'] = y_pred_final.Churn_Prob.map( lambda x: 1 if x > 0.5 else 0)
# Let's see the head
y_pred_final.head()


# ### Model Evaluation

# In[ ]:


from sklearn import metrics


# In[ ]:


# Confusion matrix 
confusion = metrics.confusion_matrix( y_pred_final.Churn, y_pred_final.predicted )
confusion


# In[ ]:


# Predicted     Churn  not_churn  __all__
# Actual
# Churn            1359   169     1528
# not_churn         256   326      582
# __all__          1615   751     2110


# In[ ]:


#Let's check the overall accuracy.
metrics.accuracy_score(y_pred_final.Churn, y_pred_final.predicted)


# In[ ]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(6, 6))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return fpr, tpr, thresholds


# In[ ]:


draw_roc(y_pred_final.Churn, y_pred_final.predicted)


# In[ ]:


#draw_roc(y_pred_final.Churn, y_pred_final.predicted)
"{:2.2f}".format(metrics.roc_auc_score(y_pred_final.Churn, y_pred_final.Churn_Prob))


# #### We see an overall AUC score of 0.83 looks like we did a decent job.
# - But we did spend a lot of effort on the features and their selection.
# - Can PCA help reduce our effort?

# ### PCA on the data

# #### Note - 
# - While computng the principal components, we must not include the entire dataset. Model building is all about doing well on the data we haven't seen yet!
# - So we'll calculate the PCs using the train data, and apply them later on the test data

# In[ ]:


X_train.shape
# We have 30 variables after creating our dummy variables for our categories


# In[ ]:


#Improting the PCA module
from sklearn.decomposition import PCA
pca = PCA(svd_solver='randomized', random_state=42)


# In[ ]:


#Doing the PCA on the train data
pca.fit(X_train)


# #### Let's plot the principal components and try to make sense of them
# - We'll plot original features on the first 2 principal components as axes

# In[ ]:


pca.components_


# In[ ]:


colnames = list(X_train.columns)
pcs_df = pd.DataFrame({'PC1':pca.components_[0],'PC2':pca.components_[1], 'Feature':colnames})
pcs_df.head()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize = (8,8))
plt.scatter(pcs_df.PC1, pcs_df.PC2)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
for i, txt in enumerate(pcs_df.Feature):
    plt.annotate(txt, (pcs_df.PC1[i],pcs_df.PC2[i]))
plt.tight_layout()
plt.show()


# We see that the fist component is in the direction where the 'charges' variables are heavy
#  - These 3 components also have the highest of the loadings

# #### Looking at the screeplot to assess the number of needed principal components

# In[ ]:


pca.explained_variance_ratio_


# In[ ]:


#Making the screeplot - plotting the cumulative variance against the number of components
get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize = (12,8))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


# #### Looks like 16 components are enough to describe 95% of the variance in the dataset
# - We'll choose 16 components for our modeling

# In[ ]:


#Using incremental PCA for efficiency - saves a lot of time on larger datasets
from sklearn.decomposition import IncrementalPCA
pca_final = IncrementalPCA(n_components=16)


# #### Basis transformation - getting the data onto our PCs

# In[ ]:


df_train_pca = pca_final.fit_transform(X_train)
df_train_pca.shape


# #### Creating correlation matrix for the principal components - we expect little to no correlation

# In[ ]:


#creating correlation matrix for the principal components
corrmat = np.corrcoef(df_train_pca.transpose())


# In[ ]:


#plotting the correlation matrix
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize = (20,10))
sns.heatmap(corrmat,annot = True)


# In[ ]:


# 1s -> 0s in diagonals
corrmat_nodiag = corrmat - np.diagflat(corrmat.diagonal())
print("max corr:",corrmat_nodiag.max(), ", min corr: ", corrmat_nodiag.min(),)
# we see that correlations are indeed very close to 0


# #### Indeed - there is no correlation between any two components! Good job, PCA!
# - We effectively have removed multicollinearity from our situation, and our models will be much more stable

# In[ ]:


#Applying selected components to the test data - 16 components
df_test_pca = pca_final.transform(X_test)
df_test_pca.shape


# #### Applying a logistic regression on our Principal Components
# - We expect to get similar model performance with significantly lower features
# - If we can do so, we would have done effective dimensionality reduction without losing any import information

# In[ ]:


#Training the model on the train data
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

learner_pca = LogisticRegression()
model_pca = learner_pca.fit(df_train_pca,y_train)


# **Note**
# 
# Note that we are fitting the original variable y with the transformed variables (principal components). This is not a problem becuase the transformation done in PCA is *linear*, which implies that you've only changed the way the new x variables are represented, though the nature of relationship between X and Y is still linear. 

# In[ ]:


#Making prediction on the test data
pred_probs_test = model_pca.predict_proba(df_test_pca)[:,1]
"{:2.2}".format(metrics.roc_auc_score(y_test, pred_probs_test))


# #### Impressive! The same result, without all the hard work on feature selection!

# Why not take it a step further and get a little more 'unsupervised' in our approach?
# This time, we'll let PCA select the number of components basen on a variance cutoff we provide

# In[ ]:


pca_again = PCA(0.90)


# In[ ]:


df_train_pca2 = pca_again.fit_transform(X_train)
df_train_pca2.shape
# we see that PCA selected 14 components


# In[ ]:


#training the regression model
learner_pca2 = LogisticRegression()
model_pca2 = learner_pca2.fit(df_train_pca2,y_train)


# In[ ]:


df_test_pca2 = pca_again.transform(X_test)
df_test_pca2.shape


# In[ ]:


#Making prediction on the test data
pred_probs_test2 = model_pca2.predict_proba(df_test_pca2)[:,1]
"{:2.2f}".format(metrics.roc_auc_score(y_test, pred_probs_test2))


# #### So there it is - a very similar result, without all the hassles. We have not only achieved dimensionality reduction, but also saved a lot of effort on feature selection.

# #### Before closing, let's also visualize the data to see if we can spot any patterns

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize = (8,8))
plt.scatter(df_train_pca[:,0], df_train_pca[:,1], c = y_train.map({0:'green',1:'red'}))
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.tight_layout()
plt.show()


# Looks like there is a good amount of separation in 2D, but probably not enough
# 
# Let's look at it in 3D, and we expect spread to be better (dimensions of variance, remember?)

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(8,8))
ax = Axes3D(fig)
# ax = plt.axes(projection='3d')
ax.scatter(df_train_pca[:,2], df_train_pca[:,0], df_train_pca[:,1], c=y_train.map({0:'green',1:'red'}))


# #### So let's try building the model with just 3 principal components!

# In[ ]:


pca_last = PCA(n_components=3)
df_train_pca3 = pca_last.fit_transform(X_train)
df_test_pca3 = pca_last.transform(X_test)
df_test_pca3.shape


# In[ ]:


#training the regression model
learner_pca3 = LogisticRegression()
model_pca3 = learner_pca3.fit(df_train_pca3,y_train)
#Making prediction on the test data
pred_probs_test3 = model_pca3.predict_proba(df_test_pca3)[:,1]
"{:2.2f}".format(metrics.roc_auc_score(y_test, pred_probs_test3))


# #### 0.82! Isn't that just amazing!

# In[ ]:


---------------------------------------#Thank You-------------------------------------------------------------------------------------------


# In[ ]:




