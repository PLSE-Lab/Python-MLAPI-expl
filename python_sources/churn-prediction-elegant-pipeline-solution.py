#!/usr/bin/env python
# coding: utf-8

# # Framing the Problem
# Supervised Classification -> predict if a customer will churn or not

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option("display.max_columns", 500)


# ## Getting the data

# In[ ]:


df = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head()


# ## Investigating the data structure

# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.describe()


# TotalCharges column has some empty cells, so it's treated as an object (string). Let's replace these empty cells with NaNs and cast to float

# In[ ]:


df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan, regex=False).astype(float)


# In[ ]:


df.describe()


# In[ ]:


df["TotalCharges"].hist()


# Resembles a power law distribution (not exactly, but OK). We can apply log transform to it

# In[ ]:


df["TotalCharges"] = df["TotalCharges"].apply(lambda x: np.log1p(x))


# In[ ]:


df["Churn"].value_counts(normalize=True)


# Our target are most Nos, I will make sure to stratify every split from now on[](http://)

# In[ ]:


map_labels = {"No": 0,
              "Yes": 1}

df["Churn"] = df["Churn"].map(map_labels)


# In[ ]:


df.isnull().sum()


# We have 11 missing values on TotalCharges, we could drop these rows, but, since we have few data points, I will opt to impute with the median later.

# In[ ]:


df = df.drop(columns=["customerID"])


# ## Creating a test set
# Stratified sampling because the target is not balanced

# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=23)
for train_index, test_index in split.split(df, df["Churn"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]


# In[ ]:


X_train = strat_train_set.drop("Churn", axis=1)
y_train = strat_train_set["Churn"].copy()


# ## Looking for correlations

# In[ ]:


X_train.corr()


# - Tenure seems to be highly positive correlated with TotalCharges
# - MonthlyCharges and TotalCharges also have moderate positive correlation

# ### Creating the preprocessing pipeline
# - It's a good practice to scale the numerical features
# - We will encode the categorical features using OneHotEncoder. Keep in mind that this introduces multicollinearity, which can be an issue for certain methods (for instance, methods that require matrix inversion). If features are highly correlated, matrices are computationally difficult to invert, which can lead to numerically unstable estimates. To reduce the correlation among variables, we can simply remove one feature column from the one-hot encoded array.
# - Let's combine both transformations with ColumnTransformer from sklearn

# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


num_attribs = ["MonthlyCharges", "TotalCharges", "tenure"]
cat_attribs = ["SeniorCitizen", "gender", "Partner", "Dependents",
               "PhoneService", "MultipleLines", "InternetService", 'OnlineSecurity', 
               'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
               'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']


num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])


# ### Stratified k-fold cross validation
# 
# Experiments by Ron Kohavi on various real-world datasets suggest that **10-fold cross-validation** offers the best tradeoff between bias and variance ( A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection , Kohavi, Ron , International Joint Conference on Artificial Intelligence (IJCAI) , 14 (12): 1137-43, 1995 ).
# 
# A slight improvement over the standard k-fold cross-validation approach is stratified k-fold cross-validation, which can yield better bias and variance estimates, especially in cases of unequal class proportions

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


# In[ ]:


from sklearn.linear_model import LogisticRegression

clf_lr = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(solver="lbfgs", max_iter=300, class_weight="balanced"))])

scores = cross_val_score(clf_lr, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)
scores.mean()


# ### Plotting the learning curve
# By plotting the model training and validation accuracies as functions of the training dataset size, we can easily detect whether the model suffers from high variance or high bias, and whether the collection of more data could help to address this problem

# In[ ]:


from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(estimator=clf_lr,
                                                        X=X_train,
                                                        y=y_train,
                                                        train_sizes=np.linspace(0.1, 1.0, 10),
                                                        cv=skf,
                                                        scoring="roc_auc")


# In[ ]:


train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(train_sizes, train_mean,
          color='blue', marker='o',
          markersize=5, label='Training AUC')
plt.fill_between(train_sizes,
                  train_mean + train_std,
                  train_mean - train_std,
                  alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean,
          color='green', linestyle='--',
          marker='s', markersize=5,
          label='Validation AUC')
plt.fill_between(train_sizes,
                  test_mean + test_std,
                  test_mean - test_std,
                  alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training examples')
plt.ylabel('ROC_AUC')
plt.legend(loc='lower right')
plt.ylim([0.8, 0.9])
plt.show()


# By the learning curve, our model seems to be OK. If something it can be **underfitting**.
# 
# - We could try a more complex model, but risk overfitting the training data
# - We could remove the regularization parameter that is on by default on LogisticRegression -> L2 norm
# - Looks like our model won't improve if we collect more data
# 
# ### Next steps
# - Explore the data better and try feature engineering
# - Tune hyperparameters
# - Stacking
# 
# Since this is just a practice example, I won't bother much, let's just select the LogisticRegression and end here

# In[ ]:


final_model = clf_lr.fit(X_train, y_train)


# ## Finally testing

# In[ ]:


X_test = strat_test_set.drop("Churn", axis=1)
y_test = strat_test_set["Churn"].copy()


final_predictions = final_model.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

print(confusion_matrix(y_test, final_predictions))
print(accuracy_score(y_test, final_predictions))
print(roc_auc_score(y_test, final_predictions))

