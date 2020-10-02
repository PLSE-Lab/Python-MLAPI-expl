#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb


# In[ ]:


SEED = 1
sns.set(rc={'figure.figsize': (9, 6)})
sns.set_style('white')


# In[ ]:


# Load the data

df = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


df.info()


# ### Good News: There's no missing data

# # EDA

# In[ ]:


# Find the ratio of churned customers

print(df['Churn'].value_counts(normalize=False))
print('\n')
print(df['Churn'].value_counts(normalize=True))


# ### The distribution is imbalanced as ratio of churned customers is only 0.26

# In[ ]:


# Replace the text values in Churn column with boolean values of 0 and 1

df['Churn'].replace({'No': 0, 'Yes': 1}, inplace=True)


# In[ ]:


target_variable = 'Churn'


# In[ ]:


def plot_categorical_column(column_name):
    """
    A generic function to plot the distribution of a categorical column, and
    the ratio of Churn in each of the values of that column.
    """
    f, (ax1, ax2) = plt.subplots(2, figsize=(9, 12))
    sns.countplot(x=column_name, data=df, ax=ax1)
    sns.pointplot(x=column_name, y=target_variable, data=df, ax=ax2)
    ax2.set_ylim(0, 0.5)


# In[ ]:


def plot_continuous_column(column_name):
    """
    A generic function to plot the distribution of a continuous column, and
    boxplot of that column for each value of Churn
    """
    f, (ax1, ax2) = plt.subplots(2, figsize=(9, 12))
    sns.distplot(df[column_name], ax=ax1)
    sns.boxplot(x='Churn', y=column_name, data=df, ax=ax2)


# ## Visualize gender column

# In[ ]:


plot_categorical_column('gender')


# ### Observation: The ratio of Female and Male is quite comparable, the churn probabilities are also almost the same in both the classes

# ## Visualize SeniorCitizen column

# In[ ]:


plot_categorical_column('SeniorCitizen')


# ### Observation: The churn probability of senior citizens is quite high with the prob being more than 0.4

# ## Visualize Partner column

# In[ ]:


plot_categorical_column('Partner')


# ### Observation: People without partners are more likely to churn

# ## Visualize Dependents column

# In[ ]:


plot_categorical_column('Dependents')


# ### Observation: Similar to partners, people without dependents are more likely to churn

# ## Visualize Tenure column

# In[ ]:


plot_continuous_column('tenure')


# ### Observation: As expected, people who have less tenure are more likely to churn. When people have been associated for long, they tend to stick around.

# ## Visualize PhoneService column

# In[ ]:


plot_categorical_column('PhoneService')


# ### Observation: No real impact of PhoneService as most people are using it.

# ## Visualize MultipleLines column

# In[ ]:


plot_categorical_column('MultipleLines')


# ### Observation: Multiple Lines is increasing the churn, although the effect is small

# ## Visualize InternetService column

# In[ ]:


plot_categorical_column('InternetService')


# ### Observation: Internet Service is looking like a major reason for churn, as we can see that the churn rate for those people who don't use internet services is real low ~ 0.07, and it becomes more than 0.4 for people who are using Fiber optic.
# ### The company should really focus on improving its Fiber optic services.

# ## Visualize OnlineSecurity column

# In[ ]:


plot_categorical_column('OnlineSecurity')


# ### Observation: People who take the Online security service are less likely to churn. Company should promote this service, maybe give a discount also if need be, because it greatly improves the churn rate.

# ## Visualize OnlineBackup column

# In[ ]:


plot_categorical_column('OnlineBackup')


# ### Observation: Similar to the Online security service, Online backup service should also be promoted as this also improves in customer retention.

# ## Visualize DeviceProtection column

# In[ ]:


plot_categorical_column('DeviceProtection')


# ### Observation: DeviceProtection service helps in reducing the churn rate.

# ## Visualize TechSupport column

# In[ ]:


plot_categorical_column('TechSupport')


# ### Observation: TechSupport service also helps in reducing the churn rate.

# ## Visualize StreamingTV column

# In[ ]:


plot_categorical_column('StreamingTV')


# ### Observation: Not much impact of StreamingTV service on churn rate.

# ## Visualize StreamingMovies column

# In[ ]:


plot_categorical_column('StreamingMovies')


# ### Observation: Not much impact of StreamingMovies service on churn rate.

# ## Visualize Contract column

# In[ ]:


plot_categorical_column('Contract')


# ### Observation: As expected, people who opt for long contracts are less likely to churn as evident from the "Two year" contract, highest churn rate is from people who opt for monthly contract.
# ### If the company can manage to convert the monthly people to take the one-year/two-year contract, the churn rate can vastly come down.

# ## Visualize PaperlessBilling column

# In[ ]:


plot_categorical_column('PaperlessBilling')


# ### Observation: Seems that people who opt for paperless bills are less likely to churn. Maybe because people generally don't spend too much time on paperless bills, they do more on hard copies. This could be one theory, could be any other reason also.

# ## Visualize PaymentMethod column

# In[ ]:


plot_categorical_column('PaymentMethod')


# ### Observation: Churn rate is highest for Electronic check payment method. As expected, automatic payment methods have low churn rate as people care less when the amount is debited automatically.

# ## Visualize MonthlyCharges column

# In[ ]:


plot_continuous_column('MonthlyCharges')


# ### Observation: People who have comparatively high monthly charges are more likely to churn

# # Modeling

# ### Replace binary text values with numbers

# In[ ]:


binary_columns_replace_dict = {
    'gender': {
        'Female': 0,
        'Male': 1
    },
    'Partner': {
        'No': 0,
        'Yes': 1
    },
    'Dependents': {
        'No': 0,
        'Yes': 1
    },
    'PhoneService': {
        'No': 0,
        'Yes': 1
    },
    'MultipleLines': {
        'No phone service': 0,
        'No': 0,
        'Yes': 1
    },
    'OnlineSecurity': {
        'No internet service': 0,
        'No': 0,
        'Yes': 1
    },
    'OnlineBackup': {
        'No internet service': 0,
        'No': 0,
        'Yes': 1
    },
    'DeviceProtection': {
        'No internet service': 0,
        'No': 0,
        'Yes': 1
    },
    'TechSupport': {
        'No internet service': 0,
        'No': 0,
        'Yes': 1
    },
    'StreamingTV': {
        'No internet service': 0,
        'No': 0,
        'Yes': 1
    },
    'StreamingMovies': {
        'No internet service': 0,
        'No': 0,
        'Yes': 1
    },
    'PaperlessBilling': {
        'No': 0,
        'Yes': 1
    }
}

for binary_col in binary_columns_replace_dict:
    df[binary_col].replace(binary_columns_replace_dict[binary_col], inplace=True)


# In[ ]:


df.info()


# ### Create dummy variables

# In[ ]:


categorical_columns = ['InternetService', 'Contract', 'PaymentMethod']

for categorical_column in categorical_columns:
    dummy_df = pd.get_dummies(df[categorical_column], prefix=categorical_column, drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)


# ### Create new features

# In[ ]:


# Create a feature for the number of internet services used

df['internet_services_count'] = df['OnlineSecurity'] + df['OnlineBackup'] + df['DeviceProtection']                                 + df['TechSupport'] + df['StreamingTV'] + df['StreamingMovies']


# In[ ]:


# Create a feature for checking if the payment is automatic or not

df['is_payment_automatic'] = df['PaymentMethod'].isin(['Bank transfer (automatic)', 'Credit card (automatic)'])


# ### Scale the data

# In[ ]:


target_variable = 'Churn'


# In[ ]:


features = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'PaperlessBilling', 'MonthlyCharges', 'internet_services_count', 'is_payment_automatic'
]

for col in df.columns:
    if not col in features and col.startswith(('InternetService_', 'Contract_')):
        features.append(col)


# In[ ]:


features


# In[ ]:


scaler = MinMaxScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(df[features]))
scaled_df.columns = features
scaled_df[target_variable] = df[target_variable]


# ### Try running different models

# In[ ]:


def evaluate_models(df, features):
    """
    Evaluate different models on the passed dataframe using the given features.
    """
    
    # Create testing and training data
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target_variable], test_size=0.2, random_state=SEED
    )

    results = {} # to store the results of the models
    models = [
        ('lr', LogisticRegression(random_state=SEED)),
        ('lda', LinearDiscriminantAnalysis()),
        ('svm', SVC(random_state=SEED)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('nb', GaussianNB()),
        ('dt', DecisionTreeClassifier(random_state=SEED)),
        ('rf', RandomForestClassifier(random_state=SEED, n_estimators=100)),
        ('et', ExtraTreesClassifier(random_state=SEED, n_estimators=100)),
        ('gb', GradientBoostingClassifier(random_state=SEED, n_estimators=100)),
        ('ada', AdaBoostClassifier(random_state=SEED)),
        ('xgb', xgb.XGBClassifier(random_state=SEED))
        
    ]

    for model_name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = model.score(X_test, y_test)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        results[model_name] = (model, accuracy, f1, cm)
        
    sorted_results = sorted(results.items(), key=lambda x: x[1][1], reverse=True)
    for model_name, (model, accuracy, f1, cm) in sorted_results:
        print(model_name, accuracy, f1)
        
    return results


# In[ ]:


results = evaluate_models(scaled_df, features)


# ### XGBoost is giving the highest accuracy of 0.8176
# 
# ### Lets apply cross validation to it

# In[ ]:


model = xgb.XGBClassifier(random_state=SEED)
cross_val_scores = cross_val_score(model, scaled_df[features], scaled_df[target_variable], cv=5)
print(cross_val_scores.mean())


# ### Cross validation accuracy: 0.8054
