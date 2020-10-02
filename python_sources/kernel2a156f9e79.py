# %% [code] {"id":"Ccdrntyekj2h"}
import pandas as pd

# %% [code] {"id":"6GPHIAN9km3c"}
df = pd.read_csv("../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv")

# %% [code] {"id":"anj7QNmkkth0","outputId":"242bf688-4333-4bef-f12f-cfe3c04a85b0"}
df.head()

# %% [code] {"id":"rLQUxIn7kwkb","outputId":"e4b1112f-f393-4157-bf87-18cd236f72c9"}
df.info()

# %% [code] {"id":"hXzBlPSck1CD","outputId":"1087fde6-9040-4b8a-e2f3-b35da0157841"}
df.describe()

# %% [code] {"id":"-FdfkAZLk6ps","outputId":"3e8857aa-f517-4979-da31-bdd7a8207ffc"}
df.isnull().any()

# %% [code] {"id":"JzZJ97lZlTSa","outputId":"96e2218d-1a03-4619-9364-2f2b5452ba26"}
df.isnull().sum()

# %% [code] {"id":"WkQUyKqxlc1b"}
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)


# %% [code] {"id":"Jrjr8ElVqjmJ"}
df['Married'].fillna(df['Married'].mode()[0], inplace=True)


# %% [code] {"id":"9c9b8VvYqz_2"}
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)

# %% [code] {"id":"EzSBHVwKrDjM"}
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace=True)

# %% [code] {"id":"hCNzxAYcri47"}
df['LoanAmount'].fillna(
df['LoanAmount'].dropna().median(),inplace=True)

# %% [code] {"id":"nSclyvq2r_Xb"}
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0],inplace=True)

# %% [code] {"id":"AWwE7w1RsI67"}
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)


# %% [code] {"id":"XFhAPFGQtpgR","outputId":"78d06283-5497-4753-f574-c33f4e21f6b4"}
df.isnull().sum()

# %% [code] {"id":"3VX7_A4zttGH","outputId":"e7b74f04-38ca-43d4-bf4e-b19850f0acbc"}
df.duplicated().any()

# %% [code] {"id":"vtXSpWBZyhhm","outputId":"4992f1d7-5069-4c05-e30f-7ef74a1a2824"}
import numpy as np
import matplotlib.pyplot as plt
import seaborn as se

# %% [code] {"id":"yEpSiXYuy38U","outputId":"9f18de99-6dfc-4c34-9eef-04f47a520497"}
plt.figure(figsize=(8,6))
df['Gender'].value_counts().plot(kind='bar', color = ('grey', 'black'))
plt.ylabel('Number of data points')
plt.xlabel('Gender')
plt.show()

# %% [code] {"id":"2VF0VexZzvXX","outputId":"d7531fb8-584c-40ef-ae5e-4eb476a3c666"}
plt.figure(figsize=(8,6))
df['Married'].value_counts().plot(kind='bar', color = ('grey', 'black'))
plt.ylabel('Number of data points')
plt.xlabel('Married')
plt.show()

# %% [code] {"id":"JMu-LSb30Rsj","outputId":"3f54dfc2-408d-4c4c-b567-e740fb24d983"}
plt.figure(figsize=(8,6))
df['Loan_Status'].value_counts().plot(kind='bar', color = ('grey', 'black'))
plt.ylabel('Number of data points')
plt.xlabel('Loan_Status')
plt.show()

# %% [code] {"id":"z48LH8yZ0mxy","outputId":"0bd9d2ed-17c0-4d57-beae-f474f2bcc302"}
plt.figure(figsize=(10,7))
se.distplot(df['LoanAmount'])
plt.show()

# %% [code] {"id":"5xSJYPih1zQL","outputId":"209f176e-392b-4378-a642-0b0998b86cf6"}
plt.figure(figsize=(10,7))
se.distplot(df['ApplicantIncome'])
plt.show()

# %% [code] {"id":"eVhYipDf2Mc6","outputId":"e6671409-0e1a-4779-8584-38133a1a5053"}
se.countplot(y=df['Gender'],hue=df['Education'])
plt.show()

# %% [code] {"id":"pKtQdOtL2_dn","outputId":"91b55233-7e51-4482-fad8-8126ea2e6fbc"}
df.dtypes

# %% [code] {"id":"OznTJfMF45Ak"}
df.drop(columns= ['Loan_ID'],axis=1,inplace=True)

# %% [code] {"id":"AeRvL9DA5M2K","outputId":"65b26e6e-689e-4067-d7ea-c1a4b59edabd"}
df.dtypes

# %% [code] {"id":"F3_AWV4Y5vrv","outputId":"c22ec240-08e9-4446-e632-7e358fe9d109"}
df['Dependents'].unique()

# %% [code] {"id":"YgNvpCzd6Mv6"}
code_numeric = {'Male': 1, 'Female': 2,
'Yes': 1, 'No': 2,
'Graduate': 1, 'Not Graduate': 2,
'Urban': 3, 'Semiurban': 2,'Rural': 1,
'Y': 1, 'N': 0,
'3+': 3}

# %% [code] {"id":"r7hGvINC6ZrD"}
df_train = df.applymap(lambda s: code_numeric.get(s) if s in code_numeric else s)

# %% [code] {"id":"bpFPG045_jFC","outputId":"ccae815d-1ad1-4d26-9109-698c4ae1fffc"}
df_train.dtypes

# %% [code] {"id":"FmbDR9E7_mfU"}
dep = pd.to_numeric(df_train['Dependents'])

# %% [code] {"id":"QE8GUopo_8xq"}
df_train.drop(['Dependents'],axis=1,inplace=True)

# %% [code] {"id":"CmrgWLatAPT8"}
df_train = pd.concat([df_train,dep],axis=1)

# %% [code] {"id":"gceaJfzlAhOn","outputId":"03de7def-486a-4c3f-d172-c0a9447f5d98"}
df_train.dtypes

# %% [code] {"id":"LijLRtSBAqkD","outputId":"8c87b2b9-95cb-40e4-862e-262042046e47"}
plt.figure(figsize=(15,6))
se.heatmap(df_train.corr(),annot=True, cmap="RdBu")
plt.show()

# %% [code] {"id":"JfoyRktRA3xC","outputId":"4d9bf1bb-cd2f-4a0e-c176-dffe99a1feaa"}
plt.figure(figsize=(12,9))
plt.scatter(x=df_train['ApplicantIncome'], y=df_train['LoanAmount'])
plt.xlabel('ApplicantIncome')
plt.ylabel('LoanAmount')
plt.show()

# %% [code] {"id":"nhw6KQ3oCyfk"}
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# %% [code] {"id":"j5sO4NihEKBV"}
y = df_train['Loan_Status']
X = df_train.drop('Loan_Status', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# %% [code] {"id":"932wmhwyET7r"}
model = LogisticRegression()

# %% [code] {"id":"NUzfGDCvEokP","outputId":"8f0eb46e-1c25-4bd1-fedc-c8f14ffdded2"}
model.fit(X_train,y_train)

# %% [code] {"id":"9johr7jDE86d"}
ypred = model.predict(X_test)

# %% [code] {"id":"_lI5lOXPFbCk"}
evaluation = f1_score(y_test, ypred)

# %% [code] {"id":"HN_3Ey8uFlFd","outputId":"9fe0b7c3-891b-4938-8a98-dd31a830aeda"}
evaluation

# %% [code] {"id":"tNoVwYFxFnOo"}
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score, f1_score)


# %% [code] {"id":"1dQ4OZtuF_Zl","outputId":"492f7e46-41e4-4b32-a0f6-5a26e1b69a22"}
print('Accuracy Score = {}'.format(accuracy_score(y_test, ypred)))

# %% [code] {"id":"AWQY7E9nGLFY","outputId":"50962885-a699-4d4a-a979-fdf303c9d83f"}
print(confusion_matrix(y_pred=ypred, y_true=y_test))

# %% [code] {"id":"T0r3VzYlGVMk","outputId":"75c37cd9-c894-4f16-f512-4794fee28162"}
print('Precision Score = {}'.format(precision_score(y_test, ypred)))
print('Recall Score = {}'.format(recall_score(y_test, ypred)))

# %% [code] {"id":"9zov5HrZHGnn"}
rfc = RandomForestClassifier(n_estimators=100, random_state=100)


# %% [code] {"id":"w9Oui6mjHaIE"}
from scipy import stats
n_features = X_train.shape[1]
param_dist = {
    'criterion': ['gini', 'entropy'],
    'max_features': stats.randint(low=1, high=n_features)
}
rscv = RandomizedSearchCV(estimator=rfc, param_distributions=param_dist, n_iter=50, cv=5, scoring='accuracy' , random_state=100)

# %% [code] {"id":"FjVDevxjKTmX","outputId":"db9c48ee-9b63-499b-ae59-13079bd94595"}
rscv.fit(X_train,y_train)

# %% [code] {"id":"UCGFPvNcMgl0"}
y_p = rscv.predict(X_test)

# %% [code] {"id":"g4DHHV6bM5tM","outputId":"0958d6cf-08b8-4ff7-a7d7-b3262c279f94"}
f1_score(y_test,y_p)

# %% [code] {"id":"gq9GQMT0NAxb"}
