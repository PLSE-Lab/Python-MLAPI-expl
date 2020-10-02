#!/usr/bin/env python
# coding: utf-8

# <h5>Context</h5>
# This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. 
# The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic
# measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger 
# database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.
# 
# <h5>Content</h5>
# The datasets consists of several medical predictor variables and one target variable, Outcome. 
# Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.

# ## Feature Information
#     Pregnancies -- Number of times pregnant
#     Glucose-- Plasma glucose concentration a 2 hours in an oral glucose tolerance test,
#     BloodPressure-- Diastolic blood pressure (mm Hg),
#     SkinThickness -- Triceps skin fold thickness (mm),
#     Insulin--2-Hour serum insulin (mu U/ml),
#     BMI -- Body mass index (weight in kg/(height in m)^2),
#     DiabetesPedigreeFunction--Diabetes pedigree function(a function which scores likelihood of diabetes based on family history)
#     Age-- Age (years),
#     Outcome -- Class variable (0 or 1) 268 of 768 are 1, the others are 0

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns',None)


# In[ ]:


import plotly.offline as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.tools as tls
import plotly.figure_factory as ff
py.init_notebook_mode(connected=True)
import squarify


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


prima=pd.read_csv("/kaggle/input/prima-diabetes/prima_diabetes.csv")


# In[ ]:





# In[ ]:


prima.shape


# In[ ]:


prima.head()


# In[ ]:


list(prima.columns)


# In[ ]:


prima.describe()


# In[ ]:


## Data Exploration
#plt.figure(figsize=(12,5))
print(prima.corr()['Outcome'])
sns.heatmap(prima.corr(),annot=True)


# In[ ]:


prima.isnull().sum()


# In[ ]:


prima[prima['Glucose']==0]


# In[ ]:


prima[prima['BloodPressure']==0]


# In[ ]:


prima[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = prima[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)


# In[ ]:


prima.isnull().sum()


# In[ ]:


# Define missing plot to detect all missing values in dataset
def missing_plot(dataset, key) :
    null_feat = pd.DataFrame(len(dataset[key]) - dataset.isnull().sum(), columns = ['Count'])
    percentage_null = pd.DataFrame((len(dataset[key]) - (len(dataset[key]) - dataset.isnull().sum()))/len(dataset[key])*100, columns = ['Count'])
    percentage_null = percentage_null.round(2)

    trace = go.Bar(x = null_feat.index, y = null_feat['Count'] ,opacity = 0.8, text = percentage_null['Count'],  textposition = 'auto',marker=dict(color = '#7EC0EE',
            line=dict(color='#000000',width=1.5)))

    layout = dict(title =  "Missing Values (count & %)")

    fig = dict(data = [trace], layout=layout)
    py.iplot(fig)


# In[ ]:


# Plotting 
missing_plot(prima, 'Outcome')


# In[ ]:


prima['Glucose'].median()


# In[ ]:


prima['Glucose'].mean()


# In[ ]:


# patient who is suffering with dIABETES Will have more Glucose level.
# patient who is not suffering with dIABETES Will have less Glucose level.


# In[ ]:


prima.groupby('Outcome').agg({'Glucose':'median'}).reset_index()


# In[ ]:


def median_target(var):   
    temp = prima[prima[var].notnull()]
    temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
    return temp


# In[ ]:


median_target('Insulin')


# In[ ]:


prima.loc[(prima['Outcome'] == 0 ) & (prima['Insulin'].isnull()), 'Insulin'] = 102.5
prima.loc[(prima['Outcome'] == 1 ) & (prima['Insulin'].isnull()), 'Insulin'] = 169.5


# In[ ]:


median_target('Glucose')


# In[ ]:


prima.loc[(prima['Outcome'] == 0 ) & (prima['Glucose'].isnull()), 'Glucose'] = 107.0
prima.loc[(prima['Outcome'] == 1 ) & (prima['Glucose'].isnull()), 'Glucose'] = 140.0


# In[ ]:


median_target('SkinThickness')


# In[ ]:


prima.loc[(prima['Outcome'] == 0 ) & (prima['SkinThickness'].isnull()), 'SkinThickness'] = 27.0
prima.loc[(prima['Outcome'] == 1 ) & (prima['SkinThickness'].isnull()), 'SkinThickness'] = 32.0


# In[ ]:


median_target('BloodPressure')


# In[ ]:


prima.loc[(prima['Outcome'] == 0 ) & (prima['BloodPressure'].isnull()), 'BloodPressure'] = 70.0
prima.loc[(prima['Outcome'] == 1 ) & (prima['BloodPressure'].isnull()), 'BloodPressure'] = 74.5


# In[ ]:


median_target('BMI')


# In[ ]:


prima.loc[(prima['Outcome'] == 0 ) & (prima['BMI'].isnull()), 'BMI'] = 30.1
prima.loc[(prima['Outcome'] == 1 ) & (prima['BMI'].isnull()), 'BMI'] = 34.3


# In[ ]:


missing_plot(prima, 'Outcome')


# In[ ]:


plt.style.use('ggplot') # Using ggplot2 style visuals 
f, ax = plt.subplots(figsize=(11, 15))
ax.set_facecolor('#fafafa')
ax.set(xlim=(-.05, 200))
plt.ylabel('Variables')
plt.title("Overview Data Set")
ax = sns.boxplot(data = prima, 
  orient = 'h', 
  palette = 'Set2')


# In[ ]:


median_target('Glucose')


# In[ ]:


prima.head()


# In[ ]:


sns.distplot(prima['Age'])


# In[ ]:


# Standard Scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
var=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
prima[var] = scaler.fit_transform(prima[var])
prima.head()


# In[ ]:


labels = "Diabetes", "Non Diabetes"
plt.title('Diabetes Status')
plt.ylabel('Condition')
prima['Outcome'].value_counts().plot.pie(explode = [0, 0.25], autopct = '%1.2f%%',
                                                shadow = True, labels = labels)


# In[ ]:


prima.head()


# In[ ]:


y=prima['Outcome']
X=prima.drop('Outcome',axis=1)


# In[ ]:


# Splitting the data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# In[ ]:


y_train


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
print(logreg.intercept_)
print(logreg.coef_)
print(metrics.accuracy_score(y_train,logreg.predict(X_train)))


# In[ ]:


print(logreg.intercept_)


# In[ ]:


print(logreg.coef_)


# In[ ]:


metrics.accuracy_score(y_train,logreg.predict(X_train))


# In[ ]:


import statsmodels.api as sm
model = sm.GLM(y_train,(sm.add_constant(X_train)),family=sm.families.Binomial()).fit()
print(model.summary())


# In[ ]:


from sklearn import metrics
y_train_pred=model.predict(sm.add_constant(X_train))
y_train_pred=round(y_train_pred).astype('int')
metrics.accuracy_score(y_train,y_train_pred)


# In[ ]:


X_test.head()


# In[ ]:


y_test_pred=model.predict(sm.add_constant(X_test))


# In[ ]:


PredictedOutcome=pd.DataFrame(y_test_pred,columns=['PredicbtedOutcome'])


# In[ ]:


PredictedOutcome.head()


# In[ ]:


Test=pd.concat([X_test,PredictedOutcome],axis=1)


# In[ ]:


Test.describe()


# In[ ]:


Test.head()


# In[ ]:


Test['Outcome']=Test['PredicbtedOutcome'].apply(lambda x: 1 if x > 0.71 else 0)


# In[ ]:


Test.head()


# In[ ]:


Test.shape


# In[ ]:


sns.countplot(Test['Outcome'])


# In[ ]:


labels = "Diabetes", "Non Diabetes"
plt.title('Diabetes Status')
plt.ylabel('Condition')
Test['Outcome'].value_counts().plot.pie(explode = [0, 0.25], autopct = '%1.2f%%',
                                                shadow = True, labels = labels)


# In[ ]:


from sklearn import metrics
metrics.confusion_matrix(y_test,Test['Outcome'])


# In[ ]:


TN= 135
TP=38
FN=43
FP=15


# ###  Accuracy : (TP+TN)/(TP+FN+FP+TN)

# In[ ]:


Accuracy= (TP+TN)/(TP+FN+FP+TN)  
print(Accuracy)


# #### Precision : Precision is a measure that tells us what proportion of customers that we classified as diabetes, 
#      actually had diabetes. The predicted positives (Customers predicted as diabetes are TP and FP) and the customers 
#      actually  diabetes are TP.
# 
#     Precision = TP / (TP+FP) 

# In[ ]:


Precision = TP / (TP+FP) 
print(Precision)


# #### Recall or Sensitivity or Hit Rate or True Positive Rate:
#                Recall is a measure that tells us what proportion of customers that actually had churned and  was classified by the algorithm as churned. The actual positives (Customers who had churned are TP and FN) and the customers classified by the model as churn are TP. (Note: FN is included because the Person actually had a churned even though the model predicted otherwise).
# 
#         Recall = TP/(TP+FN) 
# 

# In[ ]:


Recall = TP/(TP+FN) 
print(Recall)


# ## Specificity or Selectivity or True Negative Rate:
#            Specificity is a measure that tells us what proportion of patient that did NOT have diabetes, were predicted by the           model as non-diabetes. The actual negatives (patient actually NOT churned are FP and TN) and the customers classified          by us not as not churned are TN. (Note: FP is included because the Person did NOT actually have churned even though            the model     predicted     as churned).
# 
#     Specificity = TN / (TN+FP) 
# 

# In[ ]:


Specificity = TN / (TN+FP)
print(Specificity)


# ### F1 Score 

# In[ ]:


F1_Score = 2 * Precision * Recall / (Precision + Recall)
print(F1_Score)


# In[ ]:


## False Positive Rate or Fall Out or Probability of False Alarm
FPR= FP / (FP+ TN )
print(FPR)


# In[ ]:


# False Negative Rate or Miss Rate:
FNR = FN/(FN+TP)
print(FNR)


# # Prevalence:
#       Prevalence = Actual Positive / Total Population
#                   =   TP + FN / TP + TN + FP + FN 
# 

# In[ ]:


Prevalence=   (TP + FN) / (TP + TN + FP + FN) 
print(Prevalence)


# ###  A receiver operating characteristics (ROC) : 
#     ROC graphs are two-dimensional graphs in which TPR is plotted on the Y axis and FPR is plotted on the
#     X axis. An ROC graph depicts relative tradeoffs between benefits (true positives) and costs (false positives).

# In[ ]:


FPR,TPR,thresholds=metrics.roc_curve(y_test,Test['Outcome'])


# In[ ]:


plt.plot(FPR,TPR)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("FPR  or (1- Specificity)")
plt.ylabel("TPR or Sensitivity")
plt.title("ROC - Receiver Operating Characteristics")


# In[ ]:


AUC=metrics.accuracy_score(y_test,Test['Outcome'])
print(AUC)


# In[ ]:


# ROC on Train data
FPR,TPR,thresholds=metrics.roc_curve(y_train,y_train_pred)
plt.plot(FPR,TPR)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("FPR  or (1- Specificity)")
plt.ylabel("TPR or Sensitivity")
plt.title("ROC - Receiver Operating Characteristics")


# In[ ]:


AUC=metrics.accuracy_score(y_train,y_train_pred)
print(AUC)


# ## Precision and Recall threshold

# In[ ]:


Test.head()


# In[ ]:


numbers = [float(x)/10 for x in range(11) ]
for i in numbers:
    Test[i]=Test.PredicbtedOutcome.map(lambda x: 1 if x > i else 0)
Test.head()


# In[ ]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(Test['Outcome'], Test[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[ ]:


# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# #### From the curve above, 0.7 is the optimum point to take it as a cutoff probability

# In[ ]:


Test['final_predicted'] = Test.PredicbtedOutcome.map( lambda x: 1 if x > 0.7 else 0)
Test.head()


# In[ ]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_test, Test['final_predicted'])


# In[ ]:


from sklearn.metrics import precision_recall_curve
p, r, thresholds = precision_recall_curve(y_test, Test['final_predicted'])
plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# #### Hence we can see decided threshold based on precision and recall is absolutely correct!!!!
