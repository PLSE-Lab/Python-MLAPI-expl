#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[ ]:


# read the data and look at the first 5 records
df = pd.read_csv("../input/diabetes.csv")
df.head()


# In[ ]:


df.info()
#All are numeric data type


# In[ ]:


print("Summary Statistics :\n",df.describe())
#All independent variables are continuous and dependednt variable(Class) is a categorical variable.


# In[ ]:


fig,ax = plt.subplots(figsize=(10,10)) #Returns a tuple. A handle to the figure and axes of the subplot.
ax.set_xticklabels(labels=df.columns,rotation=90)
ax.set_yticklabels(labels=df.columns)
ax.tick_params(length=5,width=1,labelsize='medium')
cax = ax.matshow(df.drop("Class",axis=1).corr()) # Plot between independent variables
fig.colorbar(cax)
# plt.matshow(df.corr())


# In[ ]:


#Check for null values
df.isna().sum(axis=0)


# In[ ]:


#Boxplots for the following columns:
# 'Plasma_glucose_concentration_2 hr',
#  'blood_pressure',
#  ' Triceps_skin_fold_thickness ',
#  ' Hr2_serum_insulin',
#  'BOI',
#  ' Diabetes_pedigree_function',
#  'Age'

for i in list(df.columns[1:8]):
    fig = plt.figure()
    plt.boxplot(df[i],sym='rx')       
    plt.title(i)


# Outliers in BOI and Plasma_glucose_concentration_2 hr look erroneous with a value of 0.
# Let's transform these extreme minimum values

# In[ ]:


def outliers_treamtment(x):
    if x.name in ['Plasma_glucose_concentration_2 hr','BOI']:
        print(x.name)
        x_quantile = x.quantile([0.25,0.5,0.75])
        IQR = x_quantile[0.75] - x_quantile[0.25]
        IQR_15 = 1.5*(x_quantile[0.75] - x_quantile[0.25])
        x_lower_whisker = x_quantile[0.25] - IQR_15
        x_upper_whisker = x_quantile[0.75] + IQR_15
        #I am going to handle outliers value of <=lower_whisker and not those 
        # which are above upper_whisker as
        #they indicate meaningful data in our dataset
        #rounding up to 5th percentile value
        print('Outliers value for the column ',x.name, 'are \n',x[x<x_lower_whisker])
        x[x<x_lower_whisker] = x.quantile(0.05)
        print('After treatment, number of outliers left \n',x[x<x_lower_whisker])
    return x


# In[ ]:


df=df.apply(func= outliers_treamtment, axis = 0) # Applying outliers_treatment


# In[ ]:


print(df.loc[df['BOI']==0,'BOI'].any()) #Any value=0 left in BOI column?
plt.boxplot(df['BOI'],sym='ro') #Verifying with boxplot
plt.show()


# In[ ]:


print(df.loc[df['Plasma_glucose_concentration_2 hr']==0,'Plasma_glucose_concentration_2 hr'].any())
plt.boxplot(df['Plasma_glucose_concentration_2 hr'],sym='ro')
plt.show()


# In[ ]:


# target variable % distribution
print(df['Class'].value_counts(normalize=True))


# In[ ]:


#Segregating dependent and inedependent columns

X = df.drop(columns='Class') # independent variables
Y = df['Class'] # dependent variable


# In[ ]:


# splitting into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.3,random_state=123)


# In[ ]:


#instantiate a logistic regression model, and fit
model = LogisticRegression()
model = model.fit(X_train, Y_train)


# In[ ]:


# predict class labels for the train set. The predict fuction converts probability values > .5 to 1 else 0
Y_pred = model.predict(X_test)


# In[ ]:


# generate evaluation metrics
print("Accuracy: ", metrics.accuracy_score(Y_test, Y_pred))


# In[ ]:


#Confusion Matrix
print(metrics.confusion_matrix(Y_test,Y_pred))


# In[ ]:


57/(57+31) #Recall. TP/TP+FN


# In[ ]:


57/(57+12) #Precision TP/TP+FP


# In[ ]:


#Classification Report
print(metrics.classification_report(Y_test,Y_pred))


# In[ ]:


Y_pred_prob = model.predict_proba(X_test) # Predicted Probabilities for the test class
 # 1st Column indicates probabilites for Class 0 and 2nd column for Class 1
#Extracting probabilities for the positive class
Y_pred_prob = Y_pred_prob[:,1]
Y_pred_prob


# In[ ]:


# extract false positive, true positive rate
fpr, tpr, thresholds = metrics.roc_curve(Y_test,Y_pred_prob)
roc_auc = metrics.auc(fpr, tpr)
print(f"Area under the ROC curve : {roc_auc}",)


# In[ ]:


plt.figure(dpi=150)
plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:0.2f})")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate',color='g')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
ax2 = plt.gca().twinx()
ax2.plot(fpr, thresholds, markeredgecolor='r',linestyle='dashed', color='r')
ax2.set_ylabel('Threshold',color='r')
plt.show()


# In[ ]:


roc = pd.DataFrame({'fpr':fpr,'tpr(Sensitivity/Recall)':tpr,'1-fpr (TNR/Specificity)' : (1-fpr),                        'threshold':thresholds})
#By looking at the above graph,trying to find a sweet spot between high tpr and low 
# fpr, by varying
# threshold between 0.25 and 0.5
print(roc.loc[((roc['threshold']>0.25) & (roc['threshold'] < 0.6)),:])


# **Remarks**
# 
# Rememeber when i said that by default, model predicts Class 1 (rounds predicted 
# probability to 1) whenever probability > 0.5 and Class 0 (rounds predicted probability to 0)
# whenever prob < 0.5, thereby implying a set threshold of 0.5.
# What was the recall, then? 
# ~ 64%. 
# If you see above, Recall is between 61 to 68% for a threshold 
# range of 0.49 to 0.53 which verifies the above statement.
# 
# Question: What threshold should we choose?
# 
# Answer: Depends on what matters to us. Are we focussing on Recall or Precision.
#     Here in this problem statement, False Positives i.e.
#     wrongly predicting diabetes for a healthy person will not matter as much 
#     as False Negative i.e. predicting healhty for a diabetic person.
#     So, we would aim to reduce False Negatives (Increase Recall/True Positive Rate) even if 
#     it costs
#     us high False Positives ( and thus reduces Precision)
# A threshold of 0.30, gives us TPR(Recall of 87.5%) at a FPR of 28%
