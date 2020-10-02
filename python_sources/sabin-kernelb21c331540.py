#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import DistanceMetric
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


#Loading the data and displaying them
customers_df = pd.read_csv("/kaggle/input/predicting-churn-for-bank-customers/Churn_Modelling.csv")
customers_df.head()


# In[ ]:


#Describing the information of the dataset to see the type, null and size of the columns
customers_df.info()


# In[ ]:


#Verifying and making sure that there is no nan in the dataset
customers_df.isna().sum()


# In[ ]:


#Looking for duplicates in the dataset
customers_df[customers_df.duplicated(keep="first")].count()


# # Data Distribution

# In[ ]:


exited_customers = customers_df[customers_df["Exited"] == 1]
remained_customers = customers_df[customers_df["Exited"] == 0]

labels = ['Exited', 'Stayed']
sizes = [len(exited_customers), len(remained_customers)]

explode = (0, 0.1)  
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  
plt.tight_layout()
plt.title("Cutomers who stayed and those who remained percentage")
plt.show()


# In[ ]:



exited_Male = exited_customers[exited_customers["Gender"] == "Male"]
remained_Male = remained_customers[remained_customers["Gender"] == "Male"]
exited_Female = exited_customers[exited_customers["Gender"] == "Female"]
remained_Female = remained_customers[remained_customers["Gender"] == "Female"]

group_names=['Exited', 'Stayed']
group_size=[len(exited_customers),len(remained_customers)]
subgroup_names=['Male', 'Female', 'Male', 'Female']
subgroup_size=[len(exited_Male),len(exited_Female),len(remained_Male),len(remained_Female)]
 
# Create colors
a, b, c=[plt.cm.Blues, plt.cm.Reds, plt.cm.Greens]
 
# First Ring (outside)
fig, ax = plt.subplots()
ax.axis('equal')
mypie, _ = ax.pie(group_size, radius=1.3, labels=group_names, colors=[ b(0.6),c(0.7)] )
plt.setp( mypie, width=0.3, edgecolor='white')
 
# Second Ring (Inside)
mypie2, _ = ax.pie(subgroup_size, radius=1.3-0.3, labels=subgroup_names, labeldistance=0.7, colors=[a(0.6),c(0.3),a(0.6),c(0.3)])
plt.setp( mypie2, width=0.4, edgecolor='white')
plt.margins(0,0)
plt.title("Distribution of Male and Female who exited/stayed")
 
# show it
plt.show()


# In[ ]:


#
fig, axarr = plt.subplots(2, 2, figsize=(20, 12))
sns.countplot(x='Geography', hue = 'Exited',data = customers_df, ax=axarr[0][0])
sns.countplot(x='Gender', hue = 'Exited',data = customers_df, ax=axarr[0][1])
sns.countplot(x='HasCrCard', hue = 'Exited',data = customers_df, ax=axarr[1][0])
sns.countplot(x='IsActiveMember', hue = 'Exited',data = customers_df, ax=axarr[1][1])


#  # The probability for customers to churn is:

# In[ ]:


exit_probability = len(exited_customers)/(len(exited_customers)+len(remained_customers))
exit_probability


# ### The probability for having a female exit

# In[ ]:


exited_Female = exited_customers[exited_customers["Gender"] == "Female"]
female_exit_probability = len(exited_Female)/len((customers_df[customers_df["Gender"] == "Female"]))
female_exit_probability


# ### The probability for having a female exit

# In[ ]:


exited_Male = exited_customers[exited_customers["Gender"] == "Male"]
male_exit_probability = len(exited_Male)/len((customers_df[customers_df["Gender"] == "Male"]))
male_exit_probability


# ### The probability for having a credit card owner exit

# In[ ]:


exited_Credit_card_owner = exited_customers[exited_customers["HasCrCard"] == 1]
credit_card_owner_exit_probability = len(exited_Credit_card_owner)/len((customers_df[customers_df["HasCrCard"] == 1]))
credit_card_owner_exit_probability


# ### The probability for having someone who doesn't have a credit card owner exit

# In[ ]:


exited_Credit_card_no_owner = exited_customers[exited_customers["HasCrCard"] == 0]
credit_card_no_owner_exit_probability = len(exited_Credit_card_no_owner)/len((customers_df[customers_df["HasCrCard"] == 0]))
credit_card_no_owner_exit_probability


# ### The probability for having an active customer exit

# In[ ]:


exited_active_customer = exited_customers[exited_customers["IsActiveMember"] == 1]
active_customer_exit_probability = len(exited_active_customer)/len((customers_df[customers_df["IsActiveMember"] == 1]))
active_customer_exit_probability


# ### The probability for having a non active customer exit

# In[ ]:


exited_non_active_customer = exited_customers[exited_customers["IsActiveMember"] == 0]
non_active_customer_exit_probability = len(exited_non_active_customer) / len((customers_df[customers_df["IsActiveMember"] == 0]))
non_active_customer_exit_probability


# ### The probability for having a French customer exit

# In[ ]:


exited_french_customer = exited_customers[exited_customers["Geography"] == "France"]
french_customer_exit_probability = len(exited_french_customer) / len((customers_df[customers_df["Geography"] == "France"]))
french_customer_exit_probability


# ### The probability for having a Spain customer exit

# In[ ]:


exited_spain_customer = exited_customers[exited_customers["Geography"] == "Spain"]
spain_customer_exit_probability = len(exited_spain_customer) / len((customers_df[customers_df["Geography"] == "Spain"]))
spain_customer_exit_probability 


# ### The probability for having a Germany customer exit

# In[ ]:


exited_germany_customer = exited_customers[exited_customers["Geography"] == "Germany"]
germany_customer_exit_probability = len(exited_germany_customer) / len((customers_df[customers_df["Geography"] == "Germany"]))
germany_customer_exit_probability 


# ### The probability for having a group exit by age

# In[ ]:


exited_Teenagers = exited_customers[exited_customers["Age"] < 20]
total_Teenagers = customers_df[customers_df["Age"] < 20]
print("Probability of tennagers to churn = ",len(exited_Teenagers)/len(total_Teenagers))
exited_adults_less_than_35 = exited_customers[(exited_customers["Age"] < 35) & (exited_customers["Age"] >= 20)]
total_adults_less_than_35 = customers_df[(customers_df["Age"] < 35) & (customers_df["Age"] >= 20)]
print("Probability of adults between 20 than 35 years to churn = ",len(exited_adults_less_than_35)/len(total_adults_less_than_35))
exited_adults_less_than_40 = exited_customers[(exited_customers["Age"] < 40) & (exited_customers["Age"] >= 35)]
total_adults_less_than_40 = customers_df[(customers_df["Age"] < 40) & (customers_df["Age"] >= 35)]
print("Probability of adults between 35 than 40 years to churn = ",len(exited_adults_less_than_40)/len(total_adults_less_than_40))
exited_adults_less_than_50 = exited_customers[(exited_customers["Age"] < 50) & (exited_customers["Age"] >= 40)]
total_adults_less_than_50 = customers_df[(customers_df["Age"] < 50) & (customers_df["Age"] >= 40)]
print("Probability of adults between 40 than 50 years to churn = ",len(exited_adults_less_than_50)/len(total_adults_less_than_50))
exited_adults_greater_than_50 = exited_customers[(exited_customers["Age"] > 50)]
total_adults_greater_than_50 = customers_df[(customers_df["Age"] > 50)]
print("Probability of adults over 50 years to churn = ",len(exited_adults_greater_than_50)/len(total_adults_greater_than_50))


# In[ ]:


#Correlation on the heatmap

corr = customers_df.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


# In[ ]:


#Displaying the correlation on a heatmap using different colors and annotations
fig, ax = plt.subplots(figsize=(20,10))
sns.heatmap(corr, annot=True, annot_kws={"size": 12},ax=ax)


# From the heatmap we can see that Age is positivily correlated with our target value but with a weak correlation. This is followed by the ActiveMember which is a weak negative correlation, followed by balance with 0.12. Balance and Number of products are also negatively correlated which we do not want. 

# We have seen that we have categorical values in our data, so we need to create dummy variables for models that can not work on categorical values such as logistics regression

# In[ ]:


# Creating dummy variables
gender_dummies = pd.get_dummies(customers_df["Gender"])
geography_dummies = pd.get_dummies(customers_df["Geography"])
# Adding them to our dataframe and removing unecessary dataframes
new_customers_df = pd.concat([customers_df,gender_dummies,geography_dummies],axis=1)
#Removing unecessary columns
new_customers_df = new_customers_df.drop(["RowNumber", "CustomerId","Surname","Geography","Gender"], axis=1)
new_customers_df.head()


# # Logistic regression

# In[ ]:


X = new_customers_df.drop(["Exited"],axis=1)
y = new_customers_df["Exited"]


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
#Splitting the training set and the testing set with 70 to 30 percentage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Building the logistic and fitting the model 
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[ ]:


#Predicting using our built model
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#Building and printing the confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion matrix for the logistic regression")
print(confusion_matrix)
print(classification_report(y_test, y_pred))


# From the confusion matrix of logistic regression, we can see that we have 2352 correct prediction and 621 incorect  predictions. 
# Recall is the number of true positive over the number of true negative divided by the number of true positive plus thhe number of false negative, is very low on the people who exited
# 
# Precision which is defined as the number true positives divided by the number of true positives plus the number of false positives is also low. 
# 
# F1 score which is the balance between precision and recall is also low for the exited customers.
# 
# These metrics will help us identify a better model beyond just looking at the accuracy

# In[ ]:





# In[ ]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# # Decision trees

# Decision trees may become more powerfull when it is pruned as when you prune, it reduces the size of the tree by removing sections of the tree that provide little power to classify instance, thus improves accuracy and reduces overfitting. We will try different trees while pruning on different depth with accuracy on our explanotory variable to find out the ideal depth to cut our tree

# In[ ]:


max_depths = [30,20,15,10,9,8,7,6,5,4,3,2,1]
models = [tree.DecisionTreeClassifier(max_depth=x).fit(X_train,y_train) for x in max_depths]
print(len(X_train))
print(len(X_test))

for m,d in zip(models,max_depths):
    print('depth = ',d,
          'train_accuracy = ',round(m.score(X_train,y_train),3),
          'valid_accuracy = ',round(m.score(X_test,y_test),3))


# #### If we prune the tree at the depth of 6 that's where we get the highest accuracy

# In[ ]:


# Building the model and pruning at depth 6 
DTreemodel = tree.DecisionTreeClassifier(max_depth=6).fit(X_train,y_train) 
# Printing the accuracy both training and testing
print('train_accuracy = ',round(DTreemodel.score(X_train,y_train),3),
          'valid_accuracy = ',round(DTreemodel.score(X_test,y_test),3))
# Predict on the testing set
y_pred_D_Tree = DTreemodel.predict(X_test)


# In[ ]:


# Calculating the random forest confusion matrix then the classification report
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred_D_Tree)
print("The confusion matrix is as follow:")
print(confusion_matrix)
print(classification_report(y_test, y_pred_D_Tree))


# Both the accuracy and confusion matrix for the decision tree is way better compared to our previous logistic regression model

# In[ ]:


# Area under the curve, ROC analysis

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
DT_roc_auc = roc_auc_score(y_test, DTreemodel.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, DTreemodel.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Decision Trees  (area = %0.2f)' % DT_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# # KNN

# The KNN is a non parametric model that with an algorithm thatassumes that similar things exist in close proximity. In order to find the perfect distance metrics and number of neighbors, we will loop through different neighborghs, and by the help of a graph, we will choose which one gives the greatest accuracy, then do the same using different distance metrics

# In[ ]:


# Looping through different number of neighbors
n = np.arange(1,21)
models = [KNeighborsClassifier(n_neighbors=x).fit(X_train,y_train) for x in n]
scores_train = [models[x-1].score(X_train,y_train) for x in n]
scores_valid = [models[x-1].score(X_test,y_test) for x in n]

fig, ax = plt.subplots(figsize=(8,8))
ax.plot(n,scores_train,label='train')
ax.plot(n,scores_valid,label='valid')
ax.grid(linestyle='--')
ax.legend()
ax.set_xticks(n)
ax.set_xlabel('n neighbors')
ax.set_ylabel('accuracy')
ax.set_title('Accuracy of KNN with number of neighbors')
plt.close('all')
fig


# In[ ]:


# Loop trough different distance metrics to find the best 
n = np.arange(1,21)

models_minkowski = [KNeighborsClassifier(
    n_neighbors=x,
    metric='minkowski').fit(X_train,y_train) for x in n]
models_euclidean = [KNeighborsClassifier(
    n_neighbors=x,
    metric='euclidean').fit(X_train,y_train) for x in n]
models_manhattan = [KNeighborsClassifier(
    n_neighbors=x,
    metric='manhattan').fit(X_train,y_train) for x in n]


valid_minkowski = [models_minkowski[x-1].score(X_test,y_test) for x in n]
valid_euclidean = [models_euclidean[x-1].score(X_test,y_test) for x in n]
valid_manhattan = [models_manhattan[x-1].score(X_test,y_test) for x in n]
train_manhattan = [models_manhattan[x-1].score(X_test,y_test) for x in n]

fig, ax = plt.subplots(figsize=(8,8))
ax.plot(n,valid_minkowski,label='minkowski')
ax.plot(n,valid_euclidean,label='euclidean')
ax.plot(n,valid_manhattan,label='manhattan')
ax.grid(linestyle='--')
ax.legend()
ax.set_xticks(n)
ax.set_xlabel('n neighbors')
ax.set_ylabel('accuracy')
ax.set_title('Accuracy of KNN Validation')

plt.close('all')
fig


# In[ ]:


valid_manhattan[19]


# In[ ]:


KNNmodel = KNeighborsClassifier(n_neighbors=20, metric='manhattan').fit(X_train,y_train)
y_pred_KNN = KNNmodel.predict(X_test)


# In[ ]:


# Calculating the KNN confusion matrix then the classification report
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred_KNN)
print("The confusion matrix is as follow:")
print(confusion_matrix)
print(classification_report(y_test, y_pred_KNN))


# The accuracy is better than logistic regression but not above the decision tree

# In[ ]:


# Area under the curve, ROC analysis

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
KNN_roc_auc = roc_auc_score(y_test, KNNmodel.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, KNNmodel.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='KNN  (area = %0.2f)' % KNN_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# # Ensembles for classification

# The ensembles uses use multiple learning algorithms to obtain better predictive performance than could be obtained using only one. We are going to look at the random forest which is an ensemble of decision trees. We will find out which number of trees is better by looping through different numbers

# In[ ]:


n = [1,10,50,80,100,200,300,400,450,500,550,600,650,700,800,900,1000]
models = [RandomForestClassifier(n_estimators=x,max_depth=4,random_state=3).fit(X_train,y_train) for x in n]


# In[ ]:


train_acc = [models[x].score(X_train,y_train) for x in range(len(n))]
valid_acc = [models[x].score(X_test,y_test) for x in range(len(n))]

fig, ax = plt.subplots(figsize=(8,8))
ax.plot(range(len(n)),train_acc,label='train')
ax.plot(range(len(n)),valid_acc,label='valid')
ax.set_xticks(range(len(n)))
ax.set_xticklabels(n)
ax.set_xlabel('trees')
ax.set_ylabel('accuracy')
ax.grid(linestyle='--')
ax.legend()
ax.set_title('Random Forest accuracy with different trees')

plt.close('all')
fig.savefig('rf.png',bbox_inches='tight')
fig


# ### Based on the graph,an esemble of trees(Random forest) of 1000 trees, provides the highest testing accuracy 

# In[ ]:


RF_model = RandomForestClassifier(n_estimators=1000,max_depth=6,random_state=3).fit(X_train,y_train)
y_pred_RF = RF_model.predict(X_test)


# In[ ]:


# Calculating the random forest confusion matrix then the classification report
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred_RF)
print("The confusion matrix is as follow:")
print(confusion_matrix)
print(classification_report(y_test, y_pred_RF))


# In[ ]:


# Area under the curve, ROC analysis

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
RF_roc_auc = roc_auc_score(y_test, RF_model.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, RF_model.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Random forest  (area = %0.2f)' % RF_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# In[ ]:




