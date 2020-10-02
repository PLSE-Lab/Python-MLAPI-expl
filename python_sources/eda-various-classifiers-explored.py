#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[ ]:


df = pd.read_csv("../input/HR_comma_sep.csv")
print(df.shape)
df.head()


# In[ ]:


print(df["sales"].unique())
print(df["salary"].unique())


# In[ ]:


df.describe()


# In[ ]:


df["left"].value_counts()[1]/float(df["left"].value_counts().sum())


# In[ ]:


sns.pairplot(df,hue='left')


# From this pairplot, we have following observations:
# If the number of projects is too low(<=2), employees chances of leaving the company are high
# If the number of projects is too high(>=5), employees chances of leaving the company are high
# If monthly hours 200, employees chances of leaving are high
# Work accidents dont seem to have any corelation with the employee leaving the company
# employees chances of leaving the company are realtively high, when their number_years is between 3 and 5
# IF the last evaluation is too good or too bad, employees tend to leave the company more.
# From this it looks like tree based algorithms would work best on this problem than regression. Lets test!

# In[ ]:


df.groupby(["left","number_project"])["satisfaction_level"].count()


# In[ ]:


df.loc[df["number_project"] == 2,"Num_Project_cat"] = 0
df.loc[(df["number_project"] == 3) | (df["number_project"] == 4) | (df["number_project"] == 5),"Num_Project_cat"] = 1
df.loc[df["number_project"] > 4,"Num_Project_cat"] = 2
df.drop("number_project",axis=1)
df.head()


# In[ ]:


df.columns


# In[ ]:


sns.factorplot("sales", col="salary", data=df,kind="count",size=4, aspect=.7,hue="left")


# In[ ]:


sns.factorplot(x='time_spend_company', y="satisfaction_level", data=df,kind="box", aspect=1,hue="left")


# In[ ]:


print(set(df.sales))


# #Difference in satisfaction levels by department

# In[ ]:


sns.factorplot("left","satisfaction_level",
                   data=df, kind="box", col="sales", col_wrap=4, size=2, aspect=0.8)


# In[ ]:


sns.factorplot("left","satisfaction_level",
                   data=df, kind="box", col="salary", col_wrap=3, size=2, aspect=1)


# #Average statisfaction of employess in various departments

# In[ ]:


df.groupby(["sales", "left"])["satisfaction_level"].mean().plot(kind='barh',color=['b','r'])
plt.title("Average satisfaction level by Department")
plt.xlabel("Satisfaction Level")


# 

# In[ ]:


d = df.groupby(["time_spend_company", "left"])["satisfaction_level"].count()/df.groupby(["time_spend_company"])["satisfaction_level"].count()
d.xs(1, level='left').plot(kind='bar', color='b')

#print df.xs(10223,level='id')
#print d.xs(1, level='left')


# In[ ]:


sns.violinplot(x="satisfaction_level", y="sales", data=df, hue = 'left', scale= "count", 
               size=10, aspect = 1,  scale_hue=False)


# In[ ]:


sns.set_color_codes()
for val in set(df["salary"]):
    x = df.loc[df["salary"] == val, "satisfaction_level"]
    ax = sns.distplot(x)
plt.title("Satisfcation Level by salary band")


# #Let's convert categorical variables into dummies

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["sales"] = le.fit_transform(df["sales"])
df["salary"] = le.fit_transform(df["salary"])


# In[ ]:


sales_dummies = pd.get_dummies(df["sales"], drop_first=True, prefix="BU")
salary_dummies = pd.get_dummies(df["salary"], drop_first=True, prefix="SL")
df = pd.concat([df, sales_dummies, salary_dummies], axis=1)
df.drop(["sales","salary"], axis=1,inplace=True)
df.head()


# In[ ]:


X = pd.concat([df.ix[:,:5],  df.ix[:,7:]],axis=1)
y = df["left"]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.25, random_state =1)


# Let's try various classification algorithms!

# #Logistic Regression

# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[ ]:


y_pred = logreg.predict(X_test)
y_prob = logreg.predict_proba(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[ ]:


conmat = np.array(confusion_matrix(y_pred, y_test))
confusion_matrix = pd.DataFrame(conmat, index = ["Act_left", "Act_Not_left"], columns = ["Pred_left", "Pred_not_left"])
confusion_matrix


# In[ ]:


accuracy_score(y_test, y_pred)


# In[ ]:


logreg.coef_


# In[ ]:


from sklearn.metrics import roc_curve, auc
plt.style.use('seaborn-white')

Y_score = logreg.decision_function(X_test)

# For class malignant, find the area under the curve
FPR, TPR, THR = roc_curve(y_test, Y_score)
ROC_AUC = auc(FPR, TPR)

# Plot of a ROC curve for class 1 (has_cancer)
#plt.figure(figsize=[11,9])
plt.plot(FPR, TPR, label='ROC curve (area = %0.2f)' % ROC_AUC, linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (1-specificity)', fontsize=18)
plt.ylabel('True Positive Rate (sensitivity)', fontsize=18)
plt.title('ROC curve for HR Analytics', fontsize=18)
plt.legend(loc="lower right")
plt.show()


# 

# In[ ]:


#Lets do a Grid Search
from sklearn.model_selection import GridSearchCV
logreg = LogisticRegression()
C_vals= [11,12,13,14,15,16,17,18]
penalties = ['l1','l2']

gs = GridSearchCV(logreg, {'penalty':penalties, 'C': C_vals}, verbose = True, cv = 5, scoring = 'roc_auc')
gs.fit(X_train, y_train)


# In[ ]:


gs.best_params_


# In[ ]:


import itertools
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    return


# In[ ]:


from sklearn.metrics import classification_report
gs_logreg = LogisticRegression(C=gs.best_params_["C"], penalty=gs.best_params_["penalty"], solver='liblinear')
gs_logreg.fit(X_train, y_train)
Y_ = gs_logreg.predict(X_test)


# In[ ]:


Y_score = gs_logreg.decision_function(X_test)

# For class malignant, find the area under the curve
FPR, TPR, THR = roc_curve(y_test, Y_score)
ROC_AUC = auc(FPR, TPR)

# Plot of a ROC curve for class 1 (has_cancer)
#plt.figure(figsize=[11,9])
plt.plot(FPR, TPR, label='ROC curve (area = %0.2f)' % ROC_AUC, linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (1-specificity)', fontsize=18)
plt.ylabel('True Positive Rate (sensitivity)', fontsize=18)
plt.title('ROC curve for HR Analytics', fontsize=18)
plt.legend(loc="lower right")
plt.show()


# #Model performace hasn't been improved much!

# In[ ]:


#Let's try tree based classifier - RandomForest


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=800,random_state=10)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
accuracy_score(y_test, rf_pred)


# In[ ]:


rf


# In[ ]:


from sklearn.metrics import confusion_matrix
conmat = np.array(confusion_matrix(rf_pred, y_test))
confusion_matrix = pd.DataFrame(conmat, index = ["Act_left", "Act_Not_left"], columns = ["Pred_left", "Pred_not_left"])
confusion_matrix


# In[ ]:


features = np.argsort(rf.feature_importances_)[::-1]
print('Feature ranking:')

for f in range(df.shape[1]-2):
    print('%d. feature %d %s (%f)' % (f+1 , features[f], df.columns[features[f]],
                                      rf.feature_importances_[features[f]]))


# That's pretty good! How about Gradient Boosting and XGB?

# 

# In[ ]:





# In[ ]:


#Lets try XGB 
import xgboost as xgb


# 

# In[ ]:


dtrain = xgb.DMatrix(X_train[:].astype(float), label=y_train.astype(int)) # construct from np.array
dtest = xgb.DMatrix(X_test[:].astype(float), label=y_test.astype(int))


# In[ ]:


param = {'max_depth':6, 'eta':0.01, 'silent':1, 'objective':'binary:logistic', 'eval_metric': 'auc' }# specify validations set to watch performance
watchlist = [(dtest,'eval'), (dtrain,'train')]


# In[ ]:


num_round = 100
clf = xgb.train(param, dtrain, num_round, watchlist,early_stopping_rounds=50)


# In[ ]:


xgb_preds = clf.predict(dtest)


# In[ ]:


xgb.plot_importance(clf)


# In[ ]:


xgb.to_graphviz(clf, num_trees=2)


# In[ ]:


xgb_pred = clf.predict(dtest)


# 

# 
