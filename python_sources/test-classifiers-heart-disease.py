#!/usr/bin/env python
# coding: utf-8

# **            Test of ML classifier using Heart Disease Data**
# 
# * Data Analysis
# * Data Visiualization
# * Feature Encoding
# * Feature Scaling
# * Logistic Regression
# * Stochastic Gradient Decsent Classifier
# * Random Forest Classifier
# * Support Vector Machine
# * Random Forest
# * Gradient Boosting Classifier
# * Voting Classifier

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[ ]:


data = pd.read_csv("../input/heart.csv")
data.head(3)


# **Data discription:
# **
# * age - age in years 
# * sex - (1 = male; 0 = female) 
# * cp - chest pain type 
# * trestbps - resting blood pressure (in mm Hg on admission to the hospital) 
# * chol - serum cholestoral in mg/dl 
# * fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false) 
# * restecg - resting electrocardiographic results 
# * thalach - maximum heart rate achieved 
# * exang - exercise induced angina (1 = yes; 0 = no) 
# * oldpeak - ST depression induced by exercise relative to rest 
# * slope - the slope of the peak exercise ST segment 
# * ca - number of major vessels (0-3) colored by flourosopy 
# * thal - 3 = normal; 6 = fixed defect; 7 = reversable defect 
# * target - have disease or not (1=yes, 0=no)
# 
# There are several categorical fetures, so we have to be careful on how to deal with them.

# In[ ]:


print(data.info())


# In[ ]:


import seaborn as sns

corr_matrix = data.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr_matrix,annot=True,cmap="coolwarm",fmt=".2f",annot_kws={'size':16})
corr_matrix["target"].sort_values(ascending=False)


# Based on correlation matrix figure, we do quick data analysis with visualization.

# In[ ]:


plt.figure(figsize=(10,5))
ax1 = plt.subplot(1, 2, 1)
ax = sns.scatterplot(x="thalach", y="trestbps", hue="target",data=data)

ax1 = plt.subplot(1, 2, 2)
ax = sns.scatterplot(x="thalach", y="oldpeak", hue="target",data=data)


# The figure left shows more overlap compared to the right. This shows trestbps(-0.14) is weak feature compared to oldpeak(-0.43). It also shows due to low correlation values we have to use higher dimention features do better classification.

# In[ ]:


plt.figure(figsize=(10,5))
ax1 = plt.subplot(1, 2, 1)
sns.distplot(data["thalach"][data["target"]==0], label="Negative")
sns.distplot(data["thalach"][data["target"]==1], label="Positive")
plt.ylabel("density")
plt.xlabel("maximum heart rate achieved")
plt.legend()
ax2 = plt.subplot(1, 2, 2)
ax = sns.scatterplot(x="thalach", y="target", hue="target",data=data)
plt.xlabel("maximum heart rate achieved")
plt.show()


# Feature "thalach" a.k.a maximum heart rate achieved is interesting feature. As you can see, the person with higher max heart rate is more likely to diagonsed with heart disease.

# In[ ]:


sns.catplot(x="sex", hue = "target",kind="count", data=data);
plt.xlabel("Sex (0:Female, 1:Male)")
plt.title("Heart Disease (Target 0:Positive, 1:Negitave)")
plt.show()


#  Based on sex, female is more likely diagnosed with heart disease than male.

# In[ ]:


sns.catplot(x="cp",hue="target",kind = "count",data=data)
plt.xlabel("Chest Pain (0:Female, 1:Male)")
plt.title("Heart Disease (Target 0:Positive, 1:Negitave)")
plt.show()


# Even chest pain values looks as numerical, but it positvely correlates with target. It acts like continous variables as higher the chest pain person is more likely to be diagnosed with hear disease which makes sense. This is one of the reason I did not add this feature to categorical.

# In[ ]:


X = data.iloc[:,:-1].values
Y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2, random_state = 5)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
log_clf = LogisticRegression(solver="lbfgs")
log_clf.fit(x_train,y_train)
y_logclf_pred = log_clf.predict(x_test)
print(accuracy_score(y_test,y_logclf_pred))


# The first classifier is linear regression without any data preprocessing. Note: Linear regression does not need feature scaling, thats why it still works well.

# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_logclf_pred))
print(classification_report(y_test,y_logclf_pred))


# Full classifier performance analysis. It is worth to note one can calculate precision, recall, and f1-score from confusion matrix.

# In[ ]:


#Encoding categorical features
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
fe1_1hot = encoder.fit_transform(data['thal'].values.reshape(-1,1))
fe2_1hot = encoder.fit_transform(data['ca'].values.reshape(-1,1))
fe3_1hot = encoder.fit_transform(data['slope'].values.reshape(-1,1))

datadrop = data.drop(columns=["thal","ca","slope"])
X_new = datadrop.iloc[:,:-1].values
Y_new = datadrop.iloc[:,-1].values
X_new = np.concatenate((X_new,fe1_1hot.toarray(),fe2_1hot.toarray(),fe3_1hot.toarray()),axis=1)


# In[ ]:


from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=3)
sgd_clf.fit(x_train,y_train)
y_sgd_clf_pred = sgd_clf.predict(x_test)
print(accuracy_score(y_test,y_sgd_clf_pred))


# Wow, the SGD accuracy is only 70.5%?! Why?! Again, we have to do feature scaling since a lot of algorithms reuqires this step.

# In[ ]:


from sklearn.preprocessing import StandardScaler
stand_sca = StandardScaler()
X_trans = stand_sca.fit_transform(X_new)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X_trans,Y_new,test_size = 0.2, random_state = 5)
sgd_clf = SGDClassifier(random_state=3,n_jobs=-1)
sgd_clf.fit(x_train,y_train)
y_sgd_clf_pred = sgd_clf.predict(x_test)
print(accuracy_score(y_test,y_sgd_clf_pred))


# In[ ]:


plt.scatter([0,1],[100*0.704,100*0.868])
plt.ylabel("accuracy")
plt.annotate('Before Feature Scaling', xy=(0, 72), xytext=(0, 75), fontsize = 12,
            arrowprops=dict(facecolor='grey', shrink=0.05, linewidth = 2))

plt.annotate('After Feature Scaling', xy=(1, 85), xytext=(0.6, 82.5), fontsize = 12,
            arrowprops=dict(facecolor='grey', shrink=0.05, linewidth = 2))

plt.show()


# After feature scaling , SGD classification accuracy increased by nearly 17%!

# In[ ]:


def testclassfiersgd(max_in,tol_in):
    sgd_clf = SGDClassifier(random_state=3,n_jobs=-1,max_iter=max_in,tol=tol_in)
    sgd_clf.fit(x_train,y_train)
    y_sgd_clf_pred = sgd_clf.predict(x_test)
    acc = accuracy_score(y_test,y_sgd_clf_pred)
    return acc

max_iter_test = [10,50,100,500,1000,3000]
tol = [10,1,0.1,0.2,0.3]

acc_matrix_sgd = np.zeros((len(max_iter_test),len(tol)))

for i in range(len(max_iter_test)):
    #print(max_iter_test[i])
    for j in range(len(tol)):
        #print(tol[j])
        acc_matrix_sgd[i,j] = testclassfiersgd(max_iter_test[i],tol[j])

sns.heatmap(acc_matrix_sgd,annot=True,cmap="coolwarm",fmt=".2f",annot_kws={'size':16})
plt.ylabel("max_iter")
plt.xlabel("tol")
plt.show()


# This is my simple version of hyper parameters tuning, we can also use sklearn grid search to find best parameters.

# In[ ]:


log_clf = LogisticRegression(solver="lbfgs")
log_clf.fit(x_train,y_train)
y_logclf_pred = log_clf.predict(x_test)
print("Logistic Regression accuracy",100*accuracy_score(y_test,y_logclf_pred))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=100,random_state=1)
rf_clf.fit(x_train,y_train)
y_rf_clf_pred = rf_clf.predict(x_test)
print("Random Forest accuracy",100*accuracy_score(y_test,y_rf_clf_pred))


# In[ ]:


from sklearn.svm import SVC
svc_clf = SVC(kernel="rbf")
svc_clf.fit(x_train,y_train)
y_svc_clf_pred = svc_clf.predict(x_test)
print("SVM accuracy",100*accuracy_score(y_test,y_svc_clf_pred))


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
grbt_clf = GradientBoostingClassifier(max_depth=2,n_estimators=100,random_state=1)
grbt_clf.fit(x_train,y_train)

errors = np.zeros((100,1))
i = 0
for y_pred in grbt_clf.staged_predict(x_test):
    errors[i] = accuracy_score(y_test,y_pred)
    i = i + 1
    #print(y_pred)

best_n_estimator = np.argmax(errors)

plt.plot(errors)
plt.xlabel('number of trees');plt.ylabel('accuracy');plt.show()

grbt_clf_best = GradientBoostingClassifier(max_depth=2,n_estimators=best_n_estimator+1)
grbt_clf_best.fit(x_train,y_train)
y_test_gbrt = grbt_clf_best.predict(x_test)

print("GBR accuracy is:",100*accuracy_score(y_test,y_test_gbrt))


# In[ ]:


plt.scatter([0,1,2,3,4],[86.8,93.4,86.8,90.1,91.8])
plt.ylabel("Accuracy");plt.title("Accuracy comparision of classifiers")
plt.xticks([0,1,2,3,4],("SGD","RF","LOG","GB","SVC"))
plt.show()


# By comparing one fold validation, random forest performed the best and SGD & Logistic Regression performed the worst. 
# Finally, lets put all together, perform 5 fold cross validations to see average performance of classifiers. 

# In[ ]:


from sklearn.model_selection import cross_validate
sgd_cval=cross_validate(sgd_clf,X_trans,Y_new,cv=5)
rf_cval=cross_validate(rf_clf,X_trans,Y_new,cv=5)
log_cval=cross_validate(log_clf,X_trans,Y_new,cv=5)
grbt_cval=cross_validate(grbt_clf_best,X_trans,Y_new,cv=5)
svc_cval=cross_validate(svc_clf,X_trans,Y_new,cv=5)


# In[ ]:


all_mod_cval=np.concatenate((sgd_cval["test_score"],rf_cval["test_score"],log_cval["test_score"],
               grbt_cval["test_score"],svc_cval["test_score"]),axis=0)


# This result is different than what we got from earlier one!

# In[ ]:


plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
sns.heatmap(all_mod_cval.reshape((5,5)),annot=True,cmap="coolwarm",fmt=".2f",annot_kws={'size':16})
plt.title("5 fold cross validation")
plt.yticks([0,1,2,3,4],("SGD","RF","LOG","GB","SVC"))
plt.subplot(1,2,2)
plt.scatter([0,1,2,3,4],[sgd_cval["test_score"].mean(),
                        rf_cval["test_score"].mean(),
                        log_cval["test_score"].mean(),
                        grbt_cval["test_score"].mean(),
                        svc_cval["test_score"].mean()])
plt.ylabel("Avg Accuracy");plt.title("Avg accuracy comparision of classifiers")
plt.xticks([0,1,2,3,4],("SGD","RF","LOG","GB","SVC"))
plt.show()


# Logistic regression has the highest average accuracy through 5 fold cross validation.
# 
# **End Note: The 5-fold cross validation shows different result than earlier one fold result(same random state) for five different classifiers.So it is important to point out one should perform cross validation to determine performance of the classifier. **

# In[ ]:




