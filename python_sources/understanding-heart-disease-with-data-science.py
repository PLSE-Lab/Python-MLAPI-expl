#!/usr/bin/env python
# coding: utf-8

# # Understanding Heart Disease with Data Science
# 
# Heart disease is a term covering any disorder of the heart. It could be Congenital heart disease, Arrhythmia( irregular heartbeat), Coronary artery disease, Dilated cardiomyopathy, Myocardial infarction, Heart failure, Hypertrophic cardiomyopathy. To know more about heart disease in details you could refer to this website- [Heart Disease](https://www.medicalnewstoday.com/articles/237191.php)
# Lets see if Data Science help us understand and solve Heart disease problems.

# ![Heart Disease](https://www.hivplusmag.com/sites/hivplusmag.com/files/2017/06/22/heart-disease-750.jpg)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


Data = pd.read_csv("../input/heart.csv")
Data.head()


# In[ ]:


Data.isnull().sum()


# ## Data Visualization

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


f,ax = plt.subplots(figsize=(13,10))
sns.heatmap(Data.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)
plt.title("Correlation Matrix",fontsize=14)
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))
g=sns.barplot(Data["sex"],Data["target"],hue=Data["cp"],palette="rainbow",edgecolor='yellow')
plt.title("Target-0(SAFE), Target-1(UNSAFE)",fontsize=14)
plt.xlabel("SEX",fontsize=14)
plt.ylabel("Target",fontsize=14)
plt.xticks(np.arange(2),["FEMALE","MALE"])
plt.show()


# ## Conclusion
# * **Females with typical Angina(cp type - 0) have higher chances of Heart disease compared to Male.**
# * **Both Male and Female have higher chances of Heart Disease in case of atypical angina(cp type - 1) although females have higher chance compared to male if consided precisely.**
# * **Females are guaranteed to have Heart Disease in case of asymptomatic angina(cp type - 3) while Males have higher chances to have Heart Disease but are not guaranteed or assured to have it.**
# * **Any Kind of cheast pain should not be neglected in case of female, and doctors must be consulted immediately.**

# In[ ]:


plt.figure(figsize=(10,10))
g=sns.scatterplot(Data["age"],Data["trestbps"],hue=Data["sex"],size=Data["target"],size_order=[1,0],palette="copper_r",s=400)
#g=sns.scatterplot(Data["age"],Data["chol"],hue=Data["target"],size=Data["sex"],palette="copper_r",ax=ax[1],s=200)
plt.title("Target-0(SAFE), Target-1(UNSAFE)\nSex-0(Female) Sex-1(Male)",fontsize=14)
plt.xlabel("AGE",fontsize=14)
plt.ylabel("resting blood pressure",fontsize=14)
plt.grid(True)
plt.show()


# ## Conclusion
# * **Analyzing the patient with respect to blood pressure would be a bad criteria to judge the patient for having Heart Disease or not.**

# In[ ]:


plt.figure(figsize=(15,10))
g=sns.scatterplot(Data["age"],Data["chol"],hue=Data["sex"],size=Data["target"],palette="plasma_r",size_order=[1,0],s=800)
plt.title("Target-0(SAFE), Target-1(UNSAFE)\nSex-0(Female) Sex-1(Male)",fontsize=14)
plt.xlabel("AGE",fontsize=14)
plt.ylabel("serum cholestoral in mg/dl ",fontsize=14)
plt.ylim([100,450])
plt.grid(True)
plt.show()


# ## Conclusion
# * **For Age below 50 Females are more prone to heart disease compared to Males for high cholesterol.**
# * **For Age below 50 Females have high Chances of Heart Disease even for Cholesterol level below 190.**
# * **For Age between 50-60 Males are more prone to Heart Disease even for Cholesterol level below 190. One the other hand Males have also shown to have no heart Disease even with cholesterol level above 250.**
# * **For Age between 60-70 there is 50% chances that a Male or a Female might Heart Disease.**
# * **For age above 70 Females are assured to have a Heart Disease where as Males assured not to have it.**
# * **According to me if the age is above 50 then Cholesterol level would not be a strong criteria to judge a person having should have Heart Disease or not because there are cases where high cholesterol indicates absence of Heart Disease which should not have occured.**
# * **Males and Females should avoid junk and oily food and should exercize daily. Specially females should maintain diet and health.**
# * **It is necessary to maintaintain your diet and health after 50 especially men.**

# ### Target-0(SAFE), Target-1(UNSAFE)
# ### Sex-0(Female) Sex-1(Male)
# ### restecg: resting electrocardiographic results -- Value 0: normal -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV) -- Value 2: showing probable or definite left ventricular hypertrophy

# In[ ]:


plt.figure(figsize=(50,50))
g=sns.catplot("target","age",col="sex",hue="restecg",data=Data,palette="magma_r")
#plt.title("Target-0(SAFE), Target-1(UNSAFE)",fontsize=14)
#plt.xlabel("AGE",fontsize=14)
#plt.ylabel("Target",fontsize=14)
plt.show()


# ## Conclusion
# * **Patients showing probable or definite left ventricular hypertrophy are not confirmed to suffer from Heart Disease.**
# * **Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV) would be a bad criteria to judge the Presence of Heart Disease, the same is also in case of normal ST-T wave. **

# In[ ]:


plt.figure(figsize=(15,15))
g=sns.barplot("target","slope",hue="age",data=Data,palette="plasma_r",errwidth =0)
plt.xlabel("Target",fontsize=14)
plt.ylabel("Slope of ST segment",fontsize=14)
plt.xticks(np.arange(2),["SAFE","UNSAFE"])
plt.grid(True)
plt.show()


# ## Conclusion
# * **If The slope of the peak exercise ST segment is 1.00 or below then the patient could be assured safe from heart disease for any age group.**
# * **If The slope of the peak exercise ST segment is 1.50 or above then the patient have very high chances of having a heart disease irrespective of any age.**
# * **Slope of the S T segment could be used as a strong parameter to judge if a patient is suffering from Heart Disease or not. It could also be used along with Cholesterol level to judge if a patient could have Heart Disease or not after the age of 50.**

# In[ ]:


plt.figure(figsize=(15,15))
g=sns.barplot("age","cp",hue="target",data=Data,palette="plasma_r",errwidth =0)
plt.title("Target-0(SAFE), Target-1(UNSAFE)",fontsize=14)
plt.xlabel("AGE",fontsize=14)
plt.ylabel("(ca)number of major vessels (0-3) colored by flourosopy",fontsize=14)
plt.grid(True)
plt.show()


# ## Conclusion
# * **If the Value of ca is 1 or more then there is high chance of having a heart disease. **
# * **For patients above age of 50 this is a good parameter to judge if patient have heart disease because if value of ca is 1 or above strictly indicates the presence of heart disease.**

# # Feature Importance

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,auc,roc_curve


# In[ ]:


y = Data['target']
Data.drop("target", axis=1, inplace=True)
X = Data


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42)


# In[ ]:


Model = GradientBoostingClassifier(verbose=1, learning_rate=0.5,warm_start=True)
Model.fit(x_train, y_train)


# In[ ]:


# feature importance
print(Model.feature_importances_)


# In[ ]:


plt.figure(figsize=(15,15))
plt.bar(range(len(Model.feature_importances_)), Model.feature_importances_)
plt.title("Feature Importance")
plt.xticks(np.arange(13), Data.columns)
plt.grid(True)
plt.show()


# ## Conclusion
# * **ca test[ (ca)number of major vessels (0-3) colored by flourosopy] would be the best test to determine the presence of Heart Disease**
# * **Age and sex are not that much important to determine the presence of heart disease.**
# * **Although ECG graph  is the best way to know the condition of ones heart, but from the above graph we could see that the parameters related to ecg graph does not provide much strong support to declare whether one should have heart disease or not.**
# * **On the other hand ca, thal, cp tests provide much strong support to judge whether a person sould have heart disease or not.**

# ## Model Creation
# # Predicting if a patient is having Heart Disease or not.

# In[ ]:


y_pred = Model.predict(x_test)


# In[ ]:


print("Accuracy(GradientBoostingClassifier)\t:"+str(accuracy_score(y_test,y_pred)))
print("Precision(GradientBoostingClassifier)\t:"+str(precision_score(y_test,y_pred)))
print("Recall(GradientBoostingClassifier)\t:"+str(recall_score(y_test,y_pred)))


# In[ ]:


Model_2 = RandomForestClassifier(verbose=1,n_estimators=200,n_jobs=-1,warm_start=True)
Model_2.fit(x_train, y_train)


# In[ ]:


y_pred_2 = Model_2.predict(x_test)


# In[ ]:


print("Accuracy(RandomForestClassifier)\t:"+str(accuracy_score(y_test,y_pred_2)))
print("Precision(RandomForestClassifier)\t:"+str(precision_score(y_test,y_pred_2)))
print("Recall(RandomForestClassifier)\t:"+str(recall_score(y_test,y_pred_2)))


# In[ ]:


from xgboost import XGBClassifier
Model_3 = XGBClassifier()
Model_3.fit(x_train, y_train)


# In[ ]:


y_pred_3 = Model_3.predict(x_test)


# In[ ]:


print("Accuracy(XGBClassifier)\t:"+str(accuracy_score(y_test,y_pred_3)))
print("Precision(XGBClassifier)\t:"+str(precision_score(y_test,y_pred_3)))
print("Recall(XGBClassifier)\t:"+str(recall_score(y_test,y_pred_3)))


# In[ ]:


prob_1=Model.predict_proba(x_test)
prob_1 = prob_1[:,1]# Probalility prediction for GradientBoosting classifier
prob_2=Model_2.predict_proba(x_test)
prob_2 = prob_2[:,1]# Probalility prediction for Rangomforest classifier
prob_3=Model_3.predict_proba(x_test)
prob_3 = prob_3[:,1]# Probalility prediction for XGBoost classifier


# In[ ]:


fpr1, tpr1, _ = roc_curve(y_test, prob_1)
fpr2, tpr2, _ = roc_curve(y_test, prob_2)
fpr3, tpr3, _ = roc_curve(y_test, prob_3)
plt.figure(figsize=(14,12))
plt.title('Receiver Operating Characteristic',fontsize=14)
plt.plot(fpr1, tpr1, label = 'AUC(GradientBoosting Classifier) = %0.3f' % auc(fpr1, tpr1))
plt.plot(fpr2, tpr2, label = 'AUC(Randomforest Classifier) = %0.3f' % auc(fpr2, tpr2))
plt.plot(fpr3, tpr3, label = 'AUC(XGBoost Classifier) = %0.3f' % auc(fpr3, tpr3))
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.ylabel('True Positive Rate',fontsize=14)
plt.xlabel('False Positive Rate',fontsize=14)
plt.show()


# ## Note
# * **Every thing I have stated in this kernel are all according to my opinion and is totally based on my study over the datset with the help of data science and data analytics.**
# * **Please Comment down below and let me know your opinion about this kernel.**

# In[ ]:




