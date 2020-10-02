#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing  python  libraries
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


#Import data set
df = pd.read_csv('../input/heart.csv')


# In[ ]:


df.head(5)# this command shows first 5 rows of data frame

-------------------------Details of  Attributes of data frame-------------------------
age- in years
sex-(1 = male; 0 = female)
cp- chest pain type
trestbps- resting blood pressure (in mm Hg on admission to the hospital)
chol- serum cholestoral in mg/dl
fbs-(fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
restecg-resting electrocardiographic results
thalach-maximum heart rate achieved
exang-exercise induced angina (1 = yes; 0 = no)
oldpeak-ST depression induced by exercise relative to rest
slope-the slope of the peak exercise ST segment
ca-number of major vessels (0-3) colored by flourosopy
thal- 3 = normal; 6 = fixed defect; 7 = reversable defect
target- 1 or 0 . (0 refers no ,1 refers yes)
# In[ ]:


df.info()  


# In[ ]:


df['target'].value_counts()


# ### _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ __ _ _ _ _ _  _ _ _ _ _ _ _ _ _ _ _ _   _ _ _ ___ _ __ __ _ _ _ _ _ _ _  _ _ _ _ _   

# # Data Cleaning

# In[ ]:


#checking for Missing Data 

missing_data=df.isnull()
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")    


# # Conclusion : There is no missing values in data frame

# In[ ]:


#Correct data format

df.dtypes
#Conclusion:All the dtypes are correct format


# ### _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ __ _ _ _ _ _  _ _ _ _ _ _ _ _ _ _ _ _   _ _ _ ___ _ __ __ _ _ _ _ _ _ _  _ _ _ _ _   

# # Data Visualization

# In[ ]:


get_ipython().run_cell_magic('capture', '', '! pip install seaborn')


# In[ ]:


#correlation of independent variable and dependent variable 
df.corr()


# In[ ]:


df[['cp','target']].corr()


# In[ ]:


df.describe() # this will shows the descriptive statistics of data frame 


# In[ ]:


#Heat map
plt.figure(figsize=(15,7))
corr = df.corr()
sns.heatmap(corr, annot=True )


# In[ ]:


sns.distplot(df['age'],color='Red',hist_kws={'alpha':1,"linewidth": 2}, kde_kws={"color": "k", "lw": 3, "label": "KDE"})
#Most people age is from 40 to 60


# In[ ]:


fig,ax=plt.subplots(figsize=(16,6))
sns.pointplot(x='age',y='cp',data=df,color='Lime',hue='target',linestyles=["-", "--"])
plt.title('Age vs Cp')
#People with heart disease tend to have higher 'cphest pain' at all ages only exceptions at age 45 and 49


# In[ ]:


sns.countplot(x='ca',data=df,hue='target',palette='YlOrRd',linewidth=3)
# People with 'ca' as 0 have highest chance of heart disease


# ## Age vs target
# let's  calculate the pearsonr coefficient and p-value 

# In[ ]:



pearson_coef, p_value = stats.pearsonr(df['age'], df['target'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 


# ## Conclusion
# Since the p-value is $<$ 0.001, the correlation between age and target is statistically significant, and there is negative  linear relationship  (-0.225).

# ## sex vs target
# let's  calculate the pearsonr coefficient and p-value 

# In[ ]:


pearson_coef, p_value = stats.pearsonr(df['sex'], df['target'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 


# ## Conclusion
# Since the p-value is  <  0.001, the correlation between sex and target is statistically significant, and there is negative  linear relationship which  is very weak (-0.28).

# ## cp vs target
# let's  calculate the pearsonr coefficient and p-value 

# In[ ]:


pearson_coef, p_value = stats.pearsonr(df['cp'], df['target'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 


# ## Conclusion
# Since the p-value is  <  0.001, the correlation between cp  and target  is statistically significant, and the linear relationship is quite strong (~0.4332).

# ## trestbps vs Target
# let's  calculate the pearsonr coefficient and p-value 

# In[ ]:


pearson_coef, p_value = stats.pearsonr(df['trestbps'], df['target'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 


# ## Conclusion
# Since the p-value is  <  0.001, the correlation between trestbps and target is statistically significant, and there is negative  linear relationship which is very weak (~0.144).

# ## Chol vs target
# let's  calculate the pearsonr coefficient and p-value 

# In[ ]:


pearson_coef, p_value = stats.pearsonr(df['chol'], df['target'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 


# ## Conclusion
# Since the p-value is  >0.1, the correlation between chol and target is statistically unsignificant, and there is negative  linear relationship which is very strong (-0.854).

# ## fbs vs target
# let's  calculate the pearsonr coefficient and p-value 

# In[ ]:


pearson_coef, p_value = stats.pearsonr(df['fbs'], df['target'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 


# ## Conclusion
# Since the p-value is  >0.1, the correlation between fbs and target is statistically unsignificant, and there is negative  linear relationship which is very weak (~0.2).

# ## restecg vs target
# let's  calculate the pearsonr coefficient and p-value 

# In[ ]:


pearson_coef, p_value = stats.pearsonr(df['restecg'], df['target'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 


# ## Conclusion
# Since the p-value is  >  0.001, the correlation between restecg and target is statistically unsignificant, and there is linear relationship which is very weak (~0.137).

# ## thalach vs target
# let's  calculate the pearsonr coefficient and p-value 

# In[ ]:


pearson_coef, p_value = stats.pearsonr(df['thalach'], df['target'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 


# ## Conclusion
# Since the p-value is  <  0.001, the correlation between thalach and target is statistically significant, and there is  linear relationship which is stong (~0.42).

# ## exang vs target
# let's  calculate the pearsonr coefficient and p-value 

# In[ ]:


pearson_coef, p_value = stats.pearsonr(df['exang'], df['target'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 


# ## Conclusion
# Since the p-value is  <  0.001, the correlation between exang  and target is statistically significant, and there is negative  linear relationship which is strong (~0.43).

# ## oldpeak vs target
# let's  calculate the pearsonr coefficient and p-value 

# In[ ]:


pearson_coef, p_value = stats.pearsonr(df['oldpeak'], df['target'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 


# ## Conclusion
# Since the p-value is  <  0.001, the correlation between oldpeak and target is statistically significant, and there is negative  linear relationship which is strong (~0.43).

# ## slope vs target
# let's  calculate the pearsonr coefficient and p-value 

# In[ ]:


pearson_coef, p_value = stats.pearsonr(df['slope'], df['target'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 


# ## Conclusion
# Since the p-value is  <  0.001, the correlation between slope  and target is statistically significant, and there is linear relationship which is stong (~0.34).

# ## ca vs target
# let's  calculate the pearsonr coefficient and p-value 

# In[ ]:


pearson_coef, p_value = stats.pearsonr(df['ca'], df['target'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 


# ## Conclusion
# Since the p-value is  <  0.001, the correlation between ca and target is statistically significant, and there is negative  linear relationship which is weak (~0.39).

# ## thal vs target
# let's  calculate the pearsonr coefficient and p-value 

# In[ ]:


pearson_coef, p_value = stats.pearsonr(df['thal'], df['target'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 


# ## Conclusion
# Since the p-value is  <  0.001, the correlation between thal and target is statistically significant, and there is negative  linear relationship which is very weak (~0.344).

# # Data Modelling 

# In[ ]:



x=df.drop('target',axis=1)
x.head()
y=df['target']


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state =0)


# # Logistic Regression 

# In[ ]:


#First we import the library of Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


lr=LogisticRegression() #create an object

# now we train the model using fit
lr.fit(x_train,y_train) 

# in this line model makes predictions
predictions=lr.predict(x_test)

#we find the accuracy of model and saved in accuracy varaible
accuracy=accuracy_score(y_test,predictions)


print(predictions)
print(f"LogisticRegression  Accuracy Score is  {accuracy}")


# # Support Vector Machine Algorithm

# In[ ]:


#First we import the library of Support vector machine

from sklearn.svm import SVC

svc=SVC()

# now we train the model using fit
svc.fit(x_train,y_train) 

# in this line model is tested
sv_predictions=svc.predict(x_test)

#we find the accuracy of model and saved in sv_accuracy varaible
sv_accuracy=accuracy_score(y_test,sv_predictions)


print(sv_predictions)
print(f"Support vectoe Machine   Accuracy Score is  {sv_accuracy}")


# # Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()

# now we train the model using fit
nb.fit(x_train,y_train) 

# in this line model is tested
nb_predictions=nb.predict(x_test)

#we find the accuracy of model and saved in nb_accuracy varaible
nb_accuracy=accuracy_score(y_test,nb_predictions)


print(nb_predictions)
print(f" Gaussian Naive Bayes Algorithm Accuracy Score is  {nb_accuracy}")


# # Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=50)

# now we train the model using fit
rf.fit(x_train,y_train) 

# in this line model is tested
rf_predictions=rf.predict(x_test)

#we find the accuracy of model and saved in rf_accuracy varaible
rf_accuracy=accuracy_score(y_test,rf_predictions)


print(rf_predictions)
print(f" RAndom Forest Algorithm Accuracy Score is  {rf_accuracy}")


# ## KNN Classifier

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(3)

# now we train the model using fit
knn.fit(x_train,y_train) 

# in thisn line model is tested
knn_predictions=knn.predict(x_test)

#we find the accuracy of model and saved in sv_accuracy varaible
knn_accuracy=accuracy_score(y_test,knn_predictions)


print(knn_predictions)
print(f" K Nearest  neighbour Cklassification Accuracy Score is  {knn_accuracy}")


# # K-Fold cross Validation

# In[ ]:


from sklearn.model_selection import cross_val_score


# ## Logistic regression model performance using cross_val_score

# In[ ]:


cross_val_score(LogisticRegression(), x, y)


# ## SVC model performance using cross_val_score

# In[ ]:


cross_val_score(SVC(), x, y)


# ## Naive Bayes Performance using KFold 

# In[ ]:


cross_val_score(GaussianNB(),x,y)


# ## KNN performance using KFold

# In[ ]:


cross_val_score(KNeighborsClassifier(),x,y)


# ## Random Forest using KFold

# In[ ]:


cross_val_score(RandomForestClassifier(n_estimators=30),x,y)


# ## Parameter tunning using k fold cross validation

# In[ ]:


scores1 = cross_val_score(RandomForestClassifier(n_estimators=5),x, y, cv=10) 
np.average(scores1)


# In[ ]:


scores2 = cross_val_score(RandomForestClassifier(n_estimators=20),x, y, cv=10)
np.average(scores2)


# In[ ]:


scores3 = cross_val_score(RandomForestClassifier(n_estimators=28),x, y, cv=10)
np.average(scores3)


# In[ ]:


scores4 = cross_val_score(RandomForestClassifier(n_estimators=50),x, y, cv=10)
np.average(scores4)


# Here we used cross_val_score to fine tune our random forest classifier and figured that having around 50 trees in random forest gives best result. 

# #### _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
