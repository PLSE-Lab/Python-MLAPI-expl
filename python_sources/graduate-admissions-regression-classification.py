#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import timeit
from statsmodels.tools.eval_measures import mse, rmse
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score


# In[ ]:


# Import dataset.
df = pd.read_csv("../input/Admission_Predict.csv",sep = ",")
df.head()


# In[ ]:


# Check for null values.
df.isnull().sum()

# Drop 'Serial No.'
df = df.drop('Serial No.',axis=1)

# Renaming columns.
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(' ','_')


# In[ ]:


df.columns


# In[ ]:


# Check if the data types are correct as per the meaning of each column.
df.info()


# In[ ]:


# Descriptive statistics.
df.describe()


# In[ ]:


cols = ['GRE_Score','TOEFL_Score','University_Rating','SOP','LOR','CGPA','Chance_of_Admit']

plt.figure(figsize=(6,40))

for i in range(len(cols)):
    plt.subplot(7,1,i+1)
    plt.hist(df[cols[i]],color='pink',alpha=0.75)
    plt.title("Distribution of " + cols[i])

plt.show()


# From the above GRE_Score, TOEFL_Score and CGPA are some what normally distributed and the target variable 'Chance_of_Admit' is left skewed.<br/>
# Let us observe the correlation between the target variable and independent variables.

# In[ ]:


cols = ['GRE_Score','TOEFL_Score','University_Rating','SOP','LOR','CGPA']

plt.figure(figsize=(6,40))

for i in range(len(cols)):
    plt.subplot(6,1,i+1)
    plt.scatter(df['Chance_of_Admit'],df[cols[i]],color='brown')
    plt.title("Correlation b/w 'Chance_of_Admit' and '{}'".format(cols[i]))

plt.show()


# Above plots show that there is a linear relation between target variable and GRE_Score, TOEFL_Score and CGPA. However, let us statistically confirm this using a correlation matrix.

# In[ ]:


# Visualize correlation between independant variables and the target variable. Here, the target variable is 'Chance_of_Admit'
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(),annot = True)
plt.show()


# In[ ]:


# Correlation factors for 'Chance_of_Admit'.
df.corr()['Chance_of_Admit'].sort_values(ascending=False)


# It is clear that CGPA has highest correlation with Chance_of_Admit followed by GRE_Score and TOEFL_Score. <br/>
# Also, there is a little correlation of University_Rating, SOR and LOR with the target variable.

# <b><font color='blue'>Simple Linear Regression with OLS(Ordinary Least Squares).<br/></b>

# In[ ]:


from sklearn import linear_model

X = df.drop(['Chance_of_Admit','Research'],axis = 1)
Y = df['Chance_of_Admit']

model = linear_model.LinearRegression()
model.fit(X,Y)

print('\nCoefficients: \n', model.coef_)
print('\nIntercept: \n', model.intercept_)


# <b>Linear regression equation with coefficients and intercept(from the above result) would be:</b><br/>
# Chance_of_Admit = 0.00227608GRE_Score + 0.00275344TOEFL_Score + 0.00606202University_Rating - 0.0019614SOP + 0.0227486LOR + 0.11987489CGPA - 1.4138594435308127<br/>
# 
# But we are more interested in identifying the significance of these coefficients. We can determine this using Statistical T-test. Let us use "statsmodels" to get T-test, p values and many other statistical information.

# <b><font color='blue'>Model1:<font/> Linear Regression using statsmodels

# <b>Interpreting estimated coefficients.

# In[ ]:


import statsmodels.api as sm

# We need to manually add a constant in statsmodels' sm
X = df.drop(['Chance_of_Admit','Research'],axis = 1)
Y = df['Chance_of_Admit']

X = sm.add_constant(X)
model1 = sm.OLS(Y, X).fit()

model1.summary()


# From the above, p-values of GRE_Score, TOEFL_Score, LOR and CGPA are less than 0.05 and p-values of University_Rating and SOP are greater than 0.05.<br/>
# This implies that the coefficient of University_Rating and SOP are effectively zero from a statistical point of view. They are simply not significant, and we can say that it appears to be no relationship between Chance_of_Admit and the features (University_Rating, SOP). <br/>
# 
# Let us remove these 2 variables and see if there is an improvement in the model or not.

# <b><font color='blue'>Model2:<font/> Remove University_Rating and SOP

# In[ ]:


X = df.drop(['Chance_of_Admit','Research','University_Rating','SOP'],axis = 1)
Y = df['Chance_of_Admit']

X = sm.add_constant(X)
model2 = sm.OLS(Y, X).fit()

model2.summary()


# <b>Evaluating performance by comparing <font color='blue'>model1</font> and <font color='blue'>model2

# - We can use an F-test to compare two models if one of them is nested within the other. That is, if the feature set in a model is a subset of the feature set of the other, then we can use F-test. In this case, we say that the model with higher F statistic is superior to the other one.
# - We can also use adjusted R-squared. The higher adjusted R-squared, the better the model explains the target variable.
# - Using information criteria is also a common way of comparing different models and selecting the best one. Here, the two information criteria are Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC). Both take into consideration the sum of the squared errors (SSE), the sample size, and the number of parameters. The lower AIC and BIC, the better the model explains the target variable.

# In[ ]:


performance_stats = pd.DataFrame()

# This information is drawn from model1 and model2 summary tables above.
performance_stats['F-statistic'] = [259.9,389.9]
performance_stats['Adj_R-squared'] = [0.796,0.796]
performance_stats['AIC'] = [-1051,-1054]
performance_stats['BIC'] = [-1023,-1034]

performance_stats


# From the above, we observe that model2 is slightly better than model1(based on the 3 rules from the bullet points). <br/>
# The target variable(Chance_of_Admit) is skewed. Let us transform it to a normal distribution and test the model again.

# In[ ]:


# Transform to normal distribution using boxcox transformation.
from scipy.stats import boxcox
boxcox_Chance_of_Admit,_ = boxcox(df['Chance_of_Admit'])
plt.figure(figsize=(6,5))
plt.hist(boxcox_Chance_of_Admit)
plt.show()


# Still it is NOT a complete normal distribution but some what better than before. Let us rerun the model with the transformed target variable.

# <b><font color='blue'>Model3: <font/>Apply boxcox transformation to the target variable.

# In[ ]:


df['boxcox_Chance_of_Admit'] = boxcox_Chance_of_Admit

X = df.drop(['Chance_of_Admit','Research','University_Rating','SOP','boxcox_Chance_of_Admit'],axis = 1)
Y = df['boxcox_Chance_of_Admit']

X = sm.add_constant(X)
model3 = sm.OLS(Y, X).fit()

model3.summary()


# In[ ]:


# These values are taken from the above summary tables for model1, model2 and model3.
performance_statistics = pd.DataFrame()
performance_statistics['F-statistic'] = [259.9,389.9,443.2]
performance_statistics['Adj_R-squared'] = [0.796,0.796,0.816]
performance_statistics['AIC'] = [-1051,-1054,-1272]
performance_statistics['BIC'] = [-1023,-1034,-1252]

performance_statistics['Model'] = ['model1','model2','model3']
performance_statistics.set_index('Model')


# Wow!! After transforming target variable to normal distribution, model3's performance stats have been improved.<br/>
# It's time to make predictions now.

# <b><font color='blue'>Model4: Making predictions.

# In[ ]:


from sklearn.model_selection import train_test_split

X = df.drop(['Chance_of_Admit','Research','University_Rating','SOP','boxcox_Chance_of_Admit'],axis = 1)
Y = df['boxcox_Chance_of_Admit']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 450)
X_train = sm.add_constant(X_train)

model4 = sm.OLS(y_train, X_train).fit()
model4.summary()


# In[ ]:


# We add constant to the model as it's a best practice to do so every time!
X_test = sm.add_constant(X_test)

# We are making predictions here
y_preds = model4.predict(X_test)

plt.scatter(y_test, y_preds)
plt.plot(y_test, y_test, color="red")
plt.xlabel("true values")
plt.ylabel("predicted values")
plt.title("Admission: true and predicted values")
plt.show()


# In the graph, we see that our model predicts the higher values of the target better than the lower values. By using visualizations like this, we can more intuitively understand the performance of the models. But, we need to find more accurate ways of assessing our models. Let us find out some metrics to evaluate how good the predictions are. Below are those metrics.

# In[ ]:


rmse_ols = rmse(y_test, y_preds)

print("Mean absolute error of the prediction is: {}".format(mean_absolute_error(y_test, y_preds)))
print("Mean squared error of the prediction is: {}".format(mse(y_test, y_preds)))
print("Root mean squared error of the prediction is: {}".format(rmse_ols))
print("Mean absolute percentage error of the prediction is: {}".format(np.mean(np.abs((y_test - y_preds) / y_test)) * 100))


# <b><font color='blue'>Model5: <font/>Apply boxcox transformation to the features.

# In[ ]:


# Transform to normal distribution using boxcox transformation.
boxcox_TOEFL_Score,_ = boxcox(df['TOEFL_Score'])
plt.figure(figsize=(6,5))
plt.hist(boxcox_TOEFL_Score,color='purple',alpha=0.70)
plt.title("Distribution of TOEFL_Score")
plt.show()

boxcox_GRE_Score,_ = boxcox(df['GRE_Score'])
plt.figure(figsize=(6,5))
plt.hist(boxcox_GRE_Score,color='purple',alpha=0.70)
plt.title("Distribution of GRE_Score")
plt.show()

boxcox_CGPA,_ = boxcox(df['CGPA'])
plt.figure(figsize=(6,5))
plt.hist(boxcox_CGPA,color='purple',alpha=0.70)
plt.title("Distribution of CGPA")
plt.show()

boxcox_LOR,_ = boxcox(df['LOR'])
plt.figure(figsize=(6,5))
plt.hist(boxcox_LOR,color='purple',alpha=0.70)
plt.title("Distribution of LOR")
plt.show()


# In[ ]:


df['boxcox_TOEFL_Score'] = boxcox_TOEFL_Score
df['boxcox_GRE_Score'] = boxcox_GRE_Score
df['boxcox_CGPA'] = boxcox_CGPA
df['boxcox_LOR'] = boxcox_LOR


# In[ ]:


X1 = df.drop(['Chance_of_Admit','Research','University_Rating','SOP',
             'TOEFL_Score','GRE_Score','CGPA','boxcox_Chance_of_Admit','LOR'],axis = 1)
Y1 = df['boxcox_Chance_of_Admit']

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, Y1, test_size = 0.2, random_state = 450)
X_train1 = sm.add_constant(X_train1)

model5 = sm.OLS(y_train1, X_train1).fit()
model5.summary()


# In[ ]:


# We add constant to the model as it's a best practice to do so every time!
X_test1 = sm.add_constant(X_test1)

# We are making predictions here
y_preds1 = model5.predict(X_test1)

plt.scatter(y_test1, y_preds1)
plt.plot(y_test1, y_test1, color="red")
plt.xlabel("true values")
plt.ylabel("predicted values")
plt.title("Admission: true and predicted values")
plt.show()


# In[ ]:


rmse_ols_Nor_dist = rmse(y_test1, y_preds1)

print("Mean absolute error of the prediction is: {}".format(mean_absolute_error(y_test1, y_preds1)))
print("Mean squared error of the prediction is: {}".format(mse(y_test1, y_preds1)))
print("Root mean squared error of the prediction is: {}".format(rmse_ols_Nor_dist))
print("Mean absolute percentage error of the prediction is: {}".format(np.mean(np.abs((y_test1 - y_preds1) / y_test1)) * 100))


# In[ ]:


print("Root mean squared error of MODEL4 and MODEL5 are {}  and  {}".format(rmse(y_test, y_preds),rmse(y_test1, y_preds1)))


# We can observe that RMSE for model5 is slightly decreased due to transforming independent variables(features) to normal distributions.

# <b><font color='blue'>Linear Regression

# In[ ]:


X = df.drop(['Chance_of_Admit','Research','University_Rating','SOP',
             'TOEFL_Score','GRE_Score','CGPA','boxcox_Chance_of_Admit','LOR'],axis = 1)
Y = df['boxcox_Chance_of_Admit']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 450)

lireg = linear_model.LinearRegression()
lireg.fit(X_train,y_train)
pred_test = lireg.predict(X_test)

rmse_lireg = rmse(y_test, pred_test)

print("RMSE for the prediction of test data is",rmse_lireg)


# <b><font color='blue'>KNN Regressor

# In[ ]:


# KNN Regression model with cross validation.
from sklearn import neighbors

knn = neighbors.KNeighborsRegressor(n_neighbors=10)
X = df.drop(['Chance_of_Admit','Research','University_Rating','SOP',
             'TOEFL_Score','GRE_Score','CGPA','boxcox_Chance_of_Admit','LOR'],axis = 1)
Y = df['boxcox_Chance_of_Admit']

score = cross_val_score(knn, X, Y, cv=5)
print("Score : ",score)
print("Variance : ",score.std()**2)


# In[ ]:


# KNN Regression model with weights parameter.
knn_w = neighbors.KNeighborsRegressor(n_neighbors=10,weights='distance')
X = df.drop(['Chance_of_Admit','Research','University_Rating','SOP',
             'TOEFL_Score','GRE_Score','CGPA','boxcox_Chance_of_Admit','LOR'],axis = 1)
Y = df['boxcox_Chance_of_Admit']

score = cross_val_score(knn_w, X, Y, cv=5)
print("Score : ",score)
print("Variance : ",score.std()**2)


# In[ ]:


# KNN Regression model with RMSE
knn = neighbors.KNeighborsRegressor(n_neighbors=10)
X = df.drop(['Chance_of_Admit','Research','University_Rating','SOP',
             'TOEFL_Score','GRE_Score','CGPA','boxcox_Chance_of_Admit','LOR'],axis = 1)
Y = df['boxcox_Chance_of_Admit']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 450)

knn.fit(X_train,y_train)
pred_train = knn.predict(X_train)
pred_test = knn.predict(X_test)

rmse_knn = rmse(y_test, pred_test)

print("RMSE for the prediction of trained data is",rmse(y_train, pred_train))
print("RMSE for the prediction of test data is",rmse_knn)


# <b><font color='blue'>KNN Classifier

# Let us change the column 'Chance_of_Admit' so that it holds boolean values(0 and 1) in order to use it for classification. <br/>
# - Here on, when ever I create a classification model, I use the dataframe "data" which holds boolean value for 'Chance_of_Admit'.
# - When I create a regression model, I use the original data frame "df".

# In[ ]:


data = df[['GRE_Score','TOEFL_Score','CGPA','LOR','Chance_of_Admit']]
data.loc[data['Chance_of_Admit']>=0.50,'Chance_of_Admit'] = 1
data.loc[data['Chance_of_Admit']<0.50,'Chance_of_Admit'] = 0
data['Chance_of_Admit'] = data['Chance_of_Admit'].astype('int')


# In[ ]:


# Look at our data.
plt.figure(figsize=(7,5))

plt.scatter(
    data[data['Chance_of_Admit'] == 1].CGPA,
    data[data['Chance_of_Admit'] == 1].GRE_Score,
    color='red'
)
plt.scatter(
    data[data['Chance_of_Admit'] == 0].CGPA,
    data[data['Chance_of_Admit'] == 0].GRE_Score,
    color='blue'
)

plt.legend(['Get Admission', 'No Admission'])
plt.title('"Admission(YES/NO)" Characteristics for GREvsCGPA',fontsize=12)
plt.xlabel('CGPA',fontsize=12)
plt.ylabel('GRE_Score',fontsize=12)
plt.show()


plt.figure(figsize=(7,5))

plt.scatter(
    data[data['Chance_of_Admit'] == 1].CGPA,
    data[data['Chance_of_Admit'] == 1].TOEFL_Score,
    color='red'
)
plt.scatter(
    data[data['Chance_of_Admit'] == 0].CGPA,
    data[data['Chance_of_Admit'] == 0].TOEFL_Score,
    color='blue'
)

plt.legend(['Get Admission', 'No Admission'])
plt.title('"Admission(YES/NO)" Characteristics for TOEFLvsCGPA',fontsize=12)
plt.xlabel('CGPA',fontsize=12)
plt.ylabel('TOEFL_Score',fontsize=12)
plt.show()


plt.figure(figsize=(7,5))

plt.scatter(
    data[data['Chance_of_Admit'] == 1].GRE_Score,
    data[data['Chance_of_Admit'] == 1].TOEFL_Score,
    color='red'
)
plt.scatter(
    data[data['Chance_of_Admit'] == 0].GRE_Score,
    data[data['Chance_of_Admit'] == 0].TOEFL_Score,
    color='blue'
)

plt.legend(['Get Admission', 'No Admission'])
plt.title('"Admission(YES/NO)" Characteristics for GREvsTOEFL',fontsize=12)
plt.xlabel('GRE_Score',fontsize=12)
plt.ylabel('TOEFL_Score',fontsize=12)
plt.show()


# From the above, it is observed that students with high scores of CGPA, GRE and TOEFL have more chances of getting admissions.

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

X = data[['GRE_Score','TOEFL_Score','CGPA','LOR']]
Y = data['Chance_of_Admit']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 450)

knc = KNeighborsClassifier(n_neighbors=5)
knc.fit(X_train,y_train)
predict_test_knc = knc.predict(X_test)

print(knc.predict([[337,118,9.65,4.5]]))
print(knc.predict_proba([[337,118,9.65,4.5]]))


# In[ ]:


# Confusion matrix

cm_knc = confusion_matrix(y_test,predict_test_knc)

# Visualization of Confusion matrix.
f, ax = plt.subplots(figsize =(3,3))
sns.heatmap(cm_knc,annot = True,linewidths=0.5,linecolor="orange",fmt = ".0f",ax=ax)
plt.xlabel("predictions")
plt.ylabel("Actuals")
plt.show()


acu_score_knc = accuracy_score(y_test,predict_test_knc)
score_knc = knc.score(X_test,y_test)


print("precision_score: ", precision_score(y_test,predict_test_knc))
print("recall_score: ", recall_score(y_test,predict_test_knc))


print("f1_score: ",f1_score(y_test,predict_test_knc))

print("Accuracy Score:  ",acu_score_knc)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from scipy import stats

neighbors = KNeighborsClassifier(n_neighbors=5, weights='distance')

# Our input data frame will be the z-scores this time instead of raw data.
X = pd.DataFrame({
    'GRE_Score': stats.zscore(data.GRE_Score),
    'TOEFL_Score': stats.zscore(data.TOEFL_Score),
    'CGPA': stats.zscore(data.CGPA),
    'LOR': stats.zscore(data.LOR)
})

# Fit our model.
Y = data['Chance_of_Admit']
neighbors.fit(X, Y)

print(neighbors.predict([[337,118,9.65,4.5]]))
print(neighbors.predict_proba([[337,118,9.65,4.5]]))


# <b><font color='blue'>Visualize a Decision tree.

# In[ ]:


# Decision tree classifier.
from sklearn import tree

# A convenience for displaying visualizations.
from IPython.display import Image

# Packages for rendering our tree.
import pydotplus
import graphviz

X = data[['GRE_Score','TOEFL_Score','CGPA','LOR']]
Y = data['Chance_of_Admit']

# Initialize and train our tree.
decision_tree = tree.DecisionTreeClassifier(
    criterion='entropy',
    max_features=1,
    max_depth=4,
    random_state = 1337
)
decision_tree.fit(X, Y)

# Render our tree.
dot_data = tree.export_graphviz(
    decision_tree, out_file=None,
    feature_names=X.columns,
    class_names=['Get Admission', 'No Admission'],
    filled=True
)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())


# In[ ]:


pred = decision_tree.predict(X)


# <b><font color='blue'>Decision Tree Classifier.

# In[ ]:


# Decision tree classifier.


X = data[['GRE_Score','TOEFL_Score','CGPA','LOR']]
Y = data['Chance_of_Admit']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 450)

dct = tree.DecisionTreeClassifier()
dct.fit(X_train,y_train)
predict_test_dct = dct.predict(X_test)

# Confusion matrix
cm_dct = confusion_matrix(y_test,predict_test_dct)

# Visualization of Confusion matrix.
f, ax = plt.subplots(figsize =(3,3))
sns.heatmap(cm_dct,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("predictions")
plt.ylabel("Actuals")
plt.show()

acu_score_dct = accuracy_score(y_test,predict_test_dct)
score_dct = dct.score(X_test,y_test)

print("precision_score: ", precision_score(y_test,predict_test_dct))
print("recall_score: ", recall_score(y_test,predict_test_dct))

print("f1_score: ",f1_score(y_test,predict_test_dct))

print("Accuracy Score:  ",acu_score_dct)


# In[ ]:


# Decision tree Regressor.
X = data[['GRE_Score','TOEFL_Score','CGPA','LOR']]
Y = data['Chance_of_Admit']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 450)

model = tree.DecisionTreeRegressor()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

rmse_dct = rmse(y_test, y_pred)

print("RMSE is : ",rmse_dct)


# <b><font color='blue'>Ensemble modeling: RandomForestClassifier with Cross validation

# In[ ]:


from sklearn import ensemble
from sklearn.model_selection import cross_val_score

rfc = ensemble.RandomForestClassifier()
X = data[['GRE_Score','TOEFL_Score','CGPA','LOR']]
Y = data['Chance_of_Admit']

score = cross_val_score(rfc, X, Y, cv=10)
print("Cross validation Score : ",score)
print("Variance : ",score.std()**2)


# In[ ]:


X = data[['GRE_Score','TOEFL_Score','CGPA','LOR']]
Y = data['Chance_of_Admit']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 450)

rfc1 = ensemble.RandomForestClassifier()
rfc1.fit(X_train,y_train)
predict_test_rfc = rfc1.predict(X_test)

# Confusion matrix
cm_rfc = confusion_matrix(y_test,predict_test_rfc)

# Visualization of Confusion matrix.
f, ax = plt.subplots(figsize =(3,3))
sns.heatmap(cm_rfc,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("predictions")
plt.ylabel("Actuals")
plt.show()

acu_score_rfc = accuracy_score(y_test,predict_test_rfc)
score_rfc = rfc1.score(X_test,y_test)

print("precision_score: ", precision_score(y_test,predict_test_rfc))
print("recall_score: ", recall_score(y_test,predict_test_rfc))

print("f1_score: ",f1_score(y_test,predict_test_rfc))

print("Accuracy Score:  ",acu_score_rfc)


# <b><font color='blue'>Linear SVM

# In[ ]:


from sklearn.svm import SVC

# Instantiate our model and fit the data.
X = data[['GRE_Score','TOEFL_Score','CGPA','LOR']]
Y = data['Chance_of_Admit']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 450)

svm = SVC(kernel = 'linear')
svm.fit(X_train,y_train)
pred = svm.predict(X_test)

# Confusion matrix
cm_svc = confusion_matrix(y_test,pred)

# Visualization of Confusion matrix.
f, ax = plt.subplots(figsize =(3,3))
sns.heatmap(cm_svc,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("predictions")
plt.ylabel("Actuals")
plt.show()

acu_score_svc = accuracy_score(y_test,pred)
score_svc = svm.score(X_test,y_test)

print("precision_score: ", precision_score(y_test,pred))
print("recall_score: ", recall_score(y_test,pred))

print("f1_score: ",f1_score(y_test,pred))

print("Accuracy Score:  ",acu_score_svc)


# <b><font color='blue'>SVM Regressor

# In[ ]:


from sklearn.svm import SVR

X = data[['GRE_Score','TOEFL_Score','CGPA','LOR']]
Y = data['Chance_of_Admit']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 450)

svr = SVR()
svr.fit(X_train,y_train)
pred = svr.predict(X_test)

rmse_svr = rmse(y_test, pred)

print("SVR Score is : ",svr.score(X, Y))
print("Cross validation score is : ",cross_val_score(svr, X, Y, cv=5))
print("Variance : ",cross_val_score(svr, X, Y, cv=5).std()**2)
print("RMSE is : ",rmse_svr)


# <b><font color='blue'>Gradient Boosting

# In[ ]:


X = data[['GRE_Score','TOEFL_Score','CGPA','LOR']]
y = data['Chance_of_Admit']


# In[ ]:


# Create training and test sets.
offset = int(X.shape[0]*0.9)

# Put 90% of the data in the training set.
X_train, y_train = X[:offset], y[:offset]

# And put 10% in the test set.
X_test, y_test = X[offset:], y[offset:]


# Since we're now working with a binary outcome, we are using a classifier. Our loss function can be "deviance", or "exponential". Deviance is used for logistic regression, and we'll try that here.

# In[ ]:


# We'll make 100 iterations, use 2-deep trees, and set our loss function.
params = {'n_estimators': 100,
          'max_depth': 2,
          'loss': 'deviance'}

# Initialize and fit the model.
clf = ensemble.GradientBoostingClassifier(**params)
clf.fit(X_train, y_train)

predict_train = clf.predict(X_train)
predict_test = clf.predict(X_test)

# Accuracy tables.
table_train = pd.crosstab(y_train, predict_train, margins=True)
table_test = pd.crosstab(y_test, predict_test, margins=True)

train_tI_errors = table_train.loc[0,1] / table_train.loc['All','All']
train_tII_errors = table_train.loc[1,0] / table_train.loc['All','All']

test_tI_errors = table_test.loc[0,1]/table_test.loc['All','All']
test_tII_errors = table_test.loc[1,0]/table_test.loc['All','All']

print((
    'Training set accuracy:\n'
    'Percent Type I errors: {}\n'
    'Percent Type II errors: {}\n\n'
    'Test set accuracy:\n'
    'Percent Type I errors: {}\n'
    'Percent Type II errors: {}'
).format(train_tI_errors, train_tII_errors, test_tI_errors, test_tII_errors))


# Unlike decision trees, gradient boost solutions are not terribly easy to interpret on the surface. But they aren't quite a black box. We can get a measure of how important various features are by counting how many times a feature is used over the course of many decision trees.

# In[ ]:


feature_importance = clf.feature_importances_

# Make importances relative to max importance.
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


# <b><font color='blue'>Performance of all Regression models:

# In[ ]:


y = [rmse_ols,rmse_lireg,rmse_knn,rmse_dct,rmse_svr]
x = ['OLS','Linear_Reg','KNN','Dec_Tree','Support Vector']
plt.bar(x,y)
plt.ylabel("RMSE")
plt.title("RMSE w.r.t Regression Algorithms")
plt.show()


# <b>From the above, it is observed that OLS and Linear Regression have low RMSE values. 

# <b><font color='blue'>Performance of all Classification models:

# In[ ]:


y1 = [acu_score_knc,acu_score_dct,acu_score_rfc,acu_score_svc]
x1 = ['KNN_Classifer','Dec_Tree','Random_Forest','Support_Vector']
plt.bar(x1,y1)
plt.ylabel("Accuracy Score")
plt.title("Accuracy score w.r.t Classification Algorithms")
plt.show()


# <b>From the above, it is observed that all models have almost similar accuracy. Among them, KNN Classifier and Random Forest have better accuracy compared to others.
