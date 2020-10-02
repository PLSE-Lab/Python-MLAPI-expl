#!/usr/bin/env python
# coding: utf-8

# Predicting if a prospect would subscribe a term deposit or not. Data Analysis, Data visualization, Feature Selection and Reduction, about 10 Machine Learning models/estimators. Multilayer Perceptron(Deep Learning/Artificial Neural Network). Dataset splitted into training and testing data in order to avoid overfitting. 

# In[ ]:


import numpy
import pandas
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier


import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix

from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.preprocessing import LabelEncoder
from keras.constraints import maxnorm


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# load dataset
dataframe = pandas.read_csv("../input/bank-additional.csv")

#dataframe = dataframe.replace({'?': numpy.nan}).dropna()


# In[ ]:



# Encode Data
dataframe.job.replace(('admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)

dataframe.marital.replace(('divorced','married','single','unknown'),(1,2,3,4), inplace=True)
dataframe.education.replace(('basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown'),(1,2,3,4,5,6,7,8), inplace=True)
dataframe.default.replace(('no','yes','unknown'),(1,2,3), inplace=True)
dataframe.housing.replace(('no','yes','unknown'),(1,2,3), inplace=True)
dataframe.loan.replace(('no','yes','unknown'),(1,2,3), inplace=True)
dataframe.contact.replace(('cellular','telephone'),(1,2), inplace=True)
dataframe.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)
dataframe.day_of_week.replace(('mon','tue','wed','thu','fri'),(1,2,3,4,5), inplace=True)
dataframe.poutcome.replace(('failure','nonexistent','success'),(1,2,3), inplace=True)
dataframe.y.replace(('yes','no'),(0,1), inplace=True)

dataframe = dataframe.abs()


# In[ ]:


print("Head:", dataframe.head())


# In[ ]:


print("Statistical Description:", dataframe.describe())


# In[ ]:


print("Shape:", dataframe.shape)


# In[ ]:


print("Data Types:", dataframe.dtypes)


# In[ ]:


print("Correlation:", dataframe.corr(method='pearson'))


# 'duration'(last contact duration) has the highest correlation with the level of income(which is a negative correlation), followed by 'euribor3m'(euribor 3 month rate)
#  which is a positive correlation, 'loan'(personal loan) has the least correlation 
#  Implication: How long the last call with prospect is what mainly influences whether they'd subscribe a term deposit or not, they taken or not not taken a personal loan has the least infuence on such decision

# In[ ]:


dataset = dataframe.values


X = dataset[:,0:20]
Y = dataset[:,20] 


# In[ ]:


# feature extraction
test = SelectKBest(score_func=f_classif, k=3)
fit = test.fit(X, Y)

# scores
numpy.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)

# summarise selected features
print(features[0:10,:])


#  'loan', 'cons.conf.idx'(consumer confidence index) and 'education' were top 3 selected features for predicting 'Income'
#  using Analysis Of Variance(ANOVA) F-statistical test, Univariate Selections

# In[ ]:


#Feature Selection
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)

print("Number of Features: ", fit.n_features_)
print("Selected Features: ", fit.support_)
print("Feature Ranking: ", fit.ranking_) 


#  'previous'(number of contacts performed before this campaign and for this client), 'poutcome'(outcome of the previous marketing campaign) and 'euribor3m' were top 3 selected features/feature combination for predicting 'y'
#  using Recursive Feature Elimination, the 3rd was atually the attributes with the 2nd highest correlation with the 
#  'y' parameter

#  Dimensionality Reduction using Principal Component Analysis:

# In[ ]:


pca = PCA(n_components=3)
fit = pca.fit(X)

print("Explained Varience: ", fit.explained_variance_ratio_)


# In[ ]:


model = ExtraTreesClassifier()
model.fit(X, Y)
print("Feature Importance: ", model.feature_importances_)


# 
# # 'euribor3m', 'age' and 'job' are the top 3 features using Feature Importance from Extra Trees(Bagged decision trees)

# In[ ]:


plt.hist(dataframe.y)


# An uneven sample distribution, leaning towards mainly the "no" output class

# In[ ]:


dataframe.hist()


#  Most of the atibutes have positive skews , but  few outliers with negative skews. 
#  Most of the categorical features have uneven sample distributions
#  'euribor3m' has a near even distribution of samples accross its numeric values

# In[ ]:


dataframe.plot(kind='density', subplots=True, layout=(4,6), sharex=False, sharey=False)


# In[ ]:


dataframe.plot(kind='box', subplots=True, layout=(4,6), sharex=False, sharey=False)


# In[ ]:


scatter_matrix(dataframe)


# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataframe.corr(), vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,20,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(dataframe.columns)
ax.set_yticklabels(dataframe.columns)

#plt.show()


#  'marital_status' has the highest positive corelation as expected

# In[ ]:


# Split Data to Train and Test
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2)

num_instances = len(X)

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('L_SVM', LinearSVC()))
models.append(('SGDC', SGDClassifier()))
models.append(('ETC', ExtraTreesClassifier()))
models.append(('RFC', RandomForestClassifier()))

# Evaluations
results = []
names = []

for name, model in models:
    # Fit the model
    model.fit(X_Train, Y_Train)
    
    predictions = model.predict(X_Test)
    
    # Evaluate the model
    score = accuracy_score(Y_Test, predictions)
    mse = mean_squared_error(predictions, Y_Test)
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    results.append(mse)
    names.append(name)
    
    msg = "%s: %f (%f)" % (name, score, mse)
    print(msg)
    
    


# 
# 
#  'Linear Discriminant Analysis' and 'Logistic Regression' are the best estimators/models for this dataset, they can be further explored and their hyperparameters tuned

# In[ ]:


#Encode Categorical data
columns_encode = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome','y']
encoded_dataset = pandas.get_dummies(dataframe,columns=columns_encode)
encoded_dataset = encoded_dataset.values


X = encoded_dataset[:,0:63].astype(float)


Y_dataframe = dataframe.values
Y = Y_dataframe[:,20]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
print(encoded_Y.shape)

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, encoded_Y, test_size=0.3)
print("X_Train: ", X_Train.shape)
print("X_Test: ", X_Test.shape)
print("Y_Train: ", Y_Train.shape)
print("Y_Test: ", Y_Test.shape)


# In[ ]:


# create model
model = Sequential()
model.add(Dense(40, input_dim=63, init='uniform', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20, init='uniform', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(10, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='relu'))

# Compile model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_Train, Y_Train, epochs=100, batch_size=10)

# Evaluate the model
scores = model.evaluate(X_Test, Y_Test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# *Multilayer Perceptron Accuracy: 88.51%
# Logistic Regression Accuracy: 0.91%
# Linear Discriminant Analysis Accuracy: 0.90%*

# 

# 
