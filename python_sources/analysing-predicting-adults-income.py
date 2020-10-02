#!/usr/bin/env python
# coding: utf-8

# Predict whether an adult's income exceeds or is below $50K/yr based on census data. Data Analysis, Data visualization, Feature Selection and Reduction, about 10 Machine Learning models/estimators. Multilayer Perceptron(Deep Learning/Artificial Neural Network). Dataset splitted into training and testing data.

# In[ ]:


import numpy
import pandas
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier


import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error



from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# In[ ]:


# load dataset
dataframe = pandas.read_csv("../input/adult.csv")

dataframe = dataframe.replace({'?': numpy.nan}).dropna()


# In[ ]:



# Assign names to Columns
dataframe.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

# Encode Data
dataframe.workclass.replace(('Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'),(1,2,3,4,5,6,7,8), inplace=True)
dataframe.education.replace(('Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'),(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16), inplace=True)
dataframe.marital_status.replace(('Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'),(1,2,3,4,5,6,7), inplace=True)
dataframe.occupation.replace(('Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'),(1,2,3,4,5,6,7,8,9,10,11,12,13,14), inplace=True)
dataframe.relationship.replace(('Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'),(1,2,3,4,5,6), inplace=True)
dataframe.race.replace(('White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'),(1,2,3,4,5), inplace=True)
dataframe.sex.replace(('Female', 'Male'),(1,2), inplace=True)
dataframe.native_country.replace(('United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'),(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41), inplace=True)
dataframe.income.replace(('<=50K', '>50K'),(0,1), inplace=True)


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


# 'marital_status' has the highest correlation with the level of income(which is a negative correlation), followed by 'education_num' which is a positive correlation, 'fnlwgt' has the least correlation 

# In[ ]:


dataset = dataframe.values


X = dataset[:,0:14]
Y = dataset[:,14] 


# In[ ]:


# feature extraction
test = SelectKBest(score_func=chi2, k=3)
fit = test.fit(X, Y)

# scores
numpy.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)

# summarise selected features
print(features[0:10,:])


#  'relationship', 'capital_loss' and 'capital_gain' were top 3 selected features for predicting 'Income'
#  using chi-squared (chi2) statistical test, Univariate Selections

# In[ ]:


#Feature Selection
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)

print("Number of Features: ", fit.n_features_)
print("Selected Features: ", fit.support_)
print("Feature Ranking: ", fit.ranking_)


# 'education_num', 'marital_status' and 'sex' were top 3 selected features/feature combination for predicting 'Income'
#  using Recursive Feature Elimination, the 1st and 2n are atually the two attributes with the highest correlation with the 
#  'Income' class

#  Dimensionality Reduction: Principal Component Analysis: 

# In[ ]:


pca = PCA(n_components=3)
fit = pca.fit(X)

print("Explained Varience: ", fit.explained_variance_ratio_)


# In[ ]:


model = ExtraTreesClassifier()
model.fit(X, Y)
print("Feature Importance: ", model.feature_importances_)


# In[ ]:





# 'age', 'fnlwgt' and 'marital_status' are the top 3 features using Feature Importance from Extra Trees(Bagged decision trees)

# **VISUALIZATION**

# In[ ]:


plt.hist((dataframe.income))


# Most of the dataset's samples fall within the '<=50K' 'income' output class

# In[ ]:


dataframe.hist()


# In[ ]:


dataframe.plot(kind='density', subplots=True, layout=(4,4), sharex=False, sharey=False)


# There are a mixture of positive skews and negative skews the other attributes

# In[ ]:


scatter_matrix(dataframe)


# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataframe.corr(), vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,15,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(dataframe.columns)
ax.set_yticklabels(dataframe.columns)


# 'marital_status' has the highest positive corelation as expected

# 
# 
# 
# 
# 
# 
# **MACHINE LEARNING ESTIMATORS/MODELS**

# In[ ]:


# Split Data to Train and Test
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.3)
print("X_Train: ", X_Train.shape)
print("X_Test: ", X_Test.shape)
print("Y_Train: ", Y_Train.shape)
print("Y_Test: ", Y_Test.shape)


# In[ ]:




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
    


# 'Linear Discriminant Analysis' is the best estimators/models for this dataset, followed by 'LogisticRegression' and 'K Nearest Neighbour', they can be further explored and their hyperparameters tuned

# **DEEP LEARNING: MULTILAYER PERCEPTRON(ARTIFICIAL NEURAL NETWORK)**

# In[ ]:


# create model
model = Sequential()
model.add(Dense(28, input_dim=14, activation='relu', kernel_initializer="uniform"))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu', kernel_constraint=maxnorm(3), kernel_initializer="uniform"))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu', kernel_initializer="uniform"))
model.add(Dense(1, activation='sigmoid', kernel_initializer="uniform"))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_Train, Y_Train, epochs=300, batch_size=10)

# Evaluate the model
scores = model.evaluate(X_Test, Y_Test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

