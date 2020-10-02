#!/usr/bin/env python
# coding: utf-8

# # Customer churn
# The customer churn, also known as customer attrition, refers to the phenomenon whereby a customer leaves a company. Some studies confirmed that acquiring new customers can cost five times more than satisfying and retaining existing customers. As a matter of fact, there are a lot of benefits that encourage the tracking of the customer churn rate, for example:
# * Marketing costs to acquire new customers are high. Therefore, it is important to retain customers so that the initial investment is not
# wasted;
# * It has a direct impact on the ability to expand the company;
# * etc.
# 
# In this project our goal is to predict the probability of a customer is likely to churn using machine learning techniques.
# 

# - <a href='#1'>1. Load Data</a>
# - <a href='#2'>2. Data Manipulation</a>
# - <a href='#3'>3. Exploratory Data Analysis</a>
#     - <a href='#3.1'>3.1. Customer churn in data</a>
#     - <a href='#3.2'>3.2. Distribution of categorical variables</a>
#     - <a href='#3.3'>3.3. Distribution of continuous variables</a>
#     - <a href='#3.4'>3.4. Finding missing values</a>
#     - <a href='#3.5'>3.5. Correlation matrix</a>
#     - <a href='#3.6'>3.6. Detecting and handling outliers</a>
# - <a href='#4'>4. Data preparation</a>
# - <a href='#5'>5. Feature engineering for the baseline model</a>
#     - <a href='#5.1'>5.1. Feature importance</a>
# - <a href='#6'>6. Selecting the machine learning algorithms</a>
#     - <a href='#6.1'>6.1. Train and build baseline model</a>
#         - <a href='#6.2'>6.1.1 Splitting the dataset</a>
#         - <a href='#6.3'>6.1.2. Model fitting</a>
#     - <a href='#6.4'>6.2. Testing the baseline model</a>
#     - <a href='#6.4'>6.3. ROC-AUC performance for the models</a>
# - <a href='#7'>7. Optimization</a>
#     - <a href='#6.1'>7.1. Implementing a cross-validation based approach</a>
#     - <a href='#6.2'>7.2. Implementing hyperparameter tuning</a>
#     - <a href='#6.3'>7.3. Train models with the help of new hyperparameter</a>
#     - <a href='#6.4'>7.4. Problem with the optimization approach</a>
#         - <a href='#6.5'>7.4.1. Feature transformation</a>
#         - <a href='#6.5'>7.4.2. Voting-based ensemble model</a>
#         - <a href='#6.5'>7.4.2.1. Transformed data</a>
#         - <a href='#6.5'>7.4.2.2. Untransformed data</a>
# - <a href='#7'>8. Conclusion</a>

# In[ ]:


# Libraries
from __future__ import print_function
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #visualization
import matplotlib.pyplot as plt #visualization
get_ipython().run_line_magic('matplotlib', 'inline')

import itertools
import warnings
warnings.filterwarnings("ignore")
import os
import io
import plotly.offline as py #visualization
py.init_notebook_mode(connected=True) #visualization
import plotly.graph_objs as go #visualization
import plotly.tools as tls #visualization
import plotly.figure_factory as ff #visualization
#print(os.listdir("../input"))


# # Load Data

# In[ ]:


# Read the training dataset
training_data = pd.read_csv('../input/Churn_Modelling.csv')


# # Data Manipulation

# In[ ]:


# Print the first 5 lines of the dataset
training_data.head()


# We can see that the columns name are not consistent.

# In[ ]:


# Convert all columns heading in lowercase 
clean_column_name = []
columns = training_data.columns
for i in range(len(columns)):
    clean_column_name.append(columns[i].lower())
training_data.columns = clean_column_name


# In[ ]:


# Drop the irrelevant columns  as shown above
training_data = training_data.drop(["rownumber", "customerid", "surname"], axis = 1)


# In[ ]:


#Separating churn and non churn customers
churn     = training_data[training_data["exited"] == 1]
not_churn = training_data[training_data["exited"] == 0]


# In[ ]:


target_col = ["exited"]
cat_cols   = training_data.nunique()[training_data.nunique() < 6].keys().tolist()
cat_cols   = [x for x in cat_cols if x not in target_col]
num_cols   = [x for x in training_data.columns if x not in cat_cols + target_col]


# In[ ]:





# # Exploratory Data Analysis (EDA)

# In[ ]:


# Print the first 5 lines of the dataset
training_data.head()


# In[ ]:


# View the dimension of the dataset
training_data.shape


# In[ ]:


# Checking for unique value in the data attributes
training_data.nunique()


# As we can see the rownumber attribute is just like a counter of records, the customerid attribute is a unique identifier for a given customer and the surname attribute enter also the profiling a customer. So we are going remove them from our dataset they don't give useful information the analysis.

# In[ ]:


# Drop the irrelevant columns  as shown above
#training_data = training_data.drop(["rownumber", "customerid", "surname"], axis = 1)


# In[ ]:


# Describe the all statistical properties of the training dataset
training_data[training_data.columns[:10]].describe()


# In[ ]:


# Median
training_data[training_data.columns[:10]].median()


# In[ ]:


# Mean
training_data[training_data.columns[:10]].mean()


# ### Customer churn in the data

# In[ ]:


# Percentage per category for the target column.
percentage_labels = training_data['exited'].value_counts(normalize = True) * 100
percentage_labels


# In[ ]:


# Graphical representation of the target label percentage.
total_len = len(training_data['exited'])
sns.set()
sns.countplot(training_data.exited).set_title('Data Distribution')
ax = plt.gca()
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2.,
            height + 2,
            '{:.2f}%'.format(100 * (height/total_len)),
            fontsize=14, ha='center', va='bottom')
sns.set(font_scale=1.5)
ax.set_xlabel("Labels for exited column")
ax.set_ylabel("Numbers of records")
plt.show()


# From this chart, one can see that there are many records with the target label $0$ and fewer records with the target label $1$. One can see that the data records with a $0$ label are about $79.63 \%$, whereas $20.37 \%$ of the data records are labeled $1$. We will use all of these facts in the upcoming sections. For now, we can consider our outcome variable as imbalanced.

# ### Distribution of the categorical variables

# In[ ]:


#function  for pie plot for customer attrition types
def plot_pie(column) :
    
    trace1 = go.Pie(values  = churn[column].value_counts().values.tolist(),
                    labels  = churn[column].value_counts().keys().tolist(),
                    hoverinfo = "label+percent+name",
                    domain  = dict(x = [0,.48]),
                    name    = "Churn Customers",
                    marker  = dict(line = dict(width = 2,
                                               color = "rgb(243,243,243)")
                                  ),
                    hole    = .6
                   )
    trace2 = go.Pie(values  = not_churn[column].value_counts().values.tolist(),
                    labels  = not_churn[column].value_counts().keys().tolist(),
                    hoverinfo = "label+percent+name",
                    marker  = dict(line = dict(width = 2,
                                               color = "rgb(243,243,243)")
                                  ),
                    domain  = dict(x = [.52,1]),
                    hole    = .6,
                    name    = "Non churn customers" 
                   )


    layout = go.Layout(dict(title = column + " distribution in customer attrition ",
                            plot_bgcolor  = "rgb(243,243,243)",
                            paper_bgcolor = "rgb(243,243,243)",
                            annotations = [dict(text = "churn customers",
                                                font = dict(size = 13),
                                                showarrow = False,
                                                x = .15, y = .5),
                                           dict(text = "Non churn customers",
                                                font = dict(size = 13),
                                                showarrow = False,
                                                x = .88,y = .5
                                               )
                                          ]
                           )
                      )
    data = [trace1,trace2]
    fig  = go.Figure(data = data,layout = layout)
    py.iplot(fig)
#for all categorical columns plot pie
#for i in cat_cols :
    #plot_pie(i)


# In[ ]:


# Calling the function for plotting the pie plot for geography column
plot_pie(cat_cols[0])


# The output above shows us that the among the churned customers those who are are geographycally located in Germay have a high rate of churn with $40\%$, followed by France with $39.8\%$ and Spain with $20.3\%$. For non chun customers France is leading with $52.8\%$, Spain with $25.9\%$ and Germany with $21.3\%$.

# In[ ]:


# Calling the function for plotting the pie plot for gender column
plot_pie(cat_cols[1])


# The output above shows us that for the churn customers female have $55.9\%$, whereas male with $44.1\%$. For the case of non churn customers $57.3\%$ are male and $42.7\%$ are female.

# In[ ]:


# Calling the function for plotting the pie plot for numofproducts column
plot_pie(cat_cols[2])


# The graph above shows that among the churn customers, the rate of those who use one product is very high with $69.2\%$, followed by  those who use two products with $17.1\%$, three products with $10.8\%$, and four products with $2.95\%$. For non churn customers, customers with two products are $53.3\%$, one product are $46.2\%$, and three products are $0.58\%$.

# In[ ]:


# Calling the function for plotting the pie plot for gender column
plot_pie(cat_cols[3])


# The output above shows us that for the churn customers those who possess a card are $69.9\%$, whereas those don't possess are $30.1\%$. For the case of non churn customers $70.7\%$ possess a card and $29.3\%$ don't possess a card.

# In[ ]:


# Calling the function for plotting the pie plot for geography column
plot_pie(cat_cols[4])


# The output above shows us that the among the churned customers those who are not active members have a high rate of churn with $63.9\%$, and active members with $36.1\%$. For non chun customers active members are leading with $55.5\%$, and non active members with $44.5\%$.

# ### Distribution of the continuous variables

# 

# In[ ]:


#function  for histogram for customer churn types
def histogram(column) :
    trace1 = go.Histogram(x  = churn[column],
                          histnorm= "percent",
                          name = "Churn Customers",
                          marker = dict(line = dict(width = .5,
                                                    color = "black"
                                                    )
                                        ),
                         opacity = .9 
                         ) 
    
    trace2 = go.Histogram(x  = not_churn[column],
                          histnorm = "percent",
                          name = "Non churn customers",
                          marker = dict(line = dict(width = .5,
                                              color = "black"
                                             )
                                 ),
                          opacity = .9
                         )
    
    data = [trace1,trace2]
    layout = go.Layout(dict(title =column + " distribution in customer attrition ",
                            plot_bgcolor  = "rgb(243,243,243)",
                            paper_bgcolor = "rgb(243,243,243)",
                            xaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                             title = column,
                                             zerolinewidth=1,
                                             ticklen=5,
                                             gridwidth=2
                                            ),
                            yaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                             title = "percent",
                                             zerolinewidth=1,
                                             ticklen=5,
                                             gridwidth=2
                                            ),
                           )
                      )
    fig  = go.Figure(data=data,layout=layout)
    
    py.iplot(fig)


# **Note:** For more information on the different plot just pass the mousse hover the graph.

# In[ ]:


# Calling the function for plotting the histogram for creditscore column 
histogram(num_cols[0])


# In[ ]:


# Calling the function for plotting the histogram for creditscore column 
# Pass the mouse hover the graph for more information. 
histogram(num_cols[1])


# The graph above shows us that the customers with age of 46 are the most churned.

# In[ ]:


# Calling the function for plotting the histogram for tenure column 
histogram(num_cols[2])


# Pass the mouse hover the graph for more information. It shows us that the customers who have been with the bank just for one moth are the most churned.

# In[ ]:


# Calling the function for plotting the histogram for balance column 
histogram(num_cols[3])


# In[ ]:


# Calling the function for plotting the histogram for estimatedsalary  column 
histogram(num_cols[4])


# ## Finding missing values
# In order to find the missing values in the dataset, we need to check each and every data attribute. First, we will try to identify which attribute has a missing or null value.

# In[ ]:


training_data.isnull().sum()


# As one can see there is no missing values in the dataset.

# ## Correlation
# The term correlation refers to a mutual relationship or association between quantities. So, here, we will find out what kind of association is present among the different data attributes.

# In[ ]:


# Get the correlation matrix of the training dataset
training_data[training_data.columns].corr()


# In[ ]:


# Visualization of the correlation matrix using heatmap plot
sns.set()
sns.set(font_scale = 1.25)
sns.heatmap(training_data[training_data.columns[:10]].corr(), annot = True,fmt = ".1f")
plt.show()


# The following facts can be derived from the graph:
# *  Cells with 1.0 values are highly correlated with each other;
# * Each attribute has a very high correlation with itself, so all the diagonal values are 1.0;
# * balance attribute is negatively correlated with numberofproducts attribute. It means one attribute increases as the other decreases, and vice versa.
# 
# Before moving ahead, we need to check whether these attributes contain any outliers or insignificant values. If they do, we need to handle these outliers, so our next section is about detecting outliers from our training dataset.

# ## Detecting and Handling Outliers
# In this part, we will try to detect outliers and how to handle them. 

# In[ ]:


# Function which plot box plot for detecting outliers
trace = []
def gen_boxplot(df):
    for feature in df:
        trace.append(
            go.Box(
                name = feature,
                y = df[feature]
            )
        )

new_df = training_data[num_cols[:1]]
gen_boxplot(new_df)
data = trace
py.iplot(data)


# The box plots above don't  show us any value that are faraway from the min value and also from the max value, so there is  outliers detected.

# In[ ]:


# Function which plot box plot for detecting outliers
trace = []
def gen_boxplot(df):
    for feature in df:
        trace.append(
            go.Box(
                name = feature,
                y = df[feature]
            )
        )
new_df = training_data[num_cols[1:3]]
gen_boxplot(new_df)
data = trace
py.iplot(data)


# The graph tells us that for age there is few outliers. As we can see the two extrem values for the age box plot.

# In[ ]:


# Handling age column outliers
ageNew = []
for val in training_data.age:
    if val <= 85:
        ageNew.append(val)
    else:
        ageNew.append(training_data.age.median())
        
training_data.age = ageNew


# In[ ]:


# Function which plot box plot for detecting outliers
trace = []
def gen_boxplot(df):
    for feature in df:
        trace.append(
            go.Box(
                name = feature,
                y = df[feature]
            )
        )
new_df = training_data[num_cols[3:]]
gen_boxplot(new_df)
data = trace
py.iplot(data)


# The graph above shows nothing abnormal.

# # Data preparation

# In[ ]:


# One-Hot encoding our categorical attributes
list_cat = ['geography', 'gender']
training_data = pd.get_dummies(training_data, columns = list_cat, prefix = list_cat)


# In[ ]:


# Print the first five rows
training_data.head()


# # Feature engineering for the baseline model
# In this section, you will learn how to select features that are important in order to develop the predictive model. So right now, just to begin with, we won't focus much on deriving new features at this stage because first, we need to know which input variables / columns / data attributes / features give us at least baseline accuracy. So, in this first iteration, our focus is on the selection of features from the available training dataset.
# ## Finding out Feature importance
# We need to know which the important features are. In order to find that out, we are going to train the model using the Random Forest classifier. After that, we will have a rough idea about the important features for us.

# In[ ]:


# Import the Random Forest classifier
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


# We perform training on the Random Forest model and generate the importance of the features
X = training_data.drop('exited', axis=1)
y = training_data.exited
features_label = X.columns
forest = RandomForestClassifier (n_estimators = 10000, random_state = 0, n_jobs = -1)
forest.fit(X, y)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for i in range(X.shape[1]):
    print ("%2d) %-*s %f" % (i + 1, 30, features_label[i], importances[indices[i]]))


# In[ ]:


# Visualization of the Feature importances
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices], color = "green", align = "center")
plt.xticks(range(X.shape[1]), features_label, rotation = 90)
plt.show()


# The graph above shows the features with the highest importance value to the lowest importance value. It shows the most important features are creditscore, age, tenure, balance,  and so on.
# We will surely revisit again feature engineering in the upcoming sections.

# # Selecting Machine Learning Algorithms
# Since we are modeling a critic problem for that we need model with high performance possible. Here, we will try a couple of different machine learning algorithms in order to get an idea about which machine learning algorithm performs better. Also, we will perform a accuracy comparison amoung them. 
# As our problem is a classification problem, the algorithms that we are going to choose are as follows:
# * K-Nearest Neighbor (KNN)
# * Logistic Regression (LR)
# * AdaBoost
# * Gradient Boosting (GB)
# * RandomForest (RF)

# ## Train and build baseline model

# In[ ]:


# Import different models 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# Scoring function
from sklearn.metrics import roc_auc_score, roc_curve


# In[ ]:


X = training_data.drop('exited', axis=1)
y = training_data.exited


# As you can see in the code above, variable X contains all the columns except the target column entitled *exited*, so we have dropped this column. The reason behind dropping this attribute is that this attribute contains the answer/target/label for each row. Machine algorithms need input in terms of a key-value pair, so a target column is key and all other columns are values. We can say that a certain pattern of values will lead to a particular target value, which we need to predict using an machine learning algorithm.

# #### Splitting the dataset

# In[ ]:


# Splitting the dataset in training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


# the code above splits the training  data. We will use $75\%$ of the training data for actual training purposes, and once training is completed, we will use the remaining $25\%$ of the training data to check the training accuracy of our trained  model.

# #### Model fitting

# In[ ]:


# Initialization of the KNN
knMod = KNeighborsClassifier(n_neighbors = 5, weights = 'uniform', algorithm = 'auto', leaf_size = 30, p = 2,
                             metric = 'minkowski', metric_params = None)
# Fitting the model with training data 
knMod.fit(X_train, y_train)


# In[ ]:


# Initialization of the Logistic Regression
lrMod = LogisticRegression(penalty = 'l2', dual = False, tol = 0.0001, C = 1.0, fit_intercept = True,
                            intercept_scaling = 1, class_weight = None, 
                            random_state = None, solver = 'liblinear', max_iter = 100,
                            multi_class = 'ovr', verbose = 2)
# Fitting the model with training data 
lrMod.fit(X_train, y_train)


# In[ ]:


# Initialization of the AdaBoost model
adaMod = AdaBoostClassifier(base_estimator = None, n_estimators = 200, learning_rate = 1.0)
# Fitting the model with training data 
adaMod.fit(X_train, y_train)


# In[ ]:


# Initialization of the GradientBoosting model
gbMod = GradientBoostingClassifier(loss = 'deviance', n_estimators = 200)
# Fitting the model with training data 
gbMod.fit(X_train, y_train)


# In[ ]:


# Initialization of the Random Forest model
rfMod = RandomForestClassifier(n_estimators=10, criterion='gini')
# Fitting the model with training data 
rfMod.fit(X_train, y_train)


# ## Testing the baseline model
# Here, we will implement the code, which will give us an idea about how good or how bad our trained models perform in a validation set. We are using the mean accuracy score and the AUC-ROC score.
# We have generated five different classifiers and, after performing testing for each of them on the validation dataset, which is $25\%$ of held-out dataset from the training dataset, we will find out which model works well and gives us a reasonable baseline score.

# In[ ]:


# Compute the model accuracy on the given test data and labels
knn_acc = knMod.score(X_test, y_test)
# Return probability estimates for the test data
test_labels = knMod.predict_proba(np.array(X_test.values))[:,1]
# Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
knn_roc_auc = roc_auc_score(y_test, test_labels , average = 'macro', sample_weight = None)


# In[ ]:


# Compute the model accuracy on the given test data and labels
lr_acc = lrMod.score(X_test, y_test)
# Return probability estimates for the test data
test_labels = lrMod.predict_proba(np.array(X_test.values))[:,1]
# Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
lr_roc_auc = roc_auc_score(y_test, test_labels , average = 'macro', sample_weight = None)


# In[ ]:


# Compute the model accuracy on the given test data and labels
ada_acc = adaMod.score(X_test, y_test)
# Return probability estimates for the test data
test_labels = adaMod.predict_proba(np.array(X_test.values))[:,1]
# Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
ada_roc_auc = roc_auc_score(y_test, test_labels , average = 'macro')


# In[ ]:


# Compute the model accuracy on the given test data and labels
gb_acc = gbMod.score(X_test, y_test)
# Return probability estimates for the test data
test_labels = gbMod.predict_proba(np.array(X_test.values))[:,1]
# Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
gb_roc_auc = roc_auc_score(y_test, test_labels , average = 'macro')


# In[ ]:


# Compute the model accuracy on the given test data and labels
rf_acc = rfMod.score(X_test, y_test)
# Return probability estimates for the test data
test_labels = rfMod.predict_proba(np.array(X_test.values))[:,1]
# Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
rf_roc_auc = roc_auc_score(y_test, test_labels , average = 'macro')


# In[ ]:


models = ['KNN', 'Logistic Regression', 'AdaBoost', 'GradientBoosting', 'Random Forest']
accuracy = [knn_acc, lr_acc, ada_acc, gb_acc, rf_acc]
roc_auc = [knn_roc_auc, lr_roc_auc, ada_roc_auc, gb_roc_auc, rf_roc_auc]

d = {'accuracy': accuracy, 'roc_auc': roc_auc}
df_metrics = pd.DataFrame(d, index = models)
df_metrics


# ### ROC-AUC performance for the models

# In[ ]:


fpr_knn, tpr_knn, _ = roc_curve(y_test, knMod.predict_proba(np.array(X_test.values))[:,1])
fpr_lr, tpr_lr, _ = roc_curve(y_test, lrMod.predict_proba(np.array(X_test.values))[:,1])
fpr_ada, tpr_ada, _ = roc_curve(y_test, adaMod.predict_proba(np.array(X_test.values))[:,1])
fpr_gb, tpr_gb, _ = roc_curve(y_test, gbMod.predict_proba(np.array(X_test.values))[:,1])
fpr_rf, tpr_rf, _ = roc_curve(y_test, rfMod.predict_proba(np.array(X_test.values))[:,1])


# In[ ]:


# Plot the roc curve
plt.figure(figsize = (12,6), linewidth= 1)
plt.plot(fpr_knn, tpr_knn, label = 'KNN Score: ' + str(round(knn_roc_auc, 5)))
plt.plot(fpr_lr, tpr_lr, label = 'LR score: ' + str(round(lr_roc_auc, 5)))
plt.plot(fpr_ada, tpr_ada, label = 'AdaBoost Score: ' + str(round(ada_roc_auc, 5)))
plt.plot(fpr_gb, tpr_gb, label = 'GB Score: ' + str(round(gb_roc_auc, 5)))
plt.plot(fpr_rf, tpr_rf, label = 'RF score: ' + str(round(rf_roc_auc, 5)))
plt.plot([0,1], [0,1], 'k--', label = 'Random guessing: 0.5')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve ')
plt.legend(loc='best')
plt.show()


# In the code above we used *score()* function of scikit-learn, which give us the mean accuracy score, whereas, the *roc_auc_score()* function  provide us with the ROC-AUC score, which is more significant for us because the mean accuracy score considers only one threshold value, whereas the ROC-AUC score takes into consideration all possible threshold values and gives us the score.
# 
# As you can see in the code snippets given above, the GradientBoosting with $0.86$ and the AdaBoost with $0.84$ classifiers get a good ROC-AUC score on the validation dataset. Other classifiers, such as logistic regression, KNN, and RandomForest do not perform well on the validation set. From this stage onward, we will work with GradientBoosting and AdaBoost classifiers in order to improve their accuracy score.
# 
# In the next section, we will see what we need to do in order to increase classification accuracy since we want a model with the high accuracy possible.

# # Optimization
# In this section, we will use the following techniques in order to improve the accuracy of the classifiers :
# *  Cross-validation
# *  Hyperparameter tuning

# ## Implementing a cross-validation based approach
# Here, we are going to implement K-folds cross-validation. For the value of K, I am going to use $K=5$.

# In[ ]:


# Import the cross-validation module
from sklearn.model_selection import cross_val_score

# Function that will track the mean value and the standard deviation of the accuracy
def cvDictGen(functions, scr, X_train = X, y_train = y, cv = 5):
    cvDict = {}
    for func in functions:
        cvScore = cross_val_score(func, X_train, y_train, cv = cv, scoring = scr)
        cvDict[str(func).split('(')[0]] = [cvScore.mean(), cvScore.std()]
    
    return cvDict


# In[ ]:


mod = [knMod, lrMod, adaMod, gbMod, rfMod]
cvD = cvDictGen(mod, scr = 'roc_auc')
cvD


# As we can see, in the above output, GradietBoosting and Adaboot classifier perform well. This cross-validation score helps in order to decide which model we should select and which ones we should not go with.  Based on the mean value and the standard deviation value, we can conclude that our ROC-AUC score does not deviate much, so we are not suffering from the overfitting issue.

# ## Implementing hyperparameter tuning
# Here, we will look at how we can obtain optimal values for the parameters. So, we are going to use the *RandomizedSearchCV* hyperparameter tuning method. We will implement this method for the AdaBoost and GradientBossting models since they are the one having good performance.

# In[ ]:


# Import methods
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


# ### AdaBoost

# In[ ]:


# Possible parameters
adaHyperParams = {'n_estimators': [10,50,100,200,420]}


# In[ ]:


gridSearchAda = RandomizedSearchCV(estimator = adaMod, param_distributions = adaHyperParams, n_iter = 5,
                                   scoring = 'roc_auc')
gridSearchAda.fit(X_train, y_train)


# In[ ]:


# Display the best parameters and the score
gridSearchAda.best_params_, gridSearchAda.best_score_


# The output above shows that the optimal value.

# ### GradientBoosting

# In[ ]:


# Possibles parameters
gbHyperParams = {'loss' : ['deviance', 'exponential'],
                 'n_estimators': randint(10, 500),
                 'max_depth': randint(1,10)}


# In[ ]:


# Initialization
gridSearchGB = RandomizedSearchCV(estimator = gbMod, param_distributions = gbHyperParams, n_iter = 10,
                                   scoring = 'roc_auc')
# Fitting the model
gridSearchGB.fit(X_train, y_train)


# In[ ]:


gridSearchGB.best_params_, gridSearchGB.best_score_


# The output above shows that the optimal  values

# ## Train models with help of new hyper parameter
# Here we are going to use the optimal parameter values that we got from the hyperparameter tuning.

# In[ ]:


# GradientBoosting with the optimal parameters
bestGbModFitted = gridSearchGB.best_estimator_.fit(X_train, y_train)


# In[ ]:


# AdaBoost with the optimal parameter
bestAdaModFitted = gridSearchAda.best_estimator_.fit(X_train, y_train)


# In[ ]:


functions = [bestGbModFitted, bestAdaModFitted]
cvDictbestpara = cvDictGen(functions, scr = 'roc_auc')
cvDictbestpara


# In[ ]:


# Getting the score GradientBoosting
test_labels = bestGbModFitted.predict_proba(np.array(X_test.values))[:,1]
roc_auc_score(y_test,test_labels , average = 'macro', sample_weight = None)


# In[ ]:


# Getting the score AdaBoost
test_labels = bestAdaModFitted.predict_proba(np.array(X_test.values))[:,1]
roc_auc_score(y_test,test_labels , average = 'macro', sample_weight = None)


# We can see in the above output that there is no such an improvement.  One can still ask this question : can we further improve the accuracy of the classifiers? Sure, there is always room for improvement.

# ## Problems with the optimization approach
# Up until, we did not spend a lot of time on feature engineering. So in our best possible approach, we spent time on the transformation of features engineering. We need to implement a voting mechanism in order to generate the final probability of the prediction on the actual test dataset so that we can get the best accuracy score.
# 
# These are the two techniques that we need to apply:
# * Feature transformation
# * An ensemble ML model with a voting mechanism

# ### Feature transformation (Feature engineering)
# We will apply standard scaler/log transformation to our training dataset. The reason behind this is that we have some attributes that are very skewed and some data attributes that have values that are more spread out in nature. 

# In[ ]:


# Import the log transformation method
from sklearn.preprocessing import FunctionTransformer, StandardScaler


# In[ ]:


transformer = FunctionTransformer(np.log1p)
scaler = StandardScaler()
X_train_1 = np.array(X_train)
#X_train_transform = transformer.transform(X_train_1)
X_train_transform = scaler.fit_transform(X_train_1)


# In[ ]:


bestGbModFitted_transformed = gridSearchGB.best_estimator_.fit(X_train_transform, y_train)
bestAdaModFitted_transformed = gridSearchAda.best_estimator_.fit(X_train_transform, y_train)


# In[ ]:


cvDictbestpara_transform = cvDictGen(functions = [bestGbModFitted_transformed, bestAdaModFitted_transformed],
                                     scr='roc_auc')
cvDictbestpara_transform


# In[ ]:


# For the test set
X_test_1 = np.array(X_test)
#X_test_transform = transformer.transform(X_test_1)
X_test_transform = scaler.fit_transform(X_test_1)


# In[ ]:


test_labels=bestGbModFitted_transformed.predict_proba(np.array(X_test_transform))[:,1]
roc_auc_score(y_test,test_labels , average = 'macro', sample_weight = None)


# ### Voting-based ensemble model
# In this section, we will use a voting-based ensemble classifier. So, we implement a voting-based machine learning model for both untransformed features as well as transformed features. Let's see which version scores better on the validation dataset.

# #### For transform data

# In[ ]:


# Import the voting-based ensemble model
from sklearn.ensemble import VotingClassifier


# In[ ]:


# Initialization of the model
votingMod = VotingClassifier(estimators=[('gb', bestGbModFitted_transformed), 
                                         ('ada', bestAdaModFitted_transformed)],
                                         voting = 'soft', weights = [2,1])
# Fitting the model
votingMod = votingMod.fit(X_train_transform, y_train)


# In[ ]:


test_labels=votingMod.predict_proba(np.array(X_test_transform))[:,1]
votingMod.score(X_test_transform, y_test)


# In[ ]:


# The roc_auc score
roc_auc_score(y_test, test_labels , average = 'macro', sample_weight = None)


# #### For untransform data

# In[ ]:


# Initialization of the model
votingMod_old = VotingClassifier(estimators = [('gb', bestGbModFitted), ('ada', bestAdaModFitted)], 
                                 voting = 'soft', weights = [2,1])
# Fitting the model
votingMod_old = votingMod.fit(X_train, y_train)


# In[ ]:


test_labels = votingMod_old.predict_proba(np.array(X_test.values))[:,1]
votingMod.score(X_test, y_test)


# In[ ]:


# The roc_auc score
roc_auc_score(y_test,test_labels , average = 'macro', sample_weight = None)


# In the both cases above we have achieved about $87\%$ accuracy.  This score is by far the most efficient accuracy as per industry standards.

# # Conclusion
# In this project we build a model that predict how likely a customer is going to churn. During exploratory data analysis we found out that the female customer are the most likely to churn, customer that are located in Germany are the most churned, and also customer using only one product are the most churned. After building several model we ended up with two  GradientBoosting and AdaBoost which performed better than others followed by Random Forest. We dicided to go further with the two and implemented a voting-based for the two which will allow us to choose the best model. Since the problem is about binary classification with a imbalance dataset, I have used the most efficient metric for model performance which is the ROC-AUC score and my model achieved about $87\%$ accuary. This score is by far the most efficient accuracy as per industry standards. The model can achieve better performance providing a lot of historical data for the training phase. For the next way to explore is may be try to group value in tenure column in trimester, semester and yearly.
