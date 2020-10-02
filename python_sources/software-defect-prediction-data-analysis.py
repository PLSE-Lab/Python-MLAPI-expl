#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization

#-- plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
#--

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/jm1.csv')


# > **About this Software Defect Prediction Dataset**
#  
# This is a Promise data set made publicly available in order to encourage repeatable, verifiable, refutable, and/or improvable predictive models of software engineering.
# 
# >***Attribute Information:***
#  1. loc                                    : numeric % McCabe's line count of code
#  2. v(g)                                  : numeric % McCabe "cyclomatic complexity"
#  3. ev(g)                                : numeric % McCabe "essential complexity"
#  4. iv(g)                                 : numeric % McCabe "design complexity"
#  5. n                                      : numeric % Halstead total operators + operands
#  6. v                                       : numeric % Halstead "volume"
#  7. l                                        : numeric % Halstead "program length"
#  8. d                                      : numeric % Halstead "difficulty"
#  9. i                                        : numeric % Halstead "intelligence"
#  10. e                                     : numeric % Halstead "effort"
#  11. b                                      : numeric % Halstead 
#  12. t                                      : numeric % Halstead's time estimator
#  13. lOCode                          : numeric % Halstead's line count
#  14. lOComment                  : numeric % Halstead's count of lines of comments
#  15. lOBlank                          : numeric % Halstead's count of blank lines
#  16. lOCodeAndComment  : numeric
#  17. uniq_Op                          : numeric % unique operators
#  18. uniq_Opnd                     : numeric % unique operands
#  19. total_Op                         : numeric % total operators
#  20. total_Opnd                    : numeric % total operands
#  21. branchCount                 : numeric % of the flow graph
#  22. defects                          : {false,true} % module has/has not one or more reported defects

# > **Data Discovery & Visualization**

# In[ ]:


data.info() #informs about the data (memory usage, data types etc.)


# In[ ]:


data.head() #shows first 5 rows


# In[ ]:


data.tail() #shows last 5 rows


# In[ ]:


data.sample(10) #shows random rows (sample(number_of_rows))


# In[ ]:


data.shape #shows the number of rows and columns


# In[ ]:


data.describe() #shows simple statistics (min, max, mean, etc.)


# In[ ]:


defects_true_false = data.groupby('defects')['b'].apply(lambda x: x.count()) #defect rates (true/false)
print('False : ' , defects_true_false[0])
print('True : ' , defects_true_false[1])


# > * **Histogram**

# In[ ]:


trace = go.Histogram(
    x = data.defects,
    opacity = 0.75,
    name = "Defects",
    marker = dict(color = 'green'))

hist_data = [trace]
hist_layout = go.Layout(barmode='overlay',
                   title = 'Defects',
                   xaxis = dict(title = 'True - False'),
                   yaxis = dict(title = 'Frequency'),
)
fig = go.Figure(data = hist_data, layout = hist_layout)
iplot(fig)


# > * **Covariance**
# 
# Covariance is a measure of the directional relationship between the returns on two risky assets. A positive covariance means that asset returns move together while a negative covariance means returns move inversely.

# In[ ]:


data.corr() #shows coveriance matrix


# > * **Heatmap**

# In[ ]:


f,ax = plt.subplots(figsize = (15, 15))
sns.heatmap(data.corr(), annot = True, linewidths = .5, fmt = '.2f')
plt.show()


# > *The light color in the heat map indicates that the covariance is high. (Ex. "v-b" , "v-n", etc.)*
# 
# > *The dark color in the heat map indicates that the covariance is low. (Ex. "loc-l" , "l-d", etc.)*

# > * **Scatter Plot **

# In[ ]:


trace = go.Scatter(
    x = data.v,
    y = data.b,
    mode = "markers",
    name = "Volume - Bug",
    marker = dict(color = 'darkblue'),
    text = "Bug (b)")

scatter_data = [trace]
scatter_layout = dict(title = 'Volume - Bug',
              xaxis = dict(title = 'Volume', ticklen = 5),
              yaxis = dict(title = 'Bug' , ticklen = 5),
             )
fig = dict(data = scatter_data, layout = scatter_layout)
iplot(fig)

#two attributes with high correlation v-b > just about 1


# > **Data Preprocessing**

# In[ ]:


data.isnull().sum() #shows how many of the null


# > *No missing value. *
# 
# > *No data cleaning needed because the data is all important.*

# >*  **Outlier Detection (Box Plot)**

# In[ ]:


trace1 = go.Box(
    x = data.uniq_Op,
    name = 'Unique Operators',
    marker = dict(color = 'blue')
    )
box_data = [trace1]
iplot(box_data)


# *Showing all information when clicking on plot (min, max, q1, q2, etc.).*

# >*  **Feature Extraction**

# In[ ]:


def evaluation_control(data):    
    evaluation = (data.n < 300) & (data.v < 1000 ) & (data.d < 50) & (data.e < 500000) & (data.t < 5000)
    data['complexityEvaluation'] = pd.DataFrame(evaluation)
    data['complexityEvaluation'] = ['Succesful' if evaluation == True else 'Redesign' for evaluation in data.complexityEvaluation]


# In[ ]:


evaluation_control(data)
data


# In[ ]:


data.info()


# In[ ]:


data.groupby("complexityEvaluation").size() #complexityEvalution rates (Succesfull/redisgn)


# In[ ]:


# Histogram
trace = go.Histogram(
    x = data.complexityEvaluation,
    opacity = 0.75,
    name = 'Complexity Evaluation',
    marker = dict(color = 'darkorange')
)
hist_data = [trace]
hist_layout = go.Layout(barmode='overlay',
                   title = 'Complexity Evaluation',
                   xaxis = dict(title = 'Succesful - Redesign'),
                   yaxis = dict(title = 'Frequency')
)
fig = go.Figure(data = hist_data, layout = hist_layout)
iplot(fig)


# > * **Data Normalization  (Min-Max Normalization)**

# In[ ]:


from sklearn import preprocessing

scale_v = data[['v']]
scale_b = data[['b']]

minmax_scaler = preprocessing.MinMaxScaler()

v_scaled = minmax_scaler.fit_transform(scale_v)
b_scaled = minmax_scaler.fit_transform(scale_b)

data['v_ScaledUp'] = pd.DataFrame(v_scaled)
data['b_ScaledUp'] = pd.DataFrame(b_scaled)

data


# In[ ]:


scaled_data = pd.concat([data.v , data.b , data.v_ScaledUp , data.b_ScaledUp], axis=1)
scaled_data


# >**Model Selection**

# >* **Naive Bayes**

# In[ ]:


data.info()


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import model_selection

X = data.iloc[:, :-10].values  #Select related attribute values for selection
Y = data.complexityEvaluation.values   #Select classification attribute values


# In[ ]:


Y


# In[ ]:


#Parsing selection and verification datasets
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = validation_size, random_state = seed)


# In[ ]:


#Creation of Naive Bayes model
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()


# In[ ]:


#Calculation of ACC value by K-fold cross validation of NB model
scoring = 'accuracy'
kfold = model_selection.KFold(n_splits = 10, random_state = seed)
cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv = kfold, scoring = scoring)


# In[ ]:


cv_results


# In[ ]:


msg = "Mean : %f - Std : (%f)" % (cv_results.mean(), cv_results.std())
msg


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
#Accuracy score
from sklearn.metrics import accuracy_score
print("ACC: ",accuracy_score(y_pred,y_test))


# > * **Linear Regression**

# In[ ]:


sel_loc = data['loc']
sel_b = data['b']
selected_data = pd.concat([sel_loc, sel_b], axis=1)
selected_data
#data selected for selection


# In[ ]:


selected_data.describe() #shows simple statistics (min, max, mean, etc.)


# In[ ]:


selected_data.corr() #shows coveriance matrix


# In[ ]:


#Scatter Plot
trace = go.Scatter(
    x = data['loc'],
    y = data.b,
    mode = "markers",
    name = "Line of Code - Bug",
    marker = dict(color = 'darkmagenta'),
    text = "Bug (b)")

scatter_data = [trace]
scatter_layout = dict(title = 'Line of Code - Bug',
              xaxis = dict(title = 'Line of Code', ticklen = 5),
              yaxis = dict(title = 'Bug' , ticklen = 5),
             )
fig = dict(data = scatter_data, layout = scatter_layout)
iplot(fig)


# In[ ]:


Y = selected_data['b'].values  
X = selected_data['loc'].values  
X = X.reshape(-1,1)
#Select the X and Y values for selection
Y


# In[ ]:


#Parsing selection and verification datasets
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0) 


# In[ ]:


#Creation of Linear Regression model
from sklearn.linear_model import LinearRegression  
model = LinearRegression()  
model.fit(X_train, y_train)  


# In[ ]:


# Intercept & Coef
print("Intercept :", model.intercept_)  
print("Coef :", model.coef_)


# In[ ]:


X_test


# In[ ]:


y_pred = model.predict(X_test) 


# In[ ]:


# New data (real , estimated)
new_data = pd.DataFrame({'real': y_test, 'estimated': y_pred})  
new_data


# In[ ]:


#The nearest line of all values in the model
plt.scatter(X_train, y_train, color = 'red')
modelin_tahmin_ettigi_y = model.predict(X_train)
plt.plot(X_train, modelin_tahmin_ettigi_y, color = 'black')
plt.title('Line of Code - Bug', size = 15)  
plt.xlabel('Line of Code')  
plt.ylabel('Bug')  
plt.show() 


# In[ ]:


#The results of the model. (This uses the Least squares method and the Root mean square error methods)
#In general, as these values are calculated as the mean value and the difference difference, it is considered that the model has better estimation ability as it approaches 0.
from sklearn import metrics   
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  


# >*When we look at the values, the fact that the values are close to zero shows us that the model has good predictive ability.*

# *- THE END -*
