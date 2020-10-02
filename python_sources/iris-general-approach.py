#!/usr/bin/env python
# coding: utf-8

# In this kernel, we will see how to approach a machine learning problem using a small and well-known data set. We will perform all bit of analysis to understand relationships among features so we can create new ones, this involves some knowledge on the field that, in this case, can be obtained from Wikipedia, but you can also use the research department of your company as well your friend who owes you a favor.
# 
# Of course, some of the approaches described here are over engineered since the data is small but will need present for completeness.
# 
# ### Outline
# 
# 
# - Identify our problem and goal
# - Get the necessary data
# - Identify the target variable and see if there is skewness on target
# - Clean up
#     - Dump unnecessary data such as ids
#     - Search for Nulls and fill them if necessary
#     - Search for scaling
# - Feature Engineering
# - Preprocessing
# - Select a Model
# - Assemble the Pipeline
# - Example of an API
#    
#  
# 
# 
# 

# ![Iris sibirica](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Sibirische_Schwertlilie%2C_Iris_sibirica_06.JPG/400px-Sibirische_Schwertlilie%2C_Iris_sibirica_06.JPG)
# [Iris sibirica](https://en.wikipedia.org/wiki/Iris_sibirica)

# ### The problem
# 
# We want to classify plants of the iris genius, give it's width and length measures (sepal and petal), betweem some species.

# In[ ]:


import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')
sns.set_palette('Set2')
np.random.seed(1998)


# In[ ]:


df = pd.read_csv('../input/Iris.csv')


# In[ ]:


df.head()


# ## Identify the target variable
# Our target variable is the *Species* column on our data frame, it represent as stated the flower species
# 
# With the target we can see if is everything alright if it, that is, there is no skewness, neither outliers.

# In[ ]:


plt.figure(figsize=(10,5))
plt.title('Species Variable Counts')
_ = sns.countplot(data=df, x='Species')


# As we can see all classes of the target is evenly distributed, so we are good to go.

# ## Clean Up
# 
# It's time to start cleaning our data, we do that by search for *null, nan, n\a* values and filling them if possible, otherwise dropping if their percentage is too high to obtain a decent *mean* value

# In[ ]:


# lets see if there is null values
series = df.isnull().sum()
pd.DataFrame({'Column Name': series.index, 'Has Nulls ?': ['Yes' if v > 0 else 'No' for v in series.values]})


# Since this is a well worked out data set, there are no missing values, so we can move on

# ### There are outliers on the data ?
# 
# We will look up for outliers within the data, outliers can also damage the model perfomance, because they are too far from the mean, the model needs to adjust it's weights to compensate this, but outliers are just a minimal amout of data that doesn't need fit the model because it can't generalize enough, damaging the greater amount of data around the mean

# In[ ]:


fig, axs = plt.subplots(2, 2, figsize=(20, 10))
_ = sns.boxplot(data=df, x='Species', y='SepalWidthCm', ax=axs[0][0])
_ = sns.boxplot(data=df, x='Species', y='SepalLengthCm', ax=axs[0][1])
_ = sns.boxplot(data=df, x='Species', y='PetalWidthCm', ax=axs[1][0])
_ = sns.boxplot(data=df, x='Species', y='PetalLengthCm', ax=axs[1][1])


# Boxplots are defined on [Wikipedia](https://en.wikipedia.org/wiki/Box_plot) as:
# 
# > Box and whisker plots are uniform in their use of the box: the bottom and top of the box are always the first and third quartiles, and the band inside the box is always the second quartile (the median). But the ends of the whiskers can represent several possible alternative values, among them:
# 
# >- the minimum and maximum of all of the data[1] (as in figure 2)
# - the lowest datum still within 1.5 IQR of the lower quartile, and the highest datum still within 1.5 IQR of the upper quartile (often called the Tukey boxplot)[2][3] (as in figure 3)
# - one standard deviation above and below the mean of the data
# - the 9th percentile and the 91st percentile
# - the 2nd percentile and the 98th percentile.
# 
# > Any data not included between the whiskers should be plotted as an outlier with a dot, small circle, or star, but occasionally this is not done.

# With that kind of plot we can see that it draws some outliers, but in that case they are outliers to the whiskers, their distance is still minimal from them, so we will not consider them outliers for our problem, but instead, if they were values like 0.1 for Iris-Virginica's Sepal Length, then it would be a outlier.

# 
# 
# ### Is the data on same scale ?
# 
# With this we will check if the data is on the same scale of values, so we might apply scaling or not. Values with a ratio to high can [damage](https://stats.stackexchange.com/questions/41704/how-and-why-do-normalization-and-feature-scaling-work) the model perfomance.
# 
# We might see if all our data is between a given interval, of couse, this is a gut feeling measure, since I don't know an automatic and mathematical way to check it.

# In[ ]:


mean = df.mean()
pd.DataFrame({'Column Name': mean.index, 'Needs Scale ?': ['No' if  1 < v < 10 else 'Yes' for v in mean.values]})


# Ok, we got that our id needs scaling, buuut, we are going to drop it, since it just a number of the example, not a feature to take to the model

# In[ ]:


# lets drop the id since it is useless
df.drop('Id', axis=1, inplace=True)


# ## Feature Engineering
# 
# Since our data is all cleaned and ready, we can start working on new features to improve our model, of course, in this dataset we already get a good accuracy out of the box, but since we are over engineering here we are going to generate some.
# 

# In[ ]:


# make a copy to not mess visualizations
edf = df.copy()


# ### Looking up for correlations
# 
# Correlation is a way to see the dependency between variables, further from 0.0 strongly is the dependency, low dependency can represent variables that can be dropped, and others that can be amplified.
# 

# In[ ]:


corr = df.corr()
ax = sns.heatmap(corr, annot=True, linewidth=0.3, linecolor='w')
_ = plt.xticks(rotation=90)


# The heatmap show a strongly dependency between every variable, with the exception of Sepal Width vs Sepal Length

# ### Creating new features
# 
# Observing the features, we can see that for both sepal and petal we have length and width and petal and sepal length have strong correlation, so makes sense for the problem to generate a ratio between the two so we are applying the simple division function: $ratio(w, l) = \frac{w}{l}$

# In[ ]:


edf['SepalRatio'] = edf['SepalWidthCm'] / edf['SepalLengthCm'] 
edf['PetalRatio'] = edf['PetalWidthCm'] / edf['PetalLengthCm']


# In[ ]:


edf.head()


# Seems we just yourselves into a problem, now if we run the scaling checker as we did above, our new features will need scaling because they are less than 1

# In[ ]:


mean = edf.mean()
pd.DataFrame({'Column Name': mean.index, 'Needs Scale ?': ['No' if  1 < v < 10 else 'Yes' for v in mean.values]})


# Since that is enough of generating features for it, we can scale everything and start our model

# ## Preprocessing
# 
# Here is what we will perform:
# - Get the features from dataframe
# - Get the labels from dataframe
# - Scale the features
# - Encode the labels

# In[ ]:


X = edf.drop('Species', axis=1).as_matrix()
y = edf['Species'].values


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
# scale the features
X_scaler = MinMaxScaler()
X_scaled = X_scaler.fit_transform(X)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
y_encoder = LabelEncoder()
y_encoded = y_encoder.fit_transform(y)


# ## Model Selection
# 
# Now we have everything ready to search for a model, since this is a small problem we can use the GradientBoosting and decide if we need to try another one.
# 
# We will split the data set into a train and a small part of test, then we will train the classifier, using *GridSearchCV* to automatically search for best params for the model given an array of options. To finish we see the accuracy of the prediction, that not the only method and one of the last to be used after the model is selected using *cross validation* techiniques

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y) # we use stratify to split equally the classes

params = {'learning_rate': [0.1, 0.2, 0.4, 0.6], 'n_estimators': [100, 200, 300]}

clf = GridSearchCV(GradientBoostingClassifier(), cv=5, param_grid=params)

clf.fit(X_train, y_train)
print('Trained GradientBoostingClassifier')
print('-'*40)
print('Best Features ', clf.best_params_)


# In[ ]:


score = accuracy_score(y_test, clf.predict(X_test))
print("Accuracy Score {0:.2f}%".format(score * 100))


# ## Assemble the Pipeline
# 
# Now we will tie all the process into a pipeline where the X features will predict the class y of the given input, the parts we will include are:
# 
# - MinMaxScaler
# - Classifier
# 
# With pipeline built we can serialize it on a file so we can put on production somewhere so botanists can we do automatically classify their measures. We can use the default package from Python, named `pickle`. Also in the end of the pipeline we will use our fitted LabelEncoder to decode the predicted class.

# In[ ]:


import joblib
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class IrisRatioTransfomer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    
    
    def transform(self, measures_matrix):
        """Given an numpy matrix, it compute the measures of rate betweeen the pairs of information
        Example:
            given: np.ndarray([[1, 2, 3, 4], [1, 2, 3, 4]])
                representing sepal length, sepal width, petal length, petal width
            then: it will return the ration between 2, 1 and 4, 3, appended to the array
                as last positions
        """
        try:
            row_collector = []
            for row in measures_matrix:
                sepal_ratio = row[1] / row[0]
                petal_ratio = row[3] / row[2]
                
                row_collector.append([*row, sepal_ratio, petal_ratio])
            return np.array(row_collector)
        except KeyError as e:
            print('The input data is mal-formed, please check the documentation')
            raise KeyError(e)
    
    
pipeline = Pipeline([
    ('Iris Ratio Transformer', IrisRatioTransfomer()),
    ('Feature Scaler', X_scaler),
    ('Estimator', GradientBoostingClassifier()) # parameters got from the GridSearch, and the default ones
])

# fit the pipeline with raw data, let it do the transformations we did manually
pipeline.fit(df.drop('Species', axis=1).as_matrix(), y_encoder.transform(df['Species'].values))
# save the fitted model with the encoder for decoding
assemble = (pipeline , y_encoder)
# dump the pipeline and the label encoder so we can use at another place
joblib.dump(assemble, 'iris.pkl')
print('ok.')


# ## Defining an API
# 
# We will define a simple API to load the model, receive measures, compute the features and return the class

# In[ ]:


import joblib

class Measure():
    def __init__(self, sepal_width, sepal_length, petal_width, petal_length):
        self.data = [
            sepal_width,
            sepal_length,
            petal_width,
            petal_length
        ]

        

class Botanist():
    
    def __init__(self):
        "Load the assemble from pipeline step, and assign them to fields acessible from this class"
        
        pipeline, label_encoder = joblib.load('iris.pkl')
        self.pipeline = pipeline
        self.label_encoder = label_encoder
    
    def predict(self, measure):
        """
        Predict the species of Iris based on it's measures.
        This method accepts a single entry, for array computation use
        `bulk_predict`.
        Returns the name of the specie
        """
        return self.bulk_predict([measure])[0]
        
        
    def bulk_predict(self, measures):
        """
        Predict the species of Iris based on it's measures.
        This an array of Measure, for single computation use
        `predict`.
        Returns an array containing the names of the species on
        the same order as defined on the measures array.
        """
        feats = [measure.data for measure in measures]
        predicted = self.pipeline.predict(feats) # we need a matrix
        return self.label_encoder.inverse_transform(predicted)
        


# In[ ]:


b = Botanist()


# In[ ]:


b.predict(Measure(5.1,3.5,1.4,0.2))


# In[ ]:


measures = [
    Measure(6.7, 3.0, 5.2, 2.3), # virginica
    Measure(4.9, 3.0, 1.4, 0.2), # setosa
    Measure(5.9, 3.0, 5.1, 1.8), # virginica
    Measure(5.0, 2.3, 3.3, 1.0), # versicolor
    Measure(4.5, 3.8, 1.7, 0.1) # made up
]

results = b.bulk_predict(measures)
for r in results:
    print('The specie measured is a', r)


# ## My two cents
# Even if this kernel was fast enough to reach the point of assembling a good model, the real world is slower than that. Our data was too small to cause problems of interpretation for us, we didn't even need to consult a botanic for advice, also we are not expected to know everything about all, we should know how to apply data science and machine learning modeling, not even both, because they are a vast field on each own. On real data we should spend more time digging the meaning of variables, which of them can be turned into features, we need to experiment on each one has a linear dependency on another and if it makes sense to turn that dependency into a nonlinear, we must sleep together with the data.

# *All based on the experience of myself, as an entry guy on the field*
# 
# 
# References:
#  - [sckit-lean](scikit-learn.org)
#  - [SciPy](https://www.scipy.org)
#  - [Python Data Visualizations](https://www.kaggle.com/benhamner/python-data-visualizations)
# 

# In[ ]:




