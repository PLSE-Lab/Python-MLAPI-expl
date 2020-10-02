#!/usr/bin/env python
# coding: utf-8

# <center>
# 
# ## **<font color='green'>Heart disease UCI - Exploratory Data Analisys (EDA)**</font> 

# <center>
# <img src="http://reillyrangel.com.br/wp-content/uploads/2016/10/medicina-saude-tecnologia-computacao-microsoft-cancer-google-apple-ibm-reilly-rangel-s.jpg" width=50%/>

# Author: [Paulo Henrique Zen Messerschmidt](https://www.linkedin.com/in/paulo-henrique-zen-messerschmidt-35581661/)
# 
# 

# This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular, the Cleveland database is the only one that has been used by ML researchers to this date. ([Kaggle DataSet](https://www.kaggle.com/ronitf/heart-disease-uci))
# 
# The "target" feature refers to the presence of heart disease in the patient. It is integer value: 0 (no presence); 1 (presence of heart disease).

# The description of the dataset features can be found in [this link](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
# 

# ### Importing libraries: preparing the enviroment.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py


import warnings
warnings.filterwarnings('ignore')

py.offline.init_notebook_mode(connected=True)
import plotly.graph_objs as go

get_ipython().run_line_magic('matplotlib', 'inline')
# Plot in SVG format since this format is more sharp and legible
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")


# In[ ]:


path = '../input/heart.csv'
df = pd.read_csv(path)
df.head()


# **Features**:
# - Age: age in years;
# - Sex: 1 = male; 0 = female;
# - Cp: chest pain type:  0 - Typical angina; 1 - atypical angina; 2 - non-anginal pain; 3 - asymptomatic;
# - trestbps: resting blood pressure (in mm Hg on admission to the hospital)
# - chol: serum cholestoral in mg/dl
# - fbs: fastin blood sugar: 1 - true; 0 - false)
# - restecg: resting electrocardiographic results: 0 - normal ; 1 - wave abornamility - T wave inversions and/or ST elevation or depression of > 0.05 mV; 2 - showing probable or definite left ventricular hypertrophy by Estes' criteria
# - thalach: maximum heart rate achieved
# - exang: exercise induced angina: 1 - yes; 0 - no
# - oldpeak: ST depression induced by exercise relative to rest 
# - slope: the slope of the peak exercise ST segment: 0 - upsloping; 1 - flat; 2 - downsloping;
# - ca: number of major vessels (0-3) colored by flourosopy
# - <font color='green'>thal: 1 - normal; 2 - fixed defect; 3 - reversable defect **CHECK** </font>
# - target: 0 - Healthy; 1 - sick (heart disease)

# **We can divide these features in two groups: quantitative and categorical**
# - Quantitative features: Age, trestbps, chol, thalach, oldpeak
# - Categorical features: sex, cp, fbs, restecg, exang, slope, ca, thal
# 
# 
# Obs.: our target feature is categorical (classification problem)

# In[ ]:


num_cont_feat = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
cat_feat = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']


# For now we don't need to operate categorical variables in numerical format. Let's transform our categorical variables (sex and exang) to facilitate reading.
# 

# In[ ]:


df['sex'] = df['sex'].apply(lambda x: 'male' if x == 1 else 'female')
df['exang'] = df['exang'].map({1: 'Yes', 0:'No'})


# ### Dataset analysis

# **Let's take a look in the dataset shape**

# In[ ]:


df.shape


# 303 samples and 14 features

# **Verifying the features name**

# In[ ]:


df.columns


# **Checking the data type of each column**

# In[ ]:


df.info()


# That's good, there's no missing values (all features has 303 samples).

# **Let's see the statistics of our dataset.**

# In[ ]:


df.describe()


# Highlights:
# 
# - 25% of samples are around 61 years old or older.
# - 50% of samples have a cholesterol level up to 240.
# 

# # Data visualization

# ### Univariate visualization

# Let's configure our plots (size and style).

# In[ ]:


plt.rcParams['figure.figsize']= (12,8) # figure size
sns.set_style('darkgrid') # Style


# ## Age

# **Let's plot a histogram for age using matplotlib**

# In[ ]:


df['age'].hist(grid=True, bins=10); 
plt.title('Age distribuition')


# Apparently, the age feature not follow a normal distribution. It's not so clear in this chart. Let's plot a density plot using seaborn.
# 
# Obs.: bin param can affect the shape of distribution.

# **Density plots using seaborn**

# In[ ]:


sns.distplot(df[df['sex']=='female']['age'], rug=True, hist=True, label='female')
sns.distplot(df[df['sex']=='male']['age'], rug=True, hist=True, label='male')
plt.legend()
plt.title('Density plot of age by sex');


# 
# Density graph shows the smoothed distribution of points along the numerical axis. The density peaks
# where there is the highest concentration of points. In sum, density graphs can be considered smoothed histograms.
# 
# Recommended materials: [1](https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Density_Plots.pdf); [2](https://homepage.divms.uiowa.edu/~luke/classes/STAT4580/histdens.html)

# ### Using Plotly

# **Let's draw a histogram for the "Age" feature**

# In[ ]:


age = df['age']
layout = go.Layout(barmode='overlay')
data = go.Histogram(x=age, opacity=0.6, xbins={'size': 4})
fig = go.Figure(data=[data], layout=layout)
py.offline.iplot(fig)


# ## **Resting blood pressure (in mm Hg on admission to the hospital)**

# Let's check the distribuition

# In[ ]:


df['trestbps'].hist()
plt.title('Resting Blood pressure distribuition')


# Interesting, it seems that we have a normal distribution

# ### Let's create a density plot

# In[ ]:


sns.distplot(df['trestbps'], bins=10)
plt.title('Resting Blood pressure desnity plot');


# Let's plot a histogram for all continuous variables using the df.hist()

# In[ ]:


plt.rcParams['figure.figsize']= (15,8) # reajustar o tamanho da figura 

df[[ 'age','trestbps', 'chol', 'thalach', 'oldpeak']].hist();


# 
# - Chol: apparently is quite close to a normal distribution. However it is possible to notice a high value (acmia of 500) that can be a possible outlier!
# 
# - Oldpeak: It seems to follow a left-skewed distribution (lognormal).
# 
# - Thalach: the maximum heart rate achieved seems to follow a right-skewed distribution
# 
# - Trestbps: Resting blood pressure appears to follow a normal distribution.

# **Let's check outliers presence using boxplots**

# In[ ]:


fig, axes = plt.subplots(nrows = 1, ncols=2)
sns.boxplot(x='chol', data=df, orient='v', ax=axes[0])
sns.boxplot(x='oldpeak', data=df,  orient='v', ax=axes[1]);


# ## Categorical features

# Let's start with the target variable to see the rate of people with and without heart disease

# In[ ]:


df['target'].value_counts()


# The number of unhealthy people is higher than the number of healthy people.

# Let's check the proportion of men and women

# In[ ]:


df['sex'].value_counts()


# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(17,10))
cat_feat = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']

for idx, feature in enumerate(cat_feat):
    ax = axes[int(idx/4), idx%4]
    if feature != 'target':
        sns.countplot(x=feature, hue='target', data=df, ax=ax)


# Obs.: the error code above appears because we have 9 features (include target) but we create only 8 axes in subplots.
# 
# Let's get some insights frm this chart:
# 
# Chest pain: the heart desease diagnosis is greater among the patients that feel any chest pain.
# 
# Restegc - Eletrocardiagraph results: the rate of heart desease diagnoses higher for patients with a ST-T wabe abnormality .
# 
# Slope: The ratio of patients diagnosed with heart desease is higher for slope = 2 
# 
# Ca: The diagonosed ratio decreases fo ca between 1 and 3.
# 
# Thal: the diagnosed ratio is higher for thal = 2.

# Let's perform a multivariate analysis, comparing the number of healthy and unhealthy people by sex.
# 

# In[ ]:


plt.rcParams['figure.figsize'] = (10,8)
sns.countplot(x='target', hue='sex', data=df);
plt.title('Count of target feature by sex')


# The amount of healthy male people is greater than the amount of unhealthy. For women, the number of unhealthy women is higher. Let's create an index.

# In[ ]:


pd.crosstab(df['sex'], df['target'], normalize=True)


# We can see that most are healthy and are male.
# 
# Let's create one for each of the groups

# In[ ]:


sex_target = df.groupby(by=['sex', 'target']).size()
sex_target_pcts = sex_target.groupby(level=0).apply(lambda x: 100*x/x.sum())

sex_target_pcts


# ### Chest Pain type

# In[ ]:


sns.countplot(x='cp', hue='target', data=df)


# Most patients who experience some type of pain have heart disease

# ## Multivariate visualization

# First let's evaluate the correlation between the numeric variables in our dataset. This information is important because some machine learning algorithms can not handle correlated input variables, such as linear and logistic regression.

# Let's transform  `sex` and `exang` features applying `map()`.

# In[ ]:


df['sex'] = df['sex'].map({'female': 0, 'male': 1})
df['exang'] = df['exang'].map({'No': 0, 'Yes': 1})


# Let's create a correlation matrix using `sns.heatmap()`.

# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(df.drop(['target', 'sex', 'cp', 'fbs'], axis=1).corr(), annot=True, cmap='coolwarm');


# Apparently there are no features with a pretty strong correlation (above |0.7|)

# ## Scatterplot matrix

# When the number of variables is relatively small we can plot a grid with all variables. The diagonal line shows the distribution of each variable.

# In[ ]:


plt.figure(figsize=(30,30))
sns.set_style('darkgrid')
sns.pairplot(df[num_cont_feat])


# ## ScatterPlot

# In[ ]:


plt.rcParams['figure.figsize'] = (8,8)
sns.scatterplot(x='chol', y='trestbps', hue='sex', size=None, data=df)
plt.title(' Cholesterol vs Blood pressure in rest')


# As can be seen there is a pacient with high cholesterol. But, there's not a specific division between those that feel pain during exercise practice and those of not feel pain.
# We can use hue to filter by sex. It's also possible to filter using size = 'label_to_filer'.

# We can use a jointplot to get a scatter plot and a histogram of each feature.

# In[ ]:


sns.jointplot(x='thalach', y='trestbps',  data=df)


# We can set the parameter kind='kde' to get a smoothed version.

# In[ ]:


sns.jointplot(kind='kde', x='thalach', y='trestbps', data=df)


# This plot uses a color scale to represent the density, i.e. the local where the majority of the data points fall. In this case, the points are concentrated around 170, 130 in the axis x and y respectively

# In[ ]:


sns.scatterplot(x='age', y='chol', hue='target', data = df)


# We can do de same using `plt.scatter()`

# In[ ]:


plt.figure()
plt.scatter(df[df['target'] == 0]['age'], df[df['target'] == 0]['chol'], marker='o', c='blue', label='healthy')
plt.scatter(df[df['target'] == 1]['age'], df[df['target'] == 1]['chol'], marker='x', c='red', label='sick')

plt.legend()


# ## Boxplots - Quantitative/Categorical Plots

# Let's visualize the distribuition of chol by sex

# In[ ]:


sns.boxplot(x='sex', y='chol', data=df)


# Apparently, female patients has  higher cholesterol indices than male patients.

# Let's plot a `catplot` to visualize how the cholesterol and chest pain type are relationed.

# In[ ]:


plt.figure(figsize=(15,10))
sns.catplot(x='sex', y='chol', col='target', data=df, kind='box', height=4, aspect=.8)


# Let's visualize if the diagnose is influenced by age

# In[ ]:


sns.boxplot(y='age', x='target', data = df)


# It's no clearif the age of patient influence on diagnosis. Apparently not.

# We can use plotly to plot the boxplots

# In[ ]:


# First let's create a list to append the data to be plotted
data = []
for pain in df.cp.unique():
    data.append(go.Box(y=df[df.cp == pain].chol, name=str(pain)))

layout = go.Layout(yaxis=dict(title ='Cholesterol', zeroline=False))
                   
fig = go.Figure(data=data, layout=layout)               
py.iplot(fig, show_link=False)


# Apparently, higher colesterol values are relationed with these three types of pain. (See the max values)

# In[ ]:


# First let's create a list to append the data to be plotted
data = []
for target in df.target.unique():
    data.append(go.Box(y=df[df.target == target].thalach, name=str(target)))

layout = go.Layout(yaxis=dict(title ='maximum heart rate achieved', zeroline=False))
                   
fig = go.Figure(data=data, layout=layout)               
py.iplot(fig, show_link=False)


# Those patients that was diagnosed with heart desease had higer maxium heart rate.

# ## Prediction model

# Firstly, let's consider the problem context  before testing different algorithms for classification task.
# 
# Here, we'll first create a Decision Tree Model, for main two reasons:
# - It's a simple and well know algorithm. 
# - We can use flow diagrams to visual representations of decisions trees. Since we are predictin heart desease, we can use it to sharing the knowledge with a medical in order to ensure that our model make sense;

# **Firts let's split our dataset in train and test**

# In[ ]:


# Import train test split

from sklearn.model_selection import train_test_split


# In[ ]:


df.head()


# Here, we don't need to standardize the data in a scale because decisions tree it's not affected by data scale.
# 

# In[ ]:


# Split the DataFrame into a matrix X and vecto Y which form the train set
X, y = df.drop('target', axis=1), df['target']


# In[ ]:


X.shape, y.shape


# Here, we gonna create a train split and a holdout split, because we'll perform a [cross-validation](https://towardsdatascience.com/cross-validation-70289113a072) on our train split. 

# In[ ]:


X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.3, random_state = 17)


# In[ ]:


X_train.shape, X_holdout.shape


# Here, we'll use StratifiedKFold, since we want to keep the same proportion of labels in the train and test fold.
# We'll also use GridSearchCV, that allows us to find the best hyperparameter combination.

# In[ ]:


from sklearn.model_selection import GridSearchCV, StratifiedKFold


# In[ ]:


# Let's specify 5 kfold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)


# In[ ]:


# Let's create our hyperparameter grid using a dictionary

params = {'max_depth': np.arange(2,10), 
         'min_samples_leaf': np.arange(2,10),
         }


# Importing the DecisionTree Classifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[ ]:


tree = DecisionTreeClassifier(random_state=17)


# In[ ]:


best_tree = GridSearchCV(estimator = tree, param_grid=params, n_jobs=1, verbose=1)


# In[ ]:


best_tree.fit(X_train, y_train)


# In[ ]:


best_tree.best_params_


# In[ ]:


best_tree.best_estimator_


# In[ ]:


best_tree.best_score_


# In[ ]:


pred_holdout_better = best_tree.predict(X_holdout)


# In[ ]:


accuracy_score(y_holdout, pred_holdout_better)


# Let's visualize how decisions tree create the decisions rules

# In[ ]:


# First we'll import graphviz from sklearn.tree
from sklearn.tree import export_graphviz


# In[ ]:


export_graphviz(decision_tree=best_tree.best_estimator_,
               out_file='tree.dot', filled=True, 
                feature_names=df.drop('target', axis=1).columns)

from subprocess import call

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

from IPython.display import Image
Image(filename = 'tree.png')


# Considering that we don't want to avoid [type II errors](https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62) let's check our f1score.
# 
# 

# In[ ]:


from sklearn.metrics import f1_score


# In[ ]:


print(f1_score(y_holdout, pred_holdout_better))


# As can be seen, our f1-score is 0.75

# Let's investigate the feature importances from our model, in order to check if we can create a simpler model.

# In[ ]:


importances = best_tree.best_estimator_.feature_importances_


# In[ ]:


plt.figure(figsize=(10,4))
plt.bar(X_train.columns.values,importances)


# As can be seen, fbs, sex and thal are not useful to predict heart disease.
# Let's create a model without these features to see how the accuracy is affected.

# ## Reduced model

# Let's import [clone](https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html) from sklearn.

# In[ ]:


from sklearn.base import clone
from sklearn.metrics import f1_score, accuracy_score


# In[ ]:


X_train_reduced = X_train.drop(['sex', 'fbs', 'thal'], axis=1)
X_holdout_reduced = X_holdout.drop(['sex', 'fbs', 'thal'], axis=1)


# In[ ]:


X_train_reduced.shape, X_holdout_reduced.shape


# In[ ]:


# Train on the "best" model found from grid search earlier
tree_reduceded_clone = (clone(best_tree.best_estimator_)).fit(X_train_reduced, y_train)

# Make new predictions
reduced_predictions = tree_reduceded_clone.predict(X_holdout_reduced)

print("\nFinal Model trained on reduced data\n------")
print("Accuracy on holdout data: {:.4f}".format(accuracy_score(y_holdout, reduced_predictions)))
print("F1-score on holdout data: {:.4f}".format(f1_score(y_holdout, reduced_predictions)))


# In the case of large datasets, it is more advantageous to work with a smaller number of  features, since the model can improve its accuracy, improve its efficiency and reduce the computational cost.
