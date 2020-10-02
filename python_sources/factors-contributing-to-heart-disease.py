#!/usr/bin/env python
# coding: utf-8

# # Heart Disease - An Exploratory Analysis
# 
# The purpose of this notebook is to perform some insightful visuals while keeping the code as simple as possible. We then move to to use these insights to guide our models for prediction. <br>
# The dataset has been downloaded from [here](https://archive.ics.uci.edu/ml/datasets/Heart+Disease). You can also find a full description of the data on the same webpage.
# 

# ## Data Preparation

# In[ ]:


# We first import important packages for our analysis
import pandas as pd # pandas and numpy are for data analysis and manipulation
import numpy as np
import matplotlib.pyplot as plt # matplotlib and seaborn allow for great visualization
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv') # make sure the dataset in the same directory as the notebook
df.head()


# In[ ]:


def get_shape(df, name = 'dataframe'):
    dims = len(df.shape)
    if dims == 1:
        print('The {} has {} rows and {} column.'.format(name, df.shape[0], 1))
    elif dims == 2:
        print('The {} has {} rows and {} columns.'.format(name, df.shape[0], df.shape[1]))
    
get_shape(df)


# In[ ]:


df.isnull().sum() # Number of NA values by column


# ### Unclear column names and values
# The dataset pretty much looks clean as there are no null values. However, there is still some data preparation needed.<br>
# As we can see, the column names and the values of columns are not very clear. For instance, the values for sex are 0 and 1. This might not be very helpful for visualizatoins as it will be hard to understand the relationship between different data points. We'll create meaningful column names and column values so that anyone can understand the visualizations without going through the description.

# In[ ]:


old_names = df.columns # saving old column names which might be used later

df.columns = ['Age', 'Sex', 'Chest pain type', 'Resting blood pressure', 'Serum cholestrol',
             'Fasting blood sugar > 120 mg/dl', 'Resting ECG', 'Max heart rate', 'Exercise enduced angina',
             'Exercise enduced ST depression', 'Slope of ST', 'No. of major vessels', 'Thalassemia',
             'Diagnosis']

old_df = df.copy() # saving a copy of older dataframe

df['Sex'] = df['Sex'].map({0: 'Female', 1: 'Male'})

# Angina is chest pain or discomfort caused when your heart muscle doesn't get enough oxygen-rich blood
df['Chest pain type'] = df['Chest pain type'].map({0: 'typical angina',
                                                  1: 'atypical angina',
                                                  2: 'non-anginal pain',
                                                  3: 'asymptomatic'})
df['Fasting blood sugar > 120 mg/dl'] = df['Fasting blood sugar > 120 mg/dl'].map({0: 'No',
                                                                                  1: 'Yes'})

# An electrocardiogram (ECG or EKG) records the electrical signal from your heart to check for different heart conditions.
# Electrodes are placed on your chest to record your heart's electrical signals, which cause your heart to beat.

df['Resting ECG'] = df['Resting ECG'].map({0: 'Normal',
                                          1: 'ST-T wave abnormality',
                                          2: 'Left ventricular hypertrophy'})

df['Exercise enduced angina'] = df['Exercise enduced angina'].map({0: 'No',
                                                                  1: 'Yes'})

df['Slope of ST'] = df['Slope of ST'].map({0: 'Up-sloping',
                                          1: 'Flat',
                                          2: 'Down-sloping'})

# The description on the website doesn't not provide the right mapping.
# So I've used generic mapping i.e. Type 1, 2, 3, and 4


# Thalassemia is a blood disorder which the body makes an abnormal form or inadequate amount of hemoglobin.

df['Thalassemia'] = df['Thalassemia'].map({0: 'Type 1',
                                          1: 'Type 2', 
                                          2: 'Type 3',
                                          3: 'Type 4'})  
df['Diagnosis'] = df['Diagnosis'].map({0: 'Negative',
                                      1: 'Positive'})


# In[ ]:


df.head()


# This is a lot better than previous column names. Although some columns, such as *Exercise enduced angina*, might still not provide clarity, we can safely assume that any interested reader will refer to the description provided [here](https://archive.ics.uci.edu/ml/datasets/Heart+Disease). <br> <br>

# ### Splitting the dataset
# In real world, predicitons are made on fresh, untouched data and hence, it becomes very important that we have a set of observations which is untouched i.e. our model has never seen this set. This prevents information leakage. We will go one step further and refrain from even visualizing this set. Although this results in loss of accuracy, it helps us to mimic our analysis to real world as close as possible. So let's go!

# In[ ]:


from sklearn.model_selection import train_test_split

X = df.drop('Diagnosis', axis = 1)
y = df['Diagnosis']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 147,
                                                   shuffle = True, stratify = y)

get_shape(X_train, 'X_train datafram')
get_shape(X_test, 'X_test dataframe')
get_shape(y_train, 'y_train dataframe')
get_shape(y_test, 'y_test dataframe')


# ## Data Visualization

# In[ ]:


# We'll recombine are train data for visualization purpose. 
df_train = pd.concat([X_train, y_train], axis = 1)


# In[ ]:


# It's always good to check if predictor variable is distributed equally across different classes.
# It is often difficult to model with skewed classes.

plt.figure(figsize = (3, 4))
sns.countplot(x = df_train['Diagnosis'], color = "#FFCE00")
plt.ylabel('Counts');

# we'll use counts to set x position for text
counts = df_train['Diagnosis'].value_counts() 
# creating percentage text
percentage_text = ['{:0.0f}%'.format(x) for x in counts*100/df_train.shape[0]]
# setting each text using plt.text
for pos in range(len(percentage_text)):
    plt.text(pos-0.10, counts[pos]-10, percentage_text[pos],
            color = 'black')

# It seems like classes are almost equally split


# ### Asking questions
# Before we begin any analysis, it is highly recommended to pose some questions that might seem interesting to you. Questioning before visualizing helps in two ways - it helps us generate some curiosity towards the dataset and helps us understand our own biases towards the data. For example, I feel older patients with higher heart rate might be at higher risk. If I just did some random visualizations and found bunch of strong correlations, it would be hard to understand whether the correlations were intuitive or not. So here are some things I'm looking forward to understand:
# 
# **Blood Pressure, Serum Cholesterol**: Do resting blood pressure and serum cholestoral vary siginificantly by diagnosis? <br>
# **Major blood vessels**: Do people with lesser blood vessels by fluroscopy are more prone to heart disease? How does the average blood pressure differ by number of blood vessels? <br>
# **Older Male Patients**: Do older male patients with higher heart rate and chest complains are more likely to be diagnoised positive? <br>
# 
# **Prediciton**: Given certain conditions, how accurately we can predict heart disease in a patient?

# ### Blood Pressure, Serum Cholesterol by Diagnosis

# In[ ]:


figure, axes = plt.subplots(1, 2, figsize = (14, 5))

sns.boxplot(x = df['Resting blood pressure'], y = df['Diagnosis'], 
            color = '#6bc6fa', ax = axes[0])
axes[0].set_title('Resting Blood pressure by diagnosis')

sns.boxplot(x = df['Serum cholestrol'], y = df['Diagnosis'], 
            color = '#6bc6fa', ax = axes[1])
axes[1].set_title('Serum Cholestrol by diagnosis')
figure.tight_layout(pad = 5);


# It's surprising to know that the resting blood pressure of patients with a heart disease is quite strikingly similar to those with no heart disease. The average serum cholesterol is also quite close. However, in positive samples, we do see some more outliers with unusually high serum cholesterol

# ### Movement in major blood vessels by fluroscopy

# In[ ]:


figure, axes = plt.subplots(1, 2, figsize = (14, 5))

sns.countplot(x = df_train['No. of major vessels'], hue = df_train['Diagnosis'], 
            palette = ['#FF0055', '#6bc6fa'], ax = axes[0])
axes[0].set_title('Diagnosis by major blood vessel')
axes[0].legend(loc = 'upper right')

sns.boxplot(y = df_train['Resting blood pressure'], x = df_train['No. of major vessels'], 
            color = '#6bc6fa', ax = axes[1])
axes[1].set_title('Resting blood pressure by diagnosis')
figure.tight_layout(pad = 5);


# I would like to note that the varialble *number of major vessels* indicates movement in blood vessels seen by fluroscopy. Fluroscopy is a method in which x-rays are beamed on a patient to detect movement of a organ in a patient's body, in this case movements in blood vessels. So a *number of major vessels* value of zero **does not** mean that the patient doesn't have any blood vessels, rather there was barely any movement recorded in the blood vessels. <br>
# 
# In the graph on left, we can see patients with a major blood vessel value of zero have higher chances to be tested positive for heart disease. Very few patient with a value of three or four have chances to be tested positive. In the graph on right, we see that the median resting blood pressure still remains the same irrespective of the blood vessel movements. We can see how the distribution gets tighter as we see movements in more blood vessel.

# ### Older Male Patients

# In[ ]:


args = {'color': ['#FF0266', 'grey']}
scatters = sns.FacetGrid(row = 'Sex', col = 'Chest pain type', hue = 'Diagnosis', 
                         margin_titles = True, hue_kws = args, data = df_train)
scatters.map(plt.scatter, 'Age', 'Max heart rate');
scatters.add_legend()
plt.subplots_adjust(top=0.9) # padding for title
scatters.fig.suptitle("Heart Disease by Sex and Chest pain type");


# Now, bare with me, I understand the above graph could seem very daunting to understand but actually it is a plain-old scatterplot with row and column grids. The rows represent sex of a patient and the column represent different chest pain type. In each scatterplot the x-axis represents the age of the patient and y-axis represent maximum heart rate. <br>
# 
# From the above grids, we can see that we have larger samples of patient with anginal and non-anginal chest pain complains.
# It can be incurred from the graph above that patients with non-anginal chest pain are more prone to a heart disease irrespective of age and sex. <br><br>
# As expected patients with higher heart rate seemed to be more susceptible to heart diseases. On the other hand, age is not a good indicator of heart disease, and it seems younger patients are equally susceptible, if not more. Let's look at the age distribution of patients more closely.

# In[ ]:


figure, axes = plt.subplots(1, 3, figsize = (15, 3))

sns.distplot(df_train['Age'], ax = axes[0])
axes[0].set_title('Age distribution for all patients')
sns.distplot(df_train[df_train['Diagnosis'] == 'Positive']['Age'], ax = axes[1])
axes[1].set_title('Age distribution patients tested positive')
sns.distplot(df_train[df_train['Diagnosis'] == 'Negative']['Age'], ax = axes[2])
axes[2].set_title('Age distribution patients tested negative')
plt.setp(axes, ylim = (0, 0.07));


# The distribution of all patients peaks around age 55 which means we have more samples of patients around that age. The distribution for positive patients is flatter, indicating patients of different ages (especially age between 40 to 65) are tested positive. The disctribution for negative patients peask at age 60. This indicates that a patient at age 60 is more likely to test negative than positive! Although one should note that we have higher number of samples at that age.

# ### Correlation Plot
# 
# Now that we have sufficiently explored the variables we were interested in, let's look at the dataset as a whole to find some interesting relationships between other variables. One of the most interesting visualizations is a correlation plot. It helps us visualize relationships between all pairs of variables. A strong correlation indicates a strong relationship and hence, more predicition power. 

# In[ ]:


# Optional: Creating a custom diverging colormap
# I do not like diverging colormaps provided by matplotlib as none of them provide
# same color at the end of the spectrum. For correlation, it doesn't matter whether the values
# are positive or negative. We are interested in values at the end of the spectrum
# Hence, I'm creating a custom diverging colormap with same colors at the end.
# However, this is totally optional and you can use default colormaps.
from matplotlib import cm
from matplotlib.colors import ListedColormap

bupu = cm.get_cmap('BuPu', 128)

bupu_divergence = np.vstack((bupu(np.linspace(1, 0, 128)),
                       bupu(np.linspace(0, 1, 128))))

cust_cmap = ListedColormap(bupu_divergence, name='PurpleDiverging')

plt.figure(figsize = (10, 5))
sns.heatmap(old_df.corr(), annot = True, fmt = '0.2f', 
            vmin = -1, vmax = 1, cmap = cust_cmap);


# From the graph above, we can see strong correlations between chest paint type, max heart rate, exercise enduced angina and st depression. Max heart rate is correlated to lot many variables so I'll be cautious of this. Chest pain seemed to be highly correlated to exercise induced angine as well. Let's explore some of these variables.

# In[ ]:


figure, axes = plt.subplots(2, 2, figsize = (15, 11))
plt.subplots_adjust(hspace = 0.3)


sns.countplot(x = df_train['Diagnosis'], hue = df_train['Exercise enduced angina'], 
              palette = ['#FF0055', '#6bc6fa'], hue_order = ['Yes', 'No'], ax = axes[0,0])
axes[0,0].set_title("Diagnosis by exercise induced agina")

# Percentage labels
# diag_by_angina_labels = df.groupby(['Diagnosis', 'Exercise enduced angina'])['Age'].count()/df.shape[0]
# diag_by_angina_percentage_labels = ['{:0.0%}'.format(x) for x in diag_by_angina_labels]
# xticks = [-0.25, 0.15, 0.75, 1.15]
# for pos, xtick in zip(range(len(diag_by_angina_labels)), xticks):
#     axes[0,0].text(xtick, diag_by_angina_labels[pos], diag_by_angina_percentage_labels[pos] )


sns.countplot(x = df_train['Chest pain type'], hue = df_train['Exercise enduced angina'], 
              palette = ['#FF0055', '#6bc6fa'], hue_order = ['Yes', 'No'], ax = axes[0,1])
axes[0,1].set_title("Exercise induced agina by chest pain type")

sns.boxplot(x = df_train['Exercise enduced ST depression'], y = df_train['Diagnosis'], 
              color = '#6bc6fa', ax = axes[1,0])
axes[1,0].set_title("Diagnosis by exercise enduced ST depression")

sns.boxplot(x = df_train['Chest pain type'], y = df_train['Exercise enduced ST depression'],  
              color = '#6bc6fa', ax = axes[1,1])
axes[1,1].set_title("Exercise enduced ST depression by chest pain type");


# The above graphs provide more depth to the correlations. Surprisingly, exercise enduced angina is more common in negative cases. In the upper right graph, we can see that a strong correlation between exercise enduced angina and different chest pain type. Let's connects the dots here - so if a patient experienced angina due to exercising, they are more likely to be diagnosed to have anginal chest pain. However, if a patient patient experience angina without excerising, they are more likely to have non-anginal chest pain and hence, more likely to be diagnoised positive for a heart disease. This is a great insight we gathered from few simple graphs!
# 
# Further, we can observe from the lower left graph that the median ST depression induced by negative cases is higher than that of positive cases. The distribution is also more spread out of negative cases. In the last lower right graph, a weak correlation can be observed between chest pain and ST depression induced by exercise. <br><br>
# I feel very satisified with the visualizations created. We've explore most relationships with different variables. I feel very confident to make some predictions. So let's begin!

# ## Modelling

# In[ ]:


# Create dummy variables
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Minmax scaling
from sklearn.preprocessing import MinMaxScaler

minmaxer = MinMaxScaler()
train_minmax = minmaxer.fit_transform(X_train)
test_minmax = minmaxer.transform(X_test) # I had to use fit again as train and test don't have the same size!


# In[ ]:


# Algorithms and model selection libraries

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GridSearchCV


# <a id='base predictions'></a>
# ### Base level predictions with cross validation

# In[ ]:


def base_model(models, X, y):
    for model_name, model in  models.items():
        crossval = cross_val_score(model, X, y, cv = 5, scoring = 'roc_auc')
        
        print("\n{} with Cross Validation \n".format(model_name),
              ['{:0.3%}'.format(x) for x in crossval], "\nMean Score \n",
              '{:0.3%}'.format(np.mean(crossval)))
        
seed = 30
models = {'Logistic': LogisticRegression(random_state = seed),
          'KNN': KNeighborsClassifier(n_neighbors=20),
          'Gaussian': GaussianNB(),
          'SVC': SVC(random_state = seed),
          'Decision Tree': DecisionTreeClassifier(random_state = seed),
          'Random Forest': RandomForestClassifier(random_state = seed),
          'Gradient Boosting': GradientBoostingClassifier(random_state = seed)   
         }

base_model(models, train_minmax, y_train)


# Cross validation is helpful to checkout the performance of different algorithms. In the above run, random forest outperforms other models with an average accruacy of 90%. Let's see if we can further improve the accuracy after tuning the model using grid search

# ### Gridsearch with Random Forest
# We'll use two parameters in grid search: <br>
# **Number of estimators** - Number of decision trees using in a random forest<br>
# **min_samples_leaf** - Minimum number of samples for the last node in the tree. The tree stops splitting after the specificed minimum samples<br>
# 
# To start of let's take a large range of values so that we can later decide at what values should we narrow in.

# In[ ]:


rf_params = {'n_estimators': [10, 100, 300, 500, 1000],
            'min_samples_leaf': [1, 10, 25, 50]}

rf_grid = GridSearchCV(RandomForestClassifier(random_state=0), rf_params, cv = 7, scoring = 'roc_auc')
rf_grid.fit(train_minmax, y_train)


# In[ ]:


print("Best Validation Score: {:0.2%} ".format(rf_grid.best_score_))
print("Test Score: {:0.2%}".format(rf_grid.score(test_minmax, y_test)))
print("Best Parameters", rf_grid.best_params_)
print('\n\nChosen Model\n', rf_grid.best_estimator_)


# In[ ]:


# Visualizing scores across all parameters
scores = rf_grid.cv_results_['mean_test_score'].reshape(4, 5)
sns.heatmap(scores, cmap = 'BuPu', annot=True, fmt = '0.0%', 
           xticklabels = rf_params['n_estimators'], yticklabels=rf_params['min_samples_leaf']);

plt.xlabel('N Estimators')
plt.ylabel('Min Samples Leaf')
plt.yticks(rotation = 0);


# Looking at the grid it seems that model accuracy plateaus  when using more than 100 estimators and model underfits when using sample leaf size of 50. So let's rebuild the grid but this time we'll narrow down at values near our best estimators
# 
# **Note**: In some cases it's possible to get a higher accuracy when using 100+ estimators. However, there is barely a increase in accuracy compared to the computational cost.

# ### Fine tuning the hyperparameters of Random Forest

# In[ ]:


get_ipython().run_cell_magic('time', '', "rf_params = {'n_estimators': [10, 15, 20, 25, 50 , 75, 100],\n            'min_samples_leaf': [1, 5, 10, 15, 20, 25, 30, 50]}\n\nrf_finer = GridSearchCV(RandomForestClassifier(random_state = 0), rf_params, cv = 7, scoring = 'roc_auc')\nrf_finer.fit(train_minmax, y_train)")


# In[ ]:


print("Best Validation Score: {:0.2%} ".format(rf_finer.best_score_))
print("Test Score: {:0.2%}".format(rf_finer.score(test_minmax, y_test)))
print("\n\nBest Parameters", rf_finer.best_params_)
print('Chosen Model\n', rf_finer.best_estimator_)


# In[ ]:


# Visualizing scores across all parameters
scores = rf_finer.cv_results_['mean_test_score'].reshape(8, 7)
sns.heatmap(scores, cmap = 'BuPu', annot=True, fmt = '0.0%', 
           xticklabels = rf_params['n_estimators'], yticklabels=rf_params['min_samples_leaf']);

plt.xlabel('N Estimators')
plt.ylabel('Min Samples Leaf')
plt.yticks(rotation = 0);


# I think we have the best estimators for the model. Although we did not use all the parameters available, I believe we still did a good job with the two parameters. Let's look at the precision-recall curve and roc curve

# ### Precision Recall and ROC Curve
# Let's look at the trade-off between precision and recall. It's more likely that we would rather have more false positive than false negative cases i.e we would rather diagonise a patient as positive even though they might not have an underlying heart condition (false positive) than diagnoise a patient negative even though they do have underlying heart condiiton (false negative)! <br><br>
# 
# Remember that we are only shifting the labelling threshold but the underlying model remains the same. To give you an example, consider test sample that has a 45% probability to be diagoised as positive. If our labelling threshold is 50%, we'll label this sample as negative i.e. no heart disease, however, if we change our threshold to 40% we'll label the same sample as positive! It is therefore very critical to do a cost-benefit analysis to set a right precision-recall trade off (or sensitivity-specificity trade off).

# Let's first look at class distribution of our sample tests. It is helpful to understand precision and recall if we know how many tests are positive and negative.

# In[ ]:



plt.figure(figsize = (3, 4))
sns.countplot(x = y_test, color = "#FFCE00")
plt.ylabel('Counts');

# we'll use counts to set x position for text
total = y_test.shape[0]
test_counts = y_test.value_counts() 
# creating percentage text
test_percentage = ['{:0.0f}%'.format(x) for x in test_counts*100/total]

print('We have {} samples in our test set. {} ({}) samples are positive'       ' and {} ({}) samples are negative.'.format(total, test_counts[0], test_percentage[0], test_counts[1], test_percentage[1]))

# setting each text using plt.text
for pos in range(len(test_percentage)):
    plt.text(pos-0.10, test_counts[pos]-2.5, test_percentage[pos],
            color = 'black')

# It seems like classes are almost equally split


# In[ ]:


# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve

rf_prec, rf_rec, rf_thresh = precision_recall_curve(y_test,
                                                    pos_label = 'Positive',
                                                    probas_pred = rf_finer.best_estimator_.predict_proba(test_minmax)[:,1])

optimal_idx = np.argmin(np.abs(rf_rec - rf_prec))

plt.plot(rf_prec, rf_rec, color = 'grey')
plt.plot(rf_prec[optimal_idx], rf_rec[optimal_idx], '^', 
         markersize  = 10, color = '#FF0055')

plt.legend(('Precision Recall Curve', 'Optimal Threshold'))
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.xticks(np.arange(0.75, 1.01, 0.05))
plt.yticks(np.arange(0, 1.1, 0.1));


# We get about 80% precision and recall with the optimal threshold. We can improve the recall to upto 100% while only trading off 5-8% precision. Remember 100% recall wtih 75% precision equates to 0% false negative cases with only 25% false positive rate. <br>
# 
# For the heart disease diagnosis, this means we can very confidently diagnoise a person with a negative result with only 25% of the time diagnoising a patient as positive when they actually don't have any heart disease!

# In[ ]:


# ROC Curve
rf_tpr, rf_fpr, rf_thresh = roc_curve(y_test,
                                      pos_label = 'Positive',
                                      y_score = rf_finer.best_estimator_.predict_proba(test_minmax)[:, 1])

roc_optimal_idx = np.argmax(rf_fpr - rf_tpr)
# optimal_threshold = thresholds[optimal_idx]

plt.plot(rf_tpr, rf_fpr, color = 'grey')
plt.plot(rf_tpr[roc_optimal_idx], rf_fpr[roc_optimal_idx], marker = '^', color = '#FF0055')
plt.legend(('ROC Curve', 'Optimal Threshold'))
plt.xlabel("False Positive Rate")
plt.ylabel('True Positive Rate');


# At the optimal threshold, we have 80% true positive rate. We can increase this to 90% with only 30-35% false positive rate. 

# ### Feature Importance
# Let's look at the features which were more helpful for our model to classify our samples. A word of caution - feature importance does not imply causation but only correlation. For instance, if the model found *high blood sugar* to be very useful in predicting heart disease, it doesn't mean high blood sugar causes heart disease. In fact, there could be a third factor, such as unhealthy diet, which causes both high blood sugar and heart disease. <br> <br>
# This problem is more excarbated when using complex algorithms like random forest. Since we use multiple tree to come up with the predictions, it's difficult to tell how the model came about with the predcitions. <br><br>
# Nevertheless, it's good to know what variables contributed the most to our predictions. Combined with domain knowledge this could help our understanding of heart diseases.<br>

# In[ ]:


plt.figure(figsize = (10, 7))

features = X_train.columns
importance = rf_finer.best_estimator_.feature_importances_ 
indices = np.argsort(importance)

percentage_labels = ['{:0.0%}'.format(x) for x in importance[indices]]

plt.barh(y = range(len(indices)), color = "#FFCE00",
        width = importance[indices])
plt.yticks(range(len(indices)), features[indices])

for pos, index in zip(range(len(indices)), indices):
    plt.text(importance[index], pos , percentage_labels[pos])


# Chest pain and max heart rate contribute the most to our predicitions. These are also the variables which explored in detail so good to know that. Thalassemia also strongly contributes to the predictions. Unfortunately, we don't have much information about the different types of thalassemia.

# # Conclusion

# In this notebook we explored a real-world heart disease dataset. We explored different factors which could contribute to a heart disease. We started with some questions which intrigued us. We then looked at the dataset as a whole using a correlation plot, and dived deeper into paired relationships between different factors. <br> <br>
# After sufficient exploratory analysis, we tested different algorithms using cross validation and picked random forest for further tuning. After some hyper parameter tuning we achieved a 87% test accuracy. <br> <br>
# Finally, we utlitized the precision-recall curve and roc curve to understand the trade-off and optimal threshold values, and we looked at feature importance to see how different features were utilized by our model.

# ### Bonus round - LOO and refactoring code
# I have never tried out Leave One Out cross validation. For those hearing this for the first time, in LOO (Leave One Out) each sample is treated as a test sample. It means if you have 100 data points, the algorithm will train on random 99 data points and then predict the 100th. It does this for each data point. This is of course very costly and hence, only recommended for small sample size. <br> <br>
# I also wanted to highlight on refactoring code. When I first started programming in python, my code was very repetitive and I took little efforts to make my code more efficient. After some practice, I learned about code refactoring - a process in which you look back at your code to make it more readable and efficient. Most programmers already know about refactoring, however, this concept is fairly new to most analysts. Earlier in the notebook, we make [base level predictions](#base precit) using cross validation. The below cell is quite similar to the [base level predictions](#base precit) cells, however, the code is more repetitive and inefficient. <br><br>
# I just thought it is important to share these small nuggets. When you start programming it's only important to make sure your code runs without errors, however as you progress you can make your code much easier to read and a lot more efficient to run!
# 
# Stay safe! Have fun!

# In[ ]:


# Base level predictions with loo
logistic_loo = cross_val_score(LogisticRegression(), train_minmax, y_train, cv = LeaveOneOut())
knn_loo = cross_val_score(KNeighborsClassifier(n_neighbors=20), X_train, y_train, cv = LeaveOneOut())
gaussian_loo = cross_val_score(GaussianNB(), train_minmax, y_train, cv = LeaveOneOut())
dtree_loo = cross_val_score(DecisionTreeClassifier(max_depth = 5), X_train, y_train, cv = LeaveOneOut())
svc_loo = cross_val_score(SVC(), train_minmax, y_train, cv = LeaveOneOut())
rf_loo = cross_val_score(RandomForestClassifier(max_depth = 5), X_train, y_train, cv = LeaveOneOut())
gb_loo = cross_val_score(GradientBoostingClassifier(max_depth = 5), X_train, y_train, cv = LeaveOneOut())

print("Mean Score of Logistic with LOO \n",
     '{:0.3%}'.format(np.mean(logistic_loo)))

print("\nMean Score of KNN with LOO \n",
     '{:0.3%}'.format(np.mean(knn_loo)))

print("\nMean Score of Naive Bayes with LOO \n",
     '{:0.3%}'.format(np.mean(gaussian_loo)))

print("\nMean Score of Decision Tree with LOO \n",
     '{:0.3%}'.format(np.mean(dtree_loo)))

print("\nMean Score of SVC with LOO \n",
     '{:0.3%}'.format(np.mean(svc_loo)))

print("\nMean Score of Random Forest with LOO \n",
     '{:0.3%}'.format(np.mean(rf_loo)))

print("\nMean Score of Gradient Boosting with LOO \n",
     '{:0.3%}'.format(np.mean(gb_loo)))


# ~ The End ~
