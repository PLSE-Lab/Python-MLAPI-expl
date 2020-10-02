#!/usr/bin/env python
# coding: utf-8

# # Analysis and Prediction-Indian Liver Patients.

# This dataset was downloaded from the UCI ML Repository. The objective of this study is not to maximize the accuracy or the ROC of the predictor, but to explore the different models and see if we can help doctors improve liver diagnosis.
# 
# I would like to acknowledge Sangeetha James, whose script I forked for the work.
# 
# The dataset is relatively small (583 samples), so it is hard to draw conclusions based on it. However, there are still many things we can learn from it.
# 
# The work is primarily divided into two parts - in the first part, the dataset was loaded, cleaned and visualized. During the analysis, I came across biases in the dataset, such as more than 70% of the patients in the sample set have liver disease and  men in this sample set are more likely to have liver disease than women.
# 
# In the second part, the complete dataset was divided into training data (80%) and test data (20%). In various ML algorithms, cases were identified where data was overfit or underfit, and remedial actions were taken. In all the cases, the test score accuracies were all around 70%. As a simple estimator, which ignores all the inputs and always predicts that the person has liver disease will achieve a 71.35%, this performance is pretty bad.
# 
# The first published result on this dataset by [Ramana et. al](https://pdfs.semanticscholar.org/c92d/38a7a76c20a317de63fb9278bb10102c758b.pdf) achieves 90+ percent accuracy. I was intrigued by the large difference in scores. After a second re-read of the paper, I identified that they used 90% of their data as a training data and 10% as the test data. Their approach give more data for training, which helps make their model better, but it is risky to test the model against such a smaller amount data.
# 
# With 90% training data, the results improved marginally, but I could not achieve 90%. Another inconsequential difference in the dataset used by the paper and the UCI dataset at Kaggle is that the UCI dataset has removed columns which can be calculated from the other columns. The dataset used in the paper had columns for Albumin, Globulin and Albumin and Globulin Ratio, while the Kaggle dataset has gotten rid of Globulin column.
# 
# 
# This is a work in progress and I hope to come back and replicate the 90+% accuracy, but for now, my conclusion is that we need more data to produce a better model. And secondly, the doctors should continue to use their current approach to diagnosis which was used to populate the "Dataset" (probably a biopsy, or ultrasound / MRI) which best shows the presence or absence of liver disease.
# 

# In[1]:


#Import all required libraries for reading data, analysing and visualizing data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder


# # Exploratory Data Analysis

# In[2]:


#Read the training & test data
liver_df = pd.read_csv('../input/indian_liver_patient.csv')


# This data set contains 416 liver patient records and 167 non liver patient records collected from North East of Andhra Pradesh, India. The "Dataset" column is a class label used to divide groups into liver patient (1=liver disease) or not (2=no disease). 

# In[3]:


liver_df.head()


# In[4]:


liver_df.info()


# Here is the observation from the dataset:   
# * Only gender is a non-numeric variable. All others are numeric.
# * There are 10 features and 1 output - dataset. Value 1 indicates that the patient has liver disease and 2 indicates the patient does not have liver disease.  
# * The columns names are shortened from the original dataset at UCI. 
# 
# The columns and their description reproduced from the UCI site are below.
# 1. Age - Age of the patient 
# 2. Gender -	Gender of the patient 
# 3. TB -	Total Bilirubin 
# 4. DB -	Direct Bilirubin 
# 5. Alkphos - Alkaline Phosphotase 
# 6. Sgpt - Alamine Aminotransferase 
# 7. Sgot - Aspartate Aminotransferase 
# 8. TP -	Total Protiens 
# 9. ALB - Albumin 
# 10. A/G Ratio - Albumin and Globulin Ratio 
# 11. Selector - field used to split the data into two sets (labeled by the experts) 
# 

# In[5]:


#Describe gives statistical information about NUMERICAL columns in the dataset
liver_df.describe(include='all')
#We can see that there are missing values for Albumin_and_Globulin_Ratio as only 579 entries have valid values indicating 4 missing values.
#Gender has only 2 values - Male/Female


# In[6]:


#Which features are available in the dataset?
liver_df.columns


# In[7]:


#Check for any null values
liver_df.isnull().sum()


# The only data that is null is the Albumin_and_Globulin_Ratio - Only 4 rows are null. Lets see whether this is an important feature    

# ## Data Visualization

# In[8]:


sns.countplot(data=liver_df, x = 'Dataset', label='Count')

LD, NLD = liver_df['Dataset'].value_counts()
print('Percentage of patients diagnosed with liver disease: ',LD / (LD+NLD) * 100)


# ### Subjects in this study are more likely to have liver patients than the general public
# Out of the 583 subjects, 71.35% (416) of cases had liver disease and 28.64% (167) did not. In the general population, we do not expect to see two out of three people with liver disease. This points to a bias in the data collection, in which subjects who were more likely to have issues with the liver were more likely to be picked. In a larger study with 633,323 subjects, approximately 0.27% were found to have liver cirrhosis [1]. 
# 
# For example in this dataset, if a predictor always predicts that the subject has liver disease, he will be correct about 70% of the time, but in the real world, this predictor would be 0.27% correct.
# 
# However, this group of subjects might better reflect the distribution of people who visited a Hepatologist (liver specialist).
# 
# [1] https://www.ncbi.nlm.nih.gov/pubmed/25291348
# 

# In[9]:


sns.countplot(data=liver_df, x = 'Gender', label='Count')

M, F = liver_df['Gender'].value_counts()
malesWithLiverDisease = liver_df[(liver_df['Gender'] == 'Male') & (liver_df['Dataset'] == 1)]['Age'].count()
femalesWithLiverDisease = liver_df[(liver_df['Gender'] == 'Female') & (liver_df['Dataset'] == 1)]['Age'].count()
patientsWithLiverDisease = liver_df[liver_df['Dataset'] == 1]['Age'].count()
totalPatients = liver_df['Age'].count()
print('Percent of patients that have liver disease: ',patientsWithLiverDisease /totalPatients * 100)
print('Percent of male patients that have liver disease: ',malesWithLiverDisease /M * 100)
print('Percent of female patients that have liver disease: ',femalesWithLiverDisease /F * 100)

print('Number of patients that are male: ',M)
print('Number of patients that are female: ',F)


# 
# ### Women are less likely to have liver disease. Is this real or biased data?
# The fraction of women with liver disease (64.78%) was lower than the fraction of men with liver disease (73.46). This is contrary to studies that women are more susceptible  to liver disease [2]. The higher prevalence of liver disease among men in this dataset could be due to high rates of alcohol consumption among men in India (where women drinking alcohol was a social taboo). This might not be true in western countries.
# 
# If our aim is to predict the presence of liver disease outside of India, to avoid bias (as our model is likely to guess that being a woman makes having liver disease less likely instead of more), we should disregard the gender in this study. On the other hand, including gender will help improve results for a predictive task in India.
# 
# [2] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4388670/

# Age seems to be a factor for liver disease for both male and female genders

# In[10]:


liver_df[['Dataset','Age']].groupby(['Dataset']).mean()


# ### Age
# The mean age of patients with liver disease is about five years more than the patients without the liver disease. The older patients seem to be at a higher risk of having liver disease. It will be interesting to see if there are any correlations between age and other tests.

# ### Look at all the data together (Why it is not easy to diagnose)
# The blue box plots are the patients with liver disease and the green box plots are the patients who do not have liver disease. 
# 
# As you can see, for all the different measurements, there is a significant overlap in values between people with liver disease and people without liver disease. Therefore, no single metric can be used directly for diagnosis. 
# 
# For instance, 5% of US population has Gilbert's syndrome - a genetic condition which results in an elevated Bilirubin in blood. In people with Gilbert's syndrome, a higher bilirubin does not indicate liver disease, but combining it with other metrics like Alkaline_Phosphotase or Total_Proteins might help the estimator. The ML models will try and see if we can use all the values together to better predict the occurrence of liver disease. 
# 
# The presence of outliers in patients with liver disease and without liver disease further complicates the diagnosis. 

# In[11]:


#liverDisease_df = liver_df[liver_df['Dataset'] == 1]
#liverDisease_df.drop(['Gender', 'Dataset'], axis=1).boxplot()
#nonLiverDisease_df = liver_df[liver_df['Dataset'] == 2]
#nonLiverDisease_df.drop(['Gender', 'Dataset'], axis=1).boxplot()
fig=plt.figure(figsize=(20, 24), dpi= 80, facecolor='w', edgecolor='k')

ax = liver_df.drop(['Gender'], axis='columns').set_index('Dataset', append=True).stack().to_frame().reset_index().rename(columns={'level_2':'quantity', 0:'value'}).pipe((sns.boxplot,'data'), x='quantity', y='value', hue='Dataset')
ax.set(ylim=(0,500))


# In[12]:


# Correlation
liver_corr = liver_df.corr()
liver_corr


# In[13]:


plt.figure(figsize=(15, 15))
sns.heatmap(liver_corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 10},
           cmap= 'coolwarm')
plt.title('Correlation between features');


# ## Observation:
# The "Dataset" column which indicates the presence or absence of liver disease does not strongly correlate with one of the measured metrics. There are strong positive correlations between pairs of measured quantities (Total_Proteins & Albumin, Total_Proteins & Albumin, Alamine_Aminotransferase & Aspartate_Aminotransferase, Direct_Bilirubin & Total_Bilirubin. Without more insight into the actual meaning of these quantities, it is possible these measurements are similar in nature and we might be able to reduce the dimensionality of this problem using Principle Component Analysis (PCA).

# ## Clean up data
# 
# The dataset is pretty clean, except for a few NA's in the Albumin_and_Globulin_Ratio. There is a column for Albumin, but not one for Globulin. So we do not have any information to fill in the values. We can ignore the four rows which have this column, or replace them with the mean or median. Because of the noisy data, I chose to use the median value to fill in the missing values.
# 
# Lets prepare the input and output data frames.
# 

# In[14]:


liver_df["Albumin_and_Globulin_Ratio"] = liver_df.Albumin_and_Globulin_Ratio.fillna(liver_df['Albumin_and_Globulin_Ratio'].median())

X = liver_df.drop(['Gender', 'Dataset'], axis='columns')
X.head()


# In[15]:


X.describe()


# In[16]:


y = liver_df['Dataset'] # 1 for liver disease; 2 for no liver disease


# # Machine Learning
# 
# I tried Logistic Regression, Gaussian Naive Bayes, Decision Trees, Random forest, SVM, and Nearest neighbor algorithms. If we have a biased estimator, which always predicts that the patient has liver disease, it will have a 71.35% sucess rate. So our estimator to be of any use must do substantially better than that.

# In[17]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)


# In[18]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score

def processResults(clk, str="", X_test = X_test, X_train = X_train, y_test = y_test, y_train = y_train):
    predicted = clk.predict(X_test)
    score = round(clk.score(X_train, y_train) * 100, 2)
    score_test = round(clk.score(X_test, y_test) * 100, 2)

    print(str + 'Training score: \n', score)
    print(str + 'Test Score: \n', score_test)
    print('Accuracy: \n', accuracy_score(y_test,predicted))
    print(confusion_matrix(y_test,predicted))
    print(classification_report(y_test,predicted))
    sns.heatmap(confusion_matrix(y_test,predicted),annot=True,fmt="d")
    return score, score_test
    


# ## Logistic Regression observation
# The model has an accuracy of 71.43%. The training and testing scores are very close to each other at 71.24% and 76.07% respectively.
# 
# A simple way to check the performance of a classifier is to compare its accuracy to a biased estimator. If we had a biased estimator, which, independent of all the metrics, always predicted that the patient has liver disease, for the complete dataset we would have an accuracy of 71.35%.
# 
# The training and testing accuracy percentages being so close to each other indicate that the model has high bias and is underfitting the data. To get a better fit, we should increase the number of parameters that we fit to the data or use a different model.

# In[19]:


from sklearn.linear_model import LogisticRegression

#2) Logistic Regression
# Create logistic regression object
logreg = LogisticRegression()
# Train the model using the training sets and check score
logreg.fit(X_train, y_train)

logreg_score, logreg_score_test = processResults(logreg, "Logistic Regression ")

print('Coefficient: \n', logreg.coef_)
print('Intercept: \n', logreg.intercept_)


# ## Naive Bayesian
# The Naive Bayesian estimator fares even worse than just guessing that liver disease is present! Moving on to decision trees and random forest.

# In[20]:


# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
gauss_score, gauss_score_test = processResults(gaussian, "Gaussian Naive Bayesian ")


# ## Decision tree
# The decision tree estimator achieves 100% in the training set, while only 70.09% in the test. This indicates that we have overfit our training data. We can alleviate this by finding how deep the decision tree is and prune it until we fix the issue. So, we plotted how the training and test data accuracies vary with depth ( See figure below). The ideal point when we fit everything correctly is when the training and the test curves cross each other. But in this case, that seems to happen very close to the 2.

# In[21]:


# Decision tree
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)
decision_tree_score, decision_tree_score_test  = processResults(decision_tree, "Decision Tree ")


# In[22]:


# As max depth is 14, let's reduce it, until the test and train data scores are close to each other
def plotDecisionTreeScoresVSDepth(clk, maxDepth = 14, X_test = X_test, X_train = X_train, y_test = y_test, y_train = y_train):
    score = []
    score_test = []
    allDepth = np.arange(maxDepth,1,-1)
    for depth in allDepth:
        clk.set_params(**{'random_state': 42, 'max_depth' : depth})
        clk.fit(X_train, y_train)
        
        predicted = clk.predict(X_test)
        score.append(round(clk.score(X_train, y_train) * 100, 2))
        score_test.append(round(clk.score(X_test, y_test) * 100, 2))
    plt.plot(allDepth, score)    
    plt.plot(allDepth, score_test)
    plt.ylabel('Accuracy')
    plt.xlabel('Max depth of decision tree')
    plt.legend(['Train accuracy', 'Test accuracy'])
    plt.show()

decision_tree = DecisionTreeClassifier()        
plotDecisionTreeScoresVSDepth(decision_tree)


# ## Random forest
# The random forest is an ensemble of decision trees. Like decision trees, it too achieves 100% in the training set, while only 70.86% in the test. This indicates that we have overfit our training data. We can alleviate this by finding how deep the decision tree is and prune it until we fix the issue. The plot of the training and test data accuracies vs with depth ( See figure above). The ideal point when we fit everything correctly is when the training and the test curves cross each other. But in this case, that seems to happen very close to the 2

# In[23]:


# Random Forest
# The random forest classifier uses number of decision trees and combines the results to make a prediction
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

random_forest_score, random_forest_score_test  = processResults(random_forest, "Random Forest ")


# The random forest estimator achieves 100% in the training set, while only 72.65% in the test. This indicates that we have overfit our training data. So, lets plot the same 

# ## Support vector machines
# The Support vector classification is a highly biased 

# In[24]:


randomForest = RandomForestClassifier(n_estimators=100)        
plotDecisionTreeScoresVSDepth(randomForest)


# In[25]:


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scale', StandardScaler()),    
    ('pca', PCA(n_components=8)),
    ('svc', SVC()),
])
pipe.fit(X_train, y_train)
svcScore, svcScore_test = processResults(pipe)


# In[26]:


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scale', StandardScaler()),    
    ('pca', PCA(n_components=8)),
    ('svc', SVC()),
])
pipe.set_params(svc__C=2)
pipe.fit(X_train, y_train)
svcScore, svcScore_test = processResults(pipe)


# In[ ]:





# In[27]:


from sklearn.neighbors import KNeighborsClassifier

pipeknn = Pipeline([
    ('scale', StandardScaler()),    
    ('knn', KNeighborsClassifier(n_neighbors=5)),
])
pipeknn.fit(X_train, y_train)
knnTrainScore, knnTestScore = processResults(pipeknn)


# In[28]:


## K nearest neighbors


# In[29]:


liver_df.head()


# 

# In[31]:


###Model evaluation
#We can now rank our evaluation of all the models to choose the best one for our problem. 
models = pd.DataFrame({
    'Model': [ 'Logistic Regression', 'Gaussian Naive Bayes','Decision Tree', 'Random Forest', 'Support Vector Classifier', 'Nearest Neighbour'],
    'Score': [ logreg_score, gauss_score, decision_tree_score, random_forest_score, svcScore, knnTrainScore],
    'Test Score': [ logreg_score_test, gauss_score_test, decision_tree_score_test, random_forest_score_test, svcScore_test, knnTestScore]})
models.sort_values(by='Test Score', ascending=False)

