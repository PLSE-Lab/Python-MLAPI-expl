#!/usr/bin/env python
# coding: utf-8

# # DonorsChoose EDA Challenge
# This notebook is focused on understanding the metadata and text-based influences of whether teachers' resource proposals are approved by DonorsChoose.org. We start with individuals factors, then integrate these in ML models.

# ## Contents
# * Data Wrangling
# * Project Categories
# * Number of Previously Posted Projects
# * Grade Category
# * Approvals by State
# * Approvals by Year and Month
# * Project Essays

# ## Data Wrangling
# There are 1081830 projects within the training dataset, and 459442 projects in the test dataset. As would be expected, the test dataset does not contain the column 'project_is_approved'. 
# 

# In[21]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import warnings
pd.set_option('display.max_columns', 100)
warnings.filterwarnings('ignore')

#Read Training Data, Test Data, and Resources Data
resources = pd.read_csv('../input/resources.csv', sep=',')

train_data = pd.read_csv('../input/train.csv', sep=',')  
train_data['project_submitted_datetime'] = pd.to_datetime(train_data['project_submitted_datetime'])
train_data['year'] = train_data['project_submitted_datetime'].dt.year
train_data['month'] = train_data['project_submitted_datetime'].dt.month
train_data = train_data.merge(resources, on='id')
train_data['$total'] = train_data['price'] * train_data['quantity']
print(train_data.shape,'\n',train_data.columns)

test_data = pd.read_csv('../input/test.csv', sep=',')
test_data['project_submitted_datetime'] = pd.to_datetime(test_data['project_submitted_datetime'])
test_data['year'] = test_data['project_submitted_datetime'].dt.year
test_data['month'] = test_data['project_submitted_datetime'].dt.month
test_data = test_data.merge(resources, on='id')
test_data['$total'] = test_data['price'] * test_data['quantity']
print(test_data.shape,'\n',test_data.columns)


# ## Approvals by Project Categories
# Multiple project category keywords are listed for many projects. To use this as a feature, we will  disaggregate these into separate columns in both training and test data.

# In[22]:


#Separate project categories and tabulate--there are maximum 3 categories (by prior analysis)
train_data[['cat1','cat2','cat3']] = train_data['project_subject_categories'].str.split(',', 3, expand=True)
train_data['cat1'] = train_data['cat1'].str.strip()
train_data['cat2'] = train_data['cat2'].str.strip()
train_data['cat3'] = train_data['cat3'].str.strip()
#Get the number of tags that were assigned to each project
train_data['#project_categories'] = train_data[['cat1','cat2','cat3']].count('columns')
train_data.head(3)


# In[23]:


test_data[['cat1','cat2','cat3']] = test_data['project_subject_categories'].str.split(',',3, expand=True)
test_data['cat1'] = test_data['cat1'].str.strip()
test_data['cat2'] = test_data['cat2'].str.strip()
test_data['cat3'] = test_data['cat3'].str.strip()
#Get the number of tags that were assigned to each project
test_data['#project_categories'] = test_data[['cat1','cat2','cat3']].count('columns')
test_data.head(3)


# In the training data, there were a total of 1,480,886 project category tags, split into 9 categories. Next we gather all tags, and count/plot by projects that contain that tag (as the first tag). When the category, 'Warmth,' appeared as *first* category tag, 85% of its projects were funded, compared to only 24% of 'Special Needs' projects. 

# In[24]:


categories = pd.DataFrame(train_data[['cat1','cat2','cat3']].stack().value_counts(), columns=['#TotalTags'])
categories['#approved'] = train_data.groupby('cat1')['project_is_approved'].sum()
categories['%approved'] = categories['#approved']/categories['#TotalTags']
categories = categories.sort_values(by='%approved', ascending=False)

ax = categories[['#TotalTags','#approved']].sort_values(by='#TotalTags', ascending=False).plot(kind='bar', legend=True, fontsize=16, 
                                        figsize=(12,6), rot=30, title='Funding Counts by First Project Category')
ax.set_xlabel('Project Category', fontsize=14)
ax.set_ylabel('#Projects with Tag', fontsize=16)
print(categories.sum(), '\n', categories)


# ## Approvals by Number of Previously Posted Projects
# Perhaps those previously submitting greater numbers of projects are more successful in getting their projects approved. The prior number of previously submitted projects by a submitter varied from 0 to 451. There was quite a long right tail to the distribution, as shown in the plot (which was cutoff at 20 prior projects). To transform these into a useful feature, we can bin these into categories, defined by [0,1,2,3,4+] prior projects. Also, it could be interesting to determine if a teacher's chance of project approval improves with each submitted project.

# In[68]:




#train_data["teacher_number_of_previously_posted_projects"] = train_data["teacher_number_of_previously_posted_projects"].astype('int')
bins = [0,1,5,10,25,50,500]
train_data['prior_projects_cat'] = pd.cut(train_data["teacher_number_of_previously_posted_projects"],bins=bins,
                                         labels=['0-1','2-5','6-10','11-25','26-50','51+'])
prior = pd.DataFrame(train_data.groupby('prior_projects_cat')['project_is_approved'].count())
prior['#approved'] = train_data.groupby('prior_projects_cat')['project_is_approved'].sum()
prior.rename(columns = {'project_is_approved': '#total'}, inplace=True)
prior['#not_approved'] = prior['#total'] - prior['#approved']
prior['%_approved'] = 100 * prior['#approved']/prior['#total']

#fig, axes = plt.subplots(1,1, figsize=(16,8), sharex=True)
axA = prior[['#total','#approved','#not_approved']].plot(kind='bar', figsize=(12,6), legend=True, fontsize=16, title='#Previously Posted Projects')
axB = prior['%_approved'].plot(kind='line', secondary_y=True, fontsize=14, color='r', legend=True, alpha=1.0)
axA.set_xlabel('#Prior Projects Posted by Teacher', fontsize=16)
axA.set_ylabel('#Projects', fontsize=16)
axB.set_ylabel('Percentage Projects Approved', fontsize=16)
axB.set_ylim(70,100)
#train_data["teacher_number_of_previously_posted_projects"].describe()[['min','50%','mean','max']]

#
#
#train_data['prior_projects_cat'].value_counts()


# ## Approvals by Grade Category
# We can see that approximately 83-85% of projects were approved in the training data; these were weakly based on grade category, with Grades 3-5 slightly higher (85.4%) than Grades 9-12 (83.5%). However, this seems to be a pretty weak factor. A clear trend did emerge, however: while the number of projects submitted and approved declined by a factor of 4 from PreK-2 to grades 9-12, the average project size increased by ~50% over these same intervals. Also, the mean *approved* project $ amounts were about one-third larger than the mean *non-approved* projects.  

# In[48]:


grades = pd.DataFrame(train_data.groupby('project_grade_category')['project_is_approved'].count())
grades['#approved'] = train_data.groupby('project_grade_category')['project_is_approved'].sum()
grades['$approved'] = round(train_data[train_data['project_is_approved']==1].groupby('project_grade_category')['$total'].mean(),2)
grades.rename(columns = {'project_is_approved': '#total'}, inplace=True)
grades['#not approved'] = grades['#total'] - grades['#approved']
grades['$not approved'] = round(train_data[train_data['project_is_approved']==0].groupby('project_grade_category')['$total'].mean(),2)
grades['%approved'] = round(100 * grades['#approved']/grades['#total'],1)
grades = grades.reindex(index=['Grades PreK-2','Grades 3-5','Grades 6-8','Grades 9-12'])

ax1 = grades[['#total', '#approved', '#not approved']].plot(kind='bar', figsize=(12,6), rot=0, legend=True,
                                                               fontsize=14, color=['gray','g','r'], alpha=0.5,
                                                               title='Project Approval Counts by Grades')
ax2 = grades['%approved'].plot(kind='line', secondary_y=True, fontsize=14, legend=True, alpha=0.8)
ax1.set_xlabel('Grade Category', fontsize=16)
ax1.set_ylabel('#Projects', fontsize=16)
ax1.set_ylim(0,500000)
ax2.set_ylabel('Percentage Projects Approved', fontsize=16)
ax2.set_ylim(70,85)

ax3 = grades[['$approved', '$not approved']].plot(kind='bar', figsize=(12,6), rot=0, color=['g','r'], alpha=0.5,
                                                     fontsize=14, title='Mean Project Amount by Grades')
ax3.set_xlabel('Grade Category', fontsize=16)
ax3.set_ylabel('Mean Project Amount (USD)', fontsize=16)
grades


# ## Project Approvals and Mean Funding by State
# Perhaps some state have a better chance of getting their projects funded. Because there are dramatic differences in number of projects among states, we will use percentage project approval by state. From the following training data analysis, all states had 7d0-90% of their projects funded. The plot limits the x-axis range to 80-90, to make it easier to spot differences between states. 

# In[60]:


states = pd.DataFrame(train_data.groupby('school_state').size(), columns=['#submitted'])
states['#approved'] = train_data.groupby('school_state')['project_is_approved'].sum()
states['%approved'] = round(100 * states['#approved']/states['#submitted'], 2)
states['$approved'] = train_data[train_data['project_is_approved']==1].groupby('school_state')['$total'].mean()
states.sort_values(by='%approved', ascending=True, inplace=True)
fig, axes = plt.subplots(1,1, figsize=(16,8), sharex=True)
states['%approved'].plot(kind='line',color=['blue'], alpha=1.0, legend=True)
states['$approved'].plot(kind='bar', color=['gray'], secondary_y=True, ylim=(0,160), alpha=0.7, linewidth=9, legend=True)
axes.set_xlabel('Projects from State', fontsize=16)
axes.set_ylim(70,90)
axes.set_ylabel('% Projects Aprroved', fontsize=16)
axes.right_ax.set_ylabel('Mean Approved Project Value (USD)', fontsize=16)


# ## Project Submission by Year, and Month of Year
# Within this dataset, project submissions were only from two years: 2016 and 2017. Although more than twice as many projects were submitted in 2016, a slightly greater percentage (81 vs 79%) were approved in 2017. The most popular month for project submissions is August (start of school year) , followed by September. The least number of submissions were in May and June, at the end of the school year. 

# In[61]:


years = pd.DataFrame(train_data.groupby('year').size(), columns=['#submitted'])
years['#approved'] = train_data.groupby('year')['project_is_approved'].sum()
years['%approved'] = round(100 * years['#approved']/years['#submitted'], 2)
years.sort_values(by='%approved', ascending=True, inplace=True)
print(years)

months = pd.DataFrame(train_data.groupby('month').size(), columns=['#submitted'])
months['#approved'] = train_data.groupby('month')['project_is_approved'].sum()
months['%approved'] = round(100 * months['#approved']/months['#submitted'], 2)
months.sort_index(ascending=True, inplace=True)
ax = months[['#submitted', '#approved']].plot(kind='bar', figsize=(12,6), rot=0, 
                                                fontsize=16, title='Project Approvals by Month')
ax.set_xlabel('Month', fontsize=16)
ax.set_ylabel('# Projects', fontsize=16)


# ## Project Resource Analysis
# Each project requested a resource, which was computed from  $amount * quantity. We can analyze average funded project resources in the training data.

# In[ ]:


#Release some memory
dfs = [grades, states, years, months]
del dfs


# ## Project Essay Analysis
# The following columns contained potentially valuable text to include in analyses:
# * project_title
# * project_essay_1, _2, _3, _4
# * project_resource_summary (what is being requested)
# 
# Regarding the project essays, teachers responded to the following essay prompts:
# 1. Open with the challenge facing your students (`project_essay_1`)
# 2. Tell us more about your students (`project_essay_2`)
# 3. Inspire your potential donors with an overview of the resources you're requesting (`project_essay_3`)
# 4. Close by sharing why your project is so important (`project_essay_4`) 
# 
# First, considering how many of these were responded to by teachers, we can see that all responded to the title, resource summary, and the first two essay questions. Perhaps project success is related to particular words stated as primary challenges that students face. So, we can look at the most frequent words from Project Essay 1, build a vocabulary, and relate these to project funding success.

# In[ ]:


train_data[['project_title','project_essay_1','project_essay_2','project_essay_3', 'project_essay_4','project_resource_summary']].count(axis='rows')


# ### Text Modeling Using Bag of Words and TFIDF
# The following models essay responses against project approvals. Attempted models include:
# * Naive Bayes
# * Support Vector Machine
# 
# A generalized pipeline is created using the following:
# * Splitting the dataset into test (0.2) and train (0.8) segments
# * Transforming word data into a sparse (#samples, #features) count matrix
# * Removing stop words
# * Normalize the count matrix using tf-idf
# * Fitting a classifier to the transformed training data
# * Using the classifier to predict project approval from test data
# * Computing the AUC from actual vs. predicted test data
# * Computing the proportion of test data projects that were correctly predicted
# 
# 

# In[ ]:


'''
#Predict project approval based upon word analysis (from eassy 1)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
#from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
#import nltk
#from nltk.stem.snowball import SnowballStemmer
#stemmer = SnowballStemmer('english', ignore_stopwords=True)

#class StemmedCountVectorizer(CountVectorizer):
#    def build_analyzer(self):
#        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
#        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

#stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

X_train = train_data['project_essay_1']
y_train = train_data['project_is_approved']
X_test = test_data['project_essay_1']

#Build Naive Bayes Classifier Pipeline w/stemming
clfNB = Pipeline([('vect', CountVectorizer(stop_words='english')),
                    ('tfidf', TfidfTransformer()),
                    ('clf-nb', MultinomialNB(fit_prior=True))])
clfNB = clfNB.fit(X_train, y_train)
y_pred = clfNB.predict(X_train)
print('NB--AUC:',roc_auc_score(y_train, y_pred)) #, 'Prop. predicted:',np.mean(y_pred==y_test))

#Build SVM Classifier Pipeline w/o stemming
#Define params for grid search
params = {'vect__ngram_range': [(1,1), (1,2)],
          'tfidf__use_idf': (True, False),
          'clf-svm__alpha': (.1, .01)}

####NEXT:  add grid search

clfSVM = Pipeline([('vect', CountVectorizer(stop_words='english')),
                  ('tfidf', TfidfTransformer()),
                  ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=.1, max_iter=50, random_state=0))])
clfSVM = clfSVM.fit(X_train, y_train)
y_pred = clfSVM.predict(X_train)
print('SVM--AUC:',roc_auc_score(y_train, y_pred)) #, 'Prop. predicted:',np.mean(y_pred==y_test))                  


#X_test_counts = CountVectorizer().fit_transform(X_test)
#X_test_tfidf = TfidfTransformer(use_idf=True).fit_transform(X_test_counts)
#y_pred = clfNB.predict(X_test_tfidf)

#print(roc_auc_score(y_test, y_pred))

#vect = TfidfVectorizer(min_df=6).fit(X_train)
#vect = CountVectorizer()
#X_train_counts = vect.fit_transform(X_train) #returns [nsamp, nfeatures] document-term matrix


#clfNB = MultinomialNB(alpha=0.1).fit(X_train_vect, y_train)
#gs_clf = GridSearchCV(clfNB, {'tfidf__use_idf':(True, False), 'clf__alpha': (.1, .01)})
#clfSVM = SVC(C=10000).fit(X_train_vect, y_train)
#X_test_vect = vect.transform(X_test)

#y_pred = clfNB.predict(vect.transform(X_test))
#print('AUC:',roc_auc_score(y_test, y_pred), 'Prop. predicted:',np.mean(y_pred==y_test))
#print(gs_clf.best_score_)
'''


# ## Build an Initial Linear Classification Model
# 
# Perhaps `teacher_number_of_previously_posted_projects` might provide a good signal as to whether a DonorsChoose application will be accepted? We can hypothesize that teachers who have submitted a large number of previous projects may be more familiar with the ins and outs of the application process and less likely to make errors that would lead to a rejection.
# 
# Let's test that theory by building a simple linear classification model that predicts the `project_is_approved` value solely from the `teacher_number_of_previously_posted_projects` feature. We'll build our model in TensorFlow using a `LinearClassifier` from the high-level [Estimators API](https://www.tensorflow.org/programmers_guide/estimators). 
# 
# **NOTE:** For more practice in building TensorFlow models with `Estimator`s, see the Machine Learning Crash Course [companion exercises](https://developers.google.com/machine-learning/crash-course/exercises#programming). 
# 
# First, import the modules we'll use, which include TensorFlow, the TensorFlow [Datasets API](https://www.tensorflow.org/get_started/datasets_quickstart), [numpy](http://www.numpy.org/), and [scikit-learn](http://scikit-learn.org/) (for some convenience functions for metrics):

# In[ ]:


#import tensorflow as tf
#from tensorflow.python.data import Dataset
#import numpy as np
#import sklearn.metrics as metrics


# If you didn't import the DonorsChoose training data above, do so now:

# In[ ]:


#import pandas as pd

# Filepath to main training dataset.
#train_file_path = '../input/train.csv'

# Read data and store in DataFrame.
#train_data = pd.read_csv(train_file_path, sep=',')


# Next, define the feature (`teacher_number_of_previously_posted_projects`) and label (`project_is_approved`):

# In[ ]:


# Define predictor feature(s); start with a simple example with one feature.
#my_feature_name = 'teacher_number_of_previously_posted_projects'
#my_feature = train_data[[my_feature_name]]

# Specify the label to predict.
#my_target_name = 'project_is_approved'


# Then split the data into training and validation sets:

# In[ ]:


# Prepare training and validation sets.
#N_TRAINING = 160000
#N_VALIDATION = 100000

# Choose examples and targets for training.
#training_examples = train_data.head(N_TRAINING)[[my_feature_name]].copy()
#training_targets = train_data.head(N_TRAINING)[[my_target_name]].copy()

# Choose examples and targets for validation.
#validation_examples = train_data.tail(N_VALIDATION)[[my_feature_name]].copy()
#validation_targets = train_data.tail(N_VALIDATION)[[my_target_name]].copy()


# Then set up the input function to feed data into the model using the [Datasets API](https://www.tensorflow.org/get_started/datasets_quickstart):

# In[ ]:


'''
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
    
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
 
    # Construct a dataset, and configure batching/repeating
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified
    if shuffle:
      # Shuffle with a buffer size of 10000
      ds = ds.shuffle(10000)
    
    # Return the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels
'''


# Next, construct the `LinearClassifier`:

# In[ ]:


'''
# Learning rate for training.
learning_rate = 0.00001

# Function for constructing feature columns from input features
def construct_feature_columns(input_features):
  """Construct the TensorFlow Feature Columns.
  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])

# Create a linear classifier object.
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# Set a clipping ratio of 5.0
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)  
linear_classifier = tf.estimator.LinearClassifier(
    feature_columns=construct_feature_columns(training_examples),
    optimizer=my_optimizer
)
'''


# Create input functions for training the model, predicting on the prediction data, and predicting on the validation data:

# In[ ]:


'''
batch_size = 10

# Create input function for training
training_input_fn = lambda: my_input_fn(training_examples, 
                                        training_targets[my_target_name],
                                        batch_size=batch_size)

# Create input function for predicting on training data
predict_training_input_fn = lambda: my_input_fn(training_examples,
                                                training_targets[my_target_name],
                                                num_epochs=1, 
                                                shuffle=False)

# Create input function for predicting on validation data
predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                  validation_targets[my_target_name],
                                                  num_epochs=1, 
                                                  shuffle=False)
'''


# Finally, train the model. This may take a few minutes. When training is complete, the training and validation log losses will be output:

# In[ ]:


'''
# Train for 200 steps
linear_classifier.train(
  input_fn=training_input_fn,
  steps=200
)

# Compute predictions.    
training_probabilities = linear_classifier.predict(
    input_fn=predict_training_input_fn)
training_probabilities = np.array(
      [item['probabilities'] for item in training_probabilities])
    
validation_probabilities = linear_classifier.predict(
    input_fn=predict_validation_input_fn)
validation_probabilities = np.array(
    [item['probabilities'] for item in validation_probabilities])
    
training_log_loss = metrics.log_loss(
    training_targets, training_probabilities)
validation_log_loss = metrics.log_loss(
    validation_targets, validation_probabilities)
  
# Print the training and validation log loss.
print("Training Loss: %0.2f" % training_log_loss)
print("Validation Loss: %0.2f" % validation_log_loss)

auc = metrics.auc
'''


# Next, let's calculate the [AUC (area under the curve)](https://developers.google.com/machine-learning/glossary#AUC), which is the metric this competition uses to assess the accuracy of prediction. This may take a few minutes. When calculation is complete, the training and validation AUC values will be output:

# In[ ]:


'''
training_metrics = linear_classifier.evaluate(input_fn=predict_training_input_fn)
validation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)

print("AUC on the training set: %0.2f" % training_metrics['auc'])
print("AUC on the validation set: %0.2f" % validation_metrics['auc'])
'''


# We've achieved AUC values of 0.56, which is slightly better than random. This is a good start, but can you improve the model to achieve better results?

# ## What to Try Next
# 
# A couple ideas for model refinements you can try to see if you can improve model accuracy:
# 
# * Try adjusting the `learning_rate` and `steps` hyperparameters on the existing model.
# * Try adding some text features to the model, such as the content of the project essays (`project_essay_1`, `project_essay_2`, `project_essay_3`, `project_essay_4`). You may want to try building a vocabulary from these strings; see the Machine Learning Crash Course [Intro to Sparse Data and Embeddings exercise](https://colab.research.google.com/notebooks/mlcc/intro_to_sparse_data_and_embeddings.ipynb) for some practice on working with text data and vocabularies.  

# ## Submitting a Kaggle Entry
# 
# Once you're satisfied with your model performance, you can make predictions on the test set as follows (this may take a few minutes to run):

# In[ ]:


'''
# Filepath to main test dataset.
test_file_path = '../input/test.csv'

# Read data and store in DataFrame.
test_data = pd.read_csv(test_file_path, sep=',')

my_feature_name = 'teacher_number_of_previously_posted_projects'

# Get test features
test_examples = test_data[[my_feature_name]].copy()

# No labels in data set, so generate some placeholder values
placeholder_label_vals = [0 for i in range(0, 78035)]
test_labels = pd.DataFrame({"project_is_approved": placeholder_label_vals})

predict_test_input_fn = lambda: my_input_fn(test_examples,
                                            test_labels, # unused for prediction
                                            num_epochs=1, 
                                            shuffle=False)

# Make predictions
predictions_generator = linear_classifier.predict(input_fn=predict_test_input_fn)
predictions_list = list(predictions_generator)

# Extract probabilities
probabilities = [p["probabilities"][1] for p in predictions_list]
print("Done extracting probabilities")
'''


# We want to format our submission as a CSV with two fields for each example: `id` and our prediction for `project_is_approved`, e.g.:

# 
# id,project_is_approved
# p233245,0.54
# p096795,0.14
# p236235,0.94
# ```

# Run the following code to create a `DataFrame` in the required format:

# In[ ]:


#my_submission = pd.DataFrame({'id': test_data["id"], 'project_is_approved': probabilities})
#print(my_submission.values)


# Then write your output to CSV:

# In[ ]:


#my_submission.to_csv('my_submission.csv', index=False)


# Next, click the **Commit & Run** button to execute the entire Kaggle kernel. This will take ~10 minutes to run. 
# 
# When it's finished, you'll see the navigation bar at the top of your screen has an **Output** tab. Click on the **Output** tab, and click on the **Submit to Competition** button to submit to Kaggle.
