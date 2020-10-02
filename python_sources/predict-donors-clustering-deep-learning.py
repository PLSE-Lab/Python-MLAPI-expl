#!/usr/bin/env python
# coding: utf-8

# <h1><center><font color='coral'>Machine Learning - Pair Donors to Classroom Requests</font></center></h1>

# <img src="http://internet.savannah.chatham.k12.ga.us/schools/mes/PublishingImages/DonorsChoose_org_logo.jpg" alt="Drawing" style="width: 500px;">
# ## Problem Statement
# 
# DonorsChoose.org has funded over 1.1 million classroom requests through the support of 3 million donors, the majority of whom were making their first-ever donation to a public school. If DonorsChoose.org can motivate even a fraction of those donors to make another donation, that could have a huge impact on the number of classroom requests fulfilled. The requirement is to pair up donors to the classroom requests that will most motivate them to make an additional gift.
# 

# ## Solution Approach
# On a high level the approach I choose is to split donors with similar preferences in various segments using unsupervised learning and then train a Deep Learning model to pair the donor segments with appropriate projects. 
# 
# Every month 10s of thosands of projects are submitted to DonorsChoose. Few very important quesetions that the organization would need to think of:
#     1. Which all projects they want to promote? It can be determined either using the policy or can be another machine learning based process.
#     2. How many donors does the organization want to reach out to? Too many e-mail or project can very easily wear out the donors and too few of them would probably not help to achieve the goal. Based on analysis of data, more donors are expected to donate in certain time of the year. That's something organization should also think about. 
# 
# The predictive model which is created below can be very flexibile. What makes it a good choice is that one can iterate over the process multiple times to narrow down the number of donors one would want to approach. Here is a brief flow of how it works:
#     1. Say, you received 10 new classroom requests and want to know which donors might be inclined to donate to it.
#     2. Starting from the top, the process can be executed and at the end it would suggest a list of donors that may find the project interesting.
#     3. The initial run might return 10K donors (this can be configured, of course) or more.
#     4. In case the goal is to target a much smaller number of donors, the process can be run again but this time only for a subset of donors (Output from previous step).
#     5. On an average each project has 5 donors (median is 3). It may not be useful to send e-mail request to all potential donors in one go. The iterative process can help target donors in phases, starting with the most likely ones. 

# ## Key Insights for Developing ML Solution
# 
# Below are the key insights which will be useful for buiding the ML modes. Click [here](https://www.kaggle.com/aamitrai/data-visualization-ml-for-predicting-donors) to take a look at the analysis notebook.
# 
# - 73% customers have donated only once. If organization has some additional data, such as clickstream, it would be a lot usefulfor creating better profile the donors.
# 
# - Given that a such a high number of donors only have a single record, it would not be possible to say with certainity what motivated them to donate in the first place. For this specific reason, I would not be using the essay descriptions and other freeform text; it may lead to overfitting. I would instead rely on the combination of Project category, subcategory, resource category, and donation amount to find high level pattern.
# 
# - 1% customers have donated more than 100 times. Give the nature of operations, there should be a slightly different (more personalized) strategy for high volume donors. Deep Learning can be used to build a much better profile for such 
# 
# - Some of the donors are donating thousands of time. On DonorsChoose website, I can see that they parter with many organizations. Some of these orgs match individual donations on the project. That may be one of the reason for these outliers (cases with thousands of donation), although I may be wrong on this one. Data related to such institutions should be removed before building the models.
# 
# - 22% of the project could not get required funding. In my analysis I could not find any significant pattern that may lead to projects not being funded; though, project cost does have some influence on funding. It would be interesting to see if ML can help provide some insights on probability of the project getting funded.
# 
# - The bluk of classroom requests cost between $150 - $1,500. There are decent number of outliers though. Some projects are as expensive as $255K.
# 
# - Most donors seem to donate between $10 - $100
# 
# - A huge chunk of classroom requests orginate from California. Most of the donors are from California as well.
# 
# - A large percentage of donors prefer to donate within their own state, although, residents in norteastern US seem to have less bias. Thus, location would be an important factor in recommending the project.
# 
# 
# 
# Few features which I'll be skipping deliberately:
# - Project Essays, while informational, may not add a lot of value for ML, based on the reasons explained earlier.
# - Bulk of projects are teacher lead, thus that attribute would not be relevant for analysis.
# - Teacher's Gender should not impact donation
# - Project cost is something that I would probably experiment, but not in first iteration.
# 
# 
# 
# We identified few Data Integrity issues as well
# - There are 2 duplicate 'Projects' in project file
# - There are 5.5K Donors present in 'Donation' dataset, but missing in 'Donor' dataset. If there is no info about donors, it would make sense to drom them from recommendation process.
# 

# ## Import Required Libraries and Datasets

# In[1]:


# General libraries
import os
from collections import Counter
import warnings

# Data analysis and preparation libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly.offline import iplot
import plotly.graph_objs as go
import cufflinks as cf

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from keras import optimizers, initializers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

# Set configuration
cf.go_offline()
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 300)


# In[2]:


get_ipython().run_cell_magic('time', '', "input_dir = '../input/'\nall_projects = pd.read_csv(input_dir + 'Projects.csv')\nall_donations = pd.read_csv(input_dir + 'Donations.csv')\nall_donors = pd.read_csv(input_dir + 'Donors.csv')\nall_schools = pd.read_csv(input_dir + 'Schools.csv')\nall_resources = pd.read_csv(input_dir + 'Resources.csv')\nall_teachers = pd.read_csv(input_dir + 'Teachers.csv')")


# In[3]:


# Number of donors for building model with a subset
donor_count = 50000 

# Number of segments the donors needs to be split into
# One should choose the number of segments based on the sample size and volume of donors who need to be targeted.
n_donor_segments = 20


# #### Select Donors who have donated to 5 or less projects
# #### I have assumed that the goal is to motivate the donors who are not donating as often. So for this solution, I am only considering donors who have donated less than 5 times.

# ### Select Donation records, for building Model

# In[4]:


don_count = (all_donations.groupby('Donor ID')
             .size()
             .to_frame(name='Number of Donations')
             .reset_index()
            )

# Select target donors
target_donors = don_count[don_count['Number of Donations'] <= 5]
target_donations = all_donations[all_donations['Donor ID'].isin(target_donors['Donor ID'])]
target_donations = target_donations.sort_values('Donation Received Date', ascending=False)
target_donors = target_donations.drop_duplicates('Donor ID', keep='first')['Donor ID']
target_donors = target_donors.head(donor_count).to_frame(name='Donor ID').reset_index()

# Select target projects
target_projects = target_donors.merge(target_donations, on='Donor ID')
target_projects = target_projects['Project ID'].unique()
target_projects = all_projects[all_projects['Project ID'].isin(target_projects)]

# Select target donation
target_donations = target_donations[target_donations['Donor ID'].isin(target_donors['Donor ID'].values)]


# ### Merge with Project, Schools and Donors Dataset

# In[5]:


# merged donations
merged_donation = target_donations.merge(target_projects, on='Project ID')
merged_donation = merged_donation.merge(all_donors, on='Donor ID')
merged_donation = merged_donation.merge(all_schools, on='School ID')
merged_donation.shape


# ### Remove duplicates
# 
# Some donors have donated to the same project multiple times. Aggregate the total donation amount and remvoe duplicate records

# In[6]:


donation_cols = ['Project ID', 'Donor ID', 'Donation Amount', 'Project Subject Category Tree',
                'Project Subject Subcategory Tree', 'Project Grade Level Category', 'Project Resource Category',
                'Project Cost', 'Donor State', 'Donor Is Teacher', 'School Metro Type', 'School Percentage Free Lunch',
                'School State']

donation_master = (merged_donation.groupby(['Project ID', 'Donor ID'])
                   .agg({'Donation Amount':'sum'})
                   .rename(columns={'Donation Amount': 'Total Donation'})
                   .reset_index()
                  )

donation_master = merged_donation[donation_cols].merge(donation_master, on=['Project ID', 'Donor ID'])
donation_master = (donation_master.drop_duplicates(['Project ID', 'Donor ID'], keep='first')
                   .drop('Donation Amount', axis=1)
                   .rename(columns={'Total Donation':'Donation Amount'})
                  )
donation_master.head()


# ### Analyze distribution of project categories and resources

# In[7]:


# Project Category and Subcategory are stacked columns. A classromm request can span across multiple categories.
# I will start by exploding the columns and then analyze the trend over the years
def stack_attributes(df, target_column, separator=', '):
    df = df.dropna(subset=[target_column])
    df = (df.set_index(df.columns.drop(target_column,1).tolist())
          [target_column].str.split(separator, expand=True)
          .stack().str.strip()
          .reset_index()
          .rename(columns={0:target_column})
          .loc[:, df.columns])
    df = (df.groupby([target_column, 'Project Posted Date'])
          .size()
          .to_frame(name ='Count')
          .reset_index())
    
    return df

def plot_trend(df, target_column, chartType=go.Scatter,
              datecol='Project Posted Date', 
              ytitle='Number of relevant classroom requests'):
    trend = []
    for category in list(df[target_column].unique()):
        temp = chartType(
            x = df[df[target_column]==category][datecol],
            y = df[df[target_column]==category]['Count'],
            name=category
        )
        trend.append(temp)
    
    layout = go.Layout(
        title = 'Trend of ' + target_column,
        xaxis=dict(
            title='Year & Month',
            zeroline=False,
        ),
        yaxis=dict(
            title=ytitle,
        ),
    )
    
    fig = go.Figure(data=trend, layout=layout)
    iplot(fig)
    
proj = all_projects[['Project Subject Category Tree',
                     'Project Subject Subcategory Tree',
                     'Project Resource Category',
                     'Project Grade Level Category',
                     'Project Posted Date']].copy()
proj['Project Posted Date'] = all_projects['Project Posted Date'].str.slice(start=0, stop=4)

proj_cat = stack_attributes (proj, 'Project Subject Category Tree')
proj_sub_cat = stack_attributes (proj, 'Project Subject Subcategory Tree')
proj_res_cat = (proj.groupby(['Project Resource Category', 'Project Posted Date'])
                .size()
                .to_frame(name ='Count')
                .reset_index())
proj_grade_cat = (proj.groupby(['Project Grade Level Category', 'Project Posted Date'])
                .size()
                .to_frame(name ='Count')
                .reset_index())

plot_trend(proj_cat, 'Project Subject Category Tree')
plot_trend(proj_sub_cat, 'Project Subject Subcategory Tree')
plot_trend(proj_res_cat, 'Project Resource Category')


# ### One-hot encode 
# - Project Subject Category Tree
# - Project Subject Subcategory Tree
# - Project Resource Category
# 
# For some of the projects, there are multiple values for these attributes. I'll be splitting those, prior to one hot encoding
# 
# ### Group less prevelant categories as 'Other'
# As we saw in our analysis above, for each of the three features mentioned above, there are some categories which more popular than other. To balance the dataset, I'll club the less popular categories as 'Other'

# In[8]:


donor_master = donation_master.copy()

# One-hot encode Project Subject Category Tree
cat = (pd.DataFrame(donor_master['Project Subject Category Tree']
                     .str.split(', ').tolist())
        .stack().str.strip()
        .to_frame('col')
       )
keep_cols = ['Literacy & Language', 'Math & Science', 'Applied Learning', 'Music & The Arts']
cat[~cat['col'].isin(keep_cols)] = 'Other'

donor_master = (donor_master.drop('Project Subject Category Tree', 1)
                .join(pd.get_dummies(cat, prefix='Project Subject Category Tree', prefix_sep='_')
                      .sum(level=0))
               )
donor_master.ix[donor_master['Project Subject Category Tree_Other'] > 1, 'Project Subject Category Tree_Other'] = 1


# One-hot encode Project Subject Subcategory Tree
sub_cat = (pd.DataFrame(donor_master['Project Subject Subcategory Tree']
                     .str.split(', ').tolist())
        .stack().str.strip()
        .to_frame('col')
       )
keep_cols = ['Literacy', 'Mathematics', 'Literature & Writing', 'Special Needs', 'Early Development',
            'Environmental Science']
sub_cat[~sub_cat['col'].isin(keep_cols)] = 'Other'

donor_master = (donor_master.drop('Project Subject Subcategory Tree', 1)
                .join(pd.get_dummies(sub_cat, prefix='Project Subject Subcategory Tree', prefix_sep='_')
                      .sum(level=0))
               )
donor_master.ix[donor_master['Project Subject Subcategory Tree_Other'] > 1, 'Project Subject Subcategory Tree_Other'] = 1

# One-hot encode Project Resource Category
resrc = (pd.DataFrame(donor_master['Project Resource Category']
                     .str.split(', ').tolist())
        .stack().str.strip()
        .to_frame('col')
       )
keep_cols = ['Supplies', 'Technology', 'Books', 'Computers & Tablets']
resrc[~resrc['col'].isin(keep_cols)] = 'Other'

donor_master = (donor_master.drop('Project Resource Category', 1)
                .join(pd.get_dummies(resrc, prefix='Project Resource Category', prefix_sep='_')
                      .sum(level=0))
               )
donor_master.ix[donor_master['Project Resource Category_Other'] > 1, 'Project Resource Category_Other'] = 1
donor_master.head(10)


# ### Lable the donations as 'in-state' and 'out of state'

# In[9]:


donor_master['In State'] = donor_master['School State'] == donor_master['Donor State']
donor_master['In State'] = donor_master['In State'].astype(int)
donor_master.drop(['School State', 'Donor State'], axis=1, inplace=True)
donor_master.head()


# ### Convert 'School Percentage Free Lunch' to decimal

# In[10]:


donor_master['School Percentage Free Lunch'] = donor_master['School Percentage Free Lunch'] / 100
donor_master.head()


# ### Convert Donor 'Is Teacher' to 1/0

# In[11]:


donor_master['Donor Is Teacher'] = donor_master['Donor Is Teacher'].map(dict(Yes=1, No=0))
donor_master.head()


# ### Convert Project Cost & Donation Amount in discrete buckets, with approx. normal distribution
# 
# I've experimentd and choosen buckets to make it as practical as possible. For example, it appears a lot of people are donating $10, $25 and $50. So it makes sense to have buckets on same lines

# In[12]:


custom_bucket = [0, 179, 299, 999, 2500, 100000]
#custom_bucket = [0, 199, 399, 999, 2999, 100000]
custom_bucket_label = ['Vey Low', 'Low', 'Medium', 'High', 'Very High']
donor_master['Project Cost'] = pd.cut(donor_master['Project Cost'], custom_bucket, 
                                      labels=custom_bucket_label)

(donor_master['Project Cost'].value_counts()
                             .sort_index()
                             .iplot(kind='bar', xTitle = 'Project Cost', yTitle = "Project Count", 
                                    title = 'Distribution on Project Cost', color='green')
)

custom_bucket = [0, 4.99, 19.99, 99.99, 499.99, 1000000]
#custom_bucket = [0, 5, 25, 100, 300, 1000000]
custom_bucket_label = ['Vey Low', 'Low', 'Medium', 'High', 'Very High'] # Creating a dummy hierarchy
donor_master['Donation Amount'] = pd.cut(donor_master['Donation Amount'], custom_bucket, labels=custom_bucket_label)

(donor_master['Donation Amount'].value_counts()
                             .sort_index()
                             .iplot(kind='bar', xTitle = 'Donation Amount', yTitle = 'Donation Count',
                             title = 'Simulated Distribution on Donation Amount')
)


# ### One-hot encode remaining categorical columns

# In[13]:


cat_cols = ['Project Grade Level Category', 'School Metro Type', 'Project Cost', 'Donation Amount']
donor_master = pd.get_dummies(data=donor_master, columns=cat_cols)
donor_master.head()


# ### Create single master record for each donor

# In[14]:


donor_master_final = donor_master.drop('Project ID', axis=1).copy()
all_cols = list(donor_master_final.columns)
all_cols.remove('Donor ID')
action = {col : 'max' for col in all_cols}
action['School Percentage Free Lunch'] = 'median'
donor_master_final = donor_master_final.groupby('Donor ID').agg(action).reset_index()
donor_master_final.set_index('Donor ID', inplace=True)
donor_master_final.fillna(0, inplace=True)
donor_master_final.head(10)


# ### Perform PCA to understand the importance of features

# In[15]:


from sklearn.decomposition import PCA
pca = PCA()
projected = pca.fit(donor_master_final)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

pca = PCA(0.9).fit(donor_master_final)
pca.n_components_
# temp = PCA(pca.n_components_).fit(donor_master_final)
# donor_master_final = temp.transform(donor_master_final)


# I am not good at PCA, but it appears that all the features play a minor role and atlease 26 features are required to explain 90% variance. So, I'll be using all of these to train clustering model. I hope there would be a better way to do clustering for such sparse datasets.

# ## Donor Segmentation
# 
# I played around with k-means, gaussian mixture model, dbscan and hdbscan algorithms. K-mean and 'GaussianMixture' both seem pretty reasonable, but the latter appears to have slightly better clustering. I couldn't get desired results from dbscan and hdbscan at all. One can play around by tweaking the epsilon values.
# 
# Downside of GaussianMixture is that it tend to be a lot slower. K-mean has been by far the fastest one to run.

# In[16]:


get_ipython().run_cell_magic('time', '', "def k_donor_segment(donors):\n    kmeans = KMeans(init='k-means++', n_clusters=n_donor_segments, \n                    n_init=10, precompute_distances=True)\n    kmeans.fit(donors)\n    print('k-means intertia - {}'.format(kmeans.inertia_))\n\n    lab = kmeans.labels_\n    segments = Counter()\n    segments.update(lab)\n\n    return lab, segments\n\ndef g_donor_segment(donors):\n    # Predict GMM clusters\n    gm = GaussianMixture(n_components=n_donor_segments)\n    gm.fit(donors)\n    \n    lab = gm.predict(donors)\n    segments = Counter()\n    segments.update(lab)\n\n    return lab, segments\n\ndef d_donor_segment(donors):\n    db = DBSCAN(eps=3, min_samples=10)\n    db.fit(donors)\n\n    lab = db.labels_\n    segments = Counter()\n    segments.update(lab)\n\n    return lab, segments\n\nlabel, donor_segment = k_donor_segment(donor_master_final)\n\ndonor_segment_mapping = {donor_id : donor_seg for donor_id, donor_seg \n                   in zip(list(donor_master_final.index), list(label))}\n\ndisplay(donor_segment.most_common(10))")


# ### Prepare data for Neural Net and generate the Train & test data
# 
# Generate the project master record. The idea is to create one record for each project with all necessary features. This will be used for Deep Learning and eventually predicting suitable donors for the projects.
# 
# 1. Project 'train' dataset for training deep learning model
# 2. Project 'validation' dataset for testing the deep learning model
# 3. Project 'testing' dataset for evaluating the final accuracy

# In[17]:


col = ['Donation Amount_Vey Low', 'Donation Amount_Low', 'Donation Amount_Medium', 'Donation Amount_High',
       'Donation Amount_Very High', 'In State', 'Donor Is Teacher']
proj_master = donor_master.drop(col, axis=1).copy()

proj_donor_map = (proj_master.groupby('Project ID')['Donor ID']
                  .apply(list)
                  .to_frame('Donors')
                  .reset_index()
                 )
proj_donor_map['Donors'] = proj_donor_map['Donors'].apply(lambda x: list(set(x)))
proj_master = proj_master.merge(proj_donor_map, on='Project ID', how='inner')
proj_master.drop('Donor ID', axis=1, inplace=True)
proj_master.drop_duplicates('Project ID', keep='first')

proj_master.set_index('Project ID', inplace=True)
proj_master.fillna(0, inplace=True)
features, lables = proj_master.drop('Donors', axis=1), proj_master['Donors']


# Split the train and test dataset 
train_features, test_features, train_lables, test_lables = train_test_split(features, lables,
                                                                           test_size=.3)
# Split the test dataset into validatin and test dataset
test_features, valid_features, test_lables, valid_lables = train_test_split(test_features, test_lables,
                                                                           test_size=.5)

def one_hot_encode_labels(proj_donors, donor_segment_mapping):
    n_donor_segments = len(set(donor_segment_mapping.values()))
    lables = np.zeros(shape=(proj_donors.shape[0], n_donor_segments))
    
    def get_max(x):
        ''' For given row convert highest value to 1 and rest to zero
            For give
        '''
        max = np.unravel_index(x.argmax(), x.shape)
        x = x * 0
        x[max] = 1
        return x

    for i, values in enumerate(proj_donors):
        for val in values:
            segment = donor_segment_mapping[val]
            lables[i][segment] += 1
        lables[i] = get_max(lables[i])
        
    return lables

train_lables = one_hot_encode_labels(train_lables, donor_segment_mapping)
valid_lables = one_hot_encode_labels(valid_lables, donor_segment_mapping)
test_lables = one_hot_encode_labels(test_lables, donor_segment_mapping)


# ### Build the neural network (Keras)

# In[19]:


def build_nn(X_train, Y_train, X_valid, Y_valid,
             epochs=2000, batch_size=200,
             activation='relu',
             layer_size=[100, 50, 30],
             dropout=[.3, .2, 0]):
    
    n_input = X_train.shape[1]
    n_classes = Y_train.shape[1]
    init = initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None)
    
    # opt =  optimizers.Adam()
    # opt = optimizers.Adam(lr=0.0005, amsgrad=False)
    # opt = optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    # opt = optimizers.Adagrad(lr=0.001, epsilon=None, decay=0.0)
    opt = optimizers.Nadam(lr=0.0035, beta_1=0.1, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    checkpointer = ModelCheckpoint(filepath='donors.hdf5', verbose=False, save_best_only=True)
    model = Sequential() 
    for i, val in enumerate(range(len(layer_size))):
        if i == 0:
            model.add(Dense(layer_size[i], activation=activation, 
                            input_shape=(n_input,), kernel_initializer=init))
        else:
            model.add(Dense(layer_size[i], activation=activation, kernel_initializer=init))
        model.add(Dropout(dropout[i]))    

    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    logs = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, 
                     validation_data=(X_valid, Y_valid), shuffle=True, verbose=0,
                    callbacks=[checkpointer])
    
    model.load_weights('donors.hdf5')
    score = model.evaluate(X_train, Y_train)
    print("Training Accuracy:", score[1])
    score = model.evaluate(X_valid, Y_valid)
    print("Validation Accuracy:", score[1])

    return model, logs
    


# ### Train and validate the model

# In[20]:


get_ipython().run_cell_magic('time', '', "model, logs = build_nn(train_features.values, train_lables,\n                       valid_features.values, valid_lables, \n                       epochs=300)\n\nprint ('Model Training Completed')\n\n# Analyze accuracy over epochs\nplt.plot(logs.history['acc'])\nplt.plot(logs.history['val_acc'])\nplt.ylabel('Accuracy')\nplt.xlabel('epoch #')\nplt.legend(['train', 'test'], loc='upper left')\nplt.title('Trend of Accuracy')\nplt.show()\n\n# Analyze loss over epochs\nplt.plot(logs.history['loss'])\nplt.plot(logs.history['val_loss'])\nplt.ylabel('Loss')\nplt.xlabel('epoch #')\nplt.legend(['train', 'validation'], loc='upper left')\nplt.title('Trend of Loss')\nplt.show()")


# ### List Donors for the project
# 
# Once the model is ready, run the prediction and generate potential list of donors from donor segment. I'll be doing that on my test set
# 
# 1. Validate the accuracy on test dataset
# 2. Predict Donor Segment for test projects
# 3. Pair potential donors to the project (convert donor segment into list of donors)

# In[21]:


score = model.evaluate(test_features.values, test_lables)
print("Testing Accuracy:", score[1])

# Predict Donor segment for each project
predict_segment = model.predict_classes(test_features.values)

# Generate an dictionary with a mapping of projects and potential donors
# Reverse engineer a mapping of donors and segments
donor_cluster = {value : [] for value in set(donor_segment_mapping.values())} 
{donor_cluster[val].append(key)for key, val in donor_segment_mapping.items()}

# Predicted donors
predicted_donors = {proj : donor_cluster[seg] for 
                    proj, seg in zip(list(test_features.index), list(predict_segment))}

print ('Prediction Completed')
# Let's display number of paired donors for some projects
sample = 10
for i, key in enumerate(predicted_donors):
    print('There are {:,} potential donors for {} project'
          .format(len(predicted_donors[key]), key))
    if i == sample: break
    


# ## Conclusion
# 
# While the model may not seem as good (with accuracy ~ 45%), it is quite resonable based on the limited data. I wanted to make sure that model works even with the folks who have donated relatively few times. Thus, I only chose donors who have donated less than 5 times. The model can be extended to include other generous donors as well. However, one need to be careful that including the frequent donors, may make the dataset unbalaced and the neural net may get biased towards frequent donors. To counterbalance, one can focus that segmentation in done appropriately to distribute the donors segments in realtively uniform manner.

# ## Potential Enhancements
# 
# Experts would probably be able to further fine tune this model. There are some of the potential opportunities for imporovement:
# - Additional Data: It would help a great deal to bring in some additional data about donors which can further help optimize the predictions. Something like clickstream may prove extremly valuable.
# - Use NLP to leverage project text description, essays, etc. I deliberately skipped those since the project category, subcategory and resource categories were available. However, it is completly possible that the way an essay is written will influence individuals to donate
# - Use Donor States an School State for better personalization. I skipped it as it was increasing the feature space 2x time and the data was very sparse.
# - The above process can be run iteratively to narrow down the potential donor list.
# 

# <h1><center><font color='black'>Thanks for taking a look :)</font></center></h1>
