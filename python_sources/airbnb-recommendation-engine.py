#!/usr/bin/env python
# coding: utf-8

# This dataset is a good opportunity to practice feature engineering and modelling for a recommendation engine. There is a good quantity of data to do some intereting feature engineering with both user demographics and session logs while not being so large that it takes a long time to process the data. Below is my practice at engineering some features and fitting a model.

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
import ml_metrics as metrics
from sklearn.preprocessing import MinMaxScaler


# ## Load and split data
# 
# Begin by loading the data. 

# In[ ]:


train_users = pd.read_csv(
    '/kaggle/input/airbnb-recruiting-new-user-bookings/train_users_2.csv.zip',
    parse_dates = ['timestamp_first_active', 'date_account_created', 'date_first_booking']
)


# ## Raw Feature Analysis
# 
# Below is a quick bit of analysis on each of the raw features in the data. I'll begin with the distribution of values across each feature.

# In[ ]:


fig, ax = plt.subplots(3, 3, figsize=(20, 12))

# row one
ax[0,0].set_title('Age')
train_users['age'].hist(ax=ax[0,0])

ax[0,1].set_title('Language')
train_users['language'].value_counts().plot.bar(ax=ax[0,1])

ax[0,2].set_title('Gender')
train_users['gender'].value_counts().plot.bar(ax=ax[0,2])

# row two
ax[1,0].set_title('Signup Method')
train_users['signup_method'].value_counts().plot.bar(ax=ax[1,0])

ax[1,1].set_title('Signup Flow')
train_users['signup_flow'].value_counts().plot.bar(ax=ax[1,1])

ax[1,2].set_title('Affliate Channel')
train_users['affiliate_channel'].value_counts().plot.bar(ax=ax[1,2])

# row three
ax[2,0].set_title('Affliate Provider')
train_users['affiliate_provider'].value_counts().plot.bar(ax=ax[2,0])

ax[2,1].set_title('Signup App')
train_users['signup_app'].value_counts().plot.bar(ax=ax[2,1])

ax[2,2].set_title('First Browser')
train_users['first_browser'].value_counts().nlargest(8).plot.bar(ax=ax[2,2])


# There are a lot of NaN values in the age column and some bad data (apparently there is someone who is 2014 years old). Usually I would drop the rows but the large number of them might have a negative effect on the training. As such I will fill the NaN and outlier values with the mean age.

# In[ ]:


train_users['age'] = train_users['age'].fillna(train_users['age'].mean())
train_users.loc[train_users['age'] > 1920, 'age'] = train_users['age'].mean()


# Now I'll take a look at how the features correlate with the label. As many of the features are categorical in nature there are a lot of charts to produce which involves quite a few lines of code. I've tried to condense this as much as possible but have ended up needing to split the charts across two code blocks.

# In[ ]:


fig, ax = plt.subplots(3, 6, figsize=(20, 14))

# row one
ax[0,0].set_title('Age')
train_users.plot.scatter(x='country_destination', y='age', ax=ax[0,0])

ax[0,1].set_title('Gender - UNKNOWN')
train_users[train_users['gender'] == '-unknown-']['country_destination'].value_counts().plot.bar(ax=ax[0,1], color='firebrick')

ax[0,2].set_title('Gender - FEMALE')
train_users[train_users['gender'] == 'FEMALE']['country_destination'].value_counts().plot.bar(ax=ax[0,2], color='firebrick')

ax[0,3].set_title('Gender - MALE')
train_users[train_users['gender'] == 'MALE']['country_destination'].value_counts().plot.bar(ax=ax[0,3], color='firebrick')

ax[0,4].set_title('Gender - OTHER')
train_users[train_users['gender'] == 'OTHER']['country_destination'].value_counts().plot.bar(ax=ax[0,4], color='firebrick')

ax[0,5].set_title('Signup Flow')
train_users.plot.scatter(x='country_destination', y='signup_flow', ax=ax[0,5], color='forestgreen')

# row two
ax[1,0].set_title('Lang - en')
train_users[train_users['language'] == 'en']['country_destination'].value_counts().plot.bar(ax=ax[1,0], color='gold')

ax[1,1].set_title('Lang - fr')
train_users[train_users['language'] == 'fr']['country_destination'].value_counts().plot.bar(ax=ax[1,1], color='gold')

ax[1,2].set_title('Lang - zh')
train_users[train_users['language'] == 'zh']['country_destination'].value_counts().plot.bar(ax=ax[1,2], color='gold')

ax[1,3].set_title('Lang - es')
train_users[train_users['language'] == 'es']['country_destination'].value_counts().plot.bar(ax=ax[1,3], color='gold')

ax[1,4].set_title('Lang - it')
train_users[train_users['language'] == 'it']['country_destination'].value_counts().plot.bar(ax=ax[1,4], color='gold')

ax[1,5].set_title('Lang - de')
train_users[train_users['language'] == 'de']['country_destination'].value_counts().plot.bar(ax=ax[1,5], color='gold')

# row three
ax[2,0].set_title('Signup method - basic')
train_users[train_users['signup_method'] == 'basic']['country_destination'].value_counts().plot.bar(ax=ax[2,0], color='orchid')

ax[2,1].set_title('Signup method - facebook')
train_users[train_users['signup_method'] == 'facebook']['country_destination'].value_counts().plot.bar(ax=ax[2,1], color='orchid')

ax[2,2].set_title('Signup App - web')
train_users[train_users['signup_app'] == 'Web']['country_destination'].value_counts().plot.bar(ax=ax[2,2], color='pink')

ax[2,3].set_title('Signup App - ios')
train_users[train_users['signup_app'] == 'iOS']['country_destination'].value_counts().plot.bar(ax=ax[2,3], color='pink')

ax[2,4].set_title('Signup App - android')
train_users[train_users['signup_app'] == 'Android']['country_destination'].value_counts().plot.bar(ax=ax[2,4], color='pink')

ax[2,5].set_title('Signup App - moweb')
train_users[train_users['signup_app'] == 'Moweb']['country_destination'].value_counts().plot.bar(ax=ax[2,5], color='pink')


# ### Age
# It looks like there is some relation between age and the label. Older generations seem more likely to go to the US or other for example. It may be worth bucketing the age column so that it represents age groups rather than a linear scale though this would create more columns for the model to process. Note: I tried bucketing the age column into categorical columns but it damaged to model so I have kept the column as a numerical column. 
# 
# ### Gender
# There seems to be a pattern here genders and destinations. Men and women are similar in their first four choices (NDF, America, other and France) though men do seem to prefer Britain and Spain while women prefer Italy. Unknown gender are far less likely to make a booking than any other gender while the "other" gender is more likely to make a booking in America than they are to not make a booking. This feature then seems like a good candidate for a one hot categorical feature.
# 
# ### Signup Flow
# This is the page number the user signed up from. There doesn't seem to be a relation here either so this feature will be excluded from the model.
# 
# ### Language
# There seems to be some relation here between language and destination. English speakers for example seem much more likely to book in the US while other languages such as French seem to correlate with countries such as France more than other languages do.
# 
# ### Signup method
# There doesn't seem to be much of a relation between sign up method and the label. Consider removing this one from the model to save on processing.
# 
# ### Signup App
# There seems to be a small relation here so this will be included in the model as a one hot categorical feature. Note: when I later added this to the model it brought the score down so I have removed the feature from the model.

# In[ ]:


fig, ax = plt.subplots(3, 6, figsize=(20, 14))

# row one
ax[0,0].set_title('Aff Channel - direct')
train_users[train_users['affiliate_channel'] == 'direct']['country_destination'].value_counts().plot.bar(ax=ax[0,0])

ax[0,1].set_title('Aff Channel - sem-brand')
train_users[train_users['affiliate_channel'] == 'sem-brand']['country_destination'].value_counts().plot.bar(ax=ax[0,1])

ax[0,2].set_title('Aff Channel - sem-non-brand')
train_users[train_users['affiliate_channel'] == 'sem-non-brand']['country_destination'].value_counts().plot.bar(ax=ax[0,2])

ax[0,3].set_title('Aff Channel - other')
train_users[train_users['affiliate_channel'] == 'other']['country_destination'].value_counts().plot.bar(ax=ax[0,3])

ax[0,4].set_title('Aff Channel - api')
train_users[train_users['affiliate_channel'] == 'api']['country_destination'].value_counts().plot.bar(ax=ax[0,4])

ax[0,5].set_title('Aff Channel - seo')
train_users[train_users['affiliate_channel'] == 'seo']['country_destination'].value_counts().plot.bar(ax=ax[0,5])

# row two
ax[1,0].set_title('Aff Channel - content')
train_users[train_users['affiliate_channel'] == 'content']['country_destination'].value_counts().plot.bar(ax=ax[1,0])

ax[1,1].set_title('Aff Channel - remarketing')
train_users[train_users['affiliate_channel'] == 'remarketing']['country_destination'].value_counts().plot.bar(ax=ax[1,1])

ax[1,2].set_title('Browser - Chrome')
train_users[train_users['first_browser'] == 'Chrome']['country_destination'].value_counts().plot.bar(ax=ax[1,2], color='firebrick')

ax[1,3].set_title('Browser - Safari')
train_users[train_users['first_browser'] == 'Safari']['country_destination'].value_counts().plot.bar(ax=ax[1,3], color='firebrick')

ax[1,4].set_title('Browser - Firefox')
train_users[train_users['first_browser'] == 'Firefox']['country_destination'].value_counts().plot.bar(ax=ax[1,4], color='firebrick')

ax[1,5].set_title('Browser - IE')
train_users[train_users['first_browser'] == 'IE']['country_destination'].value_counts().plot.bar(ax=ax[1,5], color='firebrick')

# row three
ax[2,0].set_title('Aff Provider - direct')
train_users[train_users['affiliate_provider'] == 'direct']['country_destination'].value_counts().plot.bar(ax=ax[2,0], color='forestgreen')

ax[2,1].set_title('Aff Provider - google')
train_users[train_users['affiliate_provider'] == 'google']['country_destination'].value_counts().plot.bar(ax=ax[2,1], color='forestgreen')

ax[2,2].set_title('Aff Provider - other')
train_users[train_users['affiliate_provider'] == 'other']['country_destination'].value_counts().plot.bar(ax=ax[2,2], color='forestgreen')

ax[2,3].set_title('Aff Provider - craigslist')
train_users[train_users['affiliate_provider'] == 'craigslist']['country_destination'].value_counts().plot.bar(ax=ax[2,3], color='forestgreen')

ax[2,4].set_title('Aff Provider - facebook')
train_users[train_users['affiliate_provider'] == 'facebook']['country_destination'].value_counts().plot.bar(ax=ax[2,4], color='forestgreen')

ax[2,5].set_title('Aff Provider - bing')
train_users[train_users['affiliate_provider'] == 'bing']['country_destination'].value_counts().plot.bar(ax=ax[2,5], color='forestgreen')


# ### Affiliate Channel
# 
# The type of paid marketing. There seems to be some good patterns here to include in the model. NDF and US are still the most popular destinations for all channels (though in different quantities for some) but some channels such as remarketing favour destinations such as FR more than others.
# 
# ### First Browser
# 
# There doesn't seem to be much correlation between this feature and the label so it will be excluded from the model.
# 
# ### Affiliate Provider
# 
# Where the marketing is. There doesn't seem to be much of a relation here either so I'll remove this feature from the model.

# ## Time Based Features
# 
# To take account of seasonality I have pulled the month out of the account created and first active timestamps. I'll compare this against the label to see if there is a relationship.

# In[ ]:


train_users['dac_month'] = train_users['date_account_created'].dt.month
train_users['tfa_month'] = train_users['timestamp_first_active'].dt.month


# In[ ]:


fig, ax = plt.subplots(4, 3, figsize=(20, 14))

# row one
ax[0,0].set_title('Month Account Created - Jan')
train_users[train_users['dac_month'] == 1]['country_destination'].value_counts().plot.bar(ax=ax[0,0])

ax[0,1].set_title('Year Account Created - Feb')
train_users[train_users['dac_month'] == 2]['country_destination'].value_counts().plot.bar(ax=ax[0,1])

ax[0,2].set_title('Year Account Created - Mar')
train_users[train_users['dac_month'] == 3]['country_destination'].value_counts().plot.bar(ax=ax[0,2])

# row two
ax[1,0].set_title('Month Account Created - Apr')
train_users[train_users['dac_month'] == 4]['country_destination'].value_counts().plot.bar(ax=ax[1,0])

ax[1,1].set_title('Year Account Created - May')
train_users[train_users['dac_month'] == 5]['country_destination'].value_counts().plot.bar(ax=ax[1,1])

ax[1,2].set_title('Year Account Created - Jun')
train_users[train_users['dac_month'] == 6]['country_destination'].value_counts().plot.bar(ax=ax[1,2])

# row three
ax[2,0].set_title('Month Account Created - Jul')
train_users[train_users['dac_month'] == 7]['country_destination'].value_counts().plot.bar(ax=ax[2,0])

ax[2,1].set_title('Year Account Created - Aug')
train_users[train_users['dac_month'] == 8]['country_destination'].value_counts().plot.bar(ax=ax[2,1])

ax[2,2].set_title('Year Account Created - Sep')
train_users[train_users['dac_month'] == 9]['country_destination'].value_counts().plot.bar(ax=ax[2,2])

# row four
ax[3,0].set_title('Month Account Created - Oct')
train_users[train_users['dac_month'] == 10]['country_destination'].value_counts().plot.bar(ax=ax[3,0])

ax[3,1].set_title('Year Account Created - Nov')
train_users[train_users['dac_month'] == 11]['country_destination'].value_counts().plot.bar(ax=ax[3,1])

ax[3,2].set_title('Year Account Created - Dec')
train_users[train_users['dac_month'] == 12]['country_destination'].value_counts().plot.bar(ax=ax[3,2])


# While the patterns are similar across all months there is some variations in country popularity suggesting that some countries are more popular in some months. This should make a good categorical feature then.
# 
# ## Session Logs
# 
# The second set of data to play with is the session logs. This is a list of actions that users have taken on the website. Some processing needs to be done to make these logs useful to the model. A simple use of the logs would be to count the action types per user. It's worth noting that many of the users in the training dataset do not have session logs associated with them. This is because the session logs begin in 2014 while the users data begins in 2010. This will make it a little trickier to train with the logs but it is likely that there is still some use for them.
# 
# To start processing the logs I will load them in batches (to save memory).

# In[ ]:


def get_batches():
    return pd.read_csv(
        '/kaggle/input/airbnb-recruiting-new-user-bookings/sessions.csv.zip',
        usecols=['user_id', 'action_type', 'action_detail', 'secs_elapsed'],
        chunksize=10000
    )


# and group each batch by user as it is loaded.

# In[ ]:


def load_and_group(group_column, agg_method):
    groups_list = []

    for batch in get_batches():
        groups_list.append(
            batch.groupby(['user_id', group_column])['secs_elapsed'].agg([agg_method])
        )
        
    groups = pd.concat(groups_list)
    groups = groups.groupby(['user_id', group_column]).sum()
    
    return groups 


# In[ ]:


action_counts = load_and_group(group_column='action_type', agg_method='count')
action_counts.head()


# Now pivot the data so we have one row per user with each row representing the count of each action type. Some extra work needed to be done here to handle the hierarchy of the columns that is produced by the pivot.

# In[ ]:


def pivot_log_groups(log_group, group_column, agg_method):
    log_group = log_group.reset_index().pivot(index='user_id', columns=group_column, values=[agg_method]).fillna(0)
    
    log_group.columns = log_group.columns.get_level_values(1)
    log_group = log_group.rename(columns={'-unknown-': 'unknown'})
    log_group = log_group.add_suffix('_' + group_column + '_' + agg_method)
    log_group_cols = log_group.columns.values
    log_group = log_group.reset_index()
    log_group = log_group.rename(columns={'user_id': 'id'})
    
    return log_group, log_group_cols


# In[ ]:


action_counts, action_counts_cols = pivot_log_groups(action_counts, group_column='action_type', agg_method='count')
action_counts.head()


# Finally join the logs with the users so all features are in one dataframe.

# In[ ]:


train_users = train_users.merge(action_counts, how='left', on=['id']).fillna(0)
train_users.head()


# ## Split Data
# 
# To validate the models training I will split the training dataset into a training and validation dataset. I have split the dataset using the timestamp_first_active column as that is how the training and test datasets are split.

# In[ ]:


val_users = train_users[train_users['timestamp_first_active'] >= '2014-06-01']
train_users = train_users[train_users['timestamp_first_active'] < '2014-06-01']

print('Count of training users: ' + str(len(train_users)))
print('Count of validation users: ' + str(len(val_users)))


# ## Pipeline
# 
# Now that I know what features will be used in the model the pipeline can be prepared. First I'll remove the labels from the datasets.

# In[ ]:


train_labels = train_users.pop('country_destination')
val_labels = val_users.pop('country_destination')


# Then convert them to categorical columns. This means assigning an integer index to each unique value. A dictionary is saved including the index-category mapping.

# In[ ]:


train_labels = pd.Categorical(train_labels)
val_labels = pd.Categorical(val_labels)

categories = {i:category for i, category in enumerate(train_labels.categories.values)}


# Tensorflow prefers labels to be in numerical form so I'll convert the labels to their categorical indexes.

# In[ ]:


train_labels = train_labels.codes
val_labels = val_labels.codes


# The datetime columns aren't used in the model and cause some errors when loaded into a Tensorflow dataset so I'll drop them from the data.

# In[ ]:


train_users = train_users.drop(columns=['timestamp_first_active', 'date_account_created', 'date_first_booking', 'first_affiliate_tracked'])
val_users = val_users.drop(columns=['timestamp_first_active', 'date_account_created', 'date_first_booking', 'first_affiliate_tracked'])


# Now I'll put the data into Tensorflows defined data format.

# In[ ]:


tf_train_data = tf.data.Dataset.from_tensor_slices(
    (dict(train_users), train_labels)
)

tf_val_data = tf.data.Dataset.from_tensor_slices(
    (dict(val_users), val_labels)
)


# And add some shuffling and batching logic for feeding the data into the model.

# In[ ]:


tf_train_data = tf_train_data.shuffle(100).batch(32)
tf_val_data = tf_val_data.batch(len(val_users))


# The categorical features need a vocabulary of unique values defining ready for the one hot encoding. This is done here.

# In[ ]:


language_vocab = tf.feature_column.categorical_column_with_vocabulary_list(
    'language', 
    train_users['language'].unique()
)

gender_vocab = tf.feature_column.categorical_column_with_vocabulary_list(
    'gender', 
    train_users['gender'].unique()
)

channel_vocab = tf.feature_column.categorical_column_with_vocabulary_list(
    'affiliate_channel', 
    train_users['affiliate_channel'].unique()
)

dac_month_vocab = tf.feature_column.categorical_column_with_vocabulary_list(
    'dac_month', 
    train_users['dac_month'].unique()
)

tfa_month_vocab = tf.feature_column.categorical_column_with_vocabulary_list(
    'tfa_month', 
    train_users['tfa_month'].unique()
)


# In[ ]:


features = []

for col_name in action_counts_cols:
        features.append(tf.feature_column.numeric_column(col_name))


# Before finally putting all features into a dense feature layer. This layer provides the necessary pre-processing of each feature per training step before the data is inserted into the hidden layers.

# In[ ]:


features.extend([
    tf.feature_column.numeric_column('age'),
    tf.feature_column.indicator_column(language_vocab),
    tf.feature_column.indicator_column(gender_vocab),
    tf.feature_column.indicator_column(channel_vocab),
    tf.feature_column.indicator_column(dac_month_vocab),
    tf.feature_column.indicator_column(tfa_month_vocab),
])

feature_layer = tf.keras.layers.DenseFeatures(features)


# ## Define and train model
# 
# This challenge seems to be more about feature engineering than model tuning so I will use a relatively straight forward model for this task. Here I define the model as a one hidden layer model with 64 units in the layer. There is a softmax layer for applying the classifications and the feature layer from earlier has also been added.

# In[ ]:


model = tf.keras.Sequential([
    feature_layer,
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(12, activation='softmax'),
])


# I'll also use a relatively simple gradient descent optimizer and use the multi class log loss method.

# In[ ]:


optimiser = tf.keras.optimizers.Ftrl(learning_rate=0.001)

model.compile(
    optimizer=optimiser, 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy'])


# Add some callbacks to reduce the learning rate or even completely stop training if the loss plateaus for too many steps.

# In[ ]:


callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=2, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1),
]


# Finally train the model for a few epochs. As the data isn't that large or complex I have only included a small number of steps as it doesn't take long for the model to converge.

# In[ ]:


train_log = model.fit(
    tf_train_data,
    validation_data=tf_val_data,
    epochs=100,
    callbacks=callbacks
)


# ## Evaluation
# 
# Although this challenge uses NCDG to evaluate the test dataset the MAP@5 (mean average precision) has a library that is easy to use so I'll use that to evaluate the validation dataset.
# 
# To calculate these metrics, begin by getting the probabilities per class in the validation dataset.

# In[ ]:


probabilities = model.predict(tf_val_data)


# Then get the top five destinations from these probablities.

# In[ ]:


predictions = np.argpartition(-probabilities, 4, axis=1)[:,0:5:]


# To calculate mean average precision the labels need to be formatted so that every label is in an array of its own e.g. 
# 
# [[label1], [label2], [label3] ...] 
# 
# This code puts the labels in this format.

# In[ ]:


formatted_val_labels = np.array([[label] for label in val_labels])


# Then calculate the mean average precision.

# In[ ]:


metrics.mapk(formatted_val_labels.astype(str), predictions.astype(str), 5)


# ## Submission
# 
# The final thing to do is to have the model generate recommendations for the users in the test set and write them to file. Begin by loading the test dataset.

# In[ ]:


test_users = pd.read_csv(
    '/kaggle/input/airbnb-recruiting-new-user-bookings/test_users.csv.zip',
    parse_dates = ['timestamp_first_active', 'date_account_created']
)


# Then pre-process the data.

# In[ ]:


test_users['age'] = test_users['age'].fillna(train_users['age'].mean())
test_users.loc[test_users['age'] > 100, 'age'] = train_users['age'].mean()

test_users['dac_month'] = test_users['date_account_created'].dt.month
test_users['tfa_month'] = test_users['timestamp_first_active'].dt.month

test_users = test_users.drop(columns=['timestamp_first_active', 'date_account_created', 'date_first_booking', 'first_affiliate_tracked'])

test_users = test_users.merge(action_counts, how='left', on=['id']).fillna(0)


# and load it into a Tensorflow dataset.

# In[ ]:


tf_test_data = tf.data.Dataset.from_tensor_slices(
    (dict(test_users))
)

tf_test_data = tf_test_data.batch(len(test_users))


# Then make predictions on the set and get the five top destinations.

# In[ ]:


probabilities = model.predict(tf_test_data)
predictions = np.argpartition(-probabilities, 4, axis=1)[:,0:5:]


# Write the recommendations to a dataset and melt the dataset so that there are five rows per user with a recommended country next to it.

# In[ ]:


submission = pd.DataFrame(data=predictions, index=test_users['id'])

submission = pd.melt(
    submission.reset_index(), 
    id_vars=['id'], 
    value_vars=[0, 1, 2, 3, 4],
    value_name='country'
)


# Ensure the recommendations are sorted by most probable recommendation.

# In[ ]:


submission = submission.sort_values(by=['id', 'variable'])


# Set the user id column as the index and drop the extra variable column (this held the order of the recommendations).

# In[ ]:


submission = submission.set_index('id')
submission = submission.drop(columns=['variable'])


# The recommendationed destinations are still numeric indexes. Replace them with the string names of the destinations.

# In[ ]:


submission['country'] = submission['country'].replace(categories)


# Finally write the recommendations to file.

# In[ ]:


submission.to_csv('submisison.csv')

