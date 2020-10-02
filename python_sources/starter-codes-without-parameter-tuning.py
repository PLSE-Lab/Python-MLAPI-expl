#!/usr/bin/env python
# coding: utf-8

# Here is my starter codes, mainly dealing with feature engineering and without much parameter tuning. Current LB is around 0.485. 
# 
# The features I used here include the statistics on event codes, time spent on each event/type/world, as well as the accuracy information of the previous plays. 
# 
# One thing I want to model is that I want to define the "session" of each play. This "session" is not the same concept as the game_session. For example, a player might use this app from 4pm to 6pm, he might take many game sessions during this period but I want to define 4pm to 6pm as one session of his behavior. I think the reasonability is that if he has already played for a long time, even if he is talented on assessments, he is also very likely to quite and has a label '0'.
# 
# I currently use XGBoost Regressor and only use the default settings. If there are any suggestions for my codes, I will be very glad to hear for your advice!

# In[ ]:


import numpy as np
import pandas as pd

pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
pd.options.display.max_colwidth = 500


# ### 1. Read the data.

# In[ ]:


train = pd.read_csv("/kaggle/input/data-science-bowl-2019/train.csv")
test = pd.read_csv("/kaggle/input/data-science-bowl-2019/test.csv")
train_labels = pd.read_csv("/kaggle/input/data-science-bowl-2019/train_labels.csv")
sample_submission = pd.read_csv("/kaggle/input/data-science-bowl-2019/sample_submission.csv")
specs = pd.read_csv("/kaggle/input/data-science-bowl-2019/specs.csv")


# In[ ]:


train.timestamp = pd.to_datetime(train.timestamp)
test.timestamp = pd.to_datetime(test.timestamp)


# In[ ]:


# Remove all users that haven't taken any assessments.
assessment_user = train[train.type == 'Assessment'].installation_id.unique()
train = train[train.installation_id.isin(assessment_user)]
train.shape


# ### 2. Train Labels Verification.

# In[ ]:


# This is not that relevant to predicting. The code here is only used to help to make sure my understanding of how to calculate lables is consistant with the competition holders!
def calculate_accuracy(row):
    # print(row.game_session)
    game_records = train[train.game_session == row.game_session]
    if row.title == 'Bird Measurer (Assessment)':
        attempts_df = game_records[game_records.event_code==4110]
    else:
        attempts_df = game_records[game_records.event_code==4100]
    attempts_df['is_correct'] = attempts_df.event_data.str.contains('"correct":true')
    num_correct = np.sum(attempts_df['is_correct'])
    num_incorrect = attempts_df.shape[0] - num_correct
    if num_correct == 0:
        accuracy_group = 0
    elif num_incorrect == 0:
        accuracy_group = 3
    elif num_incorrect == 1:
        accuracy_group = 2
    else:
        accuracy_group = 1
    return pd.Series([num_correct, num_incorrect, accuracy_group])
        
random_train_labels = train_labels.sample(n=10, random_state=1)
random_train_labels[['my_correct', 'my_incorrect', 'my_accuracy_group']] = random_train_labels.apply(calculate_accuracy, axis=1)
random_train_labels['correct_difference'] = random_train_labels.num_correct - random_train_labels.my_correct
random_train_labels['group_diffrence'] = random_train_labels.accuracy_group - random_train_labels.my_accuracy_group
# random_train_labels.head(500)


# ### 3. Test Label Construction.

# In[ ]:


# Only the last assessment in the test set should be used to do the prediction. 
test.timestamp = pd.to_datetime(test.timestamp)
test_labels = test[['game_session', 'installation_id', 'title', 'timestamp']].drop_duplicates(subset='installation_id', keep='last')


# ### 4. Feature Calculation.

# #### 4.0 Preparation

# In[ ]:


# Use train_labels to construct training set.
train_labels = train_labels.merge(train[['game_session', 'timestamp', 'installation_id']].drop_duplicates(                                         subset=['game_session', 'installation_id']), how='left',                                  on=['game_session', 'installation_id'])


# In[ ]:


clip_names = list(train[train.type=='Clip'].title.unique())
activity_names = list(train[train.type=='Activity'].title.unique())
game_names = list(train[train.type=='Game'].title.unique())
assessment_names = list(train[train.type=='Assessment'].title.unique())


# In[ ]:


event_codes_for_game = list(train[train.type=='Game'].event_code.unique())
event_codes_for_assessment = list(train[train.type=='Assessment'].event_code.unique())
event_codes_for_activity = list(train[train.type=='Activity'].event_code.unique())
event_codes_for_game_uni_assessment_uni_activity = list(set(event_codes_for_game).union(set(event_codes_for_assessment)).union(set(event_codes_for_activity)))


# In[ ]:


type_names = ['Game', 'Activity', 'Assessment', 'Clip']
world_names = ['NONE', 'MAGMAPEAK', 'TREETOPCITY', 'CRYSTALCAVES']


# #### 4.1 Feature Names: Games

# In[ ]:


game_event_code = train[train['type']=='Game'][['title', 'event_code']].drop_duplicates()
game_event_code.shape


# In[ ]:


game_features_by_name = list() # For each of the game, calculate the different counts for each event code for each game.
for index, row in game_event_code.iterrows():
    game_features_by_name.append(row.title + ' ' + str(row.event_code))
# game_features_by_name


# In[ ]:


effective_game_codes = [2000, 2020, 2030, 3010, 3020, 3021, 4020, 4070, 4090]
summary_game_features = ['Num Give Up', 'Num Total', 'Give Up Rate', 'Mean Accuracy', 'Std Accuracy', 'Max Accuracy', 'Min Accuracy']


# In[ ]:


game_feature_1_names = ['Total Count (Game) ' + str(code) for code in effective_game_codes]
game_feature_2_names = ['Overall Game ' + name for name in summary_game_features]
game_feature_3_names = ['Current Group Game' + name for name in summary_game_features]
game_feature_4_names = ['Last 3 Group Game' + name for name in summary_game_features]


# #### 4.2 Feature Names: Assessments

# In[ ]:


assessment_event_code = train[train['type']=='Assessment'][['title', 'event_code']].drop_duplicates()
assessment_event_code.shape


# In[ ]:


assessment_features_by_name = list() # For each of the game, calculate the different counts for each event code for each game.
for index, row in assessment_event_code.iterrows():
    assessment_features_by_name.append(row.title + ' ' + str(row.event_code))
# assessment_features_by_name


# In[ ]:


effective_assessment_codes = event_codes_for_assessment
summary_assessment_features = ['Num Give Up', 'Num Total', 'Give Up Rate', 'Mean Accuracy', 'Std Accuracy', 'Max Accuracy', 'Min Accuracy']

assessment_feature_1_names = ['Total Count (Ass) ' + str(code) for code in effective_assessment_codes]
assessment_feature_2_names = ['Overall Ass ' + name for name in summary_assessment_features]
assessment_feature_3_names = ['Current Group Ass ' + name for name in summary_assessment_features]
assessment_feature_4_names = ['Last 3 Group Ass ' + name for name in summary_assessment_features]


# #### 4.3 Feature Names: Activites

# In[ ]:


activity_event_code = train[train['type']=='Activity'][['title', 'event_code']].drop_duplicates()
activity_event_code.shape


# In[ ]:


activity_features_by_name = list() # For each of the game, calculate the different counts for each event code for each game.
for index, row in activity_event_code.iterrows():
    activity_features_by_name.append(row.title + ' ' + str(row.event_code))
# activity_features_by_name


# In[ ]:


effective_activity_codes = event_codes_for_activity

activity_feature_1_names = ['Total Count (Act) ' + str(code) for code in effective_activity_codes]


# #### 4.4 Feature Names: Clips

# In[ ]:


clip_length = {'Welcome to Lost Lagoon!': 19, 'Magma Peak - Level 1': 20, 
              'Slop Problem': 60, 'Tree Top City - Level 1': 17, 'Ordering Spheres': 61, 
              'Costume Box': 61, '12 Monkeys': 109, 'Tree Top City - Level 2': 25,
              "Pirate's Tale": 80, 'Treasure Map': 156, 'Tree Top City - Level 3': 26,
              'Rulers': 126, 'Magma Peak - Level 2': 22, 'Crystal Caves - Level 1': 18,
              'Balancing Act': 72, 'Crystal Caves - Level 2': 24, 'Crystal Caves - Level 3': 19,
              'Lifting Heavy Things': 118, 'Honey Cake':  142, 'Heavy, Heavier, Heaviest': 61}


# #### 4.5 Feature Names: Time

# In[ ]:


timeFeatureName_type = [name + ' Time Total' for name in type_names]
timeFeatureName_world = [name + ' Time Total' for name in world_names]
timeFeatureName_groupstat = ['Group Time Mean', 'Group Time Std', 'Group Time Max', 'Group Time Min', 'Current Elapsed']
timeFeatureName_title = [name + ' Time Total' for name in activity_names + assessment_names + game_names]
timeFeatureName_totaltime = ['Total Time']


# #### 4.6 All Feature Names

# In[ ]:


feature_names = ['Have Records'] + game_features_by_name + assessment_features_by_name + activity_features_by_name +                timeFeatureName_type + timeFeatureName_world + timeFeatureName_groupstat +                 timeFeatureName_title + timeFeatureName_totaltime + game_feature_1_names + game_feature_2_names + game_feature_3_names + game_feature_4_names +                assessment_feature_1_names + assessment_feature_2_names + assessment_feature_3_names + assessment_feature_4_names +                clip_names


# #### 4.3 Feature Calculation

# In[ ]:


# row = train_labels.iloc[3].squeeze()

def calculate_features(row):
    # print(row.game_session)
    # Step 0: Filter all relevant records.
    if row['isTrain'] == 1:
        all_user_records = train[train.installation_id == row.installation_id]
    else:
        all_user_records = test[test.installation_id == row.installation_id]

    all_records = all_user_records[all_user_records.timestamp < row.timestamp]
    
    if all_records.shape[0] > 0:

        # Step 1: Filter records according to types.
        game_records = all_records[all_records.type == 'Game']
        assessment_records = all_records[all_records.type == 'Assessment']
        activity_records = all_records[all_records.type == 'Activity']
        clip_records = all_records[all_records.type == 'Clip']

        # Step 2: Game Records and Assessment Records Processing.

        # Step 2.1: Features by Game name or Assessment name.
        game_feature_values_by_name = [0] * len(game_features_by_name)
        assessment_feature_values_by_name = [0] * len(assessment_features_by_name)
        activity_feature_values_by_name = [0] * len(activity_features_by_name)

        game_unstack_by_name = game_records.groupby(by=['title', 'event_code']).size().unstack().fillna(0)
        assessment_unstack_by_name = assessment_records.groupby(by=['title', 'event_code']).size().unstack().fillna(0)
        activity_unstack_by_name = activity_records.groupby(by=['title', 'event_code']).size().unstack().fillna(0)

        for idx, row_ in game_unstack_by_name.iterrows():
            for column in game_unstack_by_name.columns:
                try:
                    game_feature_values_by_name[game_features_by_name.index(idx + ' ' + str(column))] = row_[column]
                except:
                    continue

        for idx, row_ in assessment_unstack_by_name.iterrows():
            for column in assessment_unstack_by_name.columns:
                try:
                    assessment_feature_values_by_name[assessment_features_by_name.index(idx + ' ' + str(column))] = row_[column]
                except:
                    continue

        for idx, row_ in activity_unstack_by_name.iterrows():
            for column in activity_unstack_by_name.columns:
                try:
                    activity_feature_values_by_name[activity_features_by_name.index(idx + ' ' + str(column))] = row_[column]
                except:
                    continue

        # Step 2.2: Features based on the time sequence.

        # Step 2.2.1: Produce a summary table based on game session.
        all_records = all_records.sort_values(by='timestamp') # To guarantee that all the records are within the time sequence.
        all_session_id = all_records.game_session.unique()
        session_start_df = all_records[['game_session', 'timestamp', 'title', 'type', 'world']].drop_duplicates(subset=['game_session'], keep='first')
        session_end_df = all_records[['game_session', 'timestamp']].drop_duplicates(subset=['game_session'], keep='last')
        session_statistics = session_start_df.merge(session_end_df, on='game_session', how='left')

        def calculate_time_spent(row):
            if row['type'] == 'Clip':
                return clip_length[row['title']]
            else:
                return pd.Timedelta(row.timestamp_y - row.timestamp_x).seconds

        session_statistics['time_spent'] = session_statistics.apply(calculate_time_spent, axis=1)

        # Step 2.2.2: Divide the sessions into different groups.
        threshold = 600 # If the gap between two sessions are more than 10 min, then there are in diffrent groups.
        current_group = 1
        session_statistics['group'] = 0
        for idx, row_ in session_statistics.iterrows():
            if idx == 0:
                session_statistics.at[idx, 'group'] = 1
            else:
                if pd.Timedelta(row_['timestamp_x'] - session_statistics.iloc[idx-1]['timestamp_y']).seconds > threshold:
                    current_group += 1
                session_statistics.at[idx, 'group'] = current_group

        # Step 2.2.3: Event Code Summarization by game session.
        summarization_df = all_records.groupby(by=['game_session', 'event_code']).size().unstack()
        other_columns = list(set(event_codes_for_game_uni_assessment_uni_activity).difference(set(all_records.groupby(by=['game_session', 'event_code']).size().unstack().columns)))
        for col in other_columns:
            summarization_df[col] = np.NaN
        summarization_df.fillna(0, inplace=True)
        session_statistics = session_statistics.merge(summarization_df, on='game_session', how='left')

        def calculate_accuracy(row):
            if row.type == 'Game' or row.type == 'Assessment':
                if row.type == 'Game':
                    num_correct = row[3021]
                    num_incorrect = row[3020]
                else:
                    if row.title == 'Bird Measurer (Assessment)':
                        attempts_df = all_records[(all_records.game_session == row.game_session) & (all_records.event_code==4110)]
                    else:
                        attempts_df = all_records[(all_records.game_session == row.game_session) & (all_records.event_code==4100)]
                    attempts_df['is_correct'] = attempts_df.event_data.str.contains('"correct":true')
                    num_correct = np.sum(attempts_df['is_correct'])
                    num_incorrect = attempts_df.shape[0] - num_correct
                if num_correct + num_incorrect == 0:
                    accuracy = 0
                else:
                    accuracy = num_correct / (num_correct + num_incorrect)
            else:
                num_correct = np.NaN
                num_incorrect = np.NaN
                accuracy = np.NaN
            return pd.Series([num_correct, num_incorrect, accuracy])
        session_statistics[['num_correct', 'num_incorrect', 'accuracy']] = session_statistics.apply(calculate_accuracy, axis=1)   

        # Step 2.2.4: Features w.r.t Time.
        def my_squeeze(df, index_name):
            s = df.squeeze()
            if not isinstance(s, pd.Series):
                s = pd.Series([s], index=[df.index[0]])
                s = s.rename_axis('group')
            return s


        # Time Spent on each type.
        timeFeatureValue_type = [0] * len([name + ' Time Total' for name in type_names])
        time_type_Series = my_squeeze(session_statistics[['type', 'time_spent']].groupby('type').sum(), 'type')
        for idx in time_type_Series.index:
            timeFeatureValue_type[type_names.index(idx)] = time_type_Series.loc[idx]

        # Time Spent on each world.
        timeFeatureValue_world = [0] * len([name + ' Time Total' for name in world_names])
        time_world_Series = my_squeeze(session_statistics[['world', 'time_spent']].groupby('world').sum(), 'world')
        for idx in time_world_Series.index:
            timeFeatureValue_world[world_names.index(idx)] = time_world_Series.loc[idx]

        # Time Spent on each title. 
        timeFeatureValue_title = [0] * len(timeFeatureName_title)
        time_title_Series = my_squeeze(session_statistics[['title', 'time_spent']].groupby('title').sum(), 'title')
        for idx in time_title_Series.index:
            try:
                timeFeatureValue_title[timeFeatureName_title.index(idx+' Time Total')] = time_title_Series.loc[idx]
            except:
                pass

        # Time Spent on each group.
        is_label_new_group = False
        time_group_Series = my_squeeze(session_statistics[['group', 'time_spent']].groupby('group').sum(), 'group')


        if pd.Timedelta(row.timestamp - session_statistics.iloc[-1].timestamp_y).seconds > threshold:
            is_label_new_group = True # The assessment is based on a new group.
            current_group_time_spent = 0.00
            previous_groups = time_group_Series
        else:
            current_group_time_spent = pd.Timedelta(row.timestamp - session_statistics.iloc[-1].timestamp_y).seconds + time_group_Series.iloc[-1]
            previous_groups = time_group_Series.iloc[0:-1]

        timeFeatureValue_groupstat = [np.mean(previous_groups), np.std(previous_groups),                                      np.max(previous_groups), np.min(previous_groups),                                      current_group_time_spent]


        timeFeatureValue_totaltime = [np.sum(timeFeatureValue_world)]

        # Step 2.2.5: Features w.r.t Games.
        game_statistics = session_statistics[session_statistics.type=='Game']

        if game_statistics.shape[0] > 0:

            # Feature Category 1: Total count.
            game_feature_1_values = list(game_statistics[effective_game_codes].sum())

            # Feature Category 2: Overall Summary.
            def calculate_statistics(s):
                return np.mean(s), np.std(s), np.max(s), np.min(s)

            def calculate_ga_summary_features(df):
                num_give_up = df[df.accuracy==0].shape[0]
                give_up_rate = num_give_up / df.shape[0]
                accuracy_mean, accuracy_std, accuracy_max, accuracy_min = calculate_statistics(df.accuracy)
                # round_complete_rate = np.sum(game_statistics.is_all_round_complete) / game_statistics.shape[0]
                return [num_give_up, df.shape[0], give_up_rate, accuracy_mean, accuracy_std, accuracy_max, accuracy_min]

            game_feature_2_values = calculate_ga_summary_features(game_statistics)

            # Feature Category 3: Group Summary.
            if is_label_new_group or (game_statistics.iloc[-1].group != session_statistics.iloc[-1].group):
                game_feature_3_values = game_feature_2_values
            else:
                game_feature_3_values = calculate_ga_summary_features(game_statistics[game_statistics.group==game_statistics.iloc[-1].group])

            # Feature Category 4: Recent Group Summary.
            if game_statistics.iloc[-1].group >= 3:
                game_feature_4_values = calculate_ga_summary_features(game_statistics[game_statistics.group>=game_statistics.iloc[-1].group-2])
            else:
                game_feature_4_values = game_feature_2_values

        else:
            game_feature_1_values = [0] * len(game_feature_1_names)
            game_feature_2_values = [np.NaN] * len(game_feature_2_names)
            game_feature_3_values = [np.NaN] * len(game_feature_3_names)
            game_feature_4_values = [np.NaN] * len(game_feature_4_names)


        # Step 2.2.6: Features w.r.t Assessments.
        assessment_statistics = session_statistics[session_statistics.type=='Assessment']

        if assessment_statistics.shape[0] > 0:

            # Feature Category 1: Total count.
            assessment_feature_1_values = list(assessment_statistics[effective_assessment_codes].sum())

            # Feature Category 2: Overall Summary.
            def calculate_statistics(s):
                return np.mean(s), np.std(s), np.max(s), np.min(s)

            def calculate_ga_summary_features(df):
                num_give_up = df[df.accuracy==0].shape[0]
                give_up_rate = num_give_up / df.shape[0]
                accuracy_mean, accuracy_std, accuracy_max, accuracy_min = calculate_statistics(df.accuracy)
                # round_complete_rate = np.sum(game_statistics.is_all_round_complete) / game_statistics.shape[0]
                return [num_give_up, df.shape[0], give_up_rate, accuracy_mean, accuracy_std, accuracy_max, accuracy_min]

            assessment_feature_2_values = calculate_ga_summary_features(assessment_statistics)

            # Feature Category 3: Group Summary.
            if is_label_new_group or (assessment_statistics.iloc[-1].group != session_statistics.iloc[-1].group):
                assessment_feature_3_values = assessment_feature_2_values
            else:
                assessment_feature_3_values = calculate_ga_summary_features(assessment_statistics[assessment_statistics.group==assessment_statistics.iloc[-1].group])

            # Feature Category 4: Recent Group Summary.
            if assessment_statistics.iloc[-1].group >= 3:
                assessment_feature_4_values = calculate_ga_summary_features(assessment_statistics[assessment_statistics.group>=assessment_statistics.iloc[-1].group-2])
            else:
                assessment_feature_4_values = assessment_feature_2_values

        else:
            assessment_feature_1_values = [0] * len(assessment_feature_1_names)
            assessment_feature_2_values = [np.NaN] * len(assessment_feature_2_names)
            assessment_feature_3_values = [np.NaN] * len(assessment_feature_3_names)
            assessment_feature_4_values = [np.NaN] * len(assessment_feature_4_names)

        # Missing Values.
        if np.sum(np.isnan(game_feature_2_values)) > 0 or np.sum(np.isnan(assessment_feature_2_values)): 
            if np.sum(np.isnan(game_feature_2_values)) == 0:
                assessment_feature_2_values = game_feature_2_values
                assessment_feature_3_values = game_feature_3_values
                assessment_feature_4_values = game_feature_4_values
            elif np.sum(np.isnan(assessment_feature_2_values)) == 0:
                game_feature_2_values = assessment_feature_2_values
                game_feature_3_values = assessment_feature_3_values
                game_feature_4_values = assessment_feature_4_values
            else:
                pass


        # Step 2.2.7: Features w.r.t Activities.
        activity_statistics = session_statistics[session_statistics.type=='Activity']
        if activity_statistics.shape[0] > 0:

            # Feature Category 1: Total count.
            activity_feature_1_values = list(activity_statistics[effective_activity_codes].sum())

        else:
            activity_feature_1_values = [0] * len(activity_feature_1_names)


        # Step 2.2.8: Features w.r.t Clips.
        clip_statistics = session_statistics[session_statistics.type=='Clip']
        total_time = 0
        clip_feature_values = [0] * len(clip_names)
        for idx, row_ in clip_statistics.iterrows():
            clip_feature_values[clip_names.index(row_.title)] += 1

        feature_values =  [1] + game_feature_values_by_name + assessment_feature_values_by_name + activity_feature_values_by_name +                    timeFeatureValue_type + timeFeatureValue_world + timeFeatureValue_groupstat +                     timeFeatureValue_title + timeFeatureValue_totaltime + game_feature_1_values + game_feature_2_values + game_feature_3_values + game_feature_4_values +                    assessment_feature_1_values + assessment_feature_2_values + assessment_feature_3_values + assessment_feature_4_values +                    clip_feature_values

        return pd.Series(feature_values)
    
    else:
        return pd.Series([0] * len(feature_names))


# In[ ]:


np.random.seed(0)
idx = np.random.permutation(np.arange(len(train_labels)))
train_labels_subset = train_labels.iloc[idx].drop_duplicates(subset=['installation_id'])


# In[ ]:


train_labels_subset['isTrain'] = 1
test_labels['isTrain'] = 0
train_labels['isTrain'] = 1


# In[ ]:


train_labels_subset[feature_names] = train_labels_subset.apply(calculate_features, axis=1)


# In[ ]:


test_labels[feature_names] = test_labels.apply(calculate_features, axis=1) 


# In[ ]:


def add_more_features(labels_dataset):
    labels_dataset['Mushroom'] = np.where(labels_dataset.title=='Mushroom Sorter (Assessment)', 1, 0)
    labels_dataset['Bird Measurer'] = np.where(labels_dataset.title=='Bird Measurer (Assessment)', 1, 0)
    labels_dataset['Cauldron'] = np.where(labels_dataset.title=='Cauldron Filler (Assessment)', 1, 0)
    labels_dataset['Cart'] = np.where(labels_dataset.title=='Cart Balancer (Assessment)', 1, 0)
    labels_dataset['Chest'] = np.where(labels_dataset.title=='Chest Sorter (Assessment)', 1, 0)
    labels_dataset['TreeTopCity'] = labels_dataset['Mushroom'] + labels_dataset['Bird Measurer']
    labels_dataset['CrystalCaves'] = labels_dataset['Cart'] + labels_dataset['Chest']
    return labels_dataset


# In[ ]:


train_labels_subset = add_more_features(train_labels_subset)


# In[ ]:


test_labels = add_more_features(test_labels)


# In[ ]:


training_data = train_labels_subset.drop(['game_session', 'title', 'installation_id', 'num_correct', 'num_incorrect', 'accuracy', 'timestamp', 'isTrain'], axis=1)


# In[ ]:


testing_data = test_labels.drop(['game_session', 'title', 'installation_id', 'timestamp', 'isTrain'], axis=1)


# ### 5. Training Models

# In[ ]:


import itertools
import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Oranges):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, color='w')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[ ]:


from sklearn.model_selection import train_test_split

y = training_data['accuracy_group']
X = training_data.drop(['accuracy_group'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, cohen_kappa_score
import itertools
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neural_network import MLPClassifier


clf = XGBRegressor()

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# In[ ]:


value_counts = y_train.value_counts()
q1, q2, q3 = pd.Series(y_pred).quantile([value_counts.loc[0]/np.sum(value_counts), (value_counts.loc[0]                             +value_counts.loc[1])/np.sum(value_counts),                              (value_counts.loc[0]+value_counts.loc[1]+value_counts.loc[2])/np.sum(value_counts)])


# In[ ]:


y_final = np.where(y_pred<q1, 0, np.where(y_pred<q2,1,np.where(y_pred<q3,2,3)))


# In[ ]:


plot_confusion_matrix(confusion_matrix(y_test, y_final), classes=['0', '1', '2', '3'])


# In[ ]:


cohen_kappa_score(y_test, y_final, weights='quadratic')


# In[ ]:


y_pred = clf.predict(testing_data)

q1, q2, q3 = pd.Series(y_pred).quantile([value_counts.loc[0]/np.sum(value_counts), (value_counts.loc[0]                             +value_counts.loc[1])/np.sum(value_counts),                              (value_counts.loc[0]+value_counts.loc[1]+value_counts.loc[2])/np.sum(value_counts)])

def find_accuracy_group(x):
    if x < q1:
        return 0
    elif x < q2:
        return 1
    elif x < q3:
        return 2
    else:
        return 3

test_labels['accuracy'] = y_pred
test_labels['accuracy_group'] = test_labels['accuracy'].apply(find_accuracy_group)
output = test_labels[['installation_id', 'accuracy_group']]
output.to_csv("submission.csv", index=False)

