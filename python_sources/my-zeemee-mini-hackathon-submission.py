#!/usr/bin/env python
# coding: utf-8

# Thanks for taking the time to review my submission. 
# 
# The following is a copy of the notebook which I submitted to the ZeeMee Mini-Hackathon competition, The model had an accuracy of .936 on the test data and a score of .949 on the training data. The model itself had gotten third place for accuracy, however it did not rank for the analysis, because I ran out of time. I have cleaned up some of the cells but have otherwise left the code intact. My notes will seem a bit critical of my entrys shortcomings, this is because I am looking specifically for areas of improvement, the success of my model wasn't lost on me.
# 
# 
# The purpose of this is mostly to demonstrate my thought process when aproaching these sorts of challenges.
# 
# I chose a random forest model to predict the outcomes for two simple reasons. The first is that I have a fair bit of practice with random forests and would be able to troubleshoot any issues that arose and refine the model to get the best outcome. The second reason is the time limit, since I have 6 hours to work on this I wanted to make sure I was able to deliver a complete final product.
# 
# If the competition were to go for longer I would be tempted to use either a nueral net, as those can produce increadible accuracy if you train it right, or a support vector machine since I've wanted to get some practice with one of those to broaden my skillset. However, in a production enviorment there are two other models I would have perfered for use: a decision tree or a perceptron model. Both the decision tree and the perceptron model have one perticular area that they work very well in: explainability. Both models could be deconstructed for answer further questions, instead of determining what the likelyhood of a perticular student enrolling in college, why not find what perticular area of study would prove the greatest impact of their success (ex: 'by taking on one more extra corricular activity you increase the likelihood of being excepted to school by 12%').

# 
# The cell below was copied directly from the MatrixDS page for the event

# ZeeMee Mini-Hackathon by MatrixDS
# 
# Get started by fork lifting this project! (Green button in upper right). Then just build an R or Python tool. Make sure if you are on a team, just use one project and add your other team members!
# 
# ZeeMee is a fast growing silicon valley startup that has a social network for high school students looking for colleges. Students use the ZeeMee platform, through Android and iOS apps, to connect with others who are interested in the same colleges.
# 
# The goal of this competition is to use data collected about the students behavior on the zeemee platform to predict if they will enroll in a specific college. The dataset contains 19 features which are described here.
# 
#     college: The college of interest that a particular student is following
#     public_profie_enabled: If the student has made their zeemee profile public
#     going: If the student has stated (in a non-binding way) on the zeemee app that they are going to the college
#     interested: If the student has stated they are interested in the college on the zeemee app
#     start_term: Which term the student is projected to begin class
#     cohort_year: Which year the student is projected to begin class
#     created_by_csv: If the students zeemee account associated to the college as part of a batch upload
#     last_login: Number of days since the last login
#     schools_followed: Number of schools followed on the zeemee platform
#     high_school: Which high school the student attends
#     transfer_status: If the student is transferring from another college
#     roomate_match_quiz: If the student filled out a ZeeMee provided quiz to match with a roomate at the college of interest
#     chat_messages_sent: Number of messages sent
#     chat_viewed: Number of chats viewed
#     videos_liked: Number of videos liked
#     videos_viewed: Number of videos viewed
#     videos_viewed_unique: Number of unique videos viewed
#     offical_videos: Number of videos produced by the college of interest
#     engaged: If the student is engaged with the college on the zeemee app
#     final_funnel_stage: What stage in the enrolment process did the student end
# 
# The goal of your model is to predict final_funnel_stage. The funnel is a series of steps that a student moves through on their way to actually showing up to class. The stages are thought of in the following progression:
# 
#     Inquired: Expressed interest in the college on the zeemee app
#     Applied: Filled out some part of an application from the college
#     Application_Complete: Completed an application from the college
#     Accepted: Accepted by the college
#     Deposited: Paid a deposit to the college
#     Enrolled: Enrolled in class at the college
# 
# The prediction of interest for this competition is to focus on identifying students that enroll (funnel stage Enrolled or Deposited). This is a binary prediction, either the student does or does not enroll. Use the two csv files in the data folder to build your model. There is a training data file and a test data file.
# 
# ----
# 
# Grading
# 
# You will be graded on two areas each consisting of a possible 5 points.
# 
# First the average accuracy of your predictions on the testing data. Average accuracy is defined as the arithmetic average of accuracy for both classes of enrolled students and non-enrolled students. Predictions should be added to the zeemee_test.csv file. The best accuracy will receive all 5 points and other submissions will receive a relative portion of points to the best performer.
# 
# Second you must submit an exploration of important features for your prediction model. The explanation of features and feature engineering will receive a subjective score from a panel from ZeeMee and MatrixDS. This score will be out of a possible 5 points. Submissions can be in the format of a notebook or rmarkdown file.
# 
# Total scores for the two areas will be added and the individual and teams with the highest talley will receive the cash and interview prizes.
# 
# ----
# 
# Submission. You will submit your solution using a public MatrixDS project at the end of the hackathon. We will grade the predictions that you append to the zeemee_test.csv file and any feature explaining documents in the analysis folder in your public project. Please represent your predictions as a numeric/binary value 0 (wont enroll) or 1 (will enroll) in the submission file. The submission form will be provided on the day of the competition.

# Part 1:
# Exploritory Data Analysis

# In[ ]:


import pandas as pd

train_data = pd.read_csv("../input/zeemee-micro-competition-data/zeemee_train.csv")
test_data = pd.read_csv("../input/zeemee-micro-competition-data/zeemee_test.csv")


# In[ ]:


train_data.columns


# In[ ]:


train_data.describe()


# In[ ]:


test_data.describe()


# In[ ]:


train_data.head()


# In[ ]:


train_data.shape


# In[ ]:


## This section is so that I can view the types of data being used for each catagory and look for missing values
## This was also useful for figuring out which values were catagorical and which were ordinal
##some sections have been commented out as the output is lengthy and not frequently useful to view


print("college: ")
display(train_data.college.unique())
print("\n\npublic profile enabled: ")
display(train_data.public_profile_enabled.unique())
print("\n\ngoing: ")
display(train_data.going.unique())
print("\n\ninterested: ")
display(train_data.interested.unique())
print("\n\nstart term: ")
display(train_data.start_term.unique())
print("\n\ncohort year: ")
display(train_data.cohort_year.unique())
print("\n\ncreated by csv: ")
display(train_data.created_by_csv.unique())
#print("\n\nlast login: ") 
#display(train_data.last_login.unique()) ##found Nan here
print("\n\nschools followed: ")
display(train_data.schools_followed.unique())
print("\n\nhigh school: ")
display(train_data.high_school.unique())
print("\n\ntransfer status: ")
display(train_data.transfer_status.unique())
print("\n\nroommate match quiz: ")
display(train_data.roommate_match_quiz.unique())
#print("\n\nchat messages sent: ")
#display(train_data.chat_messages_sent.unique())
#print("\n\nchat viewed: ")
#display(train_data.chat_viewed.unique())
#print("\n\nvideos liked: ")
#display(train_data.videos_liked.unique())
#print("\n\nvideos viewed: ")
#display(train_data.videos_viewed.unique())
#print("\n\nvideos veiwed unique: ")
#display(train_data.videos_viewed_unique.unique())
#print("\n\ntotal official videos: ")
#display(train_data.total_official_videos.unique())
print("\n\nengaged: ")
display(train_data.engaged.unique())
print("\n\nfinal funnel stage: ")
display(train_data.final_funnel_stage.unique())


# In[ ]:


display(test_data.cohort_year.unique())
display(test_data.college.unique())
display(test_data.start_term.unique())


# So we can see the test data doesn't have entrys from 2017 (which the training set has), I'm curious how much the data will be affected by seasonality. More than 75% of the entries are from 2019 in both sets either way.
# 
# If there is more time I might want to build a different model for each year to see if that will prevent overfitting.

# In[ ]:


## train_data = train_data[train_data.cohort_year != 2017]
## this line lost accuracy in the model, interesting hypothesis though


# In[ ]:


print('is null in training data: ')
display(train_data.isnull().sum())
print('\n\nis null in test data: ')
display(test_data.isnull().sum())


# *As one quick note:
# the version of the test csv was overritten at the end of the competition so the test set no longer has missing values, it also has answers in it (which are actually my predictions). The part below was written during the competition*
# 
# Odd that there are missing values but that they are so rare. I'm curious if those three entry's are outliers in the dataset or just the result of a bug.
# I had experimented with changing the missing values to 0, mean, and 1600 (max value), in the end using 0 had the most positive effect on the accuracy of the model (It's also strange that 2 missing values would even have a measurable impact)

# In[ ]:


train_data['last_login'] = train_data['last_login'].fillna(0)
## I tried running the model by filling the null entry with 0 and then again with 1500
## the assumption was that the null entry was either from the time being too long 
## to record or the entry being null because the user was loging on at the time of capture
test_data['last_login'] = test_data['last_login'].fillna(0)


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

print('Setup Complete')


# In[ ]:


def final_funnel_num(val):
    if val == 'Inquired':
        return 0
    
    if val == 'Applied':
        return 0
    
    if val == 'Accepted':
        return 0
    
    if val == 'Deposited':
        return 1
    
    if val == 'Application_Complete':
        return 0
    
    if val == 'Enrolled':
        return 1
    
    else:
        return 'error'
    
## at this point in the process you may be wondering why I'm not using one of the in built encoding methods in sklearn over making my own
## When dealing with ordinal values (values that can be converted to integers and still hold meaning) it is benificial to hand encode values to ensure your model is not loosing some of the meaning in the data

## Also worth pointing out here, initially I had thought i needed to predict all 6 outcomes for the model, when I double checked the rules I discovered that the model was supposed to predict a binary outcome of either or for a subset of the outcomes (Enrolled and Deposited)
## which is why this function seems a bit silly, about an hour and a half before the deadline i discovered this error and went for the simplest fix.
## interestingly enough when I was predicting 6 outcomes the model was 54% accurate in testing (17% is the accuracy that dice would have had)
## I'm also curious how it would have effected accuracy if i had the algorithm predict all six outcomes and then translated that result to the desired binary output


# In[ ]:


def num_bool(val):
    if val == True:
        return 1
    elif val == False:
        return 0
    else:
        return 'error'
## The return code 'error' would actually be an error when the data is fed into the model
## as a random forrest cannot handle values that are non numeric


# In[ ]:


def goin_num(val):
    if val == 'undecided':
        return 0
    if val == 'going':
        return 1
    if val == 'notgoing':
        return -1
    else:
        return 'error'


# In[ ]:


def col_num(val):
    if val == 'college1':
        return 1
    if val == 'college2':
        return 2
    if val == 'college4':
        return 4
    if val == 'college3':
        return 3
    if val == 'college5':
        return 5
    if val == 'college6':
        return 6
    if val == 'college7':
        return 7
    if val == 'college8':
        return 8
    else:
        return 'error'


# In[ ]:


def term_num(val):
    if val == 'fall':
        return 1
    if val == 'spring':
        return 2
    if val == 'summer':
        return 3
    else:
        return 'error'


# In[ ]:


## since I'm using a random forest model for this project I will be converting all values I wish to use to numbers
## most likely i will have to drop some of these values for my final project, but its nice to have options
train_data['funnel_num'] = pd.Series([final_funnel_num(x) for x in train_data.final_funnel_stage], index=train_data.index)
train_data['transfer_status_num'] = pd.Series([num_bool(x) for x in train_data.transfer_status], index=train_data.index)
train_data['public_profile_enabled_num'] = pd.Series([num_bool(x) for x in train_data.public_profile_enabled], index=train_data.index)
train_data['interested_num'] = pd.Series([num_bool(x) for x in train_data.interested], index=train_data.index)
train_data['created_by_csv_num'] = pd.Series([num_bool(x) for x in train_data.created_by_csv], index=train_data.index)
train_data['roommate_match_quiz_num'] = pd.Series([num_bool(x) for x in train_data.roommate_match_quiz], index=train_data.index)
train_data['going_num'] = pd.Series([goin_num(x) for x in train_data.going], index=train_data.index)
train_data['college_num'] = pd.Series([col_num(x) for x in train_data.college], index=train_data.index)
train_data['start_term_num'] = pd.Series([term_num(x) for x in train_data.start_term], index=train_data.index)
## Also worth mentioning: isn't it wonderful not having to scale features for a random forest model


# In[ ]:


train_data.describe()


# Part -1:
# Drop everything and focus on finishing one successful model as fast a possible entry! oh god how is it half way to time already.
# 
# By this point in the competition I realized i hadn't be executing with the right sence of urgency and needed to rush towards a solution else risk having nothing to present.

# In[ ]:


y = train_data.funnel_num
rf_features = ['cohort_year',  'going_num', 'chat_messages_sent', 'schools_followed', 'videos_liked',  'chat_viewed', 'total_official_videos',  'videos_viewed','transfer_status_num',  'videos_viewed_unique', 
               'public_profile_enabled_num', 'created_by_csv_num', 'interested_num', 'roommate_match_quiz_num', 'college_num']
X = train_data[rf_features]

## what isn't pictured here is the trial and error as I remove various components, and reran the next 2 cells to compare accuracy.
## This process also taught me something new, the order the features are in affects the model created

## As a worthwhile point to mention, I had initially guessed that using label encoding for the college would have confused a random forrest and that i was going to need to retry using one hot encoding or a sparse vector instead, 
## however using label encoding did have a positive end result (assuming I didn't overfit the model)

## start_term_num and last_login was removed as it did not improve the accuracy of the model

## Other notes:  roommate_match_quiz_num and public_profile_enabled_num had an almost insignifigant impact on accuracy, I chose to leave them in because the impact was positive
## however if this were a production enviorment i might drop it for computational efficiency


train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)


# In[ ]:


forest_model = RandomForestRegressor(random_state=0)
forest_model.fit(train_X, train_y)
td_preds = forest_model.predict(val_X)
print(mean_squared_error(val_y, td_preds))


# In[ ]:


td_preds = td_preds.round()


# In[ ]:


print(td_preds.max(), td_preds.min())
## Checking to make sure there are no nonsensicle outputs


# In[ ]:


print(mean_squared_error(val_y, td_preds))


# In[ ]:


## this cell I added after the competition. Since the data sample has a split of about 1 positive case in 10
## I wanted to compare other evaluation methods
from sklearn.metrics import roc_auc_score

print(roc_auc_score(val_y, td_preds))


# In[ ]:


full_preds = forest_model.predict(train_data[rf_features])

train_data['preds'] = full_preds.round()


# In[ ]:


train_data.head(20)


# In[ ]:


y = 0
x = 0
for index, row in train_data.iterrows():
#    print(row['funnel_num'])
    if row['funnel_num'] == row['preds']:
        x = x + 1
    else:
        y = y + 1
print("Percent accuracy of model is: ", (x/(x+y))*100)


# In[ ]:


y = 0
x = 0
n = 0
q = 0
for index, row in train_data.iterrows():
#    print(row['funnel_num'])
    if row['funnel_num'] == 1:
        x = x + 1
    else:
        y = y + 1
    if row['preds'] == 1:
        n = n + 1
    else:
        q = q + 1
        
print('Percentage of positives in sample: ', (x/(x+y))*100, '\nPercentage of predicted positives in sample: ', (n/(n+q))*100)


# Now that we have an idea of the accuracy of the model (and that its better than throwing darts at a board), let's wrap this hot mess up into the test set

# In[ ]:


test_data['transfer_status_num'] = pd.Series([num_bool(x) for x in test_data.transfer_status], index=test_data.index)
test_data['public_profile_enabled_num'] = pd.Series([num_bool(x) for x in test_data.public_profile_enabled], index=test_data.index)
test_data['interested_num'] = pd.Series([num_bool(x) for x in test_data.interested], index=test_data.index)
test_data['created_by_csv_num'] = pd.Series([num_bool(x) for x in test_data.created_by_csv], index=test_data.index)
test_data['roommate_match_quiz_num'] = pd.Series([num_bool(x) for x in test_data.roommate_match_quiz], index=test_data.index)
test_data['going_num'] = pd.Series([goin_num(x) for x in test_data.going], index=test_data.index)
test_data['college_num'] = pd.Series([col_num(x) for x in test_data.college], index=test_data.index)


# In[ ]:


test_preds = forest_model.predict(test_data[rf_features])
print('values should be 0  and 1.0: ', test_preds.min(), test_preds.max())
test_data['preds'] = test_preds.round()
test_data.head(20)


# In[ ]:


#def funnel_final(val):    
#    if val == 1:
#        return 'Inquired'
#    
#    if val == 2:
#        return 'Applied'
#    
#    if val == 3:
#        return 'Accepted'
#    
#    if val == 4:
#        return 'Deposited'
#    
#    if val == 5:
#        return 'Application_Complete'
#    
#    if val == 6:
#        return 'Enrolled'


# In[ ]:


## Now to bring this hot mess full circle
test_data['final_funnel_stage'] = pd.Series([x for x in test_data.preds], index=test_data.index)


# In[ ]:


test_data.head()


# In[ ]:


test_data = test_data[['college', 'public_profile_enabled', 'going', 'interested',
       'start_term', 'cohort_year', 'created_by_csv', 'last_login',
       'schools_followed', 'high_school', 'transfer_status',
       'roommate_match_quiz', 'chat_messages_sent', 'chat_viewed',
       'videos_liked', 'videos_viewed', 'videos_viewed_unique',
       'total_official_videos', 'engaged', 'final_funnel_stage']]
print(test_data.shape) ## checking to make sure we're back to 20 columns and that i didn't screw something up


# In[ ]:


#test_data.to_csv("zeemee_test_output.csv")
#output file for the copetition


# As you can see the greatest failing of the project was that I did not ration my time effectively, I didn't finish my EDA before I jumped into the model and had to use trial and error in its place. I had optimistically imported seaborn and matplotlib, but failed to use them. Especially with the encoding methods I used I missed out on a golden opportunity to review the differences in the training and test dataset.
# 
# Secondly with testing, I did not check for overfitting and simply submitted and crossed my fingers (as a result of running out of time). In hindsight I know that the model hadn't overfit and was similairly effective in both the training and test data.
# 
# Third I would have liked to capture false positive, and fale negitive rate for the model as well [use a confusion matrix] as opposed to just capturing MAE and percentage (these at least provided quick feedback to adjust the various itterations of the model). Ideally in a production enviorment I would have liked to tweek the model so that it had very little false negitives even if it meant that the model became too optimistic (loosing just a couple students would outweigh the benifits of cutting advertising to the one who wouldn't be attainable).
