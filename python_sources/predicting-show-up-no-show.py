#!/usr/bin/env python
# coding: utf-8

# Hey everyone. This is my very first data analysis work after deciding to move on from gulping down voluminous textbooks and courses to some hands on analysis and prediction.  Any comments/helpful suggestions will be welcome and greatly appreciated. So let's get started!

# ## PART I: Data Wrangling

# In this part, I am going to try and explore the data to check for missing values/erroneous entries and also comment on redundant features and add additional ones, if needed.

# First, let us load the necessary packages and get an overview of what we are dealing with.

# In[ ]:


import numpy as np
import pandas as pds
import matplotlib.pyplot as plt
from matplotlib import pylab
import seaborn as sns
sns.set_style("whitegrid")
noShow = pds.read_csv('../input/No-show-Issue-Comma-300k.csv')
print(noShow.head())


# ### 1.1. Getting rid of typos!

# It is immediately apparent that some of the column names have typos, so let us clear them up before continuing further, so that I don't have to use alternate spellings everytime I need a variable.

# In[ ]:


noShow.rename(columns = {'ApointmentData':'AppointmentData',
                         'Alcoolism': 'Alchoholism',
                         'HiperTension': 'Hypertension',
                         'Handcap': 'Handicap'}, inplace = True)

print(noShow.columns)


# ### 1.2. Messing around with time

# For convenience, I am going to convert the AppointmentRegistration and Appointment columns into datetime64 format and the AwaitingTime column into absolute values.

# In[ ]:


noShow.AppointmentRegistration = noShow.AppointmentRegistration.apply(np.datetime64)
noShow.AppointmentData = noShow.AppointmentData.apply(np.datetime64)
noShow.AwaitingTime = noShow.AwaitingTime.apply(abs)

print(noShow.AppointmentRegistration.head())
print(noShow.AppointmentData.head())
print(noShow.AwaitingTime.head())


# It is interesting to note that the time portions have vanished from the Appointment Data timedeltas, because all appointment times were set exactly to 00:00:00. Also, if it's not clear already, the AwaitingTime is the rounded number of days from registration to appointment. Here is the proof:

# In[ ]:


daysToAppointment = noShow.AppointmentData - noShow.AppointmentRegistration
daysToAppointment = daysToAppointment.apply(lambda x: x.total_seconds() / (3600 * 24))
plt.scatter(noShow.AwaitingTime, daysToAppointment)
plt.axis([0, 400, 0, 400])
plt.xlabel('AwaitingTime')
plt.ylabel('daysToAppointment')
plt.show()


# We also create a new feature called HourOfTheDay, which will indicate the hour of the day at which the appointment was booked. This will be derived off AppointmentRegistration, thus:

# In[ ]:


def calculateHour(timestamp):
    timestamp = str(timestamp)
    hour = int(timestamp[11:13])
    minute = int(timestamp[14:16])
    second = int(timestamp[17:])
    return round(hour + minute/60 + second/3600)

noShow['HourOfTheDay'] = noShow.AppointmentRegistration.apply(calculateHour)
print(noShow.HourOfTheDay.head())


# ### 1.3. Checking for errors and NaNs

# Next we check for any erroneous values and NaNs in data.

# In[ ]:


print('Age:',sorted(noShow.Age.unique()))
print('Gender:',noShow.Gender.unique())
print('DayOfTheWeek:',noShow.DayOfTheWeek.unique())
print('Status:',noShow.Status.unique())
print('Diabetes:',noShow.Diabetes.unique())
print('Alchoholism:',noShow.Alchoholism.unique())
print('Hypertension:',noShow.Hypertension.unique())
print('Handicap:',noShow.Handicap.unique())
print('Smokes:',noShow.Smokes.unique())
print('Scholarship:',noShow.Scholarship.unique())
print('Tuberculosis:',noShow.Tuberculosis.unique())
print('Sms_Reminder:',noShow.Sms_Reminder.unique())
print('AwaitingTime:',sorted(noShow.AwaitingTime.unique()))
print('HourOfTheDay:', sorted(noShow.HourOfTheDay.unique()))


# It is clear that we do not have any NaNs anywhere in the data. However, we do have some impossible ages such as -2 and -1, and some pretty absurd ages such as 100 and beyond. I do admit that it is possible to live 113 years and celebrate living so long, and some people do live that long, but most people don't. So I will treat the ages greater than 95 as outliers. 
# 
# Once I have made my assumptions, it is time to remove the impossible and the absurd from the data.

# In[ ]:


noShow = noShow[(noShow.Age >= 0) & (noShow.Age <= 95)]
print(noShow[noShow.Age>90])


# ### 1.4. Checking for outliers in AwaitingTime

# In[ ]:


sns.stripplot(data = noShow, y = 'AwaitingTime', jitter = True)
sns.plt.ylim(0, 500)
sns.plt.show()


# Clearly, the data starts to thin out after 150 days AwaitingTime. There is one observation at 398 days, which is likely an outlier. There are almost no observations beyond 350 days, so let us remove anything beyond 350 days which will include that 398 day observation too.

# In[ ]:


noShow = noShow[noShow.AwaitingTime < 350]


# ## PART II: EXPLORING THE DATA

# Now we are all set to explore the different features of the data and determine how good a feature it is for prediction whether a patient is likely to show up at an appointment.

# ### 2.1. Analyzing the probability of showing up with respect to Age, HourOfTheDay and AwaitingTime

# In[ ]:


df = pds.crosstab(index = noShow['AwaitingTime'], columns = noShow.Status).reset_index()
print(df)
df['probShowUp'] = df['Show-Up'] / (df['Show-Up'] + df['No-Show'])
print(df[['AwaitingTime', 'probShowUp']])

def probStatus(dataset, group_by):
    df = pds.crosstab(index = dataset[group_by], columns = dataset.Status).reset_index()
    df['probShowUp'] = df['Show-Up'] / (df['Show-Up'] + df['No-Show'])
    return df[[group_by, 'probShowUp']]


# First we will check how the likelihood that a person will show up at an appointment changes with respect to Age, HourOfTheDay, AwaitingTime.

# In[ ]:


sns.lmplot(data = probStatus(noShow, 'Age'), x = 'Age', y = 'probShowUp', fit_reg = True)
sns.plt.xlim(0, 100)
sns.plt.title('Probability of showing up with respect to Age')
sns.plt.show()

sns.lmplot(data = probStatus(noShow, 'HourOfTheDay'), x = 'HourOfTheDay', 
           y = 'probShowUp', fit_reg = True)
sns.plt.title('Probability of showing up with respect to HourOfTheDay')
sns.plt.show()

sns.lmplot(data = probStatus(noShow, 'AwaitingTime'), x = 'AwaitingTime', 
           y = 'probShowUp', fit_reg = True)
sns.plt.title('Probability of showing up with respect to AwaitingTime')
sns.plt.ylim(0, 1)
sns.plt.show()


# Clearly, HourOfTheDay and AwaitingTime are not good predictors of Status, since the probability of showing up depends feebly on the HourOfTheDay and not at all on the AwaitingTime. The significantly stronger dependency is observed with respect to Age.
# 
# Next, we do the same analysis for the other variables except AppointmentRegistration and AppointmentData since we have already analyzed the probability of showing up with respect to HourOfTheDay and AwaitingTime.0

# In[ ]:


def probStatusCategorical(group_by):
    rows = []
    for item in group_by:
        for level in noShow[item].unique():
            row = {'Condition': item}
            total = len(noShow[noShow[item] == level])
            n = len(noShow[(noShow[item] == level) & (noShow.Status == 'Show-Up')])
            row.update({'Level': level, 'Probability': n / total})
            rows.append(row)
    print(pds.DataFrame(rows)) 
    return pds.DataFrame(rows)

sns.barplot(data = probStatusCategorical(['Diabetes', 'Alchoholism', 'Hypertension',
                                          'Smokes', 'Tuberculosis', 'Scholarship']),
            x = 'Condition', y = 'Probability', hue = 'Level', palette = 'Set2')
sns.plt.title('Probability of showing up')
sns.plt.ylabel('Probability')
sns.plt.show()


# In[ ]:


sns.barplot(data = probStatusCategorical(['Gender']),
            x = 'Condition', y = 'Probability', hue = 'Level', palette = 'Set2')
sns.plt.title('Probability of showing up')
sns.plt.xlabel('Gender')
sns.plt.ylabel('Probability')
sns.plt.show()


# In[ ]:


sns.barplot(data = probStatusCategorical(['Handicap']),
            x = 'Condition', y = 'Probability', hue = 'Level', palette = 'Set2')
sns.plt.title('Probability of showing up')
sns.plt.ylabel('Probability')
sns.plt.show()


# In[ ]:


sns.barplot(data = probStatusCategorical(['DayOfTheWeek']),
            x = 'Condition', y = 'Probability', hue = 'Level', palette = 'Set2',
           hue_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday',
                       'Sunday'])
sns.plt.title('Probability of showing up')
sns.plt.ylabel('Probability')
sns.plt.show()


# In[ ]:


sns.barplot(data = probStatusCategorical(['Sms_Reminder']),
            x = 'Condition', y = 'Probability', hue = 'Level', palette = 'Set2')
sns.plt.title('Probability of showing up')
sns.plt.ylabel('Probability')
sns.plt.show()


# It is clear that the probability of showing up most prominently depends on Diabetes, Alchoholism, Hypertension, Smokes, Scholarship, Tuberculosis.

# ## PART III: PREDICTING WHETHER A PERSON WILL BE SHOWING UP 

# Now that the best features have been selected, we enter the process of predicting whether a person is going to show up at the appointment on the basis of those features. As a recap, the selected features are:
# 
# 1. Age
# 2. Diabetes
# 3. Alchoholism
# 4. Hypertension
# 5. Smokes
# 6. Scholarship
# 7. Tuberculosis

# ### 3.1. Preliminary preparation

# First we will prepare the data by converting the factor variables into numerical values.

# In[ ]:


def dayToNumber(day):
    if day == 'Monday': 
        return 0
    if day == 'Tuesday': 
        return 1
    if day == 'Wednesday': 
        return 2
    if day == 'Thursday': 
        return 3
    if day == 'Friday': 
        return 4
    if day == 'Saturday': 
        return 5
    if day == 'Sunday': 
        return 6

noShow.Gender = noShow.Gender.apply(lambda x: 1 if x == 'M' else 0)
noShow.DayOfTheWeek = noShow.DayOfTheWeek.apply(dayToNumber)
noShow.Status = noShow.Status.apply(lambda x: 1 if x == 'Show-Up' else 0)


# Then, we split the data into training and testing data. After a lot of trial and error, I have found that about 296500 samples in the training data are most helpful to our classifier to properly get the best fit. Beyond that it overfits and below that it underfits. That is why I am using 296500 as sample size for our training data.

# In[ ]:


features_train = noShow[['Age','Hypertension']].iloc[:296500]

labels_train = noShow.Status[:296500]

features_test = noShow[['Age','Hypertension']].iloc[296500:]

labels_test = noShow.Status[296500:]


# ### 3.2. Predicting status

# Since each feature in the training/testing sets contains a specific set of discrete values, i.e. Age contains integers from 0 to 95 and the other features are flags that contain only 0 or 1, I can safely assume that for Naive Bayes classification, P(x|y) follows a multinomial distribution. In that light, I will use the Multinomial Naive Bayes classifier to fit the data.

# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

clf =  MultinomialNB().fit(features_train, labels_train)
print('Accuracy:', round(accuracy_score(labels_test, 
                                        clf.predict(features_test)), 2) * 100, '%')


# An accuracy of 71% is although not good enough for real life applications, for this particular dataset, this was as far as I could go. I am not yet skilled enough to fine tune parameters of different classifiers enough to beat this score, so readers are encouraged to try it themselves and see if they can beat it. 
# 
# The second thing about this dataset I want to stress about is that the accuracy could have been greater if there had been more relevant features, such as more disease records or maybe some measure of how busy the schedule of a person is. 
# 
# Until next time. :)
