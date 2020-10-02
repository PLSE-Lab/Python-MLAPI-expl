#!/usr/bin/env python
# coding: utf-8

# This notebook presents some basic visualisations from the dataset provided. 
# 
# From the data provided, 30% of the time, the patient is "No-Show". A model using a random assignment of "Show/No-show" based on the same probability would be 58% accurate, i.e. it would be correct 58% of the time. Assuming that the person will show up is accurate 70% of the time. This also means that if our predictor does not do better than 70%, our prediction is not better than simply assuming that the person will show up for every case.

# In[ ]:


# Import some libraries
import pandas as pd
import numpy as np
import seaborn as sbs
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Read the dataset
dts = pd.read_csv("../input/No-show-Issue-Comma-300k.csv")
# Some early processing, such as binning
# Create categories
# Binning for Awaiting Time
# We consider Immediate for within 2 days, then within the week, within two weeks, within
# the month, within the trimester, within half year and everything above
bins = [-99999, -180, -90, -30, -14, -7, -2, 0]
labels = ["More than half year", "Half year", "Trimester", 
          "Month", "TwoWeeks", "Week", "Immediate"]
wait_period = pd.cut(dts.AwaitingTime, bins, labels=labels)
dts['Wait_period'] = wait_period
# Binning for age, based loosely on typical categories
# Parents tend to be more concerned for babies, Infants can't really tell what's wrong
# Child do but need parents to go to an appointment
# Teenagers are suffering from other ailment (beginning of puberty and such, memories ...)
# Young Adults may be in studies and have to work
bins = [0,2,6,12,18,40,65,np.inf]
labels=["Baby","Infant","Child","Teenager","Young adults", "Adult","Elder"]
age_cat = pd.cut(dts.Age,bins,labels=labels)
dts['Age_cat'] = age_cat
# Create a boolean for Status, with True if the patient showed up
dts.eval("Status_B = Status == 'Show-Up'", inplace=True)
# Extract the month of the visit
dts['Month'] = dts['ApointmentData'].apply(lambda x: x[5:7])
#Information about when the registration was made
dts['Reg_month'] = dts['AppointmentRegistration'].apply(lambda x: x[5:7])
dts['Reg_hour'] = dts['AppointmentRegistration'].apply(lambda x: x[11:13])
impact = {}
# How does the Status distribute?
groups = dts.groupby(['Status'])
gps = groups.size()
ax = sbs.barplot(x=gps.index.tolist(), y=gps.values)
sbs.plt.show()


# The dataset shows that about 30% of the appointments are "no-show" on average across every situation. Let's use this 30% value as the "baseline": anything that shows a significant deviation from this will be considered to be a factor.
# 
# # Wait_period
# 
# All the information relates to appointments made. This is provided as AwaitingTime (in days), which I have binned in categories. As shown below, above a trimester, there isn't much data available. 
# 
# For the categories "Immediate" and "Within the week", the no show rate is lower than the mean of 40%, for the categories "Within two weeks", "Within the month" and "Within the trimester", the no show rate are above the mean, and are increasing with time.
# 
# As there is not much data available for appointment scheduled above a trimester, the results are unlikely to be reliable.

# In[ ]:


# How does the Wait_Period distribute?
groups = dts.groupby(['Wait_period'])
gps = groups.size()
sbs.barplot(x=gps.index.tolist(), y=gps.values)
sbs.plt.show()

groups = dts[['Wait_period', 'Status','Smokes']].groupby(['Wait_period', 'Status']).count()
groups = groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()
std = groups.query('Status=="No-Show"')['Smokes'].std()
impact['Wait_period'] = std
sbs.barplot(y="Smokes", x="Wait_period", hue="Status", data=groups)
sbs.plt.show()


# # Appointment day of the week
# 
# Unsurprisingly, the weekend days see far less appointments than the other days, and it should be considered unreliable as they do have very few data points.
# 
# The no-show rate is slightly lower during the mid-week days (Tue-Thu) than it is on Monday or Friday. This is particularly visible for Mondays.
# 
# Later, we will group the days in weekends, near weekends (Monday and Friday) and Midweek.

# In[ ]:


# How does the Day of the week distribute?
groups = dts.groupby(['DayOfTheWeek'])
gps = groups.size()
sbs.barplot(x=gps.index.tolist(), y=gps.values)
sbs.plt.show()

groups = dts[['DayOfTheWeek', 'Status','Smokes']].groupby(['DayOfTheWeek', 'Status']).count()
groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()
std = groups.query('Status=="No-Show"')['Smokes'].std()
impact['DayOfTheWeek'] = std
sbs.barplot(y="Smokes", x="DayOfTheWeek", hue="Status", data=groups)
sbs.plt.show()


# # Month of appointment
# 
# Globally, the data points are well-balanced for the month of the appointment, with little peaks in March, May, July and October.
# 
# The attendance is nearly average, except in January, April, July, October and December. January sees less no-show on average, the other 4 more than average.

# In[ ]:


# How does the appointment month distribute?
groups = dts.groupby(['Month'])
gps = groups.size()
sbs.barplot(x=gps.index.tolist(), y=gps.values)
sbs.plt.show()

groups = dts[['Month', 'Status','Smokes']].groupby(['Month', 'Status']).count()
groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()
std = groups.query('Status=="No-Show"')['Smokes'].std()
impact['Month'] = std
sbs.barplot(y="Smokes", x="Month", hue="Status", data=groups)
sbs.plt.show()


# # Month of registration
# 
# Again, the distribution is rather balanced, with a notable decrease in registration in November and December.
# 
# The datapoint showing an appointment made in June or December are more susceptible to end up in a patient not showing up.  On the other end of the spectrum, appointments made in January or July will see the highest attendance.

# In[ ]:


# How does the registration month distribute?
groups = dts.groupby(['Reg_month'])
gps = groups.size()
sbs.barplot(x=gps.index.tolist(), y=gps.values)
sbs.plt.show()

groups = dts[['Reg_month', 'Status','Smokes']].groupby(['Reg_month', 'Status']).count()
groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()
std = groups.query('Status=="No-Show"')['Smokes'].std()
impact['Reg_month'] = std
sbs.barplot(y="Smokes", x="Reg_month", hue="Status", data=groups)
sbs.plt.show()


# # Hour of registration
# 
# For fun's sake, let's see if the hour at which the registration is made influence the attendance rate.
# 
# Beside the comical aspect of this, we can see that appointments are primarily made between 07:00 and 10:59, and between 13:00 and 16:59. 

# In[ ]:


# How does the registration hour distribute?
groups = dts.groupby(['Reg_hour'])
gps = groups.size()
sbs.barplot(x=gps.index.tolist(), y=gps.values)
sbs.plt.show()

groups = dts[['Reg_hour', 'Status','Smokes']].groupby(['Reg_hour', 'Status']).count()
groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()
std = groups.query('Status=="No-Show"')['Smokes'].std()
impact['Reg_hour'] = std
sbs.barplot(y="Smokes", x="Reg_hour", hue="Status", data=groups)
sbs.plt.show()


# # Age category
# 
# We opted to split the age data into categories pertaining to various stages: baby, infant ... up to elder which corresponds to people aged 65 and above.
# 
# The only two categories that tend to better than average when it comes to respecting an appointment are the "Adult" and "Elder" categories. 

# In[ ]:


# How does the Age_category distribute?
groups = dts.groupby(['Age_cat'])
gps = groups.size()
sbs.barplot(x=gps.index.tolist(), y=gps.values)
sbs.plt.show()

groups = dts[['Age_cat', 'Status','Smokes']].groupby(['Age_cat', 'Status']).count()
groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()
std = groups.query('Status=="No-Show"')['Smokes'].std()
impact['Age_cat'] = std
sbs.barplot(y="Smokes", x="Age_cat", hue="Status", data=groups)
sbs.plt.show()


# # Gender
# 
# Girls and women are twice as present in the dataset as men. 
# 
# The gender does not seem to be a very good predictor: both genders have almost the same rate of not showing up, with the girls/women a bit more likely to show than the man.

# In[ ]:


# How does the Gender distribute?
groups = dts.groupby(['Gender'])
gps = groups.size()
sbs.barplot(x=gps.index.tolist(), y=gps.values)
sbs.plt.show()

groups = dts[['Gender', 'Status','Smokes']].groupby(['Gender', 'Status']).count()
groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()
std = groups.query('Status=="No-Show"')['Smokes'].std()
impact['Gender'] = std
sbs.barplot(y="Smokes", x="Gender", hue="Status", data=groups)
sbs.plt.show()


# # SMS Reminder
# A lot has been already said about this value in other pages, especially on the presence of "2" in a small number of data points.
# 
# A bit more than 40% has not received a reminder, for which we won't look into the reasons. 
# 
# It seems that this does not work: the no-show rate is almost the same with or without a reminder. As another Kaggler said: the hospitals could simply stop sending this and not incur the cost, with potentially little to no effect on the attendance.

# In[ ]:


# How does the Day of the week distribute?
groups = dts.groupby(['Sms_Reminder'])
gps = groups.size()
sbs.barplot(x=gps.index.tolist(), y=gps.values)
sbs.plt.show()

groups = dts[['Sms_Reminder', 'Status','Smokes']].groupby(['Sms_Reminder', 'Status']).count()
groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()
std = groups.query('Status=="No-Show"')['Smokes'].std()
impact['Sms_reminder'] = std
sbs.barplot(y="Smokes", x="Sms_Reminder", hue="Status", data=groups)
sbs.plt.show()


# # Diabetes
# 
# Without looking at the data, my opinion is that the attendance will be higher for people suffering from that pathology than for the others. The reason being that diabetics require constant care and monitoring, and that some of the medicine needed is available only through prescription.
# 
# About 10% of the entries in the dataset corresponds to a diabetic patient. The attendance of that population is better than the average - about 25% of the appointments resulted in a patient not showing up, which is surprisingly high regarding our expectation of diabetics being a lot more conscious about meeting their praticians.

# In[ ]:


# How does the Diabetes distribute?
groups = dts.groupby(['Diabetes'])
gps = groups.size()
sbs.barplot(x=gps.index.tolist(), y=gps.values)
sbs.plt.show()

groups = dts[['Diabetes', 'Status','Smokes']].groupby(['Diabetes', 'Status']).count()
groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()
std = groups.query('Status=="No-Show"')['Smokes'].std()
impact['Diabetes'] = std
sbs.barplot(y="Smokes", x="Diabetes", hue="Status", data=groups)
sbs.plt.show()


# # Hypertension
# 
# Again, a pathology that requires a close monitoring and care, and that requires prescription medicine.
# 
# About 20% of the appointments in the dataset are related to someone who suffers from it. 
# 
# As for diabetes, the attendance is better than the average, again around 25%.

# In[ ]:


# How does the HiperTension distribute?
groups = dts.groupby(['HiperTension'])
gps = groups.size()
sbs.barplot(x=gps.index.tolist(), y=gps.values)
sbs.plt.show()

groups = dts[['HiperTension', 'Status','Smokes']].groupby(['HiperTension', 'Status']).count()
groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()
std = groups.query('Status=="No-Show"')['Smokes'].std()
impact['HiperTension'] = std
sbs.barplot(y="Smokes", x="HiperTension", hue="Status", data=groups)
sbs.plt.show()


# # Tuberculosis
# 
# The dataset associated with tuberculosis is very small, only 135 data points, which could be linked to the incidence rate of this disease.
# 
# Appointments made by patient with Tuberculosis tend to end in a no-show more often than average. This could be due to several things, such as when the patient feels better he does not feel compelled to attend to a follow-up appointment. Another potential explanation is that patients treated for TB die before the follow-up appointment.

# In[ ]:


# How does the Tuberculosis distribute?
groups = dts.groupby(['Tuberculosis'])
gps = groups.size()
sbs.barplot(x=gps.index.tolist(), y=gps.values)
sbs.plt.show()

groups = dts[['Tuberculosis', 'Status','Smokes']].groupby(['Tuberculosis', 'Status']).count()
groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()
std = groups.query('Status=="No-Show"')['Smokes'].std()
impact['Tuberculosis'] = std
sbs.barplot(y="Smokes", x="Tuberculosis", hue="Status", data=groups)
sbs.plt.show()


# # Handicap
# 
# This can take values from 0 to 4. The "non 0" values are very few in the dataset, with less than 6,000 data points. 
# 
# For these, the attendance seems to be better than average, with values 3 and 4 not reliable enough to be of use. Later, we will group this into "null handicap" and "non-null handicap".

# In[ ]:


# How does the Handicap distribute?
groups = dts.groupby(['Handcap'])
gps = groups.size()
sbs.barplot(x=gps.index.tolist(), y=gps.values)
sbs.plt.show()

groups = dts[['Handcap', 'Status','Smokes']].groupby(['Handcap', 'Status']).count()
groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()
std = groups.query('Status=="No-Show"')['Smokes'].std()
impact['Handcap'] = std
sbs.barplot(y="Smokes", x="Handcap", hue="Status", data=groups)
sbs.plt.show()


# # Alcoholism
# 
# The proportion of appointment made with alcoholism reported is less than 10%. 
# 
# These appointments are more often ignored than on average.

# In[ ]:


# How does the alcoholism distribute?
groups = dts.groupby(['Alcoolism'])
gps = groups.size()
sbs.barplot(x=gps.index.tolist(), y=gps.values)
sbs.plt.show()

groups = dts[['Alcoolism', 'Status','Smokes']].groupby(['Alcoolism', 'Status']).count()
groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()
std = groups.query('Status=="No-Show"')['Smokes'].std()
impact['Alcoolism'] = std
sbs.barplot(y="Smokes", x="Alcoolism", hue="Status", data=groups)
sbs.plt.show()


# # Smokers
# 
# This is a bit more prevalent than alcoholism - potentially because it is more socially acceptable to smoke than to be alcoholic.
# 
# The attendance is also worse for the appointments made by patients who reported being smokers than on average.

# In[ ]:


# How does the smoking distribute?
groups = dts.groupby(['Smokes'])
gps = groups.size()
sbs.barplot(x=gps.index.tolist(), y=gps.values)
sbs.plt.show()

groups = dts[['Tuberculosis', 'Status','Smokes']].groupby(['Smokes', 'Status']).count()
groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()
std = groups.query('Status=="No-Show"')['Tuberculosis'].std()
impact['Smokes'] = std
sbs.barplot(y="Tuberculosis", x="Smokes", hue="Status", data=groups)
sbs.plt.show()


# # Scholarship
# 
# From the owner of this dataset, the scholarship is a well-fare program provided to poor families in Brazil. Around 10% of the appointments were made by patients benefiting from this program.
# 
# Attendance-wise, these appointments tend to end up in a no-show more often than on average.

# In[ ]:


# How does the smoking distribute?
groups = dts.groupby(['Scholarship'])
gps = groups.size()
sbs.barplot(x=gps.index.tolist(), y=gps.values)
sbs.plt.show()

groups = dts[['Scholarship', 'Status','Smokes']].groupby(['Scholarship', 'Status']).count()
groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()
std = groups.query('Status=="No-Show"')['Smokes'].std()
impact['Scholarship'] = std
sbs.barplot(y="Smokes", x="Scholarship", hue="Status", data=groups)
sbs.plt.show()


# # Going for multiple variables
# 
# Some of the variables can be combined to represent a specific aspect, for example I will combine "Smokes", "Alcoolism" and "Scholarship" to get a proxy for "Social status". 
# 
# ## Socio-economic status
# 
# As said above, for this, I combined the "Smokes", "Alcoolism" and "Scholarship" variable. I suspect that a "true" measure of the socio-economic variable would probably yield better results. 
# 
# From the dataset, the majority of appointments is made by patient with no scholarship and who claimed to not smoke or be alcoholic.
# 
# The rest, less than 16% of all data points, have higher no-show rate, with a maximum of almost 47% (quasi 1 out of 2!) appointments that result in a no-show for patients with a scholarship and both an alchohol and smoking usage.
# 
# It is interesting to notice that, with one exception (both alcoholism and smoking), patients with no scholarship tend to show-up on average more than people with a scholarship.
# 
# Denoting S for scholarship, C for smokes and A for alcoholism, we can flag three clusters:
# --- and C for the higher rate of showing up, S and A for a medium rate of showing up and the rest (CA, SA, SC and SCA) for the lower rate of showing up.
# 
# Later, we will group these in these three categories.

# In[ ]:


groups = dts[['Scholarship', 'Smokes', 'Alcoolism', 'Status_B']].groupby(['Scholarship', 'Smokes', 'Alcoolism'])
gps = pd.DataFrame(groups.mean())
gps["counts"] = groups.count()["Status_B"]
gps["Show"] = groups.sum()["Status_B"]
gps


# ## Pathologies
# 
# Another angle to look at the dataset is through the pathologies: some of the variables included report whether the patient has diabetes, hypertension or tuberculosis.
# 
# On average, the patient with either Diabetes or Hypertension showed up more often than the patients with no pathology (68% show) or with "only" tuberculosis (58% show). 
# 
# Diabetes is often associated with hypertension, which shows in the numbers. 
# 
# Doing the same clustering exercise as above, and denoting D for diabetes, H for hypertension and T for tuberculosis, we have the high rate of showing up with the DHT, DT, the medium with DH, H and D, and the lower rate with HT, --- and T.
# 
# We will create a variable later to regroup these.

# In[ ]:


groups = dts.groupby(['Diabetes','HiperTension','Tuberculosis'])
gps = pd.DataFrame(groups.mean())
gps["counts"] = groups.count()["Status_B"]
gps["Show"] = groups.sum()["Status_B"]
gps[['Status_B','counts','Show']]


# #Predictions
# 
# For this section, I will create the additional variables, drop some of the existing ones and see if I can use it to predict the show/no-show with a better accuracy than the average of 70%.
# 
# The first variable is the period of the week: weekend (Saturday, Sunday), Near weekend (Monday, Friday) or Mid week (Tuesday, Wednesday and Thursday). 
# 
# Let's keep in mind that there aren't a lot of datapoint for the weekends.

# In[ ]:


def class_day(row):
    if row in ("Friday","Monday"):
        return "NearWeekend"
    if row in ("Sunday","Saturday"): #Sunday is way lower than Tuesday but that can be due to limited data
        return "Weekend"
    return "MidWeek"
dts['Day_type'] = dts['DayOfTheWeek'].apply(class_day)

# How does this distribute?
groups = dts.groupby(['Day_type'])
gps = groups.size()
sbs.barplot(x=gps.index.tolist(), y=gps.values)
sbs.plt.show()

groups = dts[['Day_type', 'Status','Smokes']].groupby(['Day_type', 'Status']).count()
groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()
std = groups.query('Status=="No-Show"')['Smokes'].std()
sbs.barplot(y="Smokes", x="Day_type", hue="Status", data=groups)
sbs.plt.show()
    


# The second variable we will change is the handcap one: we will create an HC, with False if handcap is 0 and True for anything else.

# In[ ]:


def class_hc(row):
    return row>0

dts['HC'] = dts['Handcap'].apply(class_hc)

# How does this distribute?
groups = dts.groupby(['HC'])
gps = groups.size()
sbs.barplot(x=gps.index.tolist(), y=gps.values)
sbs.plt.show()

groups = dts[['HC', 'Status','Smokes']].groupby(['HC', 'Status']).count()
groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()
std = groups.query('Status=="No-Show"')['Smokes'].std()
sbs.barplot(y="Smokes", x="HC", hue="Status", data=groups)
sbs.plt.show()


# Our third variable will gather the information about the socio-economics, based on the grouping we identified.
# 
# Denoting S for scholarship, C for smokes and A for alcoholism, we can flag three clusters:
# --- and C for the higher rate of showing up, S and A for a medium rate of showing up and the rest (CA, SA, SC and SCA) for the lower rate of showing up.

# In[ ]:


def class_se(row):
    K=0
    if row['Scholarship'] == 1:
        K += 4
    if row['Smokes'] == 1:
        K += 2
    if row['Alcoolism'] == 1:
        K += 1
    if K in (0,2):
        return "High"
    if K in (4,1):
        return "Medium"
    return "Low"
dts['Socio_Economics'] = dts[["Scholarship","Smokes","Alcoolism"]].apply(class_se, axis=1)

# How does this distribute?
groups = dts.groupby(['Socio_Economics'])
gps = groups.size()
sbs.barplot(x=gps.index.tolist(), y=gps.values)
sbs.plt.show()

groups = dts[['Socio_Economics', 'Status','Smokes']].groupby(['Socio_Economics', 'Status']).count()
groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()
std = groups.query('Status=="No-Show"')['Smokes'].std()
sbs.barplot(y="Smokes", x="Socio_Economics", hue="Status", data=groups)
sbs.plt.show()


# Lastly, our fourth variable summarises the the health information.
# 
# Doing the same clustering exercise as above, and denoting D for diabetes, H for hypertension and T for tuberculosis, we have the high rate of showing up with the DHT, DT, the medium with DH, H and D, and the lower rate with HT, --- and T.
# 

# In[ ]:


def class_health(row):
    K=0
    if row['Diabetes'] == 1:
        K += 4
    if row['HiperTension'] == 1:
        K += 2
    if row['Tuberculosis'] == 1:
        K += 1
    if K in (7,5):
        return "High"
    if K in (6,2,4):
        return "Medium"
    return "Low"
dts['Health'] = dts[["Diabetes","HiperTension","Tuberculosis"]].apply(class_health, axis=1)

# How does this distribute?
groups = dts.groupby(['Health'])
gps = groups.size()
sbs.barplot(x=gps.index.tolist(), y=gps.values)
sbs.plt.show()

groups = dts[['Health', 'Status','Smokes']].groupby(['Health', 'Status']).count()
groups = groups.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()
std = groups.query('Status=="No-Show"')['Smokes'].std()
sbs.barplot(y="Smokes", x="Health", hue="Status", data=groups)
sbs.plt.show()


# Lastly, let's prepare an addition dataframe and let's remove the columns we replaced.

# In[ ]:


mldts = dts.copy()


mldts.drop(['AppointmentRegistration','ApointmentData',
            'Status','Diabetes','Alcoolism','HiperTension','Handcap',
            'Smokes','Scholarship','Tuberculosis','Sms_Reminder',
            'Reg_hour'], inplace=True, axis=1)
# Replace gender with a 0/1 variable (0: male, 1: female)
mldts['Gender'] = mldts['Gender'].apply(lambda x: x=="F")
# Convert the categorical columns to dummy encoded ones
DT_age = pd.get_dummies(mldts['Age_cat'], prefix="AC")
#DT_dow = pd.get_dummies(mldts['DayOfTheWeek'], prefix="DOW")
DT_wp = pd.get_dummies(mldts['Wait_period'], prefix="WP")
DT_month = pd.get_dummies(mldts['Month'], prefix="MONTH")
DT_RM = pd.get_dummies(mldts['Reg_month'], prefix="RM")
DT_DT = pd.get_dummies(mldts['Day_type'], prefix="DT")
DT_SE = pd.get_dummies(mldts['Socio_Economics'], prefix="SE")
DT_Health = pd.get_dummies(mldts['Health'], prefix="Health")

mldts = pd.concat([mldts,DT_age, DT_wp, DT_month, DT_RM, DT_DT,
                   DT_SE, DT_Health], axis=1);
mldts.drop(['Age_cat','DayOfTheWeek','Wait_period','Month',
            'Reg_month','Day_type','Socio_Economics','Health'], inplace=True, axis=1)


target = mldts['Status_B']
mldts.drop('Status_B', inplace=True, axis=1)

# Logistic regression

logreg = LogisticRegression()
logreg.fit(mldts, target)
Y_pred = logreg.predict(mldts)
acc_log = logreg.score(mldts, target) * 100
print("Linear regression accuracy: %.2f" % (acc_log))

train_df = mldts.copy()

coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

cfcol = coeff_df.sort_values(by='Correlation', ascending=False)
cfcol


# Our simple model achieved 68% accuracy, which is about on par with assuming that anyone will show up.  
# 
# Let's check what the relevant features (|corr| > 0.15) are.

# In[ ]:


cfcol['abscorr'] = abs(cfcol['Correlation'])
features = cfcol.query('abscorr > 0.15')['Feature'].tolist()
print(cfcol.query('abscorr > 0.15')[['Feature', 'Correlation']])

fmldts=mldts

print("\n\nRandom Forests")
print("==============")
print("max_features=auto")
for nestimators in range(7,12):
    clf = RandomForestClassifier(n_estimators=nestimators)
    scores = cross_val_score(clf, fmldts, target)
    print("Nestimators: %2d, mean score: %.4f"%(nestimators, scores.mean()))
print("max_features=log2")
for nestimators in range(7,12):
    clf = RandomForestClassifier(n_estimators=nestimators, max_features="log2")
    scores = cross_val_score(clf, fmldts, target)
    print("Nestimators: %2d, mean score: %.4f"%(nestimators, scores.mean()))
print("max_feature=None")
for nestimators in range(7,12):
    clf = RandomForestClassifier(n_estimators=nestimators, max_features=None)
    scores = cross_val_score(clf, fmldts, target)
    print("Nestimators: %2d, mean score: %.4f"%(nestimators, scores.mean()))
print("\n\nKNN")
print("===")
for nneigh in range(3,10):
    clf = KNeighborsClassifier(n_neighbors=nneigh)
    scores = cross_val_score(clf, fmldts, target)
    print("N_neighbors: %3d, mean score: %.4f"%(nneigh, scores.mean()))
    


# In[ ]:


# Let's see if limiting the dataset to age category, socio-economic and current health 
# improves our rating

fmldts = pd.concat([DT_age,DT_SE,DT_Health,DT_month], axis=1)

print("\n\nRandom Forests")
print("==============")
print("max_features=auto")
for nestimators in range(7,12):
    clf = RandomForestClassifier(n_estimators=nestimators)
    scores = cross_val_score(clf, fmldts, target)
    print("Nestimators: %2d, mean score: %.4f"%(nestimators, scores.mean()))
print("max_features=log2")
for nestimators in range(7,12):
    clf = RandomForestClassifier(n_estimators=nestimators, max_features="log2")
    scores = cross_val_score(clf, fmldts, target)
    print("Nestimators: %2d, mean score: %.4f"%(nestimators, scores.mean()))
print("max_feature=None")
for nestimators in range(7,12):
    clf = RandomForestClassifier(n_estimators=nestimators, max_features=None)
    scores = cross_val_score(clf, fmldts, target)
    print("Nestimators: %2d, mean score: %.4f"%(nestimators, scores.mean()))


# #Conclusions
# 
# In this, I failed to make a prediction that is better than just "assuming everybody will show-up". It could be that the way I proceeded is not correct, that the data provided is not sufficient to clearly separate the people in categories between those who come and those who don't.
# 
# I suspect that it is a mix of several things. If the dataset owner can provide it, I think it would be a good idea to enrich the dataset with:
# 
# 
#  - Who scheduled the appointment? (patient, doctor, social services ...)
#  - What type of doctor is the patient willing to see? (Primary care provider, CARDIO, OBGYN, ...)
#  - Is this the first appointment (new patient) or a returning patient?
#  - What mode of transportation will the patient use?
#  - How far is the patient from the care facility location?
