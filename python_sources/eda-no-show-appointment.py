#!/usr/bin/env python
# coding: utf-8

# # Project: Investigate a Dataset (No-Show Appointments)
# 
# <div class="alert alert-warning" role="alert">
#   **NOTE:** if you are facing any issues in rendering the complete kernel, Please change the browser, change browser zoom, or clear cache.
# </div>
# 
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# Over 110 thousand of medical appointments took place in brazil, some patients did not show up for their appointments. the dataset contains 14 features including the present of the patient or not on his appointment date. The features can be classified into 4 groups, patient information (id, gender, age), appointment information (appointment id, appointment date, scheduled date, no show, sms received), health information (hypertension, diabetes, alcoholism, handicap) and social information (Neighborhood, Scholarship). by investigating the dataset, I am trying to answer the following questions:
# 
# <ul>
#     <li><b>Which genders and age groups, patients are most likely not to show up to their appointments?</b></li>
#     <li><b>Is early scheduling could be a reason for not showing to appointments? How SMS reminder may help?</b></li>
#     <li><b>At which part of the day patients are most likely to skip their scheduled appointments? Morning, Afternoon, Evening or Night?</b></li>
#     <li><b>At which day of the week patients are most likely to skip their scheduled appointments?How is that changing over the years and months?</b></li>
#     <li><b>Is there any correlation between patients positive records in hypertension, diabetes, alcoholism or / and handicap and them not showing up to their appointments?</b></li>
#     <li><b>Which neighborhood has the most no-show rate? are neighborhoods with more scholarship patients are most likely not to show?</b></li>
# </ul>

# In[ ]:


# importing packages and libraries and matplotlib for visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# ### General Properties

# In[ ]:


#importing CSV into data frame.
df = pd.read_csv("/kaggle/input/noshowappointments/KaggleV2-May-2016.csv")
rows, columns = df.shape
print("The data frame has "+ str(rows) +" rows and " + str(columns) + " columns")


# In[ ]:


#browse sample of data values and formats of each feature. 
df.head()


# In[ ]:


#browse data frame columns data types
df.info()


# In[ ]:


#print out statistical details of the numeric data.
df.describe()


# In[ ]:


#check number of not showing up patinets to an appointment on scale of 100
#group by no-show column
no_show_percentage = pd.DataFrame(df.groupby(["No-show"])["PatientId"].count())
#calculate percentage of show up and no show and store it in column No-Show
no_show_percentage["No-show"] = no_show_percentage["PatientId"] / sum(no_show_percentage["PatientId"]) * 100
no_show_percentage.drop(columns="PatientId", inplace=True)
#plot the dataframe 
no_show_percentage.plot.bar(figsize=(10,5))
plt.ylim(top=100)
plt.title("Medical Appointments",{'fontsize': 20},pad=20)
plt.xlabel("Appointment Status")
plt.xticks(np.arange(2), ('Show-Up', 'No-Show'), rotation=0)
plt.legend(["Appointment Status Rate"])


# In[ ]:


#checking the age distripution
df["Age"].describe()


# In[ ]:


#Check number of duplicated records in the data frame. 
print("Number of duplicate recrods: " + str(sum(df.duplicated())))


# In[ ]:


#assure gender has only two unique values
df["Gender"].unique()


# In[ ]:


#check neighbourhood unique list
df["Neighbourhood"].unique()


# In[ ]:


#check number of wrong values of handcap that exceeds a value of 1
print("Number of wrong handicap values: " + str(df.query("Handcap > 1")["Handcap"].count()))


# In[ ]:


#check scheduled Day and Appointment Day description
df[["ScheduledDay","AppointmentDay"]].describe()


# #### From above, we learn the following:
# <ul>
#     <li>No-Show appointment rate represented 20% of the data included in the study, and the considered to be reasonable reflection of reality. </li>
#     <li>Data does not have any null values or duplicates. </li>
#     <li>Age includes some wrong data. some records have '-1' value. </li>
#     <li>Handicap has 199 records which has invalid values of (2,3,4), and that does not match column type as Boolean. </li>
# </ul>
# 
# #### Data requires the below cleaning, transformation and conversions, to help us answering our goal questions:
# <ol>
#     <li>Fix column names spelling mistakes and apply lowercase letter and underscore word separation.</li>
#     <li>Convert scheduled day and appointment day data types from string to datetime.</li>
#     <li>Extract appointment time and classify it into 4-day parts (Morning, Afternoon, Evening, Night).</li>
#     <li>Calculating how early, by days, the appointment was scheduled.</li>
#     <li>Extract appointment year, month and weekday for appointment day.</li>
#     <li>Clean and classify age into age groups.</li>
#     <li>Correction of handicap invalid values.</li>
#     <li>Apply column data types corrections.</li>
#     <li>Drop unwanted colmuns for data set.</li>
#     <li>Order columns and store data set into new CSV.</li>
# </ol>
# 

# ### Data Cleaning, Transformation and conversions.
# 
# <b> Step 1:</b> Fix column names spelling mistakes and apply lowercase letter and underscore word separation

# In[ ]:


#new column names for columns requires word seperation with underscore or spelling mistakes
columnNames = {
            "PatientId":"patient_id", 
            "AppointmentID":"appointment_id",
            "ScheduledDay":"scheduled_day",
            "AppointmentDay":"appointment_day",
            "Hipertension":"hypertension",
            "Handcap":"handicap",
            "No-show":"no_show"
            }
df = pd.read_csv("/kaggle/input/noshowappointments/KaggleV2-May-2016.csv")
#rename columns
df.rename(columns=columnNames, inplace=True)
#lower case all columns names
df.columns = df.columns.str.lower()
df.dtypes


# <b>Step 2:</b> Convert scheduled day and appointment day data types from string to datetime

# In[ ]:


#converting columns scheduled_day and appointment_day to datetime64
df['scheduled_day'] = pd.to_datetime(df['scheduled_day'], format="%Y-%m-%d %H:%M:%S")
df['appointment_day'] = pd.to_datetime(df['appointment_day'], format="%Y-%m-%d %H:%M:%S")
#confirm new data types, as well check no null values was generated because of the transition.
df[["scheduled_day","appointment_day"]].info()


# In[ ]:


#look at the description of the date time columns
df[["scheduled_day","appointment_day"]].describe()


# ***Notice*** : all the appointments occurred between `2016-04-29` and `2016-06-08`. Hence, the data we are having is only for 2016 and for April, May and June of that year.

# <b>Step 3:</b> Extract appointment time and classify it into 4-day parts (Morning, Afternoon, Evening, Night)

# In[ ]:


df['appointment_day'].dt.time.describe()


# ***Notice***: the time of the appointment was not registered. There is no way we could no at which part of the day the appointment took place.

# <b>Step 4:</b> Calculating how early, by days, the appointment was scheduled

# In[ ]:


#schedule_days = appointment_day - scheduled_day
df["schedule_days"] = (df["appointment_day"] - df["scheduled_day"]).dt.days

df["schedule_days"].describe()


# ***Notice***: `25%+` of the records the schedule date happened after the appointment. that is a huge number to ignore as it will affect the dataset validity, let us look closer to the problem.

# In[ ]:


#check ditribuption of the data for schedule_days with negative values
ax1 = plt.subplot(1,2,1)
df.query("schedule_days < 0")["schedule_days"].hist(bins=30,figsize=(13,4))
ax1.set_title("Days of Scheduling Before Appointment (All Negative)")
ax1.set_xlabel("Delta Days (Appointment - Schedule)")
ax1.legend(["Number of Appointments"])
#check ditribuption of the data for schedule_days below that -1
ax2 = plt.subplot(1,2,2)
df.query("schedule_days < -1")["schedule_days"].hist(bins=30, figsize=(13,4))
ax2.set_title("Days of Scheduling Before Appointment (Below -1)")
ax2.set_xlabel("Delta Days (Appointment - Schedule)")
ax2.legend(["Number of Appointments"])
plt.tight_layout()


# ***Notice***: most of the invalid records states that appointment was schedule 1 day after. and only 5 records was scheduled for 2 and 7 days after. let us look closer into 1-day invalid schedule dates. 

# In[ ]:


#show the appointment date and schedule dates of appointments was scheulded 1 day after
df.query("schedule_days  == -1")[["schedule_days","scheduled_day", "appointment_day"]].head(10)


# ***Notice***: as expected, the issue of time was not registered in the appointment made the conflict. its clearly that those appointment was scheduled in the same day.
# 
# let us fix our calculation by taking the difference of the date only without time.

# In[ ]:


#apply the difference between scheduled day and appointment day with date only without time.
df["schedule_days"] = (df["appointment_day"].dt.date - df["scheduled_day"].dt.date).dt.days
#plot histogram of the negative schedule_days to confirm our results
df.query("schedule_days < 0")["schedule_days"].hist(bins=30,figsize=(10,5))
plt.title("Days of Scheduling Before Appointment (All Negative)")
plt.xlabel("Delta Days (Appointment - Schedule)")
plt.legend(["Number of Appointments"])


# ***Notice***: now we are having only 5 records of appointments was scheduled after its day. let us drop them.

# In[ ]:


#filter our appointments which was scheduled after its day.
df = df.query("schedule_days >= 0")
#look at schedule days description
df["schedule_days"].describe()


# let us group our schedule days into 4 groups based on the description of the data above:
# <ul>
#     <li>0 Days</li>
#     <li>1 to 4 Days</li>
#     <li>5 to 15 Days</li>
#     <li>Above 16 Days</li>
# </ul>

# In[ ]:


#classifier function that returns the schedule_days group
def schedule_days_classifier(schedule_days):
    if schedule_days == 0:
        return "0 Days"
    elif schedule_days >= 1 and schedule_days < 5:
        return "1-4 Days"
    elif schedule_days >= 5 and schedule_days < 16:
        return "5-15 Days"
    else:
        return "16+ Days"
#apply classifier and store it in schedule_days    
df["schedule_days"] = df["schedule_days"].apply(schedule_days_classifier)


# In[ ]:


#static order of the schedule_days group
schedule_days_order = ["0 Days", "1-4 Days", "5-15 Days", "16+ Days"]
#group and plot
df.groupby(["schedule_days"]).count()[["no_show"]].loc[schedule_days_order].plot.bar()


# ***Notice***: alot of patients takes their appointments in the same day.
# 
# for now let us drop schedule day column

# In[ ]:


#drop scheduled_day column
df.drop(columns=["scheduled_day"], inplace=True)
df.columns


# <b>Step 5:</b> Extract appointment year, month and weekday for appointment day.

# In[ ]:


#print appointment unique years
print("Appointments occured in years of: " + np.array2string(df['appointment_day'].dt.year.unique()))
#print appointment unique months
print("Appointments occured in months of: " + np.array2string(df['appointment_day'].dt.month.unique()))


# ***Notice***: It is not confirmed that the appointments took place in 2016 only and in April, May and June. I dont think this extraction for those features will be helpful to us. let us concentrate on the weekday.

# In[ ]:


#list of week_days
week_day_list = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
#week day classifier get day and returns the day name.
def week_day_classifier(day):
    return week_day_list[day];
#apply classifier and store it in week_day
df["week_day"] = df["appointment_day"].dt.weekday.apply(week_day_classifier)
#print out week day sample data
df[[ "appointment_id", "week_day"]].head()


# Let us drop appointment day column

# In[ ]:


#drop appointment_day column
df.drop(columns=["appointment_day"], inplace=True)
df.columns


# <b> Step 6:</b> Clean and classify age into age groups
# 
# let us first query patients with negative age records.

# In[ ]:


# get patient id for patients has negative age
df.query("age < 0")["patient_id"].astype(str).str[:-2]


# ***Notice***: it is only one record of patients that has negative age with patient ID : 465943158731293. Before we drop out this record, let us try to find if the patient has other records we can get his correct age from.

# In[ ]:


#find other records for patient id =465943158731293
df.query("patient_id == '465943158731293'")


# the patient has only 1 recod, let us drop it.

# In[ ]:


#filter our records with negative age.
df = df.query("age >= 0")


# In[ ]:


#let us see the age distribution
df["age"].describe()


# From the age distribution above, let us classify age into 4 age groups:
# <ul>
#     <li>[0-18) => Kids</li>
# <li>[18-37) => Adults</li>
# <li>[37-55) => Matures</li>
# <li>[55-115) => Elders</li>
#     </ul>

# In[ ]:


#age classifier function
def age_classifier(age):
    if age >= 0 and age <18:
        return "Kids"
    elif age >= 18 and age < 37:
        return "Adults"
    elif age >= 37 and age < 55:
        return "Matures"
    else:
        return "Elders"
#apply age classifier and store into age_group    
df["age_group"] = df["age"].apply(age_classifier)
#drop age column
df.drop(columns=["age"], inplace=True)
#print out patinet information smaple data
df[["patient_id", "gender", "age_group"]].head()


# <b>Step 7:</b> Correction of handicap invalid values. 

# As handicap describes is the patient is handicapped or not, then i am going to consider any value 1 or above states the patient is handicapped, and 0 state the patient is not.

# In[ ]:


#make handicap value above 1 to be equal to 1
df.loc[df.handicap >1 , 'handicap'] =1
df[["handicap"]].describe()


# <b>Step 8:</b> Apply column data types corrections.

# In[ ]:


#convert scholarship, hypertension, diabetes, alcoholism, handicap, sms_received to boolean
df["scholarship"] = df["scholarship"].astype(bool)
df["hypertension"] = df["hypertension"].astype(bool)
df["diabetes"] = df["diabetes"].astype(bool)
df["alcoholism"] = df["alcoholism"].astype(bool)
df["handicap"] = df["handicap"].astype(bool)
df["sms_received"] = df["sms_received"].astype(bool)
df[["scholarship","hypertension", "diabetes", "alcoholism", "handicap", "sms_received"]].dtypes


# In[ ]:


#Convert no_show column from Yes/No into True/False
def noshow_to_boolean(status):
    if status == 'No':
        return False
    else:
        return True
    
df["no_show"] = df["no_show"].apply(noshow_to_boolean)
df[["no_show"]].dtypes


# <b>Step 9:</b> Drop unwanted colmuns for data set.

# In[ ]:


#dropping patient_id and appointment_id
df.drop(columns=['patient_id', 'appointment_id'], inplace=True)
df.columns


# <b>Step 10</b> Order columns and store data set into new CSV.

# In[ ]:


#order data set columns
df = df[['gender', 'age_group', 'neighbourhood','scholarship','hypertension', 'diabetes',
       'alcoholism', 'handicap', 'week_day', 'schedule_days', 'sms_received','no_show']]
#store data frame into cleaned csv
df.to_csv('no_show_cleaned.csv', index=False)


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# 

# In[ ]:


#load cleaned Data Frame
df_clean = pd.read_csv('no_show_cleaned.csv')


# ### Which genders and age groups, patients are most likely not to show up to their appointments?

# Let us compare Number of Males to Number of Females and then Number of show-up and no-show-up for each gender

# In[ ]:


#group by gender
gender_all = df_clean.groupby(["gender"])[["gender"]].count()
#Calculate percentage of appointments per gender
gender_all.columns = ["Gender Rate"]
gender_all["Gender Rate"] = gender_all["Gender Rate"] / sum(gender_all["Gender Rate"]) * 100
gender_all.reset_index(inplace=True)


# In[ ]:


#group by gender and no_show
gender_by_no_show = df_clean.groupby(["gender", "no_show"])[["gender"]].count()
#calculate percentage of appointment per gender per appointment show up status
gender_by_no_show.columns = ["no_show_count"]
gender_by_no_show.reset_index(inplace=True)
gender_by_no_show.columns = ["Gender", "No Show Status", "No Show Count"]
gender_by_no_show =  pd.DataFrame(gender_by_no_show.groupby(["Gender","No Show Status"])["No Show Count"].sum() / gender_by_no_show.groupby(["Gender"])["No Show Count"].sum() * 100)
gender_by_no_show = gender_by_no_show.unstack()


# In[ ]:


fig, axs = plt.subplots(1,2,figsize=(15,5))
fig.suptitle('Appointment per Gender', fontsize=16)

#plot percentage of appointments per gender
gender_all.plot.bar(ax=axs[0])
axs[0].set_xticklabels(("Female","Male"), rotation=0)
axs[0].set_ylim(top=100)
axs[0].set_xlabel("Gender")
axs[0].legend(["% of Appointments per Gender"])

#plot percentage of appointment per gender per appointment show up status
gender_by_no_show.plot.bar(ax=axs[1], stacked=True)
axs[1].set_xticklabels(("Female","Male"), rotation=0)
axs[1].set_ylim(top=100)
axs[1].set_xlabel("Gender")
axs[1].legend(["Show Up", "No Show"])


# ***Observation 1***: Appointments of femalre patients are higher than male patients, BUT, the rate of not showing up to the appointments are closely the same.
# 
# Now, let us compare number of appointments per age groups.

# In[ ]:


#group by age group
age_group_all = df_clean.groupby(["age_group"])[["age_group"]].count()
#calculate percentage of appointments per age group 
age_group_all.columns = ["Age Group Rate"]
age_group_all["Age Group Rate"] = age_group_all["Age Group Rate"] / sum(age_group_all["Age Group Rate"]) * 100
age_group_all.reset_index(inplace=True)


# In[ ]:


#group by age group per appointment show up status
age_group_no_show = df_clean.groupby(["age_group", "no_show"])[["age_group"]].count()
#calculate percentage of appointments per age group per appointment show up status
age_group_no_show.columns = ["age_group_count"]
age_group_no_show.reset_index(inplace=True)
age_group_no_show.columns = ["Age Group", "No Show Status", "No Show Count"]
age_group_no_show = pd.DataFrame(age_group_no_show.groupby(["Age Group","No Show Status"])["No Show Count"].sum() / age_group_no_show.groupby(["Age Group"])["No Show Count"].sum() * 100)
age_group_no_show = age_group_no_show.unstack()


# In[ ]:


fig, axs = plt.subplots(1,2,figsize=(15,5))
fig.suptitle('Appointment per Age Group', fontsize=16)

#plot percentage of appointments per age group 
age_group_all.plot.bar(ax=axs[0])
axs[0].set_xticklabels(("Under 18","19 to 37", "38 to 55", "Above 55"), rotation=0)
axs[0].set_ylim(top=30)
axs[0].set_xlabel("Age Group")
axs[0].legend(["% of Appointments per Age Group"])
#plot percentage of appointments per age group per appointment show up status
age_group_no_show.plot.bar(ax=axs[1], stacked=True)
axs[1].set_xticklabels(("Under 18","19 to 37", "38 to 55", "Above 55"), rotation=0)
axs[1].set_ylim(top=100)
axs[1].set_xlabel("Age Group")
axs[1].legend(["Show Up", "No Show"])


# ***Observation 2***: Ages between 19 to 37 has greatest number of appointments, as well it has the lowest rate of not showing up to their appointments. All age groups has a change of not showing up to their appointments within range of 15 to 25%.

# ### Is early scheduling could be a reason for not showing to appointments? How SMS reminder may help?

# Let us compare Number of appointments per groups of early scheduling days 

# In[ ]:


#group by schedule days group
schedule_days_all = df_clean.groupby(["schedule_days"])[["schedule_days"]].count().loc[schedule_days_order]
#calculate percentage of appointments per schedule day groups
schedule_days_all.columns = ["Schedule Days Rate"]
schedule_days_all["Schedule Days Rate"] = schedule_days_all["Schedule Days Rate"] / sum(schedule_days_all["Schedule Days Rate"]) * 100
schedule_days_all.reset_index(inplace=True)


# In[ ]:


#group by schedule days group and appointment show up status
schedule_days_no_show = df_clean.groupby(["schedule_days", "no_show"])[["schedule_days"]].count()
#calcualte percentage of appointments per schedule day group per appointment show up status
schedule_days_no_show.columns = ["schedule_days_count"]
schedule_days_no_show.reset_index(inplace=True)
schedule_days_no_show.columns = ["Schedule Days", "No Show Status", "No Show Count"]
schedule_days_no_show = pd.DataFrame(schedule_days_no_show.groupby(["Schedule Days","No Show Status"])["No Show Count"].sum() / schedule_days_no_show.groupby(["Schedule Days"])["No Show Count"].sum() * 100)
schedule_days_no_show = schedule_days_no_show.unstack().loc[schedule_days_order]


# In[ ]:


fig, axs = plt.subplots(1,2,figsize=(15,5))
fig.suptitle('Appointment per Schedule Days', fontsize=16)

#plot percentage of appointments per schedule day groups
schedule_days_all.plot.bar(ax=axs[0])
axs[0].set_xticklabels(schedule_days_order, rotation=0)
axs[0].set_ylim(top=30)
axs[0].set_xlabel("Schedule Days")
axs[0].legend(["% of Appointments per Schedule Days"])

#plot percentage of appointments per schedule day group per appointment show up status
schedule_days_no_show.plot.bar(ax=axs[1], stacked=True)
axs[1].set_xticklabels(schedule_days_order, rotation=0)
axs[1].set_ylim(top=100)
axs[1].set_xlabel("Schedule Days")
axs[1].legend(["Show Up", "No Show"])


# ***Observation 3***: most of the patients schedule their appointments in the same day, and those patients are most likely to show up in a percentage around 95%. as early as the patient schedule their appointments are most likely not going to show up to their appointments.

# Now let us look how SMS reminders to the patients might affect the appointment show up status rate.

# In[ ]:


#get only show up appointments and group by schedule days per sms_received per appointment show up status
schedule_days_sms_showed_up = df_clean.query("no_show == False").groupby(["schedule_days", "sms_received", "no_show"])[["schedule_days"]].count()
#calcualte the percentage of scheudle days per sms received per appointment show up status
schedule_days_sms_showed_up.columns = ["schedule_days_count"]
schedule_days_sms_showed_up.reset_index(inplace=True)
schedule_days_sms_showed_up.columns = ["Schedule Days","SMS Recieved", "No Show Status", "No Show Count"]
schedule_days_sms_showed_up = pd.DataFrame(schedule_days_sms_showed_up.groupby(["Schedule Days","SMS Recieved","No Show Status"])["No Show Count"].sum() / schedule_days_sms_showed_up.groupby(["Schedule Days"])["No Show Count"].sum() * 100)
#unstack twice the data
schedule_days_sms_showed_up = schedule_days_sms_showed_up.unstack().unstack().loc[schedule_days_order]


# In[ ]:


#get only no-show appointments and group by schedule days per sms_received per appointment show up status
schedule_days_sms_no_show = df_clean.query("no_show == True").groupby(["schedule_days", "sms_received", "no_show"])[["schedule_days"]].count()
#calcualte the percentage of scheudle days per sms received per appointment show up status
schedule_days_sms_no_show.columns = ["schedule_days_count"]
schedule_days_sms_no_show.reset_index(inplace=True)
schedule_days_sms_no_show.columns = ["Schedule Days","SMS Recieved", "No Show Status", "No Show Count"]
schedule_days_sms_no_show = pd.DataFrame(schedule_days_sms_no_show.groupby(["Schedule Days","SMS Recieved","No Show Status"])["No Show Count"].sum() / schedule_days_sms_no_show.groupby(["Schedule Days"])["No Show Count"].sum() * 100)
#unstack twice the data
schedule_days_sms_no_show = schedule_days_sms_no_show.unstack().unstack().loc[schedule_days_order]


# In[ ]:


fig, axs = plt.subplots(1,2,figsize=(15,5))
fig.suptitle('SMS Reminder Affect on Early Scheduling', fontsize=16)

schedule_days_sms_showed_up.plot.bar(ax=axs[0],stacked=True);
axs[0].set_xticklabels(schedule_days_order, rotation=0)
axs[0].set_ylim(top=100)
axs[0].set_title("Show-Up Appointment")
axs[0].set_xlabel("Schedule Days")
axs[0].legend(["SMS Recieved", "SMS Not Recieved"])

schedule_days_sms_no_show.plot.bar(ax=axs[1], stacked=True)
axs[1].set_xticklabels(schedule_days_order, rotation=0)
axs[1].set_ylim(top=100)
axs[1].set_title("No-Show APpointment")
axs[1].set_xlabel("Schedule Days")
axs[1].legend(["SMS Recieved", "SMS Not Recieved"])


# ***Observation 4***: SMS reminders has small affect on the appointments was scheduled before 5+ days in an amount of 10%.

# ### At which part of the day patients are most likely to skip their scheduled appointments? Morning, Afternoon, Evening or Night?

# As the appointment time was not recorded, this question will be skipped. 

# ### At which day of the week patients are most likely to skip their scheduled appointments?How is that changing over the years and months?

# Let us compare Number of appointments per weekdays

# In[ ]:


#group by week days
weekday_all = df_clean.groupby(["week_day"])[["week_day"]].count()
#calculate percentage of appointment per week day
weekday_all.columns = ["Week Day Rate"]
weekday_all["Week Day Rate"] = weekday_all["Week Day Rate"] / sum(weekday_all["Week Day Rate"]) * 100
#order index column by weekday order
weekday_all = weekday_all.reindex(week_day_list)


# In[ ]:


#group by week days per appointment show up status
week_day_no_show = df_clean.groupby(["week_day", "no_show"])[["week_day"]].count()
#calculate percentage of appointment per week day per appointment show up status
week_day_no_show.columns = ["week_day_count"]
week_day_no_show.reset_index(inplace=True)
week_day_no_show.columns = ["Week Day", "No Show Status", "No Show Count"]
week_day_no_show = pd.DataFrame(week_day_no_show.groupby(["Week Day","No Show Status"])["No Show Count"].sum() / week_day_no_show.groupby(["Week Day"])["No Show Count"].sum() * 100)
week_day_no_show = week_day_no_show.unstack()
#order index by weekday order
week_day_no_show = week_day_no_show.reindex(week_day_list)


# In[ ]:


fig, axs = plt.subplots(1,2,figsize=(15,5))
fig.suptitle('Appointments per Weekday', fontsize=16)

#plot of percentage of appointment per week day
weekday_all.plot.bar(ax=axs[0],stacked=True);
axs[0].set_xticklabels(week_day_list, rotation=0)
axs[0].set_ylim(top=30)
axs[0].set_xlabel("Week Days")
axs[0].legend(["% of Appointments per Week Days"])

#plot of percentage of appointment per week day per appointment show up status
week_day_no_show.plot.bar(ax=axs[1], stacked=True)
axs[1].set_xticklabels(week_day_list, rotation=0)
axs[1].set_ylim(top=100)
axs[1].set_xlabel("Week Days")
axs[1].legend(["Show Up", "No Show"])


# ***Observation 5***: Patients scheudle their appointments to be on the weekdays not in the weekends. And all of the weekdays has almost equal rate of patients no-show to their appointments. 
# 
# We are unable to study the changing of no-show rate over the months and the years, as the data is only representing short interval of time. 

# ### Is there any correlation between patients positive records in hypertension, diabetes, alcoholism or / and handicap and them not showing up to their appointments?

# let us compare each health feature with rate of appointment show up status.

# In[ ]:


#group by hypertenstion per appointment show up status
hipertension_no_show = df_clean.groupby(["hypertension", "no_show"])[["no_show"]].count()
#calculate the percentage appointments of hypertentation  per appointment show up status
hipertension_no_show.columns = ["hypertension_count"]
hipertension_no_show.reset_index(inplace=True)
hipertension_no_show.columns = ["Hypertension", "No Show Status", "No Show Count"]
hipertension_no_show = pd.DataFrame(hipertension_no_show.groupby(["Hypertension","No Show Status"])["No Show Count"].sum() / hipertension_no_show.groupby(["Hypertension"])["No Show Count"].sum() * 100)
hipertension_no_show = hipertension_no_show.unstack()


# In[ ]:


#group by diabetes per appointment show up status
diabetes_no_show = df_clean.groupby(["diabetes", "no_show"])[["no_show"]].count()
#calculate the percentage appointments of diabetes per appointment show up status
diabetes_no_show.columns = ["diabetes_count"]
diabetes_no_show.reset_index(inplace=True)
diabetes_no_show.columns = ["Diabetes", "No Show Status", "No Show Count"]
diabetes_no_show = pd.DataFrame(diabetes_no_show.groupby(["Diabetes","No Show Status"])["No Show Count"].sum() / diabetes_no_show.groupby(["Diabetes"])["No Show Count"].sum() * 100)
diabetes_no_show = diabetes_no_show.unstack()


# In[ ]:


#group by diabetes per appointment show up status
alcoholism_no_show = df_clean.groupby(["alcoholism", "no_show"])[["no_show"]].count()
#calculate the percentage appointments of alcoholism per appointment show up status
alcoholism_no_show.columns = ["alcoholism_count"]
alcoholism_no_show.reset_index(inplace=True)
alcoholism_no_show.columns = ["Alcoholism", "No Show Status", "No Show Count"]
alcoholism_no_show = pd.DataFrame(alcoholism_no_show.groupby(["Alcoholism","No Show Status"])["No Show Count"].sum() / alcoholism_no_show.groupby(["Alcoholism"])["No Show Count"].sum() * 100)
alcoholism_no_show = alcoholism_no_show.unstack()


# In[ ]:


#group by handicap per appointment show up status
handcap_no_show = df_clean.groupby(["handicap", "no_show"])[["no_show"]].count()
#calculate the percentage appointments of handicap per appointment show up status
handcap_no_show.columns = ["handicap_count"]
handcap_no_show.reset_index(inplace=True)
handcap_no_show.columns = ["Handicap", "No Show Status", "No Show Count"]
handcap_no_show = pd.DataFrame(handcap_no_show.groupby(["Handicap","No Show Status"])["No Show Count"].sum() / handcap_no_show.groupby(["Handicap"])["No Show Count"].sum() * 100)
handcap_no_show = handcap_no_show.unstack()


# In[ ]:


fig, axs = plt.subplots(1,4,figsize=(20,5))
fig.suptitle('Halth Status VS No Show', fontsize=16)
#plot hypertenstion per appointment show up status
hipertension_no_show.plot.bar(ax=axs[0],stacked=True);
axs[0].set_xticklabels(("False", "True"),rotation=0)
axs[0].set_ylim(top=100)
axs[0].set_title("Hipertension VS No Show")
axs[0].set_xlabel("Hipertension")
axs[0].legend(["Show Up", "No Show"])

#plot diabetes per appointment show up status
diabetes_no_show.plot.bar(ax=axs[1], stacked=True)
axs[1].set_xticklabels(("False", "True"),rotation=0)
axs[1].set_ylim(top=100)
axs[1].set_title("Diabetes VS No Show")
axs[1].set_xlabel("Diabetes")
axs[1].legend(["Show Up", "No Show"])

#plot alcoholism per appointment show up status
alcoholism_no_show.plot.bar(ax=axs[2],stacked=True);
axs[2].set_xticklabels(("False", "True"),rotation=0)
axs[2].set_ylim(top=100)
axs[2].set_title("Alcoholism VS No Show")
axs[2].set_xlabel("Alcoholism")
axs[2].legend(["Show Up", "No Show"])

#plot handicaped per appointment show up status
handcap_no_show.plot.bar(ax=axs[3], stacked=True)
axs[3].set_xticklabels(("False", "True"), rotation=0)
axs[3].set_ylim(top=100)
axs[3].set_title("Handcap VS No Show")
axs[3].set_xlabel("Handcap")
axs[3].legend(["Show Up", "No Show"])


# ***Observation 6***: from above, all health statuses show no affect on the patient not showing to their appointments or not.

# ### Which neighborhood has the most no-show rate? are neighborhoods with more scholarship patients are most likely not to show?

# let us compare number of appointments per neighborhoods.

# In[ ]:


#group by neighbourhood per appointment show up status.
neighbourhood_all = df_clean.groupby(["neighbourhood", "no_show"])[["no_show"]].count()
neighbourhood_all.columns = ["no_show_count"]
neighbourhood_all.reset_index(inplace=True)
#Calculate percentage appointments per neighborhood per appointment show up status
neighbourhood_all["no_show_rate"] = pd.DataFrame(neighbourhood_all.groupby(["neighbourhood","no_show"])["no_show_count"].sum() / neighbourhood_all.groupby(["neighbourhood"])["no_show_count"].sum() * 100).reset_index()[["no_show_count"]]
neighbourhood_all = neighbourhood_all.groupby(["neighbourhood","no_show"])[["no_show_count", "no_show_rate"]].sum()
neighbourhood_all = neighbourhood_all.unstack()
#for neighbours has all patients showed up or all patients not showed to their appointment, substitute by 0
neighbourhood_all = neighbourhood_all.fillna(0)


# In[ ]:


#plot hypertenstion per appointment show up status
axs = neighbourhood_all["no_show_count"].sort_values(by=False).plot.bar(stacked=True, figsize=(20,5));
axs.set_xlabel("neighbourhood")
axs.legend(["Show Up", "No Show"])
axs.set_title("Appointment Per All Neigbourhoods", fontsize=16)


# let us execlude neighbourhoods which has less than 1000 appointments. The reason behind this execluding is that they dont have enough appointments to study their no show rate. As well, those neighbourhoods cannot be classified based on their patients scholarships as we are going to see in the following steps.

# In[ ]:


#excluding all neigbourhoods which has less than 1000 appointments
neighbourhood_above_1000_visits = neighbourhood_all[neighbourhood_all["no_show_count"][False] + neighbourhood_all["no_show_count"][True] > 1000]


# In[ ]:


# plot percentage of appointments per neighbourhood per appointment show up status.
axs = neighbourhood_above_1000_visits["no_show_rate"].sort_values(by=False).plot.bar(stacked=True, figsize=(20,5));
axs.set_xlabel("neighbourhood")
axs.legend(["Show Up", "No Show"])
axs.set_title("Appointment Per Neigbourhoods (1000+ Appointments)", fontsize=16)


# ***Observation 7***: all neigbourshood has no-show up appointments are around the 20%.
# 
# Let us find out the affect of the scholarships on the neighbourhoods. to do this, i am going to classify the neighbourhoods into social classes based on the rate of the patients has medical scholarships.

# In[ ]:


#group by negibourhoods per scholarships, for only neigbourhoods has more than 1000 appointments. 
neighbourhood_scholarship = df_clean.query(f"neighbourhood in {neighbourhood_above_1000_visits.index.tolist()}").groupby(["neighbourhood", "scholarship"])[["scholarship"]].count()
neighbourhood_scholarship.columns = ["scholarship_count"]
neighbourhood_scholarship.reset_index(inplace=True)
#caclualte scholraship rate per neighbourhoods
neighbourhood_scholarship["scholarship_rate"] = pd.DataFrame(neighbourhood_scholarship.groupby(["neighbourhood","scholarship"])["scholarship_count"].sum() / neighbourhood_scholarship.groupby(["neighbourhood"])["scholarship_count"].sum() * 100).reset_index()[["scholarship_count"]]
neighbourhood_scholarship = neighbourhood_scholarship.groupby(["neighbourhood", "scholarship"])[["scholarship_rate"]].sum()
neighbourhood_scholarship.reset_index(inplace=True)
#find neigbourhood scholarships distribution
neighbourhood_scholarship.query("scholarship == True").describe()


# In[ ]:


#function to classify negbourhood by the scholraship rate.
def neighbourhood_social_classifier(row):
    x = row["scholarship_rate"]
    if(row["scholarship"] == False):
        x = 100 - x
    if x >= 0.283725 and x < 8.913911:
        return "Class A"
    elif x >= 8.913911 and x < 11.761120:
        return "Class B"
    elif x >= 11.761120 and x < 14.424395:
        return "Class C"
    else:
        return "Class D"
    
#apply classigication of neighbourhoods
neighbourhood_scholarship["neighbourhood_class"] = neighbourhood_scholarship.apply(neighbourhood_social_classifier,axis=1)
neighbourhood_scholarship_class = neighbourhood_scholarship.loc[:,["neighbourhood", "neighbourhood_class"]]
#drop dublicate records
neighbourhood_scholarship_class.drop_duplicates(inplace=True)


# In <i>neighbourhood_scholarship_class</i>, it contains neighbourhood scholarship dictionary table.

# In[ ]:


#function to get neigbourhood class
def get_neighbourhood_class(value):
    return neighbourhood_scholarship_class.query(f"neighbourhood == '{value}'")["neighbourhood_class"].values[0]

neighbourhood_above_1000_visits_classed = neighbourhood_above_1000_visits.reset_index()
#Apply classification of neighbourhood
neighbourhood_above_1000_visits_classed["neighbourhood_class"] = neighbourhood_above_1000_visits_classed["neighbourhood"].apply(get_neighbourhood_class)
#group by neigbourhood class
neighbourhood_above_1000_visits_classed.drop(columns=['neighbourhood'], inplace=True, level=0)
neighbourhood_above_1000_visits_classed = neighbourhood_above_1000_visits_classed.groupby(["neighbourhood_class"]).sum().stack()
neighbourhood_above_1000_visits_classed = neighbourhood_above_1000_visits_classed[["no_show_count"]]
#caclulate percentage of appointments per neigbourhood class
neighbourhood_above_1000_visits_classed["no_show_rate"] = pd.DataFrame(neighbourhood_above_1000_visits_classed["no_show_count"] / neighbourhood_above_1000_visits_classed.groupby(["neighbourhood_class"])["no_show_count"].sum() * 100)[["no_show_count"]].values
neighbourhood_above_1000_visits_classed = neighbourhood_above_1000_visits_classed.unstack()


# In[ ]:


fig, axs = plt.subplots(1,2,figsize=(20,7))
fig.suptitle('Appointments per Neighbourhood Class', fontsize=16)

#plot neighbourhood class per appointment show up status.
neighbourhood_above_1000_visits_classed["no_show_count"].plot.bar(ax=axs[0],stacked=True);
axs[0].set_xlabel("Neighbourhood Class")
axs[0].legend(["Show Up", "No Show"])
# plot percentage of appointments class per neighbourhood per appointment show up status.
neighbourhood_above_1000_visits_classed["no_show_rate"].plot.bar(ax=axs[1],stacked=True);
axs[1].set_ylim(top=100)
axs[1].set_xlabel("Neighbourhood Class")
axs[1].legend(["Show Up", "No Show"])


# ***Observation 8***: Neighbourhoods of class A, has least patients with scholarships, has the most medical appointments. But appointments show-up rate are equals for neighbourhoods classes.

# <a id='conclusions'></a>
# ## Conclusions
# 
# From the observations above, I can state the main cause of patients not showing up to their appointments is early scheduling. In observation 3, it shows that as early as the scheduling happened, the patients are most likely not going to show up to their appointment. That is reasonable cause for many reasons. The patients might forget, or gets busy with other things on the date of the appointment. Also, from observation 4, it shows how sms reminder make a small changes on the no-show rate.
# 
# Appointment time was not registered in the data, and that could be a very useful infomration to know which part of the day the patients are most likely to skip their appointments. As well, I wanted to have longer interval of time than 3 months. Longer period of time will give us an indication on how seasons and holidays may affect the appointment show-up status.
