#!/usr/bin/env python
# coding: utf-8

# Indian liver patient data consists of 500+ records, out of which 400+ are patients. This data consists of the recorded liver parameters.
# First step of the analysis is to read the data into a pandas dataframe as demonstrated below and look at a sample of the data.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

liverData=pd.read_csv("../input/indian_liver_patient.csv")
liverData.head(3)


# To continue the analysis, we should make sure that the doesn't have any NaN's null's. The following command displays a detailed analysis of the meta data of the data set.

# In[ ]:


liverData.info()


# From the above output, we observe that "Albumin_and_Globulin_Ratio" column has 4 null values.
# We do the following modifications to the data:
# 1. To continue further, lets fill up the data frome with 0's for those particular cells. This would be very helpfull in smooth analysis of the data. Another thoughtfull approach would be to fill them up with the mean of the observed values. But for the sake of simplicity, we would fill them with 0's. I encourage you to try the analysis by replacing with mean.
# 2. We will also label the patient and non patient data based on the class labels for the sake of reader convenience.
# 
# The following code would implement our idea in a neat fashion.

# In[ ]:


liverData['Albumin_and_Globulin_Ratio'].fillna(value=0, inplace=True) #Fill the NaN's with 0's
liverData['Dataset'].replace(to_replace=1, value='patient', inplace=True) #Replacing the class labels
liverData['Dataset'].replace(to_replace=2, value='nonpatient', inplace=True) #Replacing the class labels
liverData.head(3)


# We will now look at the data set distribution.
# 
# The following questions can be answered with the following plot:
# 1. Number of male and female patients records.
# 2. Number of male and female non-patients records.

# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

df1 = pd.value_counts(liverData.loc[liverData['Gender'] == 'Male']['Dataset'], sort = True).sort_index()
df2 = pd.value_counts(liverData.loc[liverData['Gender'] == 'Female']['Dataset'], sort = True).sort_index()
df1.plot(kind='bar', color='salmon', ax=ax, position=0, width=0.25)
df2.plot(kind='bar', color='mediumturquoise', ax=ax, position=1, width=0.25)

plt.title("Patient frequency histogram", fontsize=16)
plt.text(-0.4, 240, "Male", color='salmon', fontweight='bold', fontsize=14)
plt.text(-0.4, 210, "Female", color='mediumturquoise', fontweight='bold', fontsize=14)
    


# We now have a holistic idea of the distribution of the dataset. 
# 
# The following observations can be made from the bar plot:
# 1. Liver disease is more prominent in males.
# 2. Among the females, more than 60% of females have liver disesease. Similar pattern is also observed in males.
# 
# We will now see how the actual data is distributed. The following line of code will serve the purpose.
# 
# 

# In[ ]:


liverData.describe()


# It is noticable that the average age of the dataset(most of the patients) is around 45 years. The rest of the columns are too medical for me.
# Let me remind that our goal is to train a model which can crunch the readings for any patient and predict if he has a disease or not. We will now scale all the reading between 0 and 1.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
simpleScaler=MinMaxScaler()
cols=list(liverData.drop(['Gender', 'Dataset'],axis=1).columns)
liverDataScaled=pd.DataFrame(data=liverData)
liverDataScaled[cols]=simpleScaler.fit_transform(liverData[cols])
liverDataScaled.head(3)


# Now that the data is scalled, we can convert the gender and the patient class columns into dummy variables. This process will convert the categorical data to another variable which can smoothly be used as a training variable.

# In[ ]:


liverDataScaledEncoded=pd.get_dummies(liverDataScaled)
liverDataScaledEncoded.head(3)


# In the next steps, I would like to visualise the data in the form of box plots. Box plots are great when it comes to analysing the distribution of columns in the data. For futher details, a brief, yet complete analysis on box plot can be found at http://www.physics.csbsju.edu/stats/box2.html.
# Let's segregate patient and non patient data and visualise them in the form of box plots.

# In[ ]:


boxprops = dict(linestyle='-', color='k')
medianprops = dict(linestyle='-', color='k')
plt.figure()
ax = plt.subplot(111)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
plt.title("Scaled feature (patients)", fontsize=16)
liverDataScaledEncodedBoxPlotDF=liverDataScaledEncoded.loc[liverDataScaledEncoded['Dataset_patient'] == 1].drop(
                                ['Gender_Male', 'Gender_Female', 'Dataset_patient', 'Dataset_nonpatient'],axis=1)
#liverDataScaledEncodedBoxPlotDF = liverDataScaledEncodedBoxPlotDF.sort_values(by=['Total_Bilirubin'], ascending=[True])
bp = liverDataScaledEncodedBoxPlotDF.boxplot(vert=False, showmeans=True, showfliers=False,
                boxprops=boxprops,
                medianprops=medianprops)


# In[ ]:



import matplotlib.pyplot as plt
plt.figure()
ax = plt.subplot(111)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
plt.title("Scaled feature (non-patients)", fontsize=16)
liverDataScaledEncodedBoxPlotDF=liverDataScaledEncoded.loc[liverDataScaledEncoded['Dataset_nonpatient'] == 1].drop(
                                ['Gender_Male', 'Gender_Female', 'Dataset_patient', 'Dataset_nonpatient'],axis=1)
bp = liverDataScaledEncodedBoxPlotDF.boxplot(vert=False, showmeans=True, showfliers=False,
                boxprops=boxprops,
                medianprops=medianprops)


# The observed difference in both the plots is minimal. But lets see how the regression model performs with the training and the test data. We will split the data into 75:25 for Train: test data.
# We will use Python's package sklearn to train and test a multinomial logistic regression model. 

# In[ ]:


from sklearn.cross_validation import train_test_split
train_x, test_x, train_y, test_y = train_test_split(liverDataScaledEncoded.drop(['Gender_Male', 'Gender_Female', 
                                                                            'Dataset_patient', 'Dataset_nonpatient'], axis=1), 
                                                    liverDataScaledEncoded['Dataset_patient'], train_size=0.75) 
print("Test data size: " + str(train_x.shape))
print("Test data size: " + str(test_x.shape))


# In[ ]:


from sklearn import metrics
from sklearn import linear_model
mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(train_x, train_y)
print("Multinomial Logistic regression Train Accuracy :: "+ str(metrics.accuracy_score(train_y, mul_lr.predict(train_x))))
print("Multinomial Logistic regression Test Accuracy :: "+ str(metrics.accuracy_score(test_y, mul_lr.predict(test_x))))


# For both the train and test data, we obtian an accuracy of 0.71. This does not seem to be a great result for a logistic regression in this case. I hope to get my hands on even more refined and accurate data set to continue my analysis. It would be pretty exiting to see data science actyually being used in the real world.
